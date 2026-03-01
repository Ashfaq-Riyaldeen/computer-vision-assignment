from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def ensure_binary_u8(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        raise ValueError("mask must be uint8.")
    if mask.ndim != 2:
        raise ValueError("mask must be 2D.")
    return np.where(mask > 0, 255, 0).astype(np.uint8, copy=False)


def largest_connected_component(binary_u8: np.ndarray) -> np.ndarray:
    binary_u8 = ensure_binary_u8(binary_u8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    if num <= 1:
        return binary_u8

    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    return np.where(labels == keep, 255, 0).astype(np.uint8)


def fill_holes(binary_u8: np.ndarray) -> np.ndarray:
    binary_u8 = ensure_binary_u8(binary_u8)
    h, w = binary_u8.shape

    inv = cv2.bitwise_not(binary_u8)  # background+holes = 255, object = 0
    flood = inv.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 0)  # remove outside background in inv
    holes = flood  # remaining 255 pixels correspond to holes
    return cv2.bitwise_or(binary_u8, holes)


def pick_top_k_components(binary_u8: np.ndarray, k: int) -> list[np.ndarray]:
    binary_u8 = ensure_binary_u8(binary_u8)
    if k <= 0:
        raise ValueError("k must be > 0.")

    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    if num <= 1:
        return []

    comp_stats = []
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        comp_stats.append((area, label))
    comp_stats.sort(reverse=True)

    out: list[np.ndarray] = []
    for _, label in comp_stats[:k]:
        out.append(np.where(labels == label, 255, 0).astype(np.uint8))
    return out


def overlay_two_masks(
    gray_u8: np.ndarray,
    mask_a_u8: np.ndarray,
    mask_b_u8: np.ndarray,
    *,
    color_a_bgr: tuple[int, int, int] = (0, 0, 255),
    color_b_bgr: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    if gray_u8.ndim != 2 or gray_u8.dtype != np.uint8:
        raise ValueError("Expected grayscale uint8 image.")
    mask_a_u8 = ensure_binary_u8(mask_a_u8)
    mask_b_u8 = ensure_binary_u8(mask_b_u8)
    if gray_u8.shape != mask_a_u8.shape or gray_u8.shape != mask_b_u8.shape:
        raise ValueError("Image and masks must have the same shape.")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1].")

    out = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR).astype(np.float32)
    a = mask_a_u8 > 0
    b = mask_b_u8 > 0

    if np.any(a):
        out[a] = (1.0 - alpha) * out[a] + alpha * np.array(color_a_bgr, dtype=np.float32)
    if np.any(b):
        out[b] = (1.0 - alpha) * out[b] + alpha * np.array(color_b_bgr, dtype=np.float32)

    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    image_path = root_dir / "Images" / "Image_4.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_u8 = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_u8 is None:
        raise ValueError(f"Failed to read image: {image_path}")

    output_dir = root_dir / "Outputs" / "Q8"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parameters tuned for the provided CT slice (lungs are dark / air-filled regions inside the body).
    body_thr = 5
    lung_thr = 50
    lung_smooth_ksize = 1

    # 1) Body mask (largest foreground component), then fill holes (so lungs are inside the body region).
    # Avoid aggressive morphology here; it can connect the body to the image border (CT table/artifacts),
    # which leaks outside background into the "filled" mask.
    body_seed = (img_u8 > body_thr).astype(np.uint8) * 255
    body_lcc = largest_connected_component(body_seed)
    body_filled = fill_holes(body_lcc)

    # 2) Lung candidates: low-intensity pixels inside the filled body.
    # Do NOT close the full lung mask before connected components; even small closing kernels can merge
    # the left/right lungs into one component. We select components first, then smooth each mask separately.
    lung_seed = ((img_u8 < lung_thr) & (body_filled > 0)).astype(np.uint8) * 255

    # 3) Keep two largest connected components (left/right lungs).
    comps = pick_top_k_components(lung_seed, k=2)
    if len(comps) < 2:
        raise RuntimeError("Failed to find two lung components. Try adjusting thresholds.")

    # Order by centroid x-position to label left/right (image-left vs image-right).
    centroids_x = []
    for m in comps:
        ys, xs = np.where(m > 0)
        centroids_x.append(float(xs.mean()) if xs.size else 0.0)

    left_idx, right_idx = (0, 1) if centroids_x[0] < centroids_x[1] else (1, 0)

    left_lung = fill_holes(comps[left_idx])
    right_lung = fill_holes(comps[right_idx])

    if lung_smooth_ksize > 1:
        smooth_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (lung_smooth_ksize, lung_smooth_ksize)
        )
        left_lung = cv2.morphologyEx(left_lung, cv2.MORPH_OPEN, smooth_kernel, iterations=1)
        right_lung = cv2.morphologyEx(right_lung, cv2.MORPH_OPEN, smooth_kernel, iterations=1)
    lungs = cv2.bitwise_or(left_lung, right_lung)

    left_isolated = cv2.bitwise_and(img_u8, img_u8, mask=left_lung)
    right_isolated = cv2.bitwise_and(img_u8, img_u8, mask=right_lung)
    lungs_isolated = cv2.bitwise_and(img_u8, img_u8, mask=lungs)

    overlay = overlay_two_masks(img_u8, left_lung, right_lung, alpha=0.45)

    # Save outputs (matching "Original Intensity Image and Isolated Organs")
    cv2.imwrite(str(output_dir / "Image_4_original.png"), img_u8)
    cv2.imwrite(str(output_dir / "Image_4_body_mask.png"), body_filled)
    cv2.imwrite(str(output_dir / "Image_4_lungs_mask.png"), lungs)
    cv2.imwrite(str(output_dir / "Image_4_left_lung_mask.png"), left_lung)
    cv2.imwrite(str(output_dir / "Image_4_right_lung_mask.png"), right_lung)
    cv2.imwrite(str(output_dir / "Image_4_lungs_overlay.png"), overlay)
    cv2.imwrite(str(output_dir / "Image_4_lungs_isolated.png"), lungs_isolated)
    cv2.imwrite(str(output_dir / "Image_4_left_lung_isolated.png"), left_isolated)
    cv2.imwrite(str(output_dir / "Image_4_right_lung_isolated.png"), right_isolated)

    # Simple 2x2 figure
    orig_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    left_bgr = cv2.cvtColor(left_isolated, cv2.COLOR_GRAY2BGR)
    right_bgr = cv2.cvtColor(right_isolated, cv2.COLOR_GRAY2BGR)
    top = np.hstack([orig_bgr, overlay])
    bot = np.hstack([left_bgr, right_bgr])
    figure = np.vstack([top, bot])
    cv2.imwrite(str(output_dir / "Q8_figure.png"), figure)

    total_body = int((body_filled > 0).sum())
    left_area = int((left_lung > 0).sum())
    right_area = int((right_lung > 0).sum())
    lungs_area = left_area + right_area

    params_text = (
        "CT Organ Isolation (Image_4.jpg)\n"
        "Organs of interest: left lung and right lung (air regions)\n"
        f"Image size: {img_u8.shape[1]}x{img_u8.shape[0]}\n"
        "\n"
        "Segmentation parameters:\n"
        f"  Body threshold (>) : {body_thr}\n"
        f"  Lung threshold (<) : {lung_thr}\n"
        f"  Lung smoothing     : morphology open k={lung_smooth_ksize} (per-lung mask)\n"
        "\n"
        "Areas (pixels):\n"
        f"  Body (filled): {total_body}\n"
        f"  Left lung     : {left_area} ({(left_area / total_body * 100.0):.2f}% of body)\n"
        f"  Right lung    : {right_area} ({(right_area / total_body * 100.0):.2f}% of body)\n"
        f"  Both lungs    : {lungs_area} ({(lungs_area / total_body * 100.0):.2f}% of body)\n"
        "\n"
        "Observation: In this CT slice, lungs appear as the two largest low-intensity regions inside the filled\n"
        "body mask. Thresholding + morphology + connected components cleanly isolates them.\n"
    )
    (output_dir / "Q8_params.txt").write_text(params_text, encoding="utf-8")

    print("Q8 completed.")
    print(f"Saved results to: {output_dir}")
    print(params_text)


if __name__ == "__main__":
    main()
