from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class MorphMetrics:
    fg_area_px: int
    components_total: int
    components_large: int
    components_small: int
    perimeter_px: int


@dataclass(frozen=True)
class ObjectFeature:
    label: int
    area_px: int
    perimeter_px: float


def ensure_u8_gray(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Image is None.")
    if img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError("Expected a 2D uint8 image.")
    return img


def otsu_binarize_objects(img_u8: np.ndarray) -> tuple[np.ndarray, float, str]:
    img_u8 = ensure_u8_gray(img_u8)
    t, bin_mask = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv_mask = cv2.bitwise_not(bin_mask)

    h, w = img_u8.shape

    def _largest_touches_border(mask: np.ndarray) -> bool:
        num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1:
            return True
        areas = stats[1:, cv2.CC_STAT_AREA]
        k = 1 + int(np.argmax(areas))
        x, y, ww, hh, _ = stats[k]
        return bool(x == 0 or y == 0 or x + ww == w or y + hh == h)

    bin_touches = _largest_touches_border(bin_mask)
    inv_touches = _largest_touches_border(inv_mask)

    # Prefer the mask whose largest component does NOT touch the image border (more likely to be objects).
    if bin_touches and not inv_touches:
        return inv_mask, float(t), "inv"
    if inv_touches and not bin_touches:
        return bin_mask, float(t), "bin"

    # Fallback: choose the mask with fewer connected components (less fragmented).
    num_bin, _, _, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    num_inv, _, _, _ = cv2.connectedComponentsWithStats(inv_mask, connectivity=8)
    if (num_inv - 1) <= (num_bin - 1):
        return inv_mask, float(t), "inv"
    return bin_mask, float(t), "bin"


def boundary_perimeter_px(mask_u8: np.ndarray) -> int:
    if mask_u8.ndim != 2 or mask_u8.dtype != np.uint8:
        raise ValueError("Expected a 2D uint8 binary mask.")
    mask_bin = (mask_u8 > 0).astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    er = cv2.erode(mask_bin, k3, iterations=1)
    boundary = cv2.bitwise_xor(mask_bin, er)
    return int((boundary > 0).sum())


def compute_morph_metrics(mask_u8: np.ndarray, min_large_area_px: int) -> MorphMetrics:
    mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255
    fg_area = int((mask_u8 > 0).sum())

    num, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    comp_total = int(num - 1)
    if comp_total <= 0:
        return MorphMetrics(0, 0, 0, 0, 0)

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64, copy=False)
    comp_large = int((areas >= min_large_area_px).sum())
    comp_small = int(comp_total - comp_large)
    perim = boundary_perimeter_px(mask_u8)
    return MorphMetrics(fg_area, comp_total, comp_large, comp_small, perim)


def extract_object_features(mask_u8: np.ndarray, min_area_px: int, max_objects: int = 10) -> list[ObjectFeature]:
    mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return []

    candidates: list[tuple[int, int]] = []
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area_px:
            candidates.append((area, label))
    candidates.sort(reverse=True)
    candidates = candidates[:max_objects]

    features: list[ObjectFeature] = []
    for area, label in candidates:
        comp = np.where(labels == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        perimeter = float(cv2.arcLength(contours[0], True))
        features.append(ObjectFeature(label=label, area_px=area, perimeter_px=perimeter))
    return features


def to_bgr(img_u8: np.ndarray) -> np.ndarray:
    if img_u8.ndim == 2:
        return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    return img_u8


def add_title(tile_bgr_u8: np.ndarray, title: str) -> np.ndarray:
    out = tile_bgr_u8.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        title,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def tile(img_u8: np.ndarray, title: str, size: int = 512) -> np.ndarray:
    bgr = to_bgr(img_u8)
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    return add_title(bgr, title)


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    image_path = root_dir / "Images" / "Image_6.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_u8 = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    img_u8 = ensure_u8_gray(img_u8)

    output_dir = root_dir / "Outputs" / "Q10"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Convert to a suitable binary image (objects as white/foreground).
    # For this synthetic image, objects are darker than the background; we use Otsu and choose the polarity
    # automatically (bin vs inv) so that "objects" do not form a border-touching background component.
    binary_u8, otsu_t, polarity = otsu_binarize_objects(img_u8)

    # 2) Morphological operations
    se_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    se_main = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    eroded = cv2.erode(binary_u8, se_small, iterations=1)
    dilated = cv2.dilate(binary_u8, se_small, iterations=1)
    opened = cv2.morphologyEx(binary_u8, cv2.MORPH_OPEN, se_main, iterations=1)
    closed = cv2.morphologyEx(binary_u8, cv2.MORPH_CLOSE, se_main, iterations=1)

    # 3) Feature extraction (area + perimeter) and comparison table
    min_large_area = 500  # ignore tiny components as noise
    metrics = {
        "Binary": compute_morph_metrics(binary_u8, min_large_area),
        "Erosion": compute_morph_metrics(eroded, min_large_area),
        "Dilation": compute_morph_metrics(dilated, min_large_area),
        "Opening": compute_morph_metrics(opened, min_large_area),
        "Closing": compute_morph_metrics(closed, min_large_area),
    }

    features_open = extract_object_features(opened, min_area_px=min_large_area, max_objects=20)
    features_close = extract_object_features(closed, min_area_px=min_large_area, max_objects=20)

    # Save images
    cv2.imwrite(str(output_dir / "Image_6_original.png"), img_u8)
    cv2.imwrite(str(output_dir / "Image_6_binary.png"), binary_u8)
    cv2.imwrite(str(output_dir / "Image_6_eroded.png"), eroded)
    cv2.imwrite(str(output_dir / "Image_6_dilated.png"), dilated)
    cv2.imwrite(str(output_dir / "Image_6_opened.png"), opened)
    cv2.imwrite(str(output_dir / "Image_6_closed.png"), closed)

    # Montage visualization
    row1 = np.hstack(
        [
            tile(img_u8, "Original (Image_6.jpg)"),
            tile(binary_u8, f"Binary (Otsu={otsu_t:.0f}, {polarity})"),
            tile(eroded, "Erosion (3x3 ellipse)"),
        ]
    )
    row2 = np.hstack(
        [
            tile(dilated, "Dilation (3x3 ellipse)"),
            tile(opened, "Opening (5x5 ellipse)"),
            tile(closed, "Closing (5x5 ellipse)"),
        ]
    )
    montage = np.vstack([row1, row2])
    cv2.imwrite(str(output_dir / "Q10_comparison.png"), montage)

    # Table: quantitative + qualitative comparison
    qualitative = {
        "Binary": "Raw threshold mask; contains many noisy speckles and fragmented objects.",
        "Erosion": "Shrinks objects; removes small white noise but may break thin parts and reduce area.",
        "Dilation": "Expands objects; fills small gaps but also enlarges noise and can merge close objects.",
        "Opening": "Best for noise removal (erosion then dilation); preserves main shapes fairly well.",
        "Closing": "Best for gap filling (dilation then erosion); fills small holes but may connect nearby shapes.",
    }

    lines: list[str] = []
    lines.append("Morphological Analysis (Image_6.jpg)")
    lines.append(f"Input: {image_path.name} ({img_u8.shape[1]}x{img_u8.shape[0]})")
    lines.append("")
    lines.append("Binary conversion:")
    lines.append(f"  Otsu threshold: {otsu_t:.2f}")
    lines.append(f"  Selected polarity: {polarity} (objects as foreground/white)")
    lines.append("")
    lines.append("Structuring elements:")
    lines.append("  Erosion/Dilation: ellipse 3x3, 1 iteration")
    lines.append("  Opening/Closing : ellipse 5x5, 1 iteration")
    lines.append(f"  Large-object area threshold (for features): {min_large_area} px")
    lines.append("")
    lines.append("Comparison table (quantitative):")
    lines.append("Operation | FG area(px) | Components | Large(>=thr) | Small(<thr) | Perimeter(px)")
    lines.append("-" * 78)
    for name in ["Binary", "Erosion", "Dilation", "Opening", "Closing"]:
        m = metrics[name]
        lines.append(
            f"{name:9s} | {m.fg_area_px:10d} | {m.components_total:10d} | {m.components_large:11d} | "
            f"{m.components_small:10d} | {m.perimeter_px:12d}"
        )
    lines.append("")
    lines.append("Comparison table (qualitative):")
    lines.append("Operation | Noise removal | Gap filling | Shape preservation")
    lines.append("-" * 74)
    lines.append("Erosion   | Good (removes small specks) | Poor | Low-Medium (shrinks/breaks thin parts)")
    lines.append("Dilation  | Poor (amplifies noise)      | Good (bridges small gaps) | Low (merges/expands)")
    lines.append("Opening   | Best                        | Medium | Medium-High (keeps main shapes)")
    lines.append("Closing   | Medium                      | Best   | Medium (fills holes, may connect close shapes)")
    lines.append("")
    lines.append("Basic morphological features (area/perimeter) from Opening result (largest objects):")
    if not features_open:
        lines.append("  (No large objects found.)")
    else:
        for i, f in enumerate(sorted(features_open, key=lambda x: x.area_px, reverse=True), start=1):
            lines.append(f"  Obj {i:02d}: area={f.area_px:6d} px, perimeter={f.perimeter_px:8.2f} px")
    lines.append("")
    lines.append("Basic morphological features (area/perimeter) from Closing result (largest objects):")
    if not features_close:
        lines.append("  (No large objects found.)")
    else:
        for i, f in enumerate(sorted(features_close, key=lambda x: x.area_px, reverse=True), start=1):
            lines.append(f"  Obj {i:02d}: area={f.area_px:6d} px, perimeter={f.perimeter_px:8.2f} px")
    lines.append("")
    lines.append("Observation: Opening removes isolated noise while retaining the major geometric objects. Closing fills")
    lines.append("small gaps/holes but can also connect nearby objects, increasing area and reducing the number of separate")
    lines.append("components. Erosion and dilation are more extreme and mainly useful as building blocks for opening/closing.")
    lines.append("")

    report = "\n".join(lines)
    (output_dir / "Q10_results.txt").write_text(report, encoding="utf-8")

    print("Q10 completed.")
    print(f"Saved results to: {output_dir}")
    print(report)


if __name__ == "__main__":
    main()
