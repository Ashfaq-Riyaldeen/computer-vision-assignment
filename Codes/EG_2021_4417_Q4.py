from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    img_f = img.astype(np.float32, copy=False)
    min_val = float(np.min(img_f))
    max_val = float(np.max(img_f))
    if max_val - min_val < 1e-6:
        return np.zeros_like(img_f, dtype=np.uint8)
    scaled = (img_f - min_val) * (255.0 / (max_val - min_val))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def main() -> None:
    # Project root is one level above the Codes folder
    root_dir = Path(__file__).resolve().parents[1]
    image_path = root_dir / "Images" / "Image_3.jpg"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_u8 = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_u8 is None:
        raise ValueError(f"Failed to read image: {image_path}")

    output_dir = root_dir / "Outputs" / "Q4"
    output_dir.mkdir(parents=True, exist_ok=True)

    # OpenCV pyramid functions (pyrDown/pyrUp) use the standard 5x5 Gaussian kernel internally.
    chosen_kernel = "5x5 (OpenCV pyrDown/pyrUp default)"

    levels = 3  # 3-level pyramid: L0 (original), L1, L2
    gaussian_pyr: list[np.ndarray] = [img_u8.astype(np.float32)]

    for _ in range(1, levels):
        down = cv2.pyrDown(gaussian_pyr[-1])
        gaussian_pyr.append(down)

    for i, g in enumerate(gaussian_pyr):
        out_path = output_dir / f"Image_3_gaussian_pyramid_L{i}.png"
        cv2.imwrite(str(out_path), np.clip(g, 0, 255).astype(np.uint8))

    laplacian_pyr: list[np.ndarray] = []
    for i in range(levels - 1):
        expanded = cv2.pyrUp(
            gaussian_pyr[i + 1],
            dstsize=(gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0]),
        )
        lap = gaussian_pyr[i] - expanded
        laplacian_pyr.append(lap)

        out_path = output_dir / f"Image_3_laplacian_pyramid_L{i}.png"
        cv2.imwrite(str(out_path), normalize_to_uint8(lap))

    # The last Laplacian level is the coarsest Gaussian level
    laplacian_pyr.append(gaussian_pyr[-1])
    out_path_last = output_dir / f"Image_3_laplacian_pyramid_L{levels - 1}.png"
    cv2.imwrite(str(out_path_last), np.clip(gaussian_pyr[-1], 0, 255).astype(np.uint8))

    # Optional: reconstruct to verify pyramid correctness
    recon = laplacian_pyr[-1]
    for i in range(levels - 2, -1, -1):
        recon = cv2.pyrUp(recon, dstsize=(gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0]))
        recon = recon + laplacian_pyr[i]

    recon_u8 = np.clip(recon, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / "Image_3_laplacian_reconstructed.png"), recon_u8)
    diff = cv2.absdiff(img_u8, recon_u8)
    cv2.imwrite(str(output_dir / "Image_3_laplacian_reconstruction_absdiff.png"), diff)

    print("Gaussian and Laplacian pyramids generated.")
    print(f"Saved results to: {output_dir}")
    print(f"Chosen kernel size: {chosen_kernel}")
    print(
        "Justification: a 5x5 Gaussian kernel is the standard choice for Gaussian/Laplacian pyramids because it "
        "provides enough low-pass smoothing to reduce aliasing before downsampling while keeping computation modest."
    )
    print(
        "Effect of increasing kernel size (general): larger kernels produce stronger smoothing at each pyramid step, "
        "which reduces noise/aliasing more but removes fine details sooner and makes edges less sharp."
    )


if __name__ == "__main__":
    main()

