from __future__ import annotations

from pathlib import Path

import cv2


def main() -> None:
    # Project root is one level above the Codes folder
    root_dir = Path(__file__).resolve().parents[1]
    image_path = root_dir / "Images" / "Image_3.jpg"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    output_dir = root_dir / "Outputs" / "Q3"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------
    # Gaussian filtering: different kernels
    # ---------------------------------------
    kernel_sizes = (3, 5, 11, 15)
    for k in kernel_sizes:
        blurred = cv2.GaussianBlur(img, (k, k), sigmaX=0, sigmaY=0)
        out_path = output_dir / f"Image_3_gaussian_{k}x{k}.png"
        cv2.imwrite(str(out_path), blurred)

    # ---------------------------------------
    # Gaussian filtering: vary sigma (σ)
    # For a fixed kernel size
    # ---------------------------------------
    fixed_kernel = 11
    sigma_values = (0.5, 1.0, 2.0, 4.0)
    for sigma in sigma_values:
        blurred = cv2.GaussianBlur(img, (fixed_kernel, fixed_kernel), sigmaX=sigma, sigmaY=sigma)
        out_path = output_dir / f"Image_3_gaussian_{fixed_kernel}x{fixed_kernel}_sigma_{sigma:g}.png"
        cv2.imwrite(str(out_path), blurred)

    print("Gaussian filtering completed.")
    print(f"Saved results to: {output_dir}")
    print(
        "Observation (kernel size): increasing the kernel size increases smoothing/blur and noise reduction, "
        "but reduces fine details and edge sharpness (and typically increases computation time)."
    )
    print(
        f"Observation (sigma): for the fixed {fixed_kernel}x{fixed_kernel} kernel, increasing sigma spreads the "
        "Gaussian weights more, producing stronger blur; smaller sigma preserves more detail."
    )


if __name__ == "__main__":
    main()
