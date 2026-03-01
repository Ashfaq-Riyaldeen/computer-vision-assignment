from __future__ import annotations

from pathlib import Path

import cv2


def main() -> None:
    # Project root is one level above the Codes folder
    root_dir = Path(__file__).resolve().parents[1]
    image_path = root_dir / "Images" / "Image_1.jpg"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    output_dir = root_dir / "Outputs" / "Q1"
    output_dir.mkdir(parents=True, exist_ok=True)

    kernel_sizes = [3, 5, 11, 15]
    for k in kernel_sizes:
        blurred = cv2.blur(img, (k, k))
        out_path = output_dir / f"Image_1_avg_{k}x{k}.png"
        cv2.imwrite(str(out_path), blurred)

    print("Average filtering completed.")
    print(f"Saved results to: {output_dir}")
    print(
        "Effect of increasing kernel size: stronger smoothing/blur, "
        "greater noise reduction, but more detail and edge loss."
    )


if __name__ == "__main__":
    main()
