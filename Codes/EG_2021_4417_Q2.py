from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def add_salt_and_pepper_noise(
    img: np.ndarray, corruption_rate: float, rng: np.random.Generator
) -> np.ndarray:
    if img.dtype != np.uint8:
        raise ValueError("Expected uint8 image.")
    if img.ndim not in (2, 3):
        raise ValueError("Expected a 2D (grayscale) or 3D (color) image.")
    if not (0.0 <= corruption_rate <= 1.0):
        raise ValueError("corruption_rate must be in [0, 1].")

    h, w = img.shape[:2]
    num_pixels = h * w
    num_corrupt = int(round(corruption_rate * num_pixels))

    noisy = img.copy()
    if num_corrupt == 0:
        return noisy

    flat_indices = rng.choice(num_pixels, size=num_corrupt, replace=False)
    num_salt = num_corrupt // 2
    salt_indices = flat_indices[:num_salt]
    pepper_indices = flat_indices[num_salt:]

    salt_rc = np.unravel_index(salt_indices, (h, w))
    pepper_rc = np.unravel_index(pepper_indices, (h, w))

    noisy[salt_rc] = 255
    noisy[pepper_rc] = 0
    return noisy


def main() -> None:
    # Project root is one level above the Codes folder
    root_dir = Path(__file__).resolve().parents[1]
    image_path = root_dir / "Images" / "Image_2.jpg"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    output_dir = root_dir / "Outputs" / "Q2"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # (a) Salt & pepper noise
    # ---------------------------
    rng = np.random.default_rng(4417)
    noise_levels = (0.10, 0.20)
    noisy_by_level: dict[float, np.ndarray] = {}

    for corruption_rate in noise_levels:
        noisy = add_salt_and_pepper_noise(img, corruption_rate, rng)
        noisy_by_level[corruption_rate] = noisy
        out_path = output_dir / f"Image_2_salt_pepper_{int(corruption_rate * 100)}pct.png"
        cv2.imwrite(str(out_path), noisy)

    # ---------------------------
    # (b) Median filtering
    # ---------------------------
    kernel_sizes = (3, 5, 11)

    # Median filtering on the original image (for reference)
    for k in kernel_sizes:
        filtered = cv2.medianBlur(img, k)
        out_path = output_dir / f"Image_2_median_{k}x{k}.png"
        cv2.imwrite(str(out_path), filtered)

    # Median filtering on the noisy images (denoising)
    for corruption_rate, noisy in noisy_by_level.items():
        for k in kernel_sizes:
            filtered = cv2.medianBlur(noisy, k)
            out_path = output_dir / (
                f"Image_2_salt_pepper_{int(corruption_rate * 100)}pct_median_{k}x{k}.png"
            )
            cv2.imwrite(str(out_path), filtered)

    print("Salt & pepper noise added.")
    print(f"Saved results to: {output_dir}")
    print(
        "Observation (noise): higher corruption rate increases the density of black/white speckles, "
        "reducing visible detail and making edges/textures harder to discern.\n"
        "Observation (median filter): increasing kernel size removes more impulse noise (better denoising), "
        "but it also increases smoothing and can remove fine details/thin structures and slightly distort edges."
    )


if __name__ == "__main__":
    main()
