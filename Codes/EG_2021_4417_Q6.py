from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pywt


def add_salt_and_pepper_noise(
    img_u8: np.ndarray, corruption_rate: float, rng: np.random.Generator
) -> np.ndarray:
    if img_u8.dtype != np.uint8:
        raise ValueError("Expected uint8 image.")
    if img_u8.ndim not in (2, 3):
        raise ValueError("Expected a 2D (grayscale) or 3D (color) image.")
    if not (0.0 <= corruption_rate <= 1.0):
        raise ValueError("corruption_rate must be in [0, 1].")

    h, w = img_u8.shape[:2]
    num_pixels = h * w
    num_corrupt = int(round(corruption_rate * num_pixels))

    noisy = img_u8.copy()
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


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    img_f = img.astype(np.float32, copy=False)
    min_val = float(np.min(img_f))
    max_val = float(np.max(img_f))
    if max_val - min_val < 1e-6:
        return np.zeros_like(img_f, dtype=np.uint8)
    scaled = (img_f - min_val) * (255.0 / (max_val - min_val))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def remove_high_frequency_and_reconstruct(channel_u8: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    channel_f32 = channel_u8.astype(np.float32, copy=False)

    coeffs = pywt.wavedec2(channel_f32, wavelet=wavelet, level=level, mode="symmetric")
    cA = coeffs[0]
    zero_details: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for (cH, cV, cD) in coeffs[1:]:
        zero_details.append((np.zeros_like(cH), np.zeros_like(cV), np.zeros_like(cD)))

    coeffs_smooth = [cA, *zero_details]
    recon = pywt.waverec2(coeffs_smooth, wavelet=wavelet, mode="symmetric")

    h, w = channel_u8.shape[:2]
    recon = recon[:h, :w]
    return np.clip(np.rint(recon), 0, 255).astype(np.uint8)


def main() -> None:
    # Project root is one level above the Codes folder
    root_dir = Path(__file__).resolve().parents[1]
    image_path = root_dir / "Images" / "Image_3.jpg"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_u8 = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_u8 is None:
        raise ValueError(f"Failed to read image: {image_path}")

    output_dir = root_dir / "Outputs" / "Q6"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parameters (adjust if needed)
    sp_rate = 0.10  # 10% of pixels corrupted by salt & pepper noise
    laplacian_ksize = 3
    wavelet_name = "bior4.4"  # CDF 9/7 (often referred to as "db 9/7"); alternative: "haar"
    wavelet_level = 3

    cv2.imwrite(str(output_dir / "Image_3_original.png"), img_u8)

    # SP noise image (salt & pepper)
    rng = np.random.default_rng(4417)
    noisy_u8 = add_salt_and_pepper_noise(img_u8, sp_rate, rng)
    cv2.imwrite(str(output_dir / f"Image_3_salt_pepper_{int(sp_rate * 100)}pct.png"), noisy_u8)

    # Laplacian-filtered output of I
    img_f32 = img_u8.astype(np.float32)
    lap_f32 = cv2.Laplacian(img_f32, ddepth=cv2.CV_32F, ksize=laplacian_ksize)
    cv2.imwrite(str(output_dir / "Image_3_laplacian_vis.png"), normalize_to_uint8(lap_f32))

    # I' = I + SP + L(I)
    # SP is treated as an additive term: SP = noisy - I
    sp_term_f32 = noisy_u8.astype(np.float32) - img_f32
    i_prime_f32 = img_f32 + sp_term_f32 + lap_f32
    i_prime_u8 = np.clip(np.rint(i_prime_f32), 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / "Image_3_I_prime.png"), i_prime_u8)

    # Wavelet decomposition, remove high-frequency components, and reconstruct a smooth image
    smooth_u8 = np.empty_like(i_prime_u8)
    for c in range(i_prime_u8.shape[2]):
        smooth_u8[:, :, c] = remove_high_frequency_and_reconstruct(
            i_prime_u8[:, :, c], wavelet=wavelet_name, level=wavelet_level
        )
    cv2.imwrite(str(output_dir / "Image_3_I_prime_wavelet_smooth.png"), smooth_u8)

    absdiff = cv2.absdiff(i_prime_u8, smooth_u8)
    cv2.imwrite(str(output_dir / "Image_3_I_prime_vs_smooth_absdiff.png"), absdiff)

    params_text = (
        f"Input: {image_path.name}\n"
        f"SP corruption rate: {sp_rate:.2f}\n"
        f"Laplacian ksize: {laplacian_ksize}\n"
        f"Wavelet: {wavelet_name}\n"
        f"Wavelet level: {wavelet_level}\n"
        "High-frequency removal: all detail subbands set to zero; only approximation kept.\n"
    )
    (output_dir / "Q6_params.txt").write_text(params_text, encoding="utf-8")

    print("Q6 completed.")
    print(f"Saved results to: {output_dir}")
    print(params_text)
    print(
        "Observation: removing the high-frequency wavelet coefficients suppresses edges and textures, "
        "so the reconstructed image becomes smoother/blurred while preserving large-scale intensity variations."
    )


if __name__ == "__main__":
    main()

