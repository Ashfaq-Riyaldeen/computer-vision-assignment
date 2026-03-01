from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pywt


def resize_float01(img_f01: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    if img_f01.ndim != 2:
        raise ValueError("Expected a single-channel (2D) image.")
    out_h, out_w = out_hw
    in_h, in_w = img_f01.shape

    if out_h <= 0 or out_w <= 0:
        raise ValueError("Invalid output size.")

    if out_h < in_h or out_w < in_w:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LINEAR

    resized = cv2.resize(img_f01, (out_w, out_h), interpolation=interp)
    return resized.astype(np.float32, copy=False)


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape.")

    a_f = a.astype(np.float32).ravel()
    b_f = b.astype(np.float32).ravel()
    a_f -= float(a_f.mean())
    b_f -= float(b_f.mean())

    denom = float(np.linalg.norm(a_f) * np.linalg.norm(b_f))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a_f, b_f) / denom)


def details_index_for_level(dwt_level: int, target_level: int) -> int:
    if dwt_level <= 0:
        raise ValueError("dwt_level must be >= 1.")
    if not (1 <= target_level <= dwt_level):
        raise ValueError("target_level must be in [1, dwt_level].")
    # wavedec2 returns: [cA_L, (cH_L,cV_L,cD_L), ..., (cH_1,cV_1,cD_1)]
    return dwt_level - target_level + 1


def embed_watermark_dwt(
    y_f32: np.ndarray,
    wm_u8: np.ndarray,
    *,
    wavelet: str,
    dwt_level: int,
    embed_level: int,
    band: str,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    if y_f32.ndim != 2:
        raise ValueError("Expected a single-channel (2D) host image.")
    if wm_u8.ndim != 2:
        raise ValueError("Expected a single-channel (2D) watermark image.")
    if alpha <= 0:
        raise ValueError("alpha must be > 0.")
    if band not in {"cH", "cV", "cD"}:
        raise ValueError("band must be one of: cH, cV, cD.")

    coeffs = pywt.wavedec2(y_f32, wavelet=wavelet, level=dwt_level, mode="symmetric")
    idx = details_index_for_level(dwt_level, embed_level)

    cH, cV, cD = coeffs[idx]
    target = {"cH": cH, "cV": cV, "cD": cD}[band]

    wm_f01 = wm_u8.astype(np.float32) / 255.0
    wm_resized_f01 = resize_float01(wm_f01, (target.shape[0], target.shape[1]))
    wm_centered = wm_resized_f01 - 0.5  # range ~[-0.5, 0.5]

    if band == "cH":
        cH = cH + alpha * wm_centered
    elif band == "cV":
        cV = cV + alpha * wm_centered
    else:
        cD = cD + alpha * wm_centered

    coeffs_mod = list(coeffs)
    coeffs_mod[idx] = (cH, cV, cD)

    y_watermarked = pywt.waverec2(coeffs_mod, wavelet=wavelet, mode="symmetric")
    y_watermarked = y_watermarked[: y_f32.shape[0], : y_f32.shape[1]]

    wm_resized_u8 = np.clip(np.rint(wm_resized_f01 * 255.0), 0, 255).astype(np.uint8)
    return y_watermarked.astype(np.float32, copy=False), wm_resized_u8


def extract_watermark_dwt(
    y_orig_f32: np.ndarray,
    y_watermarked_f32: np.ndarray,
    *,
    wavelet: str,
    dwt_level: int,
    embed_level: int,
    band: str,
    alpha: float,
) -> np.ndarray:
    if y_orig_f32.shape != y_watermarked_f32.shape:
        raise ValueError("Original and watermarked host images must have the same shape.")
    if alpha <= 0:
        raise ValueError("alpha must be > 0.")
    if band not in {"cH", "cV", "cD"}:
        raise ValueError("band must be one of: cH, cV, cD.")

    coeffs_o = pywt.wavedec2(y_orig_f32, wavelet=wavelet, level=dwt_level, mode="symmetric")
    coeffs_w = pywt.wavedec2(y_watermarked_f32, wavelet=wavelet, level=dwt_level, mode="symmetric")
    idx = details_index_for_level(dwt_level, embed_level)

    cH_o, cV_o, cD_o = coeffs_o[idx]
    cH_w, cV_w, cD_w = coeffs_w[idx]
    band_o = {"cH": cH_o, "cV": cV_o, "cD": cD_o}[band]
    band_w = {"cH": cH_w, "cV": cV_w, "cD": cD_w}[band]

    wm_hat_f01 = (band_w - band_o) * (1.0 / alpha) + 0.5
    wm_hat_f01 = np.clip(wm_hat_f01, 0.0, 1.0)
    return np.clip(np.rint(wm_hat_f01 * 255.0), 0, 255).astype(np.uint8)


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    image_path = root_dir / "Images" / "Image_3.jpg"
    watermark_path = root_dir / "Images" / "watermark.png"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not watermark_path.exists():
        raise FileNotFoundError(f"Watermark not found: {watermark_path}")

    img_bgr_u8 = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr_u8 is None:
        raise ValueError(f"Failed to read image: {image_path}")

    wm_u8 = cv2.imread(str(watermark_path), cv2.IMREAD_GRAYSCALE)
    if wm_u8 is None:
        raise ValueError(f"Failed to read watermark: {watermark_path}")

    output_dir = root_dir / "Outputs" / "Q7"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parameters
    wavelet_name = "haar"
    dwt_level = 4
    embed_level = 4  # embed at the coarsest detail level (same as dwt_level)
    embed_band = "cH"  # one of: cH (horizontal), cV (vertical), cD (diagonal)
    alpha = 20.0  # strength: higher -> more robust, less imperceptible

    # Work on luminance (Y) for color images.
    ycrcb_u8 = cv2.cvtColor(img_bgr_u8, cv2.COLOR_BGR2YCrCb)
    y_u8 = ycrcb_u8[:, :, 0]
    y_f32 = y_u8.astype(np.float32)

    y_w_f32, wm_resized_u8 = embed_watermark_dwt(
        y_f32,
        wm_u8,
        wavelet=wavelet_name,
        dwt_level=dwt_level,
        embed_level=embed_level,
        band=embed_band,
        alpha=alpha,
    )

    y_w_u8 = np.clip(np.rint(y_w_f32), 0, 255).astype(np.uint8)
    ycrcb_w_u8 = ycrcb_u8.copy()
    ycrcb_w_u8[:, :, 0] = y_w_u8
    watermarked_bgr_u8 = cv2.cvtColor(ycrcb_w_u8, cv2.COLOR_YCrCb2BGR)

    # Extract from the final (rounded + color-converted) watermarked image to simulate a real pipeline.
    ycrcb_from_watermarked = cv2.cvtColor(watermarked_bgr_u8, cv2.COLOR_BGR2YCrCb)
    y_from_watermarked_f32 = ycrcb_from_watermarked[:, :, 0].astype(np.float32)
    wm_extracted_subband_u8 = extract_watermark_dwt(
        y_f32,
        y_from_watermarked_f32,
        wavelet=wavelet_name,
        dwt_level=dwt_level,
        embed_level=embed_level,
        band=embed_band,
        alpha=alpha,
    )

    # Resize extracted watermark back to original watermark size for easy viewing.
    wm_extracted_u8 = cv2.resize(
        wm_extracted_subband_u8,
        (wm_u8.shape[1], wm_u8.shape[0]),
        interpolation=cv2.INTER_AREA if wm_u8.size < wm_extracted_subband_u8.size else cv2.INTER_LINEAR,
    )

    psnr_db = float(cv2.PSNR(img_bgr_u8, watermarked_bgr_u8))
    ncc = normalized_cross_correlation(wm_resized_u8, wm_extracted_subband_u8)

    cv2.imwrite(str(output_dir / "Image_3_original.png"), img_bgr_u8)
    cv2.imwrite(str(output_dir / "watermark_original.png"), wm_u8)
    cv2.imwrite(str(output_dir / "watermark_resized_for_embedding.png"), wm_resized_u8)
    cv2.imwrite(str(output_dir / "Image_3_watermarked.png"), watermarked_bgr_u8)
    cv2.imwrite(str(output_dir / "Image_3_absdiff.png"), cv2.absdiff(img_bgr_u8, watermarked_bgr_u8))
    cv2.imwrite(str(output_dir / "watermark_extracted_subband.png"), wm_extracted_subband_u8)
    cv2.imwrite(str(output_dir / "watermark_extracted.png"), wm_extracted_u8)
    cv2.imwrite(str(output_dir / "watermark_extracted_binary.png"), (wm_extracted_u8 >= 128).astype(np.uint8) * 255)

    params_text = (
        "DWT Watermarking (non-blind) on Image_3.jpg\n"
        f"Host image: {image_path.name} ({img_bgr_u8.shape[1]}x{img_bgr_u8.shape[0]})\n"
        f"Watermark: {watermark_path.name} ({wm_u8.shape[1]}x{wm_u8.shape[0]})\n"
        f"Wavelet: {wavelet_name}\n"
        f"DWT levels: {dwt_level}\n"
        f"Embedding level: {embed_level}\n"
        f"Embedding band: {embed_band}\n"
        f"Alpha (strength): {alpha}\n"
        "\n"
        "Embedding rule (in selected DWT subband):\n"
        "  C' = C + alpha * (W - 0.5)\n"
        "Extraction rule (needs original image coefficients):\n"
        "  W_hat = (C' - C)/alpha + 0.5\n"
        "\n"
        f"PSNR(original, watermarked): {psnr_db:.2f} dB\n"
        f"NCC(resized watermark, extracted subband watermark): {ncc:.4f}\n"
        "\n"
        "Observation: Increasing alpha improves extracted watermark clarity (higher NCC) but makes the watermark\n"
        "more visible in the host image (lower PSNR). Embedding in higher-level detail bands tends to be more\n"
        "robust than level-1 details but can affect low-frequency appearance if alpha is too large.\n"
    )
    (output_dir / "Q7_params.txt").write_text(params_text, encoding="utf-8")

    print("Q7 completed.")
    print(f"Saved results to: {output_dir}")
    print(params_text)


if __name__ == "__main__":
    main()
