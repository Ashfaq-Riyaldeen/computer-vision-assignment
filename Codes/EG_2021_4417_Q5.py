from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class DiffStats:
    max_abs: int
    mean_abs: float
    nonzero_ratio: float
    min_diff: int
    max_diff: int


def opencv_default_gaussian_sigma(ksize: int) -> float:
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("ksize must be a positive odd integer.")
    return 0.3 * (((ksize - 1) * 0.5) - 1) + 0.8


def gaussian_kernel_1d(ksize: int, sigma: float) -> np.ndarray:
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("ksize must be a positive odd integer.")
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")

    radius = ksize // 2
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * (sigma * sigma)))
    kernel = kernel / float(kernel.sum())
    return kernel.astype(np.float32, copy=False)


def box_filter_integral_reflect(img_u8: np.ndarray, ksize: int) -> np.ndarray:
    if img_u8.dtype != np.uint8:
        raise ValueError("Expected uint8 image.")
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("ksize must be a positive odd integer.")
    if img_u8.ndim not in (2, 3):
        raise ValueError("Expected a 2D (grayscale) or 3D (color) image.")

    pad = ksize // 2
    if img_u8.ndim == 2:
        img_3d = img_u8[..., None]
        squeeze = True
    else:
        img_3d = img_u8
        squeeze = False

    padded = np.pad(img_3d, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    integral = padded.astype(np.uint64).cumsum(axis=0).cumsum(axis=1)

    # Pad integral with a zero row/col so we can use inclusion-exclusion without conditionals.
    ii = np.zeros((integral.shape[0] + 1, integral.shape[1] + 1, integral.shape[2]), dtype=np.uint64)
    ii[1:, 1:] = integral

    h, w = img_u8.shape[:2]
    y0 = np.arange(h)[:, None]
    x0 = np.arange(w)[None, :]
    y1 = y0 + ksize
    x1 = x0 + ksize

    window_sum = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    mean = window_sum.astype(np.float32) * (1.0 / float(ksize * ksize))

    # OpenCV blur on uint8 rounds to nearest and saturates to [0,255]
    out = np.clip(np.floor(mean + 0.5), 0, 255).astype(np.uint8)
    if squeeze:
        return out[..., 0]
    return out


def median_filter_edge_u8(img_u8: np.ndarray, ksize: int, block_rows: int = 128) -> np.ndarray:
    if img_u8.dtype != np.uint8:
        raise ValueError("Expected uint8 image.")
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("ksize must be a positive odd integer.")
    if img_u8.ndim not in (2, 3):
        raise ValueError("Expected a 2D (grayscale) or 3D (color) image.")

    pad = ksize // 2
    kth = (ksize * ksize) // 2

    def _median_channel(channel: np.ndarray) -> np.ndarray:
        h, w = channel.shape
        padded = np.pad(channel, ((pad, pad), (pad, pad)), mode="edge")
        out = np.empty_like(channel)

        for y0 in range(0, h, block_rows):
            y1 = min(y0 + block_rows, h)
            block = padded[y0 : y1 + ksize - 1, :]
            windows = sliding_window_view(block, (ksize, ksize))  # (block_h, w, k, k)
            flat = windows.reshape(windows.shape[0], windows.shape[1], ksize * ksize)
            med = np.partition(flat, kth, axis=2)[..., kth]
            out[y0:y1, :] = med.astype(np.uint8, copy=False)

            print(f"  Median self-impl: rows {y0 + 1}-{y1} / {h}")

        return out

    if img_u8.ndim == 2:
        return _median_channel(img_u8)

    channels = [img_u8[:, :, c] for c in range(img_u8.shape[2])]
    filtered = [_median_channel(ch) for ch in channels]
    return np.stack(filtered, axis=2)


def gaussian_blur_separable_reflect_u8(img_u8: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    if img_u8.dtype != np.uint8:
        raise ValueError("Expected uint8 image.")
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("ksize must be a positive odd integer.")
    if img_u8.ndim not in (2, 3):
        raise ValueError("Expected a 2D (grayscale) or 3D (color) image.")

    kernel = gaussian_kernel_1d(ksize, sigma)
    pad = ksize // 2

    def _convolve_rows(channel_f32: np.ndarray) -> np.ndarray:
        h, w = channel_f32.shape
        out = np.empty((h, w), dtype=np.float32)
        for y in range(h):
            row = np.pad(channel_f32[y, :], (pad, pad), mode="reflect")
            out[y, :] = np.convolve(row, kernel, mode="valid")
            if (y + 1) % 256 == 0 or (y + 1) == h:
                print(f"  Gaussian self-impl (H): row {y + 1} / {h}")
        return out

    def _convolve_cols(channel_f32: np.ndarray) -> np.ndarray:
        # Work on a contiguous transpose so each "column" is a contiguous row.
        trans = np.ascontiguousarray(channel_f32.T)
        w, h = trans.shape
        out_t = np.empty((w, h), dtype=np.float32)
        for x in range(w):
            col_as_row = np.pad(trans[x, :], (pad, pad), mode="reflect")
            out_t[x, :] = np.convolve(col_as_row, kernel, mode="valid")
            if (x + 1) % 256 == 0 or (x + 1) == w:
                print(f"  Gaussian self-impl (V): col {x + 1} / {w}")
        return out_t.T

    if img_u8.ndim == 2:
        ch = img_u8.astype(np.float32)
        tmp = _convolve_rows(ch)
        out_f = _convolve_cols(tmp)
        return np.clip(np.floor(out_f + 0.5), 0, 255).astype(np.uint8)

    out = np.empty_like(img_u8)
    img_f = img_u8.astype(np.float32)
    for c in range(img_u8.shape[2]):
        print(f"Gaussian self-impl: channel {c + 1} / {img_u8.shape[2]}")
        tmp = _convolve_rows(img_f[:, :, c])
        out_f = _convolve_cols(tmp)
        out[:, :, c] = np.clip(np.floor(out_f + 0.5), 0, 255).astype(np.uint8)

    return out


def compute_diff_stats(a_u8: np.ndarray, b_u8: np.ndarray) -> DiffStats:
    diff = a_u8.astype(np.int16) - b_u8.astype(np.int16)
    absdiff = np.abs(diff).astype(np.int16, copy=False)
    max_abs = int(absdiff.max(initial=0))
    mean_abs = float(absdiff.mean())
    nonzero_ratio = float(np.count_nonzero(absdiff) / absdiff.size)
    min_diff = int(diff.min(initial=0))
    max_diff = int(diff.max(initial=0))
    return DiffStats(
        max_abs=max_abs,
        mean_abs=mean_abs,
        nonzero_ratio=nonzero_ratio,
        min_diff=min_diff,
        max_diff=max_diff,
    )


def save_comparison(output_dir: Path, stem: str, a_u8: np.ndarray, b_u8: np.ndarray) -> DiffStats:
    if a_u8.shape != b_u8.shape:
        raise ValueError("A and B must have the same shape.")
    if a_u8.dtype != np.uint8 or b_u8.dtype != np.uint8:
        raise ValueError("A and B must be uint8.")

    stats = compute_diff_stats(a_u8, b_u8)

    cv2.imwrite(str(output_dir / f"{stem}_A_self.png"), a_u8)
    cv2.imwrite(str(output_dir / f"{stem}_B_builtin.png"), b_u8)

    absdiff = cv2.absdiff(a_u8, b_u8)
    cv2.imwrite(str(output_dir / f"{stem}_absdiff.png"), absdiff)

    diff = a_u8.astype(np.int16) - b_u8.astype(np.int16)
    if stats.max_abs == 0:
        diff_vis = np.full(a_u8.shape, 128, dtype=np.uint8)
    else:
        diff_vis = np.clip(diff.astype(np.float32) * (127.0 / stats.max_abs) + 128.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / f"{stem}_A_minus_B_vis.png"), diff_vis)

    return stats


def main() -> None:
    # Project root is one level above the Codes folder
    root_dir = Path(__file__).resolve().parents[1]
    output_dir = root_dir / "Outputs" / "Q5"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_lines: list[str] = []

    # ---------------------------
    # Q1 (Image_1): Average filter
    # Largest kernel: 15x15
    # ---------------------------
    img1_path = root_dir / "Images" / "Image_1.jpg"
    img1 = cv2.imread(str(img1_path), cv2.IMREAD_COLOR)
    if img1 is None:
        raise ValueError(f"Failed to read image: {img1_path}")

    k1 = 15
    b1 = cv2.blur(img1, (k1, k1))
    a1 = box_filter_integral_reflect(img1, k1)
    s1 = save_comparison(output_dir, f"Q1_Image_1_box_{k1}x{k1}", a1, b1)
    summary_lines.append(
        f"Q1 box {k1}x{k1}: max|A-B|={s1.max_abs}, mean|A-B|={s1.mean_abs:.6f}, "
        f"nonzero={s1.nonzero_ratio * 100:.4f}%, diff_range=[{s1.min_diff}, {s1.max_diff}]"
    )

    # ---------------------------
    # Q2 (Image_2): Median filter
    # Largest kernel: 11x11
    # ---------------------------
    img2_path = root_dir / "Images" / "Image_2.jpg"
    img2 = cv2.imread(str(img2_path), cv2.IMREAD_COLOR)
    if img2 is None:
        raise ValueError(f"Failed to read image: {img2_path}")

    k2 = 11
    b2 = cv2.medianBlur(img2, k2)
    print("Computing self-implemented median filter (this may take a while)...")
    a2 = median_filter_edge_u8(img2, k2, block_rows=128)
    s2 = save_comparison(output_dir, f"Q2_Image_2_median_{k2}x{k2}", a2, b2)
    summary_lines.append(
        f"Q2 median {k2}x{k2}: max|A-B|={s2.max_abs}, mean|A-B|={s2.mean_abs:.6f}, "
        f"nonzero={s2.nonzero_ratio * 100:.4f}%, diff_range=[{s2.min_diff}, {s2.max_diff}]"
    )

    # ---------------------------
    # Q3 (Image_3): Gaussian filter
    # Largest kernel: 15x15 (sigma auto like OpenCV when sigma=0)
    # ---------------------------
    img3_path = root_dir / "Images" / "Image_3.jpg"
    img3 = cv2.imread(str(img3_path), cv2.IMREAD_COLOR)
    if img3 is None:
        raise ValueError(f"Failed to read image: {img3_path}")

    k3 = 15
    b3 = cv2.GaussianBlur(img3, (k3, k3), sigmaX=0, sigmaY=0)
    sigma3 = opencv_default_gaussian_sigma(k3)
    print(f"Computing self-implemented Gaussian blur (ksize={k3}, sigma~{sigma3:.3f})...")
    a3 = gaussian_blur_separable_reflect_u8(img3, k3, sigma3)
    s3 = save_comparison(output_dir, f"Q3_Image_3_gaussian_{k3}x{k3}_sigma_auto", a3, b3)
    summary_lines.append(
        f"Q3 gaussian {k3}x{k3} (sigma auto {sigma3:.3f}): max|A-B|={s3.max_abs}, mean|A-B|={s3.mean_abs:.6f}, "
        f"nonzero={s3.nonzero_ratio * 100:.4f}%, diff_range=[{s3.min_diff}, {s3.max_diff}]"
    )

    summary_text = "\n".join(summary_lines) + "\n"
    (output_dir / "Q5_A_minus_B_summary.txt").write_text(summary_text, encoding="utf-8")

    print("Q5 comparisons completed.")
    print(f"Saved results to: {output_dir}")
    print(summary_text)
    print(
        "Explanation of differences (if any): small non-zero (A-B) values typically come from implementation details "
        "such as border handling (padding rule), floating-point vs fixed-point arithmetic, and rounding/saturation. "
        "If the same kernel, sigma, and border policy are matched, differences should be very small (often 0 or +/-1), "
        "and any larger differences usually appear near image borders."
    )


if __name__ == "__main__":
    main()
