from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageMetrics:
    mean: float
    std: float
    entropy_bits: float
    laplacian_var: float
    p5: float
    p95: float
    clip_ratio: float


def entropy_bits(img_u8: np.ndarray) -> float:
    hist = np.bincount(img_u8.ravel(), minlength=256).astype(np.float64)
    p = hist / float(hist.sum())
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def compute_metrics(img_u8: np.ndarray) -> ImageMetrics:
    img_f = img_u8.astype(np.float32)
    p5, p95 = np.percentile(img_f, [5, 95])
    clip_ratio = float(((img_u8 == 0) | (img_u8 == 255)).mean())
    lap = cv2.Laplacian(img_u8, cv2.CV_64F).var()
    return ImageMetrics(
        mean=float(img_f.mean()),
        std=float(img_f.std()),
        entropy_bits=entropy_bits(img_u8),
        laplacian_var=float(lap),
        p5=float(p5),
        p95=float(p95),
        clip_ratio=clip_ratio,
    )


def contrast_stretch_percentile(
    img_u8: np.ndarray, low_percent: float = 1.0, high_percent: float = 99.0
) -> tuple[np.ndarray, float, float]:
    if img_u8.dtype != np.uint8 or img_u8.ndim != 2:
        raise ValueError("Expected a 2D uint8 image.")
    if not (0.0 <= low_percent < high_percent <= 100.0):
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 100.")

    p_low, p_high = np.percentile(img_u8.astype(np.float32), [low_percent, high_percent])
    if p_high - p_low < 1e-6:
        return img_u8.copy(), float(p_low), float(p_high)

    stretched = (img_u8.astype(np.float32) - float(p_low)) * (255.0 / float(p_high - p_low))
    stretched_u8 = np.clip(np.rint(stretched), 0, 255).astype(np.uint8)
    return stretched_u8, float(p_low), float(p_high)


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


def to_bgr(img_u8: np.ndarray) -> np.ndarray:
    if img_u8.ndim == 2:
        return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    return img_u8


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    image_path = root_dir / "Images" / "Image_5.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_u8 = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_u8 is None:
        raise ValueError(f"Failed to read image: {image_path}")

    output_dir = root_dir / "Outputs" / "Q9"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Histogram Equalization (global)
    he_u8 = cv2.equalizeHist(img_u8)

    # 2) Contrast Stretching (percentile-based linear stretch)
    cs_u8, p_low, p_high = contrast_stretch_percentile(img_u8, low_percent=1.0, high_percent=99.0)

    # 3) Noise Reduction (Gaussian filtering)
    gauss_ksize = 5
    gauss_sigma = 1.0
    ga_u8 = cv2.GaussianBlur(img_u8, (gauss_ksize, gauss_ksize), gauss_sigma)

    # Save individual results
    cv2.imwrite(str(output_dir / "Image_5_original.png"), img_u8)
    cv2.imwrite(str(output_dir / "Image_5_hist_equalized.png"), he_u8)
    cv2.imwrite(str(output_dir / "Image_5_contrast_stretched.png"), cs_u8)
    cv2.imwrite(str(output_dir / "Image_5_gaussian_filtered.png"), ga_u8)

    # Montage for comparison
    tiles = [
        add_title(to_bgr(img_u8), "Original (Image_5.jpg)"),
        add_title(to_bgr(he_u8), "Histogram Equalization"),
        add_title(to_bgr(cs_u8), f"Contrast Stretch (1-99%)"),
        add_title(to_bgr(ga_u8), f"Gaussian Filter (k={gauss_ksize}, sigma={gauss_sigma})"),
    ]
    top = np.hstack([tiles[0], tiles[1]])
    bottom = np.hstack([tiles[2], tiles[3]])
    montage = np.vstack([top, bottom])
    cv2.imwrite(str(output_dir / "Q9_comparison.png"), montage)

    # Metrics + brief comparison (saved to a text file for the report)
    metrics = {
        "Original": compute_metrics(img_u8),
        "HistogramEqualization": compute_metrics(he_u8),
        "ContrastStretching": compute_metrics(cs_u8),
        "GaussianFilter": compute_metrics(ga_u8),
    }

    lines = []
    lines.append("MRI Enhancement (Image_5.jpg)")
    lines.append(f"Input: {image_path.name} ({img_u8.shape[1]}x{img_u8.shape[0]})")
    lines.append("")
    lines.append("Contrast stretching parameters:")
    lines.append(f"  Percentiles: 1% -> {p_low:.2f}, 99% -> {p_high:.2f}")
    lines.append("")
    lines.append("Gaussian filtering parameters:")
    lines.append(f"  Kernel size: {gauss_ksize}x{gauss_ksize}, sigma={gauss_sigma}")
    lines.append("")
    lines.append("Metrics (mean, std, entropy, laplacian_var, p5, p95, clip_ratio):")
    for name, m in metrics.items():
        lines.append(
            f"  {name:22s}: mean={m.mean:7.2f}, std={m.std:7.2f}, ent={m.entropy_bits:5.2f} bits, "
            f"lapVar={m.laplacian_var:9.2f}, p5={m.p5:6.1f}, p95={m.p95:6.1f}, clip={m.clip_ratio*100:5.2f}%"
        )
    lines.append("")
    lines.append("Observations:")
    lines.append(
        "  - Histogram equalization increases global contrast strongly, but can also amplify noise and make some "
        "regions look over-enhanced (non-uniform brightness)."
    )
    lines.append(
        "  - Contrast stretching (percentile-based) improves dynamic range more gently, preserving relative "
        "intensity ordering and usually keeping the image more natural for reading."
    )
    lines.append(
        "  - Gaussian filtering reduces noise, but blurs edges and fine structures, so it is best used as a "
        "pre-processing step rather than a standalone enhancement."
    )
    lines.append("")
    lines.append(
        "Conclusion (clinical diagnosis): Among these single-step methods, contrast stretching provides the best "
        "balance between improved contrast and preservation of anatomical detail. Histogram equalization is "
        "useful for visualization but may be too aggressive; Gaussian filtering improves smoothness but can "
        "remove subtle diagnostic edges."
    )
    params_text = "\n".join(lines) + "\n"
    (output_dir / "Q9_results.txt").write_text(params_text, encoding="utf-8")

    print("Q9 completed.")
    print(f"Saved results to: {output_dir}")
    print(params_text)


if __name__ == "__main__":
    main()

