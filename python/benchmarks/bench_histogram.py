# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ihist",
#     "numpy",
#     "opencv-python-headless",
#     "scikit-image",
#     "matplotlib",
#     "fast-histogram",
#     "boost-histogram",
# ]
# ///
"""Benchmark ihist against other Python histogram libraries.

Usage:
    uv run python/benchmarks/bench_histogram.py [--output-dir DIR] [--repeats N]

Generates comparison plots saved to the output directory
(default: python/benchmarks/results).
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import boost_histogram as bh
import cv2
import fast_histogram
import matplotlib.pyplot as plt
import numpy as np
import skimage.exposure

import ihist

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

# Image side lengths → total pixels will be size*size
IMAGE_SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 10000]
WARMUP_RUNS = 2
DEFAULT_REPEATS = 7  # median of N runs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(
    height: int, width: int, channels: int, dtype: np.dtype
) -> np.ndarray:
    info = np.iinfo(dtype)
    rng = np.random.default_rng(42)
    return rng.integers(
        info.min,
        info.max,
        size=(height, width) if channels == 1 else (height, width, channels),
        dtype=dtype,
        endpoint=True,
    )


def _make_mask(height: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(123)
    return rng.integers(0, 2, size=(height, width), dtype=np.uint8)


@dataclass
class BenchResult:
    name: str
    sizes: list[int] = field(default_factory=list)
    times_us: list[float] = field(default_factory=list)


def _bench(func: Callable[[], None], repeats: int) -> float:
    """Return median execution time in microseconds."""
    for _ in range(WARMUP_RUNS):
        func()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Benchmark scenarios
# ---------------------------------------------------------------------------


def bench_gray_uint8(sizes: list[int], repeats: int) -> list[BenchResult]:
    """Grayscale uint8 histogram: all libraries."""
    results = {
        name: BenchResult(name)
        for name in [
            "ihist",
            "ihist (single-thread)",
            "numpy.bincount",
            "numpy.histogram",
            "cv2.calcHist",
            "skimage.exposure.histogram",
            "fast_histogram",
            "boost-histogram",
        ]
    }

    for size in sizes:
        npix = size * size
        img = _make_image(size, size, 1, np.uint8)
        img_flat = img.ravel()

        benchmarks: dict[str, callable] = {
            "ihist": lambda i=img: ihist.histogram(i),
            "ihist (single-thread)": lambda i=img: ihist.histogram(
                i, parallel=False
            ),
            "numpy.bincount": lambda f=img_flat: np.bincount(f, minlength=256),
            "numpy.histogram": lambda f=img_flat: np.histogram(
                f, bins=256, range=(0, 256)
            ),
            "cv2.calcHist": lambda i=img: cv2.calcHist(
                [i], [0], None, [256], [0, 256]
            ),
            "skimage.exposure.histogram": lambda i=img: (
                skimage.exposure.histogram(i, nbins=256)
            ),
            "fast_histogram": lambda f=img_flat: fast_histogram.histogram1d(
                f.astype(np.float64), bins=256, range=[0, 256]
            ),
            "boost-histogram": lambda f=img_flat: bh.numpy.histogram(
                f, bins=256, range=(0, 256)
            ),
        }

        for name, func in benchmarks.items():
            t = _bench(func, repeats)
            results[name].sizes.append(npix)
            results[name].times_us.append(t)
            print(f"  {name:>35s}  {npix:>10,} px  {t:>10.1f} us")

    return list(results.values())


def bench_rgb_uint8(sizes: list[int], repeats: int) -> list[BenchResult]:
    """RGB uint8 histogram: libraries that support multi-channel."""
    results = {
        name: BenchResult(name)
        for name in [
            "ihist",
            "ihist (single-thread)",
            "numpy.bincount (per-ch loop)",
            "cv2.calcHist (per-ch loop)",
        ]
    }

    for size in sizes:
        npix = size * size
        img = _make_image(size, size, 3, np.uint8)

        def _np_bincount_rgb(i=img):
            for c in range(3):
                np.bincount(i[:, :, c].ravel(), minlength=256)

        def _cv2_rgb(i=img):
            for c in range(3):
                cv2.calcHist([i], [c], None, [256], [0, 256])

        benchmarks: dict[str, callable] = {
            "ihist": lambda i=img: ihist.histogram(i),
            "ihist (single-thread)": lambda i=img: ihist.histogram(
                i, parallel=False
            ),
            "numpy.bincount (per-ch loop)": _np_bincount_rgb,
            "cv2.calcHist (per-ch loop)": _cv2_rgb,
        }

        for name, func in benchmarks.items():
            t = _bench(func, repeats)
            results[name].sizes.append(npix)
            results[name].times_us.append(t)
            print(f"  {name:>35s}  {npix:>10,} px  {t:>10.1f} us")

    return list(results.values())


def bench_gray_uint16(sizes: list[int], repeats: int) -> list[BenchResult]:
    """Grayscale uint16 histogram (12-bit range)."""
    results = {
        name: BenchResult(name)
        for name in [
            "ihist (12-bit)",
            "ihist (12-bit, single-thread)",
            "numpy.bincount",
            "numpy.histogram",
            "cv2.calcHist",
            "skimage.exposure.histogram",
            "fast_histogram",
            "boost-histogram",
        ]
    }

    for size in sizes:
        npix = size * size
        img = _make_image(size, size, 1, np.uint16)
        # Simulate 12-bit camera data
        img = (img >> 4).astype(np.uint16)
        img_flat = img.ravel()

        benchmarks: dict[str, callable] = {
            "ihist (12-bit)": lambda i=img: ihist.histogram(i, bits=12),
            "ihist (12-bit, single-thread)": lambda i=img: ihist.histogram(
                i, bits=12, parallel=False
            ),
            "numpy.bincount": lambda f=img_flat: np.bincount(
                f, minlength=4096
            ),
            "numpy.histogram": lambda f=img_flat: np.histogram(
                f, bins=4096, range=(0, 4096)
            ),
            "cv2.calcHist": lambda i=img: cv2.calcHist(
                [i], [0], None, [4096], [0, 4096]
            ),
            "skimage.exposure.histogram": lambda f=img_flat: (
                skimage.exposure.histogram(f, nbins=4096)
            ),
            "fast_histogram": lambda f=img_flat: fast_histogram.histogram1d(
                f.astype(np.float64), bins=4096, range=[0, 4096]
            ),
            "boost-histogram": lambda f=img_flat: bh.numpy.histogram(
                f, bins=4096, range=(0, 4096)
            ),
        }

        for name, func in benchmarks.items():
            t = _bench(func, repeats)
            results[name].sizes.append(npix)
            results[name].times_us.append(t)
            print(f"  {name:>35s}  {npix:>10,} px  {t:>10.1f} us")

    return list(results.values())


def bench_gray_uint16_full(
    sizes: list[int], repeats: int
) -> list[BenchResult]:
    """Grayscale uint16 histogram (full 16-bit range)."""
    results = {
        name: BenchResult(name)
        for name in [
            "ihist (16-bit)",
            "ihist (16-bit, single-thread)",
            "numpy.bincount",
            "numpy.histogram",
            "cv2.calcHist",
            "skimage.exposure.histogram",
            "fast_histogram",
            "boost-histogram",
        ]
    }

    for size in sizes:
        npix = size * size
        img = _make_image(size, size, 1, np.uint16)
        img_flat = img.ravel()

        benchmarks: dict[str, callable] = {
            "ihist (16-bit)": lambda i=img: ihist.histogram(i, bits=16),
            "ihist (16-bit, single-thread)": lambda i=img: ihist.histogram(
                i, bits=16, parallel=False
            ),
            "numpy.bincount": lambda f=img_flat: np.bincount(
                f, minlength=65536
            ),
            "numpy.histogram": lambda f=img_flat: np.histogram(
                f, bins=65536, range=(0, 65536)
            ),
            "cv2.calcHist": lambda i=img: cv2.calcHist(
                [i], [0], None, [65536], [0, 65536]
            ),
            "skimage.exposure.histogram": lambda f=img_flat: (
                skimage.exposure.histogram(f, nbins=65536)
            ),
            "fast_histogram": lambda f=img_flat: fast_histogram.histogram1d(
                f.astype(np.float64), bins=65536, range=[0, 65536]
            ),
            "boost-histogram": lambda f=img_flat: bh.numpy.histogram(
                f, bins=65536, range=(0, 65536)
            ),
        }

        for name, func in benchmarks.items():
            t = _bench(func, repeats)
            results[name].sizes.append(npix)
            results[name].times_us.append(t)
            print(f"  {name:>35s}  {npix:>10,} px  {t:>10.1f} us")

    return list(results.values())


def bench_masked_uint8(sizes: list[int], repeats: int) -> list[BenchResult]:
    """Grayscale uint8 with mask."""
    results = {
        name: BenchResult(name)
        for name in [
            "ihist (masked)",
            "ihist (masked, single-thread)",
            "numpy.bincount (masked)",
            "cv2.calcHist (masked)",
        ]
    }

    for size in sizes:
        npix = size * size
        img = _make_image(size, size, 1, np.uint8)
        mask = _make_mask(size, size)
        img_flat = img.ravel()
        mask_flat = mask.ravel().astype(bool)

        benchmarks: dict[str, callable] = {
            "ihist (masked)": lambda i=img, m=mask: ihist.histogram(i, mask=m),
            "ihist (masked, single-thread)": lambda i=img, m=mask: (
                ihist.histogram(i, mask=m, parallel=False)
            ),
            "numpy.bincount (masked)": lambda f=img_flat, mf=mask_flat: (
                np.bincount(f[mf], minlength=256)
            ),
            "cv2.calcHist (masked)": lambda i=img, m=mask: cv2.calcHist(
                [i], [0], m, [256], [0, 256]
            ),
        }

        for name, func in benchmarks.items():
            t = _bench(func, repeats)
            results[name].sizes.append(npix)
            results[name].times_us.append(t)
            print(f"  {name:>35s}  {npix:>10,} px  {t:>10.1f} us")

    return list(results.values())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Color scheme: ihist variants in blue shades, others get distinct colors
COLORS = {
    "ihist": "#1f77b4",
    "ihist (single-thread)": "#6baed6",
    "ihist (12-bit)": "#1f77b4",
    "ihist (12-bit, single-thread)": "#6baed6",
    "ihist (16-bit)": "#1f77b4",
    "ihist (16-bit, single-thread)": "#6baed6",
    "ihist (masked)": "#1f77b4",
    "ihist (masked, single-thread)": "#6baed6",
    "numpy.bincount": "#2ca02c",
    "numpy.bincount (per-ch loop)": "#2ca02c",
    "numpy.bincount (masked)": "#2ca02c",
    "numpy.histogram": "#98df8a",
    "cv2.calcHist": "#d62728",
    "cv2.calcHist (per-ch loop)": "#d62728",
    "cv2.calcHist (masked)": "#d62728",
    "skimage.exposure.histogram": "#9467bd",
    "fast_histogram": "#ff7f0e",
    "boost-histogram": "#8c564b",
}

MARKERS = {
    "ihist": "o",
    "ihist (single-thread)": "s",
    "ihist (12-bit)": "o",
    "ihist (12-bit, single-thread)": "s",
    "ihist (16-bit)": "o",
    "ihist (16-bit, single-thread)": "s",
    "ihist (masked)": "o",
    "ihist (masked, single-thread)": "s",
    "numpy.bincount": "^",
    "numpy.bincount (per-ch loop)": "^",
    "numpy.bincount (masked)": "^",
    "numpy.histogram": "v",
    "cv2.calcHist": "D",
    "cv2.calcHist (per-ch loop)": "D",
    "cv2.calcHist (masked)": "D",
    "skimage.exposure.histogram": "p",
    "fast_histogram": "h",
    "boost-histogram": "H",
}


def _plot_scenario(
    results: list[BenchResult],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for r in results:
        ax.plot(
            r.sizes,
            r.times_us,
            marker=MARKERS.get(r.name, "o"),
            color=COLORS.get(r.name, "gray"),
            label=r.name,
            linewidth=2,
            markersize=6,
        )

    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Number of pixels", fontsize=12)
    ax.set_ylabel("Time (µs)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _plot_speedup(
    results: list[BenchResult],
    baseline_name: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot speedup of other libraries relative to a baseline (higher = baseline is faster)."""
    baseline = next((r for r in results if r.name == baseline_name), None)
    if baseline is None:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for r in results:
        if r.name == baseline_name:
            continue
        speedups = [
            b / t for b, t in zip(baseline.times_us, r.times_us, strict=False)
        ]
        ax.plot(
            r.sizes,
            speedups,
            marker=MARKERS.get(r.name, "o"),
            color=COLORS.get(r.name, "gray"),
            label=f"{baseline_name} / {r.name}",
            linewidth=2,
            markersize=6,
        )

    ax.axhline(
        y=1.0,
        color="black",
        linestyle="--",
        alpha=0.5,
        label=f"{baseline_name} (baseline)",
    )
    ax.set_xscale("log", base=10)
    ax.set_xlabel("Number of pixels", fontsize=12)
    ax.set_ylabel(f"Speedup vs {baseline_name}", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BENCHMARKS = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ihist vs other histogram libraries"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BENCHMARKS / "results",
        help="Directory for output plots (default: benchmarks/results)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Number of timed runs per measurement (default: {DEFAULT_REPEATS})",
    )
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    repeats: int = args.repeats

    all_data: dict[str, list[dict]] = {}

    # --- Grayscale uint8 ---
    print("\n=== Grayscale uint8 ===")
    gray8 = bench_gray_uint8(IMAGE_SIZES, repeats)
    _plot_scenario(
        gray8, "Grayscale uint8 Histogram", out_dir / "gray_uint8_absolute.png"
    )
    _plot_speedup(
        gray8,
        "numpy.histogram",
        "Speedup vs numpy.histogram — Grayscale uint8",
        out_dir / "gray_uint8_speedup.png",
    )
    all_data["gray_uint8"] = [
        {"name": r.name, "sizes": r.sizes, "times_us": r.times_us}
        for r in gray8
    ]

    # --- RGB uint8 ---
    print("\n=== RGB uint8 ===")
    rgb8 = bench_rgb_uint8(IMAGE_SIZES, repeats)
    _plot_scenario(
        rgb8,
        "RGB uint8 Histogram (3 channels)",
        out_dir / "rgb_uint8_absolute.png",
    )
    _plot_speedup(
        rgb8,
        "numpy.bincount (per-ch loop)",
        "Speedup vs numpy.bincount — RGB uint8",
        out_dir / "rgb_uint8_speedup.png",
    )
    all_data["rgb_uint8"] = [
        {"name": r.name, "sizes": r.sizes, "times_us": r.times_us}
        for r in rgb8
    ]

    # --- Grayscale uint16 (12-bit) ---
    print("\n=== Grayscale uint16 (12-bit) ===")
    gray16 = bench_gray_uint16(IMAGE_SIZES, repeats)
    _plot_scenario(
        gray16,
        "Grayscale uint16 (12-bit) Histogram",
        out_dir / "gray_uint16_absolute.png",
    )
    _plot_speedup(
        gray16,
        "numpy.histogram",
        "Speedup vs numpy.histogram — uint16 12-bit",
        out_dir / "gray_uint16_speedup.png",
    )
    all_data["gray_uint16"] = [
        {"name": r.name, "sizes": r.sizes, "times_us": r.times_us}
        for r in gray16
    ]

    # --- Grayscale uint16 (16-bit) ---
    print("\n=== Grayscale uint16 (16-bit) ===")
    gray16f = bench_gray_uint16_full(IMAGE_SIZES, repeats)
    _plot_scenario(
        gray16f,
        "Grayscale uint16 (16-bit) Histogram",
        out_dir / "gray_uint16_full_absolute.png",
    )
    _plot_speedup(
        gray16f,
        "numpy.histogram",
        "Speedup vs numpy.histogram — uint16 16-bit",
        out_dir / "gray_uint16_full_speedup.png",
    )
    all_data["gray_uint16_full"] = [
        {"name": r.name, "sizes": r.sizes, "times_us": r.times_us}
        for r in gray16f
    ]

    # --- Masked uint8 ---
    print("\n=== Masked Grayscale uint8 ===")
    masked = bench_masked_uint8(IMAGE_SIZES, repeats)
    _plot_scenario(
        masked,
        "Masked Grayscale uint8 Histogram",
        out_dir / "masked_uint8_absolute.png",
    )
    _plot_speedup(
        masked,
        "numpy.bincount (masked)",
        "Speedup vs numpy.bincount — Masked uint8",
        out_dir / "masked_uint8_speedup.png",
    )
    all_data["masked_uint8"] = [
        {"name": r.name, "sizes": r.sizes, "times_us": r.times_us}
        for r in masked
    ]

    # Save raw data as JSON for reproducibility
    json_path = out_dir / "benchmark_data.json"
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\n  Raw data saved: {json_path}")

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
