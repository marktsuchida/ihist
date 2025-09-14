# This file is part of ihist
# Copyright 2025 Board of Regents of the University of Wisconsin System
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "seaborn",
# ]
# ///

import argparse
import json
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_script_dir = Path(__file__).parent.absolute()
_benchmark_dir = _script_dir / "../builddir/benchmarks"


def run_benchmark(
    pixel_type: str,
    bits: int,
    mask: bool,
    data_size: int,
    repetitions: int,
    out_json: Path,
) -> None:
    imask = 1 if mask else 0
    try:
        subprocess.run(
            [
                f"{_benchmark_dir}/api_bench",
                f"--benchmark_filter=/{pixel_type}/bits:{bits}/mask:{imask}/size:{data_size}/",
                f"--benchmark_repetitions={repetitions}",
                "--benchmark_enable_random_interleaving",
                f"--benchmark_out={out_json}",
                "--benchmark_counters_tabular",
                "--benchmark_time_unit=ms",
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        out_json.unlink(missing_ok=True)
        raise


def convert_json_entry(
    raw: dict[str, str | int | float],
) -> dict[str, str | int | float] | None:
    # We leave out aggregate entries (mean, median, stddev, cv), which are easy
    # enough to recompute anyway.
    if raw["run_type"] != "iteration":
        return None

    assert isinstance(raw["name"], str)
    name_items = raw["name"].split("/")
    implementation = name_items.pop(0)
    pixel_type = name_items.pop(0)
    name_items.remove("real_time")
    name_items.remove("process_time")
    name_dict = dict(i.split(":", 1) for i in name_items)

    return {
        "impl": implementation,
        "pixel_type": pixel_type,
        "bits": int(name_dict["bits"]),
        "mask": bool(int(name_dict["mask"])),
        "n_pixels": int(name_dict["size"]) ** 2,
        "spread_percent": int(name_dict["spread"]),
        "repetition_index": raw["repetition_index"],
        "pixels_per_second": raw["pixels_per_second"],
        "real_time_ms": raw["real_time"],
        "cpu_time_ms": raw["cpu_time"],
    }


def load_results(results_json: Path) -> pd.DataFrame:
    with open(results_json) as infile:
        data = json.load(infile)
    return pd.DataFrame(
        cooked
        for raw in data["benchmarks"]
        if (cooked := convert_json_entry(raw)) is not None
    )


def plot_results(df: pd.DataFrame) -> None:
    data = df.copy()

    def geometric_mean(v):
        return np.prod(v) ** (1 / len(v))

    # OpenCV is not always MT-enabled, so skip opencv-mt in that case.
    def calc_opencv_speedup(row):
        if row["impl"] != "opencv-mt":
            return pd.NA
        single_threaded = data[
            (data["impl"] == "opencv")
            & (data["pixel_type"] == row["pixel_type"])
            & (data["bits"] == row["bits"])
            & (data["mask"] == row["mask"])
            & (data["n_pixels"] == row["n_pixels"])
            & (data["spread_percent"] == row["spread_percent"])
            & (data["repetition_index"] == row["repetition_index"])
        ]
        assert len(single_threaded) == 1
        st_row = single_threaded.iloc[0]
        return st_row["real_time_ms"] / row["real_time_ms"]

    opencv_speedups_mask0 = (
        data[np.logical_not(data["mask"])]
        .apply(calc_opencv_speedup, axis=1)
        .dropna()
    )
    opencv_speedups_mask1 = (
        data[data["mask"]].apply(calc_opencv_speedup, axis=1).dropna()
    )
    opencv_speedup_cutoff = 1.05
    if (
        len(opencv_speedups_mask0)
        and geometric_mean(opencv_speedups_mask0) < opencv_speedup_cutoff
        and len(opencv_speedups_mask1)
        and geometric_mean(opencv_speedups_mask1) < opencv_speedup_cutoff
    ):
        data = data.drop(data[data["impl"] == "opencv-mt"].index)

    impls = sorted(data["impl"].unique())
    data["impl"] = pd.Categorical(data["impl"], categories=impls)

    spreads = sorted(data["spread_percent"].unique())
    data["spread_percent"] = pd.Categorical(
        data["spread_percent"], categories=spreads
    )

    sns.set_theme(style="whitegrid", palette="muted")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey="row")
    for i, mask in enumerate((False, True)):
        sns.stripplot(
            data[data["mask"] == mask],
            x="spread_percent",
            y="real_time_ms",
            hue="impl",
            alpha=0.75,
            ax=axes[0, i],
        )
        axes[0, i].set_title(f"mask {int(mask)}")
        sns.stripplot(
            data[data["mask"] == mask],
            x="spread_percent",
            y="pixels_per_second",
            hue="impl",
            alpha=0.75,
            ax=axes[1, i],
        )
    axes[0, 0].set_ylim(bottom=0)
    axes[1, 0].set_ylim(bottom=0)

    for ax in axes.flat:
        ax.get_legend().remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        title="impl",
    )

    pixel_type = data.iloc[0]["pixel_type"]
    bits = data.iloc[0]["bits"]
    n_pixels = data.iloc[0]["n_pixels"]
    size = int(np.sqrt(n_pixels))
    fig.suptitle(f"{pixel_type}{bits} | {size}x{size}")

    plt.subplots_adjust(
        top=0.925, bottom=0.075
    )  # Prevent title from overlapping.
    plt.show()


def results_file(
    pixel_type: str, bits: int, mask: bool, data_size: int
) -> Path:
    imask = 1 if mask else 0
    return Path(
        f"{_benchmark_dir}/benchmarks_api_{pixel_type}{bits}_mask{imask}_size{data_size}.json"
    )


def all_pixel_formats() -> list[tuple[str, int]]:
    return list((t, b) for t in ("mono", "abc", "abcx") for b in (8, 12, 16))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all",
        action="store_true",
        help="all pixel formats (but single input data size)",
    )
    parser.add_argument(
        "--pixel-type",
        choices=("mono", "abc", "abcx"),
        metavar="TYPE",
        default="mono",
    )
    parser.add_argument("--bits", type=int, metavar="BITS", default=8)
    parser.add_argument(
        "--size",
        type=int,
        metavar="SIZE",
        dest="data_size",
        default=4096,
        help="input data size (square root of pixel count)",
    )
    parser.add_argument("--repetitions", type=int, metavar="N", default=3)
    parser.add_argument("--plot", action="store_true", dest="plot")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    pixel_formats = (
        all_pixel_formats() if args.all else [(args.pixel_type, args.bits)]
    )

    for pixel_format in pixel_formats:
        for mask in (False, True):
            f = results_file(*pixel_format, mask, args.data_size)
            if args.rerun or not f.exists():
                run_benchmark(
                    *pixel_format,
                    mask,
                    data_size=args.data_size,
                    repetitions=args.repetitions,
                    out_json=f,
                )
    for pixel_format in pixel_formats:
        unmasked_f = results_file(*pixel_format, False, args.data_size)
        masked_f = results_file(*pixel_format, True, args.data_size)
        unmasked_df = load_results(unmasked_f)
        masked_df = load_results(masked_f)
        df = pd.concat([unmasked_df, masked_df], ignore_index=True)
        if args.plot:
            plot_results(df)


if __name__ == "__main__":
    main()
