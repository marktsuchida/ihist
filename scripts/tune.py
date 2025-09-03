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
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_script_dir = Path(__file__).parent.absolute()
_benchmark_dir = _script_dir / "../builddir/benchmarks"


def run_benchmark(pixel_type: str, bits: int, out_json: Path) -> None:
    pixel_type = {
        "mono": "mono",
        "abc": "rgb",
        "abcx": "rgbx",
    }[pixel_type]
    try:
        subprocess.run(
            [
                f"{_benchmark_dir}/ihist_bench",
                f"--benchmark_filter=^{pixel_type}/bits:{bits}/roi_type::two_d/.*/mt:0/",
                "--benchmark_repetitions=3",
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
    pixel_type = name_items.pop(0)
    name_items.remove("real_time")
    name_items.remove("process_time")
    name_dict = dict(i.split(":", 1) for i in name_items)

    return {
        "pixel_type": pixel_type,
        "bits": int(name_dict["bits"]),
        "roi_type": name_dict["roi_type"].lstrip(":"),
        "mask": bool(int(name_dict["mask"])),
        "mt": bool(int(name_dict["mt"])),
        "stripes": int(name_dict["stripes"]),
        "unrolls": int(name_dict["unrolls"]),
        "n_pixels": int(name_dict["size"]),
        "spread_percent": int(name_dict["spread"]),
        "mt_grain_size": int(name_dict["grainsize"]),
        "repetition_index": raw["repetition_index"],
        "pixels_per_second": raw["pixels_per_second"],
    }


def load_results(results_json: Path) -> pd.DataFrame:
    with open(results_json) as infile:
        data = json.load(infile)
    return pd.DataFrame(
        cooked
        for raw in data["benchmarks"]
        if (cooked := convert_json_entry(raw)) is not None
    )


def add_scores_to_plot(data: pd.DataFrame, **kwargs):
    # This will be called for each stripes/unrolls/mask value.
    masked = bool(data.iloc[0]["mask"])
    score_type = "masked" if masked else "nomask"

    def geo_mean(v):
        return np.exp(np.mean(np.log(v)))

    # Geometric mean of the different spreads (0, 1, 6, 25, 100) naturally
    # weight the middle values (1, 6, 25) a little more than the extreme values
    # (0, 100), which is probably what we want.
    score = geo_mean(data["pixels_per_second"] * 1e-9)
    axes = plt.gca()
    axes.text(
        0.02,
        0.98 if not masked else 0.92,
        f"{score_type}: {score:.2f}",
        transform=axes.transAxes,
        verticalalignment="top",
    )


def plot_results(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", palette="muted")
    grid = sns.FacetGrid(df, col="stripes", row="unrolls", hue="mask")
    grid.map_dataframe(
        sns.stripplot, x="spread_percent", y="pixels_per_second"
    )
    grid.map_dataframe(add_scores_to_plot)
    grid.set(ylim=(0, None))
    grid.figure.suptitle(f"{df.iloc[0]['pixel_type']}{df.iloc[0]['bits']}")
    plt.subplots_adjust(
        top=0.925, bottom=0.075
    )  # Prevent title from overlapping.
    plt.show()


def benchmark_and_plot(pixel_type: str, bits: int, *, show_plot: bool) -> None:
    results_file = Path(f"{_benchmark_dir}/{pixel_type}{bits}.json")
    if not results_file.exists():
        run_benchmark(pixel_type, bits, out_json=results_file)
    if show_plot:
        results = load_results(results_file)
        plot_results(results)


def all_pixel_formats() -> list[tuple[str, int]]:
    return list((t, b) for t in ("mono", "abc", "abcx") for b in (8, 12, 16))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="""\
            Run single-threaded benchmarks for the selected pixel format and
            plot results. If the results already exist, just plot."""
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--pixel-type",
        choices=("mono", "abc", "abcx"),
        metavar="TYPE",
        default="mono",
    )
    parser.add_argument("--bits", type=int, metavar="BITS", default=8)
    parser.add_argument("--plot", action="store_true", dest="plot")
    args = parser.parse_args()

    if args.all:
        for pixel_format in all_pixel_formats():
            benchmark_and_plot(*pixel_format, show_plot=args.plot)
    else:
        benchmark_and_plot(args.pixel_type, args.bits, show_plot=args.plot)


if __name__ == "__main__":
    main()
