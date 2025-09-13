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
    stripes: int,
    unrolls: int,
    repetitions: int,
    out_json: Path,
) -> None:
    imask = 1 if mask else 0
    try:
        subprocess.run(
            [
                f"{_benchmark_dir}/ihist_bench",
                f"--benchmark_filter=^{pixel_type}/bits:{bits}/input:2d/mask:{imask}/mt:./stripes:{stripes}/unrolls:{unrolls}/",
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
    pixel_type = name_items.pop(0)
    name_items.remove("real_time")
    name_items.remove("process_time")
    name_dict = dict(i.split(":", 1) for i in name_items)

    return {
        "pixel_type": pixel_type,
        "bits": int(name_dict["bits"]),
        "input": name_dict["input"],
        "mask": bool(int(name_dict["mask"])),
        "mt": bool(int(name_dict["mt"])),
        "stripes": int(name_dict["stripes"]),
        "unrolls": int(name_dict["unrolls"]),
        "n_pixels": int(name_dict["size"]),
        "spread_percent": int(name_dict["spread"]),
        "grain_size": int(name_dict["grainsize"]),
        "repetition_index": raw["repetition_index"],
        "pixels_per_second": raw["pixels_per_second"],
        "real_time": raw["real_time"],
        "cpu_time": raw["cpu_time"],
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

    def calc_eff(row):
        single_threaded = data[
            (data["pixel_type"] == row["pixel_type"])
            & (data["bits"] == row["bits"])
            & (data["input"] == row["input"])
            & (data["mask"] == row["mask"])
            & np.logical_not(data["mt"])
            & (data["stripes"] == row["stripes"])
            & (data["unrolls"] == row["unrolls"])
            & (data["spread_percent"] == row["spread_percent"])
            & (data["repetition_index"] == row["repetition_index"])
        ]
        assert len(single_threaded) == 1
        st_row = single_threaded.iloc[0]
        return st_row["cpu_time"] / row["cpu_time"]

    data["efficiency"] = data.apply(calc_eff, axis=1)

    # Treat grain size as categories, not continuous variable.
    grain_sizes = sorted(data['grain_size'].unique())
    grain_sizes.remove(0)
    grain_sizes = list(str(gs) for gs in grain_sizes) + ["st"]
    data['grain_size'] = pd.Categorical(
        data['grain_size'].astype(str),
        categories=[str(x) for x in grain_sizes]
    )
    data.loc[np.logical_not(data["mt"]), "grain_size"] = "st"
    data["grain_size"] = data["grain_size"].astype("category")

    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey="row")

    masking = (False, True)
    for i, mask in enumerate(masking):
        sns.stripplot(
            data[data["mask"] == mask],
            x="spread_percent",
            y="pixels_per_second",
            hue="grain_size",
            ax=axes[0, i],
        )
        axes[0, i].set_title(f"mask {int(mask)}")
        sns.stripplot(
            data[data["mask"] == mask],
            x="spread_percent",
            y="efficiency",
            hue="grain_size",
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
        title="grain_size",
    )

    fig.suptitle(f"{data.iloc[0]['pixel_type']}{data.iloc[0]['bits']}")
    plt.subplots_adjust(
        top=0.925, bottom=0.075
    )  # Prevent title from overlapping.
    plt.show()


def results_file(pixel_type: str, bits: int, mask: bool) -> Path:
    imask = 1 if mask else 0
    return Path(f"{_benchmark_dir}/mt_{pixel_type}{bits}_mask{imask}.json")


def all_pixel_formats() -> list[tuple[str, int]]:
    return list((t, b) for t in ("mono", "abc", "abcx") for b in (8, 12, 16))


def read_tuning(
    pixel_type: str, bits: int, mask: bool, file: Path
) -> tuple[int, int]:
    imask = 1 if mask else 0
    ptrn = f"TUNE\\({pixel_type}, {bits}, {imask}, ([0-9]+), ([0-9]+)\\)"
    with open(file) as f:
        while line := f.readline():
            if m := re.match(ptrn, line):
                return int(m.group(1)), int(m.group(2))
    raise ValueError(
        f"Tuning for {pixel_type}{bits}_mask{imask} not found in {file}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning", metavar="FILE", required=True)
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--pixel-type",
        choices=("mono", "abc", "abcx"),
        metavar="TYPE",
        default="mono",
    )
    parser.add_argument("--bits", type=int, metavar="BITS", default=8)
    parser.add_argument("--repetitions", type=int, metavar="N", default=5)
    parser.add_argument("--plot", action="store_true", dest="plot")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    pixel_formats = (
        all_pixel_formats() if args.all else [(args.pixel_type, args.bits)]
    )

    for pixel_format in pixel_formats:
        for mask in (False, True):
            stripes, unrolls = read_tuning(*pixel_format, mask, args.tuning)
            f = results_file(*pixel_format, mask)
            if args.rerun or not f.exists():
                run_benchmark(
                    *pixel_format,
                    mask,
                    stripes,
                    unrolls,
                    repetitions=args.repetitions,
                    out_json=f,
                )
    for pixel_format in pixel_formats:
        unmasked_f = results_file(*pixel_format, False)
        masked_f = results_file(*pixel_format, True)
        unmasked_df = load_results(unmasked_f)
        masked_df = load_results(masked_f)
        df = pd.concat([unmasked_df, masked_df])
        if args.plot:
            plot_results(df)


if __name__ == "__main__":
    main()
