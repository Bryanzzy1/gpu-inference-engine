"""Plot the jitter autopsy from an autopsy CSV.

Usage:
    python plot_autopsy.py ../results/autopsy.csv

For each label in the CSV (e.g. "naive", "naive-locked"), draws a grouped bar chart of
p50/p99/p999 per stage (h2d, launch, compute, d2h), so you can see which stage owns the
tail and how much the tail exceeds the median. Writes autopsy_<label>.png next to the CSV.

Run the autopsy twice, clocks locked vs unlocked, with different labels, to compare how
much of each stage's tail is clock-ramp jitter.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STAGES = ["h2d", "launch", "compute", "d2h", "total"]


def main(path):
    df = pd.read_csv(path)
    df["p50_us"] = df["p50_ns"] / 1000.0
    df["p99_us"] = df["p99_ns"] / 1000.0
    df["p999_us"] = df["p999_ns"] / 1000.0
    out_dir = Path(path).parent

    for label in df["label"].unique():
        sub = df[df["label"] == label].set_index("stage").reindex(STAGES)
        x = np.arange(len(STAGES))
        w = 0.27

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - w, sub["p50_us"], w, label="p50")
        ax.bar(x, sub["p99_us"], w, label="p99")
        ax.bar(x + w, sub["p999_us"], w, label="p999")
        ax.set_xticks(x, STAGES)
        ax.set_ylabel("latency (us)")
        ax.set_title(f"Jitter autopsy: {label}")
        ax.legend()
        fig.tight_layout()

        out = out_dir / f"autopsy_{label}.png"
        fig.savefig(out, dpi=120)
        print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python plot_autopsy.py <autopsy.csv>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
