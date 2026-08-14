"""Plot the latency histogram(s) from a bench_hist CSV.

Usage:
    python plot_hist.py ../data/hist.csv

Draws a log-x bar chart per label (backend). The shape is the point: a tight spike
at the median with a thin long tail is what "good p50, bad p999" looks like, and it
argues the tail case far better than a single number. Writes hist_<label>.png.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(path):
    if not Path(path).exists():
        sys.exit(f"error: no such file: {path}\nrun bench_hist first to produce it")
    df = pd.read_csv(path)
    out_dir = Path(path).parent

    for label in df["label"].unique():
        sub = df[df["label"] == label]
        centers = np.sqrt(sub["bucket_lo_ns"] * sub["bucket_hi_ns"])  # geometric mid
        widths = sub["bucket_hi_ns"] - sub["bucket_lo_ns"]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(centers, sub["count"], width=widths * 0.9, align="center")
        ax.set_xscale("log")
        ax.set_xlabel("per-inference latency (ns, log scale)")
        ax.set_ylabel("count")
        ax.set_title(f"Latency distribution: {label}")
        fig.tight_layout()

        out = out_dir / f"hist_{label}.png"
        fig.savefig(out, dpi=120)
        print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python plot_hist.py <hist.csv>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
