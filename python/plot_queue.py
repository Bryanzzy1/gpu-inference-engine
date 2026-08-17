"""Plot the offered-load sweep from queue_load.

Usage:
    python plot_queue.py ../results/queue_load.csv

Produces queue_load.png next to the CSV: p99 end-to-end latency (queue wait + service) vs
offered load, one line per batch, one panel per backend, log-log. Unstable points (offered
rate above the batch/service ceiling) are drawn hollow, so the "stability wall" where each
batch's queue explodes is visible. The takeaway: small batch has a low ceiling and blows up
under modest load; larger batch raises the ceiling. The lowest stable line at each load is
the batch a live system should pick.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main(path):
    df = pd.read_csv(path)
    df["p99_us"] = df["p99_ns"] / 1000.0
    backends = list(df["backend"].unique())
    out = Path(path).with_suffix(".png")

    fig, axes = plt.subplots(1, len(backends), figsize=(6 * len(backends), 5), squeeze=False)
    for ax, backend in zip(axes[0], backends):
        sub = df[df["backend"] == backend]
        for batch in sorted(sub["batch"].unique()):
            b = sub[sub["batch"] == batch].sort_values("offered_rows_s")
            line, = ax.plot(b["offered_rows_s"], b["p99_us"], marker="o",
                            label=f"batch {batch}")
            # Redraw unstable points hollow, in the same color, to mark the wall.
            un = b[b["stable"] == 0]
            ax.scatter(un["offered_rows_s"], un["p99_us"], facecolors="none",
                       edgecolors=line.get_color(), zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("offered load (rows/s)")
        ax.set_ylabel("p99 end-to-end latency (us)")
        ax.set_title(f"{backend}: latency vs load (hollow = unstable queue)")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python plot_queue.py <queue_load.csv>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
