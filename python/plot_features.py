"""Plot features from the feature CSV to sanity-check them.

Usage:
    python plot_features.py ../data/features.csv
Writes ../data/features_plot.png and prints quick stats.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed; save to file
import matplotlib.pyplot as plt
import pandas as pd


def main(path: str) -> None:
    df = pd.read_csv(path)

    # Seconds from the start, for readable x-axes.
    t0 = df["time_us"].iloc[0]
    t = (df["time_us"] - t0) / 1e6

    print(f"rows: {len(df):,}")
    print(df[["mid", "volatility", "imbalance", "intensity"]].describe())
    print("\nlabel balance:")
    print(df["label"].value_counts(normalize=True).round(3).to_dict())
    print(f"intensity == 1024 (saturated): {int((df['intensity'] == 1024).sum())}")

    fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    ax[0].plot(t, df["mid"], lw=0.5)
    ax[0].set_ylabel("mid price")

    ax[1].plot(t, df["volatility"], lw=0.5, color="tab:orange")
    ax[1].set_ylabel("volatility")

    ax[2].plot(t, df["imbalance"], lw=0.5, color="tab:green")
    ax[2].set_ylabel("imbalance")
    ax[2].set_ylim(-1.05, 1.05)

    ax[3].plot(t, df["intensity"], lw=0.5, color="tab:red")
    ax[3].set_ylabel("intensity")
    ax[3].set_xlabel("seconds from start")

    fig.suptitle("Feature sanity check")
    fig.tight_layout()

    out = Path(path).with_name("features_plot.png")
    fig.savefig(out, dpi=110)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python plot_features.py <features.csv>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
