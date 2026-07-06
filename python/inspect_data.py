"""
Binance spot daily aggTrades files have NO header row. Columns are:
    agg_trade_id, price, qty, first_trade_id, last_trade_id,
    timestamp, is_buyer_maker, is_best_match

We print the first rows, dtypes, and the time span so we know exactly what a
tick looks like before writing the C++ parser (Day 2).

"""

import sys
import pandas as pd

COLUMNS = [
    "agg_trade_id",
    "price",
    "qty",
    "first_trade_id",
    "last_trade_id",
    "timestamp",
    "is_buyer_maker",
    "is_best_match",
]


def detect_time_unit(ts: int) -> str:
    """Guess the epoch unit from the magnitude of a timestamp."""
    digits = len(str(int(ts)))
    return {13: "ms", 16: "us", 19: "ns"}.get(digits, f"unknown ({digits} digits)")


def main(path: str) -> None:
    df = pd.read_csv(path, header=None, names=COLUMNS)

    print(f"File: {path}")
    print(f"Rows: {len(df):,}")
    print()

    print("=== First 20 rows ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(20))
    print()

    print("=== Column dtypes ===")
    print(df.dtypes)
    print()

    t0, t1 = int(df["timestamp"].iloc[0]), int(df["timestamp"].iloc[-1])
    unit = detect_time_unit(t0)
    print("=== Time span ===")
    print(f"timestamp unit (inferred): {unit}")
    print(f"first: {t0}")
    print(f"last:  {t1}")
    if unit in ("ms", "us", "ns"):
        span = pd.to_datetime(t1, unit=unit) - pd.to_datetime(t0, unit=unit)
        print(f"first (UTC): {pd.to_datetime(t0, unit=unit)}")
        print(f"last  (UTC): {pd.to_datetime(t1, unit=unit)}")
        print(f"span: {span}")
    print()

    print("=== Price / qty summary ===")
    print(df[["price", "qty"]].describe())
    print()

    print("=== is_buyer_maker balance ===")
    print(df["is_buyer_maker"].value_counts(dropna=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python inspect_data.py <path-to-aggTrades.csv>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
