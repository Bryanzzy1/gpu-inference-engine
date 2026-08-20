"""Download and verify Binance spot aggTrades, one day or a date range.

Usage:
    python download_data.py BTCUSDT 2026-06-27                 # one day
    python download_data.py BTCUSDT 2026-06-21 2026-06-27      # inclusive range

Fetches each daily aggTrades zip and its checksum, verifies the SHA-256, then
extracts the CSV into ../data. For a range it also writes one combined CSV
(BTCUSDT-aggTrades-<start>_to_<end>.csv) in date order, which the benchmarks read
as a single larger workload. Raw CSVs are gitignored. Re-downloading a day already
on disk is skipped.
"""

import hashlib
import sys
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

BASE = "https://data.binance.vision/data/spot/daily/aggTrades"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch(url: str, dest: Path) -> None:
    print(f"GET {url}")
    urllib.request.urlretrieve(url, dest)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_day(symbol: str, day: str) -> Path:
    """Fetch, verify, and extract one day. Returns the CSV path. Skips if present."""
    DATA_DIR.mkdir(exist_ok=True)
    name = f"{symbol}-aggTrades-{day}"
    csv_path = DATA_DIR / f"{name}.csv"
    if csv_path.exists():
        print(f"have {csv_path.name}, skipping")
        return csv_path
    zip_path = DATA_DIR / f"{name}.zip"
    chk_path = DATA_DIR / f"{name}.zip.CHECKSUM"

    fetch(f"{BASE}/{symbol}/{name}.zip", zip_path)
    fetch(f"{BASE}/{symbol}/{name}.zip.CHECKSUM", chk_path)

    expected = chk_path.read_text().split()[0]
    actual = sha256(zip_path)
    if actual != expected:
        print(f"CHECKSUM MISMATCH\n  expected {expected}\n  actual   {actual}", file=sys.stderr)
        sys.exit(1)
    print("checksum OK")

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)
    print(f"extracted -> {csv_path}")
    return csv_path


def _days(start: str, end: str) -> list[str]:
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    if d1 < d0:
        sys.exit("error: end date is before start date")
    out, d = [], d0
    while d <= d1:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def main(symbol: str, start: str, end: str | None = None) -> None:
    if end is None:
        download_day(symbol, start)
        return
    days = _days(start, end)
    csvs = [download_day(symbol, d) for d in days]
    combined = DATA_DIR / f"{symbol}-aggTrades-{start}_to_{end}.csv"
    # Concatenate in date order: daily aggTrades have no header and are already sorted,
    # so appending days is a valid continuous stream for the feature engine.
    with open(combined, "wb") as out:
        for c in csvs:
            with open(c, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    out.write(chunk)
    mb = combined.stat().st_size / 1e6
    print(f"combined {len(csvs)} days -> {combined}  ({mb:.0f} MB)")


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    start = sys.argv[2] if len(sys.argv) > 2 else "2026-06-27"
    end = sys.argv[3] if len(sys.argv) > 3 else None
    main(sym, start, end)
