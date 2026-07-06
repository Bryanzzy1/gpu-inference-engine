#Download and verify one day of Binance spot aggTrades.
#Raw CSVs are gitignored here


import hashlib
import sys
import urllib.request
import zipfile
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


def main(symbol: str, date: str) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    name = f"{symbol}-aggTrades-{date}"
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
    print(f"extracted -> {DATA_DIR / (name + '.csv')}")


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    dt = sys.argv[2] if len(sys.argv) > 2 else "2026-06-27"  # default BTCUSDT 2026-06-27
    main(sym, dt)
