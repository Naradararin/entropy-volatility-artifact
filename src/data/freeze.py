"""Download once, freeze forever.

This is the ONLY module permitted to touch the network, and it is run by hand,
never by an agent. It writes Parquet files to data/raw/ and records a SHA256 of
each in data/manifest.json. Every downstream step reads the frozen files, so
results do not change when the upstream data source changes or disappears.

Usage:
    python3 -m src.data.freeze

Parameters below are frozen by PREREGISTRATION.md section 2. Changing them is an
amendment, not an edit.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Frozen by PREREGISTRATION.md section 2.
TICKERS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SPX": "^GSPC",
}
START = "2017-01-01"
END = "2025-12-31"
SOURCE = "yfinance (Yahoo Finance), unofficial API"

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
MANIFEST = RAW_DIR.parent / "manifest.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(ticker: str) -> pd.DataFrame:
    import yfinance as yf  # imported here so the rest of the repo never needs it

    df = yf.download(
        ticker,
        start=START,
        end=END,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise RuntimeError(f"no rows returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    out = df[["Close"]].rename(columns={"Close": "close"})
    out.index.name = "date"
    # Duplicate dates have been observed from this source; keep the last.
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if out["close"].isna().any():
        raise RuntimeError(f"{ticker} contains NaN closes; inspect before freezing")
    if (out["close"] <= 0).any():
        raise RuntimeError(f"{ticker} contains non-positive closes")
    return out


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        raise SystemExit(
            f"{MANIFEST} already exists. Data is frozen. Delete it deliberately "
            "and record an amendment in PREREGISTRATION.md if you really mean to refreeze."
        )

    entries = []
    for name, ticker in TICKERS.items():
        df = download(ticker)
        path = RAW_DIR / f"{name}.parquet"
        df.to_parquet(path)
        entries.append(
            {
                "name": name,
                "ticker": ticker,
                "file": str(path.relative_to(MANIFEST.parent.parent)),
                "rows": int(len(df)),
                "first_date": str(df.index.min().date()),
                "last_date": str(df.index.max().date()),
                "sha256": sha256_of(path),
            }
        )
        print(f"{name}: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")

    manifest = {
        "source": SOURCE,
        "requested_start": START,
        "requested_end": END,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": entries,
        "note": (
            "Frozen snapshot. Downstream code must read these files only. "
            "Row counts differ across assets by design: crypto trades every day, "
            "the equity index does not."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {MANIFEST}")


if __name__ == "__main__":
    main()
