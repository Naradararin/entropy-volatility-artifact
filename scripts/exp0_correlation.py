"""Experiment 0: correlation between rolling fixed-bin entropy and realized
volatility on BTC.

Descriptive only (PREREGISTRATION.md 5.1: correlation is a headline, not the
evidence). Prints results; no files are written.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.entropy import N_FIXED_BINS, fixed_bin_entropy
from src.features.rolling import WINDOW, log_returns, realized_volatility, rolling_apply

ASSETS = ["BTC", "ETH", "SPX"]

# Outer bin edge, frozen by PREREGISTRATION.md 3. Compared against, not
# redefined: this must always equal FIXED_BIN_EDGES[-2] in src/features/entropy.py.
OUTER_EDGE = 0.02


def run_asset(name: str) -> None:
    data_path = ROOT / "data" / "raw" / f"{name}.parquet"
    prices = pd.read_parquet(data_path)["close"]
    returns = log_returns(prices)

    h = rolling_apply(returns, fixed_bin_entropy, window=WINDOW)
    sigma = realized_volatility(returns, window=WINDOW)

    both = pd.concat([h, sigma], axis=1, keys=["H", "sigma"]).dropna()
    n = len(both)

    pearson_r, pearson_p = stats.pearsonr(both["H"], both["sigma"])
    spearman_r, spearman_p = stats.spearmanr(both["H"], both["sigma"])

    ceiling = np.log2(N_FIXED_BINS)
    pinned = ((both["H"] == 0.0) | np.isclose(both["H"], ceiling)).mean()

    sigma_valid = sigma.dropna()
    p10, p50, p90 = sigma_valid.quantile([0.10, 0.50, 0.90])
    exceeds_outer = (sigma_valid > OUTER_EDGE).mean()

    print(f"=== {name} ===")
    print(f"n (non-NaN windows) = {n}")
    print(f"Pearson r  = {pearson_r:.6f}  (p = {pearson_p:.3e})")
    print(f"Spearman r = {spearman_r:.6f}  (p = {spearman_p:.3e})")
    print(
        f"fraction of windows with H == 0 or H == log2({N_FIXED_BINS}) "
        f"(={ceiling:.6f}): {pinned:.6f}"
    )
    print(
        f"sigma: mean = {sigma_valid.mean():.6f}, std = {sigma_valid.std(ddof=1):.6f}, "
        f"p10 = {p10:.6f}, p50 = {p50:.6f}, p90 = {p90:.6f}"
    )
    print(f"fraction of windows with sigma > {OUTER_EDGE}: {exceeds_outer:.6f}")
    print()


def main() -> None:
    for name in ASSETS:
        run_asset(name)


if __name__ == "__main__":
    main()
