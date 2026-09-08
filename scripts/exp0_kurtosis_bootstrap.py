"""Bootstrap CI on the per-quartile excess_residual from
scripts/exp0_kurtosis_quartiles.py (BTC, ETH; unsaturated windows; P10 check).

Resamples REAL windows within each kurtosis quartile with replacement (1,000
resamples), recomputing Var(resid_real) each time and subtracting the FIXED
null_mean for that quartile. The null itself is not touched: null_mean comes
from calling exp0_kurtosis_quartiles.run_asset unmodified, which reruns the
same deterministic 1,000-replication synthetic null (fixed seeds 0..999) --
only the real-side variance is resampled here. Reports the 2.5/97.5
percentile of the resulting excess_residual distribution.

Print only; no files written.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.exp0_kurtosis_quartiles import ASSETS, N_QUARTILES, OUTER_EDGE, isotonic_residuals, run_asset
from src.features.entropy import fixed_bin_entropy
from src.features.moments import excess_kurtosis
from src.features.rolling import WINDOW, log_returns, realized_volatility, rolling_apply

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 0  # fixed base seed; per-quartile seed = BOOTSTRAP_SEED + asset_idx*100 + quartile_idx


def real_resid_and_quartiles(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Re-derive the real-data residuals and quartile assignment for the
    unsaturated subsample. Deterministic, no randomness -- reproduces exactly
    what exp0_kurtosis_quartiles.run_asset computes internally on the real
    side, without rerunning its (expensive) synthetic-null loop.
    """
    data_path = ROOT / "data" / "raw" / f"{name}.parquet"
    prices = pd.read_parquet(data_path)["close"]
    returns = log_returns(prices)

    sigma_full = realized_volatility(returns, window=WINDOW)
    h_real_full = rolling_apply(returns, fixed_bin_entropy, window=WINDOW)
    kurt_full = rolling_apply(returns, excess_kurtosis, window=WINDOW)

    both = pd.concat(
        [h_real_full, sigma_full, kurt_full], axis=1, keys=["H", "sigma", "kurt"]
    ).dropna()

    probe_returns = pd.Series(0.0, index=returns.index) * sigma_full
    probe_h = rolling_apply(probe_returns, fixed_bin_entropy, window=WINDOW)
    valid_idx = both.index.intersection(probe_h.dropna().index)

    sigma = both.loc[valid_idx, "sigma"].to_numpy()
    kurt = both.loc[valid_idx, "kurt"].to_numpy()
    h_real = both.loc[valid_idx, "H"].to_numpy()

    real_resid = isotonic_residuals(h_real, sigma)
    unsat_mask = sigma <= OUTER_EDGE
    kurt_unsat = kurt[unsat_mask]
    quartile_idx = pd.qcut(kurt_unsat, N_QUARTILES, labels=False)

    return real_resid[unsat_mask], quartile_idx


def bootstrap_excess_ci(resid_q: np.ndarray, null_mean: float, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(resid_q)
    boot_excess = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        sample = rng.choice(resid_q, size=n, replace=True)
        boot_excess[b] = np.var(sample, ddof=1) - null_mean
    lo, hi = np.percentile(boot_excess, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    rows = []
    for asset_idx, name in enumerate(ASSETS):
        quartile_rows = run_asset(name)  # unmodified: real_var, null_mean per quartile
        resid_unsat, quartile_idx = real_resid_and_quartiles(name)
        for q, qrow in enumerate(quartile_rows):
            mask = quartile_idx == q
            resid_q = resid_unsat[mask]
            seed = BOOTSTRAP_SEED + asset_idx * 100 + q
            lo, hi = bootstrap_excess_ci(resid_q, qrow["null_mean"], seed)
            rows.append(
                {
                    "asset": name,
                    "quartile": qrow["quartile"],
                    "n": qrow["n"],
                    "excess_residual": qrow["excess_residual"],
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )

    cols = ["asset", "quartile", "n", "excess_residual", "CI_low", "CI_high"]
    widths = [5, 8, 5, 16, 12, 12]
    header = "  ".join(c.rjust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for row in rows:
        line = "  ".join(
            [
                row["asset"].rjust(widths[0]),
                row["quartile"].rjust(widths[1]),
                str(row["n"]).rjust(widths[2]),
                f"{row['excess_residual']:+.6f}".rjust(widths[3]),
                f"{row['ci_low']:+.6f}".rjust(widths[4]),
                f"{row['ci_high']:+.6f}".rjust(widths[5]),
            ]
        )
        print(line)


if __name__ == "__main__":
    main()
