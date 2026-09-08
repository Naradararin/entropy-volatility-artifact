"""P10 check (PREREGISTRATION.md 4, 6 step 3): within BTC and ETH's
UNSATURATED windows (sigma <= 0.02) only, does the synthetic-null excess
residual concentrate in the top kurtosis quartile?

Reuses the exact real-vs-null residual construction from
scripts/exp0_synthetic_null.py (see that file's docstring for the two fixes
this depends on: aligning the synthetic series' longer NaN warm-up, and
regressing each series' H against ITS OWN recomputed realized sigma rather
than the empirical target used to scale its innovations). A single isotonic
fit of H on sigma over the FULL sample (all sigma regimes, per
PREREGISTRATION.md 5.2's "fitted monotone spline of H on sigma") produces the
residuals; this script partitions them into kurtosis quartiles computed only
over the unsaturated subsample, using quartile edges from the REAL data and
applying that same fixed date partition to every synthetic replication.

Realized excess kurtosis uses src.features.moments.excess_kurtosis, a new
pure function (it did not exist before this script) added with a
hand-computed fixture in fixtures/kurtosis_windows.csv and tests in
tests/test_moments.py, per CLAUDE.md's conventions. entropy.py, rolling.py,
and the frozen bin edges are unchanged.

Print only; no files written.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import isotonic_regression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.entropy import fixed_bin_entropy
from src.features.moments import excess_kurtosis
from src.features.rolling import WINDOW, log_returns, realized_volatility, rolling_apply

ASSETS = ["BTC", "ETH"]
N_REPLICATIONS = 1000
OUTER_EDGE = 0.02  # frozen outer bin edge, PREREGISTRATION.md 3
N_QUARTILES = 4


def isotonic_residuals(h: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    order = np.argsort(sigma, kind="stable")
    fitted_sorted = isotonic_regression(h[order]).x
    fitted = np.empty_like(fitted_sorted)
    fitted[order] = fitted_sorted
    return h - fitted


def run_asset(name: str) -> list[dict]:
    data_path = ROOT / "data" / "raw" / f"{name}.parquet"
    prices = pd.read_parquet(data_path)["close"]
    returns = log_returns(prices)

    sigma_full = realized_volatility(returns, window=WINDOW)
    h_real_full = rolling_apply(returns, fixed_bin_entropy, window=WINDOW)
    kurt_full = rolling_apply(returns, excess_kurtosis, window=WINDOW)

    both = pd.concat(
        [h_real_full, sigma_full, kurt_full], axis=1, keys=["H", "sigma", "kurt"]
    ).dropna()

    # Same NaN-alignment fix as exp0_synthetic_null.py: sigma_full's own
    # WINDOW-length warm-up makes sim_returns invalid ~WINDOW-1 rows longer
    # than the real series.
    probe_returns = pd.Series(0.0, index=returns.index) * sigma_full
    probe_h = rolling_apply(probe_returns, fixed_bin_entropy, window=WINDOW)
    valid_idx = both.index.intersection(probe_h.dropna().index)

    sigma = both.loc[valid_idx, "sigma"].to_numpy()
    kurt = both.loc[valid_idx, "kurt"].to_numpy()
    h_real = both.loc[valid_idx, "H"].to_numpy()

    real_resid = isotonic_residuals(h_real, sigma)

    unsat_mask = sigma <= OUTER_EDGE
    kurt_unsat = kurt[unsat_mask]
    quartile_idx = pd.qcut(kurt_unsat, N_QUARTILES, labels=False)  # 0 = lowest kurtosis

    null_resid_unsat_by_seed = []
    for seed in range(N_REPLICATIONS):
        rng = np.random.default_rng(seed)
        innovations = pd.Series(rng.standard_normal(len(returns)), index=returns.index)
        sim_returns = innovations * sigma_full

        h_sim = rolling_apply(sim_returns, fixed_bin_entropy, window=WINDOW).loc[valid_idx].to_numpy()
        sigma_sim = realized_volatility(sim_returns, window=WINDOW).loc[valid_idx].to_numpy()

        sim_resid = isotonic_residuals(h_sim, sigma_sim)
        null_resid_unsat_by_seed.append(sim_resid[unsat_mask])

    real_resid_unsat = real_resid[unsat_mask]

    rows = []
    for q in range(N_QUARTILES):
        mask = quartile_idx == q
        n_q = int(mask.sum())
        real_var = float(np.var(real_resid_unsat[mask], ddof=1))
        null_vars = np.array([float(np.var(r[mask], ddof=1)) for r in null_resid_unsat_by_seed])
        null_mean = float(null_vars.mean())
        rows.append(
            {
                "asset": name,
                "quartile": f"Q{q + 1}",
                "n": n_q,
                "kurt_lo": float(kurt_unsat[mask].min()),
                "kurt_hi": float(kurt_unsat[mask].max()),
                "var_resid_real": real_var,
                "null_mean": null_mean,
                "excess_residual": real_var - null_mean,
            }
        )
    return rows


def main() -> None:
    all_rows = []
    for name in ASSETS:
        all_rows.extend(run_asset(name))

    cols = ["asset", "quartile", "n", "kurt_lo", "kurt_hi", "var_resid_real", "null_mean", "excess_residual"]
    widths = [5, 8, 5, 10, 10, 16, 12, 16]
    header = "  ".join(c.rjust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for row in all_rows:
        line = "  ".join(
            [
                row["asset"].rjust(widths[0]),
                row["quartile"].rjust(widths[1]),
                str(row["n"]).rjust(widths[2]),
                f"{row['kurt_lo']:.3f}".rjust(widths[3]),
                f"{row['kurt_hi']:.3f}".rjust(widths[4]),
                f"{row['var_resid_real']:.6f}".rjust(widths[5]),
                f"{row['null_mean']:.6f}".rjust(widths[6]),
                f"{row['excess_residual']:+.6f}".rjust(widths[7]),
            ]
        )
        print(line)


if __name__ == "__main__":
    main()
