"""Experiment 0, synthetic-null test (PREREGISTRATION.md 5.2; amended 9).

Per the 2026-09-08 amendment, the null must be run per-asset using EACH
asset's own empirical sigma path, not a single shared null -- BTC and ETH are
mostly saturated (H pinned near the ceiling), SPX mostly is not.

Null construction: for each replication, Gaussian innovations N(0,1) are
scaled by the asset's own empirical rolling sigma at each date, producing a
return series that matches the empirical sigma path in expectation but has
Gaussian shape. H is then computed on that simulated series the same way as
on the real series. Each series' residuals are taken around an isotonic
(monotone increasing) fit of ITS OWN H against ITS OWN realized sigma --
real H against real realized sigma, each replication's H against that
replication's own realized sigma (recomputed via realized_volatility on the
simulated series, not the empirical target used to scale it) -- so sigma is
measured identically in both cases and the only thing that differs is return
shape. Pairing a replication's H with the empirical TARGET sigma instead
would inject the sampling noise of a 20-obs realized-vol estimate into
Var(resid_null) for reasons unrelated to shape, inflating it. The
saturated/unsaturated split still uses the empirical (real) sigma path, since
that fixes which calendar windows count as saturated consistently across all
1,000 replications.

Uses only existing functions from src/features/ (fixed_bin_entropy,
realized_volatility, rolling_apply) and scipy.optimize.isotonic_regression,
which is already covered by requirements.txt's scipy dependency (available
since scipy 1.12; requirements.txt currently floors at 1.10 -- flagging this,
not changing it, since floor bumps were not requested). No entropy.py,
rolling.py, or bin-edge changes. Print only; no files written.
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
from src.features.rolling import WINDOW, log_returns, realized_volatility, rolling_apply

ASSETS = ["BTC", "ETH", "SPX"]
N_REPLICATIONS = 1000
OUTER_EDGE = 0.02  # frozen outer bin edge, PREREGISTRATION.md 3


def isotonic_residuals(h: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Residuals of h around a monotone-increasing fit of h on sigma.

    isotonic_regression assumes its input is already ordered by x, so we sort
    by sigma, fit, then map fitted values back to the original order.
    """
    order = np.argsort(sigma, kind="stable")
    fitted_sorted = isotonic_regression(h[order]).x
    fitted = np.empty_like(fitted_sorted)
    fitted[order] = fitted_sorted
    return h - fitted


def var_by_subsample(residual: np.ndarray, sigma: np.ndarray) -> dict[str, float]:
    sat = sigma > OUTER_EDGE
    unsat = ~sat
    return {
        "full": float(np.var(residual, ddof=1)),
        "saturated": float(np.var(residual[sat], ddof=1)) if sat.sum() > 1 else float("nan"),
        "unsaturated": float(np.var(residual[unsat], ddof=1)) if unsat.sum() > 1 else float("nan"),
    }


def run_asset(name: str) -> None:
    data_path = ROOT / "data" / "raw" / f"{name}.parquet"
    prices = pd.read_parquet(data_path)["close"]
    returns = log_returns(prices)

    sigma_full = realized_volatility(returns, window=WINDOW)  # aligned to returns.index
    h_real_full = rolling_apply(returns, fixed_bin_entropy, window=WINDOW)

    both = pd.concat([h_real_full, sigma_full], axis=1, keys=["H", "sigma"]).dropna()

    # sigma_full itself has a WINDOW-length NaN warm-up (it needs 20 real
    # returns), so sim_returns = innovations * sigma_full carries a NaN run
    # that long at the start -- one full WINDOW longer than the real return
    # series' single leading NaN. rolling_apply on sim_returns therefore only
    # becomes valid ~WINDOW-1 rows later than the real H series. Determine
    # that sim-valid range once (deterministic: depends only on sigma_full's
    # NaN pattern, not the random draw) via the same rolling_apply used
    # everywhere else, and intersect it with the real-valid range so real and
    # null are compared over identical indices.
    probe_returns = pd.Series(0.0, index=returns.index) * sigma_full
    probe_h = rolling_apply(probe_returns, fixed_bin_entropy, window=WINDOW)
    valid_idx = both.index.intersection(probe_h.dropna().index)

    sigma = both.loc[valid_idx, "sigma"].to_numpy()
    h_real = both.loc[valid_idx, "H"].to_numpy()
    n = len(valid_idx)

    real_var = var_by_subsample(isotonic_residuals(h_real, sigma), sigma)

    null_vars: dict[str, list[float]] = {"full": [], "saturated": [], "unsaturated": []}
    for seed in range(N_REPLICATIONS):
        rng = np.random.default_rng(seed)
        innovations = pd.Series(rng.standard_normal(len(returns)), index=returns.index)
        sim_returns = innovations * sigma_full  # NaN wherever sigma_full is NaN

        h_sim_full = rolling_apply(sim_returns, fixed_bin_entropy, window=WINDOW)
        h_sim = h_sim_full.loc[valid_idx].to_numpy()

        # Regress H against THIS replication's own realized sigma (measured
        # the same way as the real-data fit: from the same window that
        # produced H), not the target sigma_full used to scale the
        # innovations. sigma_full only matches a replication's realized sigma
        # in expectation -- pairing H with the target instead of the
        # replication's own realized value would inject sampling noise from
        # that mismatch into Var(resid_null), inflating it for reasons
        # unrelated to return shape, which is the one thing this null is
        # supposed to isolate. The saturated/unsaturated SPLIT still uses the
        # real sigma path, since that fixes which calendar windows count as
        # saturated consistently across all 1,000 replications.
        sigma_sim = realized_volatility(sim_returns, window=WINDOW).loc[valid_idx].to_numpy()

        sim_var = var_by_subsample(isotonic_residuals(h_sim, sigma_sim), sigma)
        for k in null_vars:
            null_vars[k].append(sim_var[k])

    print(f"=== {name} (n={n}, seeds 0..{N_REPLICATIONS - 1}) ===")
    for subsample in ["full", "saturated", "unsaturated"]:
        real_v = real_var[subsample]
        null_arr = np.array(null_vars[subsample])
        null_arr = null_arr[~np.isnan(null_arr)]
        n_sub = (
            n
            if subsample == "full"
            else int((sigma > OUTER_EDGE).sum() if subsample == "saturated" else (sigma <= OUTER_EDGE).sum())
        )
        if len(null_arr) == 0 or np.isnan(real_v):
            print(f"[{subsample}] n={n_sub}: insufficient observations, skipped")
            continue
        null_mean = null_arr.mean()
        lo, hi = np.percentile(null_arr, [2.5, 97.5])
        inside = lo <= real_v <= hi
        excess = real_v - null_mean
        print(
            f"[{subsample}] n={n_sub}: Var(resid_real) = {real_v:.6f} | "
            f"null mean = {null_mean:.6f}, 95% CI = [{lo:.6f}, {hi:.6f}] | "
            f"excess_residual = {excess:+.6f} | real value "
            f"{'INSIDE' if inside else 'OUTSIDE'} null 95% CI"
        )
    print()


def main() -> None:
    for name in ASSETS:
        run_asset(name)


if __name__ == "__main__":
    main()
