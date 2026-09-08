"""Realized excess kurtosis for the residual-vs-kurtosis check (PREREGISTRATION.md
section 6 step 3, prediction P10).

Pure functions only: array in, number out. No printing, no file reads, no
plotting, no global state.
"""

from __future__ import annotations

import numpy as np


def excess_kurtosis(returns: np.ndarray) -> float:
    """Sample excess (Fisher) kurtosis, biased plug-in estimator.

    kurt = mean((x - mean(x))^4) / mean((x - mean(x))^2)^2 - 3

    Uses population (N, not N-1) central moments, matching the plug-in
    convention used for the entropy estimator in entropy.py. A Gaussian has
    excess kurtosis 0; fat-tailed distributions are positive.
    """
    x = np.asarray(returns, dtype=float)
    if np.isnan(x).any():
        raise ValueError("returns contain NaN; caller must drop or impute first")
    mu = x.mean()
    m2 = np.mean((x - mu) ** 2)
    if m2 == 0:
        raise ValueError("zero variance; kurtosis is undefined")
    m4 = np.mean((x - mu) ** 4)
    return float(m4 / m2**2 - 3.0)
