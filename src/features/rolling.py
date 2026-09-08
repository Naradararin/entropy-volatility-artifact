"""Realized volatility and rolling-window application.

Window convention, frozen: a value at index t is computed from observations
t-WINDOW+1 .. t inclusive. It uses no information after t. Indices with fewer
than WINDOW observations are NaN.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

WINDOW = 20  # trading days, frozen by PREREGISTRATION.md section 3


def log_returns(prices: pd.Series) -> pd.Series:
    """r_t = ln(P_t / P_{t-1}). First element is NaN."""
    return np.log(prices / prices.shift(1))


def realized_volatility(returns: pd.Series, window: int = WINDOW) -> pd.Series:
    """Rolling sample standard deviation (ddof=1) of log returns."""
    return returns.rolling(window=window, min_periods=window).std(ddof=1)


def rolling_apply(
    returns: pd.Series,
    func: Callable[[np.ndarray], float],
    window: int = WINDOW,
) -> pd.Series:
    """Apply a window -> scalar function over a rolling window.

    Used for entropy variants, which pandas cannot compute natively. Slower
    than a vectorised rolling op but correctness matters more here, and the
    series are short.
    """
    values = returns.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for end in range(window, len(values) + 1):
        chunk = values[end - window : end]
        if np.isnan(chunk).any():
            continue
        out[end - 1] = func(chunk)
    return pd.Series(out, index=returns.index)
