"""Entropy estimators for discretized return series.

This module implements the feature UNDER TEST. Every function here must be
verifiable against a hand-computed fixture. Do not add convenience wrappers,
plotting, or I/O. Pure functions only: array in, number or array out.

Bin edges are frozen by PREREGISTRATION.md section 3. Changing them requires
a logged amendment.
"""

from __future__ import annotations

import numpy as np

# Frozen by PREREGISTRATION.md 3. Intervals are right-closed:
#   (-inf, -0.02], (-0.02, -0.01], (-0.01, 0.01], (0.01, 0.02], (0.02, inf)
FIXED_BIN_EDGES = np.array([-np.inf, -0.02, -0.01, 0.01, 0.02, np.inf])
N_FIXED_BINS = len(FIXED_BIN_EDGES) - 1


def bin_counts(returns: np.ndarray, edges: np.ndarray = FIXED_BIN_EDGES) -> np.ndarray:
    """Count how many returns fall in each bin.

    Intervals are right-closed, so a return exactly equal to an edge falls in
    the LOWER bin. r = -0.02 lands in bin 0, not bin 1.

    Returns an integer array of length len(edges) - 1.
    """
    r = np.asarray(returns, dtype=float)
    if np.isnan(r).any():
        raise ValueError("returns contain NaN; caller must drop or impute first")
    # side="left" makes searchsorted right-closed: value == edge goes to lower bin.
    idx = np.searchsorted(edges, r, side="left") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    return np.bincount(idx, minlength=len(edges) - 1)


def shannon_entropy_from_counts(counts: np.ndarray) -> float:
    """Plug-in (maximum-likelihood) Shannon entropy in bits.

    H = -sum(p_i * log2(p_i)), with the convention 0 * log2(0) = 0.
    """
    c = np.asarray(counts, dtype=float)
    n = c.sum()
    if n == 0:
        raise ValueError("counts sum to zero")
    p = c[c > 0] / n
    return float(-np.sum(p * np.log2(p)))


def fixed_bin_entropy(returns: np.ndarray, edges: np.ndarray = FIXED_BIN_EDGES) -> float:
    """Plug-in Shannon entropy of returns discretized into fixed absolute bins.

    This is the feature whose redundancy with realized volatility is the
    subject of the study.
    """
    return shannon_entropy_from_counts(bin_counts(returns, edges))


def occupied_bins(returns: np.ndarray, edges: np.ndarray = FIXED_BIN_EDGES) -> int:
    """Number of bins with at least one observation (K-hat).

    Tracked separately because K is expected to rise with volatility, which is
    the mechanism by which small-sample estimator bias becomes correlated with
    volatility (PREREGISTRATION.md 4, prediction P6).
    """
    return int((bin_counts(returns, edges) > 0).sum())


def miller_madow_entropy(returns: np.ndarray, edges: np.ndarray = FIXED_BIN_EDGES) -> float:
    """Miller-Madow bias-corrected Shannon entropy in bits.

    H_MM = H_plugin + (K_hat - 1) / (2 * N * ln(2))

    The 1/ln(2) converts the correction, which is derived in nats, into bits.
    """
    counts = bin_counts(returns, edges)
    n = counts.sum()
    k_hat = int((counts > 0).sum())
    return shannon_entropy_from_counts(counts) + (k_hat - 1) / (2 * n * np.log(2))
