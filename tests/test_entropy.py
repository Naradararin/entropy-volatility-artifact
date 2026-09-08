"""Tests for the entropy feature, driven by a hand-computed fixture.

Every expected value in fixtures/entropy_windows.csv was computed by hand from
the bin counts, not by running this code. If a test here fails, the code is
wrong, not the fixture. Do not edit the fixture to make a test pass.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.entropy import (
    FIXED_BIN_EDGES,
    bin_counts,
    fixed_bin_entropy,
    miller_madow_entropy,
    occupied_bins,
    shannon_entropy_from_counts,
)
from src.features.rolling import WINDOW, log_returns, realized_volatility, rolling_apply

FIXTURE = Path(__file__).parent.parent / "fixtures" / "entropy_windows.csv"


def load_fixture():
    df = pd.read_csv(FIXTURE)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "window_id": row["window_id"],
                "returns": np.array([float(x) for x in row["returns"].split(",")]),
                "expected_counts": np.array(
                    [int(x) for x in row["expected_counts"].split(",")]
                ),
                "expected_K": int(row["expected_K"]),
                "expected_H": float(row["expected_H_bits"]),
            }
        )
    return rows


FIXTURE_ROWS = load_fixture()
IDS = [r["window_id"] for r in FIXTURE_ROWS]


@pytest.mark.parametrize("case", FIXTURE_ROWS, ids=IDS)
def test_window_has_exactly_20_returns(case):
    assert len(case["returns"]) == WINDOW


@pytest.mark.parametrize("case", FIXTURE_ROWS, ids=IDS)
def test_bin_counts_match_fixture(case):
    np.testing.assert_array_equal(bin_counts(case["returns"]), case["expected_counts"])


@pytest.mark.parametrize("case", FIXTURE_ROWS, ids=IDS)
def test_entropy_matches_fixture(case):
    assert fixed_bin_entropy(case["returns"]) == pytest.approx(
        case["expected_H"], abs=1e-6
    )


@pytest.mark.parametrize("case", FIXTURE_ROWS, ids=IDS)
def test_occupied_bins_match_fixture(case):
    assert occupied_bins(case["returns"]) == case["expected_K"]


def test_boundary_values_fall_in_lower_bin():
    """Intervals are right-closed. This is the single easiest thing to get
    wrong, and getting it wrong shifts entropy in a volatility-dependent way."""
    assert bin_counts(np.array([-0.02]))[0] == 1
    assert bin_counts(np.array([-0.01]))[1] == 1
    assert bin_counts(np.array([0.01]))[2] == 1
    assert bin_counts(np.array([0.02]))[3] == 1
    assert bin_counts(np.array([0.0200001]))[4] == 1


def test_entropy_is_bounded_by_log2_of_bin_count():
    rng = np.random.default_rng(0)
    for _ in range(50):
        r = rng.normal(0, 0.02, WINDOW)
        h = fixed_bin_entropy(r)
        assert 0.0 <= h <= np.log2(len(FIXED_BIN_EDGES) - 1) + 1e-12


def test_entropy_is_invariant_to_ordering():
    """Shannon entropy of a histogram ignores sequence. This is a property of
    the estimator, and the reason permutation entropy is a different measure."""
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.02, WINDOW)
    assert fixed_bin_entropy(r) == pytest.approx(fixed_bin_entropy(rng.permutation(r)))


def test_entropy_is_not_scale_invariant():
    """The core mechanism of the study: scaling returns changes fixed-bin
    entropy, because the bins are in absolute units. A scale-invariant
    estimator would return the same value here."""
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.005, WINDOW)
    assert fixed_bin_entropy(r) != pytest.approx(fixed_bin_entropy(r * 10))


def test_nan_input_raises():
    with pytest.raises(ValueError):
        bin_counts(np.array([0.01, np.nan]))


def test_miller_madow_exceeds_plugin_when_multiple_bins_occupied():
    r = FIXTURE_ROWS[1]["returns"]  # W2, K = 5
    assert miller_madow_entropy(r) > fixed_bin_entropy(r)


def test_miller_madow_equals_plugin_when_one_bin_occupied():
    r = FIXTURE_ROWS[0]["returns"]  # W1, K = 1, correction is zero
    assert miller_madow_entropy(r) == pytest.approx(fixed_bin_entropy(r))


def test_realized_volatility_matches_hand_computation():
    """Ten +1% and ten -1%: mean 0, sum of squares 20*1e-4, var = 2e-3/19."""
    r = pd.Series([0.01] * 10 + [-0.01] * 10)
    expected = np.sqrt(20 * 1e-4 / 19)
    assert realized_volatility(r).iloc[-1] == pytest.approx(expected)


def test_rolling_windows_are_backward_looking():
    """A value at index t must not change when data after t changes."""
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 0.02, 60))
    a = rolling_apply(r, fixed_bin_entropy)
    r2 = r.copy()
    r2.iloc[40:] = 999.0
    b = rolling_apply(r2, fixed_bin_entropy)
    pd.testing.assert_series_equal(a.iloc[:40], b.iloc[:40])


def test_rolling_output_is_nan_before_first_full_window():
    r = pd.Series(np.zeros(30))
    out = rolling_apply(r, fixed_bin_entropy)
    assert out.iloc[: WINDOW - 1].isna().all()
    assert not np.isnan(out.iloc[WINDOW - 1])


def test_log_returns_roundtrip():
    prices = pd.Series([100.0, 110.0, 99.0])
    r = log_returns(prices)
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(np.log(1.10))
