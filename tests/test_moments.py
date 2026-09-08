"""Tests for realized excess kurtosis, driven by a hand-computed fixture.

Every expected value in fixtures/kurtosis_windows.csv was computed by hand
from the closed-form moments of a two-point distribution, not by running this
code. If a test here fails, the code is wrong, not the fixture. Do not edit
the fixture to make a test pass.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.moments import excess_kurtosis
from src.features.rolling import WINDOW

FIXTURE = Path(__file__).parent.parent / "fixtures" / "kurtosis_windows.csv"


def load_fixture():
    df = pd.read_csv(FIXTURE)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "window_id": row["window_id"],
                "returns": np.array([float(x) for x in row["returns"].split(",")]),
                "expected_kurtosis": float(row["expected_kurtosis"]),
            }
        )
    return rows


FIXTURE_ROWS = load_fixture()
IDS = [r["window_id"] for r in FIXTURE_ROWS]


@pytest.mark.parametrize("case", FIXTURE_ROWS, ids=IDS)
def test_window_has_exactly_20_returns(case):
    assert len(case["returns"]) == WINDOW


@pytest.mark.parametrize("case", FIXTURE_ROWS, ids=IDS)
def test_kurtosis_matches_fixture(case):
    assert excess_kurtosis(case["returns"]) == pytest.approx(
        case["expected_kurtosis"], abs=1e-9
    )


def test_kurtosis_is_scale_and_location_invariant():
    """K1 and K3 are both 50/50 two-point distributions at different values
    and spacings; excess kurtosis of an equal-probability two-point
    distribution is exactly -2 regardless of scale or location."""
    k1 = next(c for c in FIXTURE_ROWS if c["window_id"] == "K1")["returns"]
    k3 = next(c for c in FIXTURE_ROWS if c["window_id"] == "K3")["returns"]
    assert excess_kurtosis(k1) == pytest.approx(excess_kurtosis(k3))


def test_gaussian_kurtosis_is_near_zero_for_large_n():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.02, 200_000)
    assert excess_kurtosis(x) == pytest.approx(0.0, abs=0.05)


def test_nan_input_raises():
    with pytest.raises(ValueError):
        excess_kurtosis(np.array([0.01, np.nan]))


def test_zero_variance_raises():
    with pytest.raises(ValueError):
        excess_kurtosis(np.zeros(WINDOW))
