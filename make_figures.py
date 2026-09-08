"""Generate the two figures for the technical note.

Run this LOCALLY, from the repo root, inside the venv:

    python make_figures.py

It reads the frozen Parquet files (never re-downloads anything) and the
existing tested feature functions, and writes two PNGs to figures/.
Requires matplotlib (already in requirements.txt).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.entropy import FIXED_BIN_EDGES, fixed_bin_entropy
from src.features.rolling import log_returns, realized_volatility, rolling_apply

OUTER_EDGE = FIXED_BIN_EDGES[-1]
CEILING = np.log2(5)
FIGDIR = Path("figures")
FIGDIR.mkdir(exist_ok=True)

ASSETS = {"BTC": "#f7931a", "ETH": "#627eea", "SPX": "#2e7d32"}


def load(asset: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_parquet(f"data/raw/{asset}.parquet")
    r = log_returns(df["close"])
    h = rolling_apply(r, fixed_bin_entropy, window=20)
    sigma = realized_volatility(r, window=20)
    valid = h.notna() & sigma.notna()
    return sigma[valid], h[valid]


def figure_1_scatter():
    """Figure 1: H vs sigma scatter, all three assets, with the outer bin edge
    and entropy ceiling marked, showing the saturation effect directly."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (asset, color) in zip(axes, ASSETS.items()):
        sigma, h = load(asset)
        ax.scatter(sigma, h, s=4, alpha=0.25, color=color, linewidths=0)
        ax.axvline(OUTER_EDGE, color="black", linestyle="--", linewidth=1,
                    label="outer bin edge (2%)")
        ax.axhline(CEILING, color="grey", linestyle=":", linewidth=1,
                    label="entropy ceiling")
        frac_above = float((sigma > OUTER_EDGE).mean())
        ax.set_title(f"{asset}  (n={len(sigma)}, {frac_above:.0%} beyond outer edge)")
        ax.set_xlabel("20-day realized volatility (σ)")
        ax.set_xlim(0, sigma.quantile(0.99))
    axes[0].set_ylabel("20-day fixed-bin entropy (bits)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Fixed-bin entropy vs realized volatility: saturation depends on bin-scale match")
    fig.tight_layout()
    out = FIGDIR / "fig1_entropy_vs_sigma.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    plt.close(fig)


def figure_2_kurtosis_bars():
    """Figure 2: excess residual by kurtosis quartile with CIs, from the
    already-computed registry results. Numbers are hardcoded from
    experiments/registry.jsonl so this figure matches exactly what is
    reported in the technical note; it does not recompute the bootstrap."""
    data = {
        "BTC": [(0.0111, 0.0017, 0.0211), (-0.0042, -0.0091, 0.0016),
                (0.0106, 0.0025, 0.0187), (0.0482, 0.0356, 0.0623)],
        "ETH": [(0.0074, -0.0035, 0.0174), (0.0442, 0.0214, 0.0663),
                (0.0037, -0.0065, 0.0137), (0.0773, 0.0462, 0.1101)],
        "SPX": [(-0.0007, -0.0038, 0.0023), (-0.0068, -0.0092, -0.0045),
                (-0.0077, -0.0095, -0.0058), (0.0164, 0.0116, 0.0212)],
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    quartile_labels = ["Q1", "Q2", "Q3", "Q4\n(highest kurtosis)"]
    for ax, (asset, color) in zip(axes, ASSETS.items()):
        vals = data[asset]
        means = [v[0] for v in vals]
        lo = [v[0] - v[1] for v in vals]
        hi = [v[2] - v[0] for v in vals]
        x = np.arange(4)
        ax.bar(x, means, color=color, alpha=0.7)
        ax.errorbar(x, means, yerr=[lo, hi], fmt="none", color="black", capsize=4)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(quartile_labels, fontsize=8)
        ax.set_title(asset)
    axes[0].set_ylabel("Excess residual variance\n(real − synthetic null)")
    fig.suptitle("Excess residual concentrates in the top kurtosis quartile\n(unsaturated windows only; SPX is a post-hoc check, see §5 footnote)")
    fig.tight_layout()
    out = FIGDIR / "fig2_kurtosis_quartiles.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    figure_1_scatter()
    figure_2_kurtosis_bars()
    print("\nDone. Embed figures/fig1_entropy_vs_sigma.png and "
          "figures/fig2_kurtosis_quartiles.png into technical_note.md.")
