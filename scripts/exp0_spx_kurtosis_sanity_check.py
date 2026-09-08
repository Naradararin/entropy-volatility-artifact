"""SPX kurtosis-quartile sanity check -- POST-HOC, NOT PRE-REGISTERED.

PREREGISTRATION.md 5.3 sets the decision rules for the excess_residual test.
The kurtosis-quartile-split + bootstrap-CI procedure applied here (already
run for BTC and ETH in scripts/exp0_kurtosis_quartiles.py and
scripts/exp0_kurtosis_bootstrap.py) was devised during this analysis in
direct response to the pattern those two assets showed. It is not one of the
tests specified in PREREGISTRATION.md sections 2-6. Running it on SPX now,
after seeing the BTC/ETH result, is therefore a post-hoc sanity check, not a
pre-registered replication -- log it in experiments/registry.jsonl as such,
and do not fold it into the pre-registered test count used for the
Benjamini-Hochberg correction (PREREGISTRATION.md 6, "Multiple comparisons").

Reuses run_asset from exp0_kurtosis_quartiles.py and real_resid_and_quartiles /
bootstrap_excess_ci from exp0_kurtosis_bootstrap.py verbatim -- both are
already asset-agnostic (parameterized by asset name), so none of that logic
is duplicated or rewritten here. No changes to entropy.py, rolling.py,
moments.py, the frozen bin edges, or the two reused scripts. No new
dependencies.

Restricted to SPX's unsaturated windows only (sigma <= 0.02), same as
BTC/ETH -- this restriction is inherited unchanged from run_asset.

Print only; no files written.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.exp0_kurtosis_bootstrap import bootstrap_excess_ci, real_resid_and_quartiles
from scripts.exp0_kurtosis_quartiles import run_asset

ASSET = "SPX"
BOOTSTRAP_SEED_BASE = 200  # distinct from BTC's 0 / ETH's 100 in exp0_kurtosis_bootstrap.py


def main() -> None:
    print("=== POST-HOC SANITY CHECK -- NOT PRE-REGISTERED (see PREREGISTRATION.md 5.3) ===")
    print("Kurtosis-quartile split + bootstrap CI, devised after seeing the BTC/ETH")
    print("pattern; applied here to SPX's unsaturated windows only. Log in")
    print("experiments/registry.jsonl as post-hoc, not a pre-registered replication.")
    print()

    quartile_rows = run_asset(ASSET)
    resid_unsat, quartile_idx = real_resid_and_quartiles(ASSET)

    rows = []
    for q, qrow in enumerate(quartile_rows):
        mask = quartile_idx == q
        resid_q = resid_unsat[mask]
        seed = BOOTSTRAP_SEED_BASE + q
        lo, hi = bootstrap_excess_ci(resid_q, qrow["null_mean"], seed)
        rows.append(
            {
                "asset": ASSET,
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
