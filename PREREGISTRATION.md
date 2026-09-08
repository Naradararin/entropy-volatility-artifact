# Pre-Registration: Fixed-Bin Shannon Entropy as a Re-Encoding of Realized Volatility

**Status:** FROZEN — committed before any data was downloaded
**Author:** [name]
**Date frozen:** [YYYY-MM-DD]
**Commit hash at freeze:** [filled by git]

> This document is written *before* data acquisition and analysis. Any change after
> the freeze date must be recorded in the Amendments section at the bottom, with a
> date and reason. Amendments are permitted; silent edits are not.

---

## 1. Claim under test

Shannon entropy computed on daily returns discretized into **fixed absolute bins**
is a near-deterministic monotone re-encoding of realized volatility, rather than a
distinct measure of market information. The apparent independence of the two features
is an artifact of the binning scheme, not a property of the data.

### 1.1 Why this is not a novelty claim

For Gaussian returns, differential entropy is exactly `h = ln(σ√(2πe))` — a strictly
monotone function of σ (Cover & Thomas). This is textbook. The contribution is **not**
the discovery of the relationship. It is:

- **(A) Quantification.** Measuring how completely fixed-bin discretization collapses
  H onto σ, and where the residual information lives.
- **(B) Incremental-information null.** Testing whether H adds predictive information
  *conditional on* σ, which prior work largely does not isolate.
- **(C) Small-sample bias interaction.** Showing that the occupied-bin count K is
  itself a function of σ, so plug-in estimator bias injects *additional*
  volatility-correlated structure.
- **(D) Propagation audit.** Documenting that deployed systems still treat H and σ as
  independent features.

All four are required. Any subset invites the reviewer response "this is well known."

---

## 2. Frozen data specification

| Item | Value |
|---|---|
| Assets | BTC-USD, ETH-USD, plus one equity index (S&P 500) as a non-crypto anchor |
| Frequency | Daily close |
| Period | [START] to [END] — fixed before download, never extended to improve results |
| Source | [source + exact endpoint] |
| Storage | Parquet, SHA256 recorded in `data/manifest.json` |
| Timezone | UTC, close-to-close |

**Universe note.** This study makes no cross-sectional or strategy claim, so no
point-in-time universe is required and survivorship bias does not apply. If the scope
later expands to a cross-section of altcoins, this document is void and must be
re-frozen.

---

## 3. Operational definitions

Both computed on a rolling **20-trading-day** window, aligned to the same window end.

**Log return:** `r_t = ln(P_t / P_{t-1})`

**Realized volatility:** `σ_t = std(r_{t-19..t})`, sample standard deviation (ddof=1).

**Fixed-bin Shannon entropy (the feature under test):**
Bins on `r`: `(-∞, -2%]`, `(-2%, -1%]`, `(-1%, 1%]`, `(1%, 2%]`, `(2%, ∞)`.
`H_t = -Σ p_i log₂ p_i` over the 20 observations in the window, `p_i = n_i / 20`,
with `0 log 0 := 0`. Plug-in (maximum-likelihood) estimator.

**Comparison encodings** (the known fixes, computed for contrast):
- *Quantile bins:* bin edges = 20th/40th/60th/80th percentiles of the **training-period**
  return distribution only. Edges frozen; never recomputed per window.
- *Sigma bins:* fixed bin edges expressed in units of trailing σ rather than absolute %.
- *Permutation entropy:* Bandt-Pompe, embedding dimension m=3, lag=1. Scale-invariant
  by construction.

**Bias-corrected entropy:** plug-in, Miller-Madow (`+ (K̂-1)/(2N)`, K̂ = occupied bins),
and one shrinkage estimator (James-Stein / Dirichlet). All three reported side by side.

---

## 4. Predictions, recorded before running

Fill in the "My forecast" column **before** executing any analysis. This is the
calibration record; it is worthless if written afterwards.

| # | Prediction | My forecast | Claude's forecast |
|---|---|---|---|
| P1 | Spearman ρ(H_fixed, σ) on BTC daily ≥ 0.80 | ___% | 80% |
| P2 | R² of H_fixed on a monotone spline of σ ≥ 0.85 | ___% | 70% |
| P3 | Spearman ρ(H_permutation, σ) ≤ 0.30 | ___% | 75% |
| P4 | ρ(H_quantile, σ) materially lower than ρ(H_fixed, σ) | ___% | 85% |
| P5 | Incremental R² of H_fixed for next-20d σ, controlling for current σ, < 0.01 | ___% | 60% |
| P6 | Occupied-bin count K correlates with σ at ρ ≥ 0.7 | ___% | 80% |
| P7 | The above hold on the equity index as well as on crypto | ___% | 65% |
| P8 | Residual variance of H around the monotone fit on REAL data is within 25% of the same quantity on the synthetic null | ___% | 55% |
| P9 | ≥ 25% of windows have H pinned at the floor (0) or the ceiling (log₂5) | ___% | 60% |
| P10 | Excess residual over the null, if any, concentrates in the top kurtosis quartile | ___% | 65% |

Scoring: after results are in, record actual outcomes and compute a Brier score.
Log it in `calibration_log.json`. Repeat this table for every future experiment.

---

## 5. Decision rules — what each outcome means

Written now so the interpretation is not chosen after seeing the numbers.

### 5.1 ρ is a headline, not the evidence

Under a fixed-shape location-scale return distribution with mean ≈ 0, the bin
probabilities `p_i = F(edge_i) - F(edge_{i-1})` are a function of σ alone, so in
the population limit `H = g(σ)` **exactly**. The scientific question is therefore
not "is ρ high" but **"why is ρ not 1"**, which has exactly three sources:

1. **Sampling error.** N=20 makes p̂ ≠ p. This is estimator noise, not information.
2. **Shape variation.** Kurtosis and skew are not constant; at fixed σ a change
   in shape moves p and therefore H. **This is the only genuine information H holds.**
3. **Saturation.** H is bounded above by log₂5 ≈ 2.32 and below by 0, so at
   extreme σ it stops responding, biasing ρ *downward*.

A rank or linear correlation cannot separate these. The decision rules below are
therefore based on the **synthetic null**, not on ρ.

### 5.2 The synthetic null

Simulate returns whose realized-σ path matches the empirical path but whose
distributional **shape is held constant** (Gaussian innovations scaled to the
observed rolling σ). By construction this series contains *no* information beyond
σ. Compute H on it identically. Seeds fixed and logged; 1,000 replications.

`excess_residual = Var(resid_real) - Var(resid_null)` where residuals are taken
around the fitted monotone spline of H on σ.

### 5.3 Decision rules

| Outcome | Action |
|---|---|
| `excess_residual` not distinguishable from 0 (null CI covers the real value) | **Core framing confirmed.** The apparent independence of H from σ is estimator noise. Proceed to pillars C and D. |
| `excess_residual` > 0 and concentrated in high-kurtosis windows | Framing shifts to "H carries shape information that σ misses, but only in fat-tailed regimes." Still publishable; report the size of the effect and where it lives. This is a *result*, not a failure. |
| `excess_residual` > 0 and NOT concentrated in high-kurtosis windows | Unexplained. Do not interpret. Report as an open question and investigate implementation before claiming anything. |
| Floor/ceiling saturation exceeds 25% of windows | Report prominently regardless of other outcomes. A feature that is pinned a quarter of the time has a narrow effective operating range, which is a finding in its own right. |
| ρ(H_fixed, σ) < 0.50 | Inspect saturation and implementation first — under the analytics above, a low ρ is more likely a bug or an extreme-saturation regime than evidence of independence. Only after ruling both out, treat the framing as falsified. Do **not** adjust bin edges until ρ rises. |
| P5 fails (H_fixed does add incremental predictive information) | Report it. This is the most interesting possible result against my own hypothesis. |
| Permutation entropy also correlates highly with σ | Implementation error. Debug before interpreting — this would contradict scale-invariance. |

**Stopping rule.** The bin scheme, window length, and asset list defined in §2–3 are
fixed. If any is changed, the change is logged as an amendment with a reason, and all
results are reported for **both** the original and the amended specification.

---

## 6. Analysis plan

1. Compute all features; verify each against a hand-checked fixture of 25 rows.
2. Report Pearson and Spearman ρ between each entropy variant and σ, with
   block-bootstrap confidence intervals (overlapping windows induce autocorrelation —
   naive standard errors will be wrong).
3. Fit a monotone spline of H on σ; report R² and plot residuals against realized
   kurtosis. **Expected:** residual structure concentrates in high-kurtosis periods,
   which is exactly the information coarse fixed bins discard.
3b. **Synthetic null (§5.2).** Repeat step 3 on shape-constant simulated returns
   matched to the empirical σ path. Report `excess_residual` with a bootstrap CI.
   This is the primary evidence; ρ and R² are descriptive only.
3c. **Saturation audit.** Report the fraction of windows with H = 0 and with
   H = log₂5, per asset, and the σ range over which H is responsive.
4. Incremental-information test: predict next-period σ and next-period return sign
   from {σ} vs {σ, H}. Report incremental R², likelihood-ratio test, and
   information coefficient. Time-series cross-validation only; no shuffling.
5. Estimator-bias panel: plug-in vs Miller-Madow vs shrinkage, and ρ(K, σ).
6. Repeat on the equity index.

**Multiple comparisons.** The full grid is 4 entropy variants × 3 estimators ×
3 assets × 2 targets = 72 tests, plus the synthetic-null replications (which are one pre-specified test, not 1,000). Every test run is logged in
`experiments/registry.jsonl`, including failures. Headline claims are corrected via
Benjamini-Hochberg. The count of tests run is reported in the paper.

---

## 7. Propagation audit protocol (pillar D)

Target: at least 5 public artifacts (GitHub repos, TradingView indicators, published
papers, textbook code) that compute entropy over fixed price/return bins and use it
alongside a volatility feature as though the two were independent.

For each: record URL, retrieval date, exact binning scheme, whether a volatility
feature is present, and whether any redundancy caveat appears. Archive a snapshot.

**No naming-and-shaming.** Findings are reported in aggregate, describing patterns
rather than criticising individual authors.

---

## 8. What this study does not claim

- Not a claim that entropy is useless in finance.
- Not a claim that this artifact caused anyone to lose money.
- Not a claim of novelty for the entropy-volatility relationship itself.
- Not a trading strategy, and no capital is deployed at any stage.
- No causal claim about market behaviour whatsoever.

---

## 9. Amendments

| Date | Change | Reason |
|---|---|---|
| | | |
