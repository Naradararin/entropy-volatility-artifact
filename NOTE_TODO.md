# Before this is done

1. Fill in [name], OSF link, GitHub link at the top of technical_note.md
2. Run `python make_figures.py` locally — if it errors on the import line,
   the function names in src/features/rolling.py or entropy.py may differ
   slightly from what's guessed here (log_returns, fixed_bin_entropy,
   rolling_apply, realized_volatility). Give Claude Code the error message
   and it will fix the two import lines only — do not let it touch the
   feature functions themselves.
3. Embed the two PNGs into technical_note.md after the Result 1 and
   Result 3 sections respectively:
   `![Entropy vs volatility by asset](figures/fig1_entropy_vs_sigma.png)`
   `![Excess residual by kurtosis quartile](figures/fig2_kurtosis_quartiles.png)`
4. Read the whole note once out loud before publishing — this catches
   sentences that sound confident but overclaim, which is the exact failure
   mode this whole project is about avoiding.
5. Make the repo public.
6. Commit and push.
