# Fixed-bin entropy vs realized volatility

A falsification study. The claim, predictions and decision rules are frozen in
[PREREGISTRATION.md](PREREGISTRATION.md), registered on OSF before any data was
downloaded.

**Not a trading system.** No capital is deployed at any stage.

## Status

- [ ] Pre-registration filed on OSF — DOI: 10.17605/OSF.IO/GY6FU
- [ ] Data frozen (`python3 -m src.data.freeze`)
- [ ] Experiment 0: correlation between entropy variants and realized volatility
- [ ] Incremental-information test
- [ ] Estimator-bias panel
- [ ] Propagation audit
- [ ] [GitHub repo — https://github.com/Naradararin/entropy-volatility-artifact]

## Setup

```
pip install -r requirements.txt
python3 -m src.data.freeze     # run ONCE, by hand
python3 -m pytest              # must pass in under 5s
```

## Reading order

1. `PREREGISTRATION.md` — what is being claimed and what would falsify it
2. `CLAUDE.md` — conventions and hard rules
3. `fixtures/entropy_windows.csv` — hand-computed expected values
4. `src/features/entropy.py` — the feature under test
