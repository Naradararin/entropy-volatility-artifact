# CLAUDE.md — standing rules for this repository

Read this before doing anything. These rules exist to keep agent sessions cheap
and to stop silent errors, which are the dominant failure mode in backtest-style
research: broken code does not crash, it produces a *better-looking* number.

## What this project is

A falsification study, not a trading system. The claim under test is in
PREREGISTRATION.md. No capital is deployed at any stage. There is no execution
layer, no strategy, no portfolio. Do not add one.

## Hard rules

1. **Never touch the network.** Data is frozen under `data/raw/` with hashes in
   `data/manifest.json`. If a file is missing, stop and tell the user. Do not
   re-download, do not "helpfully" fetch a fresh copy.
2. **Never modify anything under `data/`.** It is frozen. A changed hash
   invalidates every result in the repo.
3. **Never edit a fixture to make a test pass.** Values in `fixtures/` were
   computed by hand. If code and fixture disagree, the code is wrong.
4. **Never change frozen parameters.** Bin edges, window length, date range and
   asset list are fixed by PREREGISTRATION.md §2–3. Changing one requires a
   logged amendment in §9 by the user, not by an agent.
5. **No new dependencies** without being asked. See `requirements.txt`.
6. **One task per session.** The task is normally "make test X pass". Do not
   refactor adjacent code, do not add features that were not requested.

## Layout

```
PREREGISTRATION.md   frozen claim, predictions, decision rules — read first
data/raw/            frozen Parquet. read-only.
data/manifest.json   source, range, row counts, SHA256 per file
fixtures/            hand-computed expected values
src/features/        pure functions: array in, number or array out. no I/O.
src/data/            the one place allowed to talk to the network, run by hand
tests/               must run in under 5 seconds
experiments/         registry.jsonl — one line per run, including failures
```

## Code conventions

- Pure functions in `src/features/`. No printing, no file reads, no plotting,
  no global state. Everything is testable without fixtures on disk.
- Log returns in decimal (`0.01` == +1%). Entropy in bits (log base 2).
- Rolling windows are right-aligned and backward-looking: the value at index
  `t` uses `t-window+1 .. t` inclusive and nothing after `t`. There is a test
  for this; do not weaken it.
- Type hints on public functions. Docstrings state units and conventions.
- Any function whose result appears in the paper needs a fixture test with a
  value that a human computed independently.

## Look-ahead bias

The single most likely way this project produces a wrong result. Forbidden:
- `.shift(-n)`, any negative shift, on a feature
- computing bin edges, means, or standard deviations over the full sample and
  applying them to earlier windows (quantile bin edges come from the training
  period only, and are frozen)
- `center=True` in any rolling call
- shuffled cross-validation on time series

## Experiment registry

Every analysis run appends one line to `experiments/registry.jsonl`:

```json
{"ts":"...","git_commit":"...","data_hash":"...","config":{...},"seed":0,"result":{...},"outcome":"ok|failed|abandoned"}
```

Failed and abandoned runs are logged too. The paper reports the total number of
runs. This is what makes the multiple-comparisons correction honest.

## Running

```
python3 -m pytest        # full suite, under 5s
```
