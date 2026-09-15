# R12 Normalization Core A R0 — Builder Contract

## Objective

Implement only the deterministic raw BTCUSDT 1m loader/window selector and the five pure normalization transforms required by `docs/tasks/R12_NORMALIZATION_ABLATION_R0.md`.

Do not implement diagnostics, frozen projection, runtime preflight, Science allowlist entries, experiment artifacts, Central Brain, Account Physics, learner, or profitability tests in this PR.

## Existing authority

Read and implement the formulas literally from:

- `docs/tasks/R12_NORMALIZATION_ABLATION_R0.md`
- `docs/R12_CORE_OBJECTIVE_AND_QUALIFICATION_DRAFT.md`

If a formula is ambiguous, report it instead of inventing another method.

## Required package

Create a narrow package under:

`science/cb16_science/normalization/`

with only the minimum modules needed, preferably:

- `data.py`
- `transforms.py`
- `__init__.py`

No framework layer.

## Data loader

Support the canonical read-only Binance monthly ZIP layout exposed through `binance_um_1m_klines_10pairs`.

Required fixed exploratory scope for real-data smoke validation:

- BTCUSDT
- 1m
- 2020-01, 2020-02, 2020-03

Use standard-library `zipfile`/`csv` where practical. Handle Binance monthly kline CSV with or without a header deterministically. Parse at least open time, O/H/L/C/V to float64/integers as appropriate.

Never write into mounted data.

Implement validation for finite OHLCV, positive OHLC prices, non-negative volume, monotone timestamps, and OHLC ordering.

Implement deterministic approximately-even endpoint selection for at most 4096 valid represented windows with `L=64`, retaining one predecessor bar for return-based transforms. Same endpoint list must be reusable by all candidates.

## Pure transforms

Implement five pure functions exactly as frozen in `R12_NORMALIZATION_ABLATION_R0.md`:

- `N0_ENDPOINT_ANCHORED`
- `N1_PER_BAR_RETURN`
- `N2_SHARED_CLOSE_ZSCORE`
- `N3_REVIN_CHANNELWISE`
- `N4_VOL_NORMALIZED_RETURN`

Constants:

- `L = 64`
- `eps_v = 1e-12`
- `eps_std = 1e-8`
- float64 comparison math

Every successful transform returns shape `[64,5]` in channel order `O,H,L,C,V`.

Near-zero required standard deviation for N2/N4 must produce an explicit candidate-window invalid result/exception type; do not silently divide by epsilon and pretend the window is informative.

Transform code must have no dependency on GitHub, dispatcher, account state, RL, PyTorch, model weights, or OCI APIs.

## Required tests

At minimum:

1. N0 exact hand-computable formula.
2. N1 exact predecessor handling and formula.
3. N2 uses one shared close-derived mean/std for all four price channels.
4. N3 independently normalizes each channel.
5. N4 divides all four return geometry channels by close-return sigma.
6. N2/N4 explicit near-zero-sigma invalid handling.
7. N0/N1/N2/N4 preserve OHLC ordering on valid nondegenerate fixture where mathematically expected.
8. N3 geometry distortion is detectable on a constructed fixture; do not force it to preserve ordering.
9. price-scale invariance for all five methods where specified.
10. volume-unit invariance for all five methods.
11. invalid/non-positive prices fail explicitly.
12. deterministic endpoint selection.
13. CSV header and no-header parsing fixtures.
14. full existing repository test suite remains green.

## Real-data smoke check

During Builder execution only, inspect the advertised read-only BTCUSDT 2020-01..03 archives and run a bounded smoke parse/transform on a few windows. Report:

- resolved canonical data path;
- archive basenames found for the three months;
- whether parsing succeeded;
- transform output shapes/finite status on the smoke windows.

Do not commit data samples copied from the mounted dataset.

## Allowed scope

- `science/cb16_science/normalization/**`
- normalization-specific tests under `tests/`

Do not modify `config/cb16_science_allowlist.json`, dispatcher, workflows, host configuration, other science modules, or canonical semantic documents.

## Done when

Targeted tests pass, full suite passes, the real-data smoke check is reported, and the PR contains only the normalization core/data interface. End with `BUILD_REPORT`.