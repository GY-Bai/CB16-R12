# R12 Normalization Ablation R0 — Builder + Science Contract

Status: READY FOR BOUNDED IMPLEMENTATION

## Objective

Implement two deterministic Science-lane tools needed before the first Vertical Slice:

1. `runtime_preflight`: report the CPU Python/NumPy/PyTorch runtime and readability of canonical mounted R12 inputs.
2. `normalization_ablation`: compare five K-line normalization families on identical BTCUSDT 1m windows without Central Brain training.

This task must not implement Account Physics, Central Brain learning, historical profitability, or any R10/R11 training method.

## Fixed data scope

Use only the canonical read-only market-data interface advertised as `binance_um_1m_klines_10pairs` (currently `/cb16/raw/klines_1m`).

Asset: `BTCUSDT` only.
Exploratory interval: `2020-01-01` through `2020-03-31` inclusive.
Bar frequency: `1m`.
Context length: `L = 64` represented bars.
Maximum evaluation windows: `4096`.
Window selection: deterministic approximately-even spacing over all valid window endpoints in the interval; no random sampling. Every method receives exactly the same endpoint set.

This interval is exploratory normalization material only. It must not be described as a profitability or holdout result.

The loader may inspect the actual read-only archive layout and CSV header/no-header convention at runtime, but must not modify, extract persistently into, or rewrite the mounted data tree. Standard-library `zipfile`/`csv` plus NumPy is preferred.

## Runtime preflight

Implement an allowlisted Science command `cb16.runtime-preflight@v1` that writes `RESULT.json` and `REPORT.md` containing only non-secret bounded information:

- Python version;
- NumPy installed/version;
- PyTorch installed/version and CPU import success;
- CUDA availability only as a boolean diagnostic (GPU is not required);
- readability/existence booleans for canonical `/cb16/raw`, `/cb16/frozen`, `/cb16/brain_assets` inputs advertised through the data manifest;
- discovered BTCUSDT 2020-01..2020-03 archive count and basename(s), not arbitrary host-directory dumps;
- whether at least one BTCUSDT archive can be opened and at least one valid OHLCV row parsed.

Do not print secrets, home-directory inventories, credential paths, or model contents.

## Common raw-window contract

For each endpoint `t`, retain one additional preceding bar when required by return-based transforms. Validation must reject non-finite OHLCV, non-positive OHLC prices, and malformed OHLC ordering (`H < max(O,C)` or `L > min(O,C)`).

All transform calculations use float64 for the comparison.

Volume numerical floor: `eps_v = 1e-12`.
Standard-deviation floor: `eps_std = 1e-8`.

## Candidate N0 — endpoint-anchored log ratios (current R12 candidate)

For every represented bar `tau` in `[t-L+1, t]`:

- `pO = log(O_tau / C_t)`
- `pH = log(H_tau / C_t)`
- `pL = log(L_tau / C_t)`
- `pC = log(C_tau / C_t)`
- `v = log(max(V_tau / median(V_window), eps_v))`

Output shape: `[L, 5]`.

## Candidate N1 — per-bar log-return geometry

For each represented bar `tau`, using previous close `C_{tau-1}`:

- `pO = log(O_tau / C_{tau-1})`
- `pH = log(H_tau / C_{tau-1})`
- `pL = log(L_tau / C_{tau-1})`
- `pC = log(C_tau / C_{tau-1})`
- volume uses the same relative-window median formula as N0.

This requires one raw predecessor bar before the represented window. Output shape: `[L, 5]`.

## Candidate N2 — causal window z-score with shared price statistics

Within the fully observed decision window only:

- form `xO=log(O)`, `xH=log(H)`, `xL=log(L)`, `xC=log(C)`;
- compute `mu_C = mean(xC_window)` and `sd_C = std(xC_window, ddof=0)`;
- if `sd_C < eps_std`, the price channels are invalid for this candidate/window;
- normalize all four price channels with the SAME close-derived statistics: `(x - mu_C) / sd_C`;
- for volume use `xV = log(max(V, eps_v))`, then `(xV - mean(xV)) / max(std(xV), eps_std)`.

Output shape: `[L, 5]`.

The shared price statistics are intentional: OHLC order should not be broken merely by channel-specific affine transforms.

## Candidate N3 — RevIN-style per-channel instance normalization

This is a deliberately literal representation-only RevIN-style challenger, with no learned affine parameters and no denormalization stage:

- channels are `log(O)`, `log(H)`, `log(L)`, `log(C)`, `log(max(V, eps_v))`;
- independently for each channel over the current observed window, subtract that channel's window mean and divide by `max(channel_std, eps_std)`.

Output shape: `[L, 5]`.

Record any OHLC ordering violations introduced in normalized coordinates; do not repair them.

## Candidate N4 — volatility-normalized return geometry

Start from the four N1 price-return channels. Compute close-to-close log returns over the represented window, including the predecessor needed for the first return. Let `sigma = std(close_log_returns, ddof=0)`.

- if `sigma < eps_std`, the price channels are invalid for this candidate/window;
- divide all four N1 price-return channels by `sigma`;
- volume uses the same relative-window median formula as N0/N1.

Output shape: `[L, 5]`.

## Invariant probes

For every method, on the same selected windows, measure rather than assume:

1. finite-output rate;
2. causality/no-lookahead: changing raw bars strictly after endpoint `t` cannot change state `t`;
3. price-scale invariance: multiply all raw OHLC in the required source window by `100`, report maximum absolute transformed difference;
4. volume-unit invariance: multiply all raw V by `1000`, report maximum absolute transformed difference;
5. OHLC ordering preservation in transformed coordinates: fraction of represented bars satisfying `H >= max(O,C)` and `L <= min(O,C)`;
6. candle-range information: Pearson correlation across all represented bars between transformed `(H-L)` and raw `log(H/L)`; if a transform changes the natural scale, report that fact rather than coercing it;
7. numerical magnitude: per-channel mean/std and global p01/p50/p99/max-abs;
8. runtime: wall-clock normalization time over the identical window set, repeated exactly 3 times after one warm-up; report median seconds. Do not optimize methods differently.

Causality should be implemented as a deterministic test/probe using a bounded sample, not by reading future values in normal transform code.

## Frozen sensory diagnostic

This experiment compares normalization, not pretrained sensory quality. To avoid importing a second uncontrolled variable, use one deterministic frozen linear projection common to all methods:

- flatten `[L,5]` to length `320`;
- construct a fixed Gaussian projection matrix `[32,320]` from NumPy `Generator(PCG64(seed=24680))`;
- row-normalize the matrix to unit L2 norm once;
- never train or mutate it;
- `Z = W @ flatten(K_norm)`.

For each candidate report:

- fraction of finite `Z` rows;
- per-dimension variance summary (min/median/max);
- covariance effective rank, defined as `exp(-sum(p_i * log(p_i)))` over normalized non-negative covariance eigenvalues, with zero-eigenvalue terms omitted;
- representation norm p01/p50/p99;
- scale-invariance representation error after the same price/volume rescaling probes;
- deterministic checksum/hash of projection configuration, not a huge matrix dump.

Do not declare a model winner from this projection. It is only a common information/stability probe.

## Interpretation fields (machine-readable, not a winner)

For each candidate, `RESULT.json` must include explicit booleans or labels for:

- `preserves_nominal_scale_invariance`;
- `preserves_ohlc_ordering`;
- `preserves_relative_volatility_amplitude` (semantic property, not inferred from downstream profitability);
- `invertible_to_relative_price_path_up_to_common_scale` where mathematically applicable;
- `known_information_removed` as a short enum/list such as `window_location`, `window_scale`, `volatility_amplitude`, `per_channel_relative_geometry`.

Builder code must not rank or promote a winner.

## Output artifacts

Implement allowlisted Science command `cb16.normalization-ablation@v1` producing exactly:

- `experiment_spec.json`
- `RESULT.json`
- `REPORT.md`

`experiment_spec.json` freezes all values in this contract plus the exact resolved input archive basenames and code commit SHA supplied by the dispatcher.

`RESULT.json` must contain per-candidate metrics and a top-level status. Formal run status is `OK` if the experiment executes and emits complete valid evidence; a candidate may have invalid windows without turning the whole experiment into a scientific PASS/FAIL claim.

`REPORT.md` summarizes evidence compactly and explicitly states: `NO WINNER PROMOTED BY BUILDER`.

## Required implementation structure

Keep the implementation small under `science/cb16_science/normalization/` or an equally narrow package. Separate at least:

- data loading/window selection;
- transform functions;
- diagnostics/projection;
- Science entrypoint/output formatting.

Pure transform functions must not know GitHub, OCI, the dispatcher, account state, or Central Brain code.

## Tests

Add deterministic unit/integration tests for at least:

- exact N0 formula on a hand-computable window;
- exact N1 formula including predecessor handling;
- N2 uses shared close statistics for O/H/L/C;
- N3 is independently normalized per channel and may be detected as geometry-distorting;
- N4 uses close-return sigma and rejects/fails candidate-window cleanly on near-zero sigma;
- price scale invariance;
- volume unit invariance;
- no-lookahead with modified future rows;
- invalid/non-positive price handling;
- deterministic endpoint selection;
- deterministic projection checksum/output;
- result schema contains all five candidate IDs;
- preflight does not emit secret environment values;
- allowlist resolves both new Science commands;
- existing full repository suite remains green.

## Allowed files

May add normalization/preflight modules and tests, and may update only the Science allowlist as needed to expose the two fixed entrypoints.

Do not modify dispatcher semantics, GitHub workflows, OCI host configuration, R12 Account/Action/Physics contracts, or existing mounted data.

## Done when

- targeted tests pass;
- full repository tests pass;
- both entrypoints can execute locally against fixtures;
- Builder reports any actual read-only BTC archive layout it inspected;
- one coherent PR is ready for Chat-SOL review;
- no scientific winner is promoted and no Central Brain training is performed.
