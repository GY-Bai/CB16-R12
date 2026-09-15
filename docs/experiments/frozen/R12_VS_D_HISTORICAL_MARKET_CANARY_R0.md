# R12 VS-D — Historical Market-Information Canary R0

Status: **PREREGISTERED — DO NOT RUN BEFORE IMPLEMENTATION MERGE**

Machine-readable authority: `config/experiments/r12_vs_d_historical_market_canary_r0.json`

## Question

After VS-C established controlled account conditioning and delayed credit assignment, does the same minimal R12 policy learn any **out-of-sample historical market-to-consequence relation** that is absent when the training consequence is deliberately shuffled?

This is the first historical canary. It is deliberately narrower than a continuing-account or profitability claim.

## Data boundary

Use only the exact BTCUSDT Binance USDM 1m archives for **2020-01, 2020-02 and 2020-03** listed in the machine-readable spec, with their frozen SHA256 values. These months were already authorized exploratory material for normalization.

Do not open any other market-data archive. The mounted manifest must still state `final_holdout_accessed=false`.

Aggregate 1m bars deterministically to UTC 1h bars: first open, max high, min low, last close, summed volume. Every hour must contain exactly 60 contiguous minute rows.

The canonical N0 input remains one predecessor plus 64 represented bars. Consequently:

- Jan-Feb training has exactly **1375** eligible next-hour consequences;
- March validation has exactly **744 = 31 x 24** next-hour consequences;
- March is development validation, **not** final holdout.
- Validation observations may causally use trailing February history before the March consequence hour, but no March consequence may enter training and no consequence/future bar may enter its own observation.

## Scientific isolation

Every historical sample starts from the same flat account (`W=1`, `q=0`). The policy observes N0(64h) plus the flat AccountState, samples `{SHORT, FLAT, LONG}` and requested risk, executes at the next 1h bar open and is marked at that bar close.

`kappa=0` and funding is off. This R0 therefore isolates **raw market decision information**. It intentionally does not test transaction-cost survival or continuing account path dependence.

No realized winner is ever constructed as a label. Realized next-hour price movement enters only through the post-action economic reward.

## Training

Eight fresh model seeds: `3101..3108`.

For each seed, positive and control learners start bitwise identically and both use categorical direction entropy coefficient `0.005`, Beta-risk entropy `0`, Adam LR `0.001/0.001`, no replay, and 256 generations.

Each generation uses all 1375 Jan-Feb windows.

Positive reward uses the matching next-hour `(open, close)` consequence.

Control keeps the same context set and exact consequence multiset but applies a deterministic fixed-point-free permutation of consequences every generation. The control may therefore learn unconditional historical drift, but cannot consistently learn the true context-to-next-hour relation.

## Validation

No optimizer update is permitted on March.

Evaluate the untrained PRE policy, trained positive policy and trained shuffled-control policy with the deterministic argmax-direction + Beta-mean adapter on the same 744 true March consequences.

March is divided into 31 UTC day blocks of exactly 24 consequences. Do not treat the 744 hours as independent observations.

For each model seed and each required comparison, generate 4096 deterministic day-block bootstrap replicates by sampling 31 day indices with replacement. The one-sided 95% lower bound is the zero-based order statistic at sorted index `204`.

A seed passes only when all three lower bounds are strictly positive:

1. POST positive mean one-step log-growth;
2. POST positive minus POST shuffled-control mean log-growth;
3. POST positive minus PRE untrained mean log-growth.

Aggregate PASS additionally requires at least 6/8 passing seeds and strictly positive medians across seeds for the same three point-estimate quantities.

## Interpretation

PASS means only `HISTORICAL_MARKET_INFORMATION_QUALIFIED`: under zero friction, a true historical context-to-consequence relation produced out-of-sample economic improvement beyond both shuffled-relation training and the untrained policy.

FAIL is `SCIENTIFIC_FAIL`; do not rescue by adding data, costs, features, generations, model capacity or threshold changes under this experiment identity.

If PASS, the next experiment may test cost-aware / continuing-account historical behavior. If FAIL, investigate representation/objective/horizon as a new preregistered question rather than opening final holdout.
