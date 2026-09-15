# R12 VS-C — Controlled Learnability Qualification R0

Status: **PREREGISTERED — IMPLEMENT, THEN RUN IMMUTABLE SCIENCE**

This task is the first scientific qualification of the merged VS-A/VS-B spine. It does **not** test historical profitability. It asks only whether the current sequential learner can learn two known-answer economic relations under controlled synthetic conditions.

The machine-readable authority for all budgets and gates is:

`config/experiments/r12_vs_c_r0.json`

That file was committed before VS-C implementation or qualification. Builder must not change its budgets, thresholds, seeds, controls, or gates.

## Objective

Implement exactly two controlled tasks and their matched negative controls:

```text
Task A: identical market + different AccountState
        -> economically correct target differs

Task B: early action -> immediate reward carries no information
        -> later account consequence carries the credit signal
```

The final formal run must use the merged VS-A Physics and VS-B learner unchanged unless a genuine contract mismatch is found before qualification. A gate miss is `SCIENTIFIC_FAIL` at **L3 — Controlled learnability** for the specific R0 claim, not permission to rescue R0 and not a project-wide scientific verdict.

## Existing authority

Use literally:

- `config/experiments/r12_vs_c_r0.json`
- `docs/R12_FIRST_VERTICAL_SLICE_TASK.md`
- `docs/R12_MINIMAL_LEARNER_CONTRACT.md`
- `docs/tasks/R12_VS_A_PHYSICS_R0.md`
- `docs/tasks/R12_VS_B_LEARNER_SPINE_R0.md`
- current merged `science/cb16_science/vslice/*`

Do not alter N0, AccountState, Permission, Physics, Actor/Critic probability semantics, return-to-go, or generation rules to make this task pass.

## Runtime and reproducibility

Formal qualification is CPU-only.

For every run:

- use the eight paired seeds in the preregistered JSON;
- call `torch.manual_seed(seed)` before constructing each positive/control learner;
- use identical initial Actor/Critic parameters for the positive and matched control of a seed;
- use the fixed FrozenSensory seed already defined by VS-B;
- set Torch deterministic algorithms on and Torch thread count to one;
- use deterministic local RNGs for task scheduling/control permutations;
- do not use wall-clock entropy;
- no GPU, replay, PPO, GAE, Teacher, scheduler, entropy rescue, early stopping, or adaptive budget changes.

The runner must evaluate each learner with the frozen deterministic policy adapter **before training and after the final preregistered generation**.

### Locked Python runtime for the immutable Science lane

The Science lane's host `python3` does not carry VS-B PyTorch. Formal qualification must therefore launch from the committed `pyproject.toml` + `uv.lock`; absence of Torch from host Python is not permission to alter the host.

The allowlisted formal command must be argv-based, not shell text, and must be semantically equivalent to:

```text
uv run --frozen --project . --python 3.12 python -m cb16_science.vslice.qualification
```

with environment:

```text
PYTHONPATH=science
UV_PROJECT_ENVIRONMENT=/tmp/cb16-vs-c-venv
UV_CACHE_DIR=/tmp/cb16-vs-c-uv-cache
UV_LINK_MODE=copy
```

The Science profile already provides per-run writable tmpfs at `/tmp`; source/worktree and `/cb16/store` remain read-only. Thus dependency materialization may occur only in ephemeral `/tmp`, while code/spec remain immutable. Exact package versions come only from `uv.lock`.

Do not point Science at a Builder branch `.venv`. Do not write a virtualenv into the Science worktree. Do not modify host Python. If the locked UV environment cannot be created or resolved, classify the run `EXECUTION_BLOCKED`; do not change scientific gates or dependencies under R0.

Builder itself should continue to use the task packet's branch-local `UV_PROJECT_ENVIRONMENT` and shared writable `/cb16/cache/uv` cache. The formal Science command above intentionally uses its own ephemeral environment.

## Task A — Account-conditioned maintenance

### Market

Create one valid canonical raw window:

- one predecessor + 64 represented bars;
- every OHLC price = `100.0`;
- every volume = `10.0`.

Pass it through the existing `normalize_market_window()` N0 boundary and then the existing FrozenSensory. The normalized market and `Z_t` must therefore be identical across all three account cases.

### Actual account truths

Use exactly:

```text
SHORT_HALF: equity=1000, quantity=-5, mark_price=100 -> desired target=-0.5
FLAT:       equity=1000, quantity= 0, mark_price=100 -> desired target= 0.0
LONG_HALF:  equity=1000, quantity=+5, mark_price=100 -> desired target=+0.5
```

One trajectory has one decision/transition.

Physics is the preregistered Task-A config:

```text
open_next = 100
close_next = 100
kappa = 0.10
permission_bisection_iterations = 24
nominal exposure budget = 1
hard exposure limit = 1
```

There is no price edge. Economic reward is only the VS-A transaction-cost consequence. Maintaining the existing target exposure costs zero; unnecessary movement costs capital.

Do not provide the desired target as a training label. It is evaluation-only known-answer authority.

### Positive training

Each generation contains 90 trajectories:

```text
30 actual SHORT_HALF observed as SHORT_HALF
30 actual FLAT       observed as FLAT
30 actual LONG_HALF  observed as LONG_HALF
```

The Actor receives the true canonical AccountState derived from the actual truth.

### Task-A zero-relation control

Each generation also contains 90 trajectories, but the actual account and the policy-observed AccountState must be exactly independent while preserving both marginals:

```text
all 3 x 3 actual-account × observed-account pairs
10 trajectories per pair
```

Reward/Physics always use the actual AccountTruth. The Actor receives the designated observed AccountState. This is a negative control only.

Do not substitute a fixed bijection, because a fixed remapping still contains perfect information.

### Evaluation

Evaluation always uses the true AccountState and deterministic VS-B adapter.

For each of the three truths record:

- direction;
- requested_risk;
- nominal target exposure `direction * requested_risk * budget`;
- target absolute error versus `[-0.5, 0, +0.5]`;
- one-step VS-A log-growth under the deterministic action.

Record PRE and POST values.

Compute exact seed and aggregate gates from the JSON. Builder may not reinterpret them.

## Task B — Delayed consequence credit

### Cue windows

Create two valid 65-row raw OHLCV windows with endpoint close `100`, volume `10`, and 64 represented closes.

For represented index `j = 0..63`:

```text
UP close_j   = 100 * exp(-0.04 * (63-j)/63)
DOWN close_j = 100 * exp(+0.04 * (63-j)/63)
```

For each represented bar:

```text
open_j = previous represented close
first represented open = predecessor close
high = max(open, close) * 1.001
low  = min(open, close) / 1.001
volume = 10
```

Use a valid predecessor whose close equals the first represented close. Predecessor OHLC may all equal that close and volume is 10.

Create one neutral window with all OHLC `100` and volume `10`.

All windows go through canonical N0 and FrozenSensory. Do not bypass normalization.

### Two-step economic trajectory

Start every trajectory from:

```text
equity=1000, quantity=0, mark_price=100
```

Use zero transaction friction and hard exposure limit 1.

At step 0:

```text
state market = UP or DOWN cue
open_1 = 100
close_1 = 100
```

The immediate reward must be exactly zero under the frozen Physics apart from normal floating representation. Assert it in tests and fail if the contract is not met.

At step 1:

```text
state market = neutral window
AccountState = true account state after action_0
UP future:   open_2 = close_2 = 110
DOWN future: open_2 = close_2 = 90
```

By the frozen next-open timing, the position created by action_0 remains economically active through the gap before action_1 executes. With `kappa=0` and `close_2=open_2`, action_1 has no contemporaneous PnL effect. Therefore the delayed `reward_1` is the economic consequence that must credit action_0 through undiscounted return-to-go.

No realized-winner action label is allowed.

### Positive training

Every generation contains 128 complete two-step trajectories:

```text
64 UP cue
64 DOWN cue
```

Use each trajectory's own Physics rewards.

### Shuffled action-to-consequence control

Collect the same kind of trajectories, states, actions, and Physics rewards. Before learner update, preserve the complete multiset of delayed `reward_1` values but assign them to trajectories using a deterministic seed-and-generation-specific permutation with **no fixed points**.

Requirements:

- permutation must be derived only from preregistered seed + generation;
- no trajectory may retain its own delayed reward;
- `reward_0` remains its original zero;
- states and actions are unchanged;
- reward multiset is exactly preserved;
- permutation varies deterministically across generations so the control does not encode one stable remapping.

Construct new immutable trajectory records rather than mutating frozen records in place.

### Evaluation

Evaluate on the true, unshuffled environment using the deterministic adapter.

Record PRE and POST for UP and DOWN:

- direction;
- requested_risk;
- true delayed log-growth;
- categorical probabilities;
- direction probability margin:

```text
0.5 * ((p_LONG(UP) - p_SHORT(UP)) +
       (p_SHORT(DOWN) - p_LONG(DOWN)))
```

The seed and aggregate gates are exactly those in the JSON.

## Gate logic

The implementation must parse the machine-readable gates and compute them mechanically.

### Task A seed PASS

Both must hold:

- all 3 deterministic directions are correct;
- mean target-exposure absolute error <= `0.15`.

Task A aggregate PASS requires all four JSON aggregate conditions.

### Task B seed PASS

All must hold:

- UP deterministic direction = LONG;
- DOWN deterministic direction = SHORT;
- both requested risks >= `0.5`;
- true mean delayed log-growth >= `0.03`.

Task B aggregate PASS requires all four JSON aggregate conditions.

### Global PASS

Both Task A and Task B aggregate gates PASS and the full implementation test suite is green.

If implementation is correct but a scientific gate misses, top-level result classification is exactly:

`SCIENTIFIC_FAIL`

Its semantic scope is **L3 — Controlled learnability**: only the frozen R0 controlled-learning proposition is rejected. L0/L1 implementation and reward-loop evidence that independently passed remains evidence; L4 historical-information and L5 economic-robustness claims are not reached. See `docs/R12_FAILURE_LOCALITY_AND_SCIENTIFIC_LAYERS.md`.

Do not rerun with different thresholds, learning rates, budgets, architectures, seeds, friction, gap magnitude, sensory seed, or controls under the same R0 identity.

## Builder implementation scope

Builder may add small modules under:

```text
science/cb16_science/vslice/
```

Suggested names:

```text
controlled_tasks.py
qualification.py
```

Builder may add focused tests and may add one Science allowlist entry:

```text
cb16.vs-c-controlled-learnability@v1
```

The allowlist entry must use the locked UV argv/runtime defined above, not bare host `python3`.

The formal command must use the repository preregistered spec; no CLI argument may override scientific parameters in R0.

Builder may run tiny smoke fixtures or reduced-budget unit tests, but **must not execute the formal preregistered 8-seed qualification**.

## Required implementation tests

At minimum prove:

1. Task-A flat raw window passes canonical N0 and yields identical normalized market for all accounts.
2. Task-A actual account states are exactly `-0.5/0/+0.5` exposure and market representation is identical.
3. Positive Task-A batch has exact 30/30/30 balance.
4. Task-A control has every actual×observed pair exactly 10 times and independent marginals.
5. Task-A reward is produced only through merged VS-A Physics; maintaining target gives zero turnover cost.
6. Task-B UP/DOWN raw windows pass canonical market validation and are distinct after N0/FrozenSensory.
7. Task-B `reward_0 == 0` for sampled valid actions with kappa zero.
8. Task-B delayed reward changes sign consistently for a hand-set LONG versus SHORT action_0 under UP/DOWN gaps.
9. Changing action_1 with fixed action_0 does not change the two-step final equity/reward under the frozen Task-B construction.
10. Positive Task-B batch is exactly 64 UP / 64 DOWN.
11. Shuffled control permutation has no fixed points, preserves reward multiset exactly, and is deterministic for seed+generation.
12. Shuffled control does not mutate original trajectory objects.
13. PRE evaluation occurs before any learner update and POST only after exactly the preregistered generation count.
14. Positive/control learners for each paired seed begin with bitwise-identical Actor and Critic parameters.
15. Each training generation consumes only current-generation complete trajectories and increments exactly once.
16. Gate evaluator reproduces hand-computed Task-A and Task-B pass/fail examples.
17. Formal runner reads the committed JSON and offers no parameter override path.
18. Result serialization contains per-seed PRE/POST positive/control metrics and every gate component.
19. `experiment_spec.json` written to result dir is byte-for-byte equivalent in parsed JSON content to the committed spec.
20. allowlist resolves to the frozen UV command and keeps the venv/cache under `/tmp`, never in the read-only Science worktree.
21. full repository suite remains green.

## Formal artifacts

The Science entrypoint writes only under `CB16_RESULT_DIR`:

```text
experiment_spec.json
RESULT.json
REPORT.md
```

`RESULT.json` must contain:

- experiment id;
- exact commit SHA from `CB16_COMMIT_SHA`;
- runtime versions/device;
- implementation-test status supplied by the runner or recorded as preregistered requirement;
- for every paired seed: Task-A PRE/POST positive and control metrics + seed gates; Task-B same;
- aggregate gate components;
- Task-A aggregate verdict;
- Task-B aggregate verdict;
- global verdict;
- classification.

`REPORT.md` must clearly separate:

```text
Implementation correctness
Task A controlled learnability
Task B delayed-credit learnability
Negative controls
Scientific verdict
Limitations
```

Loss curves may be logged as diagnostics but cannot determine PASS.

## Forbidden scope

Do not:

- use historical market data or mounted R10/R11 assets;
- modify normalization, VS-A Physics, VS-B Actor/Critic/learner semantics;
- open final holdout;
- add replay/PPO/GAE/V-trace/Teacher/entropy rescue;
- change the preregistered JSON;
- tune based on formal results;
- add GPU/distributed/database/throughput work;
- modify Builder Orchestrator, dispatcher, workflows, sandbox, session registry, or credentials except the explicitly authorized Science allowlist entry.

## Done when

Builder is done when the synthetic environments, controls, evaluator, formal artifact writer, allowlist entry, locked UV launch, and regression tests exist and pass without running formal qualification. Chat-SOL then reviews/merges the implementation and separately triggers the immutable Science run against the exact merged commit and frozen JSON.
