# R12 VS-E — Single-Checkpoint Dual-Task Composition R0

Status: **PREREGISTERED — IMPLEMENTATION AND FORMAL RUN FORBIDDEN UNTIL THIS SPEC MERGES**

Authority class: `CURRENT_TASK_SCOPE_ONLY`.
Experiment role: `QUALIFICATION`.
Machine-readable authority: `config/experiments/r12_vs_e_single_checkpoint_composition_r0.json`.
Frozen preregistration SHA256: `6ca58656d2c29c976db762f1eb3eb6259208017ac0cae5060e5d02012c446fc5`.

## Objective

Test a capability that VS-C R3 explicitly did not establish: whether **one uninterrupted Actor/Critic learner checkpoint** can simultaneously retain the two already-controlled synthetic abilities after learning them in an interleaved schedule.

R3 remains valid in its original scope. VS-E R0 is not a reinterpretation or rescue of R3 and is unrelated to the stopped VS-D historical-development probe family.

## Existing interface / files

Reuse, without semantic change:

- VS-C R3 learner authority and optimizer;
- Task A account-conditioned maintenance environment, positive relation, zero-relation account-observation control, evaluation adapter and gates;
- Task B delayed-consequence environment, true delayed reward, shuffled-consequence control, evaluation adapter and gates;
- vectorized-batched-v1-compatible execution with at most four one-thread CPU seed workers.

Parent formal authority is VS-C R3 run `34908126391`, commit `465c8c916d0f67472b1575e177e8e24b1e29a4cf`, artifact digest `sha256:7b10221a4dea2cd245e2a00a947efb97a4894f3d6b1ee1586c7fda31cde6a276`.

## Required semantics

For each fresh paired seed `2501..2508`, construct one positive learner and one dual-control learner from bitwise-identical initial parameters.

Run exactly 256 cycles. Each cycle performs:

1. one Task A update;
2. one Task B update.

The same learner and optimizer state continue across both updates and all cycles. No parameter reset, optimizer reset, checkpoint copy, task-specific learner replacement, replay, or hidden extra update is allowed.

Learner generation is globally monotonic: Task A in cycle `c` consumes generation `2c`; Task B consumes generation `2c+1`. The inherited task-specific RNG/control generation index remains the local cycle index `c=0..255`, preserving R3 Task-A collection, Task-B collection, and Task-B shuffle streams.

Evaluate both Task A and Task B, without updating parameters, at exactly:

- PRE;
- after the final Task A update, before the final Task B update;
- after the final Task B update.

The R3-compatible final aggregate gates use the **after-final-B checkpoint** for both tasks. PRE-to-POST improvements mean PRE -> after-final-B.

A seed is a joint pass at a checkpoint only when that **same checkpoint** passes both inherited R3 per-seed Task A and Task B gates.

## Allowed scope

PASS may establish only single-checkpoint coexistence of these two synthetic controlled abilities under this exact interleaved schedule and may promote `PROMOTE_SINGLE_CHECKPOINT_DUAL_TASK_COMPOSITION`.

## Do not change

Do not change seeds, cycle count, A->B order, task batch sizes, task environments, optimizer, entropy coefficient, parent gates, controls, checkpoint selection, or add historical data. Do not tune after observing any VS-E result.

## Required tests

Before formal execution, tests must lock:

- exact preregistration bytes;
- parent VS-C R3 spec SHA/provenance;
- fresh non-overlapping seed set;
- exact 512-update schedule and global/local generation semantics;
- inherited Task A/B environments and gates;
- joint-pass definition and both-switch gate thresholds;
- no historical data access;
- exact-commit full repository suite gate.

Implementation tests must additionally prove a single learner identity and uninterrupted optimizer state are used across A/B updates, evaluation is read-only, global generation reaches 512, and paired positive/control initialization is bitwise identical.

## Done when

The preregistration merges before implementation; implementation then merges from that exact authority without modifying the spec; a formal Science run executes only the allowlisted exact-commit runner and emits `cb16.result.v2` with claim-local gate, inference, attribution and promotion fields.
