# Bootstrap Next Steps

The reduction pass is sufficiently complete to move from philosophy review into the first executable R12 slice.

1. Treat the current causal K-line normalization and minimal dimensionless `AccountState` as v1 candidates, not final scientific winners.
2. Use `docs/R12_ACTION_PERMISSION_ACCOUNT_PHYSICS.md` as the v1 candidate contract for target exposure, permission, next-open execution, transaction cost, account continuity, and transition recording.
3. Use `docs/R12_MINIMAL_LEARNER_CONTRACT.md` as the first controlled-learning baseline: strict on-policy Monte Carlo policy gradient with a learned value baseline, categorical direction head, and Beta requested-risk head.
4. Build one vertical slice:

```text
normalized K-line context
  -> frozen market representation
  -> Central Brain + AccountState
  -> direction + requested_risk
  -> permission / execution
  -> next AccountState
  -> log-equity reward
  -> complete on-policy trajectory
  -> return-to-go + value baseline
  -> actor/critic update
  -> frozen-policy evaluation
```

5. Qualify controlled learnability before historical claims. At minimum test account-conditioned action, delayed consequence credit, and matched negative/shuffled controls.
6. Add Buy & Hold economic comparison and the two-axis confidence-aware Pareto qualification path.
7. Only after the vertical slice is scientifically valid, connect a small authorized historical canary.
8. Add GAE/PPO, replay/off-policy correction, databases, GPU/distributed execution, or OCI automation only when a measured blocker requires them.

## Deferred scientific comparison

Normalization challengers are tracked in GitHub Issue #1. When scheduled, compare endpoint-anchored log-ratio, per-bar log returns, rolling causal z-score, RevIN-style normalization, and volatility-normalized returns on the same asset, same causal windows, same frozen sensory weights/tap, and the same evaluation protocol. Return the compact result package to Chat-SOL for review before final normalization authority is frozen.

Keep the first implementation local, inspectable, and small.
