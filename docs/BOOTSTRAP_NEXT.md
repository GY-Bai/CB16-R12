# Bootstrap Next Steps

The reduction pass is sufficiently complete to move from philosophy review into the first executable R12 slice.

1. Freeze the causal K-line normalization contract and minimal scale-aware `AccountState` schema.
2. Freeze the smallest action / permission / execution / account-physics semantics needed for one single-asset Trader.
3. Build one vertical slice:

```text
normalized K-line context
  -> frozen market representation
  -> Central Brain + AccountState
  -> direction + requested_risk
  -> permission / execution
  -> next AccountState
  -> log-equity learning signal
  -> compact reconstructable trajectory
  -> learner update
  -> frozen-policy evaluation
```

4. Qualify controlled learnability before historical claims. At minimum test account-conditioned action, delayed consequence credit, and matched negative/shuffled controls.
5. Add Buy & Hold economic comparison and the two-axis confidence-aware Pareto qualification path.
6. Only after the vertical slice is scientifically valid, connect a small authorized historical canary.
7. Add more replay machinery, off-policy correction, databases, GPU/distributed execution, or OCI automation only when a measured blocker requires them.

Keep the first implementation local, inspectable, and small.
