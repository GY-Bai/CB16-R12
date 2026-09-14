# R11 -> R12 Reduction Review

Status: OPEN FOR MASTER + SOL REVIEW

This is a decision worksheet, not a migration plan. R11 is a source of hard-earned invariants, useful code/tests, historical experiments, and accidental complexity. Nothing migrates merely because it exists.

## Review classes

### CORE
Project essence. Removing it would change what CB16 is.

### SIMPLIFY
Keep the invariant, replace the mechanism with a smaller one.

### DEFER
Potentially useful, but unnecessary for the first working R12 system.

### DROP
Requirement/infrastructure caused by previous implementation problems, temporary constraints, or governance accumulation and not intrinsic to CB16.

### UNKNOWN
Needs evidence or Master decision.

## Questions for each R11 requirement
1. Is this a scientific/product requirement or an implementation workaround?
2. What failure originally caused it to appear?
3. Does that failure still exist in a clean R12 architecture?
4. Can an executable test preserve the invariant instead of a governance protocol?
5. What is the smallest mechanism that preserves the intended property?
6. What capability becomes impossible if it is removed?
7. Is that capability needed before controlled learnability, historical training, or only live/production hardening?

## Initial protected essence candidates
These are candidates for review, not final frozen requirements.

- Trader acts on market + account state, not market alone.
- Truth / belief / decision / permission remain distinct.
- Nominal action and executed action remain distinct.
- Market history is reusable; experience depends on policy/account trajectory.
- Same-account temporal continuity must be real.
- Learning must be capable of changing future policy behavior.
- New data may calibrate current belief without requiring historical information to be manually erased.
- Historical recurrence should be represented primarily through learned model representation, not a large handcrafted cycle/rule activation engine.
- Scientific claims remain separated by strength: wiring != controlled learning != historical improvement != profitability.
- No final-holdout leakage.

## Initial reduction candidates
To be reviewed, not automatically deleted:

- multi-layer authority/receipt chains;
- routine SHA binding beyond Git;
- infrastructure qualification detached from a concrete scientific blocker;
- generalized distributed orchestration before real need;
- duplicate provenance systems;
- excessive branch-per-microstep workflows;
- multiple documents restating the same current state;
- custom ownership/lease systems where Git/PR/worktree already supplies isolation;
- speculative performance infrastructure;
- compatibility preservation for abandoned R11 implementations.

## Review output

| R11 concept | Class | R12 rule | Why | Test/guard |
|---|---|---|---|---|

The Master is the final decision authority on trade-offs.
