# CB16-R12 AI Infra Architecture

## Goal
Minimize coordination and transport overhead while preserving enough independent review to safely develop a stateful ML/RL system. The AI infrastructure must remain thinner than the system it helps build.

## Topology

```text
Master
  |
  v
Chat-SOL
  |  task contracts / review decisions
  v
GitHub
  |  issues / PRs / labels / CI
  v
OCI execution
  |
  v
DeepSeek API Builder
  |
  +--> code -> tests -> local iteration
  |
  +--> PR diff / Actions logs / BUILD_REPORT / artifacts
                           |
                           v
                        Chat-SOL
```

## Roles

### Master
Owns highest-level product/scientific intent, acceptable compromises, scope reductions, and decisions where multiple scientifically valid options exist.

### Chat-SOL
Owns architecture, semantic decomposition, formulas and exact logic for high-risk code, task contracts, interface ownership, PR review, and acceptance/request-changes decisions.

Chat-SOL is intentionally not an OCI execution node. Its observable world is GitHub: repository files, diffs, PR/Issue discussion, Actions jobs/logs, and selected artifacts/reports.

### DeepSeek Builder
Owns bounded implementation, local test/fix loops, mechanical refactoring, requested instrumentation, and concise build reporting. It does not own scientific meaning or architecture changes outside scope.

### GitHub
GitHub is the durable control plane, not a second scientific authority system.

- Issue = task contract
- PR = implementation workspace
- label = command/state signal when needed
- Actions = execution transport + observable logs
- Git history = normal version history
- review = semantic acceptance

### OCI
OCI is the canonical development execution environment. It should host the persistent repo/cache/environment, shared read-only data/model weights when needed, DeepSeek API access, and the self-hosted runner or equivalent thin dispatcher.

## GitHub-observable development
For each Builder task, Chat-SOL should be able to answer from GitHub:
1. What changed?
2. Why did it change?
3. What was executed?
4. Which tests passed or failed?
5. What relevant runtime failure/output occurred?
6. Does the implementation satisfy the task contract?

Default evidence is diff + targeted tests + affected-suite result + concise error/runtime summary. Request additional probes only when needed.

## Parallelism
Start with one Chat-SOL lead/reviewer and one active DeepSeek Builder. Expand to two Builders only for genuinely independent tasks. A PR has one writer; parallelism is across low-coupling PRs.

## Versioning
Normal Git is enough for ordinary development. Stronger identity is required only for released milestones, formal scientific experiments, and saved results/checkpoints that must be reproducible.

## Infra growth rule
New infrastructure is justified only by an observed blocker. Prefer direct local execution, tests/logging, bounded parallel worktrees/runners, then measured optimization. Avoid speculative distributed architecture.
