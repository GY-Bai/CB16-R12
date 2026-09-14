# AGENTS.md

Keep this file small. It contains rules that apply to nearly every coding task.

## Authority
The current task contract plus current code/tests are authoritative for implementation. Do not infer requirements from R11 unless the task explicitly imports them.

## Builder role
Builders implement bounded tasks. They do not redesign scientific semantics. For formulas, state transitions, accounting rules, learning targets, masks, terminal/truncation behavior, or promotion criteria, implement the specified logic literally and flag contradictions instead of inventing replacements.

## Workflow
1. Read the task contract.
2. Inspect only the code needed for the task.
3. Implement the smallest coherent change.
4. Run targeted tests.
5. Run the affected test suite.
6. Produce a concise BUILD_REPORT.
7. Push one coherent task branch / PR.

## Git
- One PR has one active writer at a time.
- Parallel work happens across independent PRs.
- Intermediate edit/test loops do not need commits.
- Prefer squash merge for completed bounded tasks.

## Default restraint
Do not add new distributed services, custom authority databases, custom lease/fencing systems, speculative performance infrastructure, or compatibility layers for abandoned R11 behavior unless an explicit task requires them.
