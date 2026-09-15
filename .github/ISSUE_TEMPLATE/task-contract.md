---
name: R12 Task Contract
about: Bounded implementation task for an AI Builder
title: "[R12] "
labels: []
assignees: []
---

## Trusted dispatch metadata

Edit the values below, then apply the `ds:run` (Builder) or `science:run`
(Science) label to schedule the task. This block is the only machine-read part
of the Issue: the dispatcher validates it and refuses anything malformed, and
nothing outside this block is ever executed. Use the commit SHA, not a branch
name, so the task pins an exact revision. Create the bounded task file under `docs/tasks/active/` from `docs/templates/EXAMPLE.md`; templates are never executable authority.

```cb16
mode: build
base_sha: 0000000000000000000000000000000000000000
branch: ds/task-000
task_file: docs/tasks/active/TASK.md
# pr_number: 123
# review_delta: one-line summary of the requested fix
# allow_control_plane: false
# session_affinity: branch-v1
```

`session_affinity: branch-v1` is optional and Builder-only. It opts a **new**
task branch into a resumable DSH session so later dispatches continue the same
conversation instead of starting fresh. Git stays authoritative: the resumed
turn re-reads `git status`/`diff` and the task packet, and a missing or
mismatched session silently falls back to a fresh one.

Leave it out for the legacy behaviour (a fresh session every dispatch). It is
ignored for a branch that already existed on the remote, and it must not appear
in Science metadata.

Science lane uses a different block. New formal science should first pass Stage-1 protocol review using the canonical rules in `docs/authority/` and may use `docs/templates/SCIENTIFIC_EXPERIMENT_PROTOCOL.md` as a non-authoritative drafting scaffold. The trusted dispatch block remains:

```text
mode: science
commit_sha: <exact commit>
experiment_spec: <repo-relative spec path>
result_command: <allowlisted entrypoint identifier>
```

---

# Objective

# Existing interface / files

# Required semantics

# Allowed scope

# Do not change

# Required tests

# Done when
