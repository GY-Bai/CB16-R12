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
name, so the task pins an exact revision.

```cb16
mode: build
base_sha: 0000000000000000000000000000000000000000
branch: ds/task-000
task_file: docs/tasks/EXAMPLE.md
# pr_number: 123
# review_delta: one-line summary of the requested fix
# allow_control_plane: false
```

Science lane uses a different block:

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
