# Frozen Historical Experiment Contracts

Authority class: **FROZEN_HISTORICAL_AUTHORITY**.

`frozen/` preserves task and experiment contracts used by completed historical R12 work. They remain authoritative for the original method, data scope, controls, budgets, gates, and run identity they specified.

They are **not current global governance** and must not be loaded by default when deciding a new task.

Important: status text inside these files reflects the historical moment when the contract was authored (for example `PREREGISTERED` or `READY TO BUILD`). It is preserved intentionally and is **not a statement that the task is still pending today**.

Current interpretation and promotion scope come from `../authority/`. Do not edit a frozen contract to insert later interpretations; add or change current authority instead.

`FROZEN_INDEX.json` records the historical source path, snapshot commit, and SHA256 for each frozen contract. Tests enforce those hashes so later documentation work cannot silently rewrite history.