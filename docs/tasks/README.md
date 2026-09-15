# Active Task Area

Authority class: **CURRENT_TASK_SCOPE ONLY**.

Only currently authorized bounded task contracts belong under `active/`.

A task file controls its own implementation scope but does not override `../authority/`. When the task is complete, its scientifically/provenance-relevant frozen contract moves to `../experiments/frozen/`; disposable one-time task instructions should be removed from the current tree rather than accumulated here.

Do not store templates, completed experiments, archaeology, or global rules in this directory.