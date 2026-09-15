# Example Task Contract

Authority class: `TEMPLATE_ONLY` — copy into `docs/tasks/active/`; do not dispatch this file itself.

> Copy this file when opening a new bounded Builder task. Replace every section.
> The matching GitHub Issue carries the trusted metadata block; this file is
> what the Builder reads for scope and acceptance. Formal scientific protocol
> design uses `SCIENTIFIC_EXPERIMENT_PROTOCOL.md` instead of expanding this task template.

# Objective

Describe the single bounded change, in one or two sentences.

# Existing interface / files

List the files and interfaces the task may touch.

# Required semantics

State the exact required behaviour. If a rule is high-risk (economics,
accounting, terminal/truncation behaviour, learning targets, masks), specify
it literally rather than delegating invention.

# Allowed scope

- files the Builder may modify

# Do not change

- authoritative semantics, thresholds, interfaces outside the listed scope
- anything under the dispatch control plane (`.github/`, `scripts/cb16_dispatch.py`,
  `config/cb16_science_allowlist.json`, `docs/operations/current/OCI_DSH_DISPATCH_CONTRACT.md`)
  unless the Issue metadata sets `allow_control_plane: true`

# Required tests

- the exact test command(s) the Builder must run

# Done when

- the observable conditions that make this task reviewable
