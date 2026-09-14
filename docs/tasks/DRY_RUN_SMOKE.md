# Dispatch Dry Run — Builder Lane Smoke Test

# Objective

Prove the GitHub -> OCI dispatch path works end to end by adding one trivial,
inert fixture file and returning a BUILD_REPORT. This task exists only to
exercise the bridge; it must not touch any scientific semantics.

# Existing interface / files

- `docs/dispatch_smoke/DRY_RUN_FIXTURE.md` (created by this task)

# Required semantics

Create the fixture file with deterministic content that states it was produced
by the dispatch dry run. Do not modify any other file.

# Allowed scope

- `docs/dispatch_smoke/DRY_RUN_FIXTURE.md`

# Do not change

- anything else in the repository
- the dispatch control plane

# Required tests

- `python3 -m unittest discover -s tests -t .`

# Done when

- the fixture file exists
- the repository test suite passes
- the final message ends with a `BUILD_REPORT` section
