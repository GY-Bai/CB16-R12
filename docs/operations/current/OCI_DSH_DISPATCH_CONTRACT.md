# OCI DSH Dispatch Contract

Status: **V1 OPERATIONAL DESIGN — READY TO IMPLEMENT**

Authority class: `CURRENT_OPERATIONAL`; scientific claim interpretation remains subordinate to `docs/authority/`.

Purpose: define the smallest trusted path by which Chat-SOL can schedule bounded Builder work and formal scientific runs through GitHub into the OCI execution host.

## 1. Control-plane rule

GitHub is the only remote control plane.

```text
Master / Chat-SOL
    -> GitHub Issue task contract
    -> trusted label transition
    -> GitHub Actions
    -> OCI self-hosted runner
    -> local dispatcher
        -> DSH Builder lane OR immutable Science lane
    -> PR / Actions logs / artifacts
    -> Chat-SOL review
```

The GitHub workflow is transport and validation. It is not a second agent and must not contain scientific reasoning.

## 2. Two execution lanes

### Builder lane — `ds:run`

The Builder lane may mutate one task branch/worktree.

Use it for implementation, targeted tests, instrumentation, and requested fixes.

OCI invokes DSH non-interactively from the task workspace using a fresh headless session. DSH receives the task contract and repository context, edits only the authorized workspace, and runs local tests.

DSH must not receive GitHub write credentials.

### Science lane — `science:run`

The Science lane does not invoke DSH and does not edit code.

It checks out an exact commit, loads an already frozen experiment spec, runs the declared command, and returns results/artifacts. This lane exists so formal qualification cannot be adaptively repaired by an agent after results are visible.

## 3. GitHub command protocol

A task is scheduled by adding a command label to a trusted Issue:

- `ds:run` — execute or continue the Builder task;
- `science:run` — execute a frozen scientific run;
- `sol:review` — execution finished and evidence is ready for Chat-SOL;
- `blocked` — execution cannot proceed without an external fix.

Adding the command label is the trigger. Removing a label is not a command.

The workflow should trigger on `issues: [labeled]` and run only when the newly added label is one of the command labels.

## 4. Trusted task packet

The dispatcher must not treat arbitrary shell text from an Issue as a command.

The Issue provides a small structured control block plus the human-readable task contract. Minimum Builder metadata:

```text
mode: build | fix
base_sha: <exact commit>
branch: <task branch>
task_file: <repo-relative task contract path>
pr_number: <optional existing PR>
```

Minimum Science metadata:

```text
mode: science
commit_sha: <exact commit>
experiment_spec: <repo-relative spec path>
result_command: <allowlisted repository entrypoint identifier>
```

The dispatcher validates metadata and then writes a local task packet file. Issue text must never be interpolated directly into a shell command.

## 5. OCI dispatcher responsibilities

The OCI dispatcher is deterministic glue, not an LLM.

For a Builder run it must:

1. verify repository, Issue, trigger label, trusted task metadata, and exact base commit;
2. create or reuse exactly one task branch/worktree;
3. checkout with GitHub credentials disabled inside the Builder workspace;
4. materialize a local task packet containing the contract and relevant review delta;
5. invoke DSH in the task workspace;
6. capture exit code, final DSH text, declared test outputs, and a concise BUILD_REPORT;
7. reject forbidden/control-plane file changes unless explicitly authorized by the task;
8. after DSH exits, use a separate credentialed dispatcher step to commit/push the task branch and create/update a Draft PR;
9. publish compact logs/artifacts to the Actions run;
10. move task state to `sol:review` or `blocked`.

For a Science run it must:

1. checkout the exact declared commit in a clean workspace;
2. verify the frozen experiment spec exists and is unchanged;
3. run only the declared allowlisted experiment entrypoint;
4. never invoke DSH or modify source/config after execution begins;
5. upload `RESULT.json`, `REPORT.md`, the experiment spec, and bounded logs;
6. mark the Issue `sol:review` when results are available.

## 6. DSH invocation

Use DSH headless for unattended CI-style Builder execution:

```text
dsh --profile headless "Read the local CB16 task packet and execute it exactly. Do not redesign authority. End with BUILD_REPORT."
```

Run from the task worktree so the invoking directory is the workspace root.

Each dispatch should use a fresh DSH headless session. A fix run starts a new session against the existing task branch plus the new review delta; correctness must not depend on hidden conversational state from a previous agent session.

Persist DSH session logs locally for diagnosis, but upload them only when needed. The normal review surface is PR diff + BUILD_REPORT + targeted tests + affected-suite result.

## 7. Credential and filesystem boundary

During the DSH step:

- provide only the DeepSeek credential required by the model runtime;
- do not expose GitHub write tokens, OCI cloud credentials, SSH private keys, host Docker socket, or unrelated service secrets;
- use `actions/checkout` with persisted Git credentials disabled;
- do not mount unrelated host directories into the Builder workspace;
- keep shared data/model weights read-only when possible.

Git push, PR creation, label changes, and artifact publication happen after DSH exits in fixed dispatcher steps using minimum GitHub permissions.

## 8. Public-repository boundary

A self-hosted runner must never execute untrusted pull-request code.

No OCI self-hosted workflow may be triggered by arbitrary `pull_request` or fork code. Builder/science dispatch originates only from the trusted Issue-label control path on the default branch and checks out explicitly trusted refs/commits.

Preferred deployment while active self-hosted development is either:

- keep the development repository private; or
- keep the source repository public but place the self-hosted dispatch workflow in a separate private control repository.

Do not rely on Docker alone as the trust boundary.

## 9. Review/fix loop

```text
Issue + task contract
   -> add ds:run
   -> OCI Builder run
   -> Draft PR + evidence
   -> sol:review
   -> Chat-SOL review
       -> APPROVE
       or
       -> update Issue review delta
       -> re-add ds:run
       -> same branch / same PR
```

A review correction does not create a new scientific task identity unless the scientific question/method itself changes.

## 10. Concurrency

Start with one active OCI Builder job.

The workflow should use a repository/task concurrency key so the same Issue/branch cannot run twice concurrently. Add a second Builder only after two genuinely independent tasks exist and the single-runner lane is a measured bottleneck.

## 11. Evidence returned to GitHub

Normal Builder evidence:

```text
PR diff
BUILD_REPORT
DSH exit status
requested targeted-test result
affected-suite result
bounded runtime/error log
```

Formal Science evidence:

```text
exact commit SHA
experiment_spec.json
RESULT.json
REPORT.md
bounded execution log
```

Large caches, datasets, model weights, and ordinary intermediate files stay on OCI and are not copied to GitHub.

## 12. Execution and qualification status separation

- DSH/code/test failure with a functioning execution path -> Builder report and PR remains reviewable.
- dispatcher/runner failure -> `EXECUTION_BLOCKED`.
- insufficient machine capacity -> `HARDWARE_LIMIT`.
- formal experiment executes validly but misses its frozen gate -> dispatcher-compatible `SCIENTIFIC_FAIL`.
- task/spec conflicts with canonical authority -> `CONTRACT_MISMATCH`.

The dispatcher owns transport/execution classification. It does **not** infer scientific root cause, falsification, architecture feasibility, or promotion beyond what the experiment/result declares. A red GitHub Actions conclusion caused by a scientific gate miss is not evidence that infra failed.

Where the result schema supports it, preserve separate `execution_status`, `validity_status`, `gate_results`, `claim_assessments`, and `promotion_decision`. Older result schemas remain reviewable; Chat-SOL applies `docs/authority/R12_EVIDENCE_SCOPE_AND_CLAIM_AUTHORITY.md` when interpreting them.

Infrastructure must not silently change the scientific method to convert a gate miss into a pass. Scientific reporting must likewise not widen PASS or FAIL beyond the tested claim scope.
