# R12 Infra Boundary Probe R0

Status: AUTHORIZED REPO-ONLY DIAGNOSTIC TASK

## Objective

Add one deterministic, non-scientific diagnostic entrypoint that measures the two unresolved execution-boundary questions before any host-side change is requested.

This task MUST NOT modify the OCI host, the Actions runner installation, `cb16-sandbox-runner`, systemd, DSH profiles, Docker configuration, user/group membership, `/proc` mount options, or any file outside the task worktree.

The output is evidence for Chat-SOL and the human operator. It must not self-apply a host fix.

## Required probes

Implement a repository-owned diagnostic entrypoint, allowlisted under a clearly diagnostic-only identifier such as `cb16.boundary-probe@v1`, that runs through the existing Science lane and produces `RESULT.json` and `REPORT.md`.

The probe must measure only the following facts.

### 1. Worktree write boundary

Attempt to create a uniquely named temporary probe file directly under the Science worktree, outside the designated result directory.

Record only whether the write was blocked or allowed. If creation succeeds, remove the file before exit.

Do not modify tracked repository content.

### 2. Project-store write boundary

If `CB16_STORE` is present, attempt to create a uniquely namespaced tiny temporary probe file under a dedicated path such as `$CB16_STORE/.cb16_boundary_probe/`.

Record only whether the write was blocked or allowed. If creation succeeds, remove the file and remove the probe directory if empty.

Do not inspect or modify any pre-existing store content.

If `CB16_STORE` is absent, report `NOT_PRESENT`; do not treat absence as a failure.

### 3. Parent-process credential visibility through `/proc`

The dispatcher parent process receives `CB16_GITHUB_TOKEN`, while the Science child environment is scrubbed.

The diagnostic process must first assert that its own `os.environ` does NOT contain `CB16_GITHUB_TOKEN`.

Then scan readable numeric `/proc/<pid>/environ` files for the literal key marker `CB16_GITHUB_TOKEN=`. The probe MUST NOT print, persist, hash, copy, or otherwise expose any credential value or any unrelated environment content.

Record only:

- `child_env_has_github_token: true|false`
- `parent_or_peer_proc_exposes_github_token_key: true|false`
- optionally a count of readable process environments and a count of matches

Do not emit matching PIDs unless needed for debugging; never emit environment values.

If `/proc` access is denied, report that fact explicitly rather than treating it as a match.

## Result schema

`RESULT.json` must be compact and deterministic in structure. At minimum include:

```json
{
  "schema": "cb16.infra.boundary_probe.v1",
  "diagnostic_only": true,
  "worktree_write": "BLOCKED|ALLOWED|ERROR",
  "store_write": "BLOCKED|ALLOWED|NOT_PRESENT|ERROR",
  "child_env_has_github_token": false,
  "parent_or_peer_proc_exposes_github_token_key": false,
  "proc_scan_status": "OK|DENIED|ERROR"
}
```

`REPORT.md` must explain the results without claiming that any host-side fix was applied.

## Fail-closed requirements

- The diagnostic must never print secret values.
- It must never invoke DSH.
- It must never use shell interpolation for `/proc` scanning.
- It must never follow non-numeric `/proc` entries.
- It must bound the number/size of files read and tolerate permission errors.
- Temporary probe files must be cleaned up on every path where creation succeeded.
- No network access is required.

## Repository changes allowed

- `science/cb16_science/boundary_probe.py`
- `config/cb16_science_allowlist.json`
- `tests/` for deterministic unit tests of the probe
- minimal package plumbing if strictly required

Do not modify workflows, dispatcher behavior, host setup documentation, scientific architecture documents, data manifests, or any scientific code in this task.

`config/cb16_science_allowlist.json` is part of the dispatch control plane, so this task is explicitly authorized to modify that one control-plane file only.

## Required tests

At minimum cover:

1. own child environment containing no GitHub token;
2. `/proc` scanner returns only booleans/counts and never values;
3. readable synthetic process-environment fixtures containing the key are detected without exposing values;
4. permission-denied `/proc` entries are tolerated;
5. worktree probe cleans up after an allowed write;
6. store probe cleans up after an allowed write;
7. missing store reports `NOT_PRESENT`;
8. output schema is stable and marks `diagnostic_only: true`;
9. existing repository suite remains green.

## Done when

- one Draft PR contains only the allowed repository changes;
- tests pass;
- the PR explains that no OCI host mutation was attempted;
- no Science run is triggered by the Builder task itself.

After Chat-SOL reviews and merges the PR, Chat-SOL will schedule the allowlisted Science diagnostic separately and use its result to write the OCI Agent handoff prompt if host changes are actually required.
