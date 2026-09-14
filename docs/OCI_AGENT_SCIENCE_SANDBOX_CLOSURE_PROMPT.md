# OCI Agent Prompt — R12 Science Sandbox Boundary Closure

You are operating directly on the OCI host that supports repository:

`GY-Bai/CB16-R12`

This is a **host-operator task**, not a GitHub Actions task and not a DSH Builder task.

You are authorized to inspect and minimally modify the OCI-local sandbox wrapper/configuration needed to close the Science-lane write boundary described below. You are **not** authorized to redesign CB16, change scientific semantics, grant DSH broader host access, weaken credential masks, or turn the self-hosted runner into a host-administration channel.

Do not delegate host mutation back to GitHub Actions or DSH. The human operator intentionally keeps those execution paths sandboxed and without Full Access.

---

## 1. Current authority and evidence

Repository: `GY-Bai/CB16-R12`

Science diagnostic commit:

`90b7bac499a657a7ec7e226618c8d76c927906c6`

Science diagnostic Issue:

`#20 — [R12] Infra boundary probe R0 — Science diagnostic`

GitHub Actions run:

`34846326427`

Artifact:

`cb16-science-results-20-34846326427-1`

Artifact SHA256:

`48be404178c6f7d17f4946fa9da83e770d0565a6167a1fcb0f1a57ddcd6203ae`

The deterministic boundary probe returned:

```json
{
  "schema": "cb16.infra.boundary_probe.v1",
  "diagnostic_only": true,
  "worktree_write": "ALLOWED",
  "store_write": "ALLOWED",
  "child_env_has_github_token": false,
  "parent_or_peer_proc_exposes_github_token_key": false,
  "proc_scan_status": "OK",
  "readable_proc_environments": 1,
  "denied_proc_environments": 298,
  "proc_environ_token_key_matches": 0,
  "worktree_probe_cleaned_up": true,
  "store_probe_cleaned_up": true
}
```

Interpretation:

1. The Science sandbox currently allows writes throughout its task worktree.
2. The Science sandbox currently allows persistent writes to `/cb16/store`.
3. The Science child environment does **not** receive `CB16_GITHUB_TOKEN`.
4. The probe found **no readable parent/peer process environment exposing the `CB16_GITHUB_TOKEN=` key marker**. Most peer process environments were already denied by the current OS/sandbox permissions.

Therefore this task is about the **Science write boundary only**. Do **not** modify `/proc`, PID namespace behavior, GitHub-token handling, or process permissions unless you discover a separate concrete defect and report it instead of changing it automatically.

---

## 2. Required architectural invariant

The two lanes have different authority and must not be collapsed into one permission profile.

### Builder lane

Builder / DSH needs:

- task worktree: **writable**
- `/cb16/store`: **writable**
- frozen/raw/model inputs: read-only
- Docker socket: hidden/denied
- SSH/GitHub/Docker credential material: hidden/denied

This behavior already works. **Do not break it.**

### Science lane

Science is an exact-commit, allowlisted, frozen execution lane. Its durable inputs must be immutable during a run.

Required minimum:

- `/cb16/store`: **read-only** from Science
- `/cb16/raw`: read-only
- `/cb16/frozen`: read-only
- `/cb16/brain_assets`: read-only
- Docker socket and host credentials: still hidden exactly as today
- no new host privilege

Preferred stronger boundary, if it can be implemented cleanly without destabilizing Builder:

- Science task worktree/source: read-only
- only the designated per-run result directory is writable
- optional bounded per-run scratch may be writable if required, but it must be inside the task worktree/run namespace and must not become cross-run persistent state

Current Science result location is conceptually under:

`<science-worktree>/.cb16/results`

If making the whole Science worktree read-only requires pre-creating and separately bind-mounting that result directory writable, that is acceptable. Keep the exception as narrow as possible.

If the preferred stronger worktree boundary would require a large or fragile redesign, implement the required minimum (`/cb16/store` read-only for Science), preserve the existing post-run git immutability check, and report the stronger boundary as deferred. Do not widen permissions to make the implementation easier.

---

## 3. Existing sandbox design that must be preserved

The OCI-local wrapper is the single source of truth for sandbox profiles:

`~/.local/bin/cb16-sandbox-runner`

The repository-side dispatcher asks it for the canonical profile using an interface of the form:

```bash
cb16-sandbox-runner --print-profile <workspace>
cb16-sandbox-runner --print-profile <workspace> --read-only
```

The wrapper also verifies the DSH provider profile and fails closed if the provider drifts away from the canonical Builder profile.

Existing protections that MUST remain in both lanes include the effective masking/denial of:

- `/var/run/docker.sock`
- `/run/docker.sock`
- `~/.ssh`
- `~/.config/gh`
- `~/.docker`
- `~/.git-credentials`

Do not replace these with a weaker warning-only mechanism.

Do not grant the runner or DSH sudo/root/full-host access.

Do not remove the sandbox merely to make path parity easier.

---

## 4. Preferred implementation shape

First inspect the actual current wrapper and DSH profile on the host. Do not assume this prompt reproduces their exact contents.

Prefer the smallest change that gives the wrapper **two explicit canonical profiles**:

### Profile A — Builder / workspace-write

Semantics remain exactly as the currently working Builder profile:

```text
root filesystem       read-only baseline
task worktree          writable
/cb16/store            writable
raw/frozen/assets      read-only
credential/socket masks enforced
```

The existing no-`--read-only` invocation should continue to represent this profile.

### Profile B — Science / read-only

The existing `--read-only` interface should represent the Science profile:

```text
root filesystem       read-only baseline
/cb16/store            read-only
raw/frozen/assets      read-only
credential/socket masks enforced
science source         preferably read-only
result directory       writable exception only
```

Do not maintain two independent hand-copied profile definitions if one shared base + explicit deltas can express the difference. The wrapper should remain the single source of truth.

The Science profile must fail closed if its required writable result path cannot be established safely.

Do not silently fall back from Science read-only mode to Builder workspace-write mode.

---

## 5. Repository compatibility

Do not make arbitrary repository edits in this host task.

However, inspect the current repository dispatcher at `main` and verify whether it already requests the wrapper's `--read-only` profile for Science.

If the repository dispatcher does **not** yet request `--read-only` for Science, do **not** patch/push repository code from this host task. Instead report this exact repo-side requirement back to Chat-SOL:

> `run_science_entrypoint` / `query_sandbox_profile` must request the wrapper's read-only Science profile, while Builder continues using the workspace-write profile.

If a pre-created result directory is required by the host wrapper, report the minimal repository-side prerequisite precisely, for example:

> dispatcher must create `<science-worktree>/.cb16/results` before invoking the wrapper.

Chat-SOL will schedule that repository change through the normal Builder PR flow.

The host and repository changes must not be entangled into an unreviewable single operation.

---

## 6. Required host-side verification

Perform verification with deterministic shell/Python probes. Do not invoke DeepSeek just to test filesystem permissions.

### 6.1 Builder regression

Using the canonical Builder profile, verify all of the following:

- can create/delete a temporary file inside a throwaway Builder workspace;
- can create/delete a temporary file under a dedicated throwaway namespace in `/cb16/store`;
- cannot write to an unrelated host path outside allowed writable roots;
- Docker socket remains unusable/hidden;
- SSH/GitHub/Docker credential paths remain hidden/empty/inaccessible as designed;
- raw/frozen/brain assets remain readable but not writable.

Clean up every probe file.

### 6.2 Science boundary

Using the canonical Science/read-only profile, verify all of the following:

- write to `/cb16/store/.cb16_science_boundary_probe/...` is denied;
- raw/frozen/brain assets are readable and not writable;
- Docker socket remains unusable/hidden;
- credential paths remain hidden/empty/inaccessible;
- if the stronger source boundary is implemented, direct write to Science source/worktree is denied;
- the designated Science result directory is writable;
- a write anywhere else must not be enabled merely to make result output work.

Clean up every probe file/directory that was created outside the sandbox before the read-only test.

### 6.3 Profile inspection

Capture the NUL-separated output of both:

```bash
cb16-sandbox-runner --print-profile <throwaway-workspace>
cb16-sandbox-runner --print-profile <throwaway-workspace> --read-only
```

Convert it to a readable argv listing in the report.

Confirm explicitly which arguments are shared and which arguments differ.

The difference should correspond to authority (Builder writable vs Science frozen), not unrelated environment drift.

### 6.4 DSH compatibility

Verify the existing DSH headless profile still resolves `runnerCommand` to the wrapper and that the wrapper still accepts/verifies the provider's Builder profile.

Do not run a large model task for this check. Configuration inspection plus a minimal non-model sandbox invocation is preferred.

---

## 7. No-change areas

Do not change any of the following unless a test proves a new independent blocker, in which case stop and report it first:

- `/proc` mount policy
- PID namespace policy
- GitHub-token architecture
- runner registration
- runner labels
- repository visibility
- Docker group membership
- DSH model credentials
- CB16 market/science/trading code
- `/cb16/raw` contents
- `/cb16/frozen` contents
- `/cb16/brain_assets` contents
- existing project-store data

The R0 diagnostic specifically found no readable parent/peer GitHub-token key marker, so there is currently no evidence requiring a `/proc` or token-isolation host change.

---

## 8. Safety and rollback

Before editing the wrapper/profile:

1. make a timestamped local backup of every host file you will edit;
2. record permissions/ownership;
3. do not copy secret contents into the report;
4. keep rollback commands ready;
5. if Builder regression fails, restore the previous files immediately and report the failure rather than improvising a wider profile.

Do not delete or rewrite existing `/cb16/store` data.

Do not disable the credential masks as a debugging shortcut.

---

## 9. Final report required from OCI Agent

Return a concise report to the human operator containing:

1. exact host files inspected;
2. exact host files modified;
3. before/after semantic profile summary;
4. readable Builder profile argv;
5. readable Science profile argv;
6. Builder regression results;
7. Science boundary results;
8. Docker/credential-mask regression results;
9. whether any repository-side change is still required;
10. exact minimal repository-side requirement, if any;
11. rollback path/commands;
12. any residual risk.

Do not include secret values, private keys, OAuth tokens, API keys, or raw process environments.

A successful host task does **not** itself authorize formal R12 science. After your report is returned, Chat-SOL will review it, schedule any needed repo-side patch, and rerun the same `cb16.infra.boundary_probe.v1` diagnostic. Closure requires measured evidence, not an implementation claim.
