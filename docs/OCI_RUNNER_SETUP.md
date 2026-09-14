# OCI Runner Setup (operator note)

This file records the one-time host setup that cannot be committed to the
repository. It contains **no secret values** — only names, paths, and the
commands needed to reproduce the state.

## Why a dedicated runner is required

GitHub repository-level self-hosted runners serve exactly one repository. The
pre-existing runner `japan-oci-01` (labels `self-hosted, Linux, ARM64, japan-oci`)
is registered to `GY-Bai/CB16-R10` and therefore cannot pick up jobs from
`GY-Bai/CB16-R12`. The dispatch workflows target the additional label `r12`,
which only a runner registered against `GY-Bai/CB16-R12` can carry.

## 1. Register the R12 runner

```bash
# on the OCI host, as the runner user
mkdir -p ~/cb16-r12-runner && cd ~/cb16-r12-runner
# runner tarball must match the host architecture (this host is Linux ARM64)
tar xzf ~/actions-runner-linux-arm64-*.tar.gz

# registration token: repository Settings -> Actions -> Runners -> New runner,
# or `gh api -X POST repos/GY-Bai/CB16-R12/actions/runners/registration-token`
./config.sh --url https://github.com/GY-Bai/CB16-R12 \
            --token <REGISTRATION_TOKEN> \
            --name japan-oci-r12 \
            --labels self-hosted,Linux,ARM64,japan-oci,r12 \
            --work _work --unattended

./svc.sh install && ./svc.sh start     # or run ./run.sh under a user service
```

Verify from GitHub:

```bash
gh api repos/GY-Bai/CB16-R12/actions/runners \
  --jq '.runners[] | "\(.name) \(.status) \([.labels[].name]|join(","))"'
```

The workflow `runs-on` list is `[self-hosted, Linux, ARM64, japan-oci, r12]`;
all five labels must be present on the registered runner.

## 1b. Runner service must carry the user PATH

The dispatcher and `dsh` live under `~/.local/bin`, and `systemd --user` starts
services with a minimal `PATH`. A runner unit without an explicit `PATH` fails
every dispatch with:

```text
classification: EXECUTION_BLOCKED
detail: [Errno 2] No such file or directory: 'dsh'
```

The unit therefore pins the path:

```ini
[Service]
Environment=PATH=/home/bgy/.local/bin:/home/bgy/.nvm/versions/node/v22.23.2/bin:/usr/local/bin:/usr/bin:/bin
```

## 1c. Actions policy must permit GitHub-owned actions

`GY-Bai/CB16-R12` was configured with `allowed_actions: local_only`, which
rejects `actions/checkout` and `actions/upload-artifact` before the job starts
(the run shows `startup_failure` with zero jobs). The policy was narrowed
rather than opened: only GitHub-authored actions are allowed, third-party and
verified marketplace actions stay blocked.

```bash
gh api -X PUT repos/GY-Bai/CB16-R12/actions/permissions \
  --input - <<<'{"enabled":true,"allowed_actions":"selected"}'
gh api -X PUT repos/GY-Bai/CB16-R12/actions/permissions/selected-actions \
  --input - <<<'{"github_owned_allowed":true,"verified_allowed":false,"patterns_allowed":[]}'
```

To revert to the stricter original policy:

```bash
gh api -X PUT repos/GY-Bai/CB16-R12/actions/permissions \
  --input - <<<'{"enabled":true,"allowed_actions":"local_only"}'
```

## 1d. Dispatcher-owned repository clone

The dispatcher keeps its own clone at `$CB16_WORK_ROOT/repo` and creates task
worktrees there. It deliberately does **not** create worktrees inside the
Actions workspace: `actions/checkout` deletes local branches to avoid conflicts
and, when a branch is checked out in a worktree, recreates the whole repository
instead — which orphans worktree registrations. Keeping task branches out of
the checkout repository removes that interaction entirely.

## 2. Repository variables

Set these as repository Actions **variables** (not secrets) — they are paths,
not credentials:

| Variable | Purpose | Example on this host |
| --- | --- | --- |
| `CB16_WORK_ROOT` | parent directory for per-task worktrees | `/home/bgy/cb16-worktrees` |
| `CB16_STATE_DIR` | dispatch lock/state directory | `/home/bgy/.cb16/state` |

If a variable is unset the dispatcher falls back to `~/cb16-worktrees` and
`~/.cb16/state`.

## 3. DSH runtime on the host

| Item | Value on this host |
| --- | --- |
| `dsh` executable | `/home/bgy/.local/bin/dsh` (on `PATH` for the runner service) |
| `DSH_HOME` | `/home/bgy/.dsh` |
| Headless profile | `~/.dsh/profiles/headless` (bundles `dsh-base`, `dsh-headless`, `dsh-anchored-subagent`) |
| Model credential | `~/.dsh/.credentials.yaml`, single key `DEEPSEEK_API_KEY` |

The Builder lane runs `dsh --profile headless "<fixed prompt>"` from the task
worktree. The prompt is a dispatcher constant and is never built from Issue
text. A fresh headless session is used for every dispatch; correctness never
depends on a previous conversation.

### Sandbox backends

The headless profile confines file writes: a probe that tried to write outside
the worktree was denied and produced no file. Confinement comes from
`bubblewrap`, which the dispatcher does not manage.

On this host `bwrap` was installed **without root** by unpacking the Oracle
EPEL RPM into the user prefix:

```bash
cd /tmp && dnf download --destdir=. --enablerepo=ol9_developer_EPEL bubblewrap
rpm2cpio bubblewrap-*.rpm | cpio -idm
mkdir -p ~/.local/bin && cp usr/bin/bwrap ~/.local/bin/bwrap
```

`~/.local/bin` is first on the runner's `PATH`, so the dispatcher and DSH both
find it. The equivalent privileged install is
`sudo dnf install -y --enablerepo=ol9_developer_EPEL bubblewrap`.

Without a usable sandbox backend DSH refuses to run unconfined, and a Builder
dispatch fails closed.

### Verified host state

Recorded after the bootstrap dry runs on this host:

| Item | Value |
| --- | --- |
| R12 runner name | `japan-oci-r12` (user service `cb16-r12-runner.service`) |
| Runner labels | `self-hosted, Linux, ARM64, japan-oci, r12` |
| Work tree root | `/home/bgy/cb16-worktrees` |
| Dispatch state/locks | `/home/bgy/.cb16/state` |
| Sandbox binary | `/home/bgy/.local/bin/bwrap` (bubblewrap 0.6.3) |
| Dispatch clone | `/home/bgy/cb16-worktrees/repo` (owned by the dispatcher) |

## 4. Credential boundary

* The workflow passes the ephemeral `secrets.GITHUB_TOKEN` to the dispatcher
  **only** for the post-DSH publish steps (commit, push, Draft PR, labels).
* Before starting DSH the dispatcher scrubs the child environment: every
  `*TOKEN*`, `*SECRET*`, `*KEY*`, `GH_*`, `GITHUB_*`, `SSH_*`, `CB16_*`
  variable is dropped; only `DEEPSEEK_API_KEY` and runtime essentials survive.
  Git's implicit credential lookups are disabled in the child
  (`GIT_TERMINAL_PROMPT=0`, `GIT_ASKPASS=/bin/true`, `GIT_CONFIG_GLOBAL=/dev/null`).
* The dispatcher reports `credential_warnings` when long-lived host credentials
  are readable by the runner user. Recommended hardening, in order of value:

  1. run the self-hosted runner under a **dedicated user** whose home holds no
     personal GitHub token and no SSH private keys;
  2. `gh auth logout` on the runner host once bootstrap is complete, so no
     long-lived OAuth token sits in `~/.config/gh/hosts.yml`;
  3. keep the DeepSeek credential as the only credential in the runner home.

> Residual risk: the DSH sandbox makes the filesystem read-only but still
> **readable**. A sandboxed agent can read any file the runner user can read,
> including credentials left in that user's home. The mitigations above are
> therefore about not having such credentials there in the first place.

## 5. Pushing workflow files

GitHub rejects pushes that create or modify `.github/workflows/**` when the
credential lacks the `workflow` scope:

```text
! [remote rejected] ... (refusing to allow an OAuth App to create or update
workflow `.github/workflows/...` without `workflow` scope)
```

The bootstrap account here authenticates with an OAuth token that has
`gist, read:org, repo` only. To land workflow changes from a workstation,
re-authorise once with the extra scope:

```bash
gh auth refresh -s workflow
```

Once the workflows exist on `main`, GitHub Actions uses its own ephemeral
`GITHUB_TOKEN`, which needs no extra scope.

## 6. Repository visibility

`docs/OCI_DSH_DISPATCH_CONTRACT.md` §8 prefers either a private development
repository or a separate private control repository while a self-hosted runner
is active. `GY-Bai/CB16-R12` is currently **public** (as is `CB16-R10`).

The implemented dispatch path does not widen that exposure:

* it triggers only on `issues: [labeled]`, never on `pull_request` or fork code;
* only accounts with label/triage permission can apply a command label, and the
  dispatcher additionally checks the label sender against `CB16_TRUSTED_ACTORS`;
* it checks out the trusted default branch and pins work to explicit SHAs;
* the failure mode for malformed metadata is always fail-closed.

Changing repository visibility is a Master/Chat-SOL decision, not an
infrastructure side effect, so it is deliberately left to that authority.
