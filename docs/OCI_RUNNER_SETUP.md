# OCI Runner Setup (operator note)

This file records the one-time host setup that cannot be committed to the
repository. It contains **no secret values** — only names, paths, and the
commands needed to reproduce the state.

## Why a dedicated runner is required

GitHub repository-level self-hosted runners serve exactly one repository. The
pre-existing R10 runner (the `japan-oci-01` registration on `GY-Bai/CB16-R10`) is
a different repository and therefore cannot pick up jobs from
`GY-Bai/CB16-R12`. The dispatch workflows target the additional label `r12`,
which only a runner registered against `GY-Bai/CB16-R12` can carry.

The R12 runner deliberately carries **no region in its name or labels**. The
repository is public, so a runner named after its location publishes that
location; `oci-cpu-r12` / `oci-cpu` describes the platform and the CPU-only
instance class without naming the region.

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
            --name oci-cpu-r12 \
            --labels self-hosted,Linux,ARM64,oci-cpu,r12 \
            --work _work --unattended

./svc.sh install && ./svc.sh start     # or run ./run.sh under a user service
```

Verify from GitHub:

```bash
gh api repos/GY-Bai/CB16-R12/actions/runners \
  --jq '.runners[] | "\(.name) \(.status) \([.labels[].name]|join(","))"'
```

The workflow `runs-on` list is `[self-hosted, Linux, ARM64, oci-cpu, r12]`;
all five labels must be present on the registered runner. Renaming a runner is
a re-registration, not an edit: stop the service, `./config.sh remove --token
<REMOVE_TOKEN>`, then `config.sh` again with the new name and labels, and
restart the service. Until the workflow carrying the new label is on the
default branch, dispatches find no matching runner and fail to start.

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

## 1e. Action versions and the runner floor

The dispatch workflows use `actions/checkout@v7` and
`actions/upload-artifact@v7`. Both run on `node24`, which imposes a **minimum
Actions runner version of 2.327.1** on self-hosted runners. The R12 runner
reports `2.337.0`, so the floor is satisfied.

Pinning v4 was a real defect: those releases run on Node 20, and the runner
already emits

```text
Node 20 is being deprecated. This workflow is running with Node 24 by default.
If you need to temporarily use Node 20, set
ACTIONS_ALLOW_USE_UNSECURE_NODE_VERSION=true
```

which means the platform forces a newer Node onto an action that was not built
for it. Check the runner version before upgrading either action:

```bash
grep -ohE "Runner version: [0-9.]+" ~/cb16-r12-runner/_diag/*.log | tail -n 1
```

Inputs used by these workflows were verified against the `v7.0.1` manifests:
checkout uses `ref`, `fetch-depth`, `persist-credentials`; upload-artifact uses
`name`, `path`, `if-no-files-found`, `retention-days`. All still exist.

Artifact uploads are immutable from v4 onwards, so both workflows include
`github.run_attempt` in the artifact name; without it a re-run of the same run
id is rejected with "an artifact with this name already exists".

## 1f. Host identifiers in a public log

Actions logs and artifacts of a **public** repository are world-readable, and an
OCI host name leaks more than it looks like. On this host:

```text
short: agent-vcn-a1-main-jp
fqdn : agent-vcn-a1-main-jp.sub05031942320.vcna1mainjp.oraclevcn.com
```

That encodes the cloud provider, the region (`vcna1mainjp`), the VCN name and a
subscription fragment. The dispatcher therefore scrubs every host identifier
from everything it publishes:

* the lane stdout/stderr logs and the test logs written as evidence;
* `BUILD_REPORT.md` and `dispatch_summary.json`;
* the Draft PR body and any Issue comment;
* its own console output, including failure details.

`CB16_REDACT_TERMS` adds extra comma-separated terms. `dispatch_summary.json`
records how many terms were active as `host_identifiers_redacted`.

What this does **not** remove: the runner workspace path, for example
`/home/bgy/cb16-r12-runner/_work/CB16-R12/CB16-R12`, is printed by
`actions/checkout` before the dispatcher runs and carries the user name. Moving
the runner under a neutral path is a host decision.

## 1g. Docker socket and the sandbox boundary

The DSH sandbox profile is `--ro-bind / /` plus a writable workspace, so it
**binds the whole root read-only, including `/var/run/docker.sock`**. Measured
from inside that exact profile:

```text
srw-rw----  /var/run/docker.sock
daemon reachable: Server=29.4.2
```

The runner user is in the `docker` group, so a sandboxed agent can reach the
daemon and mount the host root into a privileged container. **The sandbox
confines writes, not reads, and it is not a boundary against the Docker socket.**

Masking works if the profile is amended. Appending a `/dev/null` bind over a
path hides it (verified):

```text
before: docker.sock 可见 / SSH 私钥可读 / gh token 可读
after : docker.sock 已遮蔽 / SSH 私钥 0 字节 / gh token 0 字节
```

```bash
bwrap --ro-bind / / --dev /dev --proc /proc --die-with-parent --tmpfs /tmp \
      --bind "$WS" "$WS" \
      --ro-bind /dev/null /var/run/docker.sock \
      --ro-bind /dev/null "$HOME/.ssh/id_ed25519" \
      --ro-bind /dev/null "$HOME/.config/gh/hosts.yml" \
      -- /bin/sh -c 'true'
```

There is **no mask-paths option**: `LocalSandboxProvider.Config` exposes only
`runnerCommand`, `runnerFailureSignatures` and `probeTimeoutMs`, and
`bwrapProfileArgs` is hard-coded. There is however a supported hook:
`runnerCommand` is the *operator's assertion* of the runner invocation, and
`confine()` still appends the ordinary bwrap profile arguments after it, so a
wrapper can inject binds that win.

### Applied mitigation

`~/.local/bin/cb16-sandbox-runner` receives
`<bwrap profile args...> -- <command...>`, inserts read-hiding binds after the
profile arguments, and execs `bwrap`. It never runs the wrapped command itself:
if bwrap cannot start it prints `cb16-sandbox-runner: <detail>` and exits 127,
which is the configured fatal signature, so a broken wrapper fails closed.

It hides:

| Target | Bind | Why |
| --- | --- | --- |
| `/var/run/docker.sock`, `/run/docker.sock` | `--ro-bind /dev/null` | privileged-container escape to host root |
| `~/.ssh` | `--tmpfs` | private keys |
| `~/.config/gh` | `--tmpfs` | long-lived OAuth token with `repo` + `workflow` |
| `~/.docker` | `--tmpfs` | registry credentials |
| `~/.git-credentials` | `--ro-bind /dev/null` | stored git credentials |

Wired in through the profile patch layer (a supported operator layer, not a
vendored-code edit), in `~/.dsh/profiles/headless/cordis.patch.yml`:

```yaml
- id: sandbox
  config:
    runnerCommand:
      - /home/bgy/.local/bin/cb16-sandbox-runner
    runnerFailureSignatures:
      - "cb16-sandbox-runner: "
```

Verify the composition with `dsh --profile headless --dump-config`, and verify
the effect with a headless run that tries `docker ps` (measured:
`RESULT=DOCKER_DENIED`, while workspace writes still succeed and writes outside
the workspace are still refused).

### Residual, not fixed

The wrapper constrains **the agent**. It does not stop a workflow step, which
runs as `bgy` outside the sandbox and therefore still holds the `docker` group.
Removing `bgy` from that group is a root action
(`sudo gpasswd -d bgy docker`) and was not performed here. Measured: a
`systemd --user` service **cannot** drop a supplementary group via
`SupplementaryGroups=`; both with and without the directive the test process
reported `groups=1001(bgy),989(docker)`.

## 1h. Read-only data and package caches

`config/cb16_data_manifest.json` is the repository-owned list of read-only host
data advertised to lane children. Entries whose path is absent are dropped with
a note, so the manifest is harmless on another machine. Resolved entries appear
in `.cb16/TASK_PACKET.md` under "Read-only data available" and in
`dispatch_summary.json` as `read_only_data`.

The two primary inputs are the market data root and the frozen weights:

| Logical name | Path | What |
| --- | --- | --- |
| `binance_um_1m_klines_10pairs` | `/cb16/raw/klines_1m` | 10 USDM pairs, 1m klines, monthly zips with `.CHECKSUM`, 2020-01..2026-08 |
| `binance_um_funding_rates_10pairs` | `/cb16/raw/fundingRate` | funding history for the same pairs |
| `cb16_raw_download_manifest` | `/cb16/raw/DOWNLOAD_MANIFEST.json` | provenance: symbols, range, per-segment coverage |
| `frozen_weights_r10_lineage` | `/cb16/frozen` | Central Brain G0 weights, canonical nonlinear assets, operator reducers, sensory canary |

Secondary/historical entries are kept so an older task can still find them:
`e4_t1_curated_1m_5m_1h`, `binance_usdm_1m_raw_vault`, `frozen_body_archives`.

### Frozen weights

The R10/R11 frozen bodies ship as tarballs, which are not usable as an input
root, so they were extracted once into `/cb16/frozen` and the archive hashes
recorded in `/cb16/frozen/SOURCE_ARCHIVE_SHA256SUMS.txt`:

```bash
tar xzf CB16_R10_G0_BOOTSTRAP_RETURN_R0.tar.gz
tar xzf CB16_SHANXI_FROZEN_BODY_G0_BRAIN_R10_1_THIN_V1.tar.gz
tar xzf CB16_SHANXI_R10_2_REAL_HISTORICAL_G0_LEARNING_V1.tar.gz
sha256sum ... > /cb16/frozen/SOURCE_ARCHIVE_SHA256SUMS.txt
```

Weights present: `central_brain_g0.pt`, `central_brain_g0_r10_parent.pt`,
`CANONICAL_NONLINEAR48_SEED24680_PORTABLE.npz`, `operator_reducers_v1{,_PORTABLE}.npz`,
`R10_FROZEN_SENSORY_CANARY_V1.npz`, `SIMULATOR_SNAPSHOT_PORTABLE.npz`.

These are R10/R11-lineage artefacts imported explicitly as inputs. Whether a
given weight becomes canonical for R12 is a Master/Chat-SOL decision, not an
infrastructure one.

Because the profile currently binds `/` read-only, these paths are already
reachable. The manifest exists so the agent knows *where* they are and so a
future tightened profile has an explicit list to bind `--ro-bind`.

Package state is split: downloads are shared across branches, environments are
not. See section 1k for the shared cache; the worktree-local pieces remain:

| Variable | Value |
| --- | --- |
| `UV_CACHE_DIR` | `/cb16/cache/uv` (shared, overridable with `CB16_UV_CACHE_DIR`) |
| `PIP_CACHE_DIR` | `<worktree>/.pip-cache` |
| `UV_PROJECT_ENVIRONMENT` | `<worktree>/.venv` |
| `UV_LINK_MODE` | `copy` |

The worktree-local directories are git-ignored, so a fix cycle reuses them
without committing anything.

## 1i. Canonical path set and sandbox/runner parity

Every path advertised to a lane child must be a **real host path**, never a
sandbox-only mount. Code written in the sandboxed Builder lane is later executed
by the unconfined Science lane and, often enough, by an operator in an ordinary
shell; a path that exists in only one of those contexts is a latent break.

Canonical paths, all host-level:

| Path | Kind | Purpose |
| --- | --- | --- |
| `/cb16/raw/klines_1m` | read-only | 10 USDM pairs, 1m klines, 2020-01..2026-08 |
| `/cb16/raw/fundingRate` | read-only | funding history, verified 1:1 with the klines |
| `/cb16/frozen` | read-only | R10/R11 frozen weights and assets |
| `/cb16/brain_assets` | read-only (symlink) | Kronos / TimesFM operator and medium weights |
| `/cb16/store` | **writable, persistent** | project-level storage for checkpoints and data intermediates |

`/cb16/brain_assets` is a symlink to the real tree
(`/home/<user>/cb16_brain_assets`), so the canonical name resolves identically
inside the sandbox, in a bare runner step and in an interactive shell. Refer to
the canonical name, never to the underlying home path.

### Project store

`CB16_STORE` (default `/cb16/store`, override with `CB16_PROJECT_STORE`) is the
only writable location outside the task worktree. The dispatcher creates it if
absent and exports it to both lanes, and the task packet documents it. Use
subdirectories named after the task or issue so concurrent tasks do not collide.
The sandbox reaches it because `cb16-sandbox-runner` appends
`--bind "$CB16_STORE" "$CB16_STORE"` — a bind onto the real directory, not a
private mount, which is what keeps the path identical outside the sandbox.

Both lanes receive the same `CB16_STORE` and `CB16_DATA_MANIFEST` values; a test
asserts that, so the two lanes cannot drift apart silently.

### Both lanes are sandboxed

The Science lane originally ran its entrypoint with a bare `subprocess.run`, so
it shared the Builder lane's paths but none of its confinement. It now runs
through the same `cb16-sandbox-runner` wrapper with the same workspace-write
profile, which means:

- the same writable roots (the task worktree plus `CB16_STORE`) and the same
  masked paths (`docker.sock`, `~/.ssh`, `~/.config/gh`, `~/.docker`,
  `~/.git-credentials`);
- the result package is written inside the worktree, the only place the sandbox
  permits writes, and the dispatcher copies it out for artifact upload. The
  produced-artifact check reads the published copy, so a lane cannot satisfy it
  by writing somewhere it should not;
- `run_science_entrypoint` records `sandboxed` / `unsandboxed` in its run note
  and the dispatch summary records `science_sandbox`.

`CB16_SCIENCE_SANDBOX` selects the policy: `require` (set by the Science
workflow) fails closed when the wrapper is missing, `auto` degrades with a
recorded note, `off` disables wrapping. `CB16_SANDBOX_RUNNER` names the wrapper
explicitly when it is not on `PATH`.

Because the wrapper is a host file, a new host needs it installed before Science
dispatches can run under `require`.

### One profile, verified, not mirrored

Mirroring the provider's `bwrapProfileArgs` in the dispatcher would have left the
two lanes free to drift apart whenever DSH changed that profile. Instead the
wrapper **owns the single definition**:

```bash
cb16-sandbox-runner --print-profile <workspace> [--read-only]   # NUL separated
```

The dispatcher asks for the profile rather than carrying a copy, and the wrapper
verifies whatever the DSH provider sends against the same definition before it
execs bwrap. An unrecognised or mismatching profile fails closed with

```text
cb16-sandbox-runner: sandbox profile does not match the canonical profile
  expected: --ro-bind / / --dev /dev --proc /proc --die-with-parent --tmpfs /tmp --bind <ws> <ws>
  received: ... --unshare-net
```

so a version change shows up as a loud failure rather than as two lanes that
quietly differ. Measured after the change: a headless DSH run still starts and
writes its file (the provider's profile matches), and the Science lane reports
`science_sandbox` pointing at the wrapper.

## 1j. Which model a Builder run used

A task can be dispatched twice with a different model or reasoning effort behind
it and nothing on GitHub would show it. The dispatcher therefore reads the DSH
settings file and records the value in the dispatch summary:

```json
"builder_model": {
  "provider": "deepseek-official",
  "model": "deepseek-flash",
  "reasoningEffort": "max",
  "source": "<dsh home>/settings.yaml",
  "declared": true
}
```

It appears in `dispatch_summary.json` (uploaded as evidence) and, because the
dispatcher prints its summary, in the Actions log too.

`declared: true` is deliberate: this is the configuration the lane is launched
with, read from the same file the session header reads. The dispatcher does not
claim to have measured what the API served - that would mean parsing the session
transcript, which is not worth the fragility. The source path travels with the
value so a reviewer can check it, and a missing or unreadable settings file is
reported in a `note` instead of failing the dispatch.

The value comes from `agent-default-model` in `$DSH_HOME/settings.yaml`
(`~/.dsh/settings.yaml` by default), so changing that one file changes every
lane and every task. `dsh --profile headless --dump-config` is **not** a valid
source here: it shows the plugin default, not the runtime override.

## 1k. Builder session affinity and the shared uv cache

### Opt-in session affinity

A task branch may opt into conversation continuity with one optional Builder
metadata field:

```text
session_affinity: branch-v1
```

* absent -> the legacy path, unchanged: `dsh --profile headless ...`, a fresh
  session every dispatch;
* present -> the dispatcher routes the branch through `cb16-builder-session`
  and records `branch -> session_id` in
  `<CB16_STATE_DIR>/session-affinity/<hash>.json`.

Git stays authoritative. A resumed turn's prompt re-asserts the worktree, task
packet and review delta, and a missing, corrupt, foreign-worktree or
incompatible session degrades to `fallback-new`: a fresh session in the same
worktree, with `generation + 1` recorded.

**Legacy protection.** A branch that already existed on the remote and was never
admitted is *never* adopted into a session - the dispatcher records
`declined-existing-branch` and keeps the legacy behaviour, so editing an old
Issue's metadata cannot turn it stateful. Only a genuinely new branch
initialises affinity.

**Registry loss is recoverable.** Two files are kept per branch: the registry
(`<hash>.json`, rewritten every dispatch) and a durable admission marker
(`<hash>.admitted.json`, written once). A corrupted or deleted registry on an
admitted branch therefore yields `fallback-new` with
`resume_failure_class=registry-lost` and `generation + 1`, rather than being
mistaken for a never-admitted branch and demoted to the legacy path forever.
The marker is written before the turn runs, so a failed turn after the push
cannot cost the branch its affinity.

The dispatcher also records, per run: `session_affinity`, `session_profile`,
and a `session_route` block (`action`, `session_id`, `generation`,
`resume_attempted`, `resume_succeeded`, `resume_failure_class`, `persisted`).
The profile supports no "resume latest session" form: routing is always the
exact recorded id.

The profile itself, its command-line contract, resume validation, compaction
policy and host bootstrap live in
[`infra/dsh/cb16-builder-session/README.md`](../infra/dsh/cb16-builder-session/README.md).

### Shared uv download cache

Downloads are shared across branches; environments are not:

```text
/cb16/cache/uv          shared package cache   (CB16_UV_CACHE_DIR)
<worktree>/.venv        branch-specific environment (UV_PROJECT_ENVIRONMENT)
```

`/cb16/cache/uv` is outside the worktree, so `cb16-sandbox-runner` binds it
writable **for the Builder profile only**. Science keeps the frozen read-only
policy and is refused that write - verified on the host:

```text
Builder : cache writable YES
Science : write refused
```

The dispatcher creates the directory and never edits cache contents; uv's cache
is built for concurrent readers and writers. Each worktree keeps its own `.venv`
so branches cannot poison each other's environment.

### Context management for long-lived sessions

A resumed session accumulates context that a one-shot dispatch never did, so the
session profile pins `compaction-basic` to `thresholdRatio 0.75`,
`retainRatio 0.10` and `maxOverflowRetries 2`; the legacy profile keeps the DSH
defaults. See the plugin README for the arithmetic and for why the retry budget,
not the threshold, is what catches a boundary rejection.

## 1l. The PR review timeline in a fix round

A reviewer's instructions arrive as PR **review/comment text**, which never
enters Git, and the confined Builder agent has no GitHub credentials (`gh` is
masked and the child environment carries no token). A `mode: fix` dispatch
therefore used to hand the agent little more than `review_delta` - a terse label
such as `exact_commit_implementation_test_gate` - leaving it to guess which
instruction was newest.

For any Builder dispatch that carries `pr_number`, the dispatcher now fetches,
with its own token:

| Source | Endpoint |
| --- | --- |
| commits | `GET /repos/{repo}/pulls/{n}/commits` |
| reviews | `GET /repos/{repo}/pulls/{n}/reviews` |
| issue comments | `GET /repos/{repo}/issues/{n}/comments` |
| inline review comments | `GET /repos/{repo}/pulls/{n}/comments` |

and merges them with the commit history onto **one clock**, sorted by
timestamp, in the task packet:

```text
2026-09-14 17:02:17  COMMIT  2e13afc1  CB16 builder: issue #35 (build)
2026-09-14 17:07:50  REVIEW CHANGES_REQUESTED  GY-Bai
    One semantic blocker before merge. ...
2026-09-14 17:29:28  COMMIT  0079955d  CB16 builder: issue #35 (fix)  <- latest Builder commit
```

The section then states explicitly which instructions were posted **after** the
latest Builder commit - those are the unaddressed ones - and repeats the newest
verbatim under `### Newest instruction`, so "which instruction is current" is
never inferred from prose.

Properties worth keeping:

* **trusted actors only** - a review or comment from outside the trusted list is
  not an instruction, so a bystander cannot inject task scope;
* **best effort** - a GitHub outage records the failing sources in the packet
  and never blocks the dispatch;
* **bounded** - the newest 40 entries, bodies truncated, so a long review thread
  cannot blow up the packet;
* **redacted** - bodies pass through the same host-identifier scrubbing as the
  rest of the evidence.

## 1l. Rounds hand off through the packet, not through a session

Session affinity is **off** (`SESSION_AFFINITY_ENABLED`). A resumed session
carries every earlier round's packet, and each of those contains its own
"newest instruction" block, so a later round has to be told in prose which of
several contradictory blocks is current. A cold round has no such ambiguity,
and on the measured VS-C pair resuming also cost about $0.015 more per round:
prompt caching makes rebuilding context nearly free ($0.003/M) while every
resumed step carries the whole conversation.

`session_affinity: branch-v1` still parses, and a round that declares it is
recorded as `declined-affinity-disabled` and runs on the legacy fresh path. Set
`CB16_SESSION_AFFINITY=enabled` to restore the old behaviour.

Information now travels two ways:

| Carrier | Content |
| --- | --- |
| the packet | contract, merged PR timeline, the newest unaddressed reviewer instruction, the instruction ranking |
| `BUILD_REPORT` | what the round changed, tested, and deliberately did not do |

Each Builder round writes its report to
`<CB16_STATE_DIR>/reports/<hash>.md`, and the next round reads it into a
`## Previous round report` section. A fix-round prompt states the ranking,
notes that nothing from an earlier round is remembered, and points at that
section when it exists.

**The report is only as good as what the agent writes.** The dispatcher prefers
the agent's own `BUILD_REPORT` section and only falls back to a synthesized
stub - classification, changed files, test exit code - when the agent did not
emit one. Observed on the real VS-C rounds: two of three turns produced a
proper section (5,180 and 8,480 characters), one did not. The synthesized stub
is not a substitute for the agent's reasoning, so a round that ends without
`BUILD_REPORT` degrades the next round's context. Every dispatch records
`build_report_source: agent | synthesized` in `dispatch_summary.json`, so the
degradation is visible instead of silent.

Two defects fixed while wiring this up:

* the session runner's stdout is a JSON record, whose newlines are escaped, so
  a line-anchored `BUILD_REPORT` scan could never match and **every** agent
  report on the session path was silently replaced by the stub;
* the synthesized report used the branch name in the `Issue #` field.

## 1m. Which instruction source wins

Five places can look like an instruction, and a fix round can hold several at
once. The task packet therefore opens with an explicit ranking so the agent
never has to infer which one is current:

| Rank | Source | Authority |
| --- | --- | --- |
| 1 | the `cb16` trusted metadata block | machine-validated envelope (`mode`, `base_sha`, `branch`, `task_file`, `pr_number`, `review_delta`, `session_affinity`); decides what may run at all |
| 2 | the task contract file at `base_sha` | the task itself - scope, required tests, done-when; version-controlled and pinned |
| 3 | the newest unaddressed reviewer instruction | what the current fix round must change; narrows scope, never overrides 1-2 |
| 4 | the Issue description | the operator's framing; context only, never a contract |
| - | anything from an untrusted actor | not an instruction at all |

If sources 1-3 disagree the packet tells the agent to stop short of the
conflicting change and report the conflict in `BUILD_REPORT` rather than guess.

The same ranking is repeated in the prompt of every resumed session. A resumed
turn carries the whole earlier conversation, including an earlier task packet,
an earlier review delta and an earlier reviewer comment - all of which are
superseded. The prompt therefore states that history is not the current
instruction, points at the packet on disk as the current one, and repeats the
four ranks, so the agent cannot mistake remembered state for present state.

The Issue description is the prose outside the metadata fence. It used to be
dropped in silence, which cost real instructions - one Issue carried 1,712
characters of change request outside the fence and the agent never saw a word
of it. It is now included, truncated at 8,000 characters with a marker, and
**only when the Issue author is a trusted actor**: an untrusted author's prose
is omitted with a visible note instead of being obeyed.

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
| R12 runner name | `oci-cpu-r12` (user service `cb16-r12-runner.service`) |
| Runner labels | `self-hosted, Linux, ARM64, oci-cpu, r12` |
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
