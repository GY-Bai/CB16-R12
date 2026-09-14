# cb16-builder-session

A CB16-owned DSH profile bundle that runs **one Builder turn in an explicit,
resumable session**.

It exists so a task branch can opt into conversation continuity without ever
letting that conversation become authority.

```text
Git branch / worktree / task contract / PR   =  authoritative state
DSH session                                  =  optional cognitive cache
```

If a session disappears, is corrupt, was written in another worktree, or became
unreadable after a DSH upgrade, the turn falls back to a fresh session in the
same worktree and continues from Git. Correctness never depends on the session.

## Relationship to the legacy path

| | legacy (`headless`) | this profile |
| --- | --- | --- |
| session | new every dispatch | explicit: `--new` or `--session <id>` |
| metadata field | absent | `session_affinity: branch-v1` |
| compaction policy | DSH defaults (0.80 / 0.16 / retries 1) | `0.75 / 0.10 / retries 2` |
| sandbox | same `cb16-sandbox-runner` profile | same |
| tools, persona, model | same | same |

The legacy profile is **not modified**. `--dump-config` for the two profiles
differs only in the startup/runner rows, the compaction policy and the bundle
name comments; the sandbox row, tool rows and system prompt are identical.

## Command-line contract

```bash
# first turn on a new branch
dsh --profile cb16-builder-session --new --output-format json "<task>"

# later turn, exact session id from the dispatcher registry
dsh --profile cb16-builder-session --session <session-id> --output-format json "<task>"
```

`--new` and `--session` are mutually exclusive and one of them is required. The
profile deliberately offers **no** "resume the latest session" form: routing must
be an exact branch -> session mapping, never whichever conversation was written
last.

### Machine-readable result

```json
{
  "success": true,
  "session_id": "session-<uuid>",
  "session_action": "new | resume | fallback-new",
  "resume_attempted": false,
  "resume_succeeded": false,
  "resume_failure_class": null,
  "provider": "deepseek-official",
  "model": "deepseek-flash",
  "reasoning_effort": "max",
  "text": "<final assistant message>",
  "turn_outcome": "completed",
  "duration_ms": 1234
}
```

`resume_failure_class` is one of `session-not-found`, `cwd-mismatch`,
`persistence-unavailable`, `inspect-failed`, `resume-rejected`. A
`fallback-new` action means the turn continued correctly with a fresh session;
it is not a task failure.

## Registry loss and the admission marker

The dispatcher keeps two files per branch under
`<CB16_STATE_DIR>/session-affinity/`:

| File | Holds | Lifetime |
| --- | --- | --- |
| `<hash>.json` | the active session id and generation | rewritten every dispatch |
| `<hash>.admitted.json` | the durable statement "this branch was legitimately admitted" | written once, updated in step |

The admission marker exists because "did this branch already exist on the
remote?" stops being a usable test the moment the first dispatch pushes the
branch. Without it, a registry that is later corrupted would look exactly like a
pre-existing branch that was never admitted, and the branch would be declined
into the legacy path forever - the opposite of the promised
"registry problem -> fallback-new -> generation + 1".

Routing is therefore:

```text
registry readable          -> resume that exact id
admission marker present   -> fallback-new (registry-lost), generation + 1
branch existed remotely    -> declined-existing-branch  (true legacy protection)
otherwise                  -> new, and record the admission marker
```

The admission marker is written at decision time, before the turn runs, so a
turn that fails after the branch is pushed cannot cost the branch its affinity.

## Resume validation

Before continuing, the runner reads the persisted session through the official
non-committing `sessionPersistence.inspect()` and requires:

- the session exists and its validated header can be opened;
- `header.cwd` equals the current worktree.

A session created in another directory is a `cwd-mismatch`, never a
continuation. The registry file alone is never trusted: registry and persisted
session must agree.

## Session metadata

Official headless semantics are preserved: `meta: { cwd: process.cwd() }` and
nothing else. Branch identity lives in the dispatcher registry, **not** in
`agentPreset`; `agentPreset` remains durable semantic composition metadata and is
never used as a session tag.

## Context management

A resumed session accumulates context in a way a one-shot dispatch never did,
so this profile pins a more conservative compaction policy than the DSH default
(`thresholdRatio 0.8`, `retainRatio 0.16`, `maxOverflowRetries 1`):

```yaml
- id: compaction-basic
  config:
    thresholdRatio: 0.75
    retainRatio: 0.10
    maxOverflowRetries: 2
```

Read this before changing the numbers:

- compaction fires at `floor(contextWindow x thresholdRatio)` and never reserves
  the request's completion budget, so the ratio must stay below
  `(contextWindow - maxTokens) / contextWindow`. For the configured model
  (1,048,576 window, 131,072 completion) that bound is **0.875**; 0.75 keeps
  clear of it.
- **The threshold is not a defence against every rejection.** A measured failure
  on this host was refused at 63.4% of the window - 664,648 message tokens plus
  a 384,000 completion request, 72 tokens over the limit. No threshold below
  0.63 would have fired, so recovery-on-overflow (`maxOverflowRetries`) is the
  mechanism that catches that class.
- Compaction costs extra model calls. That is the price of a long-lived session
  and is bounded by `retainRatio`.

## Installation (host bootstrap)

The profile lives outside the repository, so a new host needs this once:

```bash
# 1. the plugin resolves its imports from the DSH installation
ln -sfn \
  "$(dirname "$(readlink -f "$(command -v dsh)")")/../node_modules" \
  <repo>/infra/dsh/cb16-builder-session/node_modules

# 2. the profile directory
P=~/.dsh/profiles/cb16-builder-session
mkdir -p "$P/node_modules"
ln -sfn <repo>/infra/dsh/cb16-builder-session "$P/node_modules/dsh-cb16-builder-session"

# 3. profile definition: bundles base + this plugin (+ the same optional
#    bundles the legacy headless profile carries), and the operator patch layer
#    that installs the sandbox runnerCommand - copy it from the headless profile
#    so both profiles share one sandbox boundary
cp ~/.dsh/profiles/headless/cordis.patch.yml "$P/cordis.patch.yml"
```

Verify the composition rather than trusting it:

```bash
diff <(dsh --profile headless --dump-config) \
     <(dsh --profile cb16-builder-session --dump-config)
```

Expect only the startup/runner rows, the compaction config and bundle-name
comments to differ.

## Version assumptions

Recorded at implementation time - a DSH upgrade that changes these APIs is
exactly what the `fallback-new` path exists for.

| Component | Version |
| --- | --- |
| `dsh` launcher | 0.1.0-rc.6 |
| `@deepseek-ai/dsh-agent`, `dsh-session`, `dsh-session-persistence`, `dsh-llm`, `dsh-cmdline` | 0.1.0-rc.8 |
| registry schema | `cb16.builder_session_affinity.v1` |
| plugin | 1.0.0 |

## Rollback

Two independent, immediate paths:

```bash
# a) stop opting in: drop `session_affinity: branch-v1` from the Issue metadata.
#    Every new dispatch then takes the untouched legacy headless path.
# b) disable this profile: remove ~/.dsh/profiles/cb16-builder-session.
#    Legacy Builder tasks are unaffected either way.
```

Rolling back does not delete persisted sessions or registry entries; they simply
stop being used.
