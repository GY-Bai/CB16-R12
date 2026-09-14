#!/usr/bin/env python3
"""CB16-R12 deterministic OCI dispatcher.

Implements the trusted GitHub -> OCI dispatch contract described in
``docs/OCI_DSH_DISPATCH_CONTRACT.md``.

This file is deterministic glue.  It validates trusted metadata, manages one
task worktree, runs either the Builder lane (headless DSH) or the immutable
Science lane (allowlisted entrypoint), collects bounded evidence, and publishes
the result.  It contains no scientific reasoning and makes no LLM calls of its
own.

Invariants enforced here:

* Issue text is never interpolated into a shell command; every external
  process is spawned with an explicit argv (``shell=False``).
* The Science lane can never invoke DSH.
* The Science lane resolves ``result_command`` through a repository-owned
  allowlist instead of accepting a command from the Issue.
* DSH never receives GitHub credentials: the child environment is scrubbed and
  no token is written into the worktree.
* Ambiguous or malformed metadata fails closed.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import json
import os
import re
import shutil
import socket
import subprocess
import sys

import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

DEFAULT_REPO = "GY-Bai/CB16-R12"

BUILDER_LABEL = "ds:run"
SCIENCE_LABEL = "science:run"
REVIEW_LABEL = "sol:review"
BLOCKED_LABEL = "blocked"

LANE_BUILDER = "builder"
LANE_SCIENCE = "science"

LABEL_LANE = {BUILDER_LABEL: LANE_BUILDER, SCIENCE_LABEL: LANE_SCIENCE}

#: Failure vocabulary from docs/TASK_REVIEW_PROTOCOL.md.
CLASS_OK = "OK"
CLASS_SCIENTIFIC_FAIL = "SCIENTIFIC_FAIL"
CLASS_EXECUTION_BLOCKED = "EXECUTION_BLOCKED"
CLASS_HARDWARE_LIMIT = "HARDWARE_LIMIT"
CLASS_EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
CLASS_CONTRACT_MISMATCH = "CONTRACT_MISMATCH"
#: Ordinary Builder implementation/test failure.  Deliberately distinct from
#: SCIENTIFIC_FAIL so a broken build is never reported as a scientific result.
CLASS_BUILDER_FAIL = "BUILDER_IMPLEMENTATION_FAIL"

EXIT_OK = 0
EXIT_CONTRACT_MISMATCH = 2
EXIT_EXECUTION_BLOCKED = 3
EXIT_HARDWARE_LIMIT = 4
EXIT_SCIENTIFIC_FAIL = 5
EXIT_EVIDENCE_INSUFFICIENT = 6
EXIT_BUILDER_FAIL = 7

#: Classifications that still publish a reviewable result.  Per
#: docs/OCI_DSH_DISPATCH_CONTRACT.md section 12, a Builder failure with a
#: functioning execution path must remain reviewable, and a formal experiment
#: that executes correctly but misses its gate keeps its evidence (negative
#: results are preserved).
PUBLISHABLE_CLASSIFICATIONS = {
    "builder": (CLASS_OK, CLASS_BUILDER_FAIL),
    "science": (CLASS_OK, CLASS_SCIENTIFIC_FAIL),
}

#: Files that belong to the dispatch control plane.  A Builder task may only
#: touch them when the trusted metadata sets ``allow_control_plane: true``.
CONTROL_PLANE_PATTERNS: Tuple[str, ...] = (
    ".github/",
    "scripts/cb16_dispatch.py",
    "config/cb16_science_allowlist.json",
    "docs/OCI_DSH_DISPATCH_CONTRACT.md",
)

LOG_LIMIT = 20000
BUILD_REPORT_MARKER = "BUILD_REPORT"
#: A report section starts the line (optionally behind markdown decoration),
#: which keeps prose mentions from being mistaken for the section itself.
_BUILD_REPORT_LINE_RE = re.compile(r"^[ \t>*#_`-]*BUILD_REPORT\b", re.MULTILINE)

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{1,80}$")
PATH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
IDENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@/-]{0,80}$")
#: Characters that must never appear in trusted metadata values.
SHELL_TEXT_RE = re.compile(r"[;&|`$<>(){}\[\]!*?~\"'\\\r\n\t]")

BUILDER_REQUIRED = ("mode", "base_sha", "branch", "task_file")
SCIENCE_REQUIRED = ("mode", "commit_sha", "experiment_spec", "result_command")
OPTIONAL_KEYS = ("pr_number", "review_delta", "allow_control_plane", "issue_title")


# --------------------------------------------------------------------------
# Errors: each carries a deterministic classification and exit code
# --------------------------------------------------------------------------


class DispatchError(Exception):
    """Base class for every deterministic dispatcher failure."""

    classification = CLASS_EXECUTION_BLOCKED
    exit_code = EXIT_EXECUTION_BLOCKED

    def __init__(self, message: str, *, detail: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or message


class ContractMismatch(DispatchError):
    classification = CLASS_CONTRACT_MISMATCH
    exit_code = EXIT_CONTRACT_MISMATCH


class ExecutionBlocked(DispatchError):
    classification = CLASS_EXECUTION_BLOCKED
    exit_code = EXIT_EXECUTION_BLOCKED


class HardwareLimit(DispatchError):
    classification = CLASS_HARDWARE_LIMIT
    exit_code = EXIT_HARDWARE_LIMIT


class EvidenceInsufficient(DispatchError):
    classification = CLASS_EVIDENCE_INSUFFICIENT
    exit_code = EXIT_EVIDENCE_INSUFFICIENT


class ScientificFail(DispatchError):
    classification = CLASS_SCIENTIFIC_FAIL
    exit_code = EXIT_SCIENTIFIC_FAIL


class BuilderImplementationFail(DispatchError):
    classification = CLASS_BUILDER_FAIL
    exit_code = EXIT_BUILDER_FAIL


# --------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------


def run(
    argv: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout: Optional[float] = None,
    check: bool = False,
    input_text: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Spawn an explicit argv.  ``shell`` is never used anywhere in this file."""

    try:
        proc = subprocess.run(
            list(argv),
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            input=input_text,
        )
    except FileNotFoundError as exc:
        raise ExecutionBlocked(f"executable not found: {argv[0]}", detail=str(exc))
    except subprocess.TimeoutExpired as exc:
        raise HardwareLimit(f"command exceeded timeout: {argv[0]}", detail=str(exc))
    if check and proc.returncode != 0:
        raise ExecutionBlocked(
            f"command failed ({proc.returncode}): {' '.join(str(a) for a in argv[:3])}",
            detail=(proc.stderr or proc.stdout or "")[:800],
        )
    return proc


def env_path(name: str, fallback: Path) -> Path:
    """Read a path from the environment, treating empty values as unset.

    GitHub expands an undefined ``vars.*`` reference to the empty string rather
    than leaving the variable unset, so an empty value must not silently become
    the current directory.
    """

    value = os.environ.get(name, "").strip()
    return Path(value) if value else fallback


def bounded(text: str, limit: int = LOG_LIMIT) -> str:
    """Trim a log to a deterministic bound so evidence stays reviewable."""

    if text is None:
        return ""
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n... [truncated {len(text) - limit} bytes] ...\n{tail}"


def redact(text: str, secrets: Iterable[str]) -> str:
    """Remove secrets from captured output before it is persisted."""

    if not text:
        return ""
    out = text
    for secret in secrets:
        if secret and len(secret) >= 8:
            out = out.replace(secret, "***REDACTED***")
    return out


def host_identifiers() -> List[str]:
    """Host names that must never reach a public log or artifact.

    Actions logs for a public repository are world-readable, and an OCI host's
    FQDN can encode the cloud provider, region, VCN name and a subscription
    fragment.  Longest first so the FQDN is replaced before its short form.
    """

    names = set()
    for getter in (socket.gethostname, socket.getfqdn):
        try:
            value = (getter() or "").strip()
        except OSError:
            continue
        if value and value != "localhost":
            names.add(value)
            names.add(value.split(".")[0])
    return sorted(names, key=len, reverse=True)


def redaction_terms(env: Mapping[str, str]) -> List[str]:
    """Host identifiers, the runner user's home path, and operator terms.

    The home path is included because it carries the account name into a
    world-readable Actions log, for example
    ``/home/<user>/cb16-worktrees/<task>``.
    """

    terms = host_identifiers()
    home = (env.get("HOME") or "").strip()
    if home and home not in ("/", "/root") and len(home) > 4:
        terms.append(home.rstrip("/"))
    extra = [term.strip() for term in env.get("CB16_REDACT_TERMS", "").split(",") if term.strip()]
    terms.extend(extra)
    # Longest first so a nested path is replaced before its parent.
    return sorted(set(terms), key=len, reverse=True)


def redact_hosts(text: str, terms: Sequence[str]) -> str:
    out = text or ""
    for term in terms:
        if term and len(term) >= 4:
            out = out.replace(term, "<host>")
    return out


def git(
    repo: Path,
    *args: str,
    env: Optional[Mapping[str, str]] = None,
    timeout: float = 300,
    check: bool = False,
) -> subprocess.CompletedProcess:
    return run(["git", *args], cwd=repo, env=env, timeout=timeout, check=check)


def git_out(repo: Path, *args: str, env: Optional[Mapping[str, str]] = None) -> str:
    return git(repo, *args, env=env, check=True).stdout.strip()


# --------------------------------------------------------------------------
# Event parsing and trust gating
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Trigger:
    repo: str
    issue_number: int
    label: str
    body: str
    title: str
    sender: str
    default_branch: str


def load_event(path: Path) -> Mapping[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ExecutionBlocked(f"cannot read event payload: {path}", detail=str(exc))
    except json.JSONDecodeError as exc:
        raise ContractMismatch("event payload is not valid JSON", detail=str(exc))


def parse_trigger(
    event: Mapping[str, Any],
    *,
    repo: str = DEFAULT_REPO,
    trusted_actors: Sequence[str] = (),
) -> Trigger:
    """Validate the Issue-label trigger.

    Adding the command label is the trigger; removing a label is not a command.
    """

    if event.get("action") != "labeled":
        raise ContractMismatch("event action is not 'labeled'; no dispatch")

    issue = event.get("issue") or {}
    label = (event.get("label") or {}).get("name")
    repository = (event.get("repository") or {}).get("full_name")

    if repository != repo:
        raise ContractMismatch(f"event repository {repository!r} is not the trusted {repo!r}")
    if label not in LABEL_LANE:
        raise ContractMismatch(f"label {label!r} is not a trusted command label; no dispatch")

    number = issue.get("number")
    if not isinstance(number, int) or number <= 0:
        raise ContractMismatch("event has no valid Issue number")

    sender = (event.get("sender") or {}).get("login") or ""
    if trusted_actors and sender not in trusted_actors:
        raise ContractMismatch(
            f"sender {sender!r} is not a trusted actor", detail="label must be applied by a trusted account"
        )

    return Trigger(
        repo=repo,
        issue_number=number,
        label=label,
        body=str(issue.get("body") or ""),
        title=str(issue.get("title") or ""),
        sender=sender,
        default_branch=str((event.get("repository") or {}).get("default_branch") or "main"),
    )


# --------------------------------------------------------------------------
# Trusted metadata
# --------------------------------------------------------------------------

_METADATA_FENCE = re.compile(r"^```+cb16[ \t]*$", re.MULTILINE)


def extract_metadata_block(body: str) -> str:
    """Return the raw text of the single trusted ``cb16`` fenced block."""

    if not body:
        raise ContractMismatch("Issue body is empty; no trusted metadata block")
    match = _METADATA_FENCE.search(body)
    if not match:
        raise ContractMismatch("Issue body has no ```cb16 metadata block")
    rest = body[match.end() :]
    end = re.search(r"^```+[ \t]*$", rest, re.MULTILINE)
    if not end:
        raise ContractMismatch("metadata block is not terminated")
    block = rest[: end.start()]
    if _METADATA_FENCE.search(rest[end.end() :]):
        raise ContractMismatch("Issue body contains more than one metadata block")
    return block


def parse_metadata_block(block: str) -> Dict[str, str]:
    """Parse ``key: value`` lines strictly.  Unknown/duplicate keys fail closed."""

    meta: Dict[str, str] = {}
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ContractMismatch(f"metadata line is not 'key: value': {line[:60]!r}")
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key:
            raise ContractMismatch("metadata line has an empty key")
        if key in meta:
            raise ContractMismatch(f"duplicate metadata key {key!r}")
        if key not in BUILDER_REQUIRED + SCIENCE_REQUIRED + OPTIONAL_KEYS:
            raise ContractMismatch(f"unknown metadata key {key!r}")
        if not value:
            raise ContractMismatch(f"metadata key {key!r} has an empty value")
        meta[key] = value
    if not meta:
        raise ContractMismatch("metadata block is empty")
    return meta


def reject_shell_text(value: str, field_name: str) -> str:
    """Refuse values that look like shell text.

    The dispatcher never uses a shell, so this is defence in depth: it keeps
    shell-looking payloads from ever reaching a task packet or a branch name.
    """

    if SHELL_TEXT_RE.search(value):
        raise ContractMismatch(
            f"metadata field {field_name!r} contains shell metacharacters and is refused"
        )
    return value


def validate_sha(value: str, field_name: str) -> str:
    value = reject_shell_text(value, field_name)
    if not SHA_RE.match(value):
        raise ContractMismatch(f"{field_name} must be a full 40-character lowercase commit SHA")
    return value


def validate_branch(value: str) -> str:
    value = reject_shell_text(value, "branch")
    if not BRANCH_RE.match(value):
        raise ContractMismatch(f"branch name {value!r} is outside the allowed character set")
    if value in ("main", "master") or value.startswith("-"):
        raise ContractMismatch(f"branch name {value!r} is not an allowed task branch")
    if ".." in value or "@{" in value or value.endswith(".lock") or "//" in value:
        raise ContractMismatch(f"branch name {value!r} is not an allowed task branch")
    if not value.startswith(("ds/", "science/")):
        raise ContractMismatch("branch name must start with 'ds/' or 'science/'")
    tail = value.split("/", 1)[1]
    if len(tail) < 2 or tail.startswith("/") or tail.endswith("/"):
        raise ContractMismatch("branch name must carry a concrete task name after its prefix")
    return value


def validate_repo_path(value: str, field_name: str) -> str:
    value = reject_shell_text(value, field_name)
    if not PATH_RE.match(value):
        raise ContractMismatch(f"{field_name} is not a plain repo-relative path")
    if value.startswith("/") or "//" in value:
        raise ContractMismatch(f"{field_name} must be a repo-relative path")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ContractMismatch(f"{field_name} must not contain empty, '.' or '..' segments")
    return value


@dataclass(frozen=True)
class TaskSpec:
    lane: str
    mode: str
    sha: str
    branch: str
    task_file: Optional[str] = None
    experiment_spec: Optional[str] = None
    result_command: Optional[str] = None
    pr_number: Optional[int] = None
    allow_control_plane: bool = False
    review_delta: Optional[str] = None
    issue_title: Optional[str] = None
    raw: Mapping[str, str] = field(default_factory=dict)


def validate_metadata(
    meta: Mapping[str, str],
    lane: str,
    *,
    allowlist: Optional[Mapping[str, Any]] = None,
) -> TaskSpec:
    """Validate trusted metadata for one lane.  Builder and Science never mix."""

    if lane not in (LANE_BUILDER, LANE_SCIENCE):
        raise ContractMismatch(f"unknown lane {lane!r}")

    required = BUILDER_REQUIRED if lane == LANE_BUILDER else SCIENCE_REQUIRED
    for key in required:
        if key not in meta:
            raise ContractMismatch(f"metadata is missing required key {key!r} for the {lane} lane")

    mode = reject_shell_text(meta["mode"], "mode")
    allowed_modes = ("build", "fix") if lane == LANE_BUILDER else ("science",)
    if mode not in allowed_modes:
        raise ContractMismatch(
            f"mode {mode!r} is not valid for the {lane} lane (expected one of {allowed_modes})"
        )

    if lane == LANE_BUILDER:
        if "commit_sha" in meta or "experiment_spec" in meta or "result_command" in meta:
            raise ContractMismatch("Builder metadata must not carry Science fields")
        sha = validate_sha(meta["base_sha"], "base_sha")
        branch = validate_branch(meta["branch"])
        task_file = validate_repo_path(meta["task_file"], "task_file")
        experiment_spec = None
        result_command = None
    else:
        if "base_sha" in meta or "task_file" in meta:
            raise ContractMismatch("Science metadata must not carry Builder fields")
        sha = validate_sha(meta["commit_sha"], "commit_sha")
        branch = validate_branch(meta["branch"]) if "branch" in meta else f"science/run-{sha[:12]}"
        task_file = None
        experiment_spec = validate_repo_path(meta["experiment_spec"], "experiment_spec")
        result_command = reject_shell_text(meta["result_command"], "result_command")
        if not IDENT_RE.match(result_command):
            raise ContractMismatch("result_command is not a valid allowlist identifier")
        if allowlist is not None and result_command not in allowlist.get("entrypoints", {}):
            raise ContractMismatch(
                f"result_command {result_command!r} is not in the repository allowlist"
            )

    pr_number: Optional[int] = None
    if "pr_number" in meta:
        raw_pr = reject_shell_text(meta["pr_number"], "pr_number")
        if not raw_pr.isdigit():
            raise ContractMismatch("pr_number must be a positive integer")
        pr_number = int(raw_pr)

    allow_control_plane = False
    if "allow_control_plane" in meta:
        raw_flag = meta["allow_control_plane"].strip().lower()
        if raw_flag not in ("true", "false"):
            raise ContractMismatch("allow_control_plane must be 'true' or 'false'")
        allow_control_plane = raw_flag == "true"

    return TaskSpec(
        lane=lane,
        mode=mode,
        sha=sha,
        branch=branch,
        task_file=task_file,
        experiment_spec=experiment_spec,
        result_command=result_command,
        pr_number=pr_number,
        allow_control_plane=allow_control_plane,
        review_delta=meta.get("review_delta"),
        issue_title=meta.get("issue_title"),
        raw=dict(meta),
    )


# --------------------------------------------------------------------------
# Allowlist
# --------------------------------------------------------------------------


def build_pr_title(issue_title: str, issue_number: int) -> str:
    """Compose the Draft PR title without doubling an existing task prefix."""

    title = (issue_title or "").strip() or f"Issue #{issue_number}"
    if not title.lstrip().upper().startswith("[R12]"):
        title = f"[R12] {title}"
    return title[:200]


def load_allowlist(path: Path) -> Mapping[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ExecutionBlocked(f"cannot read allowlist: {path}", detail=str(exc))
    except json.JSONDecodeError as exc:
        raise ContractMismatch("science allowlist is not valid JSON", detail=str(exc))
    if not isinstance(data, dict) or not isinstance(data.get("entrypoints"), dict):
        raise ContractMismatch("science allowlist must contain an 'entrypoints' object")
    return data


def load_data_manifest(path: Path) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """Resolve the read-only data manifest against this host.

    Returns the entries whose path exists plus the logical names that are
    missing here.  A missing file yields an empty manifest rather than an
    error, so the dispatcher still runs on a host without the data mounted.
    """

    if not path.exists():
        return {}, []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractMismatch("data manifest is unreadable", detail=str(exc))
    entries = data.get("read_only")
    if not isinstance(entries, dict):
        raise ContractMismatch("data manifest must contain a 'read_only' object")

    available: Dict[str, Dict[str, str]] = {}
    missing: List[str] = []
    for name, spec in entries.items():
        target = (spec or {}).get("path") if isinstance(spec, dict) else None
        if not isinstance(target, str) or not target.startswith("/"):
            raise ContractMismatch(f"data manifest entry {name!r} has no absolute path")
        if Path(target).exists():
            available[name] = {"path": target, "description": str((spec or {}).get("description", ""))}
        else:
            missing.append(name)
    return available, missing


def resolve_allowlisted_argv(entrypoint: Mapping[str, Any], worktree: Path) -> List[str]:
    argv = entrypoint.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
        raise ContractMismatch("allowlisted entrypoint has no valid 'argv'")
    for token in argv:
        if token.startswith("/") or ".." in Path(token).parts:
            raise ContractMismatch("allowlisted argv may not contain absolute or parent paths")
    return list(argv)


# --------------------------------------------------------------------------
# Workspace management
# --------------------------------------------------------------------------


def ensure_sha_present(repo: Path, sha: str) -> None:
    proc = git(repo, "cat-file", "-e", f"{sha}^{{commit}}")
    if proc.returncode != 0:
        raise ContractMismatch(f"commit {sha} does not exist in the repository")


def ensure_path_at_sha(repo: Path, sha: str, path: str) -> None:
    proc = git(repo, "cat-file", "-e", f"{sha}:{path}")
    if proc.returncode != 0:
        raise ContractMismatch(f"{path} does not exist at commit {sha}")


def worktree_for_branch(repo: Path, branch: str) -> Optional[Path]:
    proc = git(repo, "worktree", "list", "--porcelain")
    if proc.returncode != 0:
        return None
    current: Optional[Path] = None
    for line in proc.stdout.splitlines():
        if line.startswith("worktree "):
            current = Path(line[len("worktree ") :])
        elif line.startswith("branch ") and current is not None:
            ref = line[len("branch ") :]
            if ref == f"refs/heads/{branch}":
                return current
    return None


def ensure_worktree(repo: Path, work_root: Path, spec: TaskSpec) -> Tuple[Path, bool]:
    """Create or reuse exactly one worktree for the task branch.

    Returns ``(path, reused)``.
    """

    # Drop registrations whose directories no longer exist.
    git(repo, "worktree", "prune")

    existing = worktree_for_branch(repo, spec.branch)
    if existing is not None:
        return existing, True

    work_root.mkdir(parents=True, exist_ok=True)
    target = work_root / spec.branch.replace("/", "__")
    if target.exists():
        # A directory with no registration is residue from an earlier run whose
        # repository was recreated underneath it.  Drop it and start clean.
        shutil.rmtree(target, ignore_errors=True)
        if target.exists():
            raise ExecutionBlocked(f"cannot clear stale worktree directory: {target}")

    has_local = git(repo, "show-ref", "--verify", "--quiet", f"refs/heads/{spec.branch}").returncode == 0
    has_remote = (
        git(repo, "show-ref", "--verify", "--quiet", f"refs/remotes/origin/{spec.branch}").returncode == 0
    )

    if has_local:
        proc = git(repo, "worktree", "add", str(target), spec.branch)
    elif has_remote:
        proc = git(repo, "worktree", "add", str(target), "-b", spec.branch, f"origin/{spec.branch}")
    else:
        proc = git(repo, "worktree", "add", str(target), "-b", spec.branch, spec.sha)
    if proc.returncode != 0:
        raise ExecutionBlocked("failed to create task worktree", detail=bounded(proc.stderr))

    head = git_out(target, "rev-parse", "--abbrev-ref", "HEAD")
    if head != spec.branch:
        raise ExecutionBlocked(f"worktree is on {head!r} instead of the intended branch {spec.branch!r}")
    return target, False


def write_task_packet(
    worktree: Path,
    spec: TaskSpec,
    trigger: Trigger,
    *,
    review_delta: Optional[str] = None,
    data_entries: Optional[Mapping[str, Mapping[str, str]]] = None,
    caches: Optional[Mapping[str, str]] = None,
) -> Path:
    """Materialise the deterministic local task packet (never committed)."""

    packet_dir = worktree / ".cb16"
    packet_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CB16 Task Packet",
        "",
        f"Repository: {trigger.repo}",
        f"Issue number: {trigger.issue_number}",
        f"Lane: {spec.lane}",
        f"Mode: {spec.mode}",
        f"Base SHA: {spec.sha}",
        f"Task branch: {spec.branch}",
    ]
    if spec.task_file:
        lines.append(f"Task contract file: {spec.task_file}")
    if spec.experiment_spec:
        lines.append(f"Experiment spec: {spec.experiment_spec}")
    if spec.result_command:
        lines.append(f"Result command: {spec.result_command}")
    if spec.pr_number is not None:
        lines.append(f"Existing PR number: {spec.pr_number}")
    else:
        lines.append("Existing PR number: (none)")
    if review_delta:
        lines.append("")
        lines.append("## Review delta")
        lines.append("")
        lines.append(review_delta.strip())
    if data_entries:
        lines += ["", "## Read-only data available", ""]
        for name, entry in sorted(data_entries.items()):
            lines.append(f"- `{name}` -> `{entry['path']}`")
            if entry.get("description"):
                lines.append(f"  {entry['description']}")
        lines += [
            "",
            "These paths are **inputs only**: never write to them, and never copy",
            "them into the workspace unless the contract asks for it.",
        ]
    if caches:
        lines += ["", "## Package caches (workspace-local)", ""]
        for key, value in sorted(caches.items()):
            lines.append(f"- `{key}` -> `{value}`")
        lines += [
            "",
            "These directories persist for the lifetime of this worktree and are",
            "git-ignored, so package downloads are reused across fix cycles.",
        ]
    lines += [
        "",
        "## Allowed scope",
        "",
        f"Work only inside this worktree ({worktree}).",
        f"The task contract at {spec.task_file or spec.experiment_spec} defines the exact",
        "file scope for this task; honour it literally and keep the smallest footprint.",
        "",
        "## Forbidden scope",
        "",
        "Do not redesign authority, semantics, or acceptance criteria.",
        "Do not touch the dispatch control plane unless this packet authorises it.",
        "Do not create commits, tags, or pushes: the dispatcher owns Git state.",
        "",
        "## Done-when conditions",
        "",
        "The task contract file states the required tests and done-when conditions.",
        "End your final message with a `BUILD_REPORT` section.",
    ]
    packet = packet_dir / "TASK_PACKET.md"
    packet.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return packet


def collect_changed_files(worktree: Path) -> List[str]:
    # -uall lists every untracked file individually; without it a brand-new
    # directory would be reported as a single `path/` entry.
    proc = git(worktree, "status", "--porcelain", "-uall")
    if proc.returncode != 0:
        raise ExecutionBlocked("cannot read worktree status", detail=bounded(proc.stderr))
    changed: List[str] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        changed.append(path.strip('"'))
    return sorted(changed)


def control_plane_violations(changed: Sequence[str]) -> List[str]:
    hits: List[str] = []
    for path in changed:
        if path.startswith(".cb16/"):
            continue
        for pattern in CONTROL_PLANE_PATTERNS:
            if pattern.endswith("/"):
                if path.startswith(pattern):
                    hits.append(path)
                    break
            elif path == pattern:
                hits.append(path)
                break
    return sorted(set(hits))


# --------------------------------------------------------------------------
# Child environment: the credential boundary
# --------------------------------------------------------------------------

#: Environment keys DSH legitimately needs.
_ENV_ALLOW_EXACT = {
    "PATH",
    "HOME",
    "DSH_HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TZ",
    "TMPDIR",
    "USER",
    "LOGNAME",
    "SHELL",
    "NODE_PATH",
    "NVM_DIR",
    "NVM_BIN",
    "DEEPSEEK_API_KEY",
}

#: Patterns that must never reach the Builder child.
_ENV_DENY_RE = re.compile(
    r"(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|^AWS_|^OCI_|^GH_|^GITHUB_|^SSH_|_KEY$)",
    re.IGNORECASE,
)


def scrubbed_env(base: Mapping[str, str], *, allow_extra: Sequence[str] = ()) -> Dict[str, str]:
    """Build the environment handed to DSH: no GitHub write credentials.

    ``DEEPSEEK_API_KEY`` is the only credential the model runtime needs, so it
    is allowed explicitly while every token/secret-shaped variable is dropped.
    """

    allow = set(_ENV_ALLOW_EXACT) | set(allow_extra)
    env: Dict[str, str] = {}
    for key, value in base.items():
        if key in allow:
            env[key] = value
        elif _ENV_DENY_RE.search(key):
            continue
        elif key.startswith("CB16_"):
            continue
        elif key.startswith("DSH_"):
            env[key] = value
    # Neutralise implicit credential lookups that shell out to git.
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_ASKPASS"] = "/bin/true"
    env["GIT_CONFIG_GLOBAL"] = "/dev/null"
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    # Keep lane children from leaving __pycache__ inside a frozen worktree.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def git_auth_env(base: Mapping[str, str], token: Optional[str]) -> Dict[str, str]:
    """Environment carrying a one-shot git auth header for fetch/clone.

    The header travels through GIT_CONFIG_COUNT/KEY/VALUE rather than argv so
    the token never appears in a process listing.
    """

    env = publish_env(base)
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode("utf-8")).decode("ascii")
        env["GIT_CONFIG_COUNT"] = "1"
        env["GIT_CONFIG_KEY_0"] = "http.https://github.com/.extraheader"
        env["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: basic {basic}"
    return env


def ensure_repo_clone(
    clone_dir: Path,
    url: str,
    *,
    env: Mapping[str, str],
    timeout: float = 600,
) -> Path:
    """Maintain the dispatcher's own persistent clone of the trusted repository.

    Task worktrees live in this clone, never in the Actions workspace: the
    checkout step deletes local branches (and, when it cannot, recreates the
    whole repository), which would otherwise orphan worktree registrations.
    """

    if (clone_dir / ".git").exists():
        proc = git(
            clone_dir,
            "fetch",
            "--prune",
            "origin",
            "+refs/heads/*:refs/remotes/origin/*",
            env=env,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise ExecutionBlocked("cannot fetch the dispatch repository", detail=bounded(proc.stderr))
        return clone_dir

    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    proc = run(["git", "clone", "--quiet", url, str(clone_dir)], env=dict(env), timeout=timeout)
    if proc.returncode != 0:
        raise ExecutionBlocked("cannot clone the dispatch repository", detail=bounded(proc.stderr))
    return clone_dir


def publish_env(base: Mapping[str, str]) -> Dict[str, str]:
    """Environment for the credentialed publish steps.

    Publishing must use exactly the token the dispatcher was handed.  The host
    may carry its own git credential helper (for example `gh auth
    git-credential`), so global/system git config is disabled here and the push
    additionally clears `credential.helper` on the command line.
    """

    env = dict(base)
    env["GIT_CONFIG_GLOBAL"] = "/dev/null"
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_ASKPASS"] = "/bin/true"
    return env


def credential_warnings(home: Path) -> List[str]:
    """Host-provisioning warnings; never blocks, but always surfaces in evidence."""

    warnings: List[str] = []
    gh_hosts = home / ".config" / "gh" / "hosts.yml"
    if gh_hosts.exists():
        warnings.append(
            f"host credential file present and readable by the runner user: {gh_hosts} "
            "(recommend `gh auth logout` on the runner host or a dedicated runner user)"
        )
    ssh_dir = home / ".ssh"
    if ssh_dir.is_dir():
        keys = [p.name for p in ssh_dir.glob("id_*") if not p.name.endswith(".pub")]
        if keys:
            warnings.append(
                f"SSH private keys present for the runner user: {', '.join(sorted(keys))} "
                "(DSH can read them; prefer a dedicated runner user)"
            )
    return warnings


# --------------------------------------------------------------------------
# Lanes
# --------------------------------------------------------------------------


@dataclass
class RunResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


DSH_PROMPT = (
    "Read the local CB16 task packet at .cb16/TASK_PACKET.md and execute it exactly. "
    "Do not redesign authority. Do not create commits or push. End with BUILD_REPORT."
)


def invoke_dsh(
    worktree: Path,
    *,
    dsh_bin: str = "dsh",
    env: Mapping[str, str],
    timeout: float = 3600,
    profile: str = "headless",
    prompt: str = DSH_PROMPT,
) -> RunResult:
    """Run a fresh headless DSH session inside the task worktree."""

    argv = [dsh_bin, "--profile", profile, prompt]
    try:
        proc = subprocess.run(
            argv,
            cwd=str(worktree),
            env=dict(env),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise ExecutionBlocked("DSH executable not found", detail=str(exc))
    except subprocess.TimeoutExpired as exc:
        return RunResult(exit_code=124, stdout=exc.stdout or "", stderr=exc.stderr or "", timed_out=True)
    return RunResult(exit_code=proc.returncode, stdout=proc.stdout or "", stderr=proc.stderr or "")


def run_science_entrypoint(
    worktree: Path,
    spec: TaskSpec,
    *,
    allowlist: Mapping[str, Any],
    env: Mapping[str, str],
    result_dir: Path,
    timeout: float = 7200,
) -> RunResult:
    """Run the frozen Science lane.  This function must never invoke DSH.

    There is deliberately no DSH parameter or DSH call on this path: the
    Science lane exists so a formal run cannot be adaptively repaired by an
    agent after its results become visible.
    """

    entrypoint = allowlist["entrypoints"].get(spec.result_command or "")
    if entrypoint is None:
        raise ContractMismatch(f"result_command {spec.result_command!r} is not allowlisted")

    if entrypoint.get("dry_run_only") and env.get("CB16_ALLOW_DRY_RUN_ONLY") != "1":
        raise ContractMismatch(
            f"entrypoint {spec.result_command!r} is a dry-run-only entrypoint "
            "and cannot run a formal scientific execution"
        )

    argv = resolve_allowlisted_argv(entrypoint, worktree)
    child_env = dict(env)
    for key, value in (entrypoint.get("env") or {}).items():
        if not re.match(r"^[A-Z][A-Z0-9_]*$", str(key)):
            raise ContractMismatch(f"allowlisted entrypoint declares an invalid env key: {key!r}")
        child_env[str(key)] = str(value)
    child_env["CB16_RESULT_DIR"] = str(result_dir)
    child_env["CB16_RESULT_COMMAND"] = spec.result_command or ""
    child_env["CB16_COMMIT_SHA"] = spec.sha
    result_dir.mkdir(parents=True, exist_ok=True)

    try:
        proc = subprocess.run(
            argv,
            cwd=str(worktree),
            env=child_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise ExecutionBlocked("science entrypoint not found", detail=str(exc))
    except subprocess.TimeoutExpired as exc:
        return RunResult(exit_code=124, stdout=exc.stdout or "", stderr=exc.stderr or "", timed_out=True)
    return RunResult(exit_code=proc.returncode, stdout=proc.stdout or "", stderr=proc.stderr or "")


def run_dry_run(worktree: Path, spec: TaskSpec, *, result_dir: Path) -> RunResult:
    """Deterministic stand-in that exercises the whole path without real work."""

    if spec.lane == LANE_BUILDER:
        fixture = worktree / "docs" / "dispatch_smoke"
        fixture.mkdir(parents=True, exist_ok=True)
        (fixture / "DRY_RUN_FIXTURE.md").write_text(
            "# Dispatch dry-run fixture\n\n"
            "Created by scripts/cb16_dispatch.py --dry-run.\n"
            "This file exists so the dispatch path can be exercised end to end\n"
            "without touching any scientific semantics.\n",
            encoding="utf-8",
        )
        return RunResult(
            exit_code=0,
            stdout="DRY_RUN builder lane stub complete\nBUILD_REPORT: dry-run fixture written\n",
            note="dry-run stub (DSH not invoked)",
        )
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "cb16.result.v1",
        "status": "DRY_RUN",
        "commit_sha": spec.sha,
        "result_command": spec.result_command,
        "metrics": {"noop": 0},
    }
    (result_dir / "RESULT.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (result_dir / "REPORT.md").write_text(
        "# Dry run report\n\nDeterministic no-op science-lane stub.\n", encoding="utf-8"
    )
    return RunResult(
        exit_code=0,
        stdout="DRY_RUN science lane stub complete\nBUILD_REPORT: dry-run artifacts written\n",
        note="dry-run stub (allowlisted entrypoint not executed)",
    )


def run_repo_tests(worktree: Path, *, env: Mapping[str, str], timeout: float = 900) -> RunResult:
    """Run the repository-owned test entrypoint (never supplied by the Issue)."""

    argv = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."]
    if not (worktree / "tests").is_dir():
        return RunResult(exit_code=0, stdout="no tests directory present\n", note="tests skipped")
    try:
        proc = subprocess.run(
            argv, cwd=str(worktree), env=dict(env), capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        return RunResult(exit_code=124, stdout=exc.stdout or "", stderr=exc.stderr or "", timed_out=True)
    return RunResult(exit_code=proc.returncode, stdout=proc.stdout or "", stderr=proc.stderr or "")


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------


def extract_build_report(text: str) -> Optional[str]:
    """Return the final BUILD_REPORT section, ignoring incidental mentions.

    A trailing prose reference such as "the report ends with `BUILD_REPORT`"
    must not displace the real section, so only a line whose first meaningful
    token is the marker is accepted.
    """

    if not text:
        return None
    matches = list(_BUILD_REPORT_LINE_RE.finditer(text))
    if not matches:
        return None
    return text[matches[-1].start() :].strip()


def synthesize_build_report(
    *,
    spec: TaskSpec,
    classification: str,
    changed: Sequence[str],
    test_result: Optional[RunResult],
    run_result: RunResult,
) -> str:
    lines = [
        "BUILD_REPORT",
        f"Task: Issue #{spec.branch} ({spec.mode})",
        f"Branch: {spec.branch}",
        f"Classification: {classification}",
        f"Changed: {', '.join(changed) if changed else '(no file changes)'}",
    ]
    if test_result is not None:
        lines.append(f"Targeted tests: exit {test_result.exit_code}")
    lines.append(f"Lane exit status: {run_result.exit_code}")
    if run_result.note:
        lines.append(f"Note: {run_result.note}")
    lines.append("Known unresolved: see Actions log and evidence bundle")
    return "\n".join(lines)


def write_evidence(
    report_dir: Path,
    *,
    payload: Mapping[str, Any],
    build_report: str,
    logs: Mapping[str, str],
    redactions: Sequence[str] = (),
) -> Dict[str, Path]:
    """Persist evidence with host identifiers scrubbed.

    Actions logs and artifacts for a public repository are world-readable, so
    nothing written here may carry the runner host's name or FQDN.
    """

    def scrub(text: str) -> str:
        return redact_hosts(text, redactions)

    report_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    summary_path = report_dir / "dispatch_summary.json"
    summary_path.write_text(
        scrub(json.dumps(payload, indent=2, sort_keys=True)) + "\n", encoding="utf-8"
    )
    written["summary"] = summary_path
    report_path = report_dir / "BUILD_REPORT.md"
    report_path.write_text(scrub(build_report).rstrip() + "\n", encoding="utf-8")
    written["build_report"] = report_path
    for name, text in logs.items():
        path = report_dir / f"{name}.log"
        path.write_text(scrub(bounded(text)), encoding="utf-8")
        written[name] = path
    return written


# --------------------------------------------------------------------------
# Locking
# --------------------------------------------------------------------------


class TaskLock:
    """Prevents the same Issue from running twice concurrently on one host."""

    def __init__(self, state_dir: Path, issue_number: int, lane: str) -> None:
        self.path = state_dir / "locks" / f"{lane}-{issue_number}.lock"
        self._handle = None

    def __enter__(self) -> "TaskLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._handle.close()
            self._handle = None
            raise ExecutionBlocked(f"another dispatch already holds {self.path.name}")
        self._handle.write(f"pid={os.getpid()}\n")
        self._handle.flush()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._handle is not None:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._handle.close()
                self._handle = None


# --------------------------------------------------------------------------
# Publishing (runs only after the lane child has exited)
# --------------------------------------------------------------------------


def github_api(
    method: str,
    path: str,
    *,
    token: str,
    payload: Optional[Mapping[str, Any]] = None,
    timeout: float = 30,
) -> Tuple[int, Any]:
    url = path if path.startswith("http") else f"https://api.github.com{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    request.add_header("User-Agent", "cb16-dispatch/1.0")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise ExecutionBlocked(f"GitHub API {method} {url} returned HTTP {exc.code}", detail=detail)
    except urllib.error.URLError as exc:
        raise ExecutionBlocked(f"GitHub API {method} {url} unreachable", detail=str(exc))


def commit_worktree(
    worktree: Path,
    *,
    message: str,
    author_name: str = "Gengyuan Bai",
    author_email: str = "37661207+GY-Bai@users.noreply.github.com",
    env: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    changed = collect_changed_files(worktree)
    if not changed:
        return None
    # `.cb16` holds the transient task packet.  Remove it before staging so it
    # cannot be committed even if `.gitignore` is missing the entry or the agent
    # dropped extra files in there.  Note that `git add -A -- .` would abort on
    # the ignored directory, so no pathspec is passed here.
    shutil.rmtree(worktree / ".cb16", ignore_errors=True)
    child_env = publish_env(env or os.environ)
    git(worktree, "add", "-A", env=child_env, check=True)
    child_env.update(
        {
            "GIT_AUTHOR_NAME": author_name,
            "GIT_AUTHOR_EMAIL": author_email,
            "GIT_COMMITTER_NAME": author_name,
            "GIT_COMMITTER_EMAIL": author_email,
        }
    )
    proc = git(worktree, "commit", "-m", message, env=child_env)
    if proc.returncode != 0:
        raise ExecutionBlocked("git commit failed", detail=bounded(proc.stderr))
    return git_out(worktree, "rev-parse", "HEAD", env=env)


def push_branch(worktree: Path, *, branch: str, token: str, slug: str, env: Mapping[str, str]) -> None:
    """Push using a one-shot auth header that is never persisted to config."""

    basic = base64.b64encode(f"x-access-token:{token}".encode("utf-8")).decode("ascii")
    proc = git(
        worktree,
        # Clear any inherited helper so the ephemeral dispatch token is the only
        # credential that can be used for this push.
        "-c",
        "credential.helper=",
        "-c",
        f"http.https://github.com/.extraheader=AUTHORIZATION: basic {basic}",
        "push",
        f"https://github.com/{slug}.git",
        f"HEAD:refs/heads/{branch}",
        env=publish_env(env),
        timeout=300,
    )
    if proc.returncode != 0:
        raise ExecutionBlocked("git push failed", detail=redact(bounded(proc.stderr), [token]))


def create_or_update_draft_pr(
    *,
    token: str,
    slug: str,
    branch: str,
    base: str,
    title: str,
    body: str,
    pr_number: Optional[int] = None,
) -> Mapping[str, Any]:
    if pr_number is not None:
        status, data = github_api(
            "PATCH", f"/repos/{slug}/pulls/{pr_number}", token=token, payload={"body": body}
        )
        return data
    try:
        status, data = github_api(
            "POST",
            f"/repos/{slug}/pulls",
            token=token,
            payload={"title": title, "head": branch, "base": base, "body": body, "draft": True},
        )
        return data
    except ExecutionBlocked:
        # A fix cycle re-dispatches the same branch, so an open PR for this head
        # may already exist.  Update it instead of failing on the 422.
        owner = slug.split("/", 1)[0]
        status, existing = github_api(
            "GET", f"/repos/{slug}/pulls?state=open&head={owner}:{branch}", token=token
        )
        if isinstance(existing, list) and existing:
            number = existing[0]["number"]
            status, data = github_api(
                "PATCH", f"/repos/{slug}/pulls/{number}", token=token, payload={"body": body}
            )
            return data
        raise


def set_issue_labels(
    *,
    token: str,
    slug: str,
    issue_number: int,
    add: Sequence[str] = (),
    remove: Sequence[str] = (),
) -> None:
    if add:
        github_api("POST", f"/repos/{slug}/issues/{issue_number}/labels", token=token, payload={"labels": list(add)})
    for label in remove:
        try:
            github_api("DELETE", f"/repos/{slug}/issues/{issue_number}/labels/{label}", token=token)
        except ExecutionBlocked:
            pass


def comment_on_issue(*, token: str, slug: str, issue_number: int, body: str) -> None:
    github_api(
        "POST", f"/repos/{slug}/issues/{issue_number}/comments", token=token, payload={"body": bounded(body, 60000)}
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


@dataclass
class DispatchOutcome:
    classification: str
    exit_code: int
    build_report: str
    summary: Dict[str, Any]
    evidence: Dict[str, Path] = field(default_factory=dict)


def dispatch(
    *,
    lane: str,
    event: Mapping[str, Any],
    repo_dir: Path,
    work_root: Path,
    state_dir: Path,
    report_dir: Path,
    allowlist_path: Path,
    data_manifest_path: Optional[Path] = None,
    repo: str = DEFAULT_REPO,
    repo_url: Optional[str] = None,
    repo_clone_dir: Optional[Path] = None,
    trusted_actors: Sequence[str] = (),
    dry_run: bool = False,
    publish: bool = False,
    github_token: Optional[str] = None,
    dsh_bin: str = "dsh",
    dsh_timeout: float = 3600,
    dsh_invoker: Optional[Any] = None,
    science_invoker: Optional[Any] = None,
    base_env: Optional[Mapping[str, str]] = None,
) -> DispatchOutcome:
    """Run one dispatch end to end.  Pure enough to unit-test with fakes."""

    env = dict(base_env or os.environ)
    trigger = parse_trigger(event, repo=repo, trusted_actors=trusted_actors)
    if LABEL_LANE[trigger.label] != lane:
        raise ContractMismatch(
            f"label {trigger.label!r} belongs to the {LABEL_LANE[trigger.label]} lane, not {lane}"
        )

    data_manifest_path = data_manifest_path or (Path(__file__).resolve().parent.parent / "config" / "cb16_data_manifest.json")
    allowlist = load_allowlist(allowlist_path) if lane == LANE_SCIENCE else {"entrypoints": {}}
    meta = parse_metadata_block(extract_metadata_block(trigger.body))
    spec = validate_metadata(meta, lane, allowlist=allowlist)

    if repo_url:
        # Own a persistent clone instead of borrowing the Actions checkout.
        repo_path = ensure_repo_clone(
            repo_clone_dir or (work_root / "repo"),
            repo_url,
            env=git_auth_env(env, github_token),
        )
    else:
        repo_path = repo_dir

    ensure_sha_present(repo_path, spec.sha)
    if spec.task_file:
        ensure_path_at_sha(repo_path, spec.sha, spec.task_file)
    if spec.experiment_spec:
        ensure_path_at_sha(repo_path, spec.sha, spec.experiment_spec)

    warnings = credential_warnings(Path(env.get("HOME", str(Path.home()))))
    redactions = redaction_terms(env)
    data_entries, data_missing = load_data_manifest(data_manifest_path)

    with TaskLock(state_dir, trigger.issue_number, lane):
        worktree, reused = ensure_worktree(repo_path, work_root, spec)

        # Workspace-local, git-ignored caches: the sandbox only permits writes
        # under the worktree, so this is where package downloads must land to be
        # reused instead of re-fetched on every dispatch.
        caches = {
            "UV_CACHE_DIR": str(worktree / ".uv-cache"),
            "PIP_CACHE_DIR": str(worktree / ".pip-cache"),
            "UV_PROJECT_ENVIRONMENT": str(worktree / ".venv"),
        }
        for key in ("UV_CACHE_DIR", "PIP_CACHE_DIR"):
            Path(caches[key]).mkdir(parents=True, exist_ok=True)

        packet = write_task_packet(
            worktree,
            spec,
            trigger,
            review_delta=spec.review_delta,
            data_entries=data_entries,
            caches=caches,
        )

        child_env = scrubbed_env(env)
        child_env.update(caches)
        child_env["UV_LINK_MODE"] = "copy"
        child_env["CB16_DATA_MANIFEST"] = str(data_manifest_path)
        if data_entries:
            child_env["CB16_DATA_ROOT"] = str(next(iter(data_entries.values()))["path"])
        result_dir = report_dir / "results"

        if dry_run and lane == LANE_BUILDER:
            run_result = run_dry_run(worktree, spec, result_dir=result_dir)
        elif lane == LANE_BUILDER:
            invoker = dsh_invoker or invoke_dsh
            run_result = invoker(
                worktree, dsh_bin=dsh_bin, env=child_env, timeout=dsh_timeout, prompt=DSH_PROMPT
            )
        else:
            invoker = science_invoker or run_science_entrypoint
            science_env = dict(child_env)
            if dry_run:
                # A dry run still executes the real allowlisted entrypoint so the
                # allowlist/resolution/artifact path is exercised, but the
                # dry-run-only guard keeps such an entrypoint out of formal runs.
                science_env["CB16_ALLOW_DRY_RUN_ONLY"] = "1"
            run_result = invoker(
                worktree,
                spec,
                allowlist=allowlist,
                env=science_env,
                result_dir=result_dir,
                timeout=dsh_timeout,
            )

        changed = collect_changed_files(worktree)
        test_result: Optional[RunResult] = None

        classification = CLASS_OK
        if run_result.timed_out:
            classification = CLASS_HARDWARE_LIMIT
        elif not run_result.ok:
            classification = CLASS_BUILDER_FAIL if lane == LANE_BUILDER else CLASS_SCIENTIFIC_FAIL

        # Science lane must be immutable: no source or config edits after start.
        if lane == LANE_SCIENCE:
            if changed:
                raise ContractMismatch(
                    "science lane mutated the worktree; frozen runs may not edit source or config",
                    detail=", ".join(changed[:20]),
                )
            expected = allowlist["entrypoints"][spec.result_command or ""].get("produces", [])
            missing = [name for name in expected if not (result_dir / name).exists()]
            if missing and classification == CLASS_OK:
                classification = CLASS_EVIDENCE_INSUFFICIENT
        else:
            violations = control_plane_violations(changed)
            if violations and not spec.allow_control_plane:
                raise ContractMismatch(
                    "Builder touched control-plane files without authorisation",
                    detail=", ".join(violations),
                )
            test_result = run_repo_tests(worktree, env=child_env)
            if not test_result.ok and classification == CLASS_OK:
                classification = CLASS_BUILDER_FAIL

        if classification == CLASS_OK:
            report_text = extract_build_report(run_result.stdout) or synthesize_build_report(
                spec=spec,
                classification=classification,
                changed=changed,
                test_result=test_result,
                run_result=run_result,
            )
        else:
            report_text = synthesize_build_report(
                spec=spec,
                classification=classification,
                changed=changed,
                test_result=test_result,
                run_result=run_result,
            )

        summary: Dict[str, Any] = {
            "schema": "cb16.dispatch.v1",
            "repo": repo,
            "issue_number": trigger.issue_number,
            "label": trigger.label,
            "lane": lane,
            "mode": spec.mode,
            "base_sha": spec.sha,
            "branch": spec.branch,
            "task_file": spec.task_file,
            "experiment_spec": spec.experiment_spec,
            "result_command": spec.result_command,
            "worktree": str(worktree),
            "worktree_reused": reused,
            "work_repo": str(repo_path),
            "dry_run": dry_run,
            "classification": classification,
            "lane_exit_status": run_result.exit_code,
            "changed_files": changed,
            "control_plane_authorised": spec.allow_control_plane,
            "read_only_data": {name: entry["path"] for name, entry in data_entries.items()},
            "read_only_data_missing": data_missing,
            "workspace_caches": caches,
            "host_identifiers_redacted": len(redactions),
            "credential_warnings": warnings,
            "published": False,
        }

        publishable = classification in PUBLISHABLE_CLASSIFICATIONS.get(lane, ())
        if publish and publishable and not dry_run:
            if not github_token:
                raise ExecutionBlocked("publish requested but no GitHub token was provided")
            if lane == LANE_BUILDER:
                if changed:
                    commit_worktree(
                        worktree, message=f"CB16 {lane}: issue #{trigger.issue_number} ({spec.mode})"
                    )
                    push_branch(worktree, branch=spec.branch, token=github_token, slug=repo, env=env)
                pr = create_or_update_draft_pr(
                    token=github_token,
                    slug=repo,
                    branch=spec.branch,
                    base=trigger.default_branch,
                    title=build_pr_title(trigger.title, trigger.issue_number),
                    body=redact_hosts(report_text, redactions),
                    pr_number=spec.pr_number,
                )
                summary["pr_number"] = pr.get("number") if isinstance(pr, dict) else None
                summary["pr_url"] = pr.get("html_url") if isinstance(pr, dict) else None
            else:
                # The Science lane never mutates a branch, so it publishes no PR:
                # its result package travels as Actions artifacts.
                summary["result_dir"] = str(result_dir)
            summary["published"] = True
            set_issue_labels(
                token=github_token, slug=repo, issue_number=trigger.issue_number, add=[REVIEW_LABEL]
            )
        elif classification != CLASS_OK and publish and github_token:
            set_issue_labels(
                token=github_token, slug=repo, issue_number=trigger.issue_number, add=[BLOCKED_LABEL]
            )
            # A locked Issue rejects comments.  The label transition is the
            # authoritative state change, so a failed comment is recorded as a
            # warning instead of failing an already-completed dispatch.
            try:
                comment_on_issue(
                    token=github_token,
                    slug=repo,
                    issue_number=trigger.issue_number,
                    body=redact_hosts(
                        f"CB16 dispatch classified this run as `{classification}`.\n\n```\n{report_text[:2000]}\n```",
                        redactions,
                    ),
                )
            except DispatchError as exc:
                warnings.append(f"could not comment on Issue #{trigger.issue_number}: {exc.message}")

        evidence = write_evidence(
            report_dir,
            payload=summary,
            build_report=report_text,
            logs={
                "lane_stdout": run_result.stdout,
                "lane_stderr": run_result.stderr,
                "tests_stdout": (test_result.stdout if test_result else ""),
                "tests_stderr": (test_result.stderr if test_result else ""),
            },
            redactions=redactions,
        )
        exit_code = {
            CLASS_OK: EXIT_OK,
            CLASS_CONTRACT_MISMATCH: EXIT_CONTRACT_MISMATCH,
            CLASS_EXECUTION_BLOCKED: EXIT_EXECUTION_BLOCKED,
            CLASS_HARDWARE_LIMIT: EXIT_HARDWARE_LIMIT,
            CLASS_SCIENTIFIC_FAIL: EXIT_SCIENTIFIC_FAIL,
            CLASS_EVIDENCE_INSUFFICIENT: EXIT_EVIDENCE_INSUFFICIENT,
            CLASS_BUILDER_FAIL: EXIT_BUILDER_FAIL,
        }[classification]
        return DispatchOutcome(
            classification=classification,
            exit_code=exit_code,
            build_report=report_text,
            summary=summary,
            evidence=evidence,
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CB16-R12 deterministic OCI dispatcher")
    parser.add_argument("--lane", required=True, choices=[LANE_BUILDER, LANE_SCIENCE])
    parser.add_argument("--event", required=True, type=Path, help="GitHub event payload JSON")
    parser.add_argument("--repo-dir", type=Path, default=Path(os.environ.get("GITHUB_WORKSPACE", ".")))
    parser.add_argument(
        "--work-root",
        type=Path,
        default=env_path("CB16_WORK_ROOT", Path.home() / "cb16-worktrees"),
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=env_path("CB16_STATE_DIR", Path.home() / ".cb16" / "state"),
    )
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config" / "cb16_science_allowlist.json",
    )
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config" / "cb16_data_manifest.json",
    )
    parser.add_argument("--repo", default=os.environ.get("CB16_REPO", DEFAULT_REPO))
    parser.add_argument(
        "--repo-url",
        default=os.environ.get("CB16_REPO_URL") or None,
        help="trusted repository URL; when set the dispatcher keeps its own clone",
    )
    parser.add_argument("--repo-clone-dir", type=Path, default=None)
    parser.add_argument("--trusted-actors", default=os.environ.get("CB16_TRUSTED_ACTORS", "GY-Bai"))
    parser.add_argument("--dsh-bin", default=os.environ.get("CB16_DSH_BIN", "dsh"))
    parser.add_argument("--dsh-timeout", type=float, default=float(os.environ.get("CB16_DSH_TIMEOUT", "3600")))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--token-env", default="CB16_GITHUB_TOKEN")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report_dir = args.report_dir or (args.state_dir / "report")
    trusted = [a.strip() for a in args.trusted_actors.split(",") if a.strip()]
    # The token is read even without --publish because the dispatcher's own
    # clone may need it to fetch; only the publish path may use it beyond that.
    token = os.environ.get(args.token_env) or None

    summary: Dict[str, Any] = {
        "schema": "cb16.dispatch.v1",
        "lane": args.lane,
        "classification": CLASS_EXECUTION_BLOCKED,
        "dry_run": bool(args.dry_run),
    }
    try:
        outcome = dispatch(
            lane=args.lane,
            event=load_event(args.event),
            repo_dir=args.repo_dir.resolve(),
            work_root=args.work_root.resolve(),
            state_dir=args.state_dir.resolve(),
            report_dir=report_dir.resolve(),
            allowlist_path=args.allowlist.resolve(),
            data_manifest_path=args.data_manifest.resolve(),
            repo=args.repo,
            repo_url=args.repo_url,
            repo_clone_dir=args.repo_clone_dir,
            trusted_actors=trusted,
            dry_run=args.dry_run,
            publish=args.publish,
            github_token=token,
            dsh_bin=args.dsh_bin,
            dsh_timeout=args.dsh_timeout,
        )
    except DispatchError as exc:
        report_dir.mkdir(parents=True, exist_ok=True)
        summary.update({"classification": exc.classification, "detail": exc.detail})
        (report_dir / "dispatch_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        terms = redaction_terms(os.environ)
        print(f"classification: {exc.classification}", file=sys.stderr)
        print(f"detail: {redact_hosts(exc.detail, terms)}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_EXECUTION_BLOCKED
    except Exception as exc:  # noqa: BLE001 - last line of defence for privacy
        # An unexpected exception must not print an unredacted traceback into a
        # world-readable log, so it is captured, scrubbed and reported like any
        # other execution blocker.
        import traceback

        terms = redaction_terms(os.environ)
        detail = redact_hosts(traceback.format_exc(), terms)
        summary.update({"classification": CLASS_EXECUTION_BLOCKED, "detail": detail})
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "dispatch_summary.json").write_text(
            redact_hosts(json.dumps(summary, indent=2, sort_keys=True), terms) + "\n",
            encoding="utf-8",
        )
        print(f"classification: {CLASS_EXECUTION_BLOCKED}", file=sys.stderr)
        print(f"detail: {detail}", file=sys.stderr)
        return EXIT_EXECUTION_BLOCKED

    print(json.dumps(outcome.summary, indent=2, sort_keys=True))
    print(f"classification: {outcome.classification}")
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
