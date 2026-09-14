"""Diagnostic-only infra boundary probe for the CB16-R12 Science lane.

This entrypoint measures three execution-boundary facts and nothing else:

1. whether a Science-lane child may create a file directly under the task
   worktree but outside the designated result directory;
2. whether it may create a file under ``$CB16_STORE/.cb16_boundary_probe/``;
3. whether its own scrubbed environment carries ``CB16_GITHUB_TOKEN`` and
   whether any readable numeric ``/proc/<pid>/environ`` still exposes the
   literal key marker ``CB16_GITHUB_TOKEN=``.

It is deliberately not scientific: it touches no market data, no thresholds, no
model weights and no repository content, and it applies no host-side change.
It never prints, persists, hashes or copies a credential value or any unrelated
environment content -- the ``/proc`` scan reports booleans and counts only.  It
never invokes DSH, never uses a shell, never follows a non-numeric ``/proc``
entry, bounds how many files and how many bytes it reads, and needs no network.

The dispatcher runs this entrypoint with the task worktree as its working
directory, so the worktree under test is ``Path.cwd()``.
"""

from __future__ import annotations

import errno
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

SCHEMA = "cb16.infra.boundary_probe.v1"
RESULT_COMMAND = "cb16.boundary-probe@v1"

#: Environment key the dispatcher parent holds and the Science child must not.
GITHUB_TOKEN_KEY = "CB16_GITHUB_TOKEN"
#: Literal key marker searched for in ``/proc/<pid>/environ``.  Only presence
#: is computed; no value is ever read out, kept, hashed or printed.
GITHUB_TOKEN_MARKER = b"CB16_GITHUB_TOKEN="

#: Dedicated namespace for the store probe.  Nothing else in the store is
#: inspected or modified.
PROBE_DIR_NAME = ".cb16_boundary_probe"
PROBE_FILE_PREFIX = ".cb16_boundary_probe_"
PROBE_PAYLOAD = b"cb16 boundary probe scratch file; safe to delete\n"

RESULT_FILENAME = "RESULT.json"
REPORT_FILENAME = "REPORT.md"

DEFAULT_PROC_ROOT = Path("/proc")
#: Bounds so a hostile or huge process table cannot turn the scan into a
#: resource problem: at most this many environ files, at most this many bytes
#: read from each.
MAX_PROC_ENTRIES = 4096
MAX_ENVIRON_BYTES = 256 * 1024

WRITE_BLOCKED = "BLOCKED"
WRITE_ALLOWED = "ALLOWED"
WRITE_NOT_PRESENT = "NOT_PRESENT"
WRITE_ERROR = "ERROR"

PROC_OK = "OK"
PROC_DENIED = "DENIED"
PROC_ERROR = "ERROR"

WRITE_STATUSES = (WRITE_BLOCKED, WRITE_ALLOWED, WRITE_NOT_PRESENT, WRITE_ERROR)
PROC_STATUSES = (PROC_OK, PROC_DENIED, PROC_ERROR)

#: Errnos that mean the boundary refused the operation, as opposed to the probe
#: itself malfunctioning.
BLOCKING_ERRNOS = frozenset({errno.EACCES, errno.EPERM, errno.EROFS})

EXIT_OK = 0
EXIT_CREDENTIAL_BOUNDARY_VIOLATION = 3


# --------------------------------------------------------------------------
# Probe results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WriteProbeResult:
    """Outcome of one bounded write attempt."""

    status: str
    cleaned_up: bool


@dataclass(frozen=True)
class ProcScanResult:
    """Counts only: no PID, no environment content, no credential value."""

    status: str
    readable_environments: int
    token_key_matches: int
    denied_environments: int


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def classify_write_error(exc: OSError) -> str:
    """Map an OS error to ``BLOCKED`` (refused) or ``ERROR`` (unexpected)."""

    return WRITE_BLOCKED if exc.errno in BLOCKING_ERRNOS else WRITE_ERROR


def remove_probe_file(path: Path) -> bool:
    """Delete a probe file we may have created; True when no residue remains."""

    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return not path.exists()


def unique_probe_path(directory: Path) -> Path:
    """A fresh probe name.  The pid and a random suffix keep it collision-free."""

    return directory / f"{PROBE_FILE_PREFIX}{os.getpid()}_{uuid.uuid4().hex}.tmp"


def probe_file_write(target: Path) -> WriteProbeResult:
    """Create ``target`` exclusively, write a fixed payload, then delete it.

    Creation is attempted with ``O_EXCL`` semantics (mode ``"xb"``) so an
    unrelated pre-existing file is never truncated or removed.  Cleanup runs on
    every path where creation may have succeeded.
    """

    try:
        with open(target, "xb") as handle:
            handle.write(PROBE_PAYLOAD)
    except FileExistsError:
        # The name belongs to someone else; leave it untouched.
        return WriteProbeResult(WRITE_ERROR, True)
    except OSError as exc:
        return WriteProbeResult(classify_write_error(exc), remove_probe_file(target))
    return WriteProbeResult(WRITE_ALLOWED, remove_probe_file(target))


# --------------------------------------------------------------------------
# Probe 1: worktree write boundary
# --------------------------------------------------------------------------


def probe_worktree_write(worktree: Path) -> WriteProbeResult:
    """Attempt one uniquely named temporary file directly under ``worktree``."""

    return probe_file_write(unique_probe_path(worktree))


# --------------------------------------------------------------------------
# Probe 2: project-store write boundary
# --------------------------------------------------------------------------


def probe_store_write(store: Optional[str]) -> WriteProbeResult:
    """Attempt one tiny file under ``$CB16_STORE/.cb16_boundary_probe/``.

    Only that dedicated directory is created, used and (when empty) removed.
    Pre-existing store content is never listed, read or modified, and an absent
    store is reported as ``NOT_PRESENT`` rather than as a failure.
    """

    value = (store or "").strip()
    if not value:
        return WriteProbeResult(WRITE_NOT_PRESENT, True)
    root = Path(value)
    if not root.is_dir():
        # The variable is set but the store is not present here; nothing was
        # created and absence is not treated as a failure.
        return WriteProbeResult(WRITE_NOT_PRESENT, True)

    probe_dir = root / PROBE_DIR_NAME
    try:
        probe_dir.mkdir(mode=0o700, exist_ok=True)
    except OSError as exc:
        return WriteProbeResult(classify_write_error(exc), True)

    result = probe_file_write(unique_probe_path(probe_dir))
    try:
        # Removes the dedicated directory only when it is empty; a directory
        # another probe is still using is left in place.
        probe_dir.rmdir()
    except OSError:
        pass
    return result


# --------------------------------------------------------------------------
# Probe 3: parent/peer credential visibility through /proc
# --------------------------------------------------------------------------


def read_environ_bounded(path: Path, limit: int = MAX_ENVIRON_BYTES) -> bytes:
    """Read at most ``limit`` bytes of one process environment."""

    with open(path, "rb") as handle:
        return handle.read(limit)


def scan_proc_environ(
    proc_root: Path = DEFAULT_PROC_ROOT,
    *,
    max_entries: int = MAX_PROC_ENTRIES,
    max_environ_bytes: int = MAX_ENVIRON_BYTES,
    environ_reader: Callable[[Path, int], bytes] = read_environ_bounded,
) -> ProcScanResult:
    """Count readable ``/proc/<pid>/environ`` files carrying the key marker.

    Only the presence of the literal byte marker is computed.  No environment
    content, no matching PID and no credential value is returned or logged.
    Non-numeric entries (``self``, ``thread-self``, ``net``, ...) are never
    opened, and both the number of files and the bytes read per file are
    bounded.  Permission errors are tolerated and counted; if nothing at all
    was readable the scan reports ``DENIED`` instead of a match.
    """

    try:
        entries = list(os.scandir(proc_root))
    except PermissionError:
        return ProcScanResult(PROC_DENIED, 0, 0, 0)
    except OSError:
        return ProcScanResult(PROC_ERROR, 0, 0, 0)

    pids = []
    for entry in entries:
        if not entry.name.isdigit():
            continue  # never follow a non-numeric /proc entry
        try:
            if not entry.is_dir(follow_symlinks=False):
                continue  # never follow a symlinked pseudo-entry
        except OSError:
            continue
        pids.append(entry.name)
    # Deterministic order, then the bound on how many files may be read.
    pids = sorted(pids, key=int)[: max(0, max_entries)]

    readable = matches = denied = 0
    for pid in pids:
        try:
            data = environ_reader(proc_root / pid / "environ", max_environ_bytes)
        except PermissionError:
            denied += 1
            continue
        except OSError:
            # A process that exited between listing and reading, or any other
            # per-process error, is tolerated and not counted as readable.
            continue
        readable += 1
        if GITHUB_TOKEN_MARKER in data:
            matches += 1

    status = PROC_DENIED if readable == 0 and denied > 0 else PROC_OK
    return ProcScanResult(status, readable, matches, denied)


# --------------------------------------------------------------------------
# Result assembly and report
# --------------------------------------------------------------------------


def build_payload(
    *,
    worktree: Path,
    store: Optional[str],
    child_env: Mapping[str, str],
    proc_root: Path = DEFAULT_PROC_ROOT,
    environ_reader: Callable[[Path, int], bytes] = read_environ_bounded,
) -> Dict[str, Any]:
    """Run the three probes once and return the deterministic RESULT payload."""

    # The child environment is checked first: the scrubbed Science environment
    # must not carry the dispatcher credential at all.
    child_has_token = GITHUB_TOKEN_KEY in child_env

    worktree_probe = probe_worktree_write(worktree)
    store_probe = probe_store_write(store)
    proc_scan = scan_proc_environ(proc_root, environ_reader=environ_reader)

    return {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "worktree_write": worktree_probe.status,
        "store_write": store_probe.status,
        "child_env_has_github_token": child_has_token,
        "parent_or_peer_proc_exposes_github_token_key": proc_scan.token_key_matches > 0,
        "proc_scan_status": proc_scan.status,
        "readable_proc_environments": proc_scan.readable_environments,
        "proc_environ_token_key_matches": proc_scan.token_key_matches,
        "denied_proc_environments": proc_scan.denied_environments,
        "worktree_probe_cleaned_up": worktree_probe.cleaned_up,
        "store_probe_cleaned_up": store_probe.cleaned_up,
    }


def _flag(value: Any) -> str:
    return "true" if value else "false"


_WORKTREE_READING = {
    WRITE_ALLOWED: (
        "A uniquely named probe file was created directly under the Science "
        "worktree, outside the designated result directory, and deleted again "
        "before exit. The worktree is therefore writable as a whole; the result "
        "directory is a convention, not a confinement boundary. The dispatcher's "
        "immutability guard (`git status` after the run) remains the only thing "
        "that turns a persistent worktree edit into a failure."
    ),
    WRITE_BLOCKED: (
        "File creation directly under the Science worktree was refused. The "
        "write boundary is narrower than the worktree, so entrypoints must keep "
        "every artefact inside the designated result directory."
    ),
    WRITE_ERROR: (
        "The write attempt failed for a reason other than a boundary refusal. "
        "The worktree boundary is unmeasured."
    ),
}

_STORE_READING = {
    WRITE_ALLOWED: (
        "The Science child could create and delete a file under the dedicated "
        "`$CB16_STORE/.cb16_boundary_probe/` directory, so the project store is "
        "writable from the Science lane. No pre-existing store content was read "
        "or modified."
    ),
    WRITE_BLOCKED: (
        "File creation under `$CB16_STORE/.cb16_boundary_probe/` was refused. If "
        "Science tasks must persist artefacts in the project store, a host-side "
        "sandbox-profile change would be required; this diagnostic did not "
        "attempt such a change."
    ),
    WRITE_NOT_PRESENT: (
        "`CB16_STORE` was absent, or did not point at an existing directory, so "
        "the store boundary was not measured. Absence is not a failure and "
        "nothing was created."
    ),
    WRITE_ERROR: (
        "The store write attempt failed for a reason other than a boundary "
        "refusal. The store boundary is unmeasured."
    ),
}


def _proc_reading(payload: Mapping[str, Any]) -> str:
    status = payload["proc_scan_status"]
    matches = payload["proc_environ_token_key_matches"]
    readable = payload["readable_proc_environments"]
    denied = payload["denied_proc_environments"]

    if status == PROC_DENIED:
        return (
            "Access to `/proc` process environments was denied: no environment "
            "file could be read, so the scan is reported as `DENIED` rather than "
            "as a match. This fact alone does not show whether the parent "
            "credential would be visible to a process that could read `/proc`."
        )
    if status == PROC_ERROR:
        return (
            "The `/proc` scan could not be performed at all (for example `/proc` "
            "is not mounted here). This is reported as `ERROR`, not as a match."
        )

    lines = [
        f"Readable `/proc/<pid>/environ` files scanned: {readable}. Denied: "
        f"{denied}. Readable files carrying the literal key marker "
        f"`CB16_GITHUB_TOKEN=`: {matches}. The probe never read out, stored, "
        "hashed or printed any value, and it does not report matching PIDs.",
    ]
    if matches:
        lines.append(
            "The key marker was present in at least one readable environment, and "
            "the probe deliberately does not record which process it belonged to. "
            "Unless the child environment check above reported a violation, that "
            "process is not the probe itself -- typically it is the dispatcher "
            "parent."
        )
    else:
        lines.append(
            "No readable process environment exposed the key marker. This is "
            "evidence about readable processes only: denied environments were "
            "not inspected."
        )
    return " ".join(lines)


def render_report(payload: Mapping[str, Any]) -> str:
    """Explain the measured facts without claiming any host-side fix."""

    child_has_token = bool(payload["child_env_has_github_token"])
    child_line = (
        "**BOUNDARY VIOLATION** -- the Science child environment carried "
        f"`{GITHUB_TOKEN_KEY}`. No value was read, recorded or printed, and the "
        "entrypoint exits non-zero so the run cannot pass silently."
        if child_has_token
        else f"The scrubbed Science child environment did not carry `{GITHUB_TOKEN_KEY}`."
    )
    cleanup_line = (
        "Every probe file created during this run was removed.\n"
        if payload["worktree_probe_cleaned_up"] and payload["store_probe_cleaned_up"]
        else "**WARNING**: at least one probe file could not be removed; see the "
        "cleanup rows above.\n"
    )

    return (
        "# CB16 Infra Boundary Probe R0 -- REPORT\n"
        "\n"
        "Diagnostic only. This run measured execution boundaries and applied no "
        "host-side fix: the OCI host, the Actions runner, `cb16-sandbox-runner`, "
        "systemd, DSH profiles, Docker, user/group membership and `/proc` mount "
        "options were all left untouched. No network access was used.\n"
        "\n"
        f"- entrypoint: `{RESULT_COMMAND}`\n"
        f"- schema: `{SCHEMA}`\n"
        "\n"
        "## Result\n"
        "\n"
        "| Field | Value |\n"
        "| --- | --- |\n"
        f"| `worktree_write` | `{payload['worktree_write']}` |\n"
        f"| `store_write` | `{payload['store_write']}` |\n"
        f"| `child_env_has_github_token` | `{_flag(child_has_token)}` |\n"
        "| `parent_or_peer_proc_exposes_github_token_key` | "
        f"`{_flag(payload['parent_or_peer_proc_exposes_github_token_key'])}` |\n"
        f"| `proc_scan_status` | `{payload['proc_scan_status']}` |\n"
        f"| readable `/proc/<pid>/environ` files | `{payload['readable_proc_environments']}` |\n"
        f"| readable files carrying `{GITHUB_TOKEN_KEY}=` | "
        f"`{payload['proc_environ_token_key_matches']}` |\n"
        f"| denied `/proc/<pid>/environ` files | `{payload['denied_proc_environments']}` |\n"
        f"| worktree probe file cleaned up | `{_flag(payload['worktree_probe_cleaned_up'])}` |\n"
        f"| store probe file cleaned up | `{_flag(payload['store_probe_cleaned_up'])}` |\n"
        "\n"
        "## 1. Worktree write boundary\n"
        "\n"
        f"{_WORKTREE_READING.get(str(payload['worktree_write']), 'Status not recognised.')}\n"
        "\n"
        "## 2. Project-store write boundary\n"
        "\n"
        f"{_STORE_READING.get(str(payload['store_write']), 'Status not recognised.')}\n"
        "\n"
        "## 3. Parent/peer credential visibility through /proc\n"
        "\n"
        f"{child_line}\n"
        "\n"
        f"{_proc_reading(payload)}\n"
        "\n"
        "## Cleanup\n"
        "\n"
        f"{cleanup_line}"
        "\n"
        "## What this run did not do\n"
        "\n"
        "- It did not modify any host configuration or any file outside the probe "
        "paths named above.\n"
        "- It did not invoke DSH, a shell, or any external command.\n"
        "- It did not read, copy, hash, persist or print a credential value, and "
        "it reports no matching PIDs.\n"
        "- It did not inspect or modify pre-existing project-store content.\n"
        "- It did not apply, and does not claim, any host-side fix.\n"
    )


def summary_line(payload: Mapping[str, Any]) -> str:
    """One path-free, secret-free stdout line for the lane log."""

    return (
        f"{RESULT_COMMAND}: worktree_write={payload['worktree_write']} "
        f"store_write={payload['store_write']} "
        f"child_env_has_github_token={_flag(payload['child_env_has_github_token'])} "
        "parent_or_peer_proc_exposes_github_token_key="
        f"{_flag(payload['parent_or_peer_proc_exposes_github_token_key'])} "
        f"proc_scan_status={payload['proc_scan_status']}"
    )


def main() -> int:
    """Run the probe and write ``RESULT.json`` and ``REPORT.md``.

    Exits non-zero only on the one hard boundary violation the probe can detect
    on its own: its own environment carrying the dispatcher credential.
    """

    result_dir = Path(os.environ["CB16_RESULT_DIR"])
    payload = build_payload(
        worktree=Path.cwd(),
        store=os.environ.get("CB16_STORE"),
        child_env=os.environ,
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / RESULT_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (result_dir / REPORT_FILENAME).write_text(render_report(payload), encoding="utf-8")
    print(summary_line(payload))
    print("BUILD_REPORT: diagnostic artifacts written; no host change attempted")
    return EXIT_CREDENTIAL_BOUNDARY_VIOLATION if payload["child_env_has_github_token"] else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
