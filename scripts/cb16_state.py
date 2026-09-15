#!/usr/bin/env python3
"""Reclaim host state that accumulates across dispatches.

Every long-lived path on this host was created for a reason, so this tool is not
a sweep: it knows a short list of things that are provably dead, reports them by
default, and removes them only when asked.

Nothing here runs inside a dispatch. A dispatch must never delete state a
concurrent dispatch may be using, so reclamation is an operator action.

Never touched, by design:

* DSH session logs - they are evidence, they are small next to the caches, and
  deleting them is an operator decision rather than a maintenance default;
* ``<state>/reports`` - the per-branch BUILD_REPORT that the next round reads;
* the shared uv cache contents - ``uv cache prune`` knows which entries are
  still referenced and this tool does not;
* any worktree while a dispatch holds a task lock.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _run(argv: List[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=120
    )


def _du(path: Path) -> int:
    """Apparent bytes under a path.

    Shelling out to `du` rather than walking in Python: a worktree holds a
    virtualenv with tens of thousands of files, and the walk is slow enough that
    the tool looked hung. `-s -b` is the apparent size in bytes, and `-x` keeps
    the walk on one filesystem.
    """

    proc = _run(["du", "-sxb", str(path)])
    if proc.returncode != 0:
        return 0
    try:
        return int(proc.stdout.split()[0])
    except (IndexError, ValueError):
        return 0


def _human(size: int) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if size < 1024 or unit == "T":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}T"


def live_locks(state_dir: Path) -> List[Path]:
    """Lock files whose flock is actually held right now.

    The file surviving a crash means nothing: the lock is a kernel flock, so it
    is released when the owning process dies. Only a held flock means a dispatch
    is running, which is why the file's presence is never treated as liveness.
    """

    held: List[Path] = []
    for path in sorted((state_dir / "locks").glob("*.lock")):
        try:
            handle = path.open("r")
        except OSError:
            continue
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            held.append(path)
        finally:
            handle.close()
    return held


def inspect(state_dir: Path, work_root: Path, repo: Path | None) -> Dict[str, Any]:
    locks = sorted((state_dir / "locks").glob("*.lock"))
    held = live_locks(state_dir)
    dead_locks = [p for p in locks if p not in held]

    affinity = sorted((state_dir / "session-affinity").glob("*.json"))
    reports = sorted((state_dir / "reports").glob("*.md"))

    worktrees: List[Dict[str, Any]] = []
    if work_root.is_dir():
        for path in sorted(work_root.iterdir()):
            if not path.is_dir() or not (path / ".git").exists():
                continue
            size = _du(path)
            branch = ""
            proc = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=path)
            if proc.returncode == 0:
                branch = proc.stdout.strip()
            # The dispatcher's own clone lives at <work-root>/repo and is on the
            # default branch, so an ancestry test alone would happily delete it.
            # It is infrastructure, not a task worktree.
            if repo is not None and path.resolve() == repo.resolve():
                worktrees.append({
                    "path": str(path), "branch": branch, "bytes": size,
                    "merged": False, "protected": "the dispatcher clone",
                })
                continue
            merged = False
            if branch and repo is not None and (repo / ".git").exists():
                merged = (
                    _run(["git", "merge-base", "--is-ancestor", branch, "origin/main"], cwd=repo)
                    .returncode
                    == 0
                )
            worktrees.append({
                "path": str(path), "branch": branch, "bytes": size,
                "merged": merged and branch != "main",
                "protected": "" if branch != "main" else "the default branch",
            })

    sessions = Path.home() / ".dsh" / "sessions"
    session_bytes = _du(sessions) if sessions.is_dir() else 0
    session_count = len([p for p in sessions.iterdir() if p.is_dir()]) if sessions.is_dir() else 0

    return {
        "worktrees": worktrees,
        "dead_locks": [str(p) for p in dead_locks],
        "live_locks": [str(p) for p in held],
        "affinity_records": [str(p) for p in affinity],
        "branch_reports": [str(p) for p in reports],
        "session_logs": {"path": str(sessions), "count": session_count, "bytes": session_bytes},
    }


def render(report: Dict[str, Any]) -> None:
    print("== reclaimable, verified dead ==")
    dead_bytes = 0
    for lock in report["dead_locks"]:
        size = os.path.getsize(lock)
        dead_bytes += size
        print(f"  lock        {lock}  ({_human(size)})")
    for record in report["affinity_records"]:
        size = os.path.getsize(record)
        dead_bytes += size
        print(f"  affinity    {record}  ({_human(size)})")
    print(f"  subtotal: {_human(dead_bytes)}")

    print()
    print("== worktrees ==")
    if report["live_locks"]:
        print("  (a dispatch is running; worktrees are reported but will not be removed)")
    for wt in report["worktrees"]:
        if wt.get("protected"):
            mark = "KEEP  "
        else:
            mark = "merged" if wt["merged"] else "open  "
        note = f"  <- {wt['protected']}" if wt.get("protected") else ""
        print(f"  [{mark}] {wt['branch'] or '(detached)':<44} {_human(wt['bytes']):>7}{note}")
    merged_bytes = sum(w["bytes"] for w in report["worktrees"] if w["merged"])
    print(f"  merged subtotal: {_human(merged_bytes)}")

    print()
    print("== reported only, never removed here ==")
    print(f"  branch reports  {len(report['branch_reports'])} files (the cross-round handoff)")
    logs = report["session_logs"]
    print(f"  session logs    {logs['count']} sessions, {_human(logs['bytes'])} at {logs['path']}")
    print("  shared uv cache run `uv cache prune` - it knows which entries are still referenced")


def reap(report: Dict[str, Any], repo: Path | None) -> int:
    freed = 0
    if report["live_locks"]:
        print("refusing to remove worktrees while a dispatch holds a lock", file=sys.stderr)
    for lock in report["dead_locks"]:
        try:
            freed += os.path.getsize(lock)
            os.remove(lock)
            print(f"removed lock      {lock}")
        except OSError as exc:
            print(f"could not remove {lock}: {exc}", file=sys.stderr)
    for record in report["affinity_records"]:
        try:
            freed += os.path.getsize(record)
            os.remove(record)
            print(f"removed affinity  {record}")
        except OSError as exc:
            print(f"could not remove {record}: {exc}", file=sys.stderr)

    if report["live_locks"]:
        return freed
    for wt in report["worktrees"]:
        if not wt["merged"]:
            continue
        if wt.get("protected"):
            continue
        path = Path(wt["path"])
        if repo is not None:
            proc = _run(["git", "worktree", "remove", "--force", str(path)], cwd=repo)
            if proc.returncode != 0:
                print(f"could not remove {path}: {proc.stderr.strip()}", file=sys.stderr)
                continue
        else:
            shutil.rmtree(path, ignore_errors=True)
        freed += wt["bytes"]
        print(f"removed worktree  {path}  ({_human(wt['bytes'])})")
    if repo is not None:
        _run(["git", "worktree", "prune"], cwd=repo)
    return freed


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report or reclaim dispatch host state.")
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--repo", type=Path, default=None, help="dispatcher clone, for merge checks")
    parser.add_argument("--apply", action="store_true", help="actually remove (default: report only)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    state_dir = args.state_dir or Path(os.environ.get("CB16_STATE_DIR", ""))
    work_root = args.work_root or Path(os.environ.get("CB16_WORK_ROOT", ""))
    repo = args.repo
    if not state_dir or not work_root:
        parser.error("--state-dir and --work-root are required (or CB16_STATE_DIR/CB16_WORK_ROOT)")

    report = inspect(state_dir, work_root, repo)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        render(report)
    if args.apply:
        freed = reap(report, repo)
        print(f"\nfreed about {_human(freed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
