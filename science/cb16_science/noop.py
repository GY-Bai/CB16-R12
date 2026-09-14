"""Deterministic no-op entrypoint used to smoke-test the Science lane.

It writes ``RESULT.json`` and ``REPORT.md`` into ``$CB16_RESULT_DIR`` and
touches no scientific data, thresholds, or repository files.  It is marked
``dry_run_only`` in the allowlist so a formal qualification run cannot use it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> int:
    result_dir = Path(os.environ["CB16_RESULT_DIR"])
    result_dir.mkdir(parents=True, exist_ok=True)
    commit_sha = os.environ.get("CB16_COMMIT_SHA", "unknown")
    result_command = os.environ.get("CB16_RESULT_COMMAND", "unknown")

    payload = {
        "schema": "cb16.result.v1",
        "status": "DRY_RUN",
        "commit_sha": commit_sha,
        "result_command": result_command,
        "metrics": {"noop": 0},
    }
    (result_dir / "RESULT.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (result_dir / "REPORT.md").write_text(
        "# Science lane dry run\n\n"
        f"- commit: `{commit_sha}`\n"
        f"- entrypoint: `{result_command}`\n"
        "- classification: DRY_RUN (no scientific work performed)\n",
        encoding="utf-8",
    )
    print("noop science entrypoint complete")
    print("BUILD_REPORT: dry-run artifacts written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
