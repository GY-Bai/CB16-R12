"""Shared, non-collected support for the normalization/preflight test suites.

Loading by path keeps the suites independent of how ``unittest discover`` was
invoked (``-t .`` imports ``tests.*`` modules, a bare ``-s tests`` does not).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SCIENCE_DIR = REPO_ROOT / "science"
FIXTURES_PATH = Path(__file__).resolve().parent / "cb16_normalization_fixtures.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fixtures = load_module("cb16_normalization_fixtures", FIXTURES_PATH)


def entrypoint_env(
    result_dir: Path,
    *,
    data_root: Optional[Path] = None,
    extra: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """A minimal environment mirroring how the dispatcher invokes an entrypoint."""

    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(SCIENCE_DIR),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CB16_RESULT_DIR": str(result_dir),
    }
    if data_root is not None:
        env["CB16_DATA_ROOT"] = str(data_root)
    env.update(dict(extra or {}))
    return env
