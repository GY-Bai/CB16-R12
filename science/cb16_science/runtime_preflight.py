"""Runtime preflight for the CB16-R12 Science lane.

The allowlisted ``cb16.runtime-preflight@v1`` entrypoint reports only bounded,
non-secret facts:

* Python version;
* NumPy installed/version/import success;
* PyTorch installed/version, CPU import success and CUDA availability as a
  boolean diagnostic (a GPU is not required);
* existence/readability booleans for the canonical ``/cb16/raw``,
  ``/cb16/frozen`` and ``/cb16/brain_assets`` roots and for every input the
  repository data manifest advertises;
* the discovered BTCUSDT ``2020-01..2020-03`` archive count and basenames;
* whether at least one BTCUSDT archive opens and yields one valid OHLCV row.

It never dumps the environment, home-directory inventories, credential paths or
model contents, and it writes only ``RESULT.json`` and ``REPORT.md`` into
``$CB16_RESULT_DIR``.  Literal paths are reported only for canonical ``/cb16``
mounts; every other manifest input is reported as booleans.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .normalization import data as market_data
from .normalization.contract import INTERVAL_MONTHS, SYMBOL

SCHEMA = "cb16.runtime.preflight.v1"
RESULT_COMMAND = "cb16.runtime-preflight@v1"
RESULT_FILENAME = "RESULT.json"
REPORT_FILENAME = "REPORT.md"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "config" / "cb16_data_manifest.json"

CANONICAL_ROOTS: Tuple[Tuple[str, str], ...] = (
    ("raw", "/cb16/raw"),
    ("frozen", "/cb16/frozen"),
    ("brain_assets", "/cb16/brain_assets"),
)

#: Bounded look at one archive: enough to prove a valid OHLCV row can be parsed
#: without reading an arbitrary amount of market data.
MAX_PROBE_ROWS = 64

EXIT_OK = 0


def _entry_exists_readable(path: Path) -> Dict[str, Any]:
    return {
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "readable": os.access(path, os.R_OK),
    }


def _canonical_path(path: Path) -> Optional[str]:
    text = str(path)
    return text if text.startswith("/cb16/") else None


def _guard(operation: Callable[[], Any], default: Any) -> Tuple[Any, Optional[str]]:
    """Run a section so one failure cannot suppress the rest of the report."""

    try:
        return operation(), None
    except Exception as exc:  # noqa: BLE001 - a preflight reports, it does not fail
        return default, type(exc).__name__


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def python_section() -> Dict[str, Any]:
    version = platform.python_version_tuple()
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "major": int(version[0]),
        "minor": int(version[1]),
        "patch": int(version[2]),
    }


def numpy_section() -> Dict[str, Any]:
    try:
        module = importlib.import_module("numpy")
    except Exception as exc:  # noqa: BLE001 - report the failure class only
        return {
            "installed": False,
            "version": None,
            "import_ok": False,
            "import_error_class": type(exc).__name__,
        }
    return {
        "installed": True,
        "version": str(getattr(module, "__version__", "unknown")),
        "import_ok": True,
        "import_error_class": None,
    }


def torch_section() -> Dict[str, Any]:
    try:
        module = importlib.import_module("torch")
    except Exception as exc:  # noqa: BLE001 - CPU-only hosts and no-GPU hosts are valid
        return {
            "installed": False,
            "version": None,
            "cpu_import_ok": False,
            "cuda_available": False,
            "import_error_class": type(exc).__name__,
        }
    cuda_available = False
    try:
        cuda_available = bool(module.cuda.is_available())
    except Exception:  # noqa: BLE001 - a broken CUDA probe is not a hard failure
        cuda_available = False
    return {
        "installed": True,
        "version": str(getattr(module, "__version__", "unknown")),
        "cpu_import_ok": True,
        "cuda_available": cuda_available,
        "import_error_class": None,
    }


def canonical_roots_section() -> List[Dict[str, Any]]:
    return [
        dict({"label": label, "path": path}, **_entry_exists_readable(Path(path)))
        for label, path in CANONICAL_ROOTS
    ]


def resolve_manifest(env: Mapping[str, str]) -> Tuple[Optional[Path], str]:
    configured = (env.get("CB16_DATA_MANIFEST") or "").strip()
    if configured and Path(configured).is_file():
        return Path(configured), "env:CB16_DATA_MANIFEST"
    if DEFAULT_MANIFEST_PATH.is_file():
        return DEFAULT_MANIFEST_PATH, "repo_default"
    return None, "absent"


def manifest_inputs_section(env: Mapping[str, str]) -> Dict[str, Any]:
    """Per-input existence/readability booleans; literal paths only under /cb16."""

    manifest_path, source = resolve_manifest(env)
    section: Dict[str, Any] = {
        "manifest_found": manifest_path is not None,
        "manifest_source": source,
        "inputs": [],
    }
    if manifest_path is None:
        return section
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("read_only")
    if not isinstance(entries, dict):
        raise ValueError("data manifest has no 'read_only' object")
    inputs: List[Dict[str, Any]] = []
    for logical_name, spec in sorted(entries.items()):
        target = (spec or {}).get("path") if isinstance(spec, dict) else None
        if not isinstance(target, str) or not target:
            inputs.append({"logical_name": logical_name, "declared_path": False})
            continue
        entry = {"logical_name": logical_name, "declared_path": True}
        entry.update(_entry_exists_readable(Path(target)))
        entry["canonical_path"] = _canonical_path(Path(target))
        inputs.append(entry)
    section["inputs"] = inputs
    return section


@dataclass(frozen=True)
class ArchiveProbe:
    attempted_basename: Optional[str]
    opened: bool
    member_is_csv: bool
    header_detected: Optional[bool]
    rows_inspected: int
    valid_ohlcv_row_parsed: bool
    error_class: Optional[str]


def probe_archives(root: Path, symbol: str, months: Sequence[str]) -> Dict[str, Any]:
    archives, missing = market_data.discover_archives(root, symbol, months)
    section: Dict[str, Any] = {
        "symbol": symbol,
        "months": list(months),
        "archive_count": len(archives),
        "basenames": [path.name for path in archives],
        "missing_months": missing,
        "root": market_data.describe_root(root),
    }
    if not archives:
        section["probe"] = None
        return section

    for path in archives:
        try:
            times, values, info = market_data.read_archive(path)
        except market_data.DataUnavailable as exc:
            section["probe"] = {
                "attempted_basename": path.name,
                "opened": False,
                "member_is_csv": False,
                "header_detected": None,
                "rows_inspected": 0,
                "valid_ohlcv_row_parsed": False,
                "error_class": type(exc).__name__,
            }
            continue
        rows = min(MAX_PROBE_ROWS, values.shape[0])
        valid, _ = market_data.validate_ohlcv(values[:rows])
        section["probe"] = {
            "attempted_basename": path.name,
            "opened": True,
            "member_is_csv": info.member_name.lower().endswith(".csv"),
            "header_detected": info.header_detected,
            "rows_inspected": int(rows),
            "valid_ohlcv_row_parsed": bool(valid.any()),
            "error_class": None,
            "first_open_time_ms": int(times[0]) if times.shape[0] else None,
        }
        return section
    return section


# ---------------------------------------------------------------------------
# Payload and report
# ---------------------------------------------------------------------------


def build_payload(env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    environment = dict(env if env is not None else os.environ)
    commit_sha = (environment.get("CB16_COMMIT_SHA") or "").strip() or "not-supplied"
    klines_root, root_source = market_data.resolve_klines_root(SYMBOL, env=environment)

    notes: List[str] = []
    python_info, python_error = _guard(python_section, {})
    numpy_info, numpy_error = _guard(numpy_section, {})
    torch_info, torch_error = _guard(torch_section, {})
    roots, roots_error = _guard(canonical_roots_section, [])
    manifest, manifest_error = _guard(
        lambda: manifest_inputs_section(environment),
        {"manifest_found": False, "manifest_source": "unreadable", "inputs": []},
    )
    archives, archives_error = _guard(
        lambda: probe_archives(klines_root, SYMBOL, INTERVAL_MONTHS), {}
    )
    for label, error in (
        ("python", python_error),
        ("numpy", numpy_error),
        ("torch", torch_error),
        ("canonical_roots", roots_error),
        ("manifest", manifest_error),
        ("btcusdt_archives", archives_error),
    ):
        if error:
            notes.append(f"{label} section failed with {error}")

    return {
        "schema": SCHEMA,
        "command": RESULT_COMMAND,
        "status": "OK",
        "diagnostic_only": True,
        "commit_sha": commit_sha,
        "python": python_info,
        "numpy": numpy_info,
        "torch": torch_info,
        "canonical_roots": roots,
        "manifest_inputs": manifest,
        "btcusdt_archives": archives,
        "klines_root_source": root_source,
        "notes": notes,
    }


def render_report(payload: Mapping[str, Any]) -> str:
    numpy_info = payload.get("numpy") or {}
    torch_info = payload.get("torch") or {}
    archives = payload.get("btcusdt_archives") or {}
    probe = archives.get("probe") or {}
    manifest = payload.get("manifest_inputs") or {}

    lines: List[str] = [
        "# CB16 Runtime Preflight — REPORT",
        "",
        "Bounded, non-secret environment diagnostics for the Science lane.  This "
        "run trains nothing and reads no model weights.",
        "",
        "## Runtime",
        "",
        "| Component | Fact |",
        "| --- | --- |",
        f"| Python | `{(payload.get('python') or {}).get('implementation')} "
        f"{(payload.get('python') or {}).get('version')}` |",
        f"| NumPy | installed `{numpy_info.get('installed')}`, "
        f"version `{numpy_info.get('version')}`, import ok `{numpy_info.get('import_ok')}` |",
        f"| PyTorch | installed `{torch_info.get('installed')}`, "
        f"version `{torch_info.get('version')}`, CPU import ok "
        f"`{torch_info.get('cpu_import_ok')}` |",
        f"| CUDA available (diagnostic only) | `{torch_info.get('cuda_available')}` |",
        "",
        "A GPU is not required.  A missing PyTorch is reported, not fatal.",
        "",
        "## Canonical mounted inputs",
        "",
        "| Root | exists | is_dir | readable |",
        "| --- | --- | --- | --- |",
    ]
    for entry in payload.get("canonical_roots") or []:
        lines.append(
            f"| `{entry.get('path')}` | `{entry.get('exists')}` | "
            f"`{entry.get('is_dir')}` | `{entry.get('readable')}` |"
        )
    if not payload.get("canonical_roots"):
        lines.append("| (none probed) | | | |")

    lines += [
        "",
        f"Manifest advertised inputs (source `{manifest.get('manifest_source')}`, "
        f"found `{manifest.get('manifest_found')}`):",
        "",
        "| logical input | exists | is_dir | readable | canonical path |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in manifest.get("inputs") or []:
        lines.append(
            f"| `{entry.get('logical_name')}` | `{entry.get('exists')}` | "
            f"`{entry.get('is_dir')}` | `{entry.get('readable')}` | "
            f"`{entry.get('canonical_path')}` |"
        )
    if not manifest.get("inputs"):
        lines.append("| (none) | | | | |")

    lines += [
        "",
        "Only canonical `/cb16` paths are printed; other manifest inputs are "
        "reported as booleans so no host home-directory path reaches an artifact.",
        "",
        "## BTCUSDT archive probe",
        "",
        f"- symbol: `{archives.get('symbol')}`; months: "
        f"`{', '.join(archives.get('months') or [])}`",
        f"- discovered archives: `{archives.get('archive_count')}`; "
        f"basenames: {', '.join('`' + name + '`' for name in archives.get('basenames') or []) or '(none)'}",
        f"- missing months: `{archives.get('missing_months')}`",
        f"- archive open probe: opened `{probe.get('opened')}`, CSV member "
        f"`{probe.get('member_is_csv')}`, header detected "
        f"`{probe.get('header_detected')}`, rows inspected "
        f"`{probe.get('rows_inspected')}`, valid OHLCV row parsed "
        f"`{probe.get('valid_ohlcv_row_parsed')}`",
        "",
    ]
    if payload.get("notes"):
        lines += ["## Notes", ""]
        lines += [f"- {note}" for note in payload["notes"]]
        lines += [""]
    return "\n".join(lines)


def summary_line(payload: Mapping[str, Any]) -> str:
    numpy_info = payload.get("numpy") or {}
    torch_info = payload.get("torch") or {}
    archives = payload.get("btcusdt_archives") or {}
    probe = archives.get("probe") or {}
    return (
        f"{RESULT_COMMAND}: python={(payload.get('python') or {}).get('version')} "
        f"numpy={numpy_info.get('version') or 'missing'} "
        f"torch={torch_info.get('version') or 'missing'} "
        f"cuda_available={torch_info.get('cuda_available')} "
        f"btcusdt_archives={archives.get('archive_count')} "
        f"archive_open_ok={probe.get('opened')} "
        f"valid_row_parsed={probe.get('valid_ohlcv_row_parsed')}"
    )


def main() -> int:
    result_dir = Path(os.environ["CB16_RESULT_DIR"])
    payload = build_payload()
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / RESULT_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (result_dir / REPORT_FILENAME).write_text(render_report(payload), encoding="utf-8")
    print(summary_line(payload))
    print(f"BUILD_REPORT: {RESULT_COMMAND} diagnostics written; no secret values emitted")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
