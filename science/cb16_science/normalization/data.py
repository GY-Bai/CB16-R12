"""Read-only market-data loading and deterministic window selection.

The loader reads the canonical monthly Binance USDM 1m kline archives directly
from the mounted read-only tree with the standard library (``zipfile``/``csv``)
plus NumPy.  It never extracts, rewrites or persists anything from that tree.

Archive layout inspected at runtime (BTCUSDT, 2020-01..2020-03 on the mounted
``binance_um_1m_klines_10pairs`` tree):

* ``<root>/<SYMBOL>/<SYMBOL>-1m-<YYYY-MM>.zip`` plus a sibling ``.CHECKSUM``;
* exactly one CSV member ``<SYMBOL>-1m-<YYYY-MM>.csv`` per zip;
* the CSV has **no header row**; the first field is the open time in
  milliseconds and the first six columns are
  ``open_time, open, high, low, close, volume``.

A header row is tolerated and detected at runtime: the first row is treated as
a header when its first field is not an integer millisecond timestamp.
"""

from __future__ import annotations

import calendar
import csv
import hashlib
import io
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .contract import DEFAULT_KLINES_ROOT, PREDECESSOR_BARS, SYMBOL

MILLISECONDS_PER_MINUTE = 60_000
MINUTES_PER_DAY = 1_440
OHLCV_COLUMNS = 5
MIN_CSV_FIELDS = 6


class DataUnavailable(Exception):
    """The canonical read-only input cannot supply the frozen experiment scope."""


# ---------------------------------------------------------------------------
# Archive discovery
# ---------------------------------------------------------------------------


def month_bounds_ms(month: str) -> Tuple[int, int]:
    """Half-open ``[start, end)`` UTC millisecond bounds of ``YYYY-MM``."""

    year_text, month_text = month.split("-")
    year, month_number = int(year_text), int(month_text)
    days = calendar.monthrange(year, month_number)[1]
    start = datetime(year, month_number, 1, tzinfo=timezone.utc)
    start_ms = int(start.timestamp() * 1000)
    return start_ms, start_ms + days * MINUTES_PER_DAY * MILLISECONDS_PER_MINUTE


def minutes_in_month(month: str) -> int:
    year_text, month_text = month.split("-")
    days = calendar.monthrange(int(year_text), int(month_text))[1]
    return days * MINUTES_PER_DAY


def archive_basename(symbol: str, month: str) -> str:
    return f"{symbol}-1m-{month}.zip"


def archive_path(root: Path, symbol: str, month: str) -> Path:
    return Path(root) / symbol / archive_basename(symbol, month)


def resolve_klines_root(
    symbol: str = SYMBOL,
    *,
    explicit: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Tuple[Path, str]:
    """Resolve the klines root without ever writing to it.

    Explicit argument first (tests), then ``CB16_KLINES_ROOT``, then
    ``CB16_DATA_ROOT`` (the dispatcher advertises the first read-only manifest
    entry there), then the canonical default.  A candidate is accepted only
    when it actually contains the symbol directory, so a mis-set variable
    degrades to the canonical path instead of silently reading nothing.
    """

    if explicit is not None:
        return Path(explicit), "explicit"
    environment = dict(env or {})
    for key in ("CB16_KLINES_ROOT", "CB16_DATA_ROOT"):
        value = (environment.get(key) or "").strip()
        if value and (Path(value) / symbol).is_dir():
            return Path(value), f"env:{key}"
    return Path(DEFAULT_KLINES_ROOT), "default"


def discover_archives(
    root: Path, symbol: str, months: Sequence[str]
) -> Tuple[List[Path], List[str]]:
    """Return ``(existing archives, missing months)`` for the requested months.

    Only the symbol directory and the requested month names are inspected; the
    directory is never listed wholesale and nothing is written.
    """

    found: List[Path] = []
    missing: List[str] = []
    for month in months:
        candidate = archive_path(root, symbol, month)
        if candidate.is_file():
            found.append(candidate)
        else:
            missing.append(month)
    return found, missing


# ---------------------------------------------------------------------------
# Archive reading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchiveInfo:
    """Per-archive facts recorded as evidence."""

    month: str
    basename: str
    member_name: str
    header_detected: bool
    row_count: int
    first_open_time_ms: int
    last_open_time_ms: int
    rows_match_calendar_month: bool


def _parse_open_time(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        return int(float(text))


def _is_header_field(text: str) -> bool:
    value = text.strip()
    if not value:
        return True
    try:
        _parse_open_time(value)
    except ValueError:
        return True
    return False


def _month_from_archive_name(name: str) -> str:
    """Recover ``YYYY-MM`` from ``<SYMBOL>-<FREQ>-<YYYY-MM>.zip``."""

    stem = name[:-4] if name.lower().endswith(".zip") else name
    parts = stem.split("-")
    if len(parts) >= 4 and len(parts[-2]) == 4 and len(parts[-1]) == 2:
        if parts[-2].isdigit() and parts[-1].isdigit():
            return f"{parts[-2]}-{parts[-1]}"
    raise DataUnavailable(
        f"archive {name} does not follow the <SYMBOL>-<FREQUENCY>-<YYYY-MM>.zip layout"
    )


def read_archive(path: Path) -> Tuple[np.ndarray, np.ndarray, ArchiveInfo]:
    """Parse one monthly zip into ``(open_time_ms, ohlcv, info)``.

    ``ohlcv`` is ``float64`` with columns ``open, high, low, close, volume``.
    A malformed row fails closed with :class:`DataUnavailable` rather than
    being silently dropped.
    """

    month = _month_from_archive_name(path.name)
    try:
        with zipfile.ZipFile(path) as archive:
            members = sorted(
                name
                for name in archive.namelist()
                if name.lower().endswith(".csv") and not name.endswith("/")
            )
            if len(members) != 1:
                raise DataUnavailable(
                    f"archive {path.name} holds {len(members)} CSV members; exactly one is required"
                )
            member = members[0]
            with archive.open(member) as handle:
                rows = list(csv.reader(io.TextIOWrapper(handle, encoding="utf-8")))
    except zipfile.BadZipFile as exc:
        raise DataUnavailable(f"archive {path.name} is not a readable zip") from exc
    except OSError as exc:
        raise DataUnavailable(f"archive {path.name} cannot be read") from exc

    if not rows:
        raise DataUnavailable(f"archive {path.name} holds no rows")

    header_detected = _is_header_field(rows[0][0])
    data_rows = rows[1:] if header_detected else rows
    if not data_rows:
        raise DataUnavailable(f"archive {path.name} holds only a header row")

    times = np.empty(len(data_rows), dtype=np.int64)
    values = np.empty((len(data_rows), OHLCV_COLUMNS), dtype=np.float64)
    for index, row in enumerate(data_rows):
        if len(row) < MIN_CSV_FIELDS:
            raise DataUnavailable(
                f"archive {path.name} row {index + 1} has {len(row)} fields; at least "
                f"{MIN_CSV_FIELDS} are required"
            )
        try:
            times[index] = _parse_open_time(row[0])
            for column in range(OHLCV_COLUMNS):
                values[index, column] = float(row[column + 1])
        except ValueError as exc:
            raise DataUnavailable(
                f"archive {path.name} row {index + 1} is not numeric"
            ) from exc

    info = ArchiveInfo(
        month=month,
        basename=path.name,
        member_name=member,
        header_detected=header_detected,
        row_count=len(data_rows),
        first_open_time_ms=int(times[0]),
        last_open_time_ms=int(times[-1]),
        rows_match_calendar_month=len(data_rows) == minutes_in_month(month),
    )
    return times, values, info


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_ohlcv(ohlcv: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """Apply the frozen raw-bar validation.

    Returns a per-bar validity mask and counts of each rejection reason.
    Rejected bars are counted, never repaired.
    """

    values = np.asarray(ohlcv, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != OHLCV_COLUMNS:
        raise ValueError("ohlcv must have shape (N, 5)")

    finite = np.isfinite(values).all(axis=1)
    prices = values[:, :4]
    positive = (prices > 0.0).all(axis=1)
    open_, high, low, close = (values[:, 0], values[:, 1], values[:, 2], values[:, 3])
    ordered = (high >= np.maximum(open_, close)) & (low <= np.minimum(open_, close))
    valid = finite & positive & ordered

    reasons = {
        "non_finite_ohlcv": int((~finite).sum()),
        "non_positive_ohlc_price": int((finite & ~positive).sum()),
        "malformed_ohlc_ordering": int((finite & positive & ~ordered).sum()),
    }
    return valid, reasons


@dataclass(frozen=True)
class LoadedBars:
    """One contiguous, validated 1m series over the frozen interval."""

    open_time_ms: np.ndarray
    ohlcv: np.ndarray
    valid: np.ndarray
    invalid_reasons: Mapping[str, int]
    archives: Tuple[ArchiveInfo, ...]
    root_source: str
    interval_start_ms: int
    interval_end_ms: int
    continuous_60s_spacing: bool

    @property
    def count(self) -> int:
        return int(self.open_time_ms.shape[0])

    @property
    def valid_count(self) -> int:
        return int(self.valid.sum())

    @property
    def archive_basenames(self) -> List[str]:
        return [info.basename for info in self.archives]


def load_bars(
    root: Path,
    symbol: str = SYMBOL,
    months: Sequence[str] = (),
    *,
    root_source: str = "explicit",
) -> LoadedBars:
    """Load, concatenate and validate the requested monthly archives."""

    archives, missing = discover_archives(root, symbol, months)
    if missing:
        raise DataUnavailable(
            f"missing {symbol} monthly archives for: {', '.join(missing)} under the advertised root"
        )

    times: List[np.ndarray] = []
    values: List[np.ndarray] = []
    infos: List[ArchiveInfo] = []
    for path in archives:
        archive_times, archive_values, info = read_archive(path)
        if archive_times.shape[0] > 1:
            steps = np.diff(archive_times)
            if not np.all(steps == MILLISECONDS_PER_MINUTE):
                raise DataUnavailable(
                    f"archive {path.name} is not a 1m-continuous series "
                    f"({int((steps != MILLISECONDS_PER_MINUTE).sum())} irregular steps)"
                )
        times.append(archive_times)
        values.append(archive_values)
        infos.append(info)

    open_time_ms = np.concatenate(times)
    ohlcv = np.concatenate(values, axis=0)
    if open_time_ms.shape[0] > 1 and not np.all(np.diff(open_time_ms) == MILLISECONDS_PER_MINUTE):
        raise DataUnavailable(
            f"the concatenated {symbol} series is not 1m-continuous across the interval"
        )
    if np.any(np.diff(open_time_ms) <= 0):
        raise DataUnavailable(f"the concatenated {symbol} series is not strictly increasing")

    valid, reasons = validate_ohlcv(ohlcv)
    start_ms, _ = month_bounds_ms(months[0])
    _, end_ms = month_bounds_ms(months[-1])
    return LoadedBars(
        open_time_ms=open_time_ms,
        ohlcv=ohlcv,
        valid=valid,
        invalid_reasons=reasons,
        archives=tuple(infos),
        root_source=root_source,
        interval_start_ms=start_ms,
        interval_end_ms=end_ms - MILLISECONDS_PER_MINUTE,
        continuous_60s_spacing=True,
    )


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------


def endpoint_valid_mask(valid: np.ndarray, context_length: int) -> np.ndarray:
    """Bar indices that may serve as a window endpoint.

    Endpoint ``t`` is valid when the represented window ``[t-L+1, t]`` and the
    one retained predecessor bar are all raw-valid, i.e. the source window
    ``[t-L, t]`` holds ``L + 1`` consecutive valid bars.
    """

    flags = np.asarray(valid, dtype=bool)
    count = flags.shape[0]
    mask = np.zeros(count, dtype=bool)
    window = context_length + PREDECESSOR_BARS
    if count < window:
        return mask
    cumulative = np.concatenate(([0], np.cumsum(flags, dtype=np.int64)))
    complete = (cumulative[window:] - cumulative[:-window]) == window
    mask[context_length:] = complete
    return mask


def valid_endpoints(valid: np.ndarray, context_length: int) -> np.ndarray:
    return np.flatnonzero(endpoint_valid_mask(valid, context_length))


def select_evenly_spaced(endpoints: np.ndarray, max_windows: int) -> np.ndarray:
    """Select at most ``max_windows`` endpoints with deterministic even spacing.

    Positions are ``floor(i * (M - 1) / (n - 1))`` over the sorted endpoint
    array, so the first and last valid endpoints are always included and no
    random sampling is involved.
    """

    ordered = np.asarray(endpoints, dtype=np.int64)
    if ordered.ndim != 1:
        raise ValueError("endpoints must be a 1-D array")
    if max_windows < 1:
        raise ValueError("max_windows must be positive")
    total = ordered.shape[0]
    if total <= max_windows:
        return ordered.copy()
    if max_windows == 1:
        return ordered[:1].copy()
    positions = np.floor(
        np.arange(max_windows, dtype=np.float64) * (total - 1) / (max_windows - 1)
    ).astype(np.int64)
    return ordered[np.unique(positions)]


def window_matrix(ohlcv: np.ndarray, endpoints: np.ndarray, context_length: int) -> np.ndarray:
    """Materialise ``(K, L + 1, 5)`` raw windows: one predecessor plus ``L`` bars."""

    values = np.asarray(ohlcv, dtype=np.float64)
    offsets = np.arange(
        -(context_length + PREDECESSOR_BARS - 1), 1, dtype=np.int64
    )
    indices = np.asarray(endpoints, dtype=np.int64)[:, None] + offsets[None, :]
    return values[indices]


def selected_endpoint_sha256(endpoints: np.ndarray, context_length: int) -> str:
    """Deterministic digest of the selected endpoint set."""

    digest = hashlib.sha256()
    digest.update(f"cb16.endpoints.v1|L={context_length}|count={len(endpoints)}|".encode("utf-8"))
    digest.update(np.ascontiguousarray(endpoints, dtype=np.int64).tobytes())
    return digest.hexdigest()


def describe_root(root: Path) -> Dict[str, Any]:
    """Path-safe description: literal path only for canonical ``/cb16`` mounts."""

    text = str(root)
    return {
        "is_canonical_cb16_path": text.startswith("/cb16/"),
        "path": text if text.startswith("/cb16/") else None,
    }
