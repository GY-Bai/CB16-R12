"""Deterministic synthetic market-data fixtures for the normalization tests.

The fixtures reproduce the real read-only archive layout exactly (one zip per
month, one CSV member, no header row, first six columns
``open_time, open, high, low, close, volume``) but are generated from a fixed
NumPy generator so tests never depend on the mounted tree and never write to it.
"""

from __future__ import annotations

import calendar
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

MILLISECONDS_PER_MINUTE = 60_000
COLUMNS = 6


def month_start_ms(month: str) -> int:
    year, number = (int(part) for part in month.split("-"))
    return int(datetime(year, number, 1, tzinfo=timezone.utc).timestamp() * 1000)


def month_minutes(month: str) -> int:
    year, number = (int(part) for part in month.split("-"))
    return calendar.monthrange(year, number)[1] * 1440


def generate_ohlcv(
    count: int, *, seed: int = 20200101, start_price: float = 7200.0
) -> np.ndarray:
    """A deterministic, strictly positive, correctly ordered OHLCV series."""

    generator = np.random.Generator(np.random.PCG64(seed))
    close = start_price * np.exp(np.cumsum(generator.normal(0.0, 0.0009, count)))
    previous = np.concatenate(([start_price], close[:-1]))
    open_ = previous * np.exp(generator.normal(0.0, 0.0002, count))
    high = np.maximum(open_, close) * np.exp(np.abs(generator.normal(0.0, 0.0004, count)))
    low = np.minimum(open_, close) * np.exp(-np.abs(generator.normal(0.0, 0.0004, count)))
    volume = 5.0 + np.abs(generator.normal(0.0, 3.0, count))
    return np.column_stack([open_, high, low, close, volume])


def rows_to_csv(open_times: np.ndarray, ohlcv: np.ndarray) -> str:
    return (
        "\n".join(
            f"{int(open_times[index])},"
            + ",".join(f"{value:.8f}" for value in ohlcv[index])
            for index in range(ohlcv.shape[0])
        )
        + "\n"
    )


def write_month_archive(
    root: Path,
    symbol: str,
    month: str,
    *,
    ohlcv: Optional[np.ndarray] = None,
    start_price: float = 7200.0,
    seed: int = 20200101,
    header: bool = False,
    start_ms: Optional[int] = None,
) -> Path:
    """Write one monthly archive and return its path."""

    directory = Path(root) / symbol
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{symbol}-1m-{month}.zip"
    count = month_minutes(month) if ohlcv is None else int(ohlcv.shape[0])
    if ohlcv is None:
        ohlcv = generate_ohlcv(count, seed=seed, start_price=start_price)
    first = month_start_ms(month) if start_ms is None else int(start_ms)
    open_times = first + MILLISECONDS_PER_MINUTE * np.arange(count, dtype=np.int64)
    body = rows_to_csv(open_times, ohlcv)
    if header:
        body = "open_time,open,high,low,close,volume\n" + body
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{symbol}-1m-{month}.csv", body)
    return path


def build_klines_root(
    root: Path,
    *,
    symbol: str = "BTCUSDT",
    months: Sequence[str] = ("2020-01", "2020-02", "2020-03"),
    header: bool = False,
) -> Path:
    """Build a full, calendar-complete, 1m-continuous fixture tree."""

    price = 7200.0
    for index, month in enumerate(months):
        write_month_archive(
            root,
            symbol,
            month,
            start_price=price,
            seed=20200101 + index,
            header=header,
        )
        price *= 1.05
    return Path(root)


def write_series_archive(
    root: Path,
    symbol: str,
    month: str,
    ohlcv: np.ndarray,
    *,
    start_ms: Optional[int] = None,
    header: bool = False,
) -> Path:
    """Write an archive from an explicit series (used for invalid-bar cases)."""

    return write_month_archive(
        root,
        symbol,
        month,
        ohlcv=ohlcv,
        start_ms=start_ms,
        header=header,
    )
