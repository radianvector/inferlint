"""Gauge time series: the only place concurrency and KV occupancy are visible.

``num_requests_running`` and ``kv_cache_usage_perc`` are instantaneous. A before/after
pair of snapshots cannot say whether the requested concurrency ever actually ran, or
how full the cache got. Sampling them during the run can.

File format: JSON Lines. The first line is a header (``{"schema": ..., "url": ...}``),
every later line one sample. Readers skip lines they cannot parse and count them, so a
run killed mid-write still yields everything it recorded.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, cast

from .prom import Metrics
from .telemetry import Snapshot, scrape

__all__ = [
    "SCHEMA",
    "VLLM_GAUGES",
    "Sample",
    "Series",
    "from_samples",
    "read",
    "sample_row",
    "watch",
]

SCHEMA = "inferlint.series/1"

# Output field -> vLLM series. Absent series are written as null, never 0.
VLLM_GAUGES: dict[str, str] = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv_usage": "vllm:kv_cache_usage_perc",
    "preemptions": "vllm:num_preemptions_total",
    "generation_tokens": "vllm:generation_tokens_total",
    "prompt_tokens": "vllm:prompt_tokens_total",
}


@dataclass(frozen=True, slots=True)
class Sample:
    t_wall: float
    t_mono_ns: int | None
    running: float | None
    waiting: float | None
    kv_usage: float | None
    preemptions: float | None = None
    generation_tokens: float | None = None
    prompt_tokens: float | None = None


@dataclass(frozen=True)
class Series:
    header: dict[str, Any]
    samples: tuple[Sample, ...]
    skipped_lines: int = 0

    def busy(self) -> tuple[Sample, ...]:
        return tuple(s for s in self.samples if s.running)

    def peak_running(self) -> float | None:
        vals = [s.running for s in self.samples if s.running is not None]
        return max(vals) if vals else None

    def kv_usages(self) -> list[float]:
        return [s.kv_usage for s in self.samples if s.kv_usage]


def sample_row(m: Metrics, t_wall: float, t_mono_ns: int | None) -> dict[str, Any]:
    row: dict[str, Any] = {"t_wall": round(t_wall, 6), "t_mono_ns": t_mono_ns}
    for key, series in VLLM_GAUGES.items():
        row[key] = m.single(series) if key == "kv_usage" else m.total(series)
    return row


def _row_to_sample(row: dict[str, Any]) -> Sample:
    def f(k: str) -> float | None:
        v = row.get(k)
        return None if v is None else float(v)

    mono = row.get("t_mono_ns")
    return Sample(
        t_wall=float(row["t_wall"]),
        t_mono_ns=None if mono is None else int(mono),
        running=f("running"),
        waiting=f("waiting"),
        kv_usage=f("kv_usage"),
        preemptions=f("preemptions"),
        generation_tokens=f("generation_tokens"),
        prompt_tokens=f("prompt_tokens"),
    )


def read(path: str | Path) -> Series:
    header: dict[str, Any] = {}
    samples: list[Sample] = []
    skipped = 0
    with Path(path).open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(row, dict):
                skipped += 1
                continue
            row = cast(dict[str, Any], row)
            if "schema" in row:
                header = row
                continue
            try:
                samples.append(_row_to_sample(row))
            except (KeyError, TypeError, ValueError):
                skipped += 1
    return Series(header=header, samples=tuple(samples), skipped_lines=skipped)


def from_samples(samples: Iterable[Sample], header: dict[str, Any] | None = None) -> Series:
    return Series(header=header or {}, samples=tuple(samples))


def watch(
    base_url: str,
    out: IO[str],
    *,
    interval: float = 0.5,
    stop: threading.Event | None = None,
    duration: float | None = None,
    scrape_fn: Callable[[str], Snapshot] = scrape,
    sleep: Callable[[float], None] = time.sleep,
    on_error: Callable[[Exception], None] | None = None,
) -> int:
    """Sample gauges into ``out`` until ``stop`` is set or ``duration`` elapses.

    A scrape that fails is skipped rather than written as zeros: a missing sample is
    a gap in the series, and a zero would be a false reading of an idle server.
    """
    stop = stop or threading.Event()
    out.write(json.dumps({"schema": SCHEMA, "url": base_url, "interval_s": interval}) + "\n")
    t_end = None if duration is None else time.monotonic() + duration
    n = 0
    while not stop.is_set() and (t_end is None or time.monotonic() < t_end):
        try:
            snap = scrape_fn(base_url)
        except Exception as e:  # network errors, server gone: record a gap
            if on_error is not None:
                on_error(e)
        else:
            out.write(json.dumps(sample_row(snap.metrics, snap.t_wall, snap.t_mono_ns)) + "\n")
            out.flush()
            n += 1
        sleep(interval)
    return n
