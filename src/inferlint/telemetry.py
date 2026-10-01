"""Snapshots of a server's /metrics, and what may be read out of two of them.

A snapshot keeps the raw exposition text, not a parsed digest, so nothing a later
reader needs has been thrown away at capture time. Each one carries two clocks:

* ``t_wall`` (``time.time()``) to line a snapshot up with logs, and
* ``t_mono_ns`` (``time.monotonic_ns()``) to measure spans. Wall clocks step; the
  monotonic clock does not.

Both are taken on either side of the HTTP request, and the midpoint is used. The
server read its counters somewhere inside that window, so half the scrape latency is
the honest timing uncertainty of a snapshot, and it is recorded rather than assumed away.
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .prom import Metrics, parse

__all__ = [
    "SCHEMA",
    "ServerRestarted",
    "Snapshot",
    "counter_delta",
    "created_changes",
    "load",
    "scrape",
    "span_s",
]

SCHEMA = "inferlint.snapshot/1"

Fetch = Callable[[str, float], str]


def _http_get(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body: bytes = resp.read()
    return body.decode("utf-8", "replace")


@dataclass(frozen=True)
class Snapshot:
    url: str
    text: str
    t_wall: float
    t_mono_ns: int | None
    latency_s: float | None = None
    _metrics: list[Metrics] = field(default_factory=list[Metrics], repr=False, compare=False)

    @property
    def metrics(self) -> Metrics:
        if not self._metrics:
            self._metrics.append(parse(self.text))
        return self._metrics[0]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "url": self.url,
            "t_wall": self.t_wall,
            "t_mono_ns": self.t_mono_ns,
            "latency_s": self.latency_s,
            "text": self.text,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Snapshot:
        if doc.get("schema") != SCHEMA:
            raise ValueError(f"not an {SCHEMA} document (schema={doc.get('schema')!r})")
        return cls(
            url=str(doc["url"]),
            text=str(doc["text"]),
            t_wall=float(doc["t_wall"]),
            t_mono_ns=None if doc.get("t_mono_ns") is None else int(doc["t_mono_ns"]),
            latency_s=None if doc.get("latency_s") is None else float(doc["latency_s"]),
        )


def load(path: str | Path) -> Snapshot:
    """Read a snapshot file, or a bare Prometheus text file (which has no clock)."""
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    if raw.lstrip().startswith("{"):
        return Snapshot.from_json(json.loads(raw))
    return Snapshot(url=f"file:{p.name}", text=raw, t_wall=float("nan"), t_mono_ns=None)


# TensorRT-LLM answers /metrics with JSON iteration statistics and serves Prometheus text
# at /prometheus/metrics. The path that worked is remembered per server.
_FALLBACK_PATHS = ("/prometheus/metrics",)
_found_path: dict[str, str] = {}


def _is_json(text: str) -> bool:
    return text.lstrip()[:1] in ("{", "[")


def scrape(
    base_url: str,
    *,
    path: str | None = None,
    timeout: float = 10.0,
    fetch: Fetch = _http_get,
    wall: Callable[[], float] = time.time,
    mono_ns: Callable[[], int] = time.monotonic_ns,
) -> Snapshot:
    """Read the server's Prometheus metrics once, with the time of the reading.

    Without ``path``, ``/metrics`` is read; if that is not Prometheus text, the other
    places servers put it are tried, and the one that works is used from then on.
    """
    base = base_url.rstrip("/")
    if path is None and base in _found_path:
        try:
            return scrape(
                base,
                path=_found_path[base],
                timeout=timeout,
                fetch=fetch,
                wall=wall,
                mono_ns=mono_ns,
            )
        except Exception:  # another server answers here now: start again from /metrics
            del _found_path[base]
    url = base + (path or "/metrics")
    w0, m0 = wall(), mono_ns()
    text = fetch(url, timeout)
    if path is None and _is_json(text):
        for alt in _FALLBACK_PATHS:
            w0, m0 = wall(), mono_ns()
            alt_text = fetch(base + alt, timeout)
            if not _is_json(alt_text):
                url, text = base + alt, alt_text
                _found_path[base] = alt
                break
    w1, m1 = wall(), mono_ns()
    return Snapshot(
        url=url,
        text=text,
        t_wall=(w0 + w1) / 2,
        t_mono_ns=(m0 + m1) // 2,
        latency_s=(m1 - m0) / 1e9,
    )


def span_s(a: Snapshot, b: Snapshot) -> tuple[float, float]:
    """Seconds between two snapshots, and the uncertainty from their scrape latencies.

    The monotonic clock is used when both snapshots have it. It is only comparable
    within one host boot, so it is cross-checked against the wall clock: if the two
    disagree by more than a second, one of them has jumped and the span is refused.
    """
    wall = b.t_wall - a.t_wall
    unc = ((a.latency_s or 0.0) + (b.latency_s or 0.0)) / 2
    if a.t_mono_ns is not None and b.t_mono_ns is not None:
        mono = (b.t_mono_ns - a.t_mono_ns) / 1e9
        if wall == wall and abs(mono - wall) > 1.0:  # wall == wall: not NaN
            raise ValueError(
                f"monotonic span {mono:.3f}s and wall span {wall:.3f}s disagree; "
                "snapshots from different host boots, or the wall clock stepped"
            )
        return mono, unc
    if wall != wall:
        raise ValueError("snapshots carry no clock (bare Prometheus text); no span available")
    return wall, unc


class ServerRestarted(RuntimeError):
    """Two snapshots came from different server processes. Their deltas mean nothing."""

    def __init__(self, message: str, evidence: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.evidence = dict(evidence)


def created_changes(a: Snapshot, b: Snapshot) -> list[tuple[str, float, float]]:
    """``*_created`` series whose value differs between two snapshots.

    prometheus_client stamps every counter and histogram with the Unix time it was
    created. The stamp changes only when the process that owns the counter restarts.
    That catches a restart even when traffic since the restart has already pushed the
    counters past their old values, which a "counter went down" test cannot see.
    """
    before = {(s.name, s.labels): s.value for s in a.metrics if s.name.endswith("_created")}
    changed: list[tuple[str, float, float]] = []
    for s in b.metrics:
        if not s.name.endswith("_created"):
            continue
        old = before.get((s.name, s.labels))
        if old is not None and old != s.value:
            changed.append((s.name, old, s.value))
    return changed


def counter_delta(
    a: Snapshot, b: Snapshot, name: str, *, witness: str | None = None, **labels: str
) -> float | None:
    """``after - before`` for a counter, ``None`` if either snapshot lacks it.

    Some servers write a labelled counter only when it first counts something. With
    ``witness``, a series exported from the start alongside it, a snapshot that has the
    witness but not the counter reads the counter as 0.

    Raises :class:`ServerRestarted` if the server was replaced between the snapshots:
    either a ``_created`` stamp moved, or the counter went down.
    """
    moved = created_changes(a, b)
    if moved:
        name0, old, new = moved[0]
        raise ServerRestarted(
            f"server restarted between snapshots ({len(moved)} *_created stamps moved, "
            f"e.g. {name0}: {old:.3f} -> {new:.3f})",
            {"created_changed": len(moved), "example": name0, "before": old, "after": new},
        )

    def value(s: Snapshot) -> float | None:
        v = s.metrics.total(name, **labels)
        if v is None and witness is not None and s.metrics.has(witness):
            return 0.0
        return v

    v0, v1 = value(a), value(b)
    if v0 is None or v1 is None:
        return None
    if v1 < v0:
        raise ServerRestarted(
            f"{name} went down ({v0:g} -> {v1:g}); counters only reset on restart",
            {"counter": name, "before": v0, "after": v1},
        )
    return v1 - v0
