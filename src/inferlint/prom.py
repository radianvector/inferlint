"""Prometheus text-format parsing, for the subset inference servers actually emit.

Two rules shape this module:

* A series that is absent is reported as absent (``None``), never as ``0.0``. A renamed
  counter that reads as a clean zero is the classic "assertion that cannot fail".
* Samples keep all their labels. Callers select by label subset; nothing is dropped at
  parse time, so a multi-engine server cannot silently collide two series into one key.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

__all__ = ["Metrics", "Sample", "parse"]

_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?"
    r"\s+(?P<value>\S+)"
    r"(?:\s+(?P<ts>-?\d+))?\s*$"
)
_LABEL = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"\s*(?:,|$)')
_UNESCAPE = {"\\\\": "\\", '\\"': '"', "\\n": "\n"}


@dataclass(frozen=True, slots=True)
class Sample:
    name: str
    labels: tuple[tuple[str, str], ...]
    value: float

    def label(self, key: str) -> str | None:
        for k, v in self.labels:
            if k == key:
                return v
        return None


def _parse_labels(body: str) -> tuple[tuple[str, str], ...]:
    out: list[tuple[str, str]] = []
    pos = 0
    body = body.strip()
    while pos < len(body):
        m = _LABEL.match(body, pos)
        if m is None:
            raise ValueError(f"malformed label set: {{{body}}}")
        raw = m.group(2)
        val = re.sub(r"\\[\\\"n]", lambda e: _UNESCAPE[e.group(0)], raw)
        out.append((m.group(1), val))
        pos = m.end()
    return tuple(sorted(out))


def _parse_value(text: str) -> float:
    # float() already accepts NaN, +Inf, -Inf in any case.
    return float(text)


def parse(text: str) -> Metrics:
    """Parse Prometheus text exposition. Malformed sample lines raise ``ValueError``."""
    samples: list[Sample] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE.match(line)
        if m is None:
            raise ValueError(f"line {lineno}: not a Prometheus sample: {line[:120]!r}")
        labels = _parse_labels(m.group("labels")) if m.group("labels") else ()
        samples.append(Sample(m.group("name"), labels, _parse_value(m.group("value"))))
    return Metrics(samples)


class Metrics:
    """An immutable set of samples with label-subset lookup."""

    __slots__ = ("_by_name", "_samples")

    def __init__(self, samples: list[Sample]) -> None:
        self._samples = tuple(samples)
        by_name: dict[str, list[Sample]] = {}
        for s in self._samples:
            by_name.setdefault(s.name, []).append(s)
        self._by_name = {k: tuple(v) for k, v in by_name.items()}

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def names(self) -> set[str]:
        return set(self._by_name)

    def has(self, name: str) -> bool:
        return name in self._by_name

    def select(self, name: str, **labels: str) -> tuple[Sample, ...]:
        """Every sample of ``name`` whose labels include all of ``labels``."""
        want = labels.items()
        return tuple(
            s for s in self._by_name.get(name, ()) if all(s.label(k) == v for k, v in want)
        )

    def total(self, name: str, **labels: str) -> float | None:
        """Sum over matching series, or ``None`` if no series matches.

        Summing is right for counters and for count-like gauges (requests running).
        It is wrong for ratios such as KV-cache usage; use :meth:`single` for those.
        """
        found = self.select(name, **labels)
        if not found:
            return None
        return math.fsum(s.value for s in found)

    def single(self, name: str, **labels: str) -> float | None:
        """The value of exactly one matching series. More than one raises ``LookupError``."""
        found = self.select(name, **labels)
        if len(found) != 1:
            if not found:
                return None
            sets = ", ".join(str(dict(s.labels)) for s in found)
            raise LookupError(f"{name} matches {len(found)} series ({sets}); narrow the labels")
        return found[0].value

    def info(self, name: str, **labels: str) -> Mapping[str, str] | None:
        """Labels of an ``*_info`` series, which carry the data (the value is always 1)."""
        found = self.select(name, **labels)
        if not found:
            return None
        return dict(found[0].labels)

    def with_prefix(self, prefix: str) -> tuple[Sample, ...]:
        return tuple(s for s in self._samples if s.name.startswith(prefix))
