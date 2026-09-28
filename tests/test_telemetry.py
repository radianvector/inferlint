from __future__ import annotations

import json
from pathlib import Path

import pytest

from inferlint import telemetry
from inferlint.telemetry import ServerRestarted, Snapshot, counter_delta, span_s

A = 'vllm:x_total{engine="0"} 10\nvllm:x_created{engine="0"} 1000.0\n'
B = 'vllm:x_total{engine="0"} 25\nvllm:x_created{engine="0"} 1000.0\n'


def snap(text: str, wall: float = 0.0, mono: int | None = 0, lat: float | None = None) -> Snapshot:
    return Snapshot(url="u", text=text, t_wall=wall, t_mono_ns=mono, latency_s=lat)


def test_counter_delta() -> None:
    assert counter_delta(snap(A), snap(B), "vllm:x_total") == 15.0


def test_missing_counter_is_none() -> None:
    assert counter_delta(snap(A), snap(B), "vllm:y_total") is None


def test_counter_going_down_is_a_restart() -> None:
    with pytest.raises(ServerRestarted, match="went down"):
        counter_delta(snap(B), snap(A.replace("1000.0", "1000.0")), "vllm:x_total")


def test_created_stamp_catches_restart_counters_cannot() -> None:
    # New process, and traffic has already pushed the counter past its old value.
    after = 'vllm:x_total{engine="0"} 40\nvllm:x_created{engine="0"} 2000.0\n'
    with pytest.raises(ServerRestarted, match="_created"):
        counter_delta(snap(A), snap(after), "vllm:x_total")


def test_scrape_uses_midpoint_and_records_latency() -> None:
    walls = iter([100.0, 100.4])
    monos = iter([1_000_000_000, 1_400_000_000])
    s = telemetry.scrape(
        "http://h:8000",
        fetch=lambda url, t: A,
        wall=lambda: next(walls),
        mono_ns=lambda: next(monos),
    )
    assert s.url == "http://h:8000/metrics"
    assert s.t_wall == pytest.approx(100.2)
    assert s.t_mono_ns == 1_200_000_000
    assert s.latency_s == pytest.approx(0.4)


def test_span_prefers_monotonic_and_reports_uncertainty() -> None:
    a = snap(A, wall=100.0, mono=5_000_000_000, lat=0.02)
    b = snap(B, wall=130.3, mono=35_250_000_000, lat=0.04)
    span, unc = span_s(a, b)
    assert span == pytest.approx(30.25)
    assert unc == pytest.approx(0.03)


def test_span_refuses_clock_disagreement() -> None:
    a = snap(A, wall=100.0, mono=0)
    b = snap(B, wall=200.0, mono=30_000_000_000)  # wall stepped by ~70 s
    with pytest.raises(ValueError, match="disagree"):
        span_s(a, b)


def test_span_needs_a_clock() -> None:
    with pytest.raises(ValueError, match="no clock"):
        span_s(snap(A, wall=float("nan"), mono=None), snap(B, wall=float("nan"), mono=None))


def test_roundtrip(tmp_path: Path) -> None:
    s = snap(A, wall=1.5, mono=7, lat=0.01)
    p = tmp_path / "s.json"
    s.save(p)
    back = telemetry.load(p)
    assert (back.text, back.t_wall, back.t_mono_ns, back.latency_s) == (A, 1.5, 7, 0.01)
    assert json.loads(p.read_text())["schema"] == telemetry.SCHEMA


def test_load_bare_prometheus_text(fx: Path) -> None:
    s = telemetry.load(fx / "metrics" / "preempted_before.prom")
    assert s.t_mono_ns is None
    assert s.metrics.total("vllm:num_preemptions_total") == 106.0
