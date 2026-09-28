"""A live run recorded by this package's own CLI against vLLM 0.28 on an RTX 4090.

32 concurrent requests, 1,000 output tokens each (ignore_eos), CUDA graphs on, gauges
sampled every 0.25 s. The files are raw: /metrics exactly as served, snapshot clocks as
taken. These tests pin what the tripwires reported on that run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inferlint import bootlog, checks, series, telemetry
from inferlint.prom import parse
from inferlint.result import Status

LIVE = Path(__file__).parent / "fixtures" / "vllm-0.28" / "live"


@pytest.fixture(scope="module")
def run() -> tuple[telemetry.Snapshot, telemetry.Snapshot, series.Series]:
    before = telemetry.load(LIVE / "before.snapshot.json")
    after = telemetry.load(LIVE / "after.snapshot.json")
    return before, after, series.read(LIVE / "run.series.jsonl")


def test_raw_exposition_parses_completely() -> None:
    m = parse((LIVE / "raw_metrics.prom").read_text(encoding="utf-8"))
    assert m.has("python_gc_objects_collected_total")  # non-vLLM series parse too
    assert m.info("vllm:cache_config_info") is not None


def test_boot_facts() -> None:
    f = bootlog.parse_file(LIVE / "boot.log")
    assert (f.kv_pool_tokens, f.attention_block_size, f.cudagraph_actual_gib) == (26093, 784, 0.18)
    assert f.unparsed == []


def test_t1_preempted_and_the_log_is_silent(run) -> None:  # type: ignore[no-untyped-def]
    before, after, _ = run
    r = checks.check_no_preemption(before, after)
    assert (r.status, r.evidence["preemptions"]) == (Status.FAIL, 22.0)
    assert "preempt" not in (LIVE / "boot.log").read_text(encoding="utf-8").lower()


def test_t8_t9_t14(run) -> None:  # type: ignore[no-untyped-def]
    _, after, s = run
    t14 = checks.check_null_block(s, after)
    assert t14.status is Status.PASS
    assert t14.evidence["inferred_usable_blocks"] == 42
    t9 = checks.concurrency_ceiling(s, 42)
    assert t9.status is Status.WARN
    assert (t9.evidence["ceiling_min"], t9.evidence["ceiling_max"]) == (7, 10)
    assert t9.evidence["blocks_per_request_min"] == pytest.approx(4.0)
    t8 = checks.check_concurrency_reached(s, 32)
    assert t8.evidence["peak_running"] == 10.0  # = the largest ceiling, never above it


def test_running_never_exceeds_the_sample_ceiling(run) -> None:  # type: ignore[no-untyped-def]
    _, _, s = run
    for x in s.samples:
        if x.running and x.kv_usage:
            used = round(x.kv_usage * 42)
            assert x.running <= (42 * int(x.running)) // used


def test_t10_monotonic_span(run) -> None:  # type: ignore[no-untyped-def]
    before, after, _ = run
    r = checks.precise_rate(before, after)
    assert r.evidence["tokens"] == 32 * 1000  # server counter = what the client asked for
    assert r.evidence["span_s"] == pytest.approx(93.72, abs=0.01)
    assert r.evidence["span_uncertainty_s"] < 0.02


def test_series_counter_agrees_with_snapshots(run) -> None:  # type: ignore[no-untyped-def]
    before, after, s = run
    first = next(x.preemptions for x in s.samples if x.preemptions is not None)
    last = [x.preemptions for x in s.samples if x.preemptions is not None][-1]
    delta = telemetry.counter_delta(before, after, "vllm:num_preemptions_total")
    assert first is not None and last is not None and delta == last - first
