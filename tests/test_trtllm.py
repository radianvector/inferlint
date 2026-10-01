"""TensorRT-LLM 1.3.0rc29 (trtllm-serve, PyTorch backend), live on the RTX 4090.

``qwen3-8b/``: ``inferlint xray`` around the same load as every other run (32 at once,
88 prompt and 1,000 output tokens, ``vllm bench serve``) with Qwen3-8B in bf16, started
with ``--max_batch_size 32 --max_seq_len 16384`` and ``return_perf_metrics``. The 27B
model of the vLLM runs does not load: TensorRT-LLM reads weight-only compressed-tensors
checkpoints as if they had activation settings (``boot_fail_compressed_tensors_w4a16``).
"""

from __future__ import annotations

from pathlib import Path

from inferlint import benchresult, bootlog, checks, engines, report, series, telemetry
from inferlint.result import Status

RUN = Path(__file__).parent / "fixtures" / "trtllm-1.3" / "qwen3-8b"


def _snaps() -> tuple[telemetry.Snapshot, telemetry.Snapshot]:
    return (
        telemetry.load(RUN / "before.snapshot.json"),
        telemetry.load(RUN / "after.snapshot.json"),
    )


def test_boot_facts() -> None:
    f = bootlog.parse_file(RUN / "boot.log")
    assert f.engine == "trtllm" and f.ready and f.unparsed == [] and f.conflicts == {}
    assert f.version == "1.3.0rc29" and bootlog.untested_version(f) is None
    # The pool is printed twice: a dry run (544 blocks), then the pool itself.
    assert (f.kv_pool_tokens, f.available_kv_cache_gib, f.page_size) == (37760, 5.19, 32)
    assert (f.running_cap, f.requested_running, f.model_load_gib) == (32, 32, 15.31)
    assert f.selected_backends == ["TRTLLM"] and f.speculative is False
    assert f.server_args is not None and f.server_args["model"] == "/root/models/Qwen3-8B"
    assert f.server_args is not None
    assert f.server_args["capacity_scheduler_policy"] == "GUARANTEED_NO_EVICT"


def test_metrics_come_from_the_prometheus_path() -> None:
    before, after = _snaps()
    assert after.url.endswith("/prometheus/metrics")
    assert checks.engine_of(before, after) is engines.TRTLLM


def test_run_checks() -> None:
    before, after = _snaps()
    s = series.read(RUN / "run.series.jsonl")
    facts = bootlog.parse_file(RUN / "boot.log")
    assert s.engine is engines.TRTLLM

    # No preemption counter. The server log, read after the run, has no pause line.
    assert facts.pauses == 0
    t1 = checks.check_no_preemption(before, after, s, facts)
    assert t1.status is Status.PASS
    assert t1.message == "no pause in the server log, which covers 41 answered requests"
    # A log saved only up to the ready line covers no request, so it counts nothing.
    text = (RUN / "boot.log").read_text(encoding="utf-8")
    start = bootlog.parse(text[: text.index("Application startup complete")])
    assert start.pauses is None and start.served_in_log == 0
    # Logged below INFO there would be no pause lines to count; the scheduler policy,
    # which admits a request only when its whole output fits, still rules pauses out.
    facts.pauses = None
    t1 = checks.check_no_preemption(before, after, s, facts)
    assert t1.status is Status.PASS and "GUARANTEED_NO_EVICT" in t1.message
    t1 = checks.check_no_preemption(before, after, s)  # without the boot log
    assert t1.status is Status.UNKNOWN and "counts no pauses" in t1.message

    t10 = checks.precise_rate(before, after)
    assert t10.status is Status.PASS and t10.evidence["tokens"] == 32_000

    # The gauges move only when a request completes: these 32 finished together, so the
    # recording saw the probe's leftovers for the whole run, then the last step.
    assert sum(1 for x in s.samples if x.kv_usage) <= 3
    t8 = checks.check_concurrency_reached(s, 32, facts)
    assert t8.status is Status.PASS and t8.evidence["gauges_lag"] is True
    t9 = checks.concurrency_ceiling(s)
    assert t9.evidence["ceiling_min"] == 35  # 1,180 blocks / 34 per request
    assert "updates its gauges only when a request completes" in t9.message

    t15 = checks.check_client_server_agree(benchresult.load(RUN / "bench.json"), before, after)
    assert t15.status is Status.PASS and t15.evidence["server_finished"] == 32


def test_a_shortfall_with_lagging_gauges_is_cant_tell() -> None:
    s = series.read(RUN / "run.series.jsonl")
    t8 = checks.check_concurrency_reached(s, 64)
    assert t8.status is Status.UNKNOWN and "may have run more in between" in t8.message


def test_report() -> None:
    before, after = _snaps()
    rep = report.build(
        boot_logs=[RUN / "boot.log"],
        before=before,
        after=after,
        series=series.read(RUN / "run.series.jsonl"),
        requested=32,
    )
    assert rep.engine is engines.TRTLLM and rep.title == "TensorRT-LLM 1.3.0rc29 run"
    t14 = rep.result("T14")
    assert t14 is not None and t14.status is Status.NOT_APPLICABLE
    # TensorRT-LLM says why each request finished, as vLLM does.
    assert rep.summary.finished == {"length": 32}
    html = report.render(rep)
    assert "T14 (does not apply to TensorRT-LLM)" in html
    assert "On TensorRT-LLM: a request is paused for recompute" in html
    assert "On SGLang" not in html
    # the requests and KV charts; no tokens-per-second chart, since the counter moves on
    # completion only (3 changes in 126 readings)
    assert html.count("updates these readings only when a request completes") == 2
    assert 'id="fig-tput"' not in html
    assert "No output-tokens-per-second chart: TensorRT-LLM adds a request's tokens" in html


def test_pauses_are_counted_from_the_log() -> None:
    """``capacity_scheduler_policy: MAX_UTILIZATION`` and a 20,960-token pool.

    TensorRT-LLM paused 13 requests for recompute and logged each ("request ID N ->
    pause"). Its paused-requests gauge, which moves only when a request completes, never
    showed one in 171 readings.
    """
    d = RUN.parent / "pauses"
    facts = bootlog.parse_file(d / "boot.log")
    assert facts.kv_pool_tokens == 20960 and facts.pauses == 13
    assert facts.server_args is not None
    assert facts.server_args["capacity_scheduler_policy"] == "MAX_UTILIZATION"
    s = series.read(d / "run.series.jsonl")
    assert {x.paused for x in s.samples} == {0.0}
    before = telemetry.load(d / "before.snapshot.json")
    after = telemetry.load(d / "after.snapshot.json")
    t1 = checks.check_no_preemption(before, after, s, facts)
    assert t1.status is Status.FAIL
    assert t1.message == (
        "13 requests paused for recompute during the run, from the server log "
        "(TensorRT-LLM counts none)"
    )
    assert checks.check_no_preemption(before, after, s).status is Status.UNKNOWN


def test_without_iteration_stats_the_gauges_are_missing() -> None:
    """The README's command with only ``return_perf_metrics: true`` in its config.

    TensorRT-LLM then serves its counters and histograms but no running, waiting or KV
    gauges, so T8 and T9 cannot tell, and say which setting adds them. The counters still
    give T1 (from the log), T10 and T15.
    """
    d = RUN.parent / "no-iter-stats"
    after = telemetry.load(d / "after.snapshot.json")
    assert after.metrics.total("trtllm_generation_tokens_total") is not None
    assert after.metrics.total("trtllm_num_requests_running") is None
    s = series.read(d / "run.series.jsonl")
    assert s.engine is engines.TRTLLM
    t8 = checks.check_concurrency_reached(s, 32)
    t9 = checks.concurrency_ceiling(s)
    for r in (t8, t9):
        assert r.status is Status.UNKNOWN and "'enable_iter_perf_stats: true'" in r.message
    before = telemetry.load(d / "before.snapshot.json")
    t15 = checks.check_client_server_agree(
        benchresult.load(d / "bench.json"), before, after, expected_extra_requests=1
    )
    assert t15.status is Status.PASS
    # vLLM's series without a running gauge gets no TensorRT-LLM advice
    assert engines.VLLM.gauges_hint == ""
