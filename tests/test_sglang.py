"""SGLang 0.5.20, live on the same RTX 4090, model and load as the vLLM runs.

``capped/``: started with ``--max-running-requests 32 --mem-fraction-static 0.88``. For
this hybrid (attention + Gated DeltaNet) model SGLang keeps a fixed state slot per running
request, and lowered its limit to 1 running, which its boot log says once and its
``/get_server_info`` does not (it reports 32). The load tool, ``vllm bench serve`` with 32
at once, reported a peak of 32 concurrent requests.
"""

from __future__ import annotations

from pathlib import Path

from inferlint import benchresult, bootlog, checks, engines, report, series, telemetry
from inferlint.failures import Kind, classify, classify_file
from inferlint.result import Status
from inferlint.teardown import Proc, server_engine, usual_pkill_pattern

FX = Path(__file__).parent / "fixtures" / "sglang-0.5"
RUN = FX / "capped"


def _snaps() -> tuple[telemetry.Snapshot, telemetry.Snapshot]:
    return (
        telemetry.load(RUN / "before.snapshot.json"),
        telemetry.load(RUN / "after.snapshot.json"),
    )


def test_engine_is_told_from_log_and_metrics() -> None:
    text = (RUN / "boot.log").read_text(encoding="utf-8")
    assert engines.from_log(text) is engines.SGLANG
    vllm_log = (FX.parent / "vllm-0.30" / "live" / "boot.log").read_text(encoding="utf-8")
    assert engines.from_log(vllm_log) is engines.VLLM
    before, after = _snaps()
    assert checks.engine_of(before, after) is engines.SGLANG
    assert series.read(RUN / "run.series.jsonl").engine is engines.SGLANG


def test_boot_facts() -> None:
    f = bootlog.parse_file(RUN / "boot.log")
    assert f.engine == "sglang" and f.ready and f.unparsed == []
    assert (f.kv_pool_tokens, f.available_kv_cache_gib, f.page_size) == (17675, 1.08, 1)
    assert (f.model_load_gib, f.cudagraph_actual_gib, f.cudagraphs) == (17.65, 1.74, True)
    # Asked for 32 running; SGLang settled on 1, and says why once.
    assert (f.requested_running, f.running_cap) == (32, 1)
    assert f.running_cap_reason == "mamba state cache"
    assert f.requested_backend is None and f.selected_backends == ["flashinfer"]
    # The log has no version; xray reads it from the server.
    assert f.version is None and bootlog.untested_version(f) is None
    f.server_version = "0.5.20"
    assert bootlog.untested_version(f) is None
    f.server_version = "0.6.1"
    note = bootlog.untested_version(f)
    assert note is not None and note.startswith("SGLang 0.6.1 is not a tested version")


def test_starts_with_flags_differing_only_in_the_random_seed_are_compared() -> None:
    a = bootlog.parse_file(RUN / "boot.log")
    b = bootlog.parse_file(RUN / "boot.log")
    assert a.server_args is not None and b.server_args is not None
    b.server_args = {**b.server_args, "random_seed": 1234}
    (t4,) = checks.check_same_pool([a, b], ["a", "b"])
    assert t4.status is Status.PASS


def test_ignored_tracebacks_are_not_a_failed_start() -> None:
    """SGLang logs import errors it ignores (torchcodec without FFmpeg) as tracebacks."""
    text = (RUN / "boot.log").read_text(encoding="utf-8")
    assert "libavutil" in text and "Ignore import error" in text
    assert classify(text) is None
    assert checks.check_boot_failure(text).status is Status.PASS


def test_real_start_up_failures_are_named() -> None:
    f = classify_file(FX / "boot_fail_mamba_state.log")
    assert f is not None and f.kind is Kind.RUNTIME_ERROR
    assert f.detail.startswith("Not enough GPU memory for hybrid (mamba/linear-attention) state")
    # With --disable-radix-cache SGLang sizes 33 state slots for 32 requests, runs out, and
    # advises raising --mem-fraction-static above 0.790 when it was already 0.88.
    f = classify_file(FX / "boot_fail_mem_advice.log")
    assert f is not None and f.kind is Kind.VALUE_ERROR
    assert "--mem-fraction-static=0.88. Raise --mem-fraction-static above 0.790" in f.detail


def test_boot_log_checks() -> None:
    f = bootlog.parse_file(RUN / "boot.log")
    t5 = checks.check_block_size(f)
    assert (t5.status, t5.message) == (Status.PASS, "page size 1 token, as set")
    t6 = checks.check_backend_honoured(f)
    assert t6.status is Status.PASS and "SGLang chose its own (flashinfer)" in t6.message


def test_run_checks() -> None:
    before, after = _snaps()
    s = series.read(RUN / "run.series.jsonl")
    facts = bootlog.parse_file(RUN / "boot.log")

    # No retraction: SGLang writes its counter only at the first one; the gauge is there.
    assert after.metrics.total("sglang:num_retracted_requests_total") is None
    t1 = checks.check_no_preemption(before, after)
    assert (t1.status, t1.message) == (Status.PASS, "no retractions")

    t10 = checks.precise_rate(before, after)
    assert t10.status is Status.PASS and t10.evidence["tokens"] == 32_000

    t14 = checks.check_null_block(s, after)
    assert t14.status is Status.UNKNOWN and "SGLang has none to check" in t14.message

    t8 = checks.check_concurrency_reached(s, 32, facts)
    assert t8.status is Status.FAIL and t8.evidence["peak_running"] == 1
    assert "server never ran more than 1" in t8.message
    assert "SGLang lowered its limit to 1 running because of the mamba state cache" in t8.message

    # SGLang's usage gauge is its fullest pool; here the state slots, 4 of 5 in use.
    t9 = checks.concurrency_ceiling(s)
    assert t9.evidence["ceiling_min"] == t9.evidence["ceiling_max"] == 1
    assert "of the fullest memory pool (KV cache or request state)" in t9.message

    bench = benchresult.load(RUN / "bench.json")
    assert (bench.completed, bench.output_tokens, bench.max_concurrency) == (32, 32_000, 32)
    t15 = checks.check_client_server_agree(bench, before, after)
    assert t15.status is Status.PASS and t15.evidence["server_finished"] == 32


def test_report() -> None:
    before, after = _snaps()
    rep = report.build(
        boot_logs=[RUN / "boot.log"],
        before=before,
        after=after,
        series=series.read(RUN / "run.series.jsonl"),
        requested=32,
        server_version="0.5.20",
    )
    assert rep.engine is engines.SGLANG and rep.title == "SGLang 0.5.20 run"
    assert {r.tripwire for r in rep.results} >= {"T1", "T5", "T6", "T8", "T9", "T10", "T12"}
    assert "T14" not in {r.tripwire for r in rep.results}
    assert rep.summary.done == 32 and rep.summary.finished == {}
    html = report.render(rep)
    assert "T14 (does not apply to SGLang)" in html
    assert "SGLang 0.5.20" in html and "<dt>SGLang</dt><dd>0.5.20</dd>" in html
    assert "On SGLang: called a retraction" in html and "On TensorRT-LLM" not in html
    assert "updates these readings only when" not in html


def test_processes() -> None:
    launcher = Proc(10, 1, ("/opt/venv/bin/python", "-m", "sglang.launch_server"), "python")
    sched = Proc(11, 10, ("sglang::scheduler",), "sglang::schedul")
    detok = Proc(12, 10, ("sglang::detokenizer",), "sglang::detoken")
    cli = Proc(13, 1, ("/opt/venv/bin/sglang", "serve", "--model-path", "m"), "sglang")
    shell = Proc(14, 1, ("bash", "-c", "pkill -f sglang.launch_server"), "bash")
    assert [server_engine(p) for p in (launcher, sched, detok, cli, shell)] == [
        "sglang",
        "sglang",
        "sglang",
        "sglang",
        None,
    ]
    assert usual_pkill_pattern([launcher, sched]) == "sglang.launch_server"
    assert usual_pkill_pattern([cli, sched]) == "sglang serve"


LIVE = FX / "live"


def test_live_run_with_room_for_five() -> None:
    """``--max-running-requests 10 --disable-radix-cache --mem-fraction-static 0.88``.

    Ten state slots left 5,930 KV tokens. SGLang admits a request only when the pool can
    hold the output it expects, so it ran 5 at once, never 10, and retracted nothing.
    """
    f = bootlog.parse_file(LIVE / "boot.log")
    assert (f.kv_pool_tokens, f.running_cap, f.requested_running) == (5930, 10, 10)
    # After the ready line SGLang's own post-warmup step timed out (it logged
    # "post-warmup freeze_gc failed" and a traceback) and the server went on serving.
    text = (LIVE / "boot.log").read_text(encoding="utf-8")
    assert "post-warmup freeze_gc failed" in text and "TimeoutError: timed out" in text
    assert checks.check_boot_failure(text).status is Status.PASS

    before = telemetry.load(LIVE / "before.snapshot.json")
    after = telemetry.load(LIVE / "after.snapshot.json")
    s = series.read(LIVE / "run.series.jsonl")
    assert checks.check_no_preemption(before, after, s).message == "no retractions"
    t8 = checks.check_concurrency_reached(s, 32, f)
    assert t8.status is Status.FAIL and t8.evidence["peak_running"] == 5
    assert "lowered" not in t8.message and "runs at most" not in t8.message  # cap not reached
    t9 = checks.concurrency_ceiling(s)
    assert t9.evidence["ceiling_min"] == t9.evidence["ceiling_max"] == 5  # = the peak
    t15 = checks.check_client_server_agree(benchresult.load(LIVE / "bench.json"), before, after)
    assert t15.status is Status.PASS


Q8B = FX / "qwen3-8b"


def test_qwen3_8b_through_xray() -> None:
    """``inferlint xray`` with Qwen3-8B (bf16) and 32 allowed to run.

    The pool was 32,096 tokens, about vLLM's 32,336 for the same model and card. vLLM ran
    all 32 and preempted 3 times; SGLang held 3 back and retracted nothing.
    """
    f = bootlog.parse_file(Q8B / "boot.log")
    assert (f.kv_pool_tokens, f.available_kv_cache_gib, f.running_cap) == (32096, 4.4, 32)
    before = telemetry.load(Q8B / "before.snapshot.json")
    after = telemetry.load(Q8B / "after.snapshot.json")
    s = series.read(Q8B / "run.series.jsonl")
    assert s.engine is engines.SGLANG  # xray's own recording names the engine
    assert checks.check_no_preemption(before, after, s).message == "no retractions"
    t8 = checks.check_concurrency_reached(s, 32, f)
    assert t8.status is Status.FAIL and t8.evidence["peak_running"] == 29
    t9 = checks.concurrency_ceiling(s)
    assert (t9.evidence["ceiling_min"], t9.evidence["ceiling_max"]) == (30, 31)
    t15 = checks.check_client_server_agree(benchresult.load(Q8B / "bench.json"), before, after)
    assert t15.status is Status.PASS


def test_a_retraction() -> None:
    """Qwen3-8B with ``--schedule-conservativeness 0.3``: SGLang admits more than fits.

    One request was retracted with 981 tokens generated. SGLang logged a warning, and its
    counter, absent until then, appeared at 1.
    """
    d = FX / "retraction"
    before = telemetry.load(d / "before.snapshot.json")
    after = telemetry.load(d / "after.snapshot.json")
    assert before.metrics.total("sglang:num_retracted_requests_total") is None
    assert after.metrics.total("sglang:num_retracted_requests_total") == 1
    t1 = checks.check_no_preemption(before, after)
    assert t1.status is Status.FAIL
    assert t1.message == "1 retraction during the run (SGLang logs a warning for each)"
    assert "KV cache pool is full. Retract requests. #retracted_reqs: 1" in (
        d / "boot.log"
    ).read_text(encoding="utf-8")
    s = series.read(d / "run.series.jsonl")
    assert {x.preemptions for x in s.samples} == {0.0, 1.0}  # the chart marks it
