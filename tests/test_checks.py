from __future__ import annotations

from pathlib import Path

import pytest

from conftest import fixture_text
from inferlint import bootlog, checks, series, telemetry
from inferlint.result import Status, TripwireFailed


def load(fx: Path, name: str) -> telemetry.Snapshot:
    return telemetry.load(fx / "metrics" / name)


def test_t1_preempted_run_fails(fx: Path) -> None:
    r = checks.check_no_preemption(
        load(fx, "preempted_before.prom"), load(fx, "preempted_after.prom")
    )
    assert r.status is Status.FAIL
    assert r.evidence["preemptions"] == 60.0


def test_t1_same_snapshot_passes(fx: Path) -> None:
    s = load(fx, "preempted_after.prom")
    assert checks.check_no_preemption(s, s).status is Status.PASS


def test_t1_restart_fails_even_though_counters_rose(fx: Path) -> None:
    a, b = load(fx, "restart_before.prom"), load(fx, "restart_after.prom")
    k = "vllm:generation_tokens_total"
    before, after = a.metrics.total(k), b.metrics.total(k)
    assert before is not None and after is not None and after > before  # looks like progress
    r = checks.check_no_preemption(a, b)
    assert r.status is Status.FAIL
    assert r.evidence["restarted"] is True


def test_t1_missing_counter_is_unknown_and_raises_by_default(fx: Path) -> None:
    s = load(fx, "preempted_after.prom")
    stripped = telemetry.Snapshot(
        url="x",
        text="\n".join(ln for ln in s.text.splitlines() if "num_preemptions" not in ln),
        t_wall=0.0,
        t_mono_ns=0,
    )
    r = checks.check_no_preemption(stripped, stripped)
    assert r.status is Status.UNKNOWN
    with pytest.raises(TripwireFailed):
        r.raise_for_status()
    r.raise_for_status(allow_unknown=True)


def test_t1_the_log_says_nothing() -> None:
    # The run in this log preempted; vLLM 0.28 wrote no line about it at the default level.
    assert "preempt" not in fixture_text("run_preempted_silently.log").lower()


def test_t4_same_flags_different_pool() -> None:
    hi = bootlog.parse(fixture_text("boot_pool_level_hi.log"))
    lo = bootlog.parse(fixture_text("boot_pool_level_lo.log"))
    [r] = checks.check_same_pool([hi, lo], ["hi", "lo"])
    assert r.status is Status.WARN
    assert r.evidence["levels"] == [12405, 13575]


def test_t4_different_flags_are_not_compared() -> None:
    a = bootlog.parse(fixture_text("boot_eager.log"))
    b = bootlog.parse(fixture_text("boot_pool_level_lo.log"))
    assert checks.check_same_pool([a, b]) == []


def test_t4_failed_boot_is_left_out() -> None:
    lo = bootlog.parse(fixture_text("boot_pool_level_lo.log"))
    dead = bootlog.parse(fixture_text("boot_fail_accelerator.log"))
    assert checks.check_same_pool([lo, dead, lo], ["a", "dead", "b"])[0].evidence[
        "not_compared"
    ] == ["dead"]


def test_t5_block_size_forced() -> None:
    r = checks.check_block_size(bootlog.parse(fixture_text("boot_eager.log")))
    assert r.status is Status.WARN
    assert r.evidence["block_size"] == 784


def test_t6_override_detected_and_honoured_case_passes() -> None:
    bad = checks.check_backend_honoured(
        bootlog.parse(fixture_text("boot_spec_backend_override.log"))
    )
    good = checks.check_backend_honoured(bootlog.parse(fixture_text("boot_fp8_triton.log")))
    assert (
        bad.status is Status.FAIL
        and "draft model (speculative decoding) picked FLASHINFER" in bad.message
    )
    assert good.status is Status.PASS and good.message == "TRITON_ATTN was requested and used"


def test_t6_nothing_requested_says_there_was_nothing_to_check() -> None:
    r = checks.check_backend_honoured(bootlog.parse(fixture_text("boot_cudagraphs.log")))
    assert r.status is Status.PASS
    assert r.message.startswith("nothing to check: the server was started without")
    assert "vLLM chose its own (FLASH_ATTN)" in r.message


def test_t7_boot_ok_then_dies() -> None:
    text = fixture_text("boot_spec_backend_override.log")
    assert bootlog.parse(text).ready  # a boot-only gate would have passed this
    r = checks.check_boot_failure(text)
    assert (r.tripwire, r.status, r.evidence["phase"]) == ("T7", Status.FAIL, "serving")


def test_t11_kv_memory_differs_between_identical_boots() -> None:
    hi = bootlog.parse(fixture_text("boot_pool_level_hi.log"))
    lo = bootlog.parse(fixture_text("boot_pool_level_lo.log"))
    [r] = checks.check_kv_memory_stable([hi, lo], ["hi", "lo"])
    assert r.status is Status.WARN


def test_t12_named_failure() -> None:
    r = checks.check_boot_failure(fixture_text("boot_fail_accelerator.log"))
    assert (r.tripwire, r.status, r.evidence["kind"]) == ("T12", Status.FAIL, "accelerator_error")


def test_t8_t9_ladder(fx: Path) -> None:
    s = series.read(fx / "series" / "ladder.series.jsonl")
    t8 = checks.check_concurrency_reached(s, 32)
    t9 = checks.concurrency_ceiling(s)
    assert t8.status is Status.FAIL and t8.evidence["peak_running"] == 4.0
    # fixed-length requests: the same ceiling on every saturated sample, = what was measured
    assert t9.status is Status.PASS
    assert t9.evidence["ceiling_min"] == t9.evidence["ceiling_max"] == 4


def test_t8_boot_line_alongside(fx: Path) -> None:
    s = series.read(fx / "series" / "ladder.series.jsonl")
    facts = bootlog.parse(fixture_text("boot_eager.log"))
    r = checks.check_concurrency_reached(s, 4, facts)
    assert r.status is Status.PASS
    assert r.evidence["boot_line_max_concurrency"] == 1.93


def test_t14_null_block(fx: Path) -> None:
    for stem, reported in (("blocks", 45), ("ladder", 53)):
        s = series.read(fx / "series" / f"{stem}.series.jsonl")
        snap = telemetry.load(fx / "series" / f"{stem}_snapshot.prom")
        r = checks.check_null_block(s, snap)
        assert r.status is Status.PASS, r.message
        assert r.evidence["inferred_usable_blocks"] == reported - 1


def test_t14_mismatched_series_and_snapshot_fail(fx: Path) -> None:
    s = series.read(fx / "series" / "blocks.series.jsonl")
    other_boot = telemetry.load(fx / "series" / "ladder_snapshot.prom")
    assert checks.check_null_block(s, other_boot).status is Status.FAIL


def test_t10_precise_rate(fx: Path) -> None:
    r = checks.precise_rate(
        telemetry.load(fx / "metrics" / "preempted_before.snapshot.json"),
        telemetry.load(fx / "metrics" / "preempted_after.snapshot.json"),
    )
    assert r.status is Status.PASS
    assert r.evidence["span_s"] == pytest.approx(68.251, abs=1e-3)
    assert r.evidence["integer_second_span_s"] in (68, 69)


def test_t10_integer_second_timer_error() -> None:
    # 2,899 tokens in 28.99 s. Integer-second stamps record 28 or 29 depending on where in
    # a second the run started, so the same run reads +3.5% or -0.03%.
    txt0 = 'vllm:generation_tokens_total{engine="0"} 0\n'
    txt1 = 'vllm:generation_tokens_total{engine="0"} 2899\n'
    seen: dict[object, object] = {}
    for start in (1000.005, 1000.5):
        a = telemetry.Snapshot("u", txt0, t_wall=start, t_mono_ns=None)
        b = telemetry.Snapshot("u", txt1, t_wall=start + 28.99, t_mono_ns=None)
        r = checks.precise_rate(a, b)
        assert r.evidence["rate_per_s"] == pytest.approx(100.0)
        seen[r.evidence["integer_second_span_s"]] = r.evidence["integer_second_error"]
    assert set(seen) == {28, 29}
    assert seen[28] == pytest.approx(28.99 / 28 - 1)
    assert seen[29] == pytest.approx(28.99 / 29 - 1)
