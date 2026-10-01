"""The same live run on each tested vLLM release.

Same model, flags and load as the vLLM 0.28 run in ``test_live.py``: 32 concurrent
requests of 88 prompt and 1,000 output tokens (``vllm bench serve``, random dataset,
ignore_eos), CUDA graphs on, an RTX 4090. 0.29 and 0.30 were recorded by a script running
the commands one at a time (``live/``); 0.28 by ``inferlint xray`` (``xray/``), a second
run on that release. These tests pin what each tripwire reported on each release, which is
what the README's tested-versions table rests on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inferlint import benchresult, bootlog, checks, series, telemetry
from inferlint.failures import Kind, classify_file
from inferlint.result import Status

FX = Path(__file__).parent / "fixtures"

# version -> (KV pool tokens, CUDA graph GiB, usable blocks, ceiling min, ceiling max,
#             preemptions, peak running)
RUNS = {
    "0.28": (26093, 0.18, 42, 5, 10, 21, 10),
    "0.29": (26093, 0.18, 42, 5, 10, 21, 10),
    "0.30": (29127, 0.08, 47, 7, 11, 22, 11),
}


def _live(ver: str) -> Path:
    return FX / f"vllm-{ver}" / ("xray" if ver == "0.28" else "live")


@pytest.mark.parametrize("ver", RUNS)
def test_boot_facts_parse_completely(ver: str) -> None:
    pool, graphs, *_ = RUNS[ver]
    f = bootlog.parse_file(_live(ver) / "boot.log")
    assert f.vllm_version is not None and f.vllm_version.startswith(ver)
    assert (f.kv_pool_tokens, f.attention_block_size, f.cudagraph_actual_gib) == (pool, 784, graphs)
    assert f.unparsed == [] and f.ready
    assert bootlog.untested_version(f) is None


@pytest.mark.parametrize("ver", RUNS)
def test_tripwires_on_the_recorded_run(ver: str) -> None:
    _, _, usable, lo, hi, preempted, peak = RUNS[ver]
    d = _live(ver)
    before = telemetry.load(d / "before.snapshot.json")
    after = telemetry.load(d / "after.snapshot.json")
    s = series.read(d / "run.series.jsonl")

    t1 = checks.check_no_preemption(before, after)
    assert (t1.status, t1.evidence["preemptions"]) == (Status.FAIL, preempted)
    assert "preempt" not in (d / "boot.log").read_text(encoding="utf-8").lower()

    t14 = checks.check_null_block(s, after)
    assert t14.status is Status.PASS and t14.evidence["inferred_usable_blocks"] == usable

    t9 = checks.concurrency_ceiling(s, usable)
    assert t9.status is Status.WARN
    assert (t9.evidence["ceiling_min"], t9.evidence["ceiling_max"]) == (lo, hi)

    t8 = checks.check_concurrency_reached(s, 32)
    assert t8.status is Status.FAIL and t8.evidence["peak_running"] == peak

    t10 = checks.precise_rate(before, after)
    assert t10.status is Status.PASS and t10.evidence["tokens"] == 32 * 1000

    boot = (d / "boot.log").read_text(encoding="utf-8")
    assert checks.check_boot_failure(boot).status is Status.PASS


@pytest.mark.parametrize("ver", RUNS)
def test_t15_client_and_server_agree(ver: str) -> None:
    d = _live(ver)
    bench = benchresult.load(d / "bench.json")
    assert (bench.completed, bench.output_tokens, bench.max_concurrency) == (32, 32_000, 32)
    r = checks.check_client_server_agree(
        bench,
        telemetry.load(d / "before.snapshot.json"),
        telemetry.load(d / "after.snapshot.json"),
    )
    assert r.status is Status.PASS, r.message
    assert r.evidence["extra_requests"] == 0  # the benchmark skipped its test request


@pytest.mark.parametrize(
    ("ver", "name", "kind", "cause"),
    [
        (
            "0.28",
            "cccl_headers",
            Kind.JIT_TOOLCHAIN,
            "CUDA compiler and CUDA toolkit headers are incompatible",
        ),
        (
            "0.30",
            "cccl_headers",
            Kind.JIT_TOOLCHAIN,
            "CUDA compiler and CUDA toolkit headers are incompatible",
        ),
        (
            "0.30",
            "ptx_version",
            Kind.JIT_TOOLCHAIN,
            "Unsupported .version 9.4; current version is '9.0'",
        ),
        (
            "0.30",
            "link_cudart",
            Kind.JIT_TOOLCHAIN,
            "cannot find -lcudart: No such file or directory",
        ),
        ("0.29", "uva_wsl", Kind.RUNTIME_ERROR, "UVA is not available"),
    ],
)
def test_t12_names_real_start_up_failures(ver: str, name: str, kind: Kind, cause: str) -> None:
    """Real failed starts while setting up these runs (WSL, pip-installed CUDA): FlashInfer's
    kernel build three ways on 0.30 and once on 0.28 (its sampler, with nvcc 13.3 against
    CUDA 13.0 headers), and 0.29's new model runner, which needs pinned memory."""
    f = classify_file(FX / f"vllm-{ver}" / f"boot_fail_{name}.log")
    assert f is not None and (f.kind, f.detail) == (kind, cause)


def test_t8_says_admission_control_rejected_the_rest() -> None:
    """vLLM 0.30 with --max-num-queued-reqs 16 and 32 requests asked for at once."""
    d = FX / "vllm-0.30" / "capped"
    facts = bootlog.parse_file(d / "boot.log")
    assert facts.non_default_args is not None
    assert facts.non_default_args["max_num_queued_reqs"] == 16
    s = series.read(d / "run.series.jsonl")
    t8 = checks.check_concurrency_reached(s, 32, facts)
    assert t8.status is Status.FAIL
    assert "admission control (--max-num-queued-reqs 16)" in t8.message
    assert "the other 16 were rejected, not queued" in t8.message
    # Without the boot log's limit, the same recording reads as an ordinary full cache.
    plain = checks.check_concurrency_reached(s, 32)
    assert plain.status is Status.FAIL and "admission control" not in plain.message
    # The benchmark counted the 16 rejections as failed; the server finished the other 16.
    bench = benchresult.load(d / "bench.json")
    assert (bench.completed, bench.failed) == (16, 16)
    t15 = checks.check_client_server_agree(
        bench, telemetry.load(d / "before.snapshot.json"), telemetry.load(d / "after.snapshot.json")
    )
    assert t15.status is Status.PASS and t15.evidence["server_finished"] == 16


def test_two_starts_with_the_same_flags_drew_pools_48_percent_apart() -> None:
    """Qwen3-8B (bf16) on vLLM 0.30, same flags and load, started twice 20 minutes apart.

    The first start compiled the model from scratch (torch.compile 20.7 s) and vLLM counted
    1.14 GiB more consumed memory and a 1 GiB higher activation peak; the second loaded the
    compiled graph. The KV pool was 32,336 tokens, then 47,888. With the smaller pool the
    32 requests were preempted 3 times; with the larger one, not at all.
    """
    d1, d2 = FX / "vllm-0.30" / "qwen3-8b-start1", FX / "vllm-0.30" / "qwen3-8b-start2"
    f1, f2 = bootlog.parse_file(d1 / "boot.log"), bootlog.parse_file(d2 / "boot.log")
    assert (f1.kv_pool_tokens, f2.kv_pool_tokens) == (32336, 47888)
    assert (f1.compiled_fresh, f2.compiled_fresh) == (True, False)
    assert (f1.consumed_gib, f2.consumed_gib) == (16.56, 15.42)
    assert (f1.peak_activation_gib, f2.peak_activation_gib) == (1.3, 0.31)
    (t4,) = checks.check_same_pool([f1, f2], ["start1", "start2"])
    assert t4.status is Status.WARN and "48.10% apart" in t4.message
    (t11,) = checks.check_kv_memory_stable([f1, f2], ["start1", "start2"])
    assert t11.message == (
        "same flags, KV cache memory varied: 4.44 to 6.58 GiB (weights and non-torch memory "
        "15.42 to 16.56 GiB; peak activation 0.31 to 1.3 GiB; 1 of 2 starts compiled the "
        "model from scratch)"
    )
    pre = [
        checks.check_no_preemption(
            telemetry.load(d / "before.snapshot.json"), telemetry.load(d / "after.snapshot.json")
        ).evidence["preemptions"]
        for d in (d1, d2)
    ]
    assert pre == [3, 0]
