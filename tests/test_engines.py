"""Which engine wrote a file, and what each engine's real start-up failures look like."""

from __future__ import annotations

from pathlib import Path

import pytest

from inferlint import engines, telemetry
from inferlint.failures import Kind, classify, classify_file
from inferlint.prom import parse
from inferlint.series import sample_row
from inferlint.teardown import Proc, server_engine, usual_pkill_pattern

FX = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "path",
    sorted(FX.glob("*/**/*.log")),
    ids=lambda p: str(p.relative_to(FX)),
)
def test_every_fixture_log_is_told_apart(path: Path) -> None:
    """FlashInfer builds kernels under .../tensorrt_llm/, so a vLLM log can mention it."""
    want = {"vllm": engines.VLLM, "sglang": engines.SGLANG, "trtllm": engines.TRTLLM}
    folder = path.relative_to(FX).parts[0].split("-")[0]
    assert engines.from_log(path.read_text(encoding="utf-8")) is want[folder]


def test_from_names() -> None:
    assert engines.from_names(["vllm:num_requests_running", "python_info"]) is engines.VLLM
    assert engines.from_names(["sglang:num_running_reqs"]) is engines.SGLANG
    assert engines.from_names(["trtllm_num_requests_running"]) is engines.TRTLLM
    assert engines.from_names(["process_cpu_seconds_total"]) is engines.VLLM  # unknown
    assert engines.by_key("trtllm") is engines.TRTLLM and engines.by_key(None) is engines.VLLM


def test_trtllm_failures_are_named() -> None:
    # Weight-only compressed-tensors (W4A16) checkpoint: TensorRT-LLM reads the activation
    # settings that are not there. The function it failed in says what it was doing.
    f = classify_file(FX / "trtllm-1.3" / "boot_fail_compressed_tensors_w4a16.log")
    assert f is not None and f.kind is Kind.UNCLASSIFIED_EXCEPTION
    assert f.detail == (
        "'NoneType' object is not subscriptable "
        "(in update_quant_config_from_compressed_tensors, quant_config_utils.py:81)"
    )
    # No CUDA_HOME: a bare assert, then an ImportError that blames PyTorch.
    f = classify_file(FX / "trtllm-1.3" / "boot_fail_no_cuda_home.log")
    assert f is not None and f.kind is Kind.ASSERTION
    assert f.detail == "assert cuda_home is not None"


def test_an_error_the_server_kept_serving_after_is_not_a_crash() -> None:
    ready = "INFO:     Application startup complete."
    err = "TimeoutError: timed out"
    served = 'INFO:     127.0.0.1:1 - "POST /v1/completions HTTP/1.1" 200 OK'
    assert classify("\n".join([ready, "post-warmup freeze_gc failed", err, served])) is None
    f = classify("\n".join([ready, err]))
    assert f is not None and f.phase.value == "serving"


def test_trtllm_processes() -> None:
    serve = Proc(5, 1, ("/opt/venv/bin/python3", "/opt/venv/bin/trtllm-serve", "serve", "m"), "")
    assert server_engine(serve) == "trtllm"
    assert usual_pkill_pattern([serve]) == "trtllm-serve"


def test_scrape_falls_back_to_the_prometheus_path() -> None:
    """TensorRT-LLM answers /metrics with JSON and serves Prometheus text elsewhere."""
    pages = {
        "http://x:1/metrics": '[{"iter": 1}]',
        "http://x:1/prometheus/metrics": "trtllm_num_requests_running 3.0\n",
    }
    s = telemetry.scrape("http://x:1", fetch=lambda url, timeout: pages[url])
    assert s.url == "http://x:1/prometheus/metrics"
    assert s.metrics.total("trtllm_num_requests_running") == 3.0
    row = sample_row(parse(s.text), 0.0, None)
    assert row["running"] == 3.0 and "preemptions" not in row


def test_a_servers_descendants_are_its_processes() -> None:
    """TensorRT-LLM's model runs in a generically named MPI worker under trtllm-serve."""
    from inferlint.teardown import server_processes

    procs = [
        Proc(100, 1, ("bash", "run.sh"), "bash"),  # the script that started it, and us
        Proc(200, 100, ("/v/bin/python", "/v/bin/trtllm-serve", "serve", "m"), "trtllm-serve"),
        Proc(300, 200, ("prte", "--singleton", "singleton.host.200.0"), "prte"),
        Proc(400, 300, ("/v/bin/python", "-R", "-m", "mpi4py.futures.server"), "python"),
        Proc(500, 1, ("/v/bin/python", "-R", "-m", "mpi4py.futures.server"), "python"),  # other
        Proc(600, 100, ("python3", "-m", "inferlint.cli", "teardown"), "python3"),  # us
    ]
    assert [p.pid for p in server_processes(procs, self_pid=600)] == [200, 300, 400]
