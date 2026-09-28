from __future__ import annotations

import pytest

from conftest import fixture_text
from inferlint.failures import Kind, Phase, classify, shutdown_lineno


def test_accelerator_error_is_named() -> None:
    # The failure a ValueError|RuntimeError grep records with a blank reason.
    f = classify(fixture_text("boot_fail_accelerator.log"))
    assert f is not None
    assert f.kind is Kind.ACCELERATOR_ERROR
    assert f.detail == "CUDA error: device not ready"
    assert f.phase is Phase.BOOT


def test_crash_on_first_request_is_serving_phase() -> None:
    f = classify(fixture_text("boot_spec_backend_override.log"))
    assert f is not None
    assert f.kind is Kind.JIT_TOOLCHAIN
    assert f.phase is Phase.SERVING  # the boot itself looked healthy


HEALTHY_STOPPED_FROM_OUTSIDE = [
    "boot_eager.log",
    "boot_cudagraphs.log",
    "boot_fp8_triton.log",
    "run_preempted_silently.log",
]


@pytest.mark.parametrize("name", [*HEALTHY_STOPPED_FROM_OUTSIDE, "boot_pool_level_hi.log"])
def test_healthy_runs_have_no_failure(name: str) -> None:
    assert classify(fixture_text(name)) is None


@pytest.mark.parametrize("name", HEALTHY_STOPPED_FROM_OUTSIDE)
def test_shutdown_cutoff_is_what_keeps_them_clean(name: str) -> None:
    # Mutation: hide the shutdown marker and the teardown tracebacks become "failures".
    text = fixture_text(name)
    assert shutdown_lineno(text) is not None
    assert classify(text.replace("[shutdown]", "[stopping]")) is not None


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        (
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
            Kind.OUT_OF_MEMORY,
        ),
        (
            "ValueError: To serve at least one request with the models's max seq len (32768), "
            "(4.0 GiB KV cache is needed, which is larger than the available KV cache memory "
            "(1.2 GiB).",
            Kind.KV_CACHE_TOO_SMALL,
        ),
        ("RuntimeError: CUDA error: an illegal memory access was encountered", Kind.CUDA_ERROR),
        ("AssertionError", Kind.ASSERTION),
        ("ValueError: unsupported quantization", Kind.VALUE_ERROR),
        ("RuntimeError: something else", Kind.RUNTIME_ERROR),
        ("KeyError: 'model.layers.0.weight'", Kind.UNCLASSIFIED_EXCEPTION),
        (
            "vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.",
            Kind.ENGINE_DEAD,
        ),
    ],
)
def test_rules(line: str, kind: Kind) -> None:
    text = f"(EngineCore pid=1) ERROR 09-05 10:57:48 [core.py:1] {line}\n"
    f = classify(text)
    assert f is not None and f.kind is kind


def test_root_cause_beats_engine_dead() -> None:
    text = (
        "(APIServer pid=1) ERROR 01-01 00:00:00 [x.py:1] EngineDeadError: EngineCore died\n"
        "(EngineCore pid=2) ERROR 01-01 00:00:00 [x.py:1] torch.OutOfMemoryError: CUDA out of memory\n"
    )
    f = classify(text)
    assert f is not None and f.kind is Kind.OUT_OF_MEMORY


def test_clean_log() -> None:
    assert classify("(APIServer pid=1) INFO:     Application startup complete.\n") is None
