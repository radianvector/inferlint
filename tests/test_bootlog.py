from __future__ import annotations

import pytest

from conftest import fixture_text
from inferlint import bootlog


def test_eager_hybrid_boot() -> None:
    f = bootlog.parse(fixture_text("boot_eager.log"))
    assert f.vllm_version == "0.28.0"
    assert f.kv_pool_tokens == 31554
    assert f.max_concurrency == 1.93
    assert f.max_concurrency_request_len == 16384
    assert f.attention_block_size == 784
    assert f.mamba_page_padding_pct == 0.13
    assert f.available_kv_cache_gib == 2.51
    assert f.selected_backends == ["FLASH_ATTN"]
    assert f.requested_backend is None
    assert f.cudagraphs is False
    assert f.speculative is False
    assert f.ready is True
    assert f.non_default_args is not None and f.non_default_args["max_num_seqs"] == 32
    assert set(f.processes) == {"APIServer", "EngineCore"}
    assert f.unparsed == []
    assert f.conflicts == {}


def test_cudagraph_memory() -> None:
    f = bootlog.parse(fixture_text("boot_cudagraphs.log"))
    assert f.cudagraphs is True
    assert f.cudagraph_estimated_gib == 0.20
    assert f.cudagraph_actual_gib == 0.18
    assert f.kv_pool_tokens == 26093


def test_backend_requested_and_overridden() -> None:
    f = bootlog.parse(fixture_text("boot_spec_backend_override.log"))
    assert f.requested_backend == "TRITON_ATTN"
    assert f.selected_backends == ["TRITON_ATTN"]  # the target model honours it...
    assert f.drafter_backends == ["FLASHINFER"]  # ...the drafter picks its own
    assert f.speculative is True
    assert f.cudagraph_mode_downgrade == ("FULL_AND_PIECEWISE", "PIECEWISE")


def test_backend_requested_and_honoured_uses_other_line_format() -> None:
    f = bootlog.parse(fixture_text("boot_fp8_triton.log"))
    assert f.requested_backend == "TRITON_ATTN"
    # the vision-encoder lines name FLASH_ATTN and must not be read as the LLM's backend
    assert f.selected_backends == ["TRITON_ATTN"]
    assert f.unparsed == []


def test_failed_boot_has_no_pool_and_no_default() -> None:
    f = bootlog.parse(fixture_text("boot_fail_accelerator.log"))
    assert f.ready is False
    assert f.kv_pool_tokens is None
    assert f.attention_block_size is None


def test_unknown_format_is_reported_not_defaulted() -> None:
    text = fixture_text("boot_eager.log").replace(
        "GPU KV cache size: 31,554 tokens, Maximum concurrency for 16,384 tokens per request: 1.93x",
        "GPU KV cache size: 31.5k tokens (1.93x concurrency)",
    )
    f = bootlog.parse(text)
    assert f.kv_pool_tokens is None
    assert [u.field for u in f.unparsed] == ["kv_pool"]
    assert "31.5k" in f.unparsed[0].line


def test_absent_line_is_not_unparsed() -> None:
    text = "\n".join(
        ln for ln in fixture_text("boot_eager.log").splitlines() if "GPU KV cache size" not in ln
    )
    f = bootlog.parse(text)
    assert f.kv_pool_tokens is None
    assert f.unparsed == []


def test_truncated_args_line_is_unparsed() -> None:
    text = fixture_text("boot_eager.log").replace("'max_num_seqs': 32}", "'max_num_seqs': 3")
    f = bootlog.parse(text)
    assert f.non_default_args is None
    assert [u.field for u in f.unparsed] == ["non_default_args"]


def test_two_boots_in_one_file_are_a_visible_conflict() -> None:
    text = fixture_text("boot_pool_level_hi.log") + fixture_text("boot_pool_level_lo.log")
    f = bootlog.parse(text)
    assert f.kv_pool_tokens == 12405  # the last value wins, and the conflict stays visible
    assert {v[0] for v in f.conflicts["kv_pool"]} == {13575, 12405}


@pytest.mark.parametrize(
    "name",
    [
        "boot_eager.log",
        "boot_cudagraphs.log",
        "boot_spec_backend_override.log",
        "boot_fail_accelerator.log",
        "boot_fp8_triton.log",
        "boot_pool_level_hi.log",
        "boot_pool_level_lo.log",
        "run_preempted_silently.log",
    ],
)
def test_every_fixture_parses_cleanly(name: str) -> None:
    f = bootlog.parse(fixture_text(name))
    assert f.vllm_version == "0.28.0"
    assert f.unparsed == [], f.unparsed


def test_result_block_shape() -> None:
    b = bootlog.parse(fixture_text("boot_spec_backend_override.log")).result_block()
    assert b["backend_requested"] == "TRITON_ATTN"
    assert b["backend_selected"] == "TRITON_ATTN"
    assert b["drafter_backend"] == "FLASHINFER"
    assert b["kv_pool_tokens"] == 19988
    assert b["unparsed_lines"] == 0
