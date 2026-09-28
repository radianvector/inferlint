from __future__ import annotations

import math

import pytest

from inferlint.prom import parse

TEXT = r"""
# HELP vllm:num_preemptions_total Cumulative number of preemption from the engine.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{engine="0",model_name="m"} 5.0
vllm:num_preemptions_created{engine="0",model_name="m"} 1.7e9
vllm:num_requests_running{engine="0",model_name="m"} 3.0
vllm:num_requests_running{engine="1",model_name="m"} 2.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25
vllm:request_success_total{engine="0",finished_reason="length",model_name="m"} 7.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="m"} 3.0
vllm:cache_config_info{block_size="416",engine="0",kv_cache_dtype_skip_layers="[]",num_gpu_blocks="53",odd="a\"b,c=d"} 1.0
vllm:e2e_request_latency_seconds_bucket{engine="0",le="+Inf",model_name="m"} 10.0
weird_nan NaN
with_ts 4 1700000000000
"""


def test_absent_is_none_not_zero() -> None:
    m = parse(TEXT)
    assert m.total("vllm:spec_decode_num_accepted_tokens_total") is None
    assert m.single("vllm:spec_decode_num_accepted_tokens_total") is None


def test_total_sums_matching_series() -> None:
    m = parse(TEXT)
    assert m.total("vllm:num_requests_running") == 5.0
    assert m.total("vllm:num_requests_running", engine="1") == 2.0
    assert m.total("vllm:request_success_total") == 10.0
    assert m.total("vllm:request_success_total", finished_reason="stop") == 3.0


def test_single_refuses_ambiguous_ratio() -> None:
    m = parse(TEXT)
    with pytest.raises(LookupError):
        m.single("vllm:num_requests_running")
    assert m.single("vllm:kv_cache_usage_perc") == 0.25


def test_info_labels_with_escapes_and_commas() -> None:
    info = parse(TEXT).info("vllm:cache_config_info")
    assert info is not None
    assert info["num_gpu_blocks"] == "53"
    assert info["kv_cache_dtype_skip_layers"] == "[]"
    assert info["odd"] == 'a"b,c=d'


def test_special_values_and_timestamps() -> None:
    m = parse(TEXT)
    v = m.single("weird_nan")
    assert v is not None and math.isnan(v)
    assert m.single("with_ts") == 4.0
    assert m.total("vllm:e2e_request_latency_seconds_bucket", le="+Inf") == 10.0


def test_malformed_line_raises() -> None:
    with pytest.raises(ValueError, match="not a Prometheus sample"):
        parse('vllm:x{engine="0"}\n')
    with pytest.raises(ValueError, match="malformed label"):
        parse("vllm:x{engine=0} 1\n")


def test_real_fixture_parses(fx) -> None:  # type: ignore[no-untyped-def]
    m = parse((fx / "metrics" / "preempted_after.prom").read_text(encoding="utf-8"))
    assert len(m) > 300
    assert m.total("vllm:num_preemptions_total") == 166.0
