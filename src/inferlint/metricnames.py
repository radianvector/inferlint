"""Every server metric inferlint reads, named in one place, per serving engine.

The checks, the recorder and the report refer to these fields, never to a literal series
name, so a release that renames a series, or another serving engine, changes this file
and nothing else. A field is ``None`` when an engine has no such series.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SGLANG", "TRTLLM", "VLLM", "MetricNames"]


@dataclass(frozen=True)
class MetricNames:
    # gauges, recorded during the run
    running: str
    waiting: str
    kv_usage: str  # fraction of the KV pool in use, 0 to 1
    # counters, compared between the before and after snapshots
    preemptions: str | None  # None: the engine counts no preemptions
    generation_tokens: str
    prompt_tokens: str
    request_success: str
    # histograms (read as _sum / _count)
    e2e_latency: str
    first_token: str
    queue_time: str
    # an *_info series whose labels carry the cache configuration
    cache_info: str | None = None
    # the label of ``request_success`` that says why a request finished
    finished_reason_label: str | None = None
    # gauges that describe the KV pool itself
    pool_tokens: str | None = None
    page_size: str | None = None
    # a gauge of requests paused for recompute, for engines that count none
    paused: str | None = None


VLLM = MetricNames(
    running="vllm:num_requests_running",
    waiting="vllm:num_requests_waiting",
    kv_usage="vllm:kv_cache_usage_perc",
    preemptions="vllm:num_preemptions_total",
    generation_tokens="vllm:generation_tokens_total",
    prompt_tokens="vllm:prompt_tokens_total",
    request_success="vllm:request_success_total",
    e2e_latency="vllm:e2e_request_latency_seconds",
    first_token="vllm:time_to_first_token_seconds",
    queue_time="vllm:request_queue_time_seconds",
    cache_info="vllm:cache_config_info",
    finished_reason_label="finished_reason",
)

# SGLang (verified on 0.5.20) exports these with --enable-metrics. It calls a preemption
# a retraction: a running request is sent back to the queue and its KV freed.
SGLANG = MetricNames(
    running="sglang:num_running_reqs",
    waiting="sglang:num_queue_reqs",
    kv_usage="sglang:token_usage",
    preemptions="sglang:num_retracted_requests_total",
    generation_tokens="sglang:generation_tokens_total",
    prompt_tokens="sglang:prompt_tokens_total",
    request_success="sglang:num_requests_total",
    e2e_latency="sglang:e2e_request_latency_seconds",
    first_token="sglang:time_to_first_token_seconds",
    queue_time="sglang:queue_time_seconds",
    pool_tokens="sglang:max_total_num_tokens",
    page_size="sglang:page_size",
)

# TensorRT-LLM (verified on 1.3.0rc29) serves these at /prometheus/metrics when started
# with return_perf_metrics; the gauges (running, waiting, KV use, paused) only with
# enable_iter_perf_stats as well. It pauses a request for recompute where vLLM preempts,
# and exports only how many are paused at each iteration, not a count of pauses.
TRTLLM = MetricNames(
    running="trtllm_num_requests_running",
    waiting="trtllm_num_requests_waiting",
    kv_usage="trtllm_kv_cache_utilization",
    preemptions=None,
    generation_tokens="trtllm_generation_tokens_total",
    prompt_tokens="trtllm_prompt_tokens_total",
    request_success="trtllm_request_success_total",
    e2e_latency="trtllm_e2e_request_latency_seconds",
    first_token="trtllm_time_to_first_token_seconds",
    queue_time="trtllm_request_queue_time_seconds",
    finished_reason_label="finished_reason",
    paused="trtllm_num_paused_requests",
)
