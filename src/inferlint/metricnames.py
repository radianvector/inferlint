"""Every server metric inferlint reads, named in one place.

The checks, the recorder and the report refer to these fields, never to a literal series
name, so a release that renames a series, or a second serving engine, changes this file
and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["VLLM", "MetricNames"]


@dataclass(frozen=True)
class MetricNames:
    # gauges, recorded during the run
    running: str
    waiting: str
    kv_usage: str
    # counters, compared between the before and after snapshots
    preemptions: str
    generation_tokens: str
    prompt_tokens: str
    request_success: str
    # histograms (read as _sum / _count)
    e2e_latency: str
    first_token: str
    queue_time: str
    # an *_info series whose labels carry the cache configuration
    cache_info: str


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
)
