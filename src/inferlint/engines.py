"""The serving engines inferlint knows, and how to tell which one produced a file.

Each engine has its own metric names, process names and log format. Everything that
depends on them asks this module, so the checks themselves are written once. A file whose
engine cannot be told is read as vLLM's, which is what inferlint read before it knew any
other engine.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from .metricnames import SGLANG as SGLANG_METRICS
from .metricnames import TRTLLM as TRTLLM_METRICS
from .metricnames import VLLM as VLLM_METRICS
from .metricnames import MetricNames

__all__ = ["ENGINES", "SGLANG", "TRTLLM", "VLLM", "Engine", "by_key", "from_log", "from_names"]


@dataclass(frozen=True)
class Engine:
    key: str  # "vllm", "sglang", "trtllm"
    name: str  # as the project writes it
    prefix: str  # what its metric names start with
    metrics: MetricNames
    tested: tuple[str, ...]  # release series checked live on a GPU
    preemption: str  # the engine's word for evicting a running request
    serve: str  # how its server is usually started, for messages
    # counter -> a series exported from the start, for counters the engine writes only
    # once they first count (see telemetry.counter_delta)
    witnesses: tuple[tuple[str, str], ...] = ()
    logs_preemptions: bool = False  # does it write a log line when it preempts?

    def witness(self, counter: str) -> str | None:
        return dict(self.witnesses).get(counter)


VLLM = Engine(
    key="vllm",
    name="vLLM",
    prefix="vllm:",
    metrics=VLLM_METRICS,
    tested=("0.28", "0.29", "0.30"),
    preemption="preemption",
    serve="vllm serve",
)
SGLANG = Engine(
    key="sglang",
    name="SGLang",
    prefix="sglang:",
    metrics=SGLANG_METRICS,
    tested=("0.5",),
    preemption="retraction",
    serve="sglang serve",
    # The retraction counter appears at the first retraction; the gauge from the start.
    witnesses=(("sglang:num_retracted_requests_total", "sglang:num_retracted_reqs"),),
    # "KV cache pool is full. Retract requests. #retracted_reqs: N" (WARNING)
    logs_preemptions=True,
)
TRTLLM = Engine(
    key="trtllm",
    name="TensorRT-LLM",
    prefix="trtllm_",
    metrics=TRTLLM_METRICS,
    tested=("1.3",),
    preemption="pause",
    serve="trtllm-serve",
)
ENGINES: tuple[Engine, ...] = (VLLM, SGLANG, TRTLLM)

# Lines only one engine prints at start-up, in the order they are tried.
_LOG_MARKERS: tuple[tuple[Engine, re.Pattern[str]], ...] = (
    # Not a bare "tensorrt_llm/": FlashInfer ships kernels under .../tensorrt_llm/ too.
    (
        TRTLLM,
        re.compile(
            r"\[TensorRT-LLM\]|\btrtllm-serve\b|\btensorrt_llm/(?:llmapi|serve|_torch|commands)/"
        ),
    ),
    (SGLANG, re.compile(r"\bserver_args=\{|\bsglang[./]|sglang::")),
    (VLLM, re.compile(r"Initializing a V1 LLM engine|\bvllm[./]|VLLM::|\(EngineCore pid=")),
)


def by_key(key: str | None) -> Engine:
    """The engine named ``key`` ("vllm", "sglang", "trtllm"); vLLM if ``None`` or unknown."""
    return next((e for e in ENGINES if e.key == key), VLLM)


def from_names(names: Iterable[str]) -> Engine:
    """The engine whose metric prefix most series carry; vLLM when none does."""
    counts = {e.key: 0 for e in ENGINES}
    for n in names:
        for e in ENGINES:
            if n.startswith(e.prefix):
                counts[e.key] += 1
    best = max(ENGINES, key=lambda e: counts[e.key])
    return best if counts[best.key] else VLLM


def from_log(text: str) -> Engine | None:
    """The engine that wrote a boot log, from lines only it prints; None if unknown."""
    head = text[:200_000]
    for engine, rx in _LOG_MARKERS:
        if rx.search(head):
            return engine
    return None
