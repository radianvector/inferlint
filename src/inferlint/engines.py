"""The serving engines inferlint knows, and how to tell which one produced a file.

Each engine has its own metric names, process names and log format. Everything that
depends on them asks this module, so the checks themselves are written once. A file whose
engine cannot be told is read as vLLM's, which is what inferlint read before it knew any
other engine, unless the user named the engine (``--engine`` or ``INFERLINT_ENGINE``).
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .metricnames import SGLANG as SGLANG_METRICS
from .metricnames import TRTLLM as TRTLLM_METRICS
from .metricnames import VLLM as VLLM_METRICS
from .metricnames import MetricNames

__all__ = [
    "ENGINES",
    "ENV_VAR",
    "SGLANG",
    "TRTLLM",
    "VLLM",
    "Engine",
    "by_key",
    "from_command",
    "from_log",
    "from_names",
    "named_by",
    "parse_key",
]

# Names the engine for every command, for a team that always runs the same one.
ENV_VAR = "INFERLINT_ENGINE"


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
    # Are its gauges updated only when a request completes, not at every step?
    gauges_lag: bool = False
    # What to change when the server serves no Prometheus metrics.
    metrics_hint: str = ""
    # Does its generated-token counter count a request's tokens only when it finishes?
    tokens_at_finish: bool = False
    # The port its server listens on when the command gives none.
    default_port: int = 8000
    # What to change when the server serves its counters but not its gauges.
    gauges_hint: str = ""

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
    metrics_hint="vLLM serves them at /metrics without a flag; check that the address is "
    "the server's",
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
    metrics_hint="start SGLang with --enable-metrics",
    default_port=30000,
    # 0.5.20: 3 changes of sglang:generation_tokens_total in 193 readings of a 32-request run
    tokens_at_finish=True,
)
TRTLLM = Engine(
    key="trtllm",
    name="TensorRT-LLM",
    prefix="trtllm_",
    metrics=TRTLLM_METRICS,
    tested=("1.3",),
    preemption="pause",
    serve="trtllm-serve",
    # "MaxUtilizationScheduler: request ID N -> pause", at INFO (the default level)
    logs_preemptions=True,
    # Its stats collector sleeps until a request completes, then logs every step since
    # (1.3.0rc29, serve/openai_server.py), so a reading shows the state as of the last
    # completion. Requests that finish together leave the gauges still until the end.
    gauges_lag=True,
    metrics_hint="put 'return_perf_metrics: true' and 'enable_iter_perf_stats: true' in the "
    "YAML file given to trtllm-serve --config; it then serves them at /prometheus/metrics",
    # 1.3.0rc29: without enable_iter_perf_stats it serves counters and histograms only
    gauges_hint="TensorRT-LLM serves its running, waiting and KV gauges only with "
    "'enable_iter_perf_stats: true' in the YAML file given to trtllm-serve --config",
    tokens_at_finish=True,  # 1.3.0rc29: trtllm_generation_tokens_total moves on completion
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


def parse_key(value: str) -> Engine | None:
    """The engine a user named: its key or its name, in any case ("trtllm", "TensorRT-LLM")."""
    v = value.strip().lower()
    return next((e for e in ENGINES if v in (e.key, e.name.lower())), None)


def named_by(names: Iterable[str]) -> Engine | None:
    """The engine whose metric prefix most series carry; None when no series has one."""
    counts = {e.key: 0 for e in ENGINES}
    for n in names:
        for e in ENGINES:
            if n.startswith(e.prefix):
                counts[e.key] += 1
    best = max(ENGINES, key=lambda e: counts[e.key])
    return best if counts[best.key] else None


def from_names(names: Iterable[str]) -> Engine:
    """The engine whose metric prefix most series carry; vLLM when none does."""
    return named_by(names) or VLLM


# `vllm serve`, `sglang serve`: a CLI followed by its "serve" subcommand.
_SERVE_CLIS = {"vllm": VLLM, "sglang": SGLANG}
# `python -m vllm.entrypoints.openai.api_server`, `python -m sglang.launch_server`
_MODULES = (("vllm", VLLM), ("sglang", SGLANG), ("tensorrt_llm", TRTLLM))


def from_command(cmd: str | Sequence[str] | None) -> Engine | None:
    """The engine a server command starts; None if it names none.

    Reads ``vllm serve``, ``sglang serve``, ``trtllm-serve``, and ``python -m`` with one of
    their modules, also behind a path or a prefix such as ``env X=1``.
    """
    if not cmd:
        return None
    try:
        argv = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
    except ValueError:  # unbalanced quotes
        return None
    for i, tok in enumerate(argv):
        name = tok.replace("\\", "/").rsplit("/", 1)[-1]
        nxt = argv[i + 1] if i + 1 < len(argv) else ""
        if name == "trtllm-serve":
            return TRTLLM
        if name in _SERVE_CLIS and nxt == "serve":
            return _SERVE_CLIS[name]
        if tok == "-m":
            for module, engine in _MODULES:
                if nxt == module or nxt.startswith(module + "."):
                    return engine
    return None


def from_log(text: str) -> Engine | None:
    """The engine that wrote a boot log, from lines only it prints; None if unknown."""
    head = text[:200_000]
    for engine, rx in _LOG_MARKERS:
        if rx.search(head):
            return engine
    return None
