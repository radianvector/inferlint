"""Boot log -> structured facts about the server that actually started.

What a benchmark needs to know about a boot is printed once, in prose, and nowhere else:
the KV pool it drew, the attention block size it was forced to, the attention backend it
actually selected (not the one requested), and how much memory CUDA graphs took.

Every field is looked for in two steps. An **anchor** (a fixed substring) says the line
exists; a **pattern** extracts the value. If the anchor is found but no pattern matches,
the line goes into ``unparsed``. That separates "this server did not print it" from
"it printed it in a format this parser does not know", which a version bump will produce
sooner or later. A field is never filled with a default.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .engines import by_key, from_log

__all__ = [
    "TESTED_VLLM",
    "BootFacts",
    "Unparsed",
    "labels",
    "parse",
    "parse_file",
    "strip_log_prefix",
    "untested_version",
]

# vLLM release series checked live on a GPU (the README's support table lists each check).
TESTED_VLLM: tuple[str, ...] = ("0.28", "0.29", "0.30")

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# "(EngineCore pid=3475) ERROR 09-05 23:33:11 [core.py:1348] message"
_PREFIX = re.compile(
    r"^\((?P<proc>[A-Za-z_][\w:]*?)\s+pid=(?P<pid>\d+)\)\s*"
    r"(?:(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?\s+"
    r"\[[^\]]*\]\s?)?"
)


def strip_log_prefix(line: str) -> str:
    return _PREFIX.sub("", line, count=1)


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _int(s: str) -> int:
    return int(s.replace(",", ""))


@dataclass(frozen=True)
class _Field:
    name: str
    anchor: re.Pattern[str]
    patterns: tuple[re.Pattern[str], ...]
    convert: Callable[[re.Match[str]], Any]


def _f(
    name: str,
    anchor: str,
    patterns: Sequence[str],
    convert: Callable[[re.Match[str]], Any],
    *,
    anchor_is_regex: bool = False,
) -> _Field:
    rx = re.compile(anchor if anchor_is_regex else re.escape(anchor))
    return _Field(name, rx, tuple(re.compile(p) for p in patterns), convert)


def _literal(m: re.Match[str]) -> Any:
    return ast.literal_eval(m.group(1))


# Patterns are listed newest-first per field when a format changes between releases.
_FIELDS: tuple[_Field, ...] = (
    _f(
        "vllm_version",
        "Initializing a V1 LLM engine",
        [r"Initializing a V1 LLM engine \(v(?P<v>[^)\s]+)\)"],
        lambda m: m.group("v"),
    ),
    _f("non_default_args", "non-default args:", [r"non-default args: (\{.*\})\s*$"], _literal),
    _f(
        "selected_backend",
        "attention backend out of potential backends",
        [r"Using (?P<b>\w+) attention backend out of potential backends: \[(?P<c>[^\]]*)\]"],
        lambda m: (m.group("b"), tuple(x.strip(" '\"") for x in m.group("c").split(","))),
    ),
    # When a backend is requested explicitly there is no candidate list, only this line.
    # Anchored on " backend." at the end so the vision-encoder lines
    # ("Using AttentionBackendEnum.X for MMEncoderAttention.") are not mistaken for it.
    _f(
        "selected_backend",
        r"Using AttentionBackendEnum\.\w+ backend\.",
        [r"Using AttentionBackendEnum\.(?P<b>\w+) backend\.\s*$"],
        lambda m: (m.group("b"), ()),
        anchor_is_regex=True,
    ),
    _f(
        "attention_block_size",
        "Setting attention block size to",
        [r"Setting attention block size to (\d+) tokens"],
        lambda m: int(m.group(1)),
    ),
    _f(
        "mamba_page_padding_pct",
        "Padding mamba page size by",
        [r"Padding mamba page size by ([\d.]+)%"],
        lambda m: float(m.group(1)),
    ),
    _f(
        "kv_pool",
        "GPU KV cache size:",
        [
            r"GPU KV cache size: (?P<tok>[\d,]+) tokens, Maximum concurrency for "
            r"(?P<len>[\d,]+) tokens per request: (?P<x>[\d.]+)x"
        ],
        lambda m: (_int(m.group("tok")), _int(m.group("len")), float(m.group("x"))),
    ),
    _f(
        "available_kv_cache_gib",
        "Available KV cache memory:",
        [r"Available KV cache memory: ([\d.]+) GiB"],
        lambda m: _num(m.group(1)),
    ),
    _f(
        "model_load",
        "Model loading took",
        [r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds"],
        lambda m: (_num(m.group(1)), _num(m.group(2))),
    ),
    _f(
        "cudagraph_estimated_gib",
        "Estimated CUDA graph memory:",
        [r"Estimated CUDA graph memory: ([\d.]+) GiB total"],
        lambda m: _num(m.group(1)),
    ),
    _f(
        "cudagraph_actual_gib",
        "CUDA graph pool memory:",
        [r"CUDA graph pool memory: ([\d.]+) GiB \(actual\)"],
        lambda m: _num(m.group(1)),
    ),
    _f(
        "cudagraph_mode_downgrade",
        "is not supported with spec-decode",
        [r"(\w+) is not supported with spec-decode .*setting cudagraph_mode=(\w+)"],
        lambda m: (m.group(1), m.group(2)),
    ),
    # ---- SGLang (verified on 0.5.20)
    _f("server_args", "server_args={", [r"server_args=(\{.*\})\s*$"], _literal),
    _f(
        "sglang_kv_pool",
        "KV Cache is allocated.",
        [
            r"KV Cache is allocated\. dtype: \S+, #tokens: (?P<tok>\d+), "
            r"K size: (?P<k>[\d.]+) GB, V size: (?P<v>[\d.]+) GB"
        ],
        lambda m: (int(m.group("tok")), round(float(m.group("k")) + float(m.group("v")), 2)),
    ),
    _f(
        "model_load",
        "Load weight end.",
        [r"Load weight end\. elapsed=(?P<s>[\d.]+) s,.* mem usage=(?P<g>[\d.]+) GB"],
        lambda m: (_num(m.group("g")), _num(m.group("s"))),
    ),
    _f(
        "sglang_cudagraph_gib",
        r"CUDA graph end\.",
        [r"CUDA graph end\. elapsed=[\d.]+ s, mem usage=(?P<g>[\d.]+) GB"],
        lambda m: _num(m.group("g")),
        anchor_is_regex=True,
    ),
    _f(
        "sglang_running_cap_reason",
        "max_running_requests is capped to",
        [r"max_running_requests is capped to (?P<n>\d+) by the (?P<why>[^(]+?) \("],
        lambda m: (int(m.group("n")), m.group("why").strip()),
    ),
    _f(
        "sglang_limits",
        "max_total_num_tokens=",
        [r"max_total_num_tokens=(?P<tok>\d+),.* max_running_requests=(?P<n>\d+)"],
        lambda m: (int(m.group("tok")), int(m.group("n"))),
    ),
    _f(
        "sglang_default_backend",
        "Attention backend not specified.",
        [r"Attention backend not specified\. Use (?P<b>\w+) backend by default"],
        lambda m: m.group("b"),
    ),
)
# Values SGLang draws afresh at every start, so not part of a start's configuration.
_SGLANG_VOLATILE_ARGS = frozenset({"random_seed"})
# Fields a log prints more than once and that add up (one CUDA graph line per phase).
_SUMMED = frozenset({"sglang_cudagraph_gib"})

_READY_MARKERS = ("Application startup complete",)
# Backend selections after this line belong to the speculative-decoding drafter.
_DRAFTER_MARKER = "Loading drafter model"
_EAGER_MARKER = "Cudagraph is disabled under eager mode"


@dataclass(frozen=True)
class Unparsed:
    field: str
    lineno: int
    line: str


@dataclass
class BootFacts:
    engine: str | None = None  # "vllm", "sglang"; None when the log does not say
    vllm_version: str | None = None
    # The version the server reported over HTTP, for engines whose log does not print it.
    server_version: str | None = None
    non_default_args: dict[str, Any] | None = None
    server_args: dict[str, Any] | None = None  # SGLang prints every argument
    page_size: int | None = None  # SGLang's KV allocation unit, in tokens
    # The most requests the server will run at once, after its own adjustments, and why
    # it is lower than asked when it is (SGLang: "the mamba state cache").
    running_cap: int | None = None
    running_cap_reason: str | None = None
    requested_running: int | None = None
    requested_backend: str | None = None
    selected_backends: list[str] = field(default_factory=list[str])  # target model
    drafter_backends: list[str] = field(default_factory=list[str])  # speculative drafter
    candidate_backends: tuple[str, ...] = ()
    attention_block_size: int | None = None
    mamba_page_padding_pct: float | None = None
    kv_pool_tokens: int | None = None
    max_concurrency: float | None = None
    max_concurrency_request_len: int | None = None
    available_kv_cache_gib: float | None = None
    model_load_gib: float | None = None
    model_load_s: float | None = None
    cudagraphs: bool | None = None
    cudagraph_estimated_gib: float | None = None
    cudagraph_actual_gib: float | None = None
    cudagraph_mode_downgrade: tuple[str, str] | None = None
    speculative: bool | None = None
    ready: bool = False
    ready_lineno: int | None = None
    processes: dict[str, list[int]] = field(default_factory=dict[str, list[int]])
    unparsed: list[Unparsed] = field(default_factory=list[Unparsed])
    # field -> every distinct value seen, when a log holds more than one (two boots
    # appended to one file). The last value wins; the conflict is kept visible.
    conflicts: dict[str, list[Any]] = field(default_factory=dict[str, list[Any]])
    lines: int = 0

    @property
    def version(self) -> str | None:
        """The engine's version: from the log (vLLM) or from the server (SGLang)."""
        return self.vllm_version or self.server_version

    @property
    def config_args(self) -> dict[str, Any] | None:
        """The arguments that identify a start's configuration, for comparing starts.

        vLLM prints only its non-default arguments; SGLang prints all of them, less the
        ones it draws afresh at every start.
        """
        if self.non_default_args is not None:
            return self.non_default_args
        if self.server_args is not None:
            return {k: v for k, v in self.server_args.items() if k not in _SGLANG_VOLATILE_ARGS}
        return None

    def result_block(self) -> dict[str, Any]:
        """The ``boot`` block of an ``rv.result/1`` document."""
        return {
            "engine": self.engine,
            "runtime_version": self.version,
            "kv_pool_tokens": self.kv_pool_tokens,
            "attention_block_size": self.attention_block_size,
            "backend_requested": self.requested_backend,
            "backend_selected": self.selected_backends[-1] if self.selected_backends else None,
            "drafter_backend": self.drafter_backends[-1] if self.drafter_backends else None,
            "available_kv_cache_gib": self.available_kv_cache_gib,
            "cudagraph_actual_gib": self.cudagraph_actual_gib,
            "cudagraphs": self.cudagraphs,
            "speculative": self.speculative,
            "unparsed_lines": len(self.unparsed),
        }


def _record(
    facts: BootFacts, name: str, value: Any, seen: dict[str, list[Any]], in_drafter: bool
) -> None:
    vals = seen.setdefault(name, [])
    if value not in vals:
        vals.append(value)
    if name == "selected_backend":
        backend, candidates = value
        (facts.drafter_backends if in_drafter else facts.selected_backends).append(backend)
        if candidates and not in_drafter:
            facts.candidate_backends = candidates
    elif name == "kv_pool":
        facts.kv_pool_tokens, facts.max_concurrency_request_len, facts.max_concurrency = value
    elif name == "model_load":
        facts.model_load_gib, facts.model_load_s = value
    elif name == "sglang_kv_pool":
        facts.kv_pool_tokens, facts.available_kv_cache_gib = value
    elif name == "sglang_cudagraph_gib":
        facts.cudagraph_actual_gib = round((facts.cudagraph_actual_gib or 0.0) + value, 2)
    elif name == "sglang_running_cap_reason":
        facts.running_cap, facts.running_cap_reason = value
    elif name == "sglang_limits":
        facts.kv_pool_tokens, facts.running_cap = value
    elif name == "sglang_default_backend":
        facts.selected_backends.append(value)
    else:
        setattr(facts, name, value)


def parse(text: str) -> BootFacts:
    facts = BootFacts()
    seen: dict[str, list[Any]] = {}
    procs: dict[str, set[int]] = {}
    lineno = 0
    in_drafter = False
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = _ANSI.sub("", raw)
        if _DRAFTER_MARKER in line:
            in_drafter = True
        pm = _PREFIX.match(line)
        if pm:
            procs.setdefault(pm.group("proc"), set()).add(int(pm.group("pid")))
        for fld in _FIELDS:
            if not fld.anchor.search(line):
                continue
            for pat in fld.patterns:
                m = pat.search(line)
                if m is None:
                    continue
                try:
                    value = fld.convert(m)
                except (ValueError, SyntaxError):
                    continue
                _record(facts, fld.name, value, seen, in_drafter)
                break
            else:
                facts.unparsed.append(Unparsed(fld.name, lineno, strip_log_prefix(line)[:300]))
        if _EAGER_MARKER in line:
            facts.cudagraphs = False
        if not facts.ready and any(k in line for k in _READY_MARKERS):
            facts.ready = True
            facts.ready_lineno = lineno
    facts.lines = lineno
    facts.processes = {k: sorted(v) for k, v in procs.items()}
    # selected_backend legitimately repeats (one line per attention group); not a conflict.
    facts.conflicts = {
        k: v for k, v in seen.items() if len(v) > 1 and k != "selected_backend" and k not in _SUMMED
    }
    engine = from_log(text)
    facts.engine = engine.key if engine is not None else None

    if facts.server_args is not None:  # SGLang
        sa = facts.server_args
        facts.page_size = sa.get("page_size") if isinstance(sa.get("page_size"), int) else None
        mrr = sa.get("max_running_requests")
        facts.requested_running = mrr if isinstance(mrr, int) else None
        default_backend = "sglang_default_backend" in seen
        rb = sa.get("attention_backend")
        facts.requested_backend = None if default_backend or rb is None else str(rb)
        if not facts.selected_backends and rb is not None:
            facts.selected_backends.append(str(rb))
        facts.speculative = sa.get("speculative_algorithm") is not None
        if facts.cudagraphs is None:
            facts.cudagraphs = not sa.get("disable_cuda_graph", False)

    args = facts.non_default_args or {}
    if facts.non_default_args is not None:
        rb = args.get("attention_backend")
        facts.requested_backend = None if rb is None else str(rb)
        facts.speculative = args.get("speculative_config") is not None
        if facts.cudagraphs is None and args.get("enforce_eager"):
            facts.cudagraphs = False
    if facts.cudagraphs is None and facts.cudagraph_actual_gib is not None:
        facts.cudagraphs = True
    return facts


def parse_file(path: str | Path) -> BootFacts:
    return parse(Path(path).read_text(encoding="utf-8", errors="replace"))


def labels(paths: Sequence[str | Path]) -> list[str]:
    """Short, distinct names for boot logs: the file name, with folders added as needed.

    Two runs' logs are often both called boot.log; keyed by file name alone, a comparison
    of them would see one log, not two.
    """
    parts = [Path(p).parts for p in paths]
    for depth in range(1, max((len(x) for x in parts), default=1) + 1):
        out = ["/".join(x[-depth:]) for x in parts]
        if len(set(out)) == len(out):
            return out
    return [str(p) for p in paths]


def untested_version(facts: BootFacts) -> str | None:
    """A warning when the log comes from a release series inferlint was not tested on.

    Log lines and metric names change between releases. A check that cannot parse says
    Can't tell, but a format that still parses with a different meaning would not, so a
    new version is flagged up front.
    """
    v = facts.version
    if v is None:
        return None
    engine = by_key(facts.engine)
    if any(v == t or v.startswith(t + ".") for t in engine.tested):
        return None
    return (
        f"{engine.name} {v} is not a tested version (tested: {', '.join(engine.tested)}). "
        "If a check says Can't tell or a value looks wrong, a log line or metric may have "
        "changed: please report it."
    )
