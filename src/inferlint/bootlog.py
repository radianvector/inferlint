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

__all__ = ["BootFacts", "Unparsed", "parse", "parse_file", "strip_log_prefix"]

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
)

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
    vllm_version: str | None = None
    non_default_args: dict[str, Any] | None = None
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

    def result_block(self) -> dict[str, Any]:
        """The ``boot`` block of an ``rv.result/1`` document."""
        return {
            "runtime_version": self.vllm_version,
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
    facts.conflicts = {k: v for k, v in seen.items() if len(v) > 1 and k != "selected_backend"}

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
