"""Name why a server died, instead of recording a blank.

A failure grep that matches ``ValueError|RuntimeError`` misses
``torch.AcceleratorError: CUDA error: device not ready`` and writes an empty reason, and
a result whose failure has no reason is a lost result. The rules below run in priority
order: specific root causes first, generic exception types after, and the secondary
"engine is dead" message last, because it follows every root cause and explains none.

The classifier also reports *when* the server died: during boot, or after it announced
it was ready. The second is a config that looks healthy until the first request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .bootlog import strip_log_prefix

__all__ = ["Failure", "Kind", "Phase", "classify", "classify_file", "shutdown_lineno"]

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class Kind(str, Enum):
    JIT_TOOLCHAIN = "jit_toolchain"  # a kernel JIT build failed (nvcc/ninja/headers)
    OUT_OF_MEMORY = "out_of_memory"
    KV_CACHE_TOO_SMALL = "kv_cache_too_small"
    ACCELERATOR_ERROR = "accelerator_error"
    CUDA_ERROR = "cuda_error"
    ASSERTION = "assertion"
    VALUE_ERROR = "value_error"
    RUNTIME_ERROR = "runtime_error"
    UNCLASSIFIED_EXCEPTION = "unclassified_exception"
    ENGINE_DEAD = "engine_dead"


class Phase(str, Enum):
    BOOT = "boot"
    SERVING = "serving"  # died after announcing it was ready


_RULES: tuple[tuple[Kind, re.Pattern[str]], ...] = (
    # A kernel build's own error first; "Ninja build failed" only says that a step failed.
    (
        Kind.JIT_TOOLCHAIN,
        re.compile(r"(?P<d>CUDA compiler and CUDA toolkit headers are incompatible)"),
    ),
    (Kind.JIT_TOOLCHAIN, re.compile(r"\bptxas\b.*\bfatal\s*:\s*(?P<d>.+)")),
    (Kind.JIT_TOOLCHAIN, re.compile(r"\bnvcc fatal\s*:\s*(?P<d>.+)")),
    (Kind.JIT_TOOLCHAIN, re.compile(r"\bld: (?P<d>cannot find .+)")),
    (Kind.JIT_TOOLCHAIN, re.compile(r"Ninja build failed|ninja: build stopped")),
    (Kind.OUT_OF_MEMORY, re.compile(r"\bOutOfMemoryError\b|CUDA out of memory")),
    (
        Kind.KV_CACHE_TOO_SMALL,
        re.compile(
            r"No available memory for the cache blocks"
            r"|KV cache is needed, which is larger than the available KV cache memory"
        ),
    ),
    (Kind.ACCELERATOR_ERROR, re.compile(r"\bAcceleratorError: (?P<d>.+)")),
    (Kind.CUDA_ERROR, re.compile(r"\bCUDA error: (?P<d>.+)")),
    (Kind.ASSERTION, re.compile(r"^(?:\w+\.)*AssertionError\b:?(?P<d>.*)")),
    (Kind.VALUE_ERROR, re.compile(r"^(?:\w+\.)*ValueError: (?P<d>.+)")),
    (Kind.RUNTIME_ERROR, re.compile(r"^(?:\w+\.)*RuntimeError: (?P<d>.+)")),
    (
        Kind.UNCLASSIFIED_EXCEPTION,
        re.compile(r"^(?!.*EngineDeadError)(?:\w+\.)*\w*(?:Error|Exception): (?P<d>.+)"),
    ),
    (Kind.ENGINE_DEAD, re.compile(r"EngineDeadError|EngineCore encountered a fatal error")),
)
_READY = "Application startup complete"
# vLLM tags every step of an orderly stop with "[shutdown]". A server stopped from outside
# (SIGTERM from a teardown) then logs tracebacks from half-closed pipes: shutdown noise,
# not failures. A real crash is logged before the shutdown it triggers, so only the lines
# before the first marker are classified.
_SHUTDOWN = re.compile(r"\[shutdown\]")
# Tracebacks a server prints and says it ignores, such as SGLang's optional imports:
# "Ignore import error when loading ...", then "[start of X traceback]" ...
# "[end of X traceback]". They are not failures, and they can come before the real one.
# Work the server did after an error, which shows it survived it: a request answered
# with 200, or a batch run (SGLang logs one line per prefill and per decode batch).
_KEPT_SERVING = re.compile(r'"POST /\S+ HTTP/[\d.]+" 200\b|^(?:Prefill|Decode) batch\b')
_IGNORED_LINE = re.compile(r"^Ignore import error\b")
_IGNORED_START = re.compile(r"^\[start of .*traceback\]")
_IGNORED_END = re.compile(r"^\[end of .*traceback\]")


def _drop_ignored(lines: list[str]) -> list[str]:
    """Blank the lines of tracebacks the server says it ignores; line numbers are kept."""
    out: list[str] = []
    inside = False
    for ln in lines:
        if _IGNORED_START.match(ln):
            inside = True
        drop = inside or bool(_IGNORED_LINE.match(ln))
        if _IGNORED_END.match(ln):
            inside = False
        out.append("" if drop else ln)
    return out


@dataclass(frozen=True)
class Failure:
    kind: Kind
    phase: Phase
    lineno: int
    line: str
    detail: str

    def summary(self) -> str:
        return f"{self.kind.value} during {self.phase.value}: {self.detail or self.line}"


def shutdown_lineno(text: str) -> int | None:
    """Line number of the first orderly-shutdown marker, if the server was stopped."""
    for i, raw in enumerate(text.splitlines(), 1):
        if _SHUTDOWN.search(raw):
            return i
    return None


def _failed_assert(lines: list[str], lineno: int) -> str:
    """The ``assert`` statement a bare AssertionError came from, from the traceback above.

    A bare ``assert x is not None`` raises an AssertionError with no message, and the error
    that follows can blame something else entirely (TensorRT-LLM: "Decoding operators failed
    to load ... incompatibility between PyTorch and TensorRT-LLM" when CUDA_HOME is unset).
    """
    for ln in reversed(lines[max(0, lineno - 4) : lineno - 1]):
        if ln.startswith("assert "):
            return ln
    return ""


_FRAME = re.compile(r'^File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)')


def _raised_in(lines: list[str], lineno: int) -> str:
    """Where a generic exception was raised, from the last traceback frame above it.

    "'NoneType' object is not subscriptable" alone names no cause; the function it came
    from often does (TensorRT-LLM: ``update_quant_config_from_compressed_tensors``).
    """
    for ln in reversed(lines[max(0, lineno - 7) : lineno - 1]):
        m = _FRAME.match(ln)
        if m:
            name = m.group("file").replace("\\", "/").rsplit("/", 1)[-1]
            return f"in {m.group('func')}, {name}:{m.group('line')}"
    return ""


def classify(text: str) -> Failure | None:
    """The most specific failure in a log, or ``None`` if nothing failed.

    Lines after an orderly shutdown began are not considered (see ``_SHUTDOWN``), nor are
    tracebacks the server says it ignores. A server that became ready survived whatever
    it logged before that, so only the lines after its ready line can hold a failure, and
    such a failure is one while serving (T7). An error after which the server went on
    answering requests did not stop it either (SGLang: "post-warmup freeze_gc failed").
    """
    lines = [strip_log_prefix(_ANSI.sub("", raw)).strip() for raw in text.splitlines()]
    stop = shutdown_lineno(text)
    if stop is not None:
        lines = lines[: stop - 1]
    lines = _drop_ignored(lines)
    ready_at = next((i for i, ln in enumerate(lines, 1) if _READY in ln), None)
    for kind, pat in _RULES:
        for i, ln in enumerate(lines, 1):
            if ready_at is not None and i <= ready_at:
                continue
            m = pat.search(ln)
            if m is None:
                continue
            if ready_at is not None and any(_KEPT_SERVING.search(x) for x in lines[i:]):
                continue
            detail = (m.groupdict().get("d") or "").strip()
            if kind is Kind.ASSERTION and not detail:
                detail = _failed_assert(lines, i)
            if kind is Kind.UNCLASSIFIED_EXCEPTION:
                where = _raised_in(lines, i)
                detail = f"{detail} ({where})" if where else detail
            phase = Phase.SERVING if ready_at is not None and i > ready_at else Phase.BOOT
            return Failure(kind, phase, i, ln[:300], detail[:200])
    return None


def classify_file(path: str | Path) -> Failure | None:
    return classify(Path(path).read_text(encoding="utf-8", errors="replace"))
