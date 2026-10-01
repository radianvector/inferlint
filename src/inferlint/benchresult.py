"""The load tool's own account of a run, read from its result file.

Each engine's benchmark client can save one: ``vllm bench serve --save-result`` and
TensorRT-LLM's ``benchmark_serving --save-result`` write one JSON document per run;
SGLang's ``python -m sglang.benchmark.serving --output-file`` appends one JSON line per
run, and the last line is the latest run. All three use the same field names. inferlint
reads the counts it needs to compare the client's view with the server's (T15) and the
concurrency the client asked for (T8), and the latencies the client measured, which the
report uses to say what limited the run. Every field is None when the file does not have
it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

__all__ = ["BenchResult", "load", "parse"]


@dataclass(frozen=True)
class BenchResult:
    path: str
    completed: int | None = None
    failed: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_s: float | None = None
    output_throughput: float | None = None
    max_concurrency: int | None = None
    num_prompts: int | None = None
    backend: str | None = None
    # client-side latencies, in milliseconds
    mean_ttft_ms: float | None = None
    median_ttft_ms: float | None = None
    p99_ttft_ms: float | None = None
    mean_tpot_ms: float | None = None


def _int(doc: dict[str, Any], key: str) -> int | None:
    v = doc.get(key)
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _float(doc: dict[str, Any], key: str) -> float | None:
    v = doc.get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def parse(doc: dict[str, Any], path: str = "") -> BenchResult:
    backend = doc.get("backend")
    return BenchResult(
        path=path,
        completed=_int(doc, "completed"),
        failed=_int(doc, "failed"),
        input_tokens=_int(doc, "total_input_tokens"),
        output_tokens=_int(doc, "total_output_tokens"),
        duration_s=_float(doc, "duration"),
        output_throughput=_float(doc, "output_throughput"),
        max_concurrency=_int(doc, "max_concurrency"),
        num_prompts=_int(doc, "num_prompts"),
        backend=backend if isinstance(backend, str) else None,
        mean_ttft_ms=_float(doc, "mean_ttft_ms"),
        median_ttft_ms=_float(doc, "median_ttft_ms"),
        p99_ttft_ms=_float(doc, "p99_ttft_ms"),
        mean_tpot_ms=_float(doc, "mean_tpot_ms"),
    )


def load(path: str | Path) -> BenchResult:
    text = Path(path).read_text(encoding="utf-8")
    try:
        doc = json.loads(text)
    except ValueError:
        # JSON lines, one per run (SGLang's client appends): the last one is the latest run
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            raise
        doc = json.loads(lines[-1])
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: not a benchmark result (expected a JSON object)")
    return parse(cast(dict[str, Any], doc), str(path))
