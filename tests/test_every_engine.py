"""Every command on a recorded run from each engine.

The same model, card and load on vLLM 0.30, SGLang 0.5.20 and TensorRT-LLM 1.3.0rc29:
Qwen3-8B in bf16 on an RTX 4090, 32 requests at once, 88 prompt and 1,000 output tokens
each, sent by ``vllm bench serve`` through ``inferlint xray``. Each engine also has a
second start with the same flags, for the checks that compare starts (T4, T11).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inferlint.cli import main
from test_report import Balance

FX = Path(__file__).parent / "fixtures"
RUNS = {
    "vllm": FX / "vllm-0.30" / "qwen3-8b-start1",
    "sglang": FX / "sglang-0.5" / "qwen3-8b",
    "trtllm": FX / "trtllm-1.3" / "qwen3-8b",
}
SECOND = {
    "vllm": FX / "vllm-0.30" / "qwen3-8b-start2",
    "sglang": FX / "sglang-0.5" / "qwen3-8b-start2",
    "trtllm": FX / "trtllm-1.3" / "qwen3-8b-start2",
}
POOL = {"vllm": 32336, "sglang": 32096, "trtllm": 37760}
ENGINES = pytest.mark.parametrize("engine", ["vllm", "sglang", "trtllm"])


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    code = main(list(argv))
    return code, capsys.readouterr().out


def files(engine: str, *names: str) -> list[str]:
    return [str(RUNS[engine] / n) for n in names]


@ENGINES
def test_boot_facts(engine: str, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run(capsys, "boot-facts", *files(engine, "boot.log"))
    assert code == 0
    assert f"engine                       {engine}" in out
    assert f"kv_pool_tokens               {POOL[engine]}" in out


@ENGINES
def test_check_log(engine: str, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run(capsys, "check-log", *files(engine, "boot.log"))
    assert code == 0
    for tw in ("[T12] PASS", "[ T5] PASS", "[ T6] PASS"):
        assert tw in out


@ENGINES
def test_check_log_compares_two_starts(engine: str, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run(
        capsys, "check-log", str(RUNS[engine] / "boot.log"), str(SECOND[engine] / "boot.log")
    )
    if engine == "vllm":
        # the first start compiled the model from scratch and left 4.44 GiB, not 6.58
        assert "[ T4] WARN" in out and "32,336 to 47,888 tokens" in out
        assert "[T11] WARN" in out and "1 of 2 starts compiled the model from scratch" in out
    else:
        assert code == 0
        assert f"[ T4] PASS  same flags, same pool ({POOL[engine]:,} tokens)" in out
        assert "[T11] PASS" in out


@ENGINES
def test_preemption(engine: str, capsys: pytest.CaptureFixture[str]) -> None:
    snaps = files(engine, "before.snapshot.json", "after.snapshot.json")
    code, out = run(capsys, "preemption", *snaps)
    if engine == "vllm":
        assert code == 1 and "[ T1] TRIPWIRE-FAILED  3 preemptions during the run" in out
    elif engine == "sglang":
        assert code == 0 and "[ T1] PASS  no retractions" in out
    else:
        # TensorRT-LLM counts no pauses; it logs each, so the command asks for the log
        assert code == 2 and "server log saved until after the run is needed" in out
        code, out = run(capsys, "preemption", *snaps, "--boot-log", *files(engine, "boot.log"))
        assert code == 0 and "no pause in the server log, which covers 41 answered" in out


@ENGINES
def test_rate(engine: str, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run(capsys, "rate", *files(engine, "before.snapshot.json", "after.snapshot.json"))
    assert code == 0 and "[T10] PASS" in out and "integer-second timer would say" in out


@ENGINES
def test_series(engine: str, capsys: pytest.CaptureFixture[str]) -> None:
    rec, after = files(engine, "run.series.jsonl", "after.snapshot.json")
    code, out = run(capsys, "series", rec, "--snapshot", after, "--requested", "32")
    if engine == "sglang":
        assert code == 1 and "server never ran more than 29" in out
    else:
        assert code == 0 and "[ T8] PASS  reached 32 running (requested 32)" in out
    assert ("[T14] PASS" in out) is (engine == "vllm")  # vLLM's reserved block only
    if engine == "trtllm":
        assert "updates its gauges only when a request completes" in out


@ENGINES
def test_report_from_the_run_folder(
    engine: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "report.html"
    code, text = run(capsys, "report", str(RUNS[engine]), "-o", str(out))
    assert code == 0
    assert "[T15] PASS  client and server agree: 32 requests, 32,000 output tokens" in text
    assert "What limited this run:" in text
    html = out.read_text(encoding="utf-8")
    p = Balance()
    p.feed(html)
    assert p.errors == [] and p.stack == [] and p.external == []
    assert 'id="limits"' in html or 'class="limits"' in html


@ENGINES
def test_explain_names_the_engine(engine: str, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run(capsys, "explain", "T1", "--engine", engine)
    assert code == 0 and "Silent preemption" in out
    others = {"sglang": "On SGLang", "trtllm": "On TensorRT-LLM"}
    for key, note in others.items():
        assert (note in out) is (key == engine)


OWN_CLIENT = {
    "sglang": FX / "sglang-0.5" / "qwen3-8b-own-client",
    "trtllm": FX / "trtllm-1.3" / "qwen3-8b-own-client",
}


@pytest.mark.parametrize("engine", ["sglang", "trtllm"])
def test_each_engines_own_benchmark_client(
    engine: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same server and load, sent by the engine's own client through ``xray``.

    SGLang's (``python -m sglang.benchmark.serving``, native ``/generate`` endpoint) appends
    its result to a JSON-lines file; TensorRT-LLM's writes vLLM's format. Each sent one
    request it leaves out of its result: a warm-up, and an initial test request.
    """
    d = OWN_CLIENT[engine]
    code, text = run(capsys, "report", str(d), "-o", str(tmp_path / "r.html"))
    assert code == 0
    assert (
        "[T15] PASS  client and server agree: 32 requests, 32,000 output tokens, plus 1 "
        "untimed request (test or warm-up)"
    ) in text
    assert "[ T8] PASS  reached 32 running (requested 32)" in text
    if engine == "sglang":
        # more at once than with vllm bench serve (29), and 3 retractions
        assert "[ T1] TRIPWIRE-FAILED  3 retractions during the run" in text
        assert "Warmup completed with 1 sequences" in (d / "load.log").read_text(encoding="utf-8")
    else:
        assert "[ T1] PASS  no pause in the server log" in text
        assert "Initial test run completed" in (d / "load.log").read_text(encoding="utf-8")


def test_the_reference_has_every_engine_note() -> None:
    """docs/tripwires.md carries the same per-engine notes as ``explain`` and the report."""
    from inferlint.catalog import ENGINE_NOTES
    from inferlint.engines import by_key

    doc = " ".join((FX.parents[1] / "docs" / "tripwires.md").read_text(encoding="utf-8").split())
    for tw, notes in ENGINE_NOTES.items():
        for key, note in notes.items():
            assert f"**On {by_key(key).name}:** {note}" in doc, (tw, key)
