"""What limited a run, on recorded runs of all three engines; and the CLI around it."""

from __future__ import annotations

from pathlib import Path

import pytest

from inferlint import benchresult, engines, insights, report, series, telemetry
from inferlint.cli import main

FX = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _no_engine_in_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(engines.ENV_VAR, raising=False)


def build(run: str, *, bench: bool = True) -> report.Report:
    d = FX / run
    return report.build(
        boot_logs=[d / "boot.log"],
        before=telemetry.load(d / "before.snapshot.json"),
        after=telemetry.load(d / "after.snapshot.json"),
        series=series.read(d / "run.series.jsonl"),
        requested=32,
        bench=benchresult.load(d / "bench.json") if bench and (d / "bench.json").exists() else None,
    )


def said(run: str, **kw: bool) -> dict[str, insights.Insight]:
    return {i.key: i for i in insights.limits(build(run, **kw))}


# Qwen3-8B, 32 requests of 88 + 1,000 tokens at once: the load needs 34,816 tokens of cache.


def test_a_cache_too_small_and_what_each_engine_did() -> None:
    vllm = said("vllm-0.30/qwen3-8b-start1")["cache"]
    assert vllm.tone == "limit"
    assert "holds 32,336 tokens; 32 requests of 1,088 tokens need 34,816" in vllm.text
    assert "so 29 fit at full length. To make room, it preempted 3 running requests." in vllm.text
    sg = said("sglang-0.5/qwen3-8b")["cache"]
    assert sg.text.endswith("To make room, it ran at most 29 at once and queued the rest.")
    trt = said("trtllm-1.3/pauses")["cache"]
    assert "holds 20,960 tokens" in trt.text and trt.text.endswith("paused 13 running requests.")
    both = said("sglang-0.5/retraction")["cache"]
    assert both.text.endswith("queued the rest, and retracted 1 running request.")


def test_a_cache_that_holds_the_load() -> None:
    for run, pool, share in (
        ("vllm-0.30/qwen3-8b-start2", "47,888", "73%"),
        ("trtllm-1.3/qwen3-8b", "37,760", "92%"),
    ):
        got = said(run)
        assert got["cache"].tone == "ok" and f"holds {pool} tokens" in got["cache"].text
        assert got["cache"].text.endswith(f"need 34,816, {share} of it.")
        assert got["waiting"].tone == "ok" and got["pace"].tone == "ok"


def test_the_pace_says_how_many_ran_on_average() -> None:
    """The recording's average running count, and the load tool's tokens/s; no ceiling."""
    pace = said("sglang-0.5/qwen3-8b")["pace"]
    assert pace.title == "On average 17 of 32 requests ran at once"
    assert pace.text == (
        "The load kept 32 requests open; the server's recording shows 16.8 running on average "
        "while it was busy. The load tool measured 799 output tokens/s over the whole run."
    )
    assert "%" not in pace.title + pace.text and "possible" not in pace.text
    # TensorRT-LLM's gauges lag: counted from tokens instead, and called what it is
    trt = said("trtllm-1.3/pauses")["pace"]
    assert trt.title == "On average 24 of 32 requests were part-way through their output"
    assert "so on average 24.4 of the 32 requests the load kept open were in that stretch" in (
        trt.text
    )
    assert "(1,006 x 0.0243 s), including any paused mid-output." in trt.text
    assert trt.text.endswith("so the recording does not give the number running.")


def test_waiting_long_is_told_from_waiting_for_others() -> None:
    sg = said("sglang-0.5/qwen3-8b")["waiting"]
    assert sg.title == "Some requests waited much longer to start"
    assert "within 641 ms, but the slowest (p99) waited 22.1 s: they were queued" in sg.text
    capped = said("sglang-0.5/capped")["waiting"]  # everyone waited: median 285 s
    assert capped.title == "Requests waited long to start" and "285.2 s" in capped.text


def test_hybrid_models_get_the_concurrency_sentence_instead() -> None:
    """Their memory per request is not KV per token, so the cache arithmetic is left out."""
    capped = said("sglang-0.5/capped")
    assert "cache" not in capped and capped["concurrency"].tone == "limit"
    assert "ran at most 1 at once: SGLang lowered its limit to 1 running because of the mamba" in (
        capped["concurrency"].text
    )
    assert "the server's recording shows 1.0 running on average" in capped["pace"].text
    vllm = said("vllm-0.30/live")
    assert "cache" not in vllm
    assert vllm["concurrency"].text.endswith(
        "ran at most 11 at once. It also preempted 22 running requests."
    )


def test_without_the_load_tools_result() -> None:
    """The server's own counts: queue time, and the pace while it was busy."""
    got = said("vllm-0.28/live", bench=False)
    assert got["waiting"].text == "A request waited 29.3 s on average before the server started it."
    assert got["pace"].text.endswith(
        "while the server was busy, its recording shows 8.0 running on average and 343 "
        "output tokens/s."
    )


def test_at_most_three_and_nothing_after_a_restart() -> None:
    rep = build("sglang-0.5/qwen3-8b")
    assert len(insights.limits(rep)) == 3
    rep.summary.restarted = True
    assert insights.limits(rep) == []


def test_the_report_shows_them_and_the_cache_chart() -> None:
    html = report.render(build("sglang-0.5/qwen3-8b"))
    assert html.index('id="h-limits"') < html.index('id="h-happened"')  # first on the page
    assert html.count('<li class="limit limit">') == 3
    assert 'id="fig-cache"' in html and "29 of them fit." in html
    assert 'id="fig-memory"' in html
    # passes, and T14, which does not apply, folded; the failure and warning shown
    assert "5 passed; 1 does not apply to SGLang</summary>" in html


# --------------------------------------------------------------------------- the CLI


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    got = capsys.readouterr()
    return code, got.out, got.err


def test_no_arguments_prints_where_to_start(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys)
    assert code == 0 and out.startswith("inferlint checks an inference benchmark run")
    assert "inferlint report run/" in out and "INFERLINT_ENGINE" in out


def test_a_mistyped_command_gets_a_suggestion(capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run(capsys, "xrya")
    assert code == 2 and "unknown command 'xrya'. Did you mean 'xray'?" in err
    code, _, err = run(capsys, "frobnicate")
    assert code == 2 and "Did you mean" not in err and "'inferlint --help' lists them" in err


def test_help_groups_the_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    groups = ["Run and check a benchmark", "Check a saved run", "The GPU", "Learn"]
    assert [out.index(g) for g in groups] == sorted(out.index(g) for g in groups)
    for name in ("xray", "report", "check-log", "teardown", "explain", "probe"):
        assert f"    {name} " in out


def test_report_reads_a_run_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    folder = tmp_path / "run"
    shutil.copytree(FX / "sglang-0.5" / "qwen3-8b", folder)
    code, out, _ = run(capsys, "report", str(folder))
    assert code == 0 and (folder / "report.html").is_file()
    # the result file was found: T15 ran, and the concurrency came from it
    assert "[T15] PASS  client and server agree: 32 requests" in out
    assert "requested 32 concurrent" in out
    assert "What limited this run:\n  - The KV cache was too small for the load." in out
    assert out.rstrip().endswith(f"-> {folder / 'report.html'}")
    made = (folder / "report.html").read_text(encoding="utf-8")
    assert (
        "boot.log, before.snapshot.json, after.snapshot.json, run.series.jsonl, bench.json" in made
    )


def test_report_says_what_it_needs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run(capsys, "report")
    assert code == 2 and "give a run folder, or -o" in err
    code, _, err = run(capsys, "report", str(tmp_path / "nope"))
    assert code == 2 and "is not a folder" in err
    code, _, err = run(capsys, "report", str(tmp_path))
    assert code == 2 and "no run files in" in err
