from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree as ET

import pytest

from inferlint import report, series, telemetry
from inferlint.catalog import TRIPWIRES
from inferlint.cli import main
from inferlint.result import Status

FX = Path(__file__).parent / "fixtures" / "vllm-0.28"
LIVE = FX / "live"
VOID = {"meta", "br", "hr", "img", "input", "link", "col", "area", "base", "source", "wbr"}
SVG_VOID = {"line", "path", "rect", "circle", "stop"}


def live_report() -> report.Report:
    return report.build(
        boot_logs=[LIVE / "boot.log"],
        before=telemetry.load(LIVE / "before.snapshot.json"),
        after=telemetry.load(LIVE / "after.snapshot.json"),
        series=series.read(LIVE / "run.series.jsonl"),
        requested=32,
        title="Live run",
    )


class Balance(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.external: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for k, v in attrs:
            if k in ("src", "href") and v and re.match(r"(https?:)?//", v):
                self.external.append(v)
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.stack.pop()

    def handle_endtag(self, tag: str) -> None:
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> with open {self.stack[-3:]}")
            return
        self.stack.pop()


def test_live_report_runs_every_applicable_check() -> None:
    rep = live_report()
    got = {r.tripwire: r.status for r in rep.results}
    assert got == {
        "T12": Status.PASS,
        "T5": Status.WARN,
        "T6": Status.PASS,
        "T1": Status.FAIL,
        "T10": Status.PASS,
        "T14": Status.PASS,
        "T9": Status.WARN,
        "T8": Status.FAIL,
    }
    assert rep.usable_blocks == 42


def test_html_is_well_formed_and_self_contained() -> None:
    html = report.render(live_report())
    p = Balance()
    p.feed(html)
    assert p.errors == [], p.errors[:3]
    assert p.stack == [], p.stack
    assert p.external == []  # nothing fetched: works offline
    assert "@import" not in html and "url(http" not in html


def test_fragment_has_no_document_skeleton() -> None:
    frag = report.render(live_report(), standalone=False)
    assert frag.startswith("<title>")
    assert "<!doctype" not in frag.lower() and "<body" not in frag and "<html" not in frag


def test_charts_and_tables_present() -> None:
    html = report.render(live_report())
    assert html.count("<figure") == 4
    assert html.count("<details>") == 4  # every chart has a data table
    assert html.count('class="chart-data"') == 3  # time charts carry hover data


def test_preemption_markers_match_the_counter() -> None:
    rep = live_report()
    html = report.render(rep)
    labels = re.findall(r'<path class="ev"[^>]*><title>(\d+) preemptions? at', html)
    per_chart = sum(int(n) for n in labels) // 2  # the same events on two charts
    assert per_chart == 22


def test_every_finding_and_the_whole_guide_render() -> None:
    html = report.render(live_report())
    for r in live_report().results:
        assert f'<span class="tw">{r.tripwire}</span>' in html
    for t in TRIPWIRES.values():
        assert t.name.replace("'", "&#x27;") in html


def test_status_never_rests_on_colour_alone() -> None:
    html = report.render(live_report())
    for label in ("Tripwire failed", "Warning", "Pass"):
        assert f"</svg>{label}</span>" in html


def test_theme_tokens_for_all_three_states() -> None:
    html = report.render(live_report())
    assert ":root{" in html
    assert '@media (prefers-color-scheme:dark){:root:not([data-theme="light"])' in html
    assert ':root[data-theme="dark"]' in html
    assert "body{margin:0;background:var(--page)" in html


def test_summary_says_what_happened() -> None:
    rep = live_report()
    sm = rep.summary
    assert sm.done == 32 and sm.finished["length"] == 32 and sm.finished["error"] == 0
    assert (sm.output_tokens, sm.prompt_tokens, sm.preemptions) == (32_000, 2_819, 22)
    assert sm.peak_running == 10 and sm.interval_s == 0.25
    assert sm.span_s == pytest.approx(93.72, abs=0.01)
    assert sm.mean_queue_s == pytest.approx(936.91 / 32, abs=0.01)
    assert sm.boot_s == 124.0  # 17:48:43 to 17:50:47, the last stamped line before ready
    html = report.render(rep)
    assert "The load tool kept 32 requests open for 1 min 34 s." in html
    assert "finished 32 requests with no errors and generated 32,000 tokens" in html
    assert "never worked on more than 10 at once" in html
    assert "32,000 (1,000 per request)" in html
    assert "32: 32 reached their token limit, no errors" in html
    assert (
        "8 of 15. Not here: T2 and T3 (run at clean-up), T4 and T11 (need two or more "
        "boot logs), T7 (needs a live server), T13 (a rule for test scripts), T15 (needs "
        "the load tool&#x27;s result file)"
    ) in html


def test_a_fail_is_explained_as_a_measurement_finding() -> None:
    html = report.render(live_report())
    assert "The server worked; the failed tripwires are about the measurement." in html
    assert "The server finished 32 requests with no errors." in html


def test_errors_in_the_counters_change_the_summary() -> None:
    rep = live_report()
    after = rep.after
    assert after is not None
    marker = 'finished_reason="error",model_name="qwen38"} 0.0'
    assert marker in after.text
    # A new Snapshot, not dataclasses.replace: replace would share the parsed-metrics cache.
    text = after.text.replace(marker, marker[:-3] + "3.0")
    broken = telemetry.Snapshot(after.url, text, after.t_wall, after.t_mono_ns, after.latency_s)
    rep2 = report.build(
        before=rep.before, after=broken, series=rep.series, requested=32, title="Live run"
    )
    html = report.render(rep2)
    assert rep2.summary.finished["error"] == 3
    assert "(3 of them with an error)" in html
    assert "with no errors" not in html


def test_a_crash_tripwire_says_the_server_failed() -> None:
    html = report.render(report.build(boot_logs=[FX / "boot_fail_accelerator.log"]))
    assert "T12: the server itself failed." in html


def test_restart_between_readings_drops_the_counts() -> None:
    rep = report.build(
        before=telemetry.load(FX / "metrics" / "restart_before.prom"),
        after=telemetry.load(FX / "metrics" / "restart_after.prom"),
    )
    assert rep.summary.restarted and rep.summary.done is None
    assert "restarted between the before and after readings" in report.render(rep)


def test_glossaries_explain_every_result_and_the_terms() -> None:
    html = report.render(live_report())
    assert html.count('<details class="gloss"') == 2
    table = html.split('<table class="meanings">', 1)[1].split("</table>", 1)[0]
    for label in ("Tripwire failed", "Warning", "Can't tell", "Pass"):
        assert f"</svg>{label}</span>" in table
    for term in ("Token", "KV cache", "KV pool", "KV block", "Preemption", "Concurrency"):
        assert f"<dt>{term}</dt>" in html


def test_theme_switch_and_favicon() -> None:
    html = report.render(live_report())
    assert html.count("data-theme-toggle") == 1 + html.count("[data-theme-toggle]")  # one button
    assert 'class="ico-moon"' in html and 'class="ico-sun"' in html  # it shows the theme
    assert "inferlint-theme" in html  # the choice is remembered
    href = re.search(r'<link rel="icon" href="([^"]+)">', html)
    assert href is not None and href.group(1).startswith("data:image/svg+xml,")
    svg = ET.fromstring(unquote(href.group(1).split(",", 1)[1]))
    assert svg.tag.endswith("svg")
    frag = report.render(live_report(), standalone=False)
    assert 'rel="icon"' not in frag  # a fragment has no head to put it in


def test_levels_chart_is_named_in_plain_words() -> None:
    html = report.render(live_report())
    assert "How many requests ran at once, and for how long" in html
    assert "but the server never ran more than 10." in html


def test_report_without_series_or_snapshots() -> None:
    rep = report.build(boot_logs=[FX / "boot_pool_level_hi.log", FX / "boot_pool_level_lo.log"])
    html = report.render(rep)
    assert "KV pool per boot" in html  # the T4 chart appears with two boots
    assert "Requests over time" not in html


def test_mutation_changes_the_report() -> None:
    s = series.read(LIVE / "run.series.jsonl")
    flat = series.Series(
        s.header,
        tuple(
            series.Sample(
                x.t_wall,
                x.t_mono_ns,
                x.running,
                x.waiting,
                x.kv_usage,
                0.0,
                x.generation_tokens,
                x.prompt_tokens,
            )
            for x in s.samples
        ),
    )
    a = report.render(live_report())
    b = report.render(report.build(series=flat, requested=32, title="Live run"))
    assert 'class="ev"' in a and 'class="ev"' not in b


def test_cli_report_and_explain(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "r.html"
    code = main(
        [
            "report",
            "-o",
            str(out),
            "--boot-log",
            str(LIVE / "boot.log"),
            "--before",
            str(LIVE / "before.snapshot.json"),
            "--after",
            str(LIVE / "after.snapshot.json"),
            "--series",
            str(LIVE / "run.series.jsonl"),
            "--requested",
            "32",
        ]
    )
    assert code == 0 and out.stat().st_size > 50_000
    assert "8 checks (4 pass, 2 warn, 2 tripwire-failed)" in capsys.readouterr().out
    made_from = "boot.log, before.snapshot.json, after.snapshot.json, run.series.jsonl"
    assert made_from in out.read_text(encoding="utf-8")
    assert main(["explain", "t1"]) == 0
    assert "Silent preemption" in capsys.readouterr().out
    assert main(["explain", "T99"]) == 2


def test_cli_report_creates_the_output_folder(tmp_path: Path) -> None:
    out = tmp_path / "runs" / "today" / "r.html"
    assert main(["report", "-o", str(out), "--boot-log", str(LIVE / "boot.log")]) == 0
    assert out.stat().st_size > 10_000


def test_an_untested_vllm_version_is_flagged(tmp_path: Path) -> None:
    log = (FX / "boot_cudagraphs.log").read_text(encoding="utf-8")
    assert "(v0.28" in log
    newer = tmp_path / "boot.log"
    newer.write_text(log.replace("(v0.28", "(v0.99"), encoding="utf-8")
    assert 'class="version-note"' not in report.render(
        report.build(boot_logs=[FX / "boot_cudagraphs.log"])
    )
    html = report.render(report.build(boot_logs=[newer]))
    assert 'class="version-note"' in html and "vLLM 0.99" in html and "not a tested version" in html
