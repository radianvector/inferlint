"""One self-contained HTML page for a run: verdicts, headline numbers, charts, boot facts.

The page has no external resources (no fonts, scripts or images fetched), so it opens
offline on the machine that ran the benchmark and can be attached to a ticket as it is.
Every chart is drawn from the run's own files and has a data table beside it.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from html import escape
from importlib import resources
from pathlib import Path
from urllib.parse import quote

from . import __version__, bootlog, checks, insights
from .benchresult import BenchResult
from .blocks import predict_concurrency
from .catalog import ENGINE_NOTES, TRIPWIRES
from .engines import VLLM as VLLM_ENGINE
from .engines import Engine, by_key, named_by
from .result import CheckResult, Status
from .series import Series
from .svgchart import Bar, BarChart, Event, Line, RefLine, TimeChart, render_bars, render_time
from .telemetry import ServerRestarted, Snapshot, counter_delta, span_s

__all__ = ["Report", "Summary", "build", "render"]

_DASH = chr(0x2013)  # en dash, shown for a value that is absent

# The mark: a pulse that trips, on a bright tile. Used as the favicon and in the header.
_LOGO = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<defs><linearGradient id="inferlint-g" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#ffe14d"/><stop offset="1" stop-color="#ff5c1f"/>'
    "</linearGradient></defs>"
    '<rect width="32" height="32" rx="8" fill="url(#inferlint-g)"/>'
    '<path d="M4 18h7l3-9 4 15 3-6h7" fill="none" stroke="#14110f" stroke-width="3" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)
_FAVICON = "data:image/svg+xml," + quote(_LOGO, safe=" =:/,")

# Early, so a stored theme applies before the first paint.
_THEME_BOOT = (
    "<script>try{var t=localStorage.getItem('inferlint-theme');"
    "if(t==='light'||t==='dark')document.documentElement.setAttribute('data-theme',t)}"
    "catch(e){}</script>"
)

# vLLM's finished_reason label, in words.
_REASONS = {
    "length": "reached their token limit",
    "stop": "ended by the model",
    "repetition": "stopped for repeating",
    "abort": "cancelled",
    "error": "failed with an error",
}

# Why a tripwire has no result in a report: what it needs that the report was not given.
# Plural verbs; _checked() makes them singular for a single tripwire.
_NEEDS = {
    "T1": "need before and after readings",
    "T2": "run at clean-up",
    "T3": "run at clean-up",
    "T4": "need two or more boot logs",
    "T5": "need a boot log",
    "T6": "need a boot log",
    "T7": "need a live server",
    "T8": "need a recording and the concurrency asked for",
    "T9": "need a recording",
    "T10": "need before and after readings",
    "T11": "need two or more boot logs",
    "T12": "need a boot log",
    "T13": "a rule for test scripts",
    "T14": "need a recording and an after reading",
    "T15": "need the load tool's result file",
}

_CRASH_TRIPWIRES = {"T7", "T12"}  # the only ones whose failure means the server itself failed

_MEANING = {
    Status.FAIL: (
        "The tripwire caught its problem in this run. For T7 and T12 that means the "
        "server failed. For every other tripwire the server worked, but a number from the "
        "run (concurrency, throughput or latency) does not mean what it seems.",
        "Read the finding. Fix the setup and run again, or report the number together "
        "with the finding.",
    ),
    Status.WARN: (
        "Nothing is wrong on its own, but a setting or an effect changes how the numbers "
        "should be read.",
        "Keep the warning with the results, so whoever compares them knows.",
    ),
    Status.UNKNOWN: (
        "The check could not decide, because a counter or log line it needs was missing, "
        "for example from a vLLM version it does not know. It never counts as a pass.",
        "Supply the missing file, or check the vLLM version.",
    ),
    Status.PASS: (
        "The tripwire looked for its problem in this run and did not find it.",
        "Nothing. A pass covers this run only.",
    ),
    Status.NOT_APPLICABLE: (
        "The tripwire is about a behaviour this engine does not have, such as vLLM's "
        "reserved KV block (T14) on SGLang or TensorRT-LLM.",
        "Nothing.",
    ),
}

_TERMS = (
    (
        "Token",
        "The unit of text a model reads and writes, defined by its tokenizer. How much "
        "text one token covers depends on the model, the tokenizer and the language. "
        "Throughput, KV cache and cost are all counted in tokens.",
    ),
    (
        "Prompt and output tokens",
        "Prompt tokens are the input sent to the model. Output tokens are the tokens it "
        "generates in reply.",
    ),
    ("Request", "One completion request sent to the server, and its response."),
    (
        "Concurrency",
        "The number of requests in progress at the same moment. 'Asked for' is how many "
        "the load tool kept open. 'Running' is how many the scheduler was actually "
        "processing. The rest wait in its queue.",
    ),
    (
        "Waiting (queue)",
        "Requests the server has accepted but not scheduled, because the KV cache has no "
        "room for them yet.",
    ),
    (
        "KV cache",
        "GPU memory holding the attention keys and values for every token of every "
        "running request, so they are not recomputed for each new token. A request's KV "
        "grows with its length.",
    ),
    (
        "KV pool",
        "The size of the KV cache in tokens, set when vLLM starts from the GPU memory "
        "left after the model and CUDA graphs are loaded. It can differ from one start "
        "to the next (T4, T11).",
    ),
    (
        "KV block",
        "The allocation unit of the KV cache: a request always holds whole blocks, so "
        "even a short request holds a full one. The usual size is 16 tokens; models that "
        "mix attention and Mamba layers use blocks of hundreds of tokens (T5).",
    ),
    (
        "Null block",
        "One block vLLM reserves and never gives to a request, so a pool of N blocks "
        "holds requests in N - 1 of them (T14).",
    ),
    (
        "Fit in cache (ceiling)",
        "How many requests of their current size fit when the cache is full: "
        "floor(1 / share), where share is the fraction of the cache one request holds. "
        "As sequences grow, fewer fit (T9).",
    ),
    (
        "Preemption",
        "When a running request needs a block and none is free, vLLM evicts a running "
        "request: it frees that request's KV blocks and later recomputes them from the "
        "prompt and the output so far. vLLM counts it in vllm:num_preemptions_total and "
        "writes no log line (T1).",
    ),
    (
        "Output tokens per second",
        "Tokens generated per second by all requests together, from the server's own counter.",
    ),
    (
        "Time to first token",
        "Time from sending a request until the first token of the answer comes back. "
        "It includes the time spent waiting in the queue.",
    ),
    (
        "Boot log",
        "The messages vLLM prints while it starts. They record the settings it actually "
        "chose, which can differ from the ones asked for.",
    ),
    (
        "Before and after readings",
        "Copies of the server's counters taken just before and just after the test. "
        "The difference between them is what happened during the test.",
    ),
    (
        "Recording",
        "The server's gauges, read several times a second during the test. The charts "
        "are drawn from it.",
    ),
    (
        "Attention backend",
        "The code that does the model's main calculation on the GPU. vLLM has several "
        "and picks one unless it is told which (T6).",
    ),
    (
        "CUDA graphs",
        "A speed-up in which vLLM records the GPU's steps once at start-up and replays "
        "them. It costs a little memory.",
    ),
)

_ORDER = {
    Status.FAIL: 0,
    Status.WARN: 1,
    Status.UNKNOWN: 2,
    Status.PASS: 3,
    Status.NOT_APPLICABLE: 4,
}
_LABEL = {
    Status.FAIL: "Tripwire failed",
    Status.WARN: "Warning",
    Status.UNKNOWN: "Can't tell",
    Status.PASS: "Pass",
    Status.NOT_APPLICABLE: "Does not apply",
}
_QUIET = (Status.PASS, Status.NOT_APPLICABLE)  # folded under one line in the report
# Icon glyphs (not emoji) so status never rests on colour alone.
_ICON = {
    Status.PASS: '<path d="M3.5 8.5l3 3 6-7"/>',
    Status.WARN: '<path d="M8 3.5v6M8 12.2v.3"/>',
    Status.FAIL: '<path d="M4.5 4.5l7 7M11.5 4.5l-7 7"/>',
    Status.UNKNOWN: '<path d="M6 6a2 2 0 1 1 2.6 1.9c-.4.2-.6.5-.6 1v.6M8 12.2v.3"/>',
    Status.NOT_APPLICABLE: '<path d="M4.5 8h7"/>',
}


@dataclass
class Summary:
    """What the run's files say happened. Every field is None when its source is absent."""

    span_s: float | None = None
    started: float | None = None  # Unix time of the before reading
    ended: float | None = None
    boot_s: float | None = None  # first boot log: first stamped line to ready
    finished: dict[str, int] = field(default_factory=dict[str, int])  # by finished_reason
    finished_total: int | None = None  # when the server does not say why each finished
    output_tokens: int | None = None
    prompt_tokens: int | None = None
    mean_latency_s: float | None = None
    mean_first_token_s: float | None = None
    mean_queue_s: float | None = None
    preemptions: int | None = None
    peak_running: int | None = None
    interval_s: float | None = None
    readings: int | None = None
    url: str | None = None
    restarted: bool = False

    @property
    def done(self) -> int | None:
        return sum(self.finished.values()) if self.finished else self.finished_total


@dataclass
class Report:
    title: str
    results: list[CheckResult]
    boots: list[tuple[str, bootlog.BootFacts]] = field(
        default_factory=list[tuple[str, bootlog.BootFacts]]
    )
    series: Series | None = None
    before: Snapshot | None = None
    after: Snapshot | None = None
    requested: int | None = None
    usable_blocks: int | None = None
    generated: str = ""
    summary: Summary = field(default_factory=Summary)
    sources: list[str] = field(default_factory=list[str])
    engine: Engine = VLLM_ENGINE
    bench: BenchResult | None = None  # the load tool's result file, when there is one

    def result(self, tripwire: str) -> CheckResult | None:
        return next((r for r in self.results if r.tripwire == tripwire), None)


def build(
    *,
    boot_logs: Sequence[str | Path] = (),
    before: Snapshot | None = None,
    after: Snapshot | None = None,
    series: Series | None = None,
    requested: int | None = None,
    title: str | None = None,
    sources: Sequence[str] = (),
    extra: Sequence[CheckResult] = (),
    server_version: str | None = None,
    engine: str | None = None,
    bench: BenchResult | None = None,
) -> Report:
    """Run every tripwire the given files allow and collect the results.

    ``sources`` names the input files, for the report's "Made from" line. ``extra`` adds
    results the files cannot give, from a live run: the probe (T7), the teardown (T2) and
    the client/server comparison (T15). ``server_version`` is the version the live
    server reported, for engines whose boot log does not print it. ``engine`` is the
    engine the user named, for files that do not say which engine wrote them. ``bench``
    is the load tool's result file; its latencies feed "What limited this run".
    """
    results: list[CheckResult] = []
    boots: list[tuple[str, bootlog.BootFacts]] = []
    boot_s: float | None = None
    many = len(boot_logs) > 1
    for p, name in zip(boot_logs, bootlog.labels(boot_logs), strict=True):
        text = Path(p).read_text(encoding="utf-8", errors="replace")
        facts = bootlog.parse(text, engine)
        if not boots:
            boot_s = _boot_seconds(text, facts.ready_lineno)
            if facts.vllm_version is None:
                facts.server_version = server_version
        boots.append((name, facts))
        for r in (
            checks.check_boot_failure(text),
            checks.check_block_size(facts),
            checks.check_backend_honoured(facts),
        ):
            results.append(_prefixed(r, name) if many else r)
    if many:
        names = [n for n, _ in boots]
        facts_list = [f for _, f in boots]
        results.extend(checks.check_same_pool(facts_list, names))
        results.extend(checks.check_kv_memory_stable(facts_list, names))
    # The last result per tripwire: the teardown's T2, not the start-of-run gate's.
    latest = {r.tripwire: r for r in extra}
    have = {r.tripwire for r in results}
    results.extend(r for t, r in latest.items() if t not in have)
    results.extend(
        run_checks(
            before=before,
            after=after,
            series=series,
            requested=requested,
            facts=boots[0][1] if boots else None,
        )
    )
    t14 = next((r for r in results if r.tripwire == "T14"), None)
    inferred = t14.evidence.get("inferred_usable_blocks") if t14 else None
    usable = inferred if t14 and t14.status is Status.PASS and isinstance(inferred, int) else None
    now = _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z").strip()
    found = _engine(before, after, series, boots, engine)
    summary = _summarize(before, after, series, found)
    summary.boot_s = boot_s
    return Report(
        engine=found,
        title=title or _default_title(boots, found),
        results=results,
        boots=boots,
        series=series,
        before=before,
        after=after,
        requested=requested,
        usable_blocks=usable,
        generated=now,
        summary=summary,
        sources=list(sources),
        bench=bench,
    )


def _engine(
    before: Snapshot | None,
    after: Snapshot | None,
    series: Series | None,
    boots: Sequence[tuple[str, bootlog.BootFacts]],
    named: str | None = None,
) -> Engine:
    """Which engine the run's files come from: the snapshots, the recording, the boot log;
    if none says, the engine the user named, else vLLM."""
    names: set[str] = set()
    for s in (before, after):
        if s is not None:
            names |= s.metrics.names()
    found = named_by(names)
    if found is not None:
        return found
    if series is not None and isinstance(series.header.get("engine"), str):
        return series.engine
    if boots and boots[0][1].engine:
        return by_key(boots[0][1].engine)
    return by_key(named)


def run_checks(
    *,
    before: Snapshot | None,
    after: Snapshot | None,
    series: Series | None,
    requested: int | None,
    facts: bootlog.BootFacts | None = None,
) -> list[CheckResult]:
    """The tripwires a run's readings and recording allow: T1, T10, T14, T9, T8.

    T14 needs a block count from the server; engines that export none skip it.
    """
    results: list[CheckResult] = []
    if before is not None and after is not None:
        results.append(checks.check_no_preemption(before, after, series, facts))
        results.append(checks.precise_rate(before, after))
    if series is None:
        return results
    usable: int | None = None
    na = checks.does_not_apply("T14", series.engine)
    if na is not None:
        results.append(na)
    elif after is not None and checks.engine_of(after).metrics.cache_info is not None:
        t14 = checks.check_null_block(series, after)
        results.append(t14)
        inferred = t14.evidence.get("inferred_usable_blocks")
        if t14.status is Status.PASS and isinstance(inferred, int):
            usable = inferred
    results.append(checks.concurrency_ceiling(series, usable))
    if requested:
        results.append(checks.check_concurrency_reached(series, requested, facts))
    return results


def _summarize(
    before: Snapshot | None,
    after: Snapshot | None,
    s: Series | None,
    engine: Engine = VLLM_ENGINE,
) -> Summary:
    names = engine.metrics
    sm = Summary()
    if s is not None and s.samples:
        peak = s.peak_running()
        sm.peak_running = None if peak is None else int(peak)
        interval = s.header.get("interval_s")
        sm.interval_s = float(interval) if isinstance(interval, (int, float)) else None
        sm.readings = len(s.samples)
        url = s.header.get("url")
        sm.url = url if isinstance(url, str) else None
    if before is None or after is None:
        return sm
    sm.url = sm.url or after.url
    sm.started, sm.ended = before.t_wall, after.t_wall
    if sm.started != sm.started:  # NaN: bare Prometheus text carries no clock
        sm.started = sm.ended = None
    with contextlib.suppress(ValueError):  # no clock, or the clocks disagree: no span
        sm.span_s = span_s(before, after)[0]
    try:
        label = names.finished_reason_label
        if label is not None:
            for reason in _REASONS:
                n = counter_delta(before, after, names.request_success, **{label: reason})
                if n is not None:
                    sm.finished[reason] = int(n)
        else:
            n = counter_delta(before, after, names.request_success)
            sm.finished_total = None if n is None else int(n)
        out = counter_delta(before, after, names.generation_tokens)
        sm.output_tokens = None if out is None else int(out)
        prompt = counter_delta(before, after, names.prompt_tokens)
        sm.prompt_tokens = None if prompt is None else int(prompt)
        if names.preemptions is not None:
            pre = counter_delta(
                before, after, names.preemptions, witness=engine.witness(names.preemptions)
            )
            sm.preemptions = None if pre is None else int(pre)
        sm.mean_latency_s = _mean(before, after, names.e2e_latency)
        sm.mean_first_token_s = _mean(before, after, names.first_token)
        sm.mean_queue_s = _mean(before, after, names.queue_time)
    except ServerRestarted:
        sm = Summary(
            peak_running=sm.peak_running,
            interval_s=sm.interval_s,
            readings=sm.readings,
            url=sm.url,
            restarted=True,
        )
    return sm


def _mean(before: Snapshot, after: Snapshot, histogram: str) -> float | None:
    total = counter_delta(before, after, histogram + "_sum")
    count = counter_delta(before, after, histogram + "_count")
    if total is None or not count:
        return None
    return total / count


_STAMP = re.compile(r"\b(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\b")


def _boot_seconds(text: str, ready_lineno: int | None) -> float | None:
    """Seconds from the first stamped line to the last stamped line before 'ready'.

    vLLM stamps lines with month-day and time but no year; a negative span (a boot
    across New Year) is dropped rather than guessed.
    """
    if ready_lineno is None:
        return None
    first: _dt.datetime | None = None
    last: _dt.datetime | None = None
    for line in text.splitlines()[:ready_lineno]:
        m = _STAMP.search(line)
        if m is None:
            continue
        mo, d, h, mi, se = (int(g) for g in m.groups())
        try:
            t = _dt.datetime(2000, mo, d, h, mi, se)  # 2000: a leap year, so 02-29 parses
        except ValueError:
            continue
        first = first or t
        last = t
    if first is None or last is None or last < first:
        return None
    return (last - first).total_seconds()


def _prefixed(r: CheckResult, name: str) -> CheckResult:
    return CheckResult(r.tripwire, r.status, f"{name}: {r.message}", r.evidence)


def _default_title(
    boots: Sequence[tuple[str, bootlog.BootFacts]], engine: Engine = VLLM_ENGINE
) -> str:
    if boots:
        v = boots[0][1].version
        return f"{engine.name} {v} run" if v else f"{engine.name} run"
    return "Tripwire report"


# --------------------------------------------------------------------------- data


def _times(s: Series) -> list[float]:
    if all(x.t_mono_ns is not None for x in s.samples):
        t0 = s.samples[0].t_mono_ns or 0
        return [((x.t_mono_ns or 0) - t0) / 1e9 for x in s.samples]
    t0 = s.samples[0].t_wall
    return [x.t_wall - t0 for x in s.samples]


def _events(s: Series, xs: Sequence[float]) -> list[Event]:
    out: list[Event] = []
    prev: float | None = None
    for x, smp in zip(xs, s.samples, strict=True):
        p = smp.preemptions
        if p is not None and prev is not None and p > prev:
            n = int(p - prev)
            out.append(Event(x, f"{n} preemption{'s' if n > 1 else ''} at {x:.1f} s"))
        if p is not None:
            prev = p
    return out


def _ceiling(s: Series, usable: int | None) -> list[float | None]:
    out: list[float | None] = []
    for smp in s.samples:
        if smp.running and smp.kv_usage and smp.kv_usage >= 0.9:
            out.append(float(predict_concurrency(smp.kv_usage, smp.running, usable_blocks=usable)))
        else:
            out.append(None)
    return out


def _throughput(s: Series, xs: Sequence[float], window_s: float = 1.0) -> list[float | None]:
    g = [smp.generation_tokens for smp in s.samples]
    out: list[float | None] = []
    j = 0
    for i in range(len(xs)):
        while j < i and xs[i] - xs[j] > window_s:
            j += 1
        gi, gj = g[i], g[j]
        if j == i or gi is None or gj is None or xs[i] == xs[j] or gi < gj:
            out.append(None)
        else:
            out.append((gi - gj) / (xs[i] - xs[j]))
    return out


def _time_at_level(s: Series, xs: Sequence[float]) -> dict[int, float]:
    spent: dict[int, float] = {}
    for i in range(len(xs) - 1):
        r = s.samples[i].running
        if r:
            spent[int(r)] = spent.get(int(r), 0.0) + (xs[i + 1] - xs[i])
    return spent


# --------------------------------------------------------------------------- html


def _pill(status: Status) -> str:
    return (
        f'<span class="pill {status.value}"><svg viewBox="0 0 16 16" aria-hidden="true">'
        f"{_ICON[status]}</svg>{_LABEL[status]}</span>"
    )


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    th = "".join(f'<th scope="col">{escape(h)}</th>' for h in head)
    body = "".join("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in row) + "</tr>" for row in rows)
    return (
        f'<div class="tablewrap"><table><thead><tr>{th}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _figure(fid: str, title: str, caption: str, svg: str, legend: str, table: str) -> str:
    return (
        f'<figure class="chart" id="{fid}" data-chart>'
        f"<figcaption><h3>{escape(title)}</h3><p>{caption}</p></figcaption>"
        f"{legend}"
        f'<div class="plot">{svg}</div>'
        f"<details><summary>Data table</summary>{table}</details>"
        f"</figure>"
    )


def _legend(items: Sequence[tuple[str, str, str]]) -> str:
    """(kind, css class, label): kind is 'line', 'ref' or 'event'."""
    keys: list[str] = []
    for kind, cls, label in items:
        if kind == "event":
            key = (
                '<svg viewBox="0 0 12 10" aria-hidden="true">'
                '<path class="ev" d="M1 1H11L6 9Z"/></svg>'
            )
        elif kind == "ref":
            key = (
                '<svg viewBox="0 0 18 10" aria-hidden="true">'
                '<line class="ref" x1="0" x2="18" y1="5" y2="5"/></svg>'
            )
        else:
            key = (
                f'<svg viewBox="0 0 18 10" aria-hidden="true"><line class="mark {cls}" '
                f'x1="1" x2="17" y1="5" y2="5"/></svg>'
            )
        keys.append(f"<li>{key}{escape(label)}</li>")
    return f'<ul class="legend">{"".join(keys)}</ul>'


def _fmt_n(v: float | None, digits: int = 0) -> str:
    if v is None:
        return _DASH
    return f"{v:,.{digits}f}"


def _tiles(rep: Report) -> str:
    tiles: list[str] = []
    s = rep.series
    t8 = rep.result("T8")
    if s is not None:
        peak = s.peak_running()
        of = f" of {rep.requested}" if rep.requested else ""
        sub = (
            f"The load tool kept {rep.requested} requests open."
            if rep.requested
            else "Most requests the server ran at once."
        )
        state = t8.status if t8 else Status.PASS
        tiles.append(_tile("Ran at once", f"{_fmt_n(peak)}{of}", sub, state))
    t1 = rep.result("T1")
    if t1 is not None and "preemptions" in t1.evidence:
        n = float(t1.evidence["preemptions"])
        eng = rep.engine
        logged = (
            f"{eng.name} logs each." if eng.logs_preemptions else "The server log mentions none."
        )
        sub = f"Requests evicted and recomputed. {logged}"
        label = "Retractions" if eng.preemption == "retraction" else "Preemptions"
        tiles.append(_tile(label, _fmt_n(n), sub, t1.status))
    t10 = rep.result("T10")
    if t10 is not None and "rate_per_s" in t10.evidence:
        rate = float(t10.evidence["rate_per_s"])
        err = t10.evidence.get("integer_second_error")
        sub = f"over {float(t10.evidence['span_s']):.1f} s"
        if isinstance(err, float):
            sub += f"; a whole-second timer reads {err:+.2%}"
        b = rep.bench
        if b is not None and b.output_throughput and b.duration_s:
            sub = (
                f"server's count over {float(t10.evidence['span_s']):.1f} s, idle time "
                f"included; the load tool measured {b.output_throughput:,.1f} over its "
                f"{b.duration_s:.1f} s"
            )
        tiles.append(_tile("Output tokens per second", _fmt_n(rate, 1), sub, None))
    b = rep.bench
    if b is not None and b.median_ttft_ms is not None and b.p99_ttft_ms is not None:
        tiles.append(
            _tile(
                "Time to first token, median",
                _ms(b.median_ttft_ms),
                f"p99 {_ms(b.p99_ttft_ms)}; as the load tool measured it, queueing included",
                None,
            )
        )
    t14 = rep.result("T14")
    if t14 is not None and t14.evidence.get("inferred_usable_blocks"):
        u = t14.evidence["inferred_usable_blocks"]
        rep_n = t14.evidence["reported_num_gpu_blocks"]
        bs = rep.boots[0][1].attention_block_size if rep.boots else None
        sub = f"{rep_n} reported, one held back" + (f"; {bs:,} tokens per block" if bs else "")
        tiles.append(_tile("Usable KV blocks", f"{u}", sub, None))
    return (
        f'<section class="tiles" aria-label="Headline numbers">{"".join(tiles)}</section>'
        if tiles
        else ""
    )


def _tile(label: str, value: str, sub: str, status: Status | None) -> str:
    badge = _pill(status) if status is not None and status is not Status.PASS else ""
    return (
        f'<div class="tile"><div class="tile-head"><span class="tile-label">{escape(label)}</span>'
        f'{badge}</div><div class="tile-value">{escape(value)}</div>'
        f'<div class="tile-sub">{escape(sub)}</div></div>'
    )


def _finding(r: CheckResult) -> str:
    tw = TRIPWIRES.get(r.tripwire)
    name = tw.name if tw else r.tripwire
    what = f'<p class="what">{escape(tw.why)}</p>' if tw and r.status not in _QUIET else ""
    return (
        f'<li class="finding {r.status.value}">'
        f'<div class="f-status">{_pill(r.status)}</div>'
        f'<div class="f-body"><div class="f-title"><span class="tw">{escape(r.tripwire)}</span>'
        f"{escape(name)}</div>"
        f'<p class="msg">{escape(r.message)}</p>{what}</div>'
        f"</li>"
    )


def _findings(rep: Report) -> str:
    """Failures, warnings and unknowns in full; passed checks, and those that do not
    apply to this engine, folded under one line."""
    ordered = sorted(rep.results, key=lambda r: (_ORDER[r.status], _tw_num(r.tripwire)))
    shown = [r for r in ordered if r.status not in _QUIET]
    passed = [r for r in ordered if r.status in _QUIET]
    out = f'<ol class="findings">{"".join(_finding(r) for r in shown)}</ol>' if shown else ""
    if passed:
        n_pass = sum(1 for r in passed if r.status is Status.PASS)
        n_na = len(passed) - n_pass
        label = f"{n_pass} passed" if shown else f"All {n_pass} checks passed"
        if n_na:
            label += f"; {n_na} {'does' if n_na == 1 else 'do'} not apply to {rep.engine.name}"
        out += (
            f'<details class="gloss passed"><summary>{label}</summary>'
            f'<ol class="findings">{"".join(_finding(r) for r in passed)}</ol></details>'
        )
    return out


def _ms(v: float) -> str:
    return f"{v / 1000:,.1f} s" if v >= 1000 else f"{v:,.0f} ms"


def _tw_num(tw: str) -> int:
    return int(tw[1:]) if tw[1:].isdigit() else 99


def _verdict(rep: Report) -> str:
    counts = {s: sum(1 for r in rep.results if r.status is s) for s in _ORDER}
    chips = "".join(
        f'<li class="count {s.value}"><span class="n">{counts[s]}</span>{_pill(s)}</li>'
        for s in sorted(_ORDER, key=lambda s: _ORDER[s])
        if counts[s] or s is not Status.NOT_APPLICABLE
    )
    return f'<ul class="verdict" aria-label="Check results">{chips}</ul>'


def _dur(s: float) -> str:
    if s < 60:
        return f"{s:.1f} s"
    m, r = divmod(round(s), 60)
    return f"{m} min {r} s"


def _clock(start: float, end: float) -> str:
    a = _dt.datetime.fromtimestamp(start).astimezone()
    b = _dt.datetime.fromtimestamp(end).astimezone()
    off = a.utcoffset() or _dt.timedelta(0)
    mins = int(off.total_seconds() // 60)
    sign = "+" if mins >= 0 else "-"
    h, m = divmod(abs(mins), 60)
    zone = f"UTC{sign}{h}" + (f":{m:02d}" if m else "")
    return f"{a:%Y-%m-%d %H:%M:%S} to {b:%H:%M:%S} ({zone})"


def _finished(sm: Summary) -> str:
    done = sm.done or 0
    if not sm.finished:  # the server counts finished requests without saying why
        return f"{done:,}"
    parts = [f"{n:,} {_REASONS[k]}" for k, n in sm.finished.items() if n and k != "error"]
    errors = sm.finished.get("error", 0)
    parts.append(f"{errors:,} {_REASONS['error']}" if errors else "no errors")
    return f"{done:,}: " + ", ".join(parts)


def _story(rep: Report) -> str:
    sm = rep.summary
    out: list[str] = []
    if sm.restarted:
        return (
            "The server restarted between the before and after readings, so the counts "
            "between them mean nothing. Only the recording is summarised here."
        )
    if sm.span_s is not None:
        out.append(
            f"The load tool kept {rep.requested} requests open for {_dur(sm.span_s)}."
            if rep.requested
            else f"The test ran for {_dur(sm.span_s)}."
        )
    if sm.done is not None and sm.output_tokens is not None:
        errors = sm.finished.get("error", 0)
        how = "with no errors" if not errors else f"({errors:,} of them with an error)"
        if not sm.finished:  # no finish reasons: say nothing about errors
            out.append(
                f"The server finished {sm.done:,} requests and generated "
                f"{sm.output_tokens:,} tokens."
            )
        else:
            out.append(
                f"The server finished {sm.done:,} requests {how} and generated "
                f"{sm.output_tokens:,} tokens."
            )
    if sm.peak_running is not None and rep.requested and sm.peak_running < rep.requested:
        wait = (
            f", {sm.mean_queue_s:.0f} s on average before a request started"
            if sm.mean_queue_s
            else ""
        )
        out.append(
            f"It never worked on more than {sm.peak_running} at once. "
            f"The rest waited in a queue{wait}."
        )
    if sm.preemptions:
        times = "once" if sm.preemptions == 1 else f"{sm.preemptions:,} times"
        verb = "retracted" if rep.engine.preemption == "retraction" else "preempted"
        logged = (
            f"{rep.engine.name} logs each one."
            if rep.engine.logs_preemptions
            else "The server log does not mention this."
        )
        out.append(
            f"It {verb} a running request {times} to free KV cache: each one went back "
            f"to the queue and later recomputed its prompt and output so far. {logged}"
        )
    return " ".join(out)


def _facts(title: str, rows: Sequence[tuple[str, str]]) -> str:
    if not rows:
        return ""
    dl = "".join(f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in rows)
    return f'<div class="facts-group"><h3>{escape(title)}</h3><dl>{dl}</dl></div>'


def _checked(rep: Report) -> tuple[str, str]:
    na = {r.tripwire for r in rep.results if r.status is Status.NOT_APPLICABLE}
    ran = ({r.tripwire for r in rep.results} & TRIPWIRES.keys()) - na
    missing: dict[str, list[str]] = {}
    for t in TRIPWIRES:
        if t not in ran:
            why = _NEEDS.get(t, "need other input")
            if t in na or (t == "T14" and rep.engine.metrics.cache_info is None):
                why = f"do not apply to {rep.engine.name}"
            missing.setdefault(why, []).append(t)
    groups: list[str] = []
    for why, ids in missing.items():
        if len(ids) == 1:
            verb, _, rest = why.partition(" ")
            if verb in ("need", "run"):
                why = f"{verb}s {rest}"
            elif verb == "do":
                why = f"does {rest}"
        names = ids[0] if len(ids) == 1 else ", ".join(ids[:-1]) + " and " + ids[-1]
        groups.append(f"{names} ({why})")
    text = f"{len(ran)} of {len(TRIPWIRES)}"
    return ("Tripwires checked", text + (f". Not here: {', '.join(groups)}" if groups else ""))


def _happened(rep: Report) -> str:
    sm = rep.summary
    test: list[tuple[str, str]] = []
    if sm.span_s is not None:
        exact = f" ({sm.span_s:.1f} s)" if sm.span_s >= 60 else ""
        test.append(
            ("Measured time", f"{_dur(sm.span_s)}{exact}, from the before to the after reading")
        )
    if sm.started is not None and sm.ended is not None:
        test.append(("Clock time", _clock(sm.started, sm.ended)))
    if sm.boot_s is not None:
        test.append(("Server start-up", f"{_dur(sm.boot_s)}, before the test (from the boot log)"))
    done = sm.done
    if done is not None:
        test.append(("Requests finished", _finished(sm)))
    if sm.output_tokens is not None:
        per = f" ({sm.output_tokens / done:,.0f} per request)" if done else ""
        test.append(("Output tokens generated", f"{sm.output_tokens:,}{per}"))
    if sm.prompt_tokens is not None:
        per = f" ({sm.prompt_tokens / done:,.0f} per request)" if done else ""
        test.append(("Prompt tokens read", f"{sm.prompt_tokens:,}{per}"))
    if sm.mean_latency_s is not None:
        test.append(("Average time per request", f"{sm.mean_latency_s:.1f} s"))
    if sm.mean_first_token_s is not None:
        q = (
            f", {sm.mean_queue_s:.1f} s of it waiting in the queue"
            if sm.mean_queue_s is not None
            else ""
        )
        test.append(("Average wait for the first token", f"{sm.mean_first_token_s:.1f} s{q}"))

    setup: list[tuple[str, str]] = []
    watched = rep.series is not None or (rep.before is not None and rep.after is not None)
    if rep.requested:
        setup.append(("Concurrency asked for", f"{rep.requested} requests at once"))
    if watched:
        setup.append(
            (
                "Requests sent by",
                "the load tool. inferlint sends none: it only reads the server's counters.",
            )
        )
    if sm.interval_s and sm.readings:
        setup.append(
            ("Recording", f"the server read every {sm.interval_s:g} s, {sm.readings:,} readings")
        )
    if sm.url:
        setup.append(("Server address", sm.url))
    if rep.sources:
        setup.append(("Made from", ", ".join(rep.sources)))
    setup.append(_checked(rep))

    story = _story(rep)
    lede = f'<p class="lede">{escape(story)}</p>' if story else ""
    return (
        '<section class="happened" aria-labelledby="h-happened">'
        '<h2 id="h-happened">What happened</h2>'
        f"{lede}{_tiles(rep)}"
        f'<div class="facts">{_facts("The test", test)}{_facts("The setup", setup)}</div>'
        f"{_terms(rep.engine)}</section>"
    )


# The same event in the other engines' words.
_PREEMPTION_TERM = {
    "sglang": (
        "When the KV cache cannot hold the running requests' next tokens, SGLang retracts "
        "a running request: it frees that request's KV and later recomputes it from the "
        "prompt and the output so far. SGLang counts it in "
        "sglang:num_retracted_requests_total and logs a warning (T1)."
    ),
    "trtllm": (
        "With the MAX_UTILIZATION scheduler policy, TensorRT-LLM pauses a running request "
        "when the KV cache is full and later recomputes it. It counts none, and logs each "
        "pause at INFO; its default policy, GUARANTEED_NO_EVICT, pauses none (T1)."
    ),
}


def _terms(engine: Engine = VLLM_ENGINE) -> str:
    terms = [
        (k, _PREEMPTION_TERM.get(engine.key, v) if k == "Preemption" else v) for k, v in _TERMS
    ]
    dl = "".join(f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in terms)
    return (
        '<details class="gloss" id="words"><summary>Words used in this report</summary>'
        f'<dl class="terms">{dl}</dl></details>'
    )


def _status_note(rep: Report) -> str:
    fails = [r for r in rep.results if r.status is Status.FAIL]
    if not fails:
        return ""
    crashed = sorted({r.tripwire for r in fails} & _CRASH_TRIPWIRES, key=_tw_num)
    if crashed:
        text = (
            f"{' and '.join(crashed)}: the server itself failed. Any other failed tripwire "
            "is about the measurement."
        )
    else:
        text = (
            "The server worked; the failed tripwires are about the measurement. Some "
            "numbers from this run cannot be taken at face value."
        )
        sm = rep.summary
        if sm.done and sm.finished and not sm.finished.get("error"):
            text += f" The server finished {sm.done:,} requests with no errors."
    return f'<p class="status-note">{escape(text)}</p>'


def _status_glossary() -> str:
    rows = "".join(
        f"<tr><td>{_pill(s)}</td><td>{escape(_MEANING[s][0])}</td>"
        f"<td>{escape(_MEANING[s][1])}</td></tr>"
        for s in sorted(_ORDER, key=lambda s: _ORDER[s])
    )
    return (
        '<details class="gloss"><summary>What Pass, Warning, Tripwire failed, Can\'t tell '
        "and Does not apply mean</summary>"
        '<div class="tablewrap"><table class="meanings"><thead><tr>'
        '<th scope="col">Result</th><th scope="col">What it means</th>'
        '<th scope="col">What to do</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div></details>"
    )


def _theme_toggle() -> str:
    """One button, as on radianvector.com: it shows the current theme and flips it."""
    return (
        '<button type="button" class="icon-btn" data-theme-toggle aria-label="Switch theme">'
        '<svg class="ico-moon" viewBox="0 0 24 24" aria-hidden="true">'
        '<path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z"/></svg>'
        '<svg class="ico-sun" viewBox="0 0 24 24" aria-hidden="true">'
        '<circle cx="12" cy="12" r="4.2"/><path d="M12 2.5v2.2M12 19.3v2.2M4.2 4.2l1.6 1.6'
        'M18.2 18.2l1.6 1.6M2.5 12h2.2M19.3 12h2.2M4.2 19.8l1.6-1.6M18.2 5.8l1.6-1.6"/></svg>'
        "</button>"
    )


def _charts(rep: Report) -> str:
    s = rep.series
    if s is None or len(s.samples) < 2:
        return ""
    xs = _times(s)
    evs = _events(s, xs)
    figs: list[str] = []
    lag = (
        f" {escape(rep.engine.name)} updates these readings only when a request completes, "
        "so between completions a line holds its last value."
        if rep.engine.gauges_lag
        else ""
    )

    running = [smp.running for smp in s.samples]
    waiting = [smp.waiting for smp in s.samples]
    ceiling = _ceiling(s, rep.usable_blocks)
    lines = [
        Line("waiting", waiting, "s2"),
        Line("running", running, "s1"),
        Line("fit in cache", ceiling, "s3", "step"),
    ]
    refs = [RefLine(float(rep.requested), f"asked for {rep.requested}")] if rep.requested else []
    svg = render_time(
        TimeChart(
            "requests",
            xs,
            lines,
            "requests",
            refs=refs,
            events=evs,
            event_label="preemption",
            aria="Requests running and waiting over time",
        )
    )
    leg = _legend(
        [
            ("line", "s1", "running"),
            ("line", "s2", "waiting"),
            ("line", "s3", "how many fit in the cache"),
        ]
        + ([("ref", "", "concurrency asked for")] if refs else [])
        + ([("event", "", "preemption")] if evs else [])
    )
    rows = [
        [f"{x:.2f}", _fmt_n(r), _fmt_n(w), _fmt_n(c)]
        for x, r, w, c in zip(xs, running, waiting, ceiling, strict=True)
    ]
    cap = (
        "Requests the server was working on, requests queued behind them, and how many "
        "requests of their current size fit in the cache when it is full. Triangles mark "
        "preemptions." + lag
    )
    figs.append(
        _figure(
            "fig-requests",
            "Requests over time",
            cap,
            svg,
            leg,
            _table(["seconds", "running", "waiting", "fit in cache"], rows),
        )
    )

    usage = [None if smp.kv_usage is None else smp.kv_usage * 100 for smp in s.samples]
    svg = render_time(
        TimeChart(
            "kv",
            xs,
            [Line("KV cache in use", usage, "s1", "area")],
            "%",
            y_max=100,
            y_fmt=lambda v: f"{v:.0f}%",
            events=evs,
            event_label="preemption",
            aria="KV cache usage over time",
        )
    )
    denom = (
        f" The gauge moves in steps of 1/{rep.usable_blocks}: one block is held back (T14)."
        if rep.usable_blocks
        else ""
    )
    leg = _legend([("event", "", "preemption")]) if evs else ""
    rows = [[f"{x:.2f}", _fmt_n(u, 1)] for x, u in zip(xs, usage, strict=True)]
    figs.append(
        _figure(
            "fig-kv",
            "KV cache in use",
            "Share of the cache holding requests." + escape(denom) + lag,
            svg,
            leg,
            _table(["seconds", "% in use"], rows),
        )
    )

    # An engine that counts a request's tokens when it finishes draws completions here,
    # not generation; the section says why the chart is missing (_tput_note).
    tput = [] if rep.engine.tokens_at_finish else _throughput(s, xs)
    if any(v is not None for v in tput):
        svg = render_time(
            TimeChart(
                "tput",
                xs,
                [Line("tokens/s", tput, "s1")],
                "tokens/s",
                aria="Output tokens per second over time",
            )
        )
        rows = [[f"{x:.2f}", _fmt_n(v, 1)] for x, v in zip(xs, tput, strict=True)]
        figs.append(
            _figure(
                "fig-tput",
                "Output tokens per second",
                "Generated tokens over a sliding one-second window, from the server's own "
                "counter." + lag,
                svg,
                "",
                _table(["seconds", "tokens/s"], rows),
            )
        )

    spent = _time_at_level(s, xs)
    if spent:
        levels = sorted(spent, reverse=True)
        bars = [
            Bar(f"{lv} at once", spent[lv], f"{lv} running at once for {spent[lv]:.1f} s")
            for lv in levels
        ]
        svg = render_bars(
            BarChart(
                "levels", bars, "seconds", aria="Seconds spent at each number of running requests"
            )
        )
        asked = ""
        if rep.requested:
            top = levels[0]
            asked = f" The load tool asked for {rep.requested} the whole time" + (
                f", but the server never ran more than {top}." if top < rep.requested else "."
            )
        rows = [[str(lv), f"{spent[lv]:.2f}"] for lv in levels]
        figs.append(
            _figure(
                "fig-levels",
                "How many requests ran at once, and for how long",
                "Each bar is the total time the server spent working on exactly that many "
                "requests at the same time." + escape(asked),
                svg,
                "",
                _table(["running at once", "seconds"], rows),
            )
        )

    return "".join(figs)


def _tput_note(rep: Report) -> str:
    if not rep.engine.tokens_at_finish or rep.series is None or len(rep.series.samples) < 2:
        return ""
    name = escape(rep.engine.name)
    return (
        '<p class="note">'
        f"No output-tokens-per-second chart: {name} adds a request's tokens to "
        "its counter when the request finishes, so readings during the run show completions, "
        "not generation. The load tool's figure and T10 cover the whole run.</p>"
    )


def _limits(rep: Report) -> str:
    found = insights.limits(rep)
    if not found:
        return ""
    items = "".join(
        f'<li class="limit {i.tone}"><div class="l-title">{escape(i.title)}</div>'
        f"<p>{escape(i.text)}</p></li>"
        for i in found
    )
    return (
        '<section class="limits" aria-labelledby="h-limits">'
        '<h2 id="h-limits">What limited this run</h2>'
        f'<ul class="limit-list">{items}</ul></section>'
    )


def _cache_chart(rep: Report) -> str:
    need = insights.cache_need(rep)
    if need is None:
        return ""
    pool, per, n, total = need.pool, need.per_request, need.concurrency, need.total
    load = f"{n} requests at full length"
    bars = [
        Bar("KV cache", float(pool), f"the cache holds {pool:,} tokens"),
        Bar(load, total, f"{n} x {need.allocated:,.0f} tokens = {total:,.0f}"),
    ]
    svg = render_bars(
        BarChart(
            "cache",
            bars,
            "tokens",
            fmt=lambda v: f"{v:,.0f}",
            aria="Tokens the KV cache holds, and tokens the load needs",
            label_width=190,
        )
    )
    fit = (
        f" {int(pool // need.allocated)} of them fit."
        if total > pool
        else f" The cache has {pool - total:,.0f} tokens to spare."
    )
    rounded = (
        f", {need.allocated:,.0f} in whole {need.block}-token blocks"
        if need.block and need.allocated != per
        else ""
    )
    cap = (
        f"Each request holds its prompt and its output so far: about {per:,.0f} tokens at the "
        f"end{rounded}. All {n} at the end of their output need {total:,.0f}." + fit
    )
    rows = [["KV cache", f"{pool:,}"], [load, f"{total:,.0f}"]]
    return _figure(
        "fig-cache",
        "The KV cache and the load",
        escape(cap),
        svg,
        "",
        _table(["", "tokens"], rows),
    )


def _memory_chart(f: bootlog.BootFacts) -> str:
    parts = [
        ("Model weights", f.model_load_gib),
        ("KV cache", f.available_kv_cache_gib),
        ("CUDA graphs", f.cudagraph_actual_gib),
        ("Activation peak", f.peak_activation_gib),
    ]
    known = [(k, v) for k, v in parts if v]
    if len(known) < 2:
        return ""
    bars = [Bar(k, float(v), f"{k}: {v:g} GiB") for k, v in known]
    svg = render_bars(
        BarChart(
            "memory",
            bars,
            "GiB",
            fmt=lambda v: f"{v:,.2f} GiB",
            aria="GPU memory per part, from the boot log",
            label_width=150,
        )
    )
    return _figure(
        "fig-memory",
        "GPU memory at start-up",
        "What the boot log says each part took. The KV cache gets what the engine's memory "
        "setting leaves after the others, so a change in any of them changes it (T4, T11).",
        svg,
        "",
        _table(["part", "GiB"], [[k, f"{v:g}"] for k, v in known]),
    )


def _pools_chart(rep: Report) -> str:
    pooled = [(n, f.kv_pool_tokens) for n, f in rep.boots if f.kv_pool_tokens]
    if len(pooled) < 2:
        return ""
    bars = [
        Bar(n if len(n) <= 18 else n[:17] + "…", float(p), f"{n}: {p:,} tokens") for n, p in pooled
    ]
    svg = render_bars(
        BarChart(
            "pools",
            bars,
            "tokens",
            fmt=lambda v: f"{v:,.0f}",
            aria="KV pool drawn by each boot",
            label_width=150,
        )
    )
    rows = [[n, f"{p:,}"] for n, p in pooled]
    return _figure(
        "fig-pools",
        "KV pool per boot",
        "The pool each boot drew. Compare results only between boots with the same pool (T4).",
        svg,
        "",
        _table(["boot log", "pool tokens"], rows),
    )


# SGLang prints all ~500 of its settings; these are the ones that change what is measured.
_SGLANG_SETTINGS = (
    "attention_backend",
    "chunked_prefill_size",
    "context_length",
    "disable_cuda_graph",
    "disable_radix_cache",
    "kv_cache_dtype",
    "mamba_full_memory_ratio",
    "max_mamba_cache_size",
    "max_queued_requests",
    "max_running_requests",
    "max_total_tokens",
    "mem_fraction_static",
    "page_size",
    "schedule_policy",
    "speculative_algorithm",
)


def _settings(f: bootlog.BootFacts) -> dict[str, object]:
    """The start's settings worth showing: vLLM's non-default ones, SGLang's main ones."""
    if f.non_default_args is not None:
        return {k: v for k, v in f.non_default_args.items() if k not in ("model", "model_tag")}
    sa = f.server_args or {}
    return {k: sa[k] for k in _SGLANG_SETTINGS if k in sa}


def _model_name(f: bootlog.BootFacts) -> str:
    """The model's folder or repository name, from the start's arguments."""
    args = f.config_args or {}
    path = args.get("model", args.get("model_path", ""))
    return str(path or "").rstrip("/").split("/")[-1]


def _boot_table(rep: Report) -> str:
    if not rep.boots:
        return ""
    name, f = rep.boots[0]
    model = _model_name(f) or _DASH
    if f.page_size is not None:
        block = f"{f.page_size:,} token{'s' if f.page_size != 1 else ''} (page size)"
    elif f.attention_block_size:
        block = f"{f.attention_block_size:,} tokens"
    else:
        block = "16 tokens (default)"
    items = [
        (rep.engine.name, f.version or _DASH),
        ("Model", model),
        ("KV pool", f"{f.kv_pool_tokens:,} tokens" if f.kv_pool_tokens else _DASH),
        ("Attention block", block),
        ("Attention backend", ", ".join(f.selected_backends) or _DASH),
        ("Draft model backend", ", ".join(f.drafter_backends) or "none"),
        (
            "Memory for KV cache",
            f"{f.available_kv_cache_gib} GiB" if f.available_kv_cache_gib else _DASH,
        ),
        (
            "CUDA graphs",
            "off"
            if f.cudagraphs is False
            else (
                f"on, {f.cudagraph_actual_gib} GiB" if f.cudagraph_actual_gib is not None else "on"
            ),
        ),
        (
            "Boot line 'maximum concurrency'",
            f"{f.max_concurrency}x at {f.max_concurrency_request_len:,} tokens"
            if f.max_concurrency and f.max_concurrency_request_len
            else _DASH,
        ),
        (
            "Model load",
            f"{f.model_load_gib} GiB in {f.model_load_s:.0f} s"
            if f.model_load_gib and f.model_load_s
            else _DASH,
        ),
        (
            "Server settings",
            ", ".join(f"{k}={v}" for k, v in sorted(_settings(f).items())) or _DASH,
        ),
    ]
    dl = "".join(f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in items)
    return (
        f'<section class="boot"><h2>The server that ran</h2>'
        f'<p class="note">From <code>{escape(name)}</code>. Every result should carry these, '
        f"because they change from one start to the next.</p><dl>{dl}</dl>"
        f"{_memory_chart(f)}</section>"
    )


def _guide(engine: Engine = VLLM_ENGINE) -> str:
    def note(tid: str) -> str:
        text = ENGINE_NOTES.get(tid, {}).get(engine.key)
        return f'<p class="why">On {escape(engine.name)}: {escape(text)}</p>' if text else ""

    items = "".join(
        f'<li><div class="g-head"><span class="tw">{t.id}</span>{escape(t.name)}'
        f'<span class="slug">{escape(t.slug)}</span></div>'
        f'<p>{escape(t.what)}</p><p class="why">{escape(t.why)}</p>{note(t.id)}'
        f"<code>{escape(t.command)}</code></li>"
        for t in TRIPWIRES.values()
    )
    # Reference text, the same in every report: folded, one line until opened.
    return (
        f'<section class="guide" id="guide"><h2>The {len(TRIPWIRES)} tripwires</h2>'
        '<details class="gloss"><summary>What each one catches, and how to check it</summary>'
        '<div class="g-body"><p class="note">Each one is a way an inference benchmark can '
        "report a wrong number without raising an error.</p>"
        f'<ol class="g-list">{items}</ol></div></details></section>'
    )


def _asset(name: str) -> str:
    return (
        resources.files("inferlint").joinpath("assets").joinpath(name).read_text(encoding="utf-8")
    )


def render(rep: Report, *, standalone: bool = True) -> str:
    """The report as HTML. ``standalone=False`` omits the document skeleton."""
    boot = rep.boots[0][1] if rep.boots else None
    meta: list[str] = []
    if boot is not None:
        model = _model_name(boot)
        engine = f"{rep.engine.name} {boot.version}" if boot.version else rep.engine.name
        meta += [x for x in (engine, model) if x]
        if boot.kv_pool_tokens:
            meta.append(f"{boot.kv_pool_tokens:,}-token KV pool")
    if rep.requested:
        meta.append(f"concurrency asked for: {rep.requested}")
    meta_html = " · ".join(escape(m) for m in meta)
    note = bootlog.untested_version(boot) if boot is not None else None
    note_html = f'<p class="version-note">{escape(note)}</p>' if note else ""
    charts = _cache_chart(rep) + _charts(rep) + _pools_chart(rep)
    logo = _LOGO.replace("<svg ", '<svg class="logo" aria-hidden="true" ', 1)
    body = (
        '<div class="page">'
        '<header class="top"><div class="bar">'
        f'<div class="brand">{logo}inferlint</div>{_theme_toggle()}</div>'
        f"<h1>{escape(rep.title)}</h1>"
        f'<p class="meta">{meta_html}</p>{note_html}</header>'
        f"{_limits(rep)}{_happened(rep)}"
        '<section class="results" aria-labelledby="h-results">'
        f'<h2 id="h-results">Results</h2>{_verdict(rep)}{_status_note(rep)}'
        f"{_status_glossary()}{_findings(rep)}</section>"
        + (
            f'<section class="charts"><h2>What the server did</h2>{_tput_note(rep)}{charts}'
            "</section>"
            if charts
            else ""
        )
        + _boot_table(rep)
        + _guide(rep.engine)
        + f"<footer>Generated {escape(rep.generated)} by inferlint {escape(__version__)}. "
        "Every chart is drawn from the run's own files.</footer>"
        "</div>"
        '<div class="tt" role="status" hidden></div>'
        f"<script>{_asset('report.js')}</script>"
    )
    head = f"<title>{escape(rep.title)}</title>{_THEME_BOOT}<style>{_asset('report.css')}</style>"
    if not standalone:
        return head + body
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
        f'<link rel="icon" href="{_FAVICON}">'
        f"{head}</head><body>{body}</body></html>"
    )
