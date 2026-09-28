"""``inferlint``: the tripwires from the command line.

Exit status: 0 all checks passed (warnings allowed), 1 a check failed,
2 a check could not decide (a series or log line was missing). ``--strict`` turns
warnings into failures.
"""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import sys
import textwrap
import threading
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import __version__, bootlog, checks, report, series, teardown, telemetry
from .catalog import TRIPWIRES
from .probe import probe
from .result import CheckResult, Status

_MARK = {Status.PASS: "PASS", Status.WARN: "WARN", Status.FAIL: "FAIL", Status.UNKNOWN: "????"}


def _emit(results: Sequence[CheckResult], as_json: bool, strict: bool) -> int:
    if as_json:
        print(json.dumps([r.to_json() for r in results], indent=1, default=str))
    else:
        for r in results:
            print(f"[{r.tripwire:>3}] {_MARK[r.status]}  {r.message}")
    statuses = {r.status for r in results}
    if Status.FAIL in statuses or (strict and Status.WARN in statuses):
        return 1
    if Status.UNKNOWN in statuses:
        return 2
    return 0


def _out(path: str) -> Path:
    """An output path with its folder created, so ``-o runs/today/report.html`` works."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _cmd_boot_facts(a: argparse.Namespace) -> int:
    facts = bootlog.parse_file(a.log)
    doc = asdict(facts)
    if a.json:
        print(json.dumps(doc, indent=1, default=str))
    else:
        for k, v in doc.items():
            if k in ("unparsed", "conflicts", "non_default_args") or v in (None, [], {}, ()):
                continue
            print(f"{k:28} {v}")
        if facts.non_default_args:
            print(f"{'non_default_args':28} {sorted(facts.non_default_args)}")
        for c, vals in facts.conflicts.items():
            print(f"CONFLICT {c}: {vals}")
        for u in facts.unparsed:
            print(f"UNPARSED {u.field} (line {u.lineno}): {u.line}")
    return 2 if (a.strict and facts.unparsed) else 0


def _cmd_check_log(a: argparse.Namespace) -> int:
    results: list[CheckResult] = []
    all_facts: list[bootlog.BootFacts] = []
    for p in a.logs:
        text = Path(p).read_text(encoding="utf-8", errors="replace")
        facts = bootlog.parse(text)
        all_facts.append(facts)
        per = [
            checks.check_boot_failure(text),
            checks.check_block_size(facts),
            checks.check_backend_honoured(facts),
        ]
        if facts.unparsed:
            per.append(
                CheckResult(
                    "T12",
                    Status.UNKNOWN,
                    f"{len(facts.unparsed)} recognised line(s) in an unknown format",
                    {"unparsed": [asdict(u) for u in facts.unparsed]},
                )
            )
        if len(a.logs) > 1:
            for r in per:
                results.append(
                    CheckResult(r.tripwire, r.status, f"{Path(p).name}: {r.message}", r.evidence)
                )
        else:
            results.extend(per)
    if len(a.logs) > 1:
        names = [Path(p).name for p in a.logs]
        results.extend(checks.check_same_pool(all_facts, names))
        results.extend(checks.check_kv_memory_stable(all_facts, names))
    return _emit(results, a.json, a.strict)


def _cmd_snapshot(a: argparse.Namespace) -> int:
    snap = telemetry.scrape(a.url)
    snap.save(_out(a.out))
    print(f"{len(snap.metrics)} samples, scrape {snap.latency_s or 0:.3f}s -> {a.out}")
    return 0


def _cmd_watch(a: argparse.Namespace) -> int:
    stop = threading.Event()

    def on_signal(_sig: int, _frame: object) -> None:
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, on_signal)
    with _out(a.out).open("w", encoding="utf-8") as fh:
        n = series.watch(
            a.url,
            fh,
            interval=a.interval,
            stop=stop,
            duration=a.duration,
            on_error=lambda e: print(f"scrape failed: {e}", file=sys.stderr),
        )
    print(f"{n} samples -> {a.out}")
    return 0


def _cmd_preemption(a: argparse.Namespace) -> int:
    r = checks.check_no_preemption(telemetry.load(a.before), telemetry.load(a.after))
    return _emit([r], a.json, a.strict)


def _cmd_rate(a: argparse.Namespace) -> int:
    r = checks.precise_rate(telemetry.load(a.before), telemetry.load(a.after), a.counter)
    if not a.json and "integer_second_error" in r.evidence:
        r = CheckResult(
            r.tripwire,
            r.status,
            f"{r.message}; an integer-second timer would say "
            f"{r.evidence['integer_second_rate_per_s']:.2f} "
            f"({r.evidence['integer_second_error']:+.2%})",
            r.evidence,
        )
    return _emit([r], a.json, a.strict)


def _cmd_series(a: argparse.Namespace) -> int:
    s = series.read(a.series)
    results: list[CheckResult] = []
    usable: int | None = None
    if a.snapshot:
        r = checks.check_null_block(s, telemetry.load(a.snapshot))
        results.append(r)
        if r.status is Status.PASS:
            inferred = r.evidence.get("inferred_usable_blocks")
            usable = inferred if isinstance(inferred, int) else None
    results.append(checks.concurrency_ceiling(s, usable))
    if a.requested:
        facts = bootlog.parse_file(a.boot_log) if a.boot_log else None
        results.append(checks.check_concurrency_reached(s, a.requested, facts))
    return _emit(results, a.json, a.strict)


def _cmd_teardown(a: argparse.Namespace) -> int:
    procs = teardown.list_processes()
    targets = teardown.server_processes(procs)
    naive = teardown.naive_pkill_matches(procs)
    for p in targets:
        print(f"target  pid={p.pid:<8} {p.cmdline[:100]}")
    if not targets and not a.json:
        print("no vLLM server processes found")
    missed = [p for p in targets if p not in naive]
    wrong = [p for p in naive if p not in targets]
    if missed or wrong:
        print(f"(pkill -f 'vllm serve' would miss {len(missed)} and wrongly hit {len(wrong)})")
    if a.dry_run:
        return 0
    _, res = teardown.teardown(
        max_used_mib=a.max_used_mib, timeout_s=a.timeout, consecutive=a.consecutive
    )
    r = CheckResult(
        "T2",
        Status.PASS if res.clear else Status.FAIL,
        res.reason,
        {"readings": res.readings, "signalled": [p.pid for p in targets]},
    )
    return _emit([r], a.json, a.strict)


def _cmd_gpu_clear(a: argparse.Namespace) -> int:
    def live() -> int:
        return len(teardown.server_processes(teardown.list_processes()))

    res = teardown.wait_clear(
        live_servers=live,
        max_used_mib=a.max_used_mib,
        consecutive=a.consecutive,
        timeout_s=a.timeout,
    )
    r = CheckResult(
        "T2",
        Status.PASS if res.clear else Status.FAIL,
        res.reason if res.clear else f"card not clear; refusing to boot. {res.reason}",
        {"readings": res.readings},
    )
    return _emit([r], a.json, a.strict)


def _cmd_report(a: argparse.Namespace) -> int:
    inputs: list[str | None] = [*(a.boot_log or []), a.before, a.after, a.series]
    rep = report.build(
        boot_logs=a.boot_log or [],
        before=telemetry.load(a.before) if a.before else None,
        after=telemetry.load(a.after) if a.after else None,
        series=series.read(a.series) if a.series else None,
        requested=a.requested,
        title=a.title,
        sources=[Path(p).name for p in inputs if p],
    )
    _out(a.out).write_text(report.render(rep), encoding="utf-8")
    counts = {s: sum(1 for r in rep.results if r.status is s) for s in Status}
    summary = ", ".join(f"{n} {s.value}" for s, n in counts.items() if n)
    print(f"{len(rep.results)} checks ({summary}) -> {a.out}")
    return 0


def _cmd_explain(a: argparse.Namespace) -> int:
    ids = [i.upper() for i in a.ids] or list(TRIPWIRES)
    unknown = [i for i in ids if i not in TRIPWIRES]
    if unknown:
        print(f"unknown tripwire: {', '.join(unknown)} (known: T1-T{len(TRIPWIRES)})")
        return 2
    width = min(shutil.get_terminal_size((88, 20)).columns, 88)
    for i in ids:
        t = TRIPWIRES[i]
        print(f"{t.id}  {t.name}")
        for para in (t.what, "Why it matters: " + t.why):
            print(textwrap.fill(para, width, initial_indent="    ", subsequent_indent="    "))
        print(f"    Check: {t.command}")
        print()
    return 0


def _cmd_probe(a: argparse.Namespace) -> int:
    r = probe(a.url, model=a.model, concurrency=a.concurrency, max_tokens=a.max_tokens)
    return _emit([r], a.json, a.strict)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="inferlint", description=__doc__.splitlines()[0] if __doc__ else ""
    )
    ap.add_argument("--version", action="version", version=f"inferlint {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="machine-readable output")
    common.add_argument("--strict", action="store_true", help="treat warnings as failures")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name: str, help_: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help_, parents=[common])

    p = add("boot-facts", "parse a vLLM boot log into structured facts")
    p.add_argument("log")
    p.set_defaults(fn=_cmd_boot_facts)

    p = add("check-log", "run the log-based tripwires (T4 T5 T6 T7 T11 T12) on boot logs")
    p.add_argument("logs", nargs="+")
    p.set_defaults(fn=_cmd_check_log)

    p = add("snapshot", "save a /metrics snapshot with timing")
    p.add_argument("url")
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=_cmd_snapshot)

    p = add("watch", "sample running/waiting/KV-usage gauges to a JSONL series")
    p.add_argument("url")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--duration", type=float, default=None, help="seconds; default until SIGTERM")
    p.set_defaults(fn=_cmd_watch)

    p = add("preemption", "T1: did the server preempt between two snapshots?")
    p.add_argument("before")
    p.add_argument("after")
    p.set_defaults(fn=_cmd_preemption)

    p = add("rate", "T10: token throughput from the snapshots' own clocks")
    p.add_argument("before")
    p.add_argument("after")
    p.add_argument("--counter", default=checks.GENERATION_TOKENS)
    p.set_defaults(fn=_cmd_rate)

    p = add("series", "T8 T9 T14: concurrency reached, ceiling, block count from a series")
    p.add_argument("series")
    p.add_argument("--snapshot", help="a snapshot from the same boot (for T14)")
    p.add_argument("--requested", type=int, help="client concurrency (for T8)")
    p.add_argument("--boot-log", help="boot log, to set the boot line's claim beside T8")
    p.set_defaults(fn=_cmd_series)

    for name, fn, help_ in (
        ("teardown", _cmd_teardown, "T2: stop every vLLM process and wait for a clear card"),
        ("gpu-clear", _cmd_gpu_clear, "T2: exit non-zero unless the card is clear (boot gate)"),
    ):
        p = add(name, help_)
        p.add_argument("--max-used-mib", type=int, default=teardown.DEFAULT_MAX_USED_MIB)
        p.add_argument("--consecutive", type=int, default=2)
        p.add_argument("--timeout", type=float, default=120.0 if name == "teardown" else 10.0)
        if name == "teardown":
            p.add_argument("--dry-run", action="store_true", help="list targets, signal nothing")
        p.set_defaults(fn=fn)

    p = add("report", "write an HTML report: verdicts, charts and boot facts for a run")
    p.add_argument("-o", "--out", required=True, help="HTML file to write")
    p.add_argument("--boot-log", action="append", help="boot log (repeat for several boots)")
    p.add_argument("--before", help="snapshot taken before the run")
    p.add_argument("--after", help="snapshot taken after the run")
    p.add_argument("--series", help="gauge series recorded during the run")
    p.add_argument("--requested", type=int, help="client concurrency")
    p.add_argument("--title", help="report title")
    p.set_defaults(fn=_cmd_report)

    p = add("explain", "explain the tripwires in plain words")
    p.add_argument("ids", nargs="*", help="e.g. T1 T9; default: all")
    p.set_defaults(fn=_cmd_explain)

    p = add("probe", "T7: one request, a concurrent burst, then a health check")
    p.add_argument("url")
    p.add_argument("--model")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=16)
    p.set_defaults(fn=_cmd_probe)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    fn: Any = a.fn
    return int(fn(a))


if __name__ == "__main__":
    sys.exit(main())
