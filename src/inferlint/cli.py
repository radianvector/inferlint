"""inferlint: checks an inference server's benchmark run for silent problems (tripwires).

Exit status: 0 every tripwire passed (warnings allowed), 1 a tripwire failed,
2 a tripwire could not decide (a series or log line was missing), or the input came from
another engine than the one named. ``--strict`` turns warnings into failures.

The engine (vLLM, SGLang, TensorRT-LLM) is told from the input: metric names, boot log,
process names. ``--engine`` or the ``INFERLINT_ENGINE`` environment variable names it;
an input from another engine is then refused, and an input that names none is read as
that engine's.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import (
    __version__,
    benchresult,
    bootlog,
    checks,
    engines,
    gpu,
    insights,
    report,
    series,
    teardown,
    telemetry,
    xray,
)
from .catalog import ENGINE_NOTES, TRIPWIRES, lookup
from .probe import probe
from .result import LABELS, CheckResult, Status

_MARK = LABELS


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


def _named(a: argparse.Namespace) -> engines.Engine | None:
    """The engine the user named, by --engine or INFERLINT_ENGINE; None to tell it."""
    key: str | None = getattr(a, "engine", None)
    return engines.by_key(key) if key else None


def _how_named(a: argparse.Namespace) -> str:
    if getattr(a, "engine_from_env", False):
        return f"{engines.ENV_VAR}={a.engine}"
    return f"--engine {a.engine}"


def _other_engine(
    a: argparse.Namespace, found: engines.Engine | None, what: str, verb: str = "came from"
) -> bool:
    """Say so on stderr when ``what`` came from another engine than the one named."""
    named = _named(a)
    if named is None or found is None or found is named:
        return False
    print(
        f"inferlint: {_how_named(a)}, but {what} {verb} {found.name}. "
        "Check the server address or the file, or name the engine it came from.",
        file=sys.stderr,
    )
    return True


def _note_unknown(a: argparse.Namespace, found: engines.Engine | None, what: str) -> None:
    """Say on stderr when nothing tells the engine and none was named: vLLM is assumed."""
    if found is None and _named(a) is None:
        print(
            f"note: {what} does not say which engine it came from; read as vLLM's "
            f"(--engine or {engines.ENV_VAR} names it)",
            file=sys.stderr,
        )


def _metrics_hint(a: argparse.Namespace) -> str:
    named = _named(a)
    if named is not None:
        return named.metrics_hint
    return "; ".join(f"for {e.name}, {e.metrics_hint}" for e in engines.ENGINES)


def _snapshot_engine(snaps: Sequence[telemetry.Snapshot]) -> engines.Engine | None:
    names: set[str] = set()
    for s in snaps:
        names |= s.metrics.names()
    return engines.named_by(names)


def _series_engine(s: series.Series) -> engines.Engine | None:
    """The engine a recording names in its header; None for one that names none."""
    key = s.header.get("engine")
    return engines.by_key(key) if isinstance(key, str) else None


def _note_version(facts: bootlog.BootFacts) -> None:
    """Warn on stderr (so --json output stays clean) about an untested release."""
    note = bootlog.untested_version(facts)
    if note:
        print(f"note: {note}", file=sys.stderr)


def _cmd_boot_facts(a: argparse.Namespace) -> int:
    text = Path(a.log).read_text(encoding="utf-8", errors="replace")
    found = engines.from_log(text)
    if _other_engine(a, found, a.log):
        return 2
    _note_unknown(a, found, a.log)
    facts = bootlog.parse(text, a.engine)
    _note_version(facts)
    doc = asdict(facts)
    if a.json:
        print(json.dumps(doc, indent=1, default=str))
    else:
        for k, v in doc.items():
            skip = ("unparsed", "conflicts", "non_default_args", "server_args")
            if k in skip or v in (None, [], {}, ()):
                continue
            print(f"{k:28} {v}")
        if facts.non_default_args:
            print(f"{'non_default_args':28} {sorted(facts.non_default_args)}")
        if facts.server_args:
            print(f"{'server_args':28} {len(facts.server_args)} settings (--json lists them)")
        for c, vals in facts.conflicts.items():
            print(f"CONFLICT {c}: {vals}")
        for u in facts.unparsed:
            print(f"UNPARSED {u.field} (line {u.lineno}): {u.line}")
    return 2 if (a.strict and facts.unparsed) else 0


def _cmd_check_log(a: argparse.Namespace) -> int:
    results: list[CheckResult] = []
    all_facts: list[bootlog.BootFacts] = []
    names = bootlog.labels(a.logs)
    for p, name in zip(a.logs, names, strict=True):
        text = Path(p).read_text(encoding="utf-8", errors="replace")
        found = engines.from_log(text)
        if _other_engine(a, found, p):
            return 2
        _note_unknown(a, found, p)
        facts = bootlog.parse(text, a.engine)
        if not all_facts:
            _note_version(facts)
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
                    CheckResult(r.tripwire, r.status, f"{name}: {r.message}", r.evidence)
                )
        else:
            results.extend(per)
    if len(a.logs) > 1:
        results.extend(checks.check_same_pool(all_facts, names))
        results.extend(checks.check_kv_memory_stable(all_facts, names))
    return _emit(results, a.json, a.strict)


def _served_snapshot(a: argparse.Namespace) -> telemetry.Snapshot | None:
    """A snapshot of the server at ``a.url``; None (said on stderr) when it serves no
    engine's metrics, or another engine's than the one named."""
    try:
        snap = telemetry.scrape(a.url)
    except (OSError, ValueError) as e:  # URLError and HTTPError are OSErrors
        snap, why = None, f" ({e})"
    else:
        why = ""
    found = _snapshot_engine([snap]) if snap is not None else None
    if found is None:
        print(
            f"inferlint: no vLLM, SGLang or TensorRT-LLM metrics at {a.url}{why}: "
            f"{_metrics_hint(a)}",
            file=sys.stderr,
        )
        return None
    return None if _other_engine(a, found, f"the server at {a.url}", "runs") else snap


def _cmd_snapshot(a: argparse.Namespace) -> int:
    snap = _served_snapshot(a)
    if snap is None:
        return 2
    snap.save(_out(a.out))
    print(f"{len(snap.metrics)} samples, scrape {snap.latency_s or 0:.3f}s -> {a.out}")
    return 0


def _cmd_watch(a: argparse.Namespace) -> int:
    stop = threading.Event()

    def on_signal(_sig: int, _frame: object) -> None:
        stop.set()

    if _served_snapshot(a) is None:
        return 2
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


def _snapshots(a: argparse.Namespace) -> tuple[telemetry.Snapshot, telemetry.Snapshot] | None:
    """The before and after snapshots; None (said on stderr) if another engine's."""
    before, after = telemetry.load(a.before), telemetry.load(a.after)
    found = _snapshot_engine([before, after])
    if _other_engine(a, found, "the snapshots"):
        return None
    _note_unknown(a, found, "the snapshots")
    return before, after


def _cmd_preemption(a: argparse.Namespace) -> int:
    snaps = _snapshots(a)
    if snaps is None:
        return 2
    s = series.read(a.series) if a.series else None
    if s is not None and _other_engine(a, _series_engine(s), a.series):
        return 2
    facts = bootlog.parse_file(a.boot_log, a.engine) if a.boot_log else None
    if (
        facts is not None
        and facts.engine
        and _other_engine(a, engines.by_key(facts.engine), a.boot_log)
    ):
        return 2
    r = checks.check_no_preemption(*snaps, s, facts)
    return _emit([r], a.json, a.strict)


def _cmd_rate(a: argparse.Namespace) -> int:
    snaps = _snapshots(a)
    if snaps is None:
        return 2
    r = checks.precise_rate(*snaps, a.counter)
    if not a.json:
        r = checks.with_timer_comparison(r)
    return _emit([r], a.json, a.strict)


def _cmd_series(a: argparse.Namespace) -> int:
    s = series.read(a.series)
    if _other_engine(a, _series_engine(s), a.series):
        return 2
    results: list[CheckResult] = []
    usable: int | None = None
    snap = telemetry.load(a.snapshot) if a.snapshot else None
    if snap is not None and checks.engine_of(snap).metrics.cache_info is not None:
        r = checks.check_null_block(s, snap)
        results.append(r)
        if r.status is Status.PASS:
            inferred = r.evidence.get("inferred_usable_blocks")
            usable = inferred if isinstance(inferred, int) else None
    results.append(checks.concurrency_ceiling(s, usable))
    if a.requested:
        facts = bootlog.parse_file(a.boot_log, a.engine) if a.boot_log else None
        results.append(checks.check_concurrency_reached(s, a.requested, facts))
    return _emit(results, a.json, a.strict)


def _procs_readable() -> bool:
    """Finding server processes reads /proc, so it needs Linux (or WSL)."""
    return Path("/proc").is_dir()


def _needs_linux(what: str, evidence: dict[str, Any] | None = None) -> CheckResult:
    return CheckResult(
        "T2",
        Status.UNKNOWN,
        f"{what} needs Linux: it finds the server's processes in /proc. "
        "Run it where the server runs.",
        evidence or {},
    )


def _cmd_teardown(a: argparse.Namespace) -> int:
    if not _procs_readable():
        return _emit([_needs_linux("teardown")], a.json, a.strict)
    procs = teardown.list_processes()
    named = _named(a)
    targets = teardown.server_processes(procs, engine=a.engine)
    pattern = teardown.usual_pkill_pattern(targets)
    naive = teardown.naive_pkill_matches(procs, pattern)
    for p in targets:
        print(f"target  pid={p.pid:<8} {p.cmdline[:100]}")
    if not targets and not a.json:
        which = named.name if named else "vLLM, SGLang or TensorRT-LLM"
        print(f"no server processes found ({which})")
    others = [p for p in teardown.server_processes(procs) if p not in targets]
    if others and not a.json:
        kinds = sorted({engines.by_key(k).name for p in others if (k := teardown.server_engine(p))})
        which = " and ".join(kinds)
        print(f"left alone ({_how_named(a)}): {len(others)} processes of {which} servers")
    missed = [p for p in targets if p not in naive]
    wrong = [p for p in naive if p not in targets]
    if missed or wrong:
        print(f"(pkill -f '{pattern}' would miss {len(missed)} and wrongly hit {len(wrong)})")
    if a.dry_run:
        return 0
    _, res = teardown.teardown(
        max_used_mib=a.max_used_mib,
        timeout_s=a.timeout,
        consecutive=a.consecutive,
        engine=a.engine,
    )
    r = CheckResult(
        "T2",
        Status.PASS if res.clear else Status.FAIL,
        res.reason,
        {"readings": res.readings, "signalled": [p.pid for p in targets]},
    )
    return _emit([r], a.json, a.strict)


def _mib_text(v: int | None) -> str:
    return "?" if v is None else f"{v:,} MiB"


def _print_gpus(gpus: Sequence[gpu.GpuInfo]) -> None:
    if not gpus:
        print("no NVIDIA GPU found: nvidia-smi is missing or failed")
    for g in gpus:
        kind = ", ".join(
            x
            for x in (
                g.family,
                f"compute capability {g.compute_cap}" if g.compute_cap else None,
                f"driver {g.driver}" if g.driver else None,
            )
            if x
        )
        print(f"GPU {g.index}   {g.name}")
        if kind:
            print(f"        {kind}")
        print(
            f"        memory: {_mib_text(g.memory_total_mib)} total, "
            f"{_mib_text(g.memory_used_mib)} used, {_mib_text(g.memory_free_mib)} free"
        )


def _cmd_gpu_inspect(a: argparse.Namespace) -> int:
    try:
        gpus = gpu.read_gpus()
    except (OSError, subprocess.SubprocessError):
        gpus = []
    facts = [g.to_json() for g in gpus]
    if not a.json:
        _print_gpus(gpus)
    if not _procs_readable():
        r = _needs_linux("Checking that the GPU is free", {"gpus": facts})
        return _emit([r], a.json, a.strict)

    def live() -> int:
        return len(teardown.server_processes(teardown.list_processes()))

    res = teardown.wait_clear(
        read_used=teardown.read_gpu_used_mib,
        live_servers=live,
        max_used_mib=a.max_used_mib,
        consecutive=a.consecutive,
        timeout_s=a.timeout,
    )
    r = CheckResult(
        "T2",
        Status.PASS if res.clear else Status.FAIL,
        res.reason if res.clear else f"card not clear; refusing to boot. {res.reason}",
        {"readings": res.readings, "gpus": facts},
    )
    return _emit([r], a.json, a.strict)


def _first(folder: Path, name: str, pattern: str) -> str | None:
    """``folder/name`` if it exists, else the first file matching ``pattern``."""
    if (folder / name).is_file():
        return str(folder / name)
    found = sorted(folder.glob(pattern))
    return str(found[0]) if found else None


def _bench_in(folder: Path) -> str | None:
    """The load tool's result file in a run folder: bench.json, or a JSON file like one."""
    if (folder / "bench.json").is_file():
        return str(folder / "bench.json")
    for f in sorted(folder.glob("*.json")):
        if f.name.endswith(".snapshot.json") or f.name == "results.json":
            continue
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and "completed" in doc and "total_output_tokens" in doc:
            return str(f)
    return None


def _print_limits(found: Sequence[insights.Insight]) -> None:
    if found:
        print("What limited this run:")
        for i in found:
            print(f"  - {i.title}. {i.text}")


def _cmd_report(a: argparse.Namespace) -> int:
    run = Path(a.run) if a.run else None
    if run is not None and not run.is_dir():
        print(f"inferlint: {a.run} is not a folder", file=sys.stderr)
        return 2
    if run is None and not a.out:
        print("inferlint: give a run folder, or -o and the files to read", file=sys.stderr)
        return 2
    # Files named on the command line win; the rest are found in the run folder.
    logs: list[str] = list(a.boot_log or [])
    if run is not None:
        if not logs:
            log = _first(run, "boot.log", "*boot*.log")
            logs = [log] if log else []
        a.before = a.before or _first(run, "before.snapshot.json", "*before*.json")
        a.after = a.after or _first(run, "after.snapshot.json", "*after*.json")
        a.series = a.series or _first(run, "run.series.jsonl", "*.series.jsonl")
        a.bench_result = a.bench_result or _bench_in(run)
        if not (logs or a.before or a.after or a.series):
            print(
                f"inferlint: no run files in {run} (boot.log, *.snapshot.json, *.series.jsonl)",
                file=sys.stderr,
            )
            return 2
    out = a.out or str((run or Path()) / "report.html")
    inputs: list[str | None] = [*logs, a.before, a.after, a.series, a.bench_result]
    before = telemetry.load(a.before) if a.before else None
    after = telemetry.load(a.after) if a.after else None
    recorded = series.read(a.series) if a.series else None
    snaps = [x for x in (before, after) if x is not None]
    found: list[tuple[engines.Engine | None, str]] = []
    if snaps:
        found.append((_snapshot_engine(snaps), "the snapshots"))
    if recorded is not None:
        found.append((_series_engine(recorded), str(a.series)))
    for log in logs:
        text = Path(log).read_text(encoding="utf-8", errors="replace")
        found.append((engines.from_log(text), log))
    if any(_other_engine(a, e, what) for e, what in found):
        return 2
    if found and not any(e for e, _ in found):
        _note_unknown(a, None, "no input")
    bench = benchresult.load(a.bench_result) if a.bench_result else None
    extra: list[CheckResult] = []
    if bench is not None and before is not None and after is not None:
        load_log = run / "load.log" if run is not None else None
        untimed = 0
        if load_log is not None and load_log.is_file():
            untimed = xray.untimed_requests(load_log.read_text(encoding="utf-8", errors="replace"))
        extra.append(
            checks.check_client_server_agree(bench, before, after, expected_extra_requests=untimed)
        )
    rep = report.build(
        boot_logs=logs,
        before=before,
        after=after,
        series=recorded,
        requested=a.requested or (bench.max_concurrency if bench else None),
        title=a.title,
        sources=[Path(f).name for f in inputs if f],
        engine=a.engine,
        extra=extra,
        bench=bench,
    )
    _out(out).write_text(report.render(rep), encoding="utf-8")
    if a.json:
        print(json.dumps([r.to_json() for r in rep.results], indent=1, default=str))
        return 0
    for r in rep.results:
        print(f"[{r.tripwire:>3}] {_MARK[r.status]}  {r.message}")
    _print_limits(insights.limits(rep))
    print(f"{_counts(rep.results)} -> {out}")
    return 0


def _counts(results: Sequence[CheckResult]) -> str:
    """'8 checks (4 pass, 2 warn, 2 tripwire-failed)'."""
    counts = {s: sum(1 for r in results if r.status is s) for s in Status}
    summary = ", ".join(
        f"{n} {'unknown' if s is Status.UNKNOWN else LABELS[s].lower()}"
        for s, n in counts.items()
        if n
    )
    return f"{len(results)} checks ({summary})"


def _cmd_explain(a: argparse.Namespace) -> int:
    keys = a.ids or list(TRIPWIRES)
    found = [lookup(k) for k in keys]
    unknown = [k for k, t in zip(keys, found, strict=True) if t is None]
    if unknown:
        print(
            f"unknown tripwire: {', '.join(unknown)} "
            f"(known: T1-T{len(TRIPWIRES)}, or a name such as silent-preemption)"
        )
        return 2
    width = min(shutil.get_terminal_size((88, 20)).columns, 88)
    for t in found:
        assert t is not None
        print(f"{t.id}  {t.name}  ({t.slug})")
        named = _named(a)
        notes = [
            f"On {engines.by_key(k).name}: {v}"
            for k, v in ENGINE_NOTES.get(t.id, {}).items()
            if named is None or k == named.key
        ]
        for para in (t.what, "Why it matters: " + t.why, *notes):
            print(textwrap.fill(para, width, initial_indent="    ", subsequent_indent="    "))
        print(f"    Check: {t.command}")
        print()
    return 0


def _cmd_xray(a: argparse.Namespace) -> int:
    load = list(a.load)
    if load and load[0] == "--":
        load = load[1:]
    if not load:
        print("xray needs the load command after --, e.g. -- vllm bench serve ...")
        return 2
    out = Path(a.out or time.strftime("xray-%Y%m%d-%H%M%S"))
    plan = xray.Plan(
        out=out,
        load=load,
        url=a.url,
        serve=a.serve,
        boot_log=Path(a.boot_log) if a.boot_log else None,
        requested=a.requested,
        bench_result=Path(a.bench_result) if a.bench_result else None,
        interval_s=a.interval,
        ready_timeout_s=a.ready_timeout,
        probe=not a.no_probe,
        title=a.title,
        engine=a.engine,
    )
    o = xray.run(plan, say=lambda line: print(line, flush=True))
    print(_counts(o.results))
    if o.report is not None:
        print(f"report: {o.report}")
    print(f"files:  {out}")
    return xray.exit_code(o, a.strict)


def _cmd_probe(a: argparse.Namespace) -> int:
    r = probe(a.url, model=a.model, concurrency=a.concurrency, max_tokens=a.max_tokens)
    return _emit([r], a.json, a.strict)


# The commands as --help lists them, by what they are for. Each line is also the
# command's own description.
_GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "Run and check a benchmark",
        (
            (
                "xray",
                "start the server (optional), record it around the load command, check the "
                "run, write report.html, stop the server",
            ),
        ),
    ),
    (
        "Check a saved run",
        (
            ("report", "check a run folder or its files again: verdicts, and report.html"),
            ("check-log", "the log-based tripwires (T4 T5 T6 T7 T11 T12) on boot logs"),
            ("boot-facts", "what a boot log says: version, KV cache, memory, settings"),
            ("preemption", "T1: did the server preempt between two snapshots?"),
            ("rate", "T10: token throughput from the snapshots' own clocks"),
            ("series", "T8 T9 T14: concurrency reached, ceiling, block count from a recording"),
        ),
    ),
    (
        "Record a server yourself",
        (
            ("snapshot", "save the server's metrics, with the time of the reading"),
            ("watch", "record the running, waiting and KV-cache readings to a series file"),
            ("probe", "T7: one request, a concurrent burst, then a health check"),
        ),
    ),
    (
        "The GPU",
        (
            ("gpu-inspect", "T2: which GPU this is, its memory, and whether it is free"),
            ("teardown", "T2: stop inference servers and wait for a free card"),
        ),
    ),
    ("Learn", (("explain", "what each tripwire catches, and how to check it"),)),
)
_HELP = {name: text for _, cmds in _GROUPS for name, text in cmds}

_START = f"""inferlint checks an inference benchmark run for silent problems (tripwires).

Start here:
  inferlint xray -o run/ --serve "vllm serve MODEL" -- vllm bench serve --model MODEL ...
      start the server, run the load, check the run, write run/report.html, stop the server
  inferlint xray -o run/ --url http://127.0.0.1:8000 -- <load command>
      the same, against a server that is already running
  inferlint report run/
      check a saved run folder again, and rewrite its report.html
  inferlint explain T1
      what a tripwire catches

vLLM, SGLang and TensorRT-LLM are told apart from their files; to name the engine,
set {engines.ENV_VAR} or pass --engine.

Every command: inferlint --help"""


def _commands_text() -> str:
    lines = ["commands:"]
    for group, cmds in _GROUPS:
        lines.append(f"  {group}")
        for name, text in cmds:
            wrapped = textwrap.wrap(text, 62)
            lines.append(f"    {name:<12} {wrapped[0]}")
            lines += [f"    {'':<12} {w}" for w in wrapped[1:]]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="inferlint",
        description=__doc__.splitlines()[0] if __doc__ else "",
        epilog=_commands_text(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--version", action="version", version=f"inferlint {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="machine-readable output")
    common.add_argument("--strict", action="store_true", help="treat warnings as failures")
    engine = argparse.ArgumentParser(add_help=False)
    engine.add_argument(
        "--engine",
        type=_engine_arg,
        metavar="{vllm,sglang,trtllm}",
        help=f"the serving engine (default: ${engines.ENV_VAR}, else told from the input)",
    )
    sub = ap.add_subparsers(
        dest="cmd", required=True, metavar="<command>", help="one of the commands below"
    )

    def add(name: str, by_engine: bool = True) -> argparse.ArgumentParser:
        parents = [common, engine] if by_engine else [common]
        return sub.add_parser(name, description=_HELP[name], parents=parents)

    p = add("boot-facts")
    p.add_argument("log")
    p.set_defaults(fn=_cmd_boot_facts)

    p = add("check-log")
    p.add_argument("logs", nargs="+")
    p.set_defaults(fn=_cmd_check_log)

    p = add("snapshot")
    p.add_argument("url")
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=_cmd_snapshot)

    p = add("watch")
    p.add_argument("url")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--duration", type=float, default=None, help="seconds; default until SIGTERM")
    p.set_defaults(fn=_cmd_watch)

    p = add("preemption")
    p.add_argument("before")
    p.add_argument("after")
    p.add_argument("--series", help="recording of the run, for engines that count no preemptions")
    p.add_argument("--boot-log", help="the server's boot log (its scheduler policy)")
    p.set_defaults(fn=_cmd_preemption)

    p = add("rate")
    p.add_argument("before")
    p.add_argument("after")
    p.add_argument(
        "--counter", default=None, help="counter to divide (default: the generated tokens)"
    )
    p.set_defaults(fn=_cmd_rate)

    p = add("series")
    p.add_argument("series")
    p.add_argument("--snapshot", help="a snapshot from the same boot (for T14)")
    p.add_argument("--requested", type=int, help="client concurrency (for T8)")
    p.add_argument("--boot-log", help="boot log, to set the boot line's claim beside T8")
    p.set_defaults(fn=_cmd_series)

    for name, fn in (("teardown", _cmd_teardown), ("gpu-inspect", _cmd_gpu_inspect)):
        # gpu-inspect: the card is free only when no engine's server holds it.
        p = add(name, by_engine=name == "teardown")
        p.add_argument("--max-used-mib", type=int, default=teardown.DEFAULT_MAX_USED_MIB)
        p.add_argument("--consecutive", type=int, default=2)
        p.add_argument("--timeout", type=float, default=120.0 if name == "teardown" else 10.0)
        if name == "teardown":
            p.add_argument("--dry-run", action="store_true", help="list targets, signal nothing")
        p.set_defaults(fn=fn)

    p = add("report")
    p.add_argument(
        "run",
        nargs="?",
        help="a run folder, such as xray's: its files are found by name",
    )
    p.add_argument("-o", "--out", help="HTML file to write (default: RUN/report.html)")
    p.add_argument("--boot-log", action="append", help="boot log (repeat for several boots)")
    p.add_argument("--before", help="snapshot taken before the run")
    p.add_argument("--after", help="snapshot taken after the run")
    p.add_argument("--series", help="gauge series recorded during the run")
    p.add_argument("--bench-result", help="the load tool's result file (vllm bench serve)")
    p.add_argument("--requested", type=int, help="client concurrency (default: the result's)")
    p.add_argument("--title", help="report title")
    p.set_defaults(fn=_cmd_report)

    p = add("explain")
    p.add_argument("ids", nargs="*", help="codes or names, e.g. T1 falling-ceiling; default: all")
    p.set_defaults(fn=_cmd_explain)

    p = add("xray")
    p.add_argument("-o", "--out", help="folder for every file of the run (default: xray-<time>)")
    p.add_argument("--serve", help='start the server with this command, e.g. "vllm serve M ..."')
    p.add_argument("--url", default="http://127.0.0.1:8000", help="the server's address")
    p.add_argument("--boot-log", help="the running server's boot log (when not using --serve)")
    p.add_argument("--requested", type=int, help="concurrency asked for (default: from the load)")
    p.add_argument("--bench-result", help="the load tool's result file (default: found)")
    p.add_argument("--interval", type=float, default=0.25, help="seconds between readings")
    p.add_argument("--ready-timeout", type=float, default=900.0, help="seconds to wait for boot")
    p.add_argument("--no-probe", action="store_true", help="skip the 9 probe requests (T7)")
    p.add_argument("--title", help="report title")
    p.add_argument("load", nargs=argparse.REMAINDER, help="-- the load command")
    p.set_defaults(fn=_cmd_xray)

    p = add("probe", False)
    p.add_argument("url")
    p.add_argument("--model")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=16)
    p.set_defaults(fn=_cmd_probe)
    return ap


def _engine_arg(value: str) -> str:
    e = engines.parse_key(value)
    if e is None:
        known = ", ".join(x.key for x in engines.ENGINES)
        raise argparse.ArgumentTypeError(f"unknown engine {value!r} (known: {known})")
    return e.key


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(_START)
        return 0
    if not args[0].startswith("-") and args[0] not in _HELP:
        close = difflib.get_close_matches(args[0], list(_HELP), n=1, cutoff=0.6)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        print(
            f"inferlint: unknown command '{args[0]}'.{hint} 'inferlint --help' lists them.",
            file=sys.stderr,
        )
        return 2
    a = build_parser().parse_args(args)
    if hasattr(a, "engine") and a.engine is None:
        env = os.environ.get(engines.ENV_VAR, "").strip()
        if env:
            e = engines.parse_key(env)
            if e is None:
                known = ", ".join(x.key for x in engines.ENGINES)
                print(
                    f"inferlint: {engines.ENV_VAR}={env!r} is not an engine (known: {known})",
                    file=sys.stderr,
                )
                return 2
            a.engine, a.engine_from_env = e.key, True
    fn: Any = a.fn
    return int(fn(a))


if __name__ == "__main__":
    sys.exit(main())
