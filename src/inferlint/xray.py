"""``inferlint xray``: one command around a whole benchmark run.

It does in order what the separate commands do one at a time:

1. With ``serve``: check the GPU is free (T2), start the server with its output saved to
   ``boot.log``, and wait until it is ready. Without it, attach to a running server.
2. Check the boot log (T5, T6, T7, T12) and send real requests (T7).
3. Read the server's counters, record its gauges while the load command runs, and read
   the counters again.
4. Check the run (T1, T8, T9, T10, T14, and T15 when the load tool saved a result file).
5. With ``serve``: stop every server process and wait for a free GPU (T2), even when an
   earlier step failed or the run was interrupted.
6. Write ``report.html`` and ``results.json``.

Everything lands in one folder, so the run can be checked again later with the
separate commands.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import cast

from . import benchresult, bootlog, checks, report, series, teardown, telemetry
from .probe import probe
from .result import LABELS, CheckResult, Status

__all__ = [
    "Outcome",
    "Plan",
    "exit_code",
    "load_concurrency",
    "prepare_load",
    "run",
    "server_version",
    "untimed_requests",
]

READY_MARKER = "Application startup complete"

# POSIX only; --serve refuses to run where /proc is missing, so Windows never calls them.
_killpg: Callable[[int, int], None] | None = getattr(os, "killpg", None)
_SIGKILL: int = getattr(signal, "SIGKILL", 9)


@dataclass
class Plan:
    out: Path
    load: list[str]
    url: str = "http://127.0.0.1:8000"
    serve: str | None = None
    boot_log: Path | None = None
    requested: int | None = None
    bench_result: Path | None = None
    interval_s: float = 0.25
    ready_timeout_s: float = 900.0
    probe: bool = True
    title: str | None = None


@dataclass
class Outcome:
    results: list[CheckResult] = field(default_factory=list[CheckResult])
    load_exit: int | None = None
    report: Path | None = None
    problem: str | None = None  # why the run stopped early, if it did
    interrupted: bool = False


@dataclass
class _Run:
    """What one run collected; filled in step by step, so a run cut short keeps its files."""

    server: subprocess.Popen[bytes] | None = None
    boot_log: Path | None = None
    facts: bootlog.BootFacts | None = None
    before: telemetry.Snapshot | None = None
    after: telemetry.Snapshot | None = None
    recorded: series.Series | None = None
    bench: benchresult.BenchResult | None = None
    requested: int | None = None
    files: list[str] = field(default_factory=list[str])


# --------------------------------------------------------------------------- load command


def _flag_value(cmd: Sequence[str], flag: str) -> str | None:
    for i, tok in enumerate(cmd):
        if tok == flag and i + 1 < len(cmd):
            return cmd[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None


def load_concurrency(cmd: Sequence[str]) -> int | None:
    """The concurrency a load command asks for, from its ``--max-concurrency``."""
    v = _flag_value(cmd, "--max-concurrency")
    return int(v) if v is not None and v.isdigit() else None


def _is_vllm_bench_serve(cmd: Sequence[str]) -> bool:
    return any(a == "bench" and b == "serve" for a, b in pairwise(cmd))


def prepare_load(cmd: Sequence[str], out: Path) -> tuple[list[str], Path | None]:
    """The load command to run, and where its result file will be.

    For ``vllm bench serve`` without ``--save-result``, the flags that save the result
    into the run folder are added, so the client's own counts can be compared with the
    server's (T15). A command that already saves its result is left as it is.
    """
    cmd = list(cmd)
    if not _is_vllm_bench_serve(cmd):
        return cmd, None
    if "--save-result" not in cmd:
        target = out / "bench.json"
        cmd += ["--save-result", "--result-dir", str(out), "--result-filename", target.name]
        return cmd, target
    name = _flag_value(cmd, "--result-filename")
    if name is None:
        return cmd, None  # vLLM names the file; found after the run by _newest_json
    return cmd, Path(_flag_value(cmd, "--result-dir") or ".") / name


def untimed_requests(load_output: str) -> int:
    """Requests ``vllm bench serve`` sent but left out of its result file, from its output.

    It prints "Initial test run completed." after one test request (whether it sends one
    depends on ``--ready-check-timeout-sec`` and its default, which has changed between
    releases) and "Warming up with N requests..." for ``--num-warmups``.
    """
    n = 1 if "Initial test run completed." in load_output else 0
    m = re.search(r"Warming up with (\d+) requests", load_output)
    return n + (int(m.group(1)) if m else 0)


def _newest_json(folder: Path, since: float) -> Path | None:
    found = [p for p in folder.glob("*.json") if p.stat().st_mtime >= since]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def _run_load(cmd: Sequence[str], log: Path, say: Callable[[str], None]) -> int:
    """Run the load command, showing its output and keeping a copy in ``load.log``."""
    with log.open("w", encoding="utf-8") as fh:
        p = subprocess.Popen(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert p.stdout is not None
        for line in p.stdout:
            fh.write(line)
            say(line.rstrip("\n"))
        return p.wait()


# --------------------------------------------------------------------------- server


def _linux() -> bool:
    """Stopping a server finds its processes in /proc, so --serve needs Linux (or WSL)."""
    return Path("/proc").is_dir()


def _healthy(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as r:
            return int(r.status) == 200
    except (urllib.error.URLError, OSError):
        return False


def server_version(url: str, timeout: float = 5.0) -> str | None:
    """The version the server reports: vLLM and TensorRT-LLM on ``/version``, SGLang on
    ``/get_server_info``."""
    for path in ("/version", "/get_server_info"):
        try:
            with urllib.request.urlopen(url.rstrip("/") + path, timeout=timeout) as r:
                doc: object = json.loads(r.read().decode("utf-8", "replace"))
        except (urllib.error.URLError, OSError, ValueError):
            continue
        if isinstance(doc, dict):
            v = cast(dict[str, object], doc).get("version")
            if isinstance(v, str) and v:
                return v
    return None


def _start_server(cmd: str, log: Path) -> subprocess.Popen[bytes]:
    with log.open("wb") as fh:
        # A session of its own, so the whole server (API process and engine) can be
        # signalled at once. The child keeps its own copy of the log file handle.
        return subprocess.Popen(
            shlex.split(cmd), stdout=fh, stderr=subprocess.STDOUT, start_new_session=True
        )


def _wait_ready(
    proc: subprocess.Popen[bytes], log: Path, url: str, timeout_s: float, poll_s: float = 2.0
) -> str | None:
    """None once the server is ready; otherwise why it is not."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return f"the server exited with code {proc.returncode} before it was ready"
        if READY_MARKER in log.read_text(encoding="utf-8", errors="replace") and _healthy(url):
            return None
        time.sleep(poll_s)
    return f"the server was not ready after {timeout_s:g} s"


def _stop_server(proc: subprocess.Popen[bytes] | None, grace_s: float = 30.0) -> CheckResult:
    """Stop the server's session, then any other vLLM process, and wait for a free GPU."""
    if proc is not None and proc.poll() is None and _killpg is not None:
        try:
            _killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            _killpg(proc.pid, _SIGKILL)
            proc.wait(timeout=10)
        except ProcessLookupError:
            pass
    leftovers, res = teardown.teardown()
    return CheckResult(
        "T2",
        Status.PASS if res.clear else Status.FAIL,
        res.reason,
        {"readings": res.readings, "left_after_the_server_stopped": [p.pid for p in leftovers]},
    )


def _gpu_free() -> CheckResult:
    def live() -> int:
        return len(teardown.server_processes(teardown.list_processes()))

    res = teardown.wait_clear(live_servers=live, timeout_s=10.0)
    return CheckResult(
        "T2",
        Status.PASS if res.clear else Status.FAIL,
        res.reason if res.clear else f"card not clear; refusing to start a server. {res.reason}",
        {"readings": res.readings},
    )


def _boot_checks(log: Path) -> list[CheckResult]:
    text = log.read_text(encoding="utf-8", errors="replace")
    facts = bootlog.parse(text)
    return [
        checks.check_boot_failure(text),
        checks.check_block_size(facts),
        checks.check_backend_honoured(facts),
    ]


# --------------------------------------------------------------------------- the run


def _line(r: CheckResult) -> str:
    return f"[{r.tripwire:>3}] {LABELS[r.status]}  {r.message}"


def run(plan: Plan, say: Callable[[str], None] = print) -> Outcome:
    """Run the whole sequence. Findings never raise; they are in ``Outcome.results``."""
    plan.out.mkdir(parents=True, exist_ok=True)
    o = Outcome()
    st = _Run(boot_log=plan.boot_log)

    def record(r: CheckResult) -> None:
        o.results.append(r)
        say(_line(r))

    try:
        _collect(plan, o, st, record, say)
    except KeyboardInterrupt:
        o.interrupted, o.problem = True, "interrupted"
    finally:
        if st.server is not None:
            say("== stop the server")
            try:
                record(_stop_server(st.server))
            except KeyboardInterrupt:
                o.interrupted = True
    if o.problem:
        say(o.problem)
    if st.boot_log is not None or (st.before is not None and st.after is not None):
        o.report = plan.out / "report.html"
        o.report.write_text(report.render(_report(plan, o, st)), encoding="utf-8")
    (plan.out / "results.json").write_text(
        json.dumps([r.to_json() for r in o.results], indent=1, default=str), encoding="utf-8"
    )
    return o


def _collect(
    plan: Plan,
    o: Outcome,
    st: _Run,
    record: Callable[[CheckResult], None],
    say: Callable[[str], None],
) -> None:
    out = plan.out
    if plan.serve is not None:
        if not _linux():
            o.problem = "--serve needs Linux (or WSL): stopping the server reads /proc"
            return
        say("== is the GPU free?")
        gate = _gpu_free()
        record(gate)
        if gate.status is not Status.PASS:
            o.problem = "the GPU is not free; start no server on it"
            return
        st.boot_log = out / "boot.log"
        say(f"== start the server (output in {st.boot_log})")
        st.server = _start_server(plan.serve, st.boot_log)
        why = _wait_ready(st.server, st.boot_log, plan.url, plan.ready_timeout_s)
        if why is not None:
            o.problem = why
            for r in _boot_checks(st.boot_log):
                record(r)
            return
    elif not _healthy(plan.url):
        o.problem = f"no server answers at {plan.url}/health"
        return

    if st.boot_log is not None:
        say("== boot log")
        st.facts = bootlog.parse_file(st.boot_log)
        if st.facts.vllm_version is None:  # SGLang does not print its version
            st.facts.server_version = server_version(plan.url)
        note = bootlog.untested_version(st.facts)
        if note:
            say(f"note: {note}")
        for r in _boot_checks(st.boot_log):
            record(r)
        st.files.append(st.boot_log.name)
    if plan.probe:
        say("== probe: one request, then 8 at once")
        record(probe(plan.url))

    st.before = telemetry.scrape(plan.url)
    st.before.save(out / "before.snapshot.json")
    stop = threading.Event()
    series_path = out / "run.series.jsonl"
    cmd, result_path = prepare_load(plan.load, out)
    with series_path.open("w", encoding="utf-8") as fh:
        watcher = threading.Thread(
            target=series.watch,
            args=(plan.url, fh),
            kwargs={"interval": plan.interval_s, "stop": stop},
            daemon=True,
        )
        watcher.start()
        say("== load: " + shlex.join(cmd))
        t_load = time.time()
        try:
            o.load_exit = _run_load(cmd, out / "load.log", say)
        finally:
            st.after = telemetry.scrape(plan.url)
            st.after.save(out / "after.snapshot.json")
            stop.set()
            watcher.join(timeout=10)
    st.files += ["before.snapshot.json", "after.snapshot.json", series_path.name]
    st.recorded = series.read(series_path)
    if o.load_exit:
        say(f"the load command exited with code {o.load_exit}")

    path = plan.bench_result or result_path
    if path is None and _is_vllm_bench_serve(cmd):
        path = _newest_json(Path(_flag_value(cmd, "--result-dir") or "."), t_load)
    if path is not None and path.is_file():
        st.bench = benchresult.load(path)
        st.files.append(path.name)
    st.requested = plan.requested or load_concurrency(cmd)
    if st.requested is None and st.bench is not None:
        st.requested = st.bench.max_concurrency

    if st.boot_log is not None and st.facts is not None:
        # The server log now covers the run too (TensorRT-LLM logs its pauses there).
        version = st.facts.server_version
        st.facts = bootlog.parse_file(st.boot_log)
        st.facts.server_version = version
    say("== checks")
    for r in report.run_checks(
        before=st.before,
        after=st.after,
        series=st.recorded,
        requested=st.requested,
        facts=st.facts,
    ):
        record(checks.with_timer_comparison(r) if r.tripwire == "T10" else r)
    if st.bench is not None:
        untimed = 0
        if _is_vllm_bench_serve(cmd):
            untimed = untimed_requests(
                (out / "load.log").read_text(encoding="utf-8", errors="replace")
            )
        record(
            checks.check_client_server_agree(
                st.bench,
                st.before,
                st.after,
                expected_extra_requests=untimed,
            )
        )


def _report(plan: Plan, o: Outcome, st: _Run) -> report.Report:
    return report.build(
        boot_logs=[st.boot_log] if st.boot_log is not None else [],
        before=st.before,
        after=st.after,
        series=st.recorded,
        requested=st.requested,
        title=plan.title,
        sources=st.files,
        extra=[r for r in o.results if r.tripwire in ("T2", "T7", "T15")],
        server_version=st.facts.server_version if st.facts is not None else None,
    )


def exit_code(o: Outcome, strict: bool = False) -> int:
    """0 all passed, 1 a tripwire failed or the run did not complete, 2 could not decide."""
    statuses = {r.status for r in o.results}
    if o.interrupted:
        return 130
    if o.problem or o.load_exit or Status.FAIL in statuses:
        return 1
    if strict and Status.WARN in statuses:
        return 1
    if Status.UNKNOWN in statuses:
        return 2
    return 0
