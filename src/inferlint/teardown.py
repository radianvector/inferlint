"""Stop a vLLM or SGLang server completely, and prove the GPU is free before the next boot.

Two traps, both of which produce numbers rather than errors:

* ``pkill -f "vllm serve"`` signals the API server and misses the engine. vLLM renames
  its engine child to ``VLLM::EngineCore``, which no longer contains ``vllm serve``. The
  engine keeps its weights and KV cache resident, the next boot loads a second copy on
  top of it, and the benchmark runs on whatever memory is left. SGLang renames its
  children the same way (``sglang::scheduler``, ``sglang::detokenizer``).
* The same pattern run from ``bash -c '... pkill -f "vllm serve" ...'`` matches the
  shell's own command line and kills the shell.

So processes are matched structurally (argv, not a substring of the whole command line),
the caller and its ancestors are never candidates, and the card counts as free only after
*consecutive* low readings with no server process alive. A crashing engine can read low
for a moment part-way through its own teardown.

Process discovery reads ``/proc`` and is Linux-only (the servers are too). GPU memory is read with
``nvidia-smi``; both are injectable for tests.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .gpu import nvidia_smi_path

__all__ = [
    "ClearResult",
    "Proc",
    "is_server_process",
    "list_processes",
    "naive_pkill_matches",
    "read_gpu_used_mib",
    "server_engine",
    "server_processes",
    "teardown",
    "usual_pkill_pattern",
    "wait_clear",
]

DEFAULT_MAX_USED_MIB = 800
# Absent on Windows, where this module only ever runs under test.
_SIGKILL: int = getattr(signal, "SIGKILL", 9)


@dataclass(frozen=True)
class Proc:
    pid: int
    ppid: int
    argv: tuple[str, ...]
    comm: str

    @property
    def cmdline(self) -> str:
        return " ".join(self.argv)


def _read_stat_ppid(stat: str) -> int:
    # /proc/<pid>/stat: "pid (comm) state ppid ..." where comm may contain spaces/parens.
    return int(stat[stat.rindex(")") + 2 :].split()[1])


def list_processes(proc_root: str | Path = "/proc") -> list[Proc]:
    root = Path(proc_root)
    out: list[Proc] = []
    for d in root.iterdir():
        if not d.name.isdigit():
            continue
        try:
            raw = (d / "cmdline").read_bytes()
            stat = (d / "stat").read_text(encoding="utf-8", errors="replace")
            comm = (d / "comm").read_text(encoding="utf-8", errors="replace").strip()
        except OSError:  # exited while we looked, or not ours to read
            continue
        argv = tuple(a.decode("utf-8", "replace") for a in raw.split(b"\0") if a)
        # A renamed process (setproctitle) may pad argv[0] with spaces.
        argv = tuple(a.strip() for a in argv if a.strip())
        try:
            ppid = _read_stat_ppid(stat)
        except (ValueError, IndexError):
            continue
        out.append(Proc(int(d.name), ppid, argv, comm))
    return out


# Per engine: the prefix its renamed children carry, its CLI, and its server modules.
# vLLM's engine renames itself VLLM::EngineCore; SGLang's scheduler and detokenizer
# become sglang::scheduler and sglang::detokenizer.
_RENAMED = {"VLLM::": "vllm", "sglang::": "sglang"}
# CLI name -> engine; each is followed by a "serve" subcommand.
_CLIS = {"vllm": "vllm", "sglang": "sglang", "trtllm-serve": "trtllm"}
_ENTRYPOINT_MODULES = {
    "vllm.entrypoints.openai.api_server": "vllm",
    "vllm.entrypoints.cli.main": "vllm",
    "sglang.launch_server": "sglang",
}


def server_engine(p: Proc) -> str | None:
    """Which engine's server process this is ("vllm", "sglang", "trtllm"), or None."""
    for prefix, engine in _RENAMED.items():
        if p.comm.startswith(prefix) or (p.argv and p.argv[0].startswith(prefix)):
            return engine
    argv = p.argv
    if not argv:
        return None
    # `vllm serve ...`, `sglang serve ...`, `trtllm-serve serve ...`, or the same run
    # as `python /path/to/bin/vllm serve ...`
    for i in (0, 1):
        if i + 1 >= len(argv) or argv[i + 1] != "serve":
            continue
        name = Path(argv[i]).name
        if name in _CLIS and (i == 0 or Path(argv[0]).name.startswith("python")):
            return _CLIS[name]
    # `python -m vllm.entrypoints...`, `python -m sglang.launch_server`
    if Path(argv[0]).name.startswith("python"):
        for i, a in enumerate(argv[:-1]):
            if a == "-m" and argv[i + 1] in _ENTRYPOINT_MODULES:
                return _ENTRYPOINT_MODULES[argv[i + 1]]
    return None


def is_server_process(p: Proc) -> bool:
    """True for a vLLM, SGLang or TensorRT-LLM server, including renamed children."""
    return server_engine(p) is not None


def usual_pkill_pattern(targets: Sequence[Proc]) -> str:
    """The ``pkill -f`` pattern people use for the server these processes belong to."""
    if any(server_engine(p) == "trtllm" for p in targets):
        return "trtllm-serve"
    if any(server_engine(p) == "sglang" for p in targets):
        launched_as_module = any("sglang.launch_server" in p.argv for p in targets)
        return "sglang.launch_server" if launched_as_module else "sglang serve"
    return "vllm serve"


def _ancestors(pid: int, procs: Sequence[Proc]) -> set[int]:
    parent = {p.pid: p.ppid for p in procs}
    seen = {pid}
    cur = pid
    while cur in parent and parent[cur] not in seen and parent[cur] > 0:
        cur = parent[cur]
        seen.add(cur)
    return seen


def server_processes(
    procs: Sequence[Proc], self_pid: int | None = None, engine: str | None = None
) -> list[Proc]:
    """Server processes and everything they started, never ``self_pid`` or its ancestors.

    Descendants count because some engines hold the GPU in a process with a generic
    name: TensorRT-LLM's model runs in ``python -m mpi4py.futures.server`` under ``prte``,
    both started by ``trtllm-serve``. With ``engine``, only that engine's servers.
    """
    me = os.getpid() if self_pid is None else self_pid
    protected = _ancestors(me, procs)

    def wanted(p: Proc) -> bool:
        found = server_engine(p)
        return found is not None and (engine is None or found == engine)

    found = {p.pid for p in procs if p.pid not in protected and wanted(p)}
    children: dict[int, list[int]] = {}
    for p in procs:
        children.setdefault(p.ppid, []).append(p.pid)
    todo = list(found)
    while todo:
        for child in children.get(todo.pop(), []):
            if child not in found and child not in protected:
                found.add(child)
                todo.append(child)
    return [p for p in procs if p.pid in found]


def naive_pkill_matches(procs: Iterable[Proc], pattern: str = "vllm serve") -> list[Proc]:
    """What ``pkill -f PATTERN`` would signal. Kept to show what it gets wrong."""
    rx = re.compile(pattern)
    return [p for p in procs if rx.search(p.cmdline)]


def read_gpu_used_mib(nvidia_smi: str | None = None) -> list[int]:
    exe = nvidia_smi or nvidia_smi_path()
    out = subprocess.run(
        [exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    return [int(x) for x in out.split()]


@dataclass
class ClearResult:
    clear: bool
    readings: list[tuple[float, int | None, int]] = field(
        default_factory=list[tuple[float, int | None, int]]
    )  # (elapsed_s, max used MiB or None if unreadable, live server processes)
    reason: str = ""


def wait_clear(
    *,
    read_used: Callable[[], list[int]] = read_gpu_used_mib,
    live_servers: Callable[[], int],
    max_used_mib: int = DEFAULT_MAX_USED_MIB,
    consecutive: int = 2,
    interval_s: float = 2.0,
    timeout_s: float = 120.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> ClearResult:
    """Poll until ``consecutive`` readings in a row are clear, or time runs out.

    A reading is clear only if no server process is alive *and* every GPU is at or below
    ``max_used_mib``. Any unclear reading resets the count.
    """
    if consecutive < 1:
        raise ValueError("consecutive must be >= 1")
    res = ClearResult(clear=False)
    t0 = clock()
    run = 0
    while True:
        n_live = live_servers()
        try:
            vals = read_used()
        except (OSError, subprocess.SubprocessError, ValueError):
            vals = []
        # No reading is not a clear reading.
        used = max(vals) if vals else None
        res.readings.append((round(clock() - t0, 3), used, n_live))
        if n_live == 0 and used is not None and used <= max_used_mib:
            run += 1
            if run >= consecutive:
                res.clear = True
                res.reason = f"{consecutive} consecutive clear readings"
                return res
        else:
            run = 0
        if clock() - t0 >= timeout_s:
            last = res.readings[-1]
            res.reason = (
                f"not clear after {timeout_s:g}s: last reading {last[1]} MiB used, "
                f"{last[2]} server processes alive"
            )
            return res
        sleep(interval_s)


def _signal(pids: Iterable[int], sig: int) -> list[int]:
    sent: list[int] = []
    for pid in pids:
        try:
            os.kill(pid, sig)
            sent.append(pid)
        except ProcessLookupError:
            pass
    return sent


def teardown(
    *,
    grace_s: float = 15.0,
    list_procs: Callable[[], list[Proc]] = list_processes,
    kill: Callable[[Iterable[int], int], list[int]] = _signal,
    read_used: Callable[[], list[int]] = read_gpu_used_mib,
    self_pid: int | None = None,
    max_used_mib: int = DEFAULT_MAX_USED_MIB,
    consecutive: int = 2,
    interval_s: float = 2.0,
    timeout_s: float = 120.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    engine: str | None = None,
) -> tuple[list[Proc], ClearResult]:
    """SIGTERM every server process, SIGKILL survivors after ``grace_s``, then wait for clear.

    With ``engine``, only that engine's servers are signalled; the card is still clear
    only when no server of any engine is left.
    """
    targets = server_processes(list_procs(), self_pid, engine)

    def ours() -> int:
        return len(server_processes(list_procs(), self_pid, engine))

    def live() -> int:
        return len(server_processes(list_procs(), self_pid))

    kill([p.pid for p in targets], signal.SIGTERM)
    deadline = clock() + grace_s
    while clock() < deadline and ours():
        sleep(0.5)
    survivors = server_processes(list_procs(), self_pid, engine)
    if survivors:
        kill([p.pid for p in survivors], _SIGKILL)
    result = wait_clear(
        read_used=read_used,
        live_servers=live,
        max_used_mib=max_used_mib,
        consecutive=consecutive,
        interval_s=interval_s,
        timeout_s=timeout_s,
        sleep=sleep,
        clock=clock,
    )
    return targets, result
