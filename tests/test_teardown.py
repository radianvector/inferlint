from __future__ import annotations

from pathlib import Path

import pytest

from inferlint import teardown
from inferlint.teardown import Proc, is_server_process, naive_pkill_matches, server_processes

# pid, ppid, argv, comm. pid 400 is "us": a shell one-liner running the teardown.
TABLE: list[tuple[int, int, list[str], str]] = [
    (1, 0, ["/sbin/init"], "init"),
    (100, 1, ["bash", "run_campaign.sh"], "bash"),
    (200, 100, ["/opt/venv/bin/python", "/opt/venv/bin/vllm", "serve", "/models/m"], "vllm"),
    (201, 200, ["VLLM::EngineCore"], "VLLM::EngineCor"),
    (300, 1, ["python", "-m", "vllm.entrypoints.openai.api_server", "--model", "m"], "python"),
    (301, 300, ["VLLM::Worker_TP0  "], "VLLM::Worker_TP"),
    (400, 1, ["bash", "-lc", "pkill -f 'vllm serve'; inferlint teardown"], "bash"),
    (401, 400, ["python", "-m", "inferlint.cli", "teardown"], "python"),
    (500, 1, ["tail", "-f", "logs/vllm serve.log"], "tail"),
    (600, 1, ["vllm", "serve", "/models/other"], "vllm"),
]


def procs() -> list[Proc]:
    return [Proc(pid, ppid, tuple(argv), comm) for pid, ppid, argv, comm in TABLE]


def test_structural_match_finds_engines_and_spares_bystanders() -> None:
    found = {p.pid for p in server_processes(procs(), self_pid=401)}
    assert found == {200, 201, 300, 301, 600}


def test_naive_pkill_misses_engines_and_hits_the_shell() -> None:
    naive = {p.pid for p in naive_pkill_matches(procs())}
    assert 201 not in naive and 301 not in naive  # engines holding the GPU survive
    assert 400 in naive  # the shell running pkill kills itself
    assert 500 in naive  # and a bystander whose argv merely mentions the phrase


def test_never_targets_self_or_ancestors() -> None:
    # Even if our own argv looked like a server, we and our parents are protected.
    table = [*procs(), Proc(700, 600, ("vllm", "serve", "x"), "vllm")]
    found = {p.pid for p in server_processes(table, self_pid=700)}
    assert 700 not in found and 600 not in found


@pytest.mark.parametrize(
    ("argv", "comm", "expected"),
    [
        (["vllm", "serve", "m"], "vllm", True),
        (["/usr/bin/python3.12", "/venv/bin/vllm", "serve", "m"], "python3", True),
        (["python", "-m", "vllm.entrypoints.cli.main", "serve"], "python", True),
        (["VLLM::EngineCore"], "VLLM::EngineCor", True),
        ([], "VLLM::EngineCor", True),  # argv wiped, comm still tells
        (["vllm", "bench", "serve"], "vllm", False),
        (["bash", "-c", "vllm serve m"], "bash", False),  # the wrapper, not the server
        (["grep", "vllm serve"], "grep", False),
        (["/opt/tools/vllm", "serve-docs"], "vllm", False),
    ],
)
def test_is_server_process(argv: list[str], comm: str, expected: bool) -> None:
    assert is_server_process(Proc(1, 0, tuple(argv), comm)) is expected


def test_list_processes_reads_proc(tmp_path: Path) -> None:
    for pid, ppid, argv, comm in TABLE[:4]:
        d = tmp_path / str(pid)
        d.mkdir()
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
        (d / "stat").write_text(f"{pid} ({comm}) S {ppid} 1 1 0 -1\n")
        (d / "comm").write_text(comm + "\n")
    (tmp_path / "self").mkdir()
    (tmp_path / "999").mkdir()  # vanished mid-scan: no files
    got = {p.pid: p for p in teardown.list_processes(tmp_path)}
    assert set(got) == {1, 100, 200, 201}
    assert got[201].argv == ("VLLM::EngineCore",)
    assert got[200].ppid == 100


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def run_wait(readings: list[int | None], live: list[int], **kw: object) -> teardown.ClearResult:
    it_r, it_l = iter(readings), iter(live)
    clock = Clock()

    def read() -> list[int]:
        v = next(it_r)
        if v is None:
            raise OSError("nvidia-smi failed")
        return [v]

    return teardown.wait_clear(
        read_used=read,
        live_servers=lambda: next(it_l),
        sleep=clock.sleep,
        clock=clock,
        **kw,  # type: ignore[arg-type]
    )


def test_one_low_reading_is_not_proof() -> None:
    # A crashing engine reads low mid-teardown, then the survivor is seen again.
    r = run_wait([18900, 300, 18500, 250, 240], [1, 0, 1, 0, 0], interval_s=2.0)
    assert r.clear
    assert len(r.readings) == 5  # needed the 4th and 5th in a row


def test_low_memory_with_a_live_engine_is_not_clear() -> None:
    r = run_wait([100] * 10, [1] * 10, interval_s=2.0, timeout_s=10.0)
    assert not r.clear
    assert "1 server processes alive" in r.reason


def test_unreadable_gpu_is_not_clear() -> None:
    r = run_wait([None] * 10, [0] * 10, interval_s=2.0, timeout_s=10.0)
    assert not r.clear


def test_empty_gpu_list_is_not_clear() -> None:
    clock = Clock()
    r = teardown.wait_clear(
        read_used=list, live_servers=lambda: 0, sleep=clock.sleep, clock=clock, timeout_s=4.0
    )
    assert not r.clear


def test_teardown_sigterm_then_sigkill_survivors() -> None:
    table = procs()
    alive = {201}  # EngineCore ignores SIGTERM
    sent: list[tuple[int, int]] = []
    clock = Clock()

    def listing() -> list[Proc]:
        return [p for p in table if not is_server_process(p) or p.pid in alive]

    def kill(pids: object, sig: int) -> list[int]:
        pl = list(pids)  # type: ignore[call-overload]
        sent.extend((pid, sig) for pid in pl)
        if sig != 15:
            alive.difference_update(pl)
        return pl

    targets, res = teardown.teardown(
        list_procs=listing,
        kill=kill,
        read_used=lambda: [120],
        self_pid=401,
        sleep=clock.sleep,
        clock=clock,
        grace_s=3.0,
    )
    assert {p.pid for p in targets} == {201}
    assert sent[0] == (201, 15)
    assert sent[-1][0] == 201 and sent[-1][1] != 15
    assert res.clear
