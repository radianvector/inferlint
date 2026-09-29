"""GPU facts from nvidia-smi, and the gpu-inspect command around the free-card check."""

from __future__ import annotations

import json
import subprocess

import pytest

from inferlint import cli, gpu, teardown
from inferlint.cli import main
from inferlint.gpu import GpuInfo

# Read from the RTX 4090 the fixtures were recorded on (idle, Windows driver 616.56).
RTX4090 = "0, NVIDIA GeForce RTX 4090, 24564, 0, 24138, 616.56, 8.9\n"


def test_parse_one_gpu() -> None:
    (g,) = gpu.parse(RTX4090)
    assert g == GpuInfo(0, "NVIDIA GeForce RTX 4090", 24564, 0, 24138, "616.56", "8.9")
    assert g.family == "Ada Lovelace"
    assert g.to_json()["family"] == "Ada Lovelace"


@pytest.mark.parametrize(
    ("cap", "expected"),
    [
        ("6.1", "Pascal"),
        ("7.0", "Volta"),
        ("7.5", "Turing"),
        ("8.0", "Ampere"),
        ("8.6", "Ampere"),
        ("8.9", "Ada Lovelace"),
        ("9.0", "Hopper"),
        ("10.0", "Blackwell"),
        ("12.0", "Blackwell"),
        ("5.2", None),
        ("", None),
        (None, None),
        ("n/a", None),
    ],
)
def test_family(cap: str | None, expected: str | None) -> None:
    assert gpu.family(cap) == expected


def test_several_gpus_and_missing_values() -> None:
    # A made-up second card whose memory nvidia-smi cannot report.
    text = RTX4090 + "1, NVIDIA A100-SXM4-40GB, [N/A], [N/A], [N/A], 550.54, 8.0\n"
    a, b = gpu.parse(text)
    assert (a.index, b.index) == (0, 1)
    assert b.memory_total_mib is None and b.memory_free_mib is None
    assert b.family == "Ampere"


def test_a_comma_in_the_name_stays_in_the_name() -> None:
    # Made up: no current NVIDIA name has a comma, but nothing stops one.
    (g,) = gpu.parse("0, NVIDIA Example, Rev 2, 8192, 0, 8192, 550.54, 8.6\n")
    assert g.name == "NVIDIA Example, Rev 2"
    assert (g.memory_total_mib, g.compute_cap) == (8192, "8.6")


def test_an_old_driver_without_compute_cap_is_asked_again() -> None:
    calls: list[str] = []

    def query(args: list[str]) -> str:
        calls.append(args[0])
        if "compute_cap" in args[0]:
            raise subprocess.CalledProcessError(2, "nvidia-smi")
        return "0, NVIDIA GeForce RTX 4090, 24564, 0, 24138, 470.82\n"  # made-up old driver

    (g,) = gpu.read_gpus(query)
    assert len(calls) == 2
    assert g.compute_cap is None and g.family is None and g.driver == "470.82"


@pytest.fixture
def one_idle_4090(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "read_gpus", lambda: gpu.parse(RTX4090))
    monkeypatch.setattr(teardown, "list_processes", list)
    monkeypatch.setattr(teardown, "read_gpu_used_mib", lambda: [0])


@pytest.mark.usefixtures("one_idle_4090")
def test_gpu_inspect_prints_the_card_and_passes_when_free(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_procs_readable", lambda: True)
    assert main(["gpu-inspect", "--consecutive", "1"]) == 0
    out = capsys.readouterr().out
    assert "GPU 0   NVIDIA GeForce RTX 4090" in out
    assert "Ada Lovelace, compute capability 8.9, driver 616.56" in out
    assert "memory: 24,564 MiB total, 0 MiB used, 24,138 MiB free" in out
    assert "[ T2] PASS" in out


@pytest.mark.usefixtures("one_idle_4090")
def test_gpu_inspect_fails_when_a_server_holds_the_card(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_procs_readable", lambda: True)
    # 21,768 MiB is what the killed server's engine held in the recorded run (T2).
    monkeypatch.setattr(teardown, "read_gpu_used_mib", lambda: [21768])
    code = main(["gpu-inspect", "--consecutive", "1", "--timeout", "0"])
    assert code == 1
    assert "card not clear; refusing to boot" in capsys.readouterr().out


@pytest.mark.usefixtures("one_idle_4090")
def test_gpu_inspect_json_carries_the_facts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_procs_readable", lambda: True)
    assert main(["gpu-inspect", "--json", "--consecutive", "1"]) == 0
    (r,) = json.loads(capsys.readouterr().out)
    assert r["status"] == "pass"
    assert r["evidence"]["gpus"][0]["family"] == "Ada Lovelace"


@pytest.mark.usefixtures("one_idle_4090")
def test_without_linux_it_shows_the_card_and_cannot_tell(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_procs_readable", lambda: False)
    assert main(["gpu-inspect"]) == 2
    out = capsys.readouterr().out
    assert "NVIDIA GeForce RTX 4090" in out
    assert "needs Linux" in out
    assert main(["teardown"]) == 2


def test_no_nvidia_smi_is_said_plainly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def missing() -> list[GpuInfo]:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gpu, "read_gpus", missing)
    monkeypatch.setattr(cli, "_procs_readable", lambda: False)
    assert main(["gpu-inspect"]) == 2
    assert "no NVIDIA GPU found" in capsys.readouterr().out
