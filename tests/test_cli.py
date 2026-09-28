from __future__ import annotations

import json
from pathlib import Path

import pytest

from inferlint.cli import main


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    code = main(list(argv))
    return code, capsys.readouterr().out


def test_boot_facts(fx: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run(capsys, "boot-facts", str(fx / "boot_eager.log"))
    assert code == 0
    assert "kv_pool_tokens               31554" in out


def test_boot_facts_json(fx: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run(capsys, "boot-facts", "--json", str(fx / "boot_eager.log"))
    assert code == 0
    assert json.loads(out)["attention_block_size"] == 784


def test_check_log_exit_codes(fx: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(capsys, "check-log", str(fx / "boot_fp8_triton.log"))[0] == 0
    code, out = run(capsys, "check-log", "--strict", str(fx / "boot_fp8_triton.log"))
    assert code == 1  # T5 warns: block size forced
    code, out = run(capsys, "check-log", str(fx / "boot_spec_backend_override.log"))
    assert code == 1 and "[ T6] FAIL" in out and "[ T7] FAIL" in out


def test_check_log_across_boots(fx: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run(
        capsys, "check-log", str(fx / "boot_pool_level_hi.log"), str(fx / "boot_pool_level_lo.log")
    )
    assert code == 0
    assert "[ T4] WARN" in out and "9.43% apart" in out


def test_preemption(fx: Path, capsys: pytest.CaptureFixture[str]) -> None:
    m = fx / "metrics"
    code, out = run(
        capsys, "preemption", str(m / "preempted_before.prom"), str(m / "preempted_after.prom")
    )
    assert code == 1 and "60 preemptions" in out


def test_rate(fx: Path, capsys: pytest.CaptureFixture[str]) -> None:
    m = fx / "metrics"
    code, out = run(
        capsys,
        "rate",
        str(m / "preempted_before.snapshot.json"),
        str(m / "preempted_after.snapshot.json"),
    )
    assert code == 0 and "integer-second timer would say" in out


def test_rate_without_clock_is_unknown(fx: Path, capsys: pytest.CaptureFixture[str]) -> None:
    m = fx / "metrics"
    code, _ = run(capsys, "rate", str(m / "preempted_before.prom"), str(m / "preempted_after.prom"))
    assert code == 2


def test_series(fx: Path, capsys: pytest.CaptureFixture[str]) -> None:
    s = fx / "series"
    code, out = run(
        capsys,
        "series",
        str(s / "ladder.series.jsonl"),
        "--snapshot",
        str(s / "ladder_snapshot.prom"),
        "--requested",
        "32",
    )
    assert code == 1
    assert "[T14] PASS" in out and "[ T9] PASS" in out and "[ T8] FAIL" in out
