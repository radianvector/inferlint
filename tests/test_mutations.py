"""Every tripwire, watched failing.

Each row runs a check on real evidence, then on the same evidence with one number or
line changed, and requires the verdict to flip. This shows that each check can fail
as well as pass.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from conftest import FIXTURES, fixture_text
from inferlint import bootlog, checks, series, telemetry
from inferlint.result import CheckResult, Status

Edit = Callable[[str], str]


def ident(s: str) -> str:
    return s


def sub(pattern: str, repl: str, count: int = 0) -> Edit:
    def f(s: str) -> str:
        out, n = re.subn(pattern, repl, s, count=count)
        assert n, f"mutation {pattern!r} matched nothing: the fixture changed under the test"
        return out

    return f


def snap(text: str) -> telemetry.Snapshot:
    return telemetry.Snapshot("fixture", text, t_wall=0.0, t_mono_ns=None)


def t1(
    edit_after: Edit, before: str = "preempted_before.prom", after: str = "preempted_after.prom"
) -> CheckResult:
    a = fixture_text(f"metrics/{before}")
    b = edit_after(fixture_text(f"metrics/{after}"))
    return checks.check_no_preemption(snap(a), snap(b))


def t4(edit_lo: Edit) -> CheckResult:
    hi = bootlog.parse(fixture_text("boot_pool_level_hi.log"))
    lo = bootlog.parse(edit_lo(fixture_text("boot_pool_level_lo.log")))
    return checks.check_same_pool([hi, lo])[0]


def t11(edit_lo: Edit) -> CheckResult:
    hi = bootlog.parse(fixture_text("boot_pool_level_hi.log"))
    lo = bootlog.parse(edit_lo(fixture_text("boot_pool_level_lo.log")))
    return checks.check_kv_memory_stable([hi, lo])[0]


def log_check(
    name: str, fn: Callable[[bootlog.BootFacts], CheckResult]
) -> Callable[[Edit], CheckResult]:
    return lambda e: fn(bootlog.parse(e(fixture_text(name))))


def text_check(name: str, fn: Callable[[str], CheckResult]) -> Callable[[Edit], CheckResult]:
    return lambda e: fn(e(fixture_text(name)))


def series_check(
    stem: str, fn: Callable[[series.Series, telemetry.Snapshot], CheckResult]
) -> Callable[[Edit], CheckResult]:
    s = series.read(FIXTURES / "series" / f"{stem}.series.jsonl")
    return lambda e: fn(s, snap(e(fixture_text(f"series/{stem}_snapshot.prom"))))


def t8(requested: int) -> Callable[[Edit], CheckResult]:
    s = series.read(FIXTURES / "series" / "ladder.series.jsonl")
    return lambda e: checks.check_concurrency_reached(s, requested)


@dataclass(frozen=True)
class Mutant:
    id: str
    run: Callable[[Edit], CheckResult]
    edit: Edit
    baseline: Status
    mutated: Status


MUTANTS = [
    # T1: the 12-user run preempted 60 times; make the after-count equal the before-count
    Mutant(
        "T1-count",
        t1,
        sub(r"(vllm:num_preemptions_total\{[^}]*\}) 166\.0", r"\1 106.0"),
        Status.FAIL,
        Status.PASS,
    ),
    # T1: remove the counter entirely -> must be UNKNOWN, never PASS
    Mutant(
        "T1-absent",
        t1,
        sub(r"(?m)^vllm:num_preemptions_total.*\n", ""),
        Status.FAIL,
        Status.UNKNOWN,
    ),
    # T1: a restart that counters cannot see; give B the same _created stamps as A and the
    # check can no longer tell the processes apart (and reads a plausible delta instead)
    Mutant(
        "T1-created",
        lambda e: t1(e, "restart_before.prom", "restart_after.prom"),
        lambda s: re.sub(
            r"(?m)^(vllm:\w+_created\{[^}]*\}) .*$",
            lambda m: m.group(0).rsplit(" ", 1)[0] + " " + _CREATED_A.get(m.group(1), "0"),
            s,
        ),
        Status.FAIL,
        Status.FAIL,
    ),  # still FAIL (it did preempt) - see test below
    Mutant(
        "T4",
        t4,
        sub(r"GPU KV cache size: 12,405", "GPU KV cache size: 13,575"),
        Status.WARN,
        Status.PASS,
    ),
    Mutant(
        "T5",
        log_check("boot_eager.log", checks.check_block_size),
        sub(r"Setting attention block size to 784", "Setting attention block size to 16"),
        Status.WARN,
        Status.PASS,
    ),
    Mutant(
        "T6",
        log_check("boot_spec_backend_override.log", checks.check_backend_honoured),
        sub(r"Using FLASHINFER attention backend", "Using TRITON_ATTN attention backend"),
        Status.FAIL,
        Status.PASS,
    ),
    Mutant(
        "T7",
        text_check("boot_spec_backend_override.log", checks.check_boot_failure),
        lambda s: "\n".join(
            ln for ln in s.splitlines() if " ERROR " not in ln and "Traceback" not in ln
        ),
        Status.FAIL,
        Status.PASS,
    ),
    Mutant("T8", t8(32), ident, Status.FAIL, Status.FAIL),
    Mutant("T8-reached", t8(4), ident, Status.PASS, Status.PASS),
    Mutant(
        "T11",
        t11,
        sub(r"Available KV cache memory: 1\.44", "Available KV cache memory: 1.59"),
        Status.WARN,
        Status.PASS,
    ),
    Mutant(
        "T12",
        text_check("boot_fail_accelerator.log", checks.check_boot_failure),
        lambda s: "\n".join(
            ln for ln in s.splitlines() if "Error" not in ln and "Traceback" not in ln
        ),
        Status.FAIL,
        Status.PASS,
    ),
    Mutant(
        "T14",
        series_check("blocks", checks.check_null_block),
        sub(r'num_gpu_blocks="45"', 'num_gpu_blocks="44"'),
        Status.PASS,
        Status.FAIL,
    ),
    Mutant(
        "T14-absent",
        series_check("blocks", checks.check_null_block),
        sub(r'num_gpu_blocks="45",', ""),
        Status.PASS,
        Status.UNKNOWN,
    ),
]

_CREATED_A = {
    m.group(1): m.group(2)
    for m in re.finditer(
        r"(?m)^(vllm:\w+_created\{[^}]*\}) (\S+)$", fixture_text("metrics/restart_before.prom")
    )
}


@pytest.mark.parametrize("m", MUTANTS, ids=[m.id for m in MUTANTS])
def test_mutant(m: Mutant) -> None:
    base = m.run(ident)
    assert base.status is m.baseline, f"baseline: {base.message}"
    mutated = m.run(m.edit)
    assert mutated.status is m.mutated, f"mutant: {mutated.message}"
    if m.baseline is not m.mutated:
        return
    # Rows with equal statuses flip something else; their assertions live below.


def test_t1_created_mutation_hides_the_restart() -> None:
    real = t1(ident, "restart_before.prom", "restart_after.prom")
    assert real.evidence.get("restarted") is True
    m = next(x for x in MUTANTS if x.id == "T1-created")
    hidden = t1(m.edit, "restart_before.prom", "restart_after.prom")
    assert hidden.evidence.get("restarted") is None  # without _created, it looks like a run


def test_t12_unknown_exception_is_named_not_blank() -> None:
    text = fixture_text("boot_fail_accelerator.log").replace("AcceleratorError", "NovelError")
    r = checks.check_boot_failure(text)
    assert r.status is Status.FAIL
    assert r.evidence["kind"] in ("cuda_error", "unclassified_exception")
    assert r.message.strip()


def test_t8_boundary_is_exact() -> None:
    s = series.read(FIXTURES / "series" / "ladder.series.jsonl")
    assert checks.check_concurrency_reached(s, 4).status is Status.PASS
    assert checks.check_concurrency_reached(s, 5).status is Status.FAIL
