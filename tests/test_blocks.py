from __future__ import annotations

import random
from pathlib import Path

import pytest

from inferlint import series
from inferlint.blocks import infer_blocks, predict_concurrency


@pytest.mark.parametrize("n", [7, 44, 52, 53, 97, 400, 1021])
def test_infer_recovers_denominator(n: int) -> None:
    rng = random.Random(n)
    ks = rng.sample(range(1, n + 1), k=min(n, 12))
    inf = infer_blocks(k / n for k in ks)
    # a random sample of numerators can share a factor with n only by bad luck; seeded
    assert inf.blocks == n
    assert inf.trusted


def test_divisor_can_fit_by_luck_so_few_readings_are_untrusted() -> None:
    # all even multiples of 1/52 are also multiples of 1/26
    inf = infer_blocks([2 / 52, 4 / 52, 8 / 52])
    assert inf.blocks == 26
    assert not inf.trusted


def test_non_fraction_readings_are_indeterminate() -> None:
    inf = infer_blocks([0.1234567, 0.2345678, 0.3456789, 0.456789, 0.56789])
    assert inf.blocks is None


def test_zero_and_empty() -> None:
    assert infer_blocks([]).blocks is None
    assert infer_blocks([0.0, 0.0]).blocks is None


def test_fixture_series_denominator(fx: Path) -> None:
    s = series.read(fx / "series" / "blocks.series.jsonl")
    inf = infer_blocks(s.kv_usages())
    assert (inf.blocks, inf.trusted) == (44, True)


@pytest.mark.parametrize(
    ("usage", "running", "expected"),
    [(0.125, 1, 8), (0.308, 1, 3), (0.423, 1, 2), (0.538, 1, 1), (0.25, 2, 8), (0.164, 1, 6)],
)
def test_ceiling(usage: float, running: float, expected: int) -> None:
    assert predict_concurrency(usage, running) == expected


def test_ceiling_block_arithmetic_is_immune_to_float_error() -> None:
    # 1/8 of 48 blocks, read back a hair high: float floor says 7, blocks say 8.
    noisy = 6 / 48 + 1e-12
    assert int(1 / noisy) == 7
    assert predict_concurrency(noisy, 1, usable_blocks=48) == 8


def test_ceiling_rejects_idle_sample() -> None:
    with pytest.raises(ValueError):
        predict_concurrency(0.0, 0)
