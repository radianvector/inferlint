"""Block-level reading of the KV cache from the usage gauge alone.

vLLM reports its pool in tokens, allocates it in blocks, and exposes usage as a
fraction: ``kv_cache_usage_perc = used_blocks / usable_blocks``. Every reading is
therefore an exact multiple of ``1/usable_blocks``, and the least common denominator of
the readings recovers the block count without any access to the server.

Checked against ground truth, this exposes something the docs do not say: the
denominator is ``num_gpu_blocks - 1``, not ``num_gpu_blocks``. One block is reserved
as a null block and never holds a request's KV, so the capacity implied by
``cache_config_info`` overstates the usable pool by one block (several percent of
the pool when there are only a few dozen large blocks).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

__all__ = ["BlockInference", "infer_blocks", "predict_concurrency"]

# Minimum distinct non-zero readings before an inferred count is trusted. With fewer, a
# divisor of the true count fits by luck (all-even multiples of 1/52 also fit 1/26).
MIN_DISTINCT = 5


@dataclass(frozen=True)
class BlockInference:
    blocks: int | None  # usable blocks (the gauge's denominator), None if indeterminate
    distinct: int
    max_residual: float  # worst |usage * blocks - round(usage * blocks)|, in blocks
    trusted: bool

    @property
    def reason(self) -> str:
        if self.blocks is None:
            return "readings are not multiples of any 1/N with N within range"
        if not self.trusted:
            return f"only {self.distinct} distinct readings; a divisor of N could fit by luck"
        return "ok"


def infer_blocks(
    usages: Iterable[float], *, max_blocks: int = 1 << 16, tol: float = 1e-6
) -> BlockInference:
    """Least N such that every usage reading is k/N for integer k.

    Each reading is reduced to its best rational approximation with denominator at
    most ``max_blocks``; N is the least common multiple of those denominators. A reading
    that is not a clean fraction (an average across engines, say) produces a residual
    larger than ``tol`` and the result is reported as indeterminate, not rounded.
    """
    vals = sorted({v for v in usages if v and 0.0 < v <= 1.0})
    if not vals:
        return BlockInference(None, 0, 0.0, False)
    n = 1
    for v in vals:
        n = math.lcm(n, Fraction(v).limit_denominator(max_blocks).denominator)
        if n > max_blocks:
            return BlockInference(None, len(vals), float("inf"), False)
    worst = max(abs(v * n - round(v * n)) for v in vals)
    if worst > tol * n:
        return BlockInference(None, len(vals), worst, False)
    return BlockInference(n, len(vals), worst, len(vals) >= MIN_DISTINCT)


def predict_concurrency(usage: float, running: float, *, usable_blocks: int | None = None) -> int:
    """How many requests like the running ones fit: ``floor(1 / share)``.

    ``share`` is the fraction of the cache one request holds, read from a single
    sample as ``usage / running``. With the block count known, the arithmetic is done
    in whole blocks, so a reading of exactly 1/8 cannot floor to 7 through float error.
    """
    if running <= 0 or usage <= 0:
        raise ValueError("need a sample with requests running and non-zero usage")
    if usable_blocks is not None:
        used = round(usage * usable_blocks)
        if used == 0:
            raise ValueError("usage rounds to zero blocks")
        return (usable_blocks * int(running)) // used
    return math.floor(running / usage + 1e-9)
