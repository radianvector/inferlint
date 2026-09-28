"""The tripwires. Each returns a :class:`CheckResult`; none raises on a finding.

IDs match the catalogue in ``docs/tripwires.md``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from .blocks import infer_blocks, predict_concurrency
from .bootlog import BootFacts
from .failures import Phase, classify
from .result import CheckResult, Status
from .series import Series
from .telemetry import ServerRestarted, Snapshot, counter_delta, span_s

__all__ = [
    "check_backend_honoured",
    "check_block_size",
    "check_boot_failure",
    "check_concurrency_reached",
    "check_kv_memory_stable",
    "check_no_preemption",
    "check_null_block",
    "check_same_pool",
    "concurrency_ceiling",
    "precise_rate",
]

PREEMPTIONS = "vllm:num_preemptions_total"
GENERATION_TOKENS = "vllm:generation_tokens_total"
CACHE_INFO = "vllm:cache_config_info"


def check_no_preemption(before: Snapshot, after: Snapshot) -> CheckResult:
    """T1. vLLM preempts silently: no log line at the default level, only a counter.

    A preempted request loses its computed KV and is recomputed from the start, so a run
    that assumed uninterrupted generation (a long-horizon test, a latency number) is void.
    """
    try:
        n = counter_delta(before, after, PREEMPTIONS)
    except ServerRestarted as e:
        return CheckResult("T1", Status.FAIL, str(e), {"restarted": True, **e.evidence})
    if n is None:
        return CheckResult(
            "T1",
            Status.UNKNOWN,
            f"{PREEMPTIONS} absent from a snapshot; preemption cannot be ruled out",
            {"series": PREEMPTIONS},
        )
    if n > 0:
        return CheckResult(
            "T1", Status.FAIL, f"{n:g} preemptions during the run", {"preemptions": n}
        )
    return CheckResult("T1", Status.PASS, "no preemptions", {"preemptions": 0})


def _config_key(f: BootFacts) -> str:
    return json.dumps(f.non_default_args, sort_keys=True, default=str)


def _groups(
    facts: Sequence[BootFacts], labels: Sequence[str] | None
) -> tuple[list[list[tuple[str, BootFacts]]], list[str]]:
    """Boots that came up, grouped by identical flags; plus the names left out."""
    names = list(labels) if labels is not None else [f"boot{i}" for i in range(len(facts))]
    by_cfg: dict[str, list[tuple[str, BootFacts]]] = {}
    skipped: list[str] = []
    for n, f in zip(names, facts, strict=True):
        if not f.ready or f.non_default_args is None:
            skipped.append(n)
            continue
        by_cfg.setdefault(_config_key(f), []).append((n, f))
    return [g for g in by_cfg.values() if len(g) > 1], skipped


def check_same_pool(
    facts: Sequence[BootFacts], labels: Sequence[str] | None = None
) -> list[CheckResult]:
    """T4. The KV pool is a property of the boot, not of the config.

    Boots with identical flags can draw different pools, and pools come in discrete
    levels, so a small difference in tokens can be a whole allocation step. Results from
    boots with different pools are not comparable without saying so. Boots are compared
    only with boots of the same flags; boots that never came up are left out.
    """
    groups, skipped = _groups(facts, labels)
    out: list[CheckResult] = []
    for g in groups:
        pools = {n: f.kv_pool_tokens for n, f in g}
        ev: dict[str, object] = {"pools": pools, "not_compared": skipped}
        missing = [n for n, p in pools.items() if p is None]
        if missing:
            out.append(
                CheckResult("T4", Status.UNKNOWN, f"no KV pool line in: {', '.join(missing)}", ev)
            )
            continue
        distinct = sorted({p for p in pools.values() if p is not None})
        label = ", ".join(pools)
        if len(distinct) == 1:
            out.append(
                CheckResult(
                    "T4",
                    Status.PASS,
                    f"same flags, same pool ({distinct[0]:,} tokens): {label}",
                    ev,
                )
            )
            continue
        lo, hi = distinct[0], distinct[-1]
        ev["levels"] = distinct
        out.append(
            CheckResult(
                "T4",
                Status.WARN,
                f"same flags, {len(distinct)} different pools: {lo:,} to {hi:,} tokens "
                f"({(hi - lo) / lo:.2%} apart) across {label}; "
                "compare results only within one pool",
                ev,
            )
        )
    return out


def check_block_size(facts: BootFacts, default: int = 16) -> CheckResult:
    """T5. Hybrid (attention + Mamba) models force the attention block size up.

    The block becomes the allocation unit: every request holds a whole number of blocks,
    so with a 784-token block a 100-token request costs as much cache as a 784-token one.
    """
    bs = facts.attention_block_size
    args = facts.non_default_args or {}
    if bs is None:
        if not facts.ready:
            return CheckResult("T5", Status.UNKNOWN, "boot did not get far enough to size blocks")
        return CheckResult(
            "T5",
            Status.PASS,
            "no block-size override printed",
            {"block_size": args.get("block_size", default)},
        )
    ev: dict[str, object] = {
        "block_size": bs,
        "requested": args.get("block_size"),
        "mamba_page_padding_pct": facts.mamba_page_padding_pct,
    }
    if facts.kv_pool_tokens:
        ev["pool_blocks"] = facts.kv_pool_tokens / bs
    if bs == args.get("block_size", default):
        return CheckResult("T5", Status.PASS, f"block size {bs}", ev)
    return CheckResult(
        "T5",
        Status.WARN,
        f"attention block size forced to {bs} tokens (requested "
        f"{args.get('block_size', default)}); each request's KV is allocated in {bs}-token units",
        ev,
    )


def check_kv_memory_stable(
    facts: Sequence[BootFacts], labels: Sequence[str] | None = None
) -> list[CheckResult]:
    """T11. The memory left for KV cache can differ between boots of one config.

    CUDA-graph capture memory varies boot to boot. A small loss can drop the pool by one
    whole level, so comparisons across boots need the per-boot figure recorded.
    """
    groups, skipped = _groups(facts, labels)
    out: list[CheckResult] = []
    for g in groups:
        avail = {n: f.available_kv_cache_gib for n, f in g}
        ev: dict[str, object] = {
            "available_kv_cache_gib": avail,
            "cudagraph_actual_gib": {n: f.cudagraph_actual_gib for n, f in g},
            "not_compared": skipped,
        }
        if any(v is None for v in avail.values()):
            out.append(
                CheckResult(
                    "T11", Status.UNKNOWN, "a boot log lacks 'Available KV cache memory'", ev
                )
            )
            continue
        vals = sorted({v for v in avail.values() if v is not None})
        if len(vals) == 1:
            out.append(
                CheckResult(
                    "T11", Status.PASS, f"same flags, {vals[0]} GiB for KV cache on every boot", ev
                )
            )
        else:
            out.append(
                CheckResult(
                    "T11",
                    Status.WARN,
                    f"same flags, KV cache memory varied: {vals[0]} to {vals[-1]} GiB",
                    ev,
                )
            )
    return out


def check_backend_honoured(facts: BootFacts) -> CheckResult:
    """T6. A requested attention backend is not applied everywhere, and nothing says so.

    ``--attention-backend`` sets the target model's backend. A speculative-decoding
    drafter chooses its own from a candidate list, so a backend known to fail on this
    machine can be in use while the boot log shows the requested one being honoured.
    """
    if facts.non_default_args is None:
        return CheckResult(
            "T6", Status.UNKNOWN, "no non-default args line; cannot tell what was requested"
        )
    ev: dict[str, object] = {
        "requested": facts.requested_backend,
        "target": facts.selected_backends,
        "drafter": facts.drafter_backends,
        "speculative": facts.speculative,
    }
    if facts.requested_backend is None:
        chosen = ", ".join(facts.selected_backends) or "one"
        msg = (
            "nothing to check: the server was started without --attention-backend, so vLLM "
            f"chose its own ({chosen}) and nothing could be ignored"
        )
        if facts.speculative is False:
            msg += "; there was no draft model either"
        return CheckResult("T6", Status.PASS, msg, ev)
    req = facts.requested_backend
    if not facts.selected_backends and not facts.drafter_backends:
        return CheckResult("T6", Status.UNKNOWN, "no backend selection line in the log", ev)
    off_target = sorted({b for b in facts.selected_backends if b != req})
    off_drafter = sorted({b for b in facts.drafter_backends if b != req})
    if off_target:
        return CheckResult(
            "T6",
            Status.FAIL,
            f"{req} was requested, but the main model used {', '.join(off_target)}",
            ev,
        )
    if off_drafter:
        return CheckResult(
            "T6",
            Status.FAIL,
            f"{req} was requested and the main model uses it, but the draft model "
            f"(speculative decoding) picked {', '.join(off_drafter)}",
            ev,
        )
    return CheckResult("T6", Status.PASS, f"{req} was requested and used", ev)


def check_boot_failure(log_text: str) -> CheckResult:
    """T7 + T12. Did the server die, when, and why.

    A config that boots and then dies on its first request passes any boot-only gate;
    ``phase == serving`` is that case.
    """
    f = classify(log_text)
    if f is None:
        return CheckResult("T12", Status.PASS, "no failure in log")
    tw = "T7" if f.phase is Phase.SERVING else "T12"
    return CheckResult(
        tw,
        Status.FAIL,
        f.summary(),
        {"kind": f.kind.value, "phase": f.phase.value, "lineno": f.lineno, "line": f.line},
    )


def check_concurrency_reached(
    series: Series, requested: int, facts: BootFacts | None = None
) -> CheckResult:
    """T8. Did the requested concurrency ever actually run at once?

    A client with 32 requests in flight measures a 32-user server only if the server
    ran 32 at once. The boot line ``Maximum concurrency for N tokens per request``
    assumes every request fills the full context, so it is no substitute for looking.
    """
    peak = series.peak_running()
    ev: dict[str, object] = {"requested": requested, "peak_running": peak}
    if facts is not None and facts.max_concurrency is not None:
        ev["boot_line_max_concurrency"] = facts.max_concurrency
        ev["boot_line_request_len"] = facts.max_concurrency_request_len
    if peak is None:
        return CheckResult("T8", Status.UNKNOWN, "series has no num_requests_running samples", ev)
    if peak < requested:
        return CheckResult(
            "T8",
            Status.FAIL,
            f"requested {requested} concurrent, server never ran more than {peak:g}",
            ev,
        )
    return CheckResult("T8", Status.PASS, f"reached {peak:g} running (requested {requested})", ev)


def concurrency_ceiling(
    series: Series, usable_blocks: int | None = None, *, saturated: float = 0.9
) -> CheckResult:
    """T9. How many requests fit, sample by sample: ``floor(1 / share)``.

    ``share = usage / running`` is the fraction of the cache one running request holds.
    It is read from every sample where the cache is the binding constraint (usage at or
    above ``saturated``), not from one sample, because it is not a constant: a request
    holds more blocks as its output grows, so the ceiling falls during a run, and each
    fall evicts requests (T1). A ceiling that moved is reported as a warning, since a
    single "concurrency" figure for such a run describes none of it.
    """
    busy = [s for s in series.samples if s.running and s.kv_usage]
    if not busy:
        return CheckResult("T9", Status.UNKNOWN, "no sample with requests running and KV in use")
    full = [s for s in busy if (s.kv_usage or 0.0) >= saturated]
    basis = "saturated" if full else "busy"
    full = full or busy
    ceilings: list[int] = []
    shares: list[float] = []
    for s in full:
        assert s.kv_usage is not None and s.running is not None
        ceilings.append(predict_concurrency(s.kv_usage, s.running, usable_blocks=usable_blocks))
        shares.append(s.kv_usage / s.running)
    fullest = max(busy, key=lambda s: s.kv_usage or 0.0)
    assert fullest.kv_usage is not None and fullest.running is not None
    at_fullest = predict_concurrency(fullest.kv_usage, fullest.running, usable_blocks=usable_blocks)
    lo, hi = min(ceilings), max(ceilings)
    ev: dict[str, object] = {
        "ceiling_min": lo,
        "ceiling_max": hi,
        "ceiling_at_fullest": at_fullest,
        "share_min": min(shares),
        "share_max": max(shares),
        "samples": len(full),
        "basis": basis,
        "usable_blocks": usable_blocks,
    }
    if usable_blocks:
        ev["blocks_per_request_min"] = min(shares) * usable_blocks
        ev["blocks_per_request_max"] = max(shares) * usable_blocks
    if lo == hi:
        held = (
            f"{shares[0]:.1%}"
            if f"{min(shares):.1%}" == f"{max(shares):.1%}"
            else f"{min(shares):.1%} to {max(shares):.1%}"
        )
        return CheckResult(
            "T9",
            Status.PASS,
            f"one request holds {held} of the cache; ceiling = floor(1/share) = {lo} "
            f"on every {basis} sample",
            ev,
        )
    return CheckResult(
        "T9",
        Status.WARN,
        f"ceiling moved between {lo} and {hi} as requests grew (one request held "
        f"{min(shares):.1%} to {max(shares):.1%} of the cache); concurrency was not a "
        "constant of this run",
        ev,
    )


def check_null_block(series: Series, snapshot: Snapshot) -> CheckResult:
    """T14. The usage gauge's denominator is ``num_gpu_blocks - 1``.

    Checked by inferring the denominator from the gauge alone and comparing it with the
    ``num_gpu_blocks`` label the server exports. Doubles as a self-test of
    :func:`infer_blocks`: if either side changes, this goes red.
    """
    info = snapshot.metrics.info(CACHE_INFO)
    reported_s = None if info is None else info.get("num_gpu_blocks")
    if reported_s is None or not reported_s.isdigit():
        return CheckResult("T14", Status.UNKNOWN, f"{CACHE_INFO} has no num_gpu_blocks label")
    reported = int(reported_s)
    inf = infer_blocks(series.kv_usages())
    ev: dict[str, object] = {
        "reported_num_gpu_blocks": reported,
        "inferred_usable_blocks": inf.blocks,
        "distinct_readings": inf.distinct,
    }
    if inf.blocks is None or not inf.trusted:
        return CheckResult("T14", Status.UNKNOWN, f"block count not inferable: {inf.reason}", ev)
    if inf.blocks == reported - 1:
        return CheckResult(
            "T14",
            Status.PASS,
            f"gauge denominator is {inf.blocks} = num_gpu_blocks ({reported}) - 1 null block",
            ev,
        )
    return CheckResult(
        "T14",
        Status.FAIL,
        f"gauge denominator is {inf.blocks}, expected num_gpu_blocks - 1 = {reported - 1}; "
        "the usage semantics changed, or the series is not from this boot",
        ev,
    )


def precise_rate(
    before: Snapshot, after: Snapshot, counter: str = GENERATION_TOKENS
) -> CheckResult:
    """T10. Throughput from the snapshots' own float clocks.

    Tokens counted between two snapshots divided by the time between those same two
    snapshots. A shell ``$(date +%s)`` difference lands within +-1 s of the truth, which
    on a 30-second run is several percent and does not cancel across the arms of a
    comparison.
    """
    try:
        tokens = counter_delta(before, after, counter)
        span, unc = span_s(before, after)
    except (ServerRestarted, ValueError) as e:
        return CheckResult("T10", Status.UNKNOWN, str(e))
    if tokens is None:
        return CheckResult("T10", Status.UNKNOWN, f"{counter} absent from a snapshot")
    if span <= 0:
        return CheckResult("T10", Status.UNKNOWN, f"non-positive span {span:.3f}s")
    int_span = int(after.t_wall) - int(before.t_wall)
    rate = tokens / span
    ev: dict[str, object] = {
        "tokens": tokens,
        "span_s": span,
        "span_uncertainty_s": unc,
        "rate_per_s": rate,
        "integer_second_span_s": int_span,
    }
    if int_span > 0:
        naive = tokens / int_span
        ev["integer_second_rate_per_s"] = naive
        ev["integer_second_error"] = naive / rate - 1
    return CheckResult("T10", Status.PASS, f"{rate:.2f} tokens/s over {span:.3f}s", ev)
