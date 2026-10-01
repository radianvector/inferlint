"""The tripwires. Each returns a :class:`CheckResult`; none raises on a finding.

IDs match the catalogue in ``docs/tripwires.md``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from .benchresult import BenchResult
from .blocks import infer_blocks, predict_concurrency
from .bootlog import BootFacts
from .engines import Engine, from_names
from .failures import Phase, classify
from .metricnames import VLLM
from .result import CheckResult, Status
from .series import Series
from .telemetry import ServerRestarted, Snapshot, counter_delta, span_s

__all__ = [
    "check_backend_honoured",
    "check_block_size",
    "check_boot_failure",
    "check_client_server_agree",
    "check_concurrency_reached",
    "check_kv_memory_stable",
    "check_no_preemption",
    "check_null_block",
    "check_same_pool",
    "concurrency_ceiling",
    "precise_rate",
    "with_timer_comparison",
]

PREEMPTIONS = VLLM.preemptions
GENERATION_TOKENS = VLLM.generation_tokens
CACHE_INFO = VLLM.cache_info


def engine_of(*snapshots: Snapshot) -> Engine:
    """The engine that served these snapshots, from their metric names (vLLM if unknown)."""
    names: set[str] = set()
    for s in snapshots:
        names |= s.metrics.names()
    return from_names(names)


def _delta(before: Snapshot, after: Snapshot, engine: Engine, name: str) -> float | None:
    return counter_delta(before, after, name, witness=engine.witness(name))


def check_no_preemption(
    before: Snapshot, after: Snapshot, series: Series | None = None
) -> CheckResult:
    """T1. vLLM preempts silently: no log line at the default level, only a counter.

    A preempted request loses its computed KV and is recomputed from the start, so a run
    that assumed uninterrupted generation (a long-horizon test, a latency number) is void.
    SGLang calls it a retraction and logs a warning for each, but the cost is the same.
    TensorRT-LLM pauses a request for recompute and counts nothing; only the recording
    (``series``) can show that it happened.
    """
    engine = engine_of(before, after)
    word = engine.preemption
    if engine.metrics.preemptions is None:
        return _paused_in_recording(engine, series)
    name = engine.metrics.preemptions
    try:
        n = _delta(before, after, engine, name)
    except ServerRestarted as e:
        return CheckResult("T1", Status.FAIL, str(e), {"restarted": True, **e.evidence})
    ev: dict[str, object] = {"engine": engine.key, "series": name}
    if n is None:
        return CheckResult(
            "T1",
            Status.UNKNOWN,
            f"{name} absent from a snapshot; {word} cannot be ruled out",
            ev,
        )
    if n > 0:
        msg = f"{n:g} {word}s during the run"
        if engine.logs_preemptions:
            msg += f" ({engine.name} logs a warning for each)"
        return CheckResult("T1", Status.FAIL, msg, {**ev, "preemptions": n})
    return CheckResult("T1", Status.PASS, f"no {word}s", {**ev, "preemptions": 0})


def _paused_in_recording(engine: Engine, series: Series | None) -> CheckResult:
    """T1 for an engine that exports only a gauge of paused requests, from the recording.

    A pause that starts and ends between two readings is not seen, so a recording can
    show that pauses happened, and a lower bound on them, but never that none did.
    """
    gauge = engine.metrics.paused
    ev: dict[str, object] = {"engine": engine.key, "series": gauge}
    seen = [s.paused for s in series.samples if s.paused is not None] if series else []
    if not seen:
        return CheckResult(
            "T1",
            Status.UNKNOWN,
            f"{engine.name} exports no {engine.preemption} count, and there is no recording "
            "of its paused requests",
            ev,
        )
    busy = [p for p in seen if p > 0]
    ev |= {"readings": len(seen), "readings_with_paused": len(busy), "paused_max": max(seen)}
    if busy:
        return CheckResult(
            "T1",
            Status.FAIL,
            f"requests were paused for recompute in {len(busy)} of {len(seen)} readings, up "
            f"to {max(busy):g} at once ({engine.name} counts no pauses, so the number is "
            "unknown)",
            ev,
        )
    return CheckResult(
        "T1",
        Status.UNKNOWN,
        f"no paused request in {len(seen)} readings; {engine.name} counts no pauses, so one "
        "between readings cannot be ruled out",
        ev,
    )


def _config_key(f: BootFacts) -> str:
    return json.dumps([f.engine, f.config_args], sort_keys=True, default=str)


def _groups(
    facts: Sequence[BootFacts], labels: Sequence[str] | None
) -> tuple[list[list[tuple[str, BootFacts]]], list[str]]:
    """Boots that came up, grouped by identical flags; plus the names left out."""
    names = list(labels) if labels is not None else [f"boot{i}" for i in range(len(facts))]
    by_cfg: dict[str, list[tuple[str, BootFacts]]] = {}
    skipped: list[str] = []
    for n, f in zip(names, facts, strict=True):
        if not f.ready or f.config_args is None:
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
    if bs is None and facts.page_size is not None:
        return CheckResult(
            "T5",
            Status.PASS,
            f"page size {facts.page_size} token{'s' if facts.page_size != 1 else ''}, as set",
            {"page_size": facts.page_size},
        )
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
    if facts.config_args is None:
        return CheckResult(
            "T6", Status.UNKNOWN, "no arguments line in the log; cannot tell what was requested"
        )
    server = {"sglang": "SGLang"}.get(facts.engine or "", "vLLM")
    ev: dict[str, object] = {
        "requested": facts.requested_backend,
        "target": facts.selected_backends,
        "drafter": facts.drafter_backends,
        "speculative": facts.speculative,
    }
    if facts.requested_backend is None:
        chosen = ", ".join(facts.selected_backends) or "one"
        msg = (
            "nothing to check: the server was started without --attention-backend, so "
            f"{server} chose its own ({chosen}) and nothing could be ignored"
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

    A client with 32 requests in flight measures a server running 32 at once only if the
    server ran 32 at once. The boot line ``Maximum concurrency for N tokens per request``
    assumes every request fills the full context, so it is no substitute for looking.

    vLLM 0.29 added admission control: with ``--max-num-queued-reqs N`` at most N
    requests are in flight (running plus waiting), and the rest are rejected with HTTP 503
    instead of queued. When the boot log sets that limit and the recording reached it,
    the message says the requests were rejected, not that the cache was full.
    """
    peak = series.peak_running()
    in_flight = [
        (s.running or 0.0) + (s.waiting or 0.0)
        for s in series.samples
        if s.running is not None and s.waiting is not None
    ]
    peak_in_flight = max(in_flight) if in_flight else None
    args = (facts.non_default_args if facts is not None else None) or {}
    cap = args.get("max_num_queued_reqs")
    ev: dict[str, object] = {
        "requested": requested,
        "peak_running": peak,
        "peak_in_flight": peak_in_flight,
    }
    if facts is not None and facts.max_concurrency is not None:
        ev["boot_line_max_concurrency"] = facts.max_concurrency
        ev["boot_line_request_len"] = facts.max_concurrency_request_len
    if isinstance(cap, int):
        ev["max_num_queued_reqs"] = cap
    if args.get("max_num_queued_tokens") is not None:
        ev["max_num_queued_tokens"] = args["max_num_queued_tokens"]
    if peak is None:
        return CheckResult("T8", Status.UNKNOWN, "series has no num_requests_running samples", ev)
    if peak >= requested:
        return CheckResult(
            "T8", Status.PASS, f"reached {peak:g} running (requested {requested})", ev
        )
    msg = f"requested {requested} concurrent, server never ran more than {peak:g}"
    limit = cap if isinstance(cap, int) and cap < requested else None
    if limit is not None and peak_in_flight is not None and peak_in_flight >= limit:
        msg += (
            f"; admission control (--max-num-queued-reqs {limit}) kept at most {limit} in "
            f"flight, so the other {requested - limit} were rejected, not queued"
        )
    run_cap = _running_cap(facts)
    if run_cap is not None:
        n, why = run_cap
        ev["running_cap"] = n
        if peak >= n and n < requested:  # the cap was reached, so it was the limit
            msg += f"; {why}"
    return CheckResult("T8", Status.FAIL, msg, ev)


def _running_cap(facts: BootFacts | None) -> tuple[int, str] | None:
    """The most requests the server runs at once by its own setting, and how it was set."""
    if facts is None:
        return None
    if facts.running_cap is not None:  # SGLang prints the limit it settled on
        n, asked = facts.running_cap, facts.requested_running
        if asked is not None and n < asked:
            because = (
                f" because of the {facts.running_cap_reason}" if facts.running_cap_reason else ""
            )
            return n, (
                f"SGLang lowered its limit to {n} running{because}, although it was started "
                f"with --max-running-requests {asked}"
            )
        return n, f"the server runs at most {n} at once (--max-running-requests {n})"
    seqs = (facts.non_default_args or {}).get("max_num_seqs")
    if isinstance(seqs, int):
        return seqs, f"the server runs at most {seqs} at once (--max-num-seqs {seqs})"
    return None


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
    # vLLM's gauge is the KV cache; SGLang's is whichever of its pools is fullest, which
    # for a hybrid model can be the per-request state slots rather than the KV cache.
    pool = (
        "the fullest memory pool (KV cache or request state)"
        if series.engine.key == "sglang"
        else "the cache"
    )
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
            f"one request holds {held} of {pool}; ceiling = floor(1/share) = {lo} "
            f"on every {basis} sample",
            ev,
        )
    return CheckResult(
        "T9",
        Status.WARN,
        f"ceiling moved between {lo} and {hi} as requests grew (one request held "
        f"{min(shares):.1%} to {max(shares):.1%} of {pool}); concurrency was not a "
        "constant of this run",
        ev,
    )


def check_null_block(series: Series, snapshot: Snapshot) -> CheckResult:
    """T14. The usage gauge's denominator is ``num_gpu_blocks - 1``.

    Checked by inferring the denominator from the gauge alone and comparing it with the
    ``num_gpu_blocks`` label the server exports. Doubles as a self-test of
    :func:`infer_blocks`: if either side changes, this goes red.
    """
    engine = engine_of(snapshot)
    cache_info = engine.metrics.cache_info
    if cache_info is None:
        return CheckResult(
            "T14",
            Status.UNKNOWN,
            f"{engine.name} exports no block count to check the usage gauge against",
            {"engine": engine.key},
        )
    info = snapshot.metrics.info(cache_info)
    reported_s = None if info is None else info.get("num_gpu_blocks")
    if reported_s is None or not reported_s.isdigit():
        return CheckResult("T14", Status.UNKNOWN, f"{cache_info} has no num_gpu_blocks label")
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


def check_client_server_agree(
    bench: BenchResult,
    before: Snapshot,
    after: Snapshot,
    *,
    expected_extra_requests: int = 0,
) -> CheckResult:
    """T15. Did the server do the work the load tool says it sent, and no other work?

    The load tool's result file counts the requests it completed and the output tokens it
    received. The server's counters between the before and after snapshots count what the
    server finished. A load tool can send requests it leaves out of its result (vllm
    bench serve's initial test request and warm-ups, which its output reports); those are
    ``expected_extra_requests``. Any other difference means other traffic shared the
    server, or the client counted work the server did not do.
    """
    names = engine_of(before, after).metrics
    try:
        done = counter_delta(before, after, names.request_success)
        tokens = counter_delta(before, after, names.generation_tokens)
    except ServerRestarted as e:
        return CheckResult("T15", Status.UNKNOWN, str(e), {"restarted": True, **e.evidence})
    ev: dict[str, object] = {
        "client_completed": bench.completed,
        "client_failed": bench.failed,
        "client_output_tokens": bench.output_tokens,
        "server_finished": done,
        "server_output_tokens": tokens,
        "expected_extra_requests": expected_extra_requests,
        "result_file": bench.path,
    }
    if done is None or tokens is None:
        return CheckResult("T15", Status.UNKNOWN, "a server counter is missing from a snapshot", ev)
    if bench.completed is None or bench.output_tokens is None:
        return CheckResult(
            "T15", Status.UNKNOWN, "the result file has no completed or output-token count", ev
        )
    extra = int(done) - bench.completed
    ev["extra_requests"] = extra
    if extra > expected_extra_requests:
        other = extra - expected_extra_requests
        return CheckResult(
            "T15",
            Status.FAIL,
            f"the server finished {other} more request{'s' if other != 1 else ''} than the "
            "load tool sent; other traffic shared the server during the run",
            ev,
        )
    if extra < expected_extra_requests:
        missing = expected_extra_requests - extra
        return CheckResult(
            "T15",
            Status.FAIL,
            f"the load tool counted {missing} more completed request"
            f"{'s' if missing != 1 else ''} than the server finished",
            ev,
        )
    per = bench.output_tokens / bench.completed if bench.completed else 0.0
    extra_tokens = tokens - bench.output_tokens
    ev["extra_output_tokens"] = extra_tokens
    # The extra requests produce output too; a mismatch beyond what they can explain means
    # the client and the server counted different tokens.
    if not 0 <= extra_tokens <= expected_extra_requests * max(per, 1.0) * 1.5:
        return CheckResult(
            "T15",
            Status.WARN,
            f"requests agree, but the server generated {tokens:,.0f} output tokens and the "
            f"load tool counted {bench.output_tokens:,}",
            ev,
        )
    msg = (
        f"client and server agree: {bench.completed} requests, "
        f"{bench.output_tokens:,} output tokens"
    )
    if extra:
        msg += f", plus {extra} untimed request{'s' if extra != 1 else ''} (test or warm-up)"
    return CheckResult("T15", Status.PASS, msg, ev)


def with_timer_comparison(r: CheckResult) -> CheckResult:
    """T10's message with what an integer-second timer would have said, for the terminal."""
    if "integer_second_error" not in r.evidence:
        return r
    return CheckResult(
        r.tripwire,
        r.status,
        f"{r.message}; an integer-second timer would say "
        f"{r.evidence['integer_second_rate_per_s']:.2f} "
        f"({r.evidence['integer_second_error']:+.2%})",
        r.evidence,
    )


def precise_rate(before: Snapshot, after: Snapshot, counter: str | None = None) -> CheckResult:
    """T10. Throughput from the snapshots' own float clocks.

    Tokens counted between two snapshots divided by the time between those same two
    snapshots. A shell ``$(date +%s)`` difference lands within +-1 s of the truth, which
    on a 30-second run is several percent and does not cancel across the arms of a
    comparison. ``counter`` defaults to the engine's generated-token counter.
    """
    if counter is None:
        counter = engine_of(before, after).metrics.generation_tokens
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
