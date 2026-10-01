"""What limited a run: up to three plain sentences drawn from its files.

The tripwires say whether a run's numbers can be trusted. These say what decided them:

* **cache**: could the KV cache hold the load? The load at full length needs
  ``concurrency x (prompt + output tokens per request)``, rounded up to whole blocks; the
  boot log says how many tokens the cache holds.
* **waiting**: did requests wait for others to finish before they started?
* **concurrency** (when the cache sentence does not apply, or the cache had room): did the
  server run as many requests at once as the load kept open, and if not, its own limit?
* **pace**: while a request ran it got a token every *TPOT* ms, so *N* running at once
  could produce ``N x 1000 / TPOT`` tokens per second. The share of that the run reached
  is what queueing, preemption and prompt processing left; the recording says how many
  requests ran at once on average.

Each is arithmetic on numbers the report already shows. A sentence is left out when its
numbers are missing, and for hybrid models, whose memory per request the token count does
not describe, the cache sentence is left out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Literal

from .checks import running_cap
from .result import Status

if TYPE_CHECKING:
    from .report import Report

__all__ = ["Busy", "CacheNeed", "Insight", "busy", "cache_need", "limits", "per_request_tokens"]

Tone = Literal["limit", "ok"]


@dataclass(frozen=True)
class Insight:
    key: str  # "cache", "waiting", "pace"
    title: str
    text: str
    tone: Tone  # "limit": this held the run back; "ok": it did not


def limits(rep: Report) -> list[Insight]:
    """What limited the run, most decisive first; empty when the files do not say."""
    if rep.summary.restarted:
        return []
    first = _cache(rep) or _concurrency_short(rep)
    out = [i for i in (first, _waiting(rep), _pace(rep)) if i is not None]
    return out[:3]


# --------------------------------------------------------------------------- numbers


def per_request_tokens(rep: Report) -> float | None:
    """Prompt plus output tokens per finished request: the server's count, else the client's."""
    sm = rep.summary
    if sm.done and sm.prompt_tokens is not None and sm.output_tokens is not None:
        return (sm.prompt_tokens + sm.output_tokens) / sm.done
    b = rep.bench
    if b and b.completed and b.input_tokens is not None and b.output_tokens is not None:
        return (b.input_tokens + b.output_tokens) / b.completed
    return None


def _block(rep: Report) -> int | None:
    f = rep.boots[0][1] if rep.boots else None
    if f is None:
        return None
    if f.page_size:
        return f.page_size
    if f.attention_block_size:
        return f.attention_block_size
    return 16 if rep.engine.key == "vllm" else None


def _hybrid(rep: Report) -> bool:
    """A model whose state per request is not KV per token (attention plus Mamba/linear)."""
    f = rep.boots[0][1] if rep.boots else None
    if f is None:
        return False
    return bool(f.running_cap_reason) or (f.attention_block_size or 0) > 64


def _concurrency(rep: Report) -> int | None:
    n = rep.requested
    done = rep.summary.done or (rep.bench.completed if rep.bench else None)
    if n and done:
        return min(n, done)
    return n


def _tpot_ms(rep: Report) -> float | None:
    """Milliseconds per output token while a request runs: the client's, else the server's."""
    if rep.bench and rep.bench.mean_tpot_ms:
        return rep.bench.mean_tpot_ms
    sm = rep.summary
    if sm.mean_latency_s is None or sm.mean_first_token_s is None or not sm.done:
        return None
    if not sm.output_tokens or sm.output_tokens / sm.done <= 1:
        return None
    decode_s = sm.mean_latency_s - sm.mean_first_token_s
    return 1000 * decode_s / (sm.output_tokens / sm.done - 1) if decode_s > 0 else None


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}" if n == 1 else f"{n:,} {word}s"


@dataclass(frozen=True)
class Busy:
    seconds: float  # time with at least one request running
    mean_running: float  # requests running on average over that time
    tokens_per_s: float | None  # output tokens per second over that time


def busy(rep: Report) -> Busy | None:
    """The recording's busy time, read between samples; None for engines whose gauges lag."""
    s = rep.series
    if s is None or len(s.samples) < 2 or rep.engine.gauges_lag:
        return None
    exact = all(x.t_mono_ns is not None for x in s.samples)
    span = weighted = tokens = 0.0
    counted = False
    for a, b in pairwise(s.samples):
        if not a.running or a.running <= 0:
            continue
        if exact and a.t_mono_ns is not None and b.t_mono_ns is not None:
            dt = (b.t_mono_ns - a.t_mono_ns) / 1e9
        else:
            dt = b.t_wall - a.t_wall
        if dt <= 0:
            continue
        span += dt
        weighted += a.running * dt
        if a.generation_tokens is not None and b.generation_tokens is not None:
            tokens += max(b.generation_tokens - a.generation_tokens, 0.0)
            counted = True
    if span <= 0:
        return None
    return Busy(span, weighted / span, tokens / span if counted else None)


# --------------------------------------------------------------------------- the sentences


def _peak(rep: Report) -> float | None:
    """The most requests that ran at once, when T8 found it short of the load."""
    t8 = rep.result("T8")
    if t8 is None or t8.status is not Status.FAIL:
        return None
    peak = t8.evidence.get("peak_running")
    return float(peak) if isinstance(peak, int | float) else None


def _preempted(rep: Report) -> str | None:
    t1 = rep.result("T1")
    n = t1.evidence.get("preemptions") if t1 is not None else None
    if not isinstance(n, int | float) or n <= 0:
        return None
    verb = {"retraction": "retracted", "pause": "paused"}.get(rep.engine.preemption, "preempted")
    return f"{verb} {_plural(int(n), 'running request')}"


@dataclass(frozen=True)
class CacheNeed:
    pool: int  # tokens the KV cache holds
    per_request: float  # prompt plus output tokens per request
    allocated: float  # the same, in whole blocks
    block: int | None
    concurrency: int

    @property
    def total(self) -> float:
        return self.concurrency * self.allocated


def cache_need(rep: Report) -> CacheNeed | None:
    """The cache against the load at full length; None when a number is missing, or for a
    hybrid model, whose memory per request a token count does not describe."""
    f = rep.boots[0][1] if rep.boots else None
    pool = f.kv_pool_tokens if f is not None else None
    per = per_request_tokens(rep)
    n = _concurrency(rep)
    if not pool or not per or not n or _hybrid(rep):
        return None
    block = _block(rep)
    alloc = math.ceil(per / block) * block if block else per
    return CacheNeed(pool, per, alloc, block, n)


def _cache(rep: Report) -> Insight | None:
    c = cache_need(rep)
    if c is None:
        return None
    pool, per, n, alloc, need = c.pool, c.per_request, c.concurrency, c.allocated, c.total
    about = "about " if per != int(per) else ""
    load = f"{n} requests of {about}{per:,.0f} tokens"
    if need > pool:
        did: list[str] = []
        peak = _peak(rep)
        if peak is not None:
            did.append(f"ran at most {peak:g} at once and queued the rest")
        pre = _preempted(rep)
        if pre:
            did.append(pre)
        then = f" To make room, it {', and '.join(did)}." if did else ""
        return Insight(
            "cache",
            "The KV cache was too small for the load",
            f"The cache holds {pool:,} tokens; {load} need {need:,.0f} to run to the end "
            f"together, so {int(pool // alloc)} fit at full length.{then}",
            "limit",
        )
    short = _concurrency_short(rep)
    if short is not None:
        return Insight(
            "cache",
            "The cache had room; something else held requests back",
            f"The cache holds {pool:,} tokens, enough for {load} ({need:,.0f}). {short.text}",
            "limit",
        )
    return Insight(
        "cache",
        "The KV cache held the whole load",
        f"The cache holds {pool:,} tokens; {load} need {need:,.0f}, {need / pool:.0%} of it.",
        "ok",
    )


def _concurrency_short(rep: Report) -> Insight | None:
    """The server ran fewer at once than the load kept open; why, when it said."""
    peak = _peak(rep)
    if peak is None:
        return None
    why = ""
    f = rep.boots[0][1] if rep.boots else None
    cap = running_cap(f)
    if cap is not None and peak >= cap[0]:
        why = f": {cap[1]}"
    asked = f"kept {rep.requested} requests open" if rep.requested else "sent more"
    pre = _preempted(rep)
    also = f" It also {pre}." if pre else ""
    return Insight(
        "concurrency",
        "The server ran fewer requests at once than the load sent",
        f"The load {asked}, but the server ran at most {peak:g} at once{why}.{also}",
        "limit",
    )


def _ms(v: float) -> str:
    return f"{v / 1000:,.1f} s" if v >= 1000 else f"{v:,.0f} ms"


def _waiting(rep: Report) -> Insight | None:
    b = rep.bench
    queued = _peak(rep) is not None
    if b and b.median_ttft_ms and b.p99_ttft_ms:
        med, p99 = b.median_ttft_ms, b.p99_ttft_ms
        if p99 > max(3 * med, med + 1000):
            why = ": they were queued until running requests ended" if queued else ""
            return Insight(
                "waiting",
                "Some requests waited much longer to start",
                f"Half the requests got their first token within {_ms(med)}, but the slowest "
                f"(p99) waited {_ms(p99)}{why}.",
                "limit",
            )
        if med >= 2000:
            why = ", most of it in the queue" if queued else ""
            return Insight(
                "waiting",
                "Requests waited long to start",
                f"Half the requests waited more than {_ms(med)} for their first token, the "
                f"slowest (p99) {_ms(p99)}{why}.",
                "limit",
            )
        return Insight(
            "waiting",
            "No request waited long to start",
            f"Time to first token: {_ms(med)} for half the requests, {_ms(p99)} at p99.",
            "ok",
        )
    q = rep.summary.mean_queue_s
    if q is not None and q >= 1.0:
        return Insight(
            "waiting",
            "Requests waited in the queue",
            f"A request waited {q:,.1f} s on average before the server started it.",
            "limit",
        )
    return None


def _pace(rep: Report) -> Insight | None:
    tpot, n = _tpot_ms(rep), _concurrency(rep)
    bz = busy(rep)
    if rep.bench and rep.bench.output_throughput:
        actual, how = rep.bench.output_throughput, "The run averaged"
    elif bz is not None and bz.tokens_per_s:
        actual, how = bz.tokens_per_s, "While busy, the server averaged"
    else:
        return None
    if not tpot or not n:
        return None
    ideal = n * 1000 / tpot
    share = actual / ideal
    head = (
        f"A running request got a token every {tpot:.1f} ms, so {n} at once could produce "
        f"about {ideal:,.0f} tokens/s. {how} {actual:,.0f}, {share:.0%} of that"
    )
    if share >= 0.9:
        return Insight("pace", "The run kept its requests busy", f"{head}.", "ok")
    if bz is not None and bz.mean_running < 0.9 * n:
        avg = round(bz.mean_running)
        why = f": on average {_plural(avg, 'request')} ran at once, not {n}"
    else:
        reasons = [r for r in (_preempted(rep),) if r]
        why = (
            f": the server {reasons[0]}, and the rest went to waiting and prompt processing"
            if reasons
            else ": the rest went to waiting, preemption and prompt processing"
        )
    return Insight("pace", f"The run reached {share:.0%} of its pace", f"{head}{why}.", "limit")
