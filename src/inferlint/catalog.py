"""Every tripwire: what goes wrong, why it changes results, and how to check for it.

Used by ``inferlint explain`` and by the HTML report, so the explanation a reader sees is the
same everywhere. The wording is technical: the precise term first, then a short gloss. The
tutorial on the documentation site explains the same ideas in plainer words.

Each tripwire has a stable code (T1-T15, in the order they were found) and a name for
people (``silent-preemption``); ``inferlint explain`` accepts either.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ENGINE_NOTES", "TRIPWIRES", "Tripwire", "lookup"]

# vLLM releases the version-specific statements below were verified on.
VERIFIED = "verified on vLLM 0.28, 0.29 and 0.30"


@dataclass(frozen=True)
class Tripwire:
    id: str
    slug: str
    name: str
    what: str
    why: str
    command: str


TRIPWIRES: dict[str, Tripwire] = {
    t.id: t
    for t in (
        Tripwire(
            "T1",
            "silent-preemption",
            "Silent preemption",
            "When the KV cache has no free block for a running request, vLLM preempts a "
            "request: it frees that request's KV blocks and later recomputes them from the "
            "prompt and the output so far. vLLM increments vllm:num_preemptions_total and "
            f"writes no log line at the default log level ({VERIFIED}).",
            "A preempted request waits in the queue again and repeats work, so latency and "
            "throughput include recomputation the client never sees. Searching the server "
            "log finds nothing.",
            "inferlint preemption before.json after.json",
        ),
        Tripwire(
            "T2",
            "orphaned-engine",
            "Orphaned engine process",
            'pkill -f "vllm serve" matches only the API-server process. The engine process, '
            "which holds the weights and the KV cache in GPU memory, renames itself "
            "VLLM::EngineCore and can outlive its parent when the parent is killed.",
            "The next server starts in whatever GPU memory is left and sizes a smaller KV "
            "pool, with no error. One low memory reading is not proof the GPU is free, so "
            "inferlint waits for two in a row.",
            "inferlint teardown  |  inferlint gpu-inspect",
        ),
        Tripwire(
            "T3",
            "cleanup-self-kill",
            "Cleanup kills its own shell",
            'pkill -f "vllm serve" inside a bash -c one-liner also matches that shell, '
            "because the shell's own command line contains the pattern.",
            "The shell is killed before the rest of the cleanup runs, and the script ends "
            "without an error message.",
            "inferlint teardown",
        ),
        Tripwire(
            "T4",
            "pool-size-jump",
            "KV pool size jumps between starts",
            "vLLM sizes the KV pool at each start from the GPU memory left after loading "
            "the model and capturing CUDA graphs, and the pool comes in discrete steps. "
            "Starts with identical flags can land on different steps; T11 is the cause.",
            "Two runs with the same flags can differ by a whole step (9.4% in the recorded "
            "example), which reads as a performance difference and is not one.",
            "inferlint check-log boot1.log boot2.log",
        ),
        Tripwire(
            "T5",
            "large-kv-blocks",
            "Large KV block size",
            "For hybrid attention/Mamba models, vLLM raises the attention block size (the KV "
            "cache allocation unit) from 16 tokens to hundreds, so that attention pages and "
            "Mamba state pages line up.",
            "Every request holds whole blocks, so a short request occupies a full block and "
            "far fewer requests fit than the pool's token count suggests.",
            "inferlint check-log boot.log",
        ),
        Tripwire(
            "T6",
            "backend-ignored",
            "Requested attention backend ignored",
            "--attention-backend sets the target model's attention backend. With speculative "
            "decoding, the draft model selects its own backend from a candidate list and "
            "does not apply the flag.",
            "In the recorded case Triton was requested and logged for the target model, but "
            "the draft model selected FlashInfer, which failed on the first request.",
            "inferlint check-log boot.log",
        ),
        Tripwire(
            "T7",
            "crash-after-ready",
            "Crash after ready",
            "A server can log 'Application startup complete' and then fail on its first "
            "request, when a code path start-up never ran is reached (for example a kernel "
            "compiled on first use).",
            "A gate that only waits for readiness passes a server that cannot serve. "
            "inferlint probe sends real requests; inferlint check-log reports T7 when the "
            "log shows a failure after ready.",
            "inferlint probe http://127.0.0.1:8000",
        ),
        Tripwire(
            "T8",
            "concurrency-not-reached",
            "Concurrency not reached",
            "A load tool's concurrency is the number of requests it keeps open. "
            "vllm:num_requests_running is the number the scheduler actually runs; the rest "
            "wait in its queue. Admission-control limits (--max-num-queued-reqs, "
            "--max-num-queued-tokens, vLLM 0.29 and later) can also reject requests.",
            "A result labelled 'concurrency 32' can describe a server that ran far fewer "
            "requests at once (never more than 10 in the recorded vLLM run).",
            "inferlint series run.series.jsonl --requested 32",
        ),
        Tripwire(
            "T9",
            "falling-ceiling",
            "Concurrency ceiling falls as sequences grow",
            "With the KV cache full, the number of requests that fit is floor(1 / share), "
            "where share is the fraction of the cache one running request holds. The share "
            "grows with sequence length, so the ceiling falls during a run, and each fall "
            "preempts requests (T1).",
            "A run has no single maximum concurrency, so one figure for it describes none "
            "of the run.",
            "inferlint series run.series.jsonl",
        ),
        Tripwire(
            "T10",
            "integer-second-timer",
            "Integer-second timing",
            "Timing a run with whole seconds, such as $(date +%s), is off by up to one "
            "second at each end.",
            "On a 30-second run that is about 3%, enough to reverse a close comparison. "
            "inferlint uses the sub-second clocks recorded with each snapshot.",
            "inferlint rate before.json after.json",
        ),
        Tripwire(
            "T11",
            "kv-memory-drift",
            "KV memory budget drifts between starts",
            "The 'Available KV cache memory' vLLM reports at start-up can differ between "
            "starts with identical flags, mostly because CUDA-graph capture memory varies.",
            "A small drift can move the KV pool to a different step (T4). Record the "
            "per-start figure with every result.",
            "inferlint check-log boot1.log boot2.log",
        ),
        Tripwire(
            "T12",
            "unexplained-failure",
            "Start-up failure without a cause",
            "When a server fails to start, a search of the log for 'Error' usually finds a "
            "wrapper line rather than the root cause, and the failure is recorded with no "
            "reason.",
            "A failure without its cause cannot be fixed, reproduced or compared.",
            "inferlint check-log boot.log",
        ),
        Tripwire(
            "T13",
            "stale-waiter",
            "Stale waiter adopts the next server",
            "A test script stopped while it waits for its server to become ready can leave "
            "its wait loop running. That loop sees the next script's server come up, "
            "reports it as its own, and later shuts it down.",
            "The next test dies partway through with no visible cause.",
            "see docs/orchestration.md",
        ),
        Tripwire(
            "T14",
            "null-block",
            "Reserved null block",
            "vLLM exports num_gpu_blocks = N but reserves one block (the null block), and "
            f"vllm:kv_cache_usage_perc counts out of N - 1 ({VERIFIED}).",
            "Capacity computed from the exported block count is one block too high.",
            "inferlint series run.series.jsonl --snapshot after.json",
        ),
        Tripwire(
            "T15",
            "client-server-mismatch",
            "Client and server counts disagree",
            "The load tool's own counts for the run (requests completed, output tokens) "
            "are compared with the server's counters between the before and after "
            "snapshots.",
            "If the server did more work than the load tool sent, other traffic shared the "
            "server and the result is contaminated. If it did less, the load tool counted "
            "work the server did not do.",
            "inferlint xray -o results/ -- vllm bench serve ...",
        ),
    )
}

# How a tripwire differs on SGLang and TensorRT-LLM, keyed by tripwire, then engine.
# Verified on SGLang 0.5.20 and TensorRT-LLM 1.3.0rc29. The descriptions above are as vLLM
# shows each problem, where most were first found; docs/tripwires.md carries these notes
# too (tests/test_every_engine.py checks that it does).
ENGINE_NOTES: dict[str, dict[str, str]] = {
    "T1": {
        "sglang": "called a retraction, counted in sglang:num_retracted_requests_total "
        "(exported only after the first one) and logged as a warning each time.",
        "trtllm": "a request is paused for recompute and nothing counts it. Each pause is "
        "logged at INFO, and inferlint counts those lines in the server log. The default "
        "scheduler policy, GUARANTEED_NO_EVICT, pauses none.",
    },
    "T2": {
        "sglang": "the workers are sglang::scheduler and sglang::detokenizer. In tests "
        "they exited when the launcher was killed.",
        "trtllm": "the model runs in python -m mpi4py.futures.server under prte, both "
        "started by trtllm-serve. In tests they exited when trtllm-serve was killed.",
    },
    "T5": {
        "sglang": "KV is allocated per token (page size 1). For hybrid attention/Mamba "
        "models a fixed state slot per running request limits concurrency instead (T8).",
        "trtllm": "KV is allocated in blocks of tokens_per_block tokens, 32 by default.",
    },
    "T8": {
        "sglang": "the running limit can be lowered below --max-running-requests, for "
        "hybrid models to fit the per-request state slots. The log says so once; "
        "/get_server_info still reports the value the server was started with.",
        "trtllm": "gauges are updated only when a request completes, so a recording can "
        "miss the peak.",
    },
    "T9": {
        "sglang": "the usage gauge, sglang:token_usage, is the fullest memory pool: for a "
        "hybrid model it can be the state slots rather than the KV cache. It is rounded to "
        "two decimals.",
        "trtllm": "gauges are updated only when a request completes, so the ceiling can "
        "rest on few readings.",
    },
    "T14": {
        "sglang": "does not apply; the reserved null block is vLLM's.",
        "trtllm": "does not apply; the reserved null block is vLLM's.",
    },
}

_BY_SLUG = {t.slug: t for t in TRIPWIRES.values()}


def lookup(key: str) -> Tripwire | None:
    """A tripwire by code (``T1``, ``t1``) or name (``silent-preemption``)."""
    k = key.strip()
    return TRIPWIRES.get(k.upper()) or _BY_SLUG.get(k.lower())
