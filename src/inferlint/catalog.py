"""Every tripwire in plain words: what goes wrong, why it matters, how to check.

Used by ``inferlint explain`` and by the HTML report, so the explanation a reader sees is the
same everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TRIPWIRES", "Tripwire"]


@dataclass(frozen=True)
class Tripwire:
    id: str
    name: str
    what: str
    why: str
    command: str


TRIPWIRES: dict[str, Tripwire] = {
    t.id: t
    for t in (
        Tripwire(
            "T1",
            "Silent preemption",
            "When the GPU memory that holds requests in progress (the KV cache) fills up, "
            "vLLM pauses a request and throws away its working notes. Before the request "
            "can continue, it has to re-read everything so far to rebuild them. vLLM "
            "writes nothing in its log when this happens. Only a counter moves.",
            "Each paused request waits and then repeats work, so it finishes later than it "
            "should. Timing results include this hidden delay, and long requests suffer "
            "most. Searching the log for it finds nothing, even when it happened.",
            "inferlint preemption before.json after.json",
        ),
        Tripwire(
            "T2",
            "Engine left running",
            'The usual way to stop vLLM, pkill -f "vllm serve", only reaches the front-end '
            "process. The engine process, which holds the model in GPU memory, renames "
            "itself VLLM::EngineCore and can keep running after the front end is gone.",
            "The next test starts on a GPU that is already mostly full and produces wrong "
            "numbers with no error. One low memory reading is not proof the GPU is free.",
            "inferlint teardown  |  inferlint gpu-inspect",
        ),
        Tripwire(
            "T3",
            "Cleanup kills itself",
            'Running pkill -f "vllm serve" inside a one-line shell command also kills that '
            "shell, because the shell's own command line contains the same words.",
            "The rest of the cleanup never runs and the script ends with no message.",
            "inferlint teardown",
        ),
        Tripwire(
            "T4",
            "Memory pool changes between starts",
            "Each time vLLM starts, it measures free memory and sets aside a pool for "
            "requests in progress. The same settings can get a different pool on the next "
            "start, and pool sizes come in fixed steps.",
            "Two runs with identical settings can differ by a whole step (9.4% in the "
            "recorded example), which looks like a real speed difference and is not one.",
            "inferlint check-log boot1.log boot2.log",
        ),
        Tripwire(
            "T5",
            "Large memory blocks",
            "Models that mix attention layers with Mamba layers make vLLM hand out cache "
            "memory in large blocks, hundreds of tokens each, instead of 16-token blocks.",
            "A short request still occupies a whole block, so far fewer requests fit than "
            "the pool's token count suggests.",
            "inferlint check-log boot.log",
        ),
        Tripwire(
            "T6",
            "Requested backend ignored",
            "The attention backend is the code that does the model's main calculation on "
            "the GPU. vLLM has several, such as FlashAttention, FlashInfer and Triton, and "
            "you can choose one when you start the server with --attention-backend. This "
            "checks that vLLM really used the one you asked for. With speculative decoding "
            "on, the small draft model that helps the main model picks its own backend and "
            "quietly ignores your choice.",
            "In the recorded case Triton was requested, but the draft model picked "
            "FlashInfer, which crashed on the first request, while the log showed Triton "
            "in use.",
            "inferlint check-log boot.log",
        ),
        Tripwire(
            "T7",
            "Ready, then crashes",
            "A server can start, announce that it is ready, and crash as soon as the first "
            "real request arrives.",
            "A check that only waits for 'ready' passes a server that cannot serve anything.",
            "inferlint probe http://127.0.0.1:8000",
        ),
        Tripwire(
            "T8",
            "Fewer users than requested",
            "A load tool can keep 32 requests open, but the server may only work on a few "
            "at a time and queue the rest.",
            "A result labelled '32 users' can describe a 10-user server.",
            "inferlint series run.series.jsonl --requested 32",
        ),
        Tripwire(
            "T9",
            "Capacity shrinks as answers grow",
            "The number of requests that fit at once is 1 divided by the share of cache one "
            "request holds. Each answer takes more cache as it gets longer, so fewer fit "
            "as a run goes on, and the server evicts some to make room.",
            "A run has no single 'maximum concurrency'. It changes while the run is going, "
            "and every drop causes preemptions (T1).",
            "inferlint series run.series.jsonl",
        ),
        Tripwire(
            "T10",
            "Whole-second timers",
            "Timing a run with whole seconds, such as $(date +%s), can be off by almost a "
            "second either way.",
            "On a 30-second test that is about 3%, enough to flip a close comparison.",
            "inferlint rate before.json after.json",
        ),
        Tripwire(
            "T11",
            "KV memory varies between starts",
            "The memory left for the pool after loading the model and capturing CUDA "
            "graphs can differ slightly from one start to the next.",
            "A small difference can drop the pool by a whole step (T4).",
            "inferlint check-log boot1.log boot2.log",
        ),
        Tripwire(
            "T12",
            "Failures with no reason",
            "When a server fails to start, simple log searches often miss the real error "
            "and record a blank reason.",
            "A failure without a reason cannot be fixed, repeated or trusted.",
            "inferlint check-log boot.log",
        ),
        Tripwire(
            "T13",
            "Old script adopts the new server",
            "A test script stopped while it waits for its server can mistake the next "
            "script's server for its own, report it ready, then shut it down.",
            "The next test dies partway through for no visible reason.",
            "see docs/orchestration.md",
        ),
        Tripwire(
            "T14",
            "One block is reserved",
            "vLLM reports N cache blocks but always keeps one empty. Its usage gauge "
            "counts out of N - 1.",
            "Capacity worked out from the reported block count is slightly too high.",
            "inferlint series run.series.jsonl --snapshot after.json",
        ),
    )
}
