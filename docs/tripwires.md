# The tripwires

Each entry gives the symptom, why it changes a benchmark's numbers, what the check does,
and the command that reproduces it from files in this repository, then what differs on
SGLang and TensorRT-LLM. The symptoms are described as vLLM shows them, where most of
them were first found. Fixtures are real boot logs and metrics from an RTX 4090 (WSL2,
driver 616.56), with local paths removed: vLLM 0.28 in `tests/fixtures/vllm-0.28/`, 0.29
and 0.30 in their own folders, SGLang 0.5.20 in `tests/fixtures/sglang-0.5/` and
TensorRT-LLM 1.3.0rc29 in `tests/fixtures/trtllm-1.3/`.

Codes (T1 to T15) are permanent, in the order the tripwires were found. Each also has a
name, and `inferlint explain` accepts either: `inferlint explain T9`,
`inferlint explain falling-ceiling`.

"Verified" says how each tripwire has been checked:
**live 0.28 / 0.29 / 0.30**: measured by this code on a running vLLM server of that
version; **SGLang, TensorRT-LLM**: measured live on SGLang 0.5.20 and TensorRT-LLM
1.3.0rc29;
**fixture**: tested against recorded evidence in `tests/fixtures/`;
**mock**: tested against a mock server with known behaviour.

| id | name | trap | check | verified |
|---|---|---|---|---|
| T1 | silent-preemption | a preempted request leaves no log line (vLLM) or no count (TensorRT-LLM) | `preemption` | live 0.28 / 0.29 / 0.30; SGLang, TensorRT-LLM |
| T2 | orphaned-engine | `pkill -f` on the server command misses the engine's processes | `teardown`, `gpu-inspect` | live 0.28 / 0.29 / 0.30; SGLang, TensorRT-LLM |
| T3 | cleanup-self-kill | `pkill -f` in a shell one-liner kills the shell | `teardown` | fixture |
| T4 | pool-size-jump | the KV pool is sized per start, in steps | `check-log` over several starts | fixture 0.28 (a jump); live 0.29 / 0.30 (2 starts each, same pool); SGLang, TensorRT-LLM (2 starts each, same pool) |
| T5 | large-kv-blocks | on vLLM, hybrid models force a large attention block | `check-log` | live 0.28 / 0.29 / 0.30; SGLang, TensorRT-LLM |
| T6 | backend-ignored | the requested attention backend is not applied everywhere | `check-log` | fixture 0.28 |
| T7 | crash-after-ready | a config starts, then dies on its first request | `probe`, `check-log` | live 0.28 / 0.29 / 0.30 (healthy), fixture 0.28, mock; SGLang, TensorRT-LLM (healthy) |
| T8 | concurrency-not-reached | requested concurrency is not achieved concurrency | `series --requested N` | live 0.28 / 0.29 / 0.30; admission control live 0.30; SGLang, TensorRT-LLM |
| T9 | falling-ceiling | the concurrency ceiling is `floor(1 / share)`, and it falls | `series` | live 0.28 / 0.29 / 0.30; SGLang, TensorRT-LLM |
| T10 | integer-second-timer | integer-second timers are off by up to a second at each end | `rate` | live 0.28 / 0.29 / 0.30; SGLang, TensorRT-LLM |
| T11 | kv-memory-drift | KV memory differs between starts with the same flags | `check-log` over several starts | fixture 0.28 (drift); live 0.29 / 0.30 (2 starts each, same memory); SGLang, TensorRT-LLM (2 starts each, same memory) |
| T12 | unexplained-failure | start-up failures recorded without a cause | `check-log` | live 0.28 / 0.29 / 0.30 (healthy); real failures on 0.28, 0.29, 0.30, SGLang and TensorRT-LLM |
| T13 | stale-waiter | a killed campaign's waiter adopts the next server | see `orchestration.md` | documented |
| T14 | null-block | vLLM's usage gauge counts out of `num_gpu_blocks - 1` | `series --snapshot` | live 0.28 / 0.29 / 0.30 (vLLM only) |
| T15 | client-server-mismatch | the load tool's counts differ from the server's | `xray` | live 0.28 / 0.29 / 0.30, mock; SGLang, TensorRT-LLM, also with their own benchmark clients |

All commands are `inferlint <command>`. `inferlint xray` runs every check that applies
around one benchmark run.

---

## T1: silent preemption

**Symptom.** When the KV cache has no free block for a running request, vLLM preempts a
request: it frees that request's KV blocks and later recomputes them from the prompt and
the output so far. At the default log level nothing is written about it. The only trace
is the counter `vllm:num_preemptions_total` (verified on vLLM 0.28, 0.29 and 0.30; 0.30
also exports a per-request histogram, `vllm:request_num_preemptions`).

**Why it matters.** A preempted request's latency includes waiting in the queue again and
recomputing its KV, so a latency or long-context result that assumed uninterrupted
generation is void. Searching server logs for "preempt" returns zero lines whether or not
it happened.

**Check.** The counter's delta between two snapshots. An absent counter is Can't tell,
never zero. A server restarted between the snapshots is detected by the `*_created`
stamps, which catches a restart even when traffic has already pushed the counters past
their old values (a "counter went down" test misses this).

**Measured.** The same 32-request run preempted 22 times on vLLM 0.28, 21 on 0.29 and 22 on
0.30, and 21 times in a second 0.28 run under `xray`. The server log mentioned none of
them.

```console
$ inferlint preemption tests/fixtures/vllm-0.28/metrics/preempted_{before,after}.prom
[ T1] TRIPWIRE-FAILED  60 preemptions during the run
$ inferlint preemption tests/fixtures/vllm-0.28/metrics/restart_{before,after}.prom
[ T1] TRIPWIRE-FAILED  server restarted between snapshots (41 *_created stamps moved, ...)
```

**On SGLang:** called a retraction, counted in sglang:num_retracted_requests_total
(exported only after the first one) and logged as a warning each time.

**On TensorRT-LLM:** a request is paused for recompute and nothing counts it. Each pause
is logged at INFO, and inferlint counts those lines in the server log. The default
scheduler policy, GUARANTEED_NO_EVICT, pauses none.

**Measured on SGLang and TensorRT-LLM:** SGLang retracted 1 request with
`--schedule-conservativeness 0.3`, and 3 when SGLang's own benchmark client sent the
load; none with `vllm bench serve` and the default settings. TensorRT-LLM paused 13
requests with `capacity_scheduler_policy: MAX_UTILIZATION` and a 20,960-token pool; its
paused-requests gauge showed none of them in 171 readings, and inferlint counted all 13
from its log. With the default policy it paused none.

## T2: orphaned engine process

**Symptom.** vLLM renames its engine child process to `VLLM::EngineCore`. The phrase
`vllm serve` is no longer in its command line, so the usual cleanup signals the API
server and not the process holding the weights and KV cache.

**Why it matters.** If the parent dies uncleanly, the engine keeps the GPU. The next
start sizes its KV pool from what is left, with no error. One low memory reading does not
prove the card is free either: a dying engine can read low partway through its own exit.

**Measured.** The API server was killed with SIGKILL on each version. Twenty seconds later
`VLLM::EngineCore` was still alive, re-parented, and held 21,768 MiB (0.28), 21,522 MiB
(0.29) and 21,872 MiB (0.30). `inferlint gpu-inspect` refused to start a server.
`inferlint teardown` found the engine, which `pkill -f 'vllm serve'` would have missed,
and the card read 0 MiB afterwards.

**Check.** Processes are matched on their argv (`vllm serve`, `python -m
vllm.entrypoints...`, any `VLLM::*`), never including the caller or its ancestors.
SIGTERM, then SIGKILL survivors, then require *consecutive* readings with no server
process and every GPU below a threshold. `inferlint gpu-inspect` is the start gate: it
shows which GPU this is and its memory, and exits non-zero on a card that is not free.

**On SGLang:** the workers are sglang::scheduler and sglang::detokenizer. In tests they
exited when the launcher was killed.

**On TensorRT-LLM:** the model runs in python -m mpi4py.futures.server under prte, both
started by trtllm-serve. In tests they exited when trtllm-serve was killed.

## T3: cleanup kills its own shell

**Symptom.** `bash -lc '...; pkill -f "vllm serve"'` matches the shell's own command
line, which contains the pattern, and kills it. The one-liner exits with status 143 and
never runs its remaining commands.

**Check.** `teardown` never uses substring matching and never targets its own ancestry.
`tests/test_teardown.py` shows the naive pattern hitting the shell and a bystander
(`tail -f "vllm serve.log"`) while missing both engines.

## T4: KV pool size jumps between starts

**Symptom.** Two starts with identical flags can report different `GPU KV cache size`
values. The values fall on a ladder of discrete steps, so the difference is a whole
allocation step, not noise. T11 is the cause.

**Why it matters.** Throughput and preemption depend on the pool. Two results compared
without their pools may be comparing two different servers. The pool also changes between
vLLM releases: with the same flags, model and GPU, 0.28 and 0.29 sized 26,093 tokens and
0.30 sized 29,127.

**Check.** Groups boot logs by identical non-default args and compares pools within each
group. Every result should carry its own start's pool.

```console
$ inferlint check-log tests/fixtures/vllm-0.28/boot_pool_level_{hi,lo}.log
[ T4] WARN  same flags, 2 different pools: 12,405 to 13,575 tokens (9.43% apart) ...
[T11] WARN  same flags, KV cache memory varied: 1.44 to 1.59 GiB
```

**On SGLang and TensorRT-LLM:** Two starts of each with the same flags drew the same pool
(32,096 and 37,760 tokens). SGLang starts that differ only in the random seed it draws
for itself are compared as the same flags.

## T5: large KV block size

**Symptom.** For hybrid attention/Mamba models, vLLM raises the attention block size
until the attention page is at least as large as the Mamba state page: 400 to 1,568
tokens in the fixtures, 784 on every live run, where the default is 16.

**Why it matters.** The block is the allocation unit. With a 784-token block, a
100-token request holds 784 tokens of cache, and the pool is a few dozen blocks, so
capacity moves in large steps.

**On SGLang:** KV is allocated per token (page size 1). For hybrid attention/Mamba models
a fixed state slot per running request limits concurrency instead (T8).

**On TensorRT-LLM:** KV is allocated in blocks of tokens_per_block tokens, 32 by default.

## T6: requested attention backend ignored

**Symptom.** `--attention-backend TRITON_ATTN` sets the target model's backend. With
speculative decoding on, the draft model picks its own backend from a candidate list. In
the fixture it picked FlashInfer, whose kernel build then failed on the first request.

**Check.** Separates the target model's selection from the drafter's (backend lines after
`Loading drafter model`) and fails if either differs from the request.

```console
$ inferlint check-log tests/fixtures/vllm-0.28/boot_spec_backend_override.log
[ T6] TRIPWIRE-FAILED  TRITON_ATTN was requested and the main model uses it, but the draft model (speculative decoding) picked FLASHINFER
```

## T7: crash after ready

**Symptom.** The boot log ends with `Application startup complete`. The first request
reaches a kernel compiled on first use, and the engine dies.

**Check.** `inferlint probe` sends one request, then 8 at once, then checks `/health`.
`check-log` reports a failure after readiness as T7 and one before readiness as T12. The
probe is tested against a mock server with known failure modes (`tests/test_probe.py`).

## T8: concurrency not reached

**Symptom.** A load tool's concurrency is the number of requests it keeps open.
`vllm:num_requests_running` is the number the scheduler actually runs; the rest wait in
its queue. When the KV cache is small, it runs a handful. The boot line `Maximum
concurrency for N tokens per request` assumes every request fills the full context, so it
can neither confirm nor refute this.

**Admission control.** vLLM 0.29 added `--max-num-queued-reqs`: at most that many
requests in flight (running plus waiting), and the rest are rejected with HTTP 503 rather
than queued. When the boot log sets the limit and the recording reached it, T8 says the
requests were rejected, not that the cache was full. On vLLM 0.30 with
`--max-num-queued-reqs 16` and 32 requests asked for, the benchmark marked 16 requests as
failed ("Never received a valid chunk to calculate TTFT") without saying why; T8
reported:

```console
[ T8] TRIPWIRE-FAILED  requested 32 concurrent, server never ran more than 11; admission control (--max-num-queued-reqs 16) kept at most 16 in flight, so the other 16 were rejected, not queued
```

**Check.** Peak `num_requests_running` from a gauge series recorded during the run.

```console
$ inferlint series tests/fixtures/vllm-0.28/series/ladder.series.jsonl --requested 32
[ T8] TRIPWIRE-FAILED  requested 32 concurrent, server never ran more than 4
```

**On SGLang:** the running limit can be lowered below --max-running-requests, for hybrid
models to fit the per-request state slots. The log says so once; /get_server_info still
reports the value the server was started with.

**On TensorRT-LLM:** gauges are updated only when a request completes, so a recording can
miss the peak.

**Measured on SGLang and TensorRT-LLM:** With Qwen3-8B and 32 at once, SGLang ran at most
29 under `vllm bench serve` and 32 under its own client; TensorRT-LLM ran 32 under both.

## T9: concurrency ceiling falls as sequences grow

`share = usage / running` is the fraction of the cache one running request holds, and
`floor(1 / share)` requests fit. With the block count known (T14), the arithmetic is done
in whole blocks, so a reading of exactly 1/8 cannot floor to 7 through float error.

The share is not a constant of the config. A request holds more blocks as its output
grows, so the ceiling falls during a run, and every fall evicts running requests (T1).
The check reads the share from every sample where the cache is the binding constraint
(usage at or above 90%) and reports the range. It passes when the ceiling held still and
warns when it moved.

**Measured.** The 32-request run (1,000 output tokens each, 784-token blocks): on vLLM
0.28, 42 usable blocks, a fresh request held 4 and 10 fit; requests held up to 6 blocks
each on average and the ceiling fell to 7. On 0.29 it fell from 10 to 5; on 0.30, with 47
usable blocks, from 11 to 7. A second 0.28 run, under `xray`, fell from 10 to 5, so the
lowest point varies between runs. Running never exceeded the ceiling computed from the same
sample (`tests/test_live.py`). On fixed-length requests (the ladder fixture) the ceiling
is 4 on every saturated sample, and 4 is exactly the peak T8 measured.

**On SGLang:** the usage gauge, sglang:token_usage, is the fullest memory pool: for a
hybrid model it can be the state slots rather than the KV cache. It is rounded to two
decimals.

**On TensorRT-LLM:** gauges are updated only when a request completes, so the ceiling can
rest on few readings.

## T10: integer-second timing

**Symptom.** `wall=$(( $(date +%s) - t0 ))` is a difference of two integer-second stamps,
so it lands within ±1 s of the true duration. On a 29-second run that is ±3.5%, and it
lands independently on each arm of a comparison.

**Check.** Tokens counted between two snapshots divided by the time between those same
snapshots, from a monotonic clock, with the scrape latency recorded as the timing
uncertainty. The integer-second figure is shown alongside for comparison.

On the vLLM 0.28 run the server's generation counter moved by exactly 32,000 tokens (32
requests of 1,000) over 93.720 s monotonic, with 0.007 s of scrape uncertainty: 341.44
tokens/s. An integer-second timer read 344.09 (+0.77%). The error grows as runs shorten.

## T11: KV memory budget drifts between starts

CUDA-graph capture and allocator state vary between starts. A small difference in the
memory left for the KV cache can move the pool down a whole step (T4). `check-log`
reports `Available KV cache memory` and CUDA-graph memory per start, grouped by flags.

**On SGLang and TensorRT-LLM:** Two starts of each with the same flags left the same
memory for the KV cache (4.4 and 5.19 GiB).

## T12: start-up failure without a cause

**Symptom.** A failure search for `ValueError|RuntimeError` misses
`torch.AcceleratorError: CUDA error: device not ready` and records a blank. A kernel
build failure is logged as "Ninja build failed", with the cause several lines away.

**Check.** An ordered classifier: specific root causes first (a kernel build's own error,
out of memory, KV cache too small, accelerator and CUDA errors), generic exception types
after, "engine dead" last because it follows every root cause and explains none.
Tracebacks after an orderly shutdown (vLLM's `[shutdown]` lines, as when a teardown stops
the server) are not failures and are ignored.

**Measured.** Setting up the 0.29 and 0.30 runs on WSL with pip-installed CUDA produced
four real start-up failures, and T12 named each cause in one line:

```console
[T12] TRIPWIRE-FAILED  jit_toolchain during boot: CUDA compiler and CUDA toolkit headers are incompatible
[T12] TRIPWIRE-FAILED  jit_toolchain during boot: Unsupported .version 9.4; current version is '9.0'
[T12] TRIPWIRE-FAILED  jit_toolchain during boot: cannot find -lcudart: No such file or directory
[T12] TRIPWIRE-FAILED  runtime_error during boot: UVA is not available
```

The first three are FlashInfer 0.6.18 (vLLM 0.29 and 0.30) building its sampling
kernels: it needs nvcc, its CRT headers and NVVM at the same version as the CUDA runtime
torch installs (13.0), and links against `libcudart` and `libcuda`. The fourth is vLLM
0.29's new model runner, which needs pinned host memory that vLLM does not use on WSL;
`VLLM_USE_V2_MODEL_RUNNER=0` selects the previous runner.

A fifth came from vLLM 0.28 under `inferlint xray`: the same headers message as the first,
from FlashInfer 0.6.16 compiling its sampler with nvcc 13.3 against CUDA 13.0 headers.
The recorded 0.28 run had set `VLLM_USE_FLASHINFER_SAMPLER=0`, so that sampler was never
built; with the same setting, xray's 0.28 run started normally. The logs are in
`tests/fixtures/vllm-0.28/`, `vllm-0.29/` and `vllm-0.30/`.

**On SGLang and TensorRT-LLM:** Real start-up failures from each are in the fixtures and
named: SGLang running out of memory for a hybrid model's state slots, and advising to
raise `--mem-fraction-static` above 0.790 when it was already 0.88; TensorRT-LLM failing
to read a weight-only compressed-tensors checkpoint (named by the function it failed in),
and a bare `assert cuda_home is not None`. SGLang logs tracebacks it ignores and errors
it survives; neither counts as a failure.

## T13: stale waiter adopts the next server

A shell `trap` does not run while the script is blocked in a child. A campaign killed
while waiting for readiness can see the *replacement* campaign's server on the same port,
report it ready, and then run its deferred cleanup against it. See `orchestration.md`.

## T14: reserved null block

**Finding.** `vllm:kv_cache_usage_perc` readings are exact multiples of `1/N`, and N is
recoverable from the readings alone. It is one less than the `num_gpu_blocks` the server
exports in `vllm:cache_config_info` (verified on vLLM 0.28, 0.29 and 0.30: 42 of 43, 42
of 43, 47 of 48). One block is reserved as a null block and never holds a request's KV.
On a pool of 45 large blocks, that is 2.2% of the advertised capacity.

**Check.** Infers N from the gauge (least common denominator of the readings, trusted
only with at least five distinct values) and compares it with the exported count. A
change in either side, or a series and snapshot from different starts, turns it red.

```console
$ inferlint series tests/fixtures/vllm-0.28/series/blocks.series.jsonl \
      --snapshot tests/fixtures/vllm-0.28/series/blocks_snapshot.prom
[T14] PASS  gauge denominator is 44 = num_gpu_blocks (45) - 1 null block
```

**On SGLang:** does not apply; the reserved null block is vLLM's.

**On TensorRT-LLM:** does not apply; the reserved null block is vLLM's.

## T15: client and server counts disagree

**Symptom.** A load tool reports what it sent and received; the server's counters say what
it did. If another client shares the server during the run (a second benchmark, a health
check that generates, a colleague), the server's numbers include that load and the
benchmark's results are contaminated. If the load tool counts work the server did not do,
its numbers are wrong in the other direction.

**Check.** `inferlint xray` saves the load tool's result file and compares its
completed requests and output tokens with the server's `vllm:request_success_total` and
`vllm:generation_tokens_total` between the before and after snapshots. Requests
`vllm bench serve` sends but leaves out of its result (its initial test request and
warm-ups, which its output reports) are allowed for.

**Measured.** On vLLM 0.28, 0.29 and 0.30 the server's counts matched the benchmark
exactly: 32 requests and 32,000 output tokens. With `--max-num-queued-reqs 16` on 0.30,
the server rejected 16 of the 32 requests. `vllm bench serve` counted them as failed, each with
"Never received a valid chunk to calculate TTFT"; T8 named admission control as the
cause, and T15 matched the 16 requests and 16,000 output tokens the server did finish.

**On SGLang and TensorRT-LLM:** `xray` also asks their own benchmark clients to save a
result: SGLang's `python -m sglang.benchmark.serving` appends one JSON line per run to
`--output-file`, and its warm-up request is allowed for; TensorRT-LLM's `python -m
tensorrt_llm.serve.scripts.benchmark_serving` writes vLLM's format. With either engine
and either client, the counts matched: 32 requests and 32,000 output tokens.
