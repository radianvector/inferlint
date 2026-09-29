# The tripwires

Each entry gives the symptom, why it changes a benchmark's numbers, what the check does,
and the command that reproduces it from files in this repository. Fixtures are real
vLLM 0.28 boot logs and `/metrics` output from an RTX 4090, with paths removed.

"Status" says how far each tripwire has been verified:
**fixture**: tested against recorded evidence in `tests/fixtures/`;
**live 0.28**: re-measured by this code on a running vLLM 0.28;
**current**: re-measured on the current vLLM release (pending for all).

| id | trap | check | status |
|---|---|---|---|
| T1 | preemption is silent | `inferlint preemption` | fixture, live 0.28 |
| T2 | `pkill -f "vllm serve"` misses the engine | `inferlint teardown`, `inferlint gpu-inspect` | fixture, live 0.28 |
| T3 | `pkill -f` in a shell one-liner kills the shell | structural matching in `teardown` | fixture, live |
| T4 | the KV pool is drawn per boot, in levels | `inferlint check-log` over several boots | fixture |
| T5 | hybrid models force a large attention block | `inferlint check-log` | fixture, live 0.28 |
| T6 | the requested attention backend is not applied everywhere | `inferlint check-log` | fixture |
| T7 | a config can boot, then die on its first request | `inferlint probe`, `inferlint check-log` | fixture, mock server, live 0.28 (healthy case) |
| T8 | requested concurrency is not achieved concurrency | `inferlint series --requested N` | fixture, live 0.28 |
| T9 | the concurrency ceiling is `floor(1 / share)`, and it moves | `inferlint series` | fixture, live 0.28 |
| T10 | integer-second timers are worth several percent | `inferlint rate` | fixture, live 0.28 |
| T11 | KV memory differs between boots of one config | `inferlint check-log` over several boots | fixture |
| T12 | boot failures recorded without a reason | `inferlint check-log` | fixture, live 0.28 (healthy case) |
| T13 | a killed campaign's waiter adopts the next server | see `orchestration.md` | documented |
| T14 | the usage gauge's denominator is `num_gpu_blocks - 1` | `inferlint series --snapshot` | fixture, live 0.28 |

---

## T1: preemption is silent

**Symptom.** Under KV-cache pressure vLLM preempts running requests: it frees their
cache and later recomputes them from the start. At the default log level nothing is
written about it. The only trace is `vllm:num_preemptions_total`.

**Why it matters.** A preempted request's latency includes a full recompute, so a
latency or long-context result that assumed uninterrupted generation is void. Grepping
server logs for "preempt" returns zero lines whether or not it happened, so it is not
a check at all.

**Check.** The counter's delta between two snapshots. An absent counter is `UNKNOWN`,
never zero. A server restarted between the snapshots fails loudly, detected by the
`*_created` stamps, which catches a restart even when traffic has already pushed the
counters past their old values (a "counter went down" test misses this).

```console
$ inferlint preemption tests/fixtures/vllm-0.28/metrics/preempted_{before,after}.prom
[ T1] FAIL  60 preemptions during the run
$ inferlint preemption tests/fixtures/vllm-0.28/metrics/restart_{before,after}.prom
[ T1] FAIL  server restarted between snapshots (41 *_created stamps moved, ...)
```

## T2: `pkill -f "vllm serve"` misses the engine

**Symptom.** vLLM renames its engine child process to `VLLM::EngineCore`. The phrase
`vllm serve` is no longer in its command line, so the usual cleanup signals the API
server and not the process holding the weights and KV cache.

**Why it matters.** If the parent dies uncleanly, the engine keeps the GPU. The next
boot loads a second copy into what is left and produces numbers from a starved server.
One low memory reading does not prove the card is free either: a dying engine can read
low partway through its own teardown.

**Measured.** In the live run the API server was killed with SIGKILL. Twenty seconds
later `VLLM::EngineCore` was still alive, re-parented, and the card showed 21,768 MiB in
use. `inferlint gpu-inspect` refused to boot. `inferlint teardown` found the engine (which
`pkill -f 'vllm serve'` would have missed) and the card read 0 MiB afterwards.

**Check.** Processes are matched on their argv (`vllm serve`, `python -m
vllm.entrypoints...`, any `VLLM::*`), never including the caller or its ancestors.
SIGTERM, then SIGKILL survivors, then require *consecutive* readings with no server
process and every GPU below a threshold. `inferlint gpu-inspect` is the boot gate: it
shows which GPU this is and its memory, and exits non-zero on a dirty card.

## T3: `pkill -f` in a shell one-liner kills the shell

**Symptom.** `bash -lc '...; pkill -f "vllm serve"'` matches the shell's own command
line, which contains the pattern, and kills it. The one-liner exits with status 143
and never runs its remaining commands.

**Check.** `teardown` never uses substring matching and never targets its own ancestry.
`tests/test_teardown.py` shows the naive pattern hitting the shell and a bystander
(`tail -f "vllm serve.log"`) while missing both engines.

## T4: the KV pool is drawn per boot, in levels

**Symptom.** Two boots with identical flags can report different `GPU KV cache size`
values. The values fall on a ladder of discrete levels, so the difference is a whole
allocation step, not noise.

**Why it matters.** Throughput and preemption depend on the pool. Two results compared
without their pools may be comparing two different servers.

**Check.** Groups boot logs by identical non-default args and compares pools within
each group. Every result should carry its own boot's pool.

```console
$ inferlint check-log tests/fixtures/vllm-0.28/boot_pool_level_{hi,lo}.log
[ T4] WARN  same flags, 2 different pools: 12,405 to 13,575 tokens (9.43% apart) ...
[T11] WARN  same flags, KV cache memory varied: 1.44 to 1.59 GiB
```

## T5: hybrid models force a large attention block

**Symptom.** On hybrid attention + Mamba models, vLLM raises the attention block size
until the attention page is at least as large as the Mamba state page: 400 to 1,568
tokens in the fixtures, where the default is 16.

**Why it matters.** The block is the allocation unit. With a 784-token block, a
100-token request holds 784 tokens of cache, and the pool is a few dozen blocks, so
capacity moves in large steps.

## T6: the requested attention backend is not applied everywhere

**Symptom.** `--attention-backend TRITON_ATTN` sets the target model's backend. With
speculative decoding on, the drafter picks its own backend from a candidate list. In
the fixture it picked FlashInfer, whose JIT build then failed on the first request.

**Check.** Separates the target model's selection from the drafter's (backend lines
after `Loading drafter model`) and fails if either differs from the request.

```console
$ inferlint check-log tests/fixtures/vllm-0.28/boot_spec_backend_override.log
[ T6] FAIL  TRITON_ATTN was requested and the main model uses it, but the draft model (speculative decoding) picked FLASHINFER
```

## T7: a config can boot, then die on its first request

**Symptom.** The boot log ends with `Application startup complete`. The first request
reaches a lazily compiled kernel and the engine dies.

**Check.** `inferlint probe` sends one request, then a concurrent burst, then checks
`/health`. `check-log` reports a failure after readiness as T7 (serving phase), and one
before readiness as T12. The probe is tested against a mock server with known
failure modes (`tests/test_probe.py`).

## T8: requested concurrency is not achieved concurrency

**Symptom.** A client with 32 requests in flight measures a 32-user server only if the
server runs 32 at once. When the cache is small, it runs a handful and queues the rest.
The boot line `Maximum concurrency for N tokens per request` assumes every request
fills the full context, so it can neither confirm nor refute this.

**Check.** Peak `num_requests_running` from a gauge series sampled during the run.

```console
$ inferlint series tests/fixtures/vllm-0.28/series/ladder.series.jsonl --requested 32
[ T8] FAIL  requested 32 concurrent, server never ran more than 4
```

## T9: the concurrency ceiling is `floor(1 / share)`, and it moves

`share = usage / running` is the fraction of the cache one running request holds, and
`floor(1 / share)` requests fit. With the block count known (T14), the arithmetic is done
in whole blocks, so a reading of exactly 1/8 cannot floor to 7 through float error.

The share is not a constant of the config. A request holds more blocks as its output
grows, so the ceiling falls during a run, and every fall evicts running requests (T1).
The check therefore reads the share from every sample where the cache is the binding
constraint (usage at or above 90%) and reports the range. It passes when the ceiling
held still and warns when it moved.

In the live run (32 requests, 1,000 output tokens each, 784-token blocks, 42 usable
blocks), a fresh request held 4.0 blocks and 10 fit. Requests grew to 6.0 blocks as they
crossed block boundaries, and the ceiling fell to 7. The preemption counter stepped at
each fall, 22 times in all. Running never exceeded the ceiling computed from the same
sample (`tests/test_live.py`). On fixed-length requests (the ladder fixture) the
ceiling is 4 on every saturated sample, and 4 is exactly the peak T8 measured.

## T10: integer-second timers are worth several percent

**Symptom.** `wall=$(( $(date +%s) - t0 ))` is a difference of two integer-second
stamps, so it lands within ±1 s of the true duration. On a 29-second run that is ±3.5%,
and it lands independently on each arm of a comparison.

**Check.** Tokens counted between two snapshots divided by the time between those same
snapshots, from a monotonic clock, with the scrape latency recorded as the timing
uncertainty. The integer-second figure is shown alongside for comparison.

In the live run the server's generation counter moved by exactly 32,000 tokens (32
requests of 1,000) over 93.720 s monotonic, with 0.007 s of scrape uncertainty: 341.44
tokens/s. An integer-second timer read 344.09 (+0.77%). The error grows as runs shorten.

## T11: KV memory differs between boots of one config

CUDA-graph capture and allocator state vary between boots. A small difference in the
memory left for KV cache can move the pool down a whole level (T4). `check-log`
reports `Available KV cache memory` and CUDA-graph memory per boot, grouped by config.

## T12: boot failures recorded without a reason

**Symptom.** A failure grep for `ValueError|RuntimeError` misses
`torch.AcceleratorError: CUDA error: device not ready` and records a blank.

**Check.** An ordered classifier: specific root causes first (JIT toolchain, OOM, KV
cache too small, accelerator and CUDA errors), generic exception types after,
"engine dead" last because it follows every root cause and explains none. Tracebacks
after an orderly shutdown (vLLM's `[shutdown]` lines, as when a teardown stops the
server) are not failures and are ignored.

## T13: a killed campaign's waiter adopts the next server

A shell `trap` does not run while the script is blocked in a child. A campaign killed
while waiting for readiness can see the *replacement* campaign's server on the same port,
report it ready, and then run its deferred cleanup against it. See `orchestration.md`.

## T14: the usage gauge's denominator is `num_gpu_blocks - 1`

**Finding.** `vllm:kv_cache_usage_perc` readings are exact multiples of `1/N`, and N is
recoverable from the readings alone. It is always one less than the `num_gpu_blocks`
the server exports in `vllm:cache_config_info`. One block is reserved as a null block
and never holds a request's KV. On a pool of 45 large blocks, that is 2.2% of the
advertised capacity.

**Check.** Infers N from the gauge (least common denominator of the readings, trusted
only with at least five distinct values) and compares it with the exported count. A
change in either side, or a series and snapshot from different boots, turns it red.

```console
$ inferlint series tests/fixtures/vllm-0.28/series/blocks.series.jsonl \
      --snapshot tests/fixtures/vllm-0.28/series/blocks_snapshot.prom
[T14] PASS  gauge denominator is 44 = num_gpu_blocks (45) - 1 null block
```
