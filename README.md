# inferlint

[![CI](https://github.com/radianvector/inferlint/actions/workflows/ci.yml/badge.svg)](https://github.com/radianvector/inferlint/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/inferlint)](https://pypi.org/project/inferlint/)
[![Python](https://img.shields.io/pypi/pyversions/inferlint)](https://pypi.org/project/inferlint/)
[![License](https://img.shields.io/github/license/radianvector/inferlint)](https://github.com/radianvector/inferlint/blob/main/LICENSE)

**inferlint measures what your inference server actually did—and flags what it didn't
tell you.**

For vLLM, SGLang and TensorRT-LLM. Measure achieved concurrency, preemptions, KV-cache
behavior, GPU state, timing quality, and other signals that can silently invalidate
inference results.

A serving benchmark can be wrong without an error. vLLM preempts a running request and
writes no log line; only a counter moves. TensorRT-LLM pauses requests for recompute and
counts none. SGLang can lower its running limit at start-up and still report the limit
it was given. A config can start cleanly and fail on its first request. Two starts with
the same flags can size different KV pools. Killing the API server can leave the engine
process holding the GPU. None of these raise an error, and each changes the numbers.

inferlint runs around a benchmark. It reads the server's boot log and Prometheus
counters, measures what happened, and flags each of these problems when it occurs. It
tells the engine from the server command, its metrics and its log. Your load tool (each
engine's own benchmark client, or any other) sends the requests; inferlint checks what
the server did with them.

[Tutorial](https://radianvector.github.io/inferlint/tutorial.html) ·
[Example report](https://radianvector.github.io/inferlint/example-report.html) ·
[Reference](https://github.com/radianvector/inferlint/blob/main/docs/tripwires.md) ·
[Changelog](https://github.com/radianvector/inferlint/blob/main/CHANGELOG.md)

## Try it in 30 seconds, without a GPU

The repository includes the files from a recorded run:

```bash
pipx install inferlint
git clone --depth 1 https://github.com/radianvector/inferlint && cd inferlint
inferlint series tests/fixtures/vllm-0.28/live/run.series.jsonl \
    --snapshot tests/fixtures/vllm-0.28/live/after.snapshot.json --requested 32
```

```console
[T14] PASS  gauge denominator is 42 = num_gpu_blocks (43) - 1 null block
[ T9] WARN  ceiling moved between 7 and 10 as requests grew (one request held 9.5% to 14.3% of the cache); concurrency was not a constant of this run
[ T8] TRIPWIRE-FAILED  requested 32 concurrent, server never ran more than 10
```

That run is vLLM 0.28. Recorded SGLang and TensorRT-LLM runs are in
`tests/fixtures/sglang-0.5/` and `tests/fixtures/trtllm-1.3/`; the same commands read them.

## Check your own run with one command

The same load, 32 requests at once with 88 prompt and 1,000 output tokens each, sent by
each engine's own benchmark client:

```bash
# vLLM
inferlint xray -o run/ --serve "vllm serve MODEL --max-model-len 16384" \
    -- vllm bench serve --model MODEL --dataset-name random --random-input-len 88 \
       --random-output-len 1000 --ignore-eos --num-prompts 32 --max-concurrency 32

# SGLang: its metrics need --enable-metrics
inferlint xray -o run/ --serve "sglang serve --model-path MODEL --enable-metrics" \
    -- python -m sglang.benchmark.serving --model MODEL --dataset-name random-ids \
       --random-input-len 88 --random-output-len 1000 --random-range-ratio 1 \
       --num-prompts 32 --max-concurrency 32

# TensorRT-LLM: its metrics need two settings in the --config file
printf 'return_perf_metrics: true\nenable_iter_perf_stats: true\n' > llm_api.yaml
inferlint xray -o run/ --serve "trtllm-serve serve MODEL --config llm_api.yaml" \
    -- python -m tensorrt_llm.serve.scripts.benchmark_serving --model MODEL \
       --dataset-name random --random-ids --random-input-len 88 --random-output-len 1000 \
       --ignore-eos --num-prompts 32 --max-concurrency 32
```

Output from the vLLM 0.28 run described below, on an RTX 4090 (paths and the load
command shortened, the benchmark's own output left out):

```console
== is the GPU free?
[ T2] PASS  2 consecutive clear readings
== start the server (output in run/boot.log)
== boot log
[T12] PASS  no failure in log
[ T5] WARN  attention block size forced to 784 tokens (requested 16); each request's KV is allocated in 784-token units
[ T6] PASS  nothing to check: the server was started without --attention-backend, so vLLM chose its own (FLASH_ATTN) and nothing could be ignored; there was no draft model either
== probe: one request, then 8 at once
[ T7] PASS  1 + 8 concurrent requests served, server alive
== load: vllm bench serve --model MODEL ... --max-concurrency 32 --save-result --result-dir run --result-filename bench.json
== checks
[ T1] TRIPWIRE-FAILED  21 preemptions during the run
[T10] PASS  314.99 tokens/s over 101.590s; an integer-second timer would say 313.73 (-0.40%)
[T14] PASS  gauge denominator is 42 = num_gpu_blocks (43) - 1 null block
[ T9] WARN  ceiling moved between 5 and 10 as requests grew (one request held 9.5% to 19.0% of the cache); concurrency was not a constant of this run
[ T8] TRIPWIRE-FAILED  requested 32 concurrent, server never ran more than 10
[T15] PASS  client and server agree: 32 requests, 32,000 output tokens
== stop the server
[ T2] PASS  2 consecutive clear readings
== what limited this run
- The server ran fewer requests at once than the load sent. The load kept 32 requests open, but the server ran at most 10 at once. It also preempted 21 running requests.
- Requests waited long to start. Half the requests waited more than 28.9 s for their first token, the slowest (p99) 73.4 s, most of it in the queue.
- The run reached 26% of its pace. A running request got a token every 25.1 ms, so 32 at once could produce about 1,273 tokens/s. The run averaged 336, 26% of that: on average 8 requests ran at once, not 32.
12 checks (8 pass, 2 warn, 2 tripwire-failed)
report: run/report.html
files:  run
```

`xray` checks that the GPU is free, starts the server and saves its boot log, sends 9
probe requests, records the server's counters and gauges around the load command after
`--`, runs every tripwire that applies, says what limited the run, and writes
`run/report.html` and `run/results.json`. It always stops the server at the end, also
after a failure or Ctrl-C. It finds the server at the `--host` and `--port` of the
`--serve` command, or the engine's default port (8000; SGLang 30000). Without `--serve`
it attaches to a server that is already running (`--url`, `--boot-log`). Exit status: 0
when every tripwire passed, 1 when one failed or the run did not complete, 2 when one
could not decide.

For the three engines' benchmark clients (`vllm bench serve`,
`python -m sglang.benchmark.serving`, `python -m tensorrt_llm.serve.scripts.benchmark_serving`),
`xray` asks the client to save its result in the run folder, compares the client's own
counts with the server's (T15), and reads `--max-concurrency` as the concurrency asked
for (T8). Any other load tool runs as it is.

To check the folder again later, on any computer: `inferlint report run/`. It finds the
files by name, prints the verdicts, and rewrites `run/report.html`.

**One engine for the whole team.** inferlint tells the engine from what it reads, so no
setting is needed. A team that always runs the same engine can name it once:

```bash
export INFERLINT_ENGINE=sglang     # or --engine sglang on any command
```

An input from another engine is then refused with exit status 2 (a snapshot from a
server on the wrong port, a file from another run), and `teardown` stops only that
engine's servers. [docs/engines.md](https://github.com/radianvector/inferlint/blob/main/docs/engines.md)
lists what each engine needs and where the tripwires differ.

## One run, measured

vLLM 0.28 on an RTX 4090, a 27B hybrid attention/Mamba model with 4-bit weights and
CUDA graphs on. 32 concurrent requests, 1,000 output tokens each:

```console
$ inferlint preemption before.json after.json
[ T1] TRIPWIRE-FAILED  22 preemptions during the run
$ grep -ci preempt boot.log
0
$ inferlint series run.series.jsonl --snapshot after.json --requested 32
[T14] PASS  gauge denominator is 42 = num_gpu_blocks (43) - 1 null block
[ T9] WARN  ceiling moved between 7 and 10 as requests grew (one request held 9.5% to 14.3% of the cache); concurrency was not a constant of this run
[ T8] TRIPWIRE-FAILED  requested 32 concurrent, server never ran more than 10
```

- The client held 32 requests in flight. The server never ran more than 10 at once.
- A fresh request held 4 KV blocks, so 10 fit. As outputs crossed the 784-token block
  boundary, running requests held up to 6 blocks each on average, and the number that
  fit fell as low as 7. Each fall evicted running requests, 22 times in all, and the
  server log says nothing about it.
- The server exports 43 KV blocks, but the usage gauge counts in 42nds. One block is
  reserved and never holds a request.

Then the API server was killed with SIGKILL, as a crash or the OOM killer would:

```console
$ pkill -9 -f "vllm[ ]serve"; sleep 20
$ ps -eo pid,ppid,rss,args | grep -E "[V]LLM::|[v]llm serve"
   1030     423 2733144 VLLM::EngineCore
$ inferlint gpu-inspect          # the GPU's model and memory lines are left out here
[ T2] TRIPWIRE-FAILED  card not clear; refusing to boot. ... 21768 MiB used, 1 server processes alive
$ inferlint teardown
target  pid=1030     VLLM::EngineCore
(pkill -f 'vllm serve' would miss 1 and wrongly hit 0)
[ T2] PASS  2 consecutive clear readings
```

The engine outlived its parent and kept 21.3 GiB of the card. The usual cleanup pattern
cannot see it, and the next start would have loaded a second copy into what was left.
The same run on vLLM 0.29 and 0.30 gave the same findings (below). The `xray` output
above is a second run on 0.28: 21 preemptions instead of 22, and a lowest ceiling of 5
instead of 7, with the same tripwires failing and warning. The raw files are in
[`tests/fixtures/`](https://github.com/radianvector/inferlint/tree/main/tests/fixtures),
and `tests/test_live.py` and `tests/test_compat.py` pin every number.

## One model, three engines

Qwen3-8B in bf16 on the same RTX 4090, 16,384-token context, at most 32 running, and the
same load sent by `vllm bench serve` to each engine:

| | vLLM 0.30 | SGLang 0.5.20 | TensorRT-LLM 1.3.0rc29 |
|---|---|---|---|
| KV pool | 32,336 tokens (47,888 on a second start) | 32,096 tokens | 37,760 tokens |
| Most running at once (T8) | 32 | 29 | 32 |
| Preempted (T1) | 3 preemptions, no log line | 0 retractions | 0 pauses in its log |
| Output tokens per second | 1,274 | 799 | 1,389 |
| Client and server counts (T15) | agree | agree | agree |

Each engine's own benchmark client, on the same servers: TensorRT-LLM's gave the same
result (32 at once, no pauses, 1,405 tokens per second). SGLang's, which sends requests
to SGLang's native `/generate` endpoint, ran all 32 at once with 3 retractions, at 1,337
to 1,342 tokens per second in three runs, against 799 and 800 in two runs with
`vllm bench serve`; whether the endpoint or the client makes the difference has not
been tested. With SGLang's defaults for the running limit (2048) and context length,
its own client reached 821. The two vLLM starts had the same flags; the first compiled the model from scratch and left
4.44 GiB for the KV cache instead of 6.58 GiB (T4, T11). [docs/engines.md](https://github.com/radianvector/inferlint/blob/main/docs/engines.md)
has the details.

## Tested versions

vLLM 0.28, 0.29 and 0.30, SGLang 0.5.20 and TensorRT-LLM 1.3.0rc29 (a release
candidate), each live on an RTX 4090 (WSL2, driver 616.56).

### vLLM

Each tripwire on each tested release, with the 27B model and load above:

| tripwire | what the run showed | 0.28 (two runs) | 0.29 | 0.30 |
|---|---|---|---|---|
| T1 silent-preemption | preemptions; none in the log | live: 22 / 21 | live: 21 | live: 22 |
| T2 orphaned-engine | MiB the engine kept after the API server was killed | live: 21,768 | live: 21,522 | live: 21,872 |
| T4 pool-size-jump | KV pool, starts with the same flags | recorded: a 9.4% jump | live: same in 2 starts | live: same in 2 starts |
| T5 large-kv-blocks | attention block size, tokens | live: 784 | live: 784 | live: 784 |
| T6 backend-ignored | requested backend not applied to the drafter | recorded | not tested | not tested |
| T7 crash-after-ready | probe after ready | live (healthy); a crash recorded | live (healthy) | live (healthy) |
| T8 concurrency-not-reached | most requests running at once, of 32 | live: 10 / 10 | live: 10 | live: 11; admission control |
| T9 falling-ceiling | concurrency ceiling, highest → lowest | live: 10 → 7 / 10 → 5 | live: 10 → 5 | live: 11 → 7 |
| T10 integer-second-timer | rate from exact timestamps | live | live | live |
| T11 kv-memory-drift | KV memory, starts with the same flags | recorded: drift | live: same in 2 starts | live: same in 2 starts |
| T12 unexplained-failure | real start-up failures named | live: 1; recorded: 1 | live: 1 | live: 3 |
| T14 null-block | usable of exported KV blocks | live: 42 of 43 | live: 42 of 43 | live: 47 of 48 |
| T15 client-server-mismatch | load tool's counts against the server's | live: agree | live: agree | live: agree; also with 16 rejected |

T3 and T13 concern shell scripts, not an engine, and do not depend on its version. The
counts differ a little from run to run: 0.28 shows both of its runs.

### SGLang and TensorRT-LLM

Each tripwire with Qwen3-8B (bf16) and the load above, and with the 27B model where it
loads (TensorRT-LLM fails to read its weight-only 4-bit checkpoint, and T12 names why):

| tripwire | what the run showed | SGLang 0.5.20 | TensorRT-LLM 1.3.0rc29 |
|---|---|---|---|
| T1 silent-preemption | requests preempted (SGLang: retracted; TensorRT-LLM: paused) | live: 1 retraction with `--schedule-conservativeness 0.3`; 3 with SGLang's own benchmark client; none with `vllm bench serve` | live: 13 pauses, counted from its log, with `MAX_UTILIZATION`; none with the default policy |
| T2 orphaned-engine | engine processes after the front end was killed | live: none; its workers ended too | live: none; its MPI workers ended too |
| T4 pool-size-jump | KV pool, starts with the same flags | live: same in 2 starts | live: same in 2 starts |
| T5 large-kv-blocks | KV allocation unit, tokens | live: 1 | live: 32 |
| T6 backend-ignored | requested backend not applied to the drafter | not tested | not tested |
| T7 crash-after-ready | probe after ready | live (healthy); an error it survived is not a crash | live (healthy) |
| T8 concurrency-not-reached | most requests running at once, of 32 | live: 29; 1 with the 27B model, where it lowered its own limit | live: 32 |
| T9 falling-ceiling | concurrency ceiling, highest → lowest | live: 31 → 30 | live: 35, from 2 readings |
| T10 integer-second-timer | rate from exact timestamps | live | live |
| T11 kv-memory-drift | KV memory, starts with the same flags | live: same in 2 starts | live: same in 2 starts |
| T12 unexplained-failure | real start-up failures named | live: 2 | live: 2 |
| T14 null-block | reserved KV block | does not apply | does not apply |
| T15 client-server-mismatch | load tool's counts against the server's | live: agree, with `vllm bench serve` and with SGLang's own client | live: agree, with `vllm bench serve` and with TensorRT-LLM's own client |

**live**: measured on a running server of that version. **recorded**: tested on recorded
files from that version. A boot log from a release not listed here makes inferlint print
a warning (SGLang's version is read from the server), since log lines and metric names
change between releases.

Setting up the vLLM runs on WSL with pip-installed CUDA hit five start-up failures, and T12
named the cause of each (details in [docs/tripwires.md](https://github.com/radianvector/inferlint/blob/main/docs/tripwires.md#t12-start-up-failure-without-a-cause)).
The fixes: `VLLM_USE_V2_MODEL_RUNNER=0` on 0.29; on 0.29 and 0.30, a CUDA 13.0 compiler
matching torch's CUDA runtime for FlashInfer's kernel builds; on 0.28,
`VLLM_USE_FLASHINFER_SAMPLER=0`, as in the recorded run.

## What it checks

| id | name | trap | command |
|---|---|---|---|
| T1 | silent-preemption | a preempted request leaves no log line (vLLM) or no count (TensorRT-LLM) | `inferlint preemption BEFORE AFTER` |
| T2 | orphaned-engine | `pkill -f "vllm serve"` misses the renamed `VLLM::EngineCore`; engines run in processes the server command does not name | `inferlint teardown`, `inferlint gpu-inspect` |
| T3 | cleanup-self-kill | `pkill -f` in a shell one-liner kills the shell | `inferlint teardown` |
| T4 | pool-size-jump | the KV pool is sized per start, in discrete steps | `inferlint check-log BOOT...` |
| T5 | large-kv-blocks | on vLLM, hybrid models force a large attention block (the allocation unit) | `inferlint check-log` |
| T6 | backend-ignored | the requested attention backend is not applied to the speculative drafter | `inferlint check-log` |
| T7 | crash-after-ready | a config starts, then dies on its first request | `inferlint probe URL`, `inferlint check-log` |
| T8 | concurrency-not-reached | requested concurrency is not achieved concurrency | `inferlint series --requested N` |
| T9 | falling-ceiling | the concurrency ceiling is `floor(1 / share)`, and it falls as requests grow | `inferlint series` |
| T10 | integer-second-timer | integer-second timers are off by up to a second at each end | `inferlint rate BEFORE AFTER` |
| T11 | kv-memory-drift | KV memory differs between starts with the same flags | `inferlint check-log BOOT...` |
| T12 | unexplained-failure | start-up failures recorded without a cause | `inferlint check-log` |
| T13 | stale-waiter | a killed campaign's waiter adopts the next server | [docs/orchestration.md](https://github.com/radianvector/inferlint/blob/main/docs/orchestration.md) |
| T14 | null-block | vLLM's KV usage gauge counts out of `num_gpu_blocks - 1` | `inferlint series --snapshot` |
| T15 | client-server-mismatch | the load tool's counts differ from the server's | `inferlint xray` |

Each is described in [docs/tripwires.md](https://github.com/radianvector/inferlint/blob/main/docs/tripwires.md)
with its symptom, why it changes results, and a command that reproduces it from files in
this repository, and what differs on each engine. `inferlint explain T9` (or
`inferlint explain falling-ceiling`) prints the same in the terminal; `--engine sglang`
shows only that engine's notes.

## Install

```console
pipx install inferlint
inferlint --help
```

[pipx](https://pipx.pypa.io) gives `inferlint` an environment of its own and puts the
command on your PATH, so it works in any terminal and from any folder, with nothing to
activate. `uv tool install inferlint` does the same. To use it as a library, install it
into your project's environment with `pip install inferlint` (see
[Use it from Python](#use-it-from-python)).

Python 3.10+, no dependencies. `teardown`, `xray --serve`, and the free-GPU check in
`gpu-inspect` find the server's processes through `/proc`, so they need Linux (or WSL),
where the engines run. Everything else runs anywhere.

## Use the commands one at a time

`xray` runs the sequence below for you. The separate commands are for scripts that need
to control each step. A test with inferlint takes as long as your benchmark. It checks
one test at a time; it is not a monitor for a server that runs for days (Prometheus and
Grafana do that job).

Use two terminals on the GPU machine, in the same folder: **A** runs the server, **B**
runs everything else.

```bash
# 1. Before the server starts (B)
inferlint gpu-inspect                  # which GPU, its memory; fails if an old server still holds it

# 2. Start the server (A): your usual command, saving its start-up output
vllm serve MODEL 2>&1 | tee boot.log   # wait for "Application startup complete"
#    or: sglang serve --model-path MODEL --enable-metrics 2>&1 | tee boot.log
#    or: trtllm-serve serve MODEL --config llm_api.yaml 2>&1 | tee boot.log

#    then check it (B)
inferlint check-log boot.log           # block size, attention backend, start-up failures
inferlint probe http://127.0.0.1:8000  # sends 9 small requests: does it answer?

# 3. The test (B)
inferlint snapshot http://127.0.0.1:8000 -o before.json
inferlint watch http://127.0.0.1:8000 -o run.series.jsonl --interval 0.25 &
vllm bench serve --model MODEL --max-concurrency 32 ...   # your benchmark sends the load
inferlint snapshot http://127.0.0.1:8000 -o after.json
kill %1                                # stops the recording

# 4. The verdict: one HTML page (any time later, on any computer)
inferlint report -o run.html --requested 32 --boot-log boot.log \
    --before before.json --after after.json --series run.series.jsonl

# 5. Stop the server (B)
inferlint teardown                     # stops every server process, waits until the GPU is free
```

- **Your benchmark** sends the requests and reports speed and latency. inferlint reads
  the server's side and says which of those numbers hold.
- **`--requested`** is the concurrency your benchmark asked for (`--max-concurrency` in
  all three engines' benchmark clients). Requests, tokens and preemptions come from the
  server's counters.
- **SGLang and TensorRT-LLM** serve metrics only when asked: `--enable-metrics`, and
  `return_perf_metrics: true` with `enable_iter_perf_stats: true` (without it, no
  running, waiting or KV gauges) in the file given to `trtllm-serve --config`. SGLang
  listens on port 30000 unless given `--port`, so its URL is `http://127.0.0.1:30000`.
  TensorRT-LLM counts no pauses and logs each one, so give `preemption` and `report` its
  log saved until after the run (`--boot-log boot.log`).
- **Each command does one job and exits**, except `watch`. They only read, except
  `probe` (9 small requests) and `teardown`, which stops the server: don't run it
  against a server that is serving users.
- **The same verdicts in the terminal:** `inferlint preemption before.json after.json`
  (T1), `inferlint rate before.json after.json` (T10),
  `inferlint series run.series.jsonl --snapshot after.json --requested 32` (T8, T9, T14).
- **T4 and T11 compare start-ups:** keep the log of each start with the same settings and
  pass them all (`--boot-log boot1.log --boot-log boot2.log`).
- **In scripts,** `inferlint gpu-inspect || exit 1` and
  `inferlint probe http://127.0.0.1:8000 || exit 1` stop a run before it produces bad
  numbers. [docs/orchestration.md](https://github.com/radianvector/inferlint/blob/main/docs/orchestration.md)
  covers scripts that run many tests.

A failed tripwire prints as `TRIPWIRE-FAILED`: the tripwire caught its problem. For T7
and T12 that means the server failed; for every other tripwire the server worked, but a
number from the run does not mean what it seems. Exit status is 0 when every tripwire
passes (warnings allowed), 1 when one fails, and 2 when one cannot decide because a
series or a log line was missing. A missing series is never read as zero. A tripwire
about a behaviour the engine does not have prints `N/A` and says so (T14, vLLM's reserved
KV block, on SGLang and TensorRT-LLM); it does not change the exit status. `--json` gives
machine-readable results (`"status": "fail"` for a failed tripwire, `"not_applicable"`
for one that does not apply); `--strict` fails on warnings.

### The report

`inferlint report` turns a run's files into one HTML page. It opens with what limited
the run, in up to three sentences: whether the KV cache could hold the load, whether
requests waited to start, and how much of the throughput its per-token speed allows the
run reached. Then a plain-English account of what happened (how long the test ran,
requests finished and failed, tokens generated, time spent queued, which tripwires were
checked and which need other files), the verdicts, with passed checks folded under one
line, charts of what the server did over time and of its KV cache against the load, the
boot facts and GPU memory, and a guide to every tripwire. Collapsible glossaries explain
the terms (token, KV pool, block, preemption) and what Pass, Warning, Tripwire failed,
Can't tell and Does not apply mean.

The page fetches nothing, so it opens offline and can be attached to a ticket as it is.
It has a light/dark switch, every chart has a data table, and hovering (or the arrow
keys) reads values at any moment.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/radianvector/inferlint/main/docs/report-dark.png">
  <img alt="The report for the live run: what limited it, what happened, the headline numbers, 2 failed tripwires and 2 warnings" src="https://raw.githubusercontent.com/radianvector/inferlint/main/docs/report.png">
</picture>

**See a full example.**
[The complete report for the run above](https://radianvector.github.io/inferlint/example-report.html)
has the charts, the findings and both glossaries. The same file is in this repository as
`docs/example-report.html`, and it works offline.

**Try it without a GPU.** The example is built from the recorded run in this repository.
Download the repository for its files, and make the report yourself:

```bash
git clone https://github.com/radianvector/inferlint
cd inferlint
inferlint report tests/fixtures/vllm-0.28/live --requested 32 \
    --title "vLLM 0.28 on an RTX 4090" -o example-report.html
```

## Use it from Python

Install it into your project's environment with `pip install inferlint`. Every check
returns a `CheckResult` with a `status` (pass, warn, fail, unknown or not_applicable),
a `message`, and
the `evidence` behind it.

```python
import subprocess

from inferlint import bootlog, checks, telemetry

url = "http://127.0.0.1:8000"
before = telemetry.scrape(url)
subprocess.run(["vllm", "bench", "serve", "--model", "MODEL"], check=True)  # your benchmark
after = telemetry.scrape(url)

result = checks.check_no_preemption(before, after)
print(result.status.value, result.message)  # fail 22 preemptions during the run
result.raise_for_status()  # raises TripwireFailed on a failed tripwire or a Can't tell

facts = bootlog.parse_file("boot.log")
print(facts.attention_block_size, facts.kv_pool_tokens)  # 784 26093
record = {"boot": facts.result_block()}  # pool, block size, backend: keep with your results
```

`raise_for_status()` lets warnings through; `raise_for_status(allow_warn=False)` stops on
them too.

| module | what it gives you |
|---|---|
| `bootlog` | the facts in a start-up log (`parse_file`) |
| `telemetry` | the server's counters (`scrape`, `load`) |
| `series` | recordings made during a run (`watch`, `read`) |
| `checks` | the tripwires, each returning a `CheckResult` |
| `catalog` | each tripwire's code, name and explanation (`TRIPWIRES`, `lookup`) |
| `probe` | the first-request check (T7) |
| `gpu` | which GPU, its family and memory (`read_gpus`) |
| `teardown` | stopping the server and waiting for a free GPU (Linux) |
| `report` | the HTML report (`build`, `render`) |
| `benchresult` | a benchmark client's result file: vLLM's, SGLang's or TensorRT-LLM's (`load`) |
| `xray` | the whole sequence in one call (`run`, `Plan`) |
| `engines` | the three engines and how each is told apart (`from_command`, `from_log`) |
| `metricnames` | every metric name inferlint reads (`VLLM`, `SGLANG`, `TRTLLM`) |

## How it is tested

- **Real evidence.** Fixtures are boot logs, metrics and benchmark results from vLLM
  0.28, 0.29 and 0.30, SGLang 0.5.20 and TensorRT-LLM 1.3.0rc29 on an RTX 4090, with
  paths removed. Each check is tested against the case it exists for, and
  `tests/test_every_engine.py` runs every command on a run from each engine.
- **Every check is watched failing.** `tests/test_mutations.py` changes one number or
  line in real evidence from each engine and requires the verdict to flip.
- **Instruments against ground truth.** The probe runs against a mock server with known
  failure modes. Block-count inference (T14) is checked against the count the server
  exports. The teardown's process matching runs on a synthetic `/proc` that includes the
  shell one-liner and a bystander. GPU facts are parsed from real `nvidia-smi` output.
- **Unknown formats are reported, not defaulted.** When a known log line changes format
  in a new release, `boot-facts` lists it as unparsed instead of returning a value, and
  a boot log from an untested release prints a warning.
- **`xray` against a mock server.** The whole sequence runs in the tests against a mock
  server of each engine with known counters, including other traffic (T15) and a server
  that dies during start-up.

## Contribute

The most useful contribution is a recorded run from an engine version or GPU that is not
in the tables above: `inferlint xray -o run/ ...` saves every file needed.
[CONTRIBUTING.md](https://github.com/radianvector/inferlint/blob/main/CONTRIBUTING.md)
says how to share it, how to set up the code, and what a new tripwire needs. GitHub
Actions runs the tests, lint and type checks on every push and pull request, on Python
3.10 to 3.13. This tests inferlint's own code; it is not a GitHub Action for your
benchmarks. Security problems: see
[SECURITY.md](https://github.com/radianvector/inferlint/blob/main/SECURITY.md).

## Status

Alpha. Tested live on vLLM 0.28, 0.29 and 0.30, SGLang 0.5.20 and TensorRT-LLM
1.3.0rc29, on one GPU model (RTX 4090). Next: more GPUs and models from contributed
runs.

## About

`inferlint` is made by [RadianVector](https://radianvector.com). RadianVector develops
tools and research for evaluating and improving AI systems.

## License

Copyright 2026 RadianVector.

This project, including the recorded evidence in `tests/fixtures/`, is licensed under the
Apache License, Version 2.0. See [LICENSE](https://github.com/radianvector/inferlint/blob/main/LICENSE)
for details.
