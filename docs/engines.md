# vLLM, SGLang and TensorRT-LLM

inferlint reads all three engines. It tells the engine from the server command, its
metric names and its boot log, so the commands are the same for all three, and
`inferlint xray` starts, records, checks and stops any of them.

Tested live on vLLM 0.28, 0.29 and 0.30, SGLang 0.5.20 and TensorRT-LLM 1.3.0rc29 (a
release candidate), all on the same RTX 4090 (WSL2, driver 616.56). Other versions get
an untested-version warning where inferlint can read the version: from the boot log for
vLLM and TensorRT-LLM, and from the server for SGLang, which `xray` asks.

## Naming the engine

inferlint tells the engine from what it reads, so no flag is needed. A team that runs one
engine can name it once:

```bash
export INFERLINT_ENGINE=sglang     # or --engine sglang on any command
```

With an engine named:

- an input from another engine is refused with exit status 2, for example a snapshot
  taken from a vLLM server on the wrong port;
- an input that names no engine (a boot log cut short) is read as the named engine's,
  instead of as vLLM's;
- `teardown` stops only that engine's servers, and `explain` shows only its notes.

`--engine` takes precedence over the variable. `xray --serve` reads the engine from the
server command before it starts anything; when the server serves no metrics, it stops
and says what to change for that engine.

## What each engine needs

- **vLLM**: nothing; it serves its metrics at `/metrics`.
- **SGLang**: start it with `--enable-metrics`. It listens on port 30000 unless given
  `--port`; `xray` reads the port from the `--serve` command, else uses the engine's
  default.
- **TensorRT-LLM**: put `return_perf_metrics: true` and `enable_iter_perf_stats: true`
  in the YAML file given to `trtllm-serve --config`. inferlint then reads Prometheus text
  from `/prometheus/metrics`; `/metrics` returns JSON. With `return_perf_metrics` alone,
  TensorRT-LLM serves its counters but no running, waiting or KV gauges, and T8 and T9
  say "can't tell" and name the missing setting (recorded in
  `tests/fixtures/trtllm-1.3/no-iter-stats/`). Keep its log at INFO, the default: T1
  counts the pauses it logs there, so `preemption` and `report` need that log saved
  until after the run (`--boot-log`); `xray` saves it.

## Each engine's benchmark client

`xray` recognises each engine's own benchmark client, asks it to save its result in the
run folder, and compares the client's counts with the server's (T15):

| client | result file | requests it leaves out of its result |
|---|---|---|
| `vllm bench serve` | `--save-result`: one JSON document | an initial test request, and `--num-warmups` |
| `python -m sglang.benchmark.serving` (or `sglang.bench_serving`) | `--output-file`: one JSON line appended per run | `--warmup-requests`, 1 by default |
| `python -m tensorrt_llm.serve.scripts.benchmark_serving` | `--save-result`: vLLM's format | an initial test request |

All three use vLLM's field names and print the requests they leave out, which `xray`
reads from their output. `inferlint report RUN_FOLDER` finds `bench.json` or
`bench.jsonl` in the folder. Any load tool works; the others run as they are, without
T15.

```bash
inferlint xray -o run/ --serve "sglang serve --model-path MODEL --enable-metrics" \
    -- python -m sglang.benchmark.serving --model MODEL --dataset-name random-ids \
       --random-input-len 88 --random-output-len 1000 --random-range-ratio 1 \
       --num-prompts 32 --max-concurrency 32

printf 'return_perf_metrics: true\nenable_iter_perf_stats: true\n' > llm_api.yaml
inferlint xray -o run/ --serve "trtllm-serve serve MODEL --config llm_api.yaml" \
    -- python -m tensorrt_llm.serve.scripts.benchmark_serving --model MODEL \
       --dataset-name random --random-ids --random-input-len 88 --random-output-len 1000 \
       --ignore-eos --num-prompts 32 --max-concurrency 32
```

## One model, three engines

Qwen3-8B in bf16, 16,384-token context, at most 32 running, and the same load as in the
README: `vllm bench serve`, 32 requests at once, 88 prompt and 1,000 output tokens each.

| | vLLM 0.30, start 1 | vLLM 0.30, start 2 | SGLang 0.5.20 | TensorRT-LLM 1.3.0rc29 |
|---|---|---|---|---|
| KV pool | 32,336 tokens | 47,888 tokens | 32,096 tokens | 37,760 tokens |
| Most running at once (T8) | 32 | 32 | 29 | 32 |
| Preemptions (T1) | 3, no log line | 0 | 0 retractions | 0 pauses in its log |
| Output tokens per second | 1,274 | 1,378 | 799 | 1,389 |
| Engine left on the GPU after SIGKILL of the front end (T2) | 22,034 MiB | not tested | none | none |
| Client and server counts (T15) | agree | agree | agree | agree |

The two vLLM starts had the same flags. The first compiled the model from scratch, and
vLLM counted 1.14 GiB more for weights and non-torch memory and a 1 GiB higher activation
peak, which left 4.44 GiB for the KV cache instead of 6.58 GiB (T4, T11). SGLang's two
starts drew the same pool, as did TensorRT-LLM's.

**With each engine's own client.** The same servers and load, sent by SGLang's and
TensorRT-LLM's own benchmark clients through `xray`:

| | SGLang 0.5.20, `sglang.benchmark.serving` | TensorRT-LLM 1.3.0rc29, `benchmark_serving` |
|---|---|---|
| Most running at once (T8) | 32 | 32 |
| Preempted (T1) | 3 retractions | 0 pauses in its log |
| Output tokens per second (the client's count) | 1,342 (1,337 and 1,340 in two repeats) | 1,405 |
| Time to first token, median | 304 ms | 371 ms |
| Client and server counts (T15) | agree, plus its 1 warm-up request | agree, plus its 1 test request |

TensorRT-LLM gave the same result under both clients. SGLang ran more at once under its
own client, which sends requests to SGLang's native `/generate` endpoint rather than
`/v1/completions`, and generated 1.7 times as many tokens per second: 1,337 to 1,342 in
three runs, against 799 and 800 in two runs with `vllm bench serve`. Whether the
endpoint or the client makes the difference has not been tested. The server settings
matter as much: with SGLang's defaults for the running limit (2048) and context length,
and the same memory fraction, its own client ran at most 31 at once and reached 821.

**SGLang's default memory fraction.** Started without `--mem-fraction-static`, SGLang
0.5.20 chose 0.704 on this card (against 0.88 in the runs above) and sized a 3,411-token
KV pool. With the same load, 3 requests fit at full length: it ran at most 5 at once and
3.7 on average, retracted 8, and its client measured 203 output tokens/s. inferlint
reported each of these (T1, T8, and what limited the run).

The files are in `tests/fixtures/vllm-0.30/qwen3-8b-start1/`, `qwen3-8b-start2/`,
`tests/fixtures/sglang-0.5/qwen3-8b/`, `qwen3-8b-start2/` and `qwen3-8b-own-client/`, and
`tests/fixtures/trtllm-1.3/qwen3-8b/`, `qwen3-8b-start2/` and `qwen3-8b-own-client/`.
`tests/test_every_engine.py` runs every command on them.

## How the tripwires differ

**T1.** SGLang calls a preemption a retraction. It counts it in
`sglang:num_retracted_requests_total`, which it exports only after the first one (inferlint
reads it as 0 until then), and logs a warning for each. TensorRT-LLM pauses a request
for recompute, counts none, and logs each pause at INFO ("request ID 72 -> pause");
inferlint counts those lines. Its default scheduler policy, GUARANTEED_NO_EVICT, admits
a request only when its whole output fits and pauses none. With MAX_UTILIZATION and a
20,960-token pool it paused 13 requests.

**T2.** In these tests, killing SGLang's launcher or TensorRT-LLM's `trtllm-serve` with
SIGKILL also ended their workers (`sglang::scheduler` and `sglang::detokenizer`; `prte`
and `python -m mpi4py.futures.server`), and the GPU was free. `inferlint teardown` finds
those workers, which `pkill -f` on the server command does not match.

**T5.** SGLang allocates KV per token (page size 1). For hybrid attention/Mamba models it
keeps a fixed state slot per running request instead, which T8 reports when it limits the
run. TensorRT-LLM allocates 32-token blocks.

**T8.** SGLang can lower its running limit below `--max-running-requests`. With
Qwen3.8-27B (a hybrid model) and `--max-running-requests 32 --mem-fraction-static 0.88`
it settled on 1, said so once in its log, and still reported 32 from
`/get_server_info`; the benchmark reported a peak of 32 concurrent requests. T8 names the
lowered limit when the run reached it.

**T8, T9 on TensorRT-LLM.** Its gauges (requests running, KV use, paused requests) are
updated only when a request completes. When requests finish together, a recording sees
little until the end; T8 says "can't tell" for a shortfall it may have missed, and T9
says how many readings it rests on.

**T10 and the report's tokens-per-second chart.** SGLang and TensorRT-LLM add a
request's generated tokens to their counter only when the request finishes: in a
32-request run the counter moved 3 times in 193 readings on SGLang and 3 times in 126 on
TensorRT-LLM, against 89 times in 137 on vLLM. A rate between two readings taken before and after the load (T10) is right; a rate
over a one-second window during the load shows completions, so the report leaves that
chart out for these two engines and says why.

**T9.** SGLang's usage gauge is its fullest memory pool, which for a hybrid model can be
the state slots rather than the KV cache.

**T12 and T7.** SGLang logs tracebacks it ignores ("Ignore import error") and errors it
survives ("post-warmup freeze_gc failed"); neither counts as a failure. Its log says
"Application startup complete" before it can serve, and `/health` answers 503 until it
can; `xray` waits for both.

**T14** is about vLLM's reserved null block and does not apply to the other engines.

## Not tested

Multi-GPU and multi-node setups, speculative decoding (T6) on SGLang and TensorRT-LLM,
other GPUs, and other engines.
