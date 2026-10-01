# SGLang and TensorRT-LLM

inferlint reads SGLang and TensorRT-LLM as well as vLLM. It tells the engine from the
server's metric names and boot log, so the commands are the same for all three, and
`inferlint xray` starts, records, checks and stops any of them.

Tested on SGLang 0.5.20 and TensorRT-LLM 1.3.0rc29 (a release candidate), on the same
RTX 4090 (WSL2, driver 616.56) as the vLLM runs. Other versions get an untested-version
warning where inferlint can read the version: from the boot log for vLLM and TensorRT-LLM,
and from the server for SGLang, which `xray` asks.

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

- **SGLang**: start it with `--enable-metrics`.
- **TensorRT-LLM**: put `return_perf_metrics: true` in the YAML file given to
  `trtllm-serve --config`. inferlint then reads Prometheus text from
  `/prometheus/metrics`; `/metrics` returns JSON. Keep its log at INFO, the default: T1
  counts the pauses it logs there.

For example:

```bash
inferlint xray -o run/ --serve "python -m sglang.launch_server --model-path MODEL --enable-metrics" \
    -- vllm bench serve --model MODEL --dataset-name random --random-input-len 88 \
       --random-output-len 1000 --ignore-eos --num-prompts 32 --max-concurrency 32
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

The files are in `tests/fixtures/vllm-0.30/qwen3-8b-start1/`, `qwen3-8b-start2/`,
`tests/fixtures/sglang-0.5/qwen3-8b/` and `tests/fixtures/trtllm-1.3/qwen3-8b/`.

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

**T9.** SGLang's usage gauge is its fullest memory pool, which for a hybrid model can be
the state slots rather than the KV cache.

**T12 and T7.** SGLang logs tracebacks it ignores ("Ignore import error") and errors it
survives ("post-warmup freeze_gc failed"); neither counts as a failure. Its log says
"Application startup complete" before it can serve, and `/health` answers 503 until it
can; `xray` waits for both.

**T14** is about vLLM's reserved null block and does not apply to the other engines.

## Not tested

Multi-GPU and multi-node setups, speculative decoding (T6) on these engines, and other
engines.
