# Changelog

Each release on [PyPI](https://pypi.org/project/inferlint/) and its
[GitHub release](https://github.com/radianvector/inferlint/releases). Dates are the PyPI
upload dates (UTC).

## Unreleased

### Added

- **SGLang and TensorRT-LLM.** inferlint tells the engine from the server's metric names
  and boot log; the commands, `xray` and the report work the same way for all three.
  Tested on SGLang 0.5.20 and TensorRT-LLM 1.3.0rc29. See
  [docs/engines.md](https://github.com/radianvector/inferlint/blob/main/docs/engines.md).
- **`--engine vllm|sglang|trtllm`**, or the `INFERLINT_ENGINE` environment variable, names
  the engine for every command that reads one. Without it the engine is told from the
  input as before. With it, an input from another engine (a wrong port, a file from
  another run) is refused with exit status 2, and an input that names no engine is read
  as the named one. `teardown --engine X` stops only X's servers, and `explain --engine X`
  shows only X's notes.
- `xray` takes the engine from the `--serve` command before it starts anything, and stops
  with the setting to change when the server serves no metrics (SGLang:
  `--enable-metrics`; TensorRT-LLM: `return_perf_metrics: true`).
- T11 says which part of vLLM's memory split moved between starts, and how many starts
  compiled the model from scratch.

### Changed

- `teardown`, `gpu-inspect` and `xray` count every process a server started as part of
  it.
- T12 names a bare `AssertionError` by its `assert` line, and a generic exception by the
  function it was raised in.

### Fixed

- T12 no longer reports tracebacks the server says it ignores, or errors logged before
  the server became ready, as a failed start; T7 no longer reports an error after which
  the server kept serving.
- T8 names a server's own running limit only when the run reached it.

## 0.2.0 (2026-10-01)

### Added

- **`inferlint xray`**: one command around a whole benchmark run. With
  `--serve "vllm serve ..."` it checks the GPU is free, starts the server and saves its
  boot log, waits until it is ready, and always stops it at the end, including after a
  failure or Ctrl-C. Around the load command given after `--` it takes the before and
  after snapshots, records the gauges, runs every tripwire the files allow, and writes
  `report.html` and `results.json` into one folder. `--url` attaches to a server that is
  already running.
- **T15, client and server counts disagree**: the load tool's result file
  (`vllm bench serve --save-result`) is compared with the server's counters. Extra work
  on the server means other traffic shared it; `xray` saves the result file
  automatically for `vllm bench serve`.
- **Tested on vLLM 0.29 and 0.30**, as well as 0.28, on an RTX 4090, with recorded
  fixtures for each and a tested-versions table in the README. `xray` itself ran live on
  all three.
- A warning when a boot log comes from a vLLM release inferlint has not been tested on.
- Names for tripwires as well as codes (`T1`, `silent-preemption`); `inferlint explain`
  accepts either.
- T12 names the root cause of a kernel build failure (`ptxas fatal`, `nvcc fatal`, a
  missing library) instead of "Ninja build failed".
- T8 says when admission control (`--max-num-queued-reqs`, vLLM 0.29 and later) rejected
  requests, rather than the KV cache holding them back.

### Changed

- A failed tripwire is shown as `TRIPWIRE-FAILED` in the terminal and "Tripwire failed"
  in the report, so it reads as a finding about the run, not a server crash. The JSON
  status stays `"fail"`, and exit codes are unchanged.
- Explanations use precise technical terms (`explain`, the report's tripwire guide);
  the tutorial keeps the plain-language versions. "Users" became "concurrent requests".
- Renamed tripwires: T2 Orphaned engine process, T3 Cleanup kills its own shell, T4 KV
  pool size jumps between starts, T5 Large KV block size, T7 Crash after ready, T8
  Concurrency not reached, T9 Concurrency ceiling falls as sequences grow, T10
  Integer-second timing, T11 KV memory budget drifts between starts, T12 Start-up
  failure without a cause, T13 Stale waiter adopts the next server, T14 Reserved null
  block.
- Version-specific statements say which vLLM versions they were verified on.
- The report uses the reader's system fonts; it still fetches nothing.
- Every metric name is defined in one place (`inferlint.metricnames`).

### Fixed

- `check-log` and `report` treated two boot logs with the same file name
  (`run1/boot.log`, `run2/boot.log`) as one, so T4 and T11 could compare a log with
  itself. Logs are now labelled with as much of their path as tells them apart.
- The first line of `inferlint --help` showed reStructuredText backticks.

## 0.1.1 (2026-09-29)

- README and package summary: "Measures what your inference server actually did, and
  flags what it didn't tell you."
- Package links: homepage on radianvector.com.

## 0.1.0 (2026-09-29)

First release: 14 tripwires for vLLM benchmark runs, checked from boot logs, `/metrics`
snapshots and a gauge recording. Commands: `gpu-inspect`, `check-log`, `boot-facts`,
`probe`, `snapshot`, `watch`, `preemption`, `rate`, `series`, `report`, `teardown`,
`explain`. Verified on vLLM 0.28 on an RTX 4090.
