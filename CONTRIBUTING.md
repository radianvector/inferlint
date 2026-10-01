# Contributing

The most useful contribution is evidence from a vLLM version or GPU that inferlint has
not been tested on. Every tripwire is checked against recorded files from real runs, so
a new boot log or `/metrics` capture is what lets the next release say it works there.

## Contribute a recorded run

1. Run a benchmark with `inferlint xray`. It saves every file inferlint reads into one
   folder:

   ```bash
   inferlint xray -o run/ --serve "vllm serve MODEL ..." \
       -- vllm bench serve --model MODEL --max-concurrency 32 ...
   ```

   Without `--serve`, start the server yourself with its output saved
   (`vllm serve ... 2>&1 | tee boot.log`) and pass `--boot-log boot.log`.

2. Remove local paths and anything private. Boot logs contain the model path and your
   home directory:

   ```bash
   sed -i -e "s#$HOME#~#g" -e "s#/path/to/models#/models#g" run/boot.log
   ```

   Check `boot.log` and `load.log` before sharing them. The snapshots and the recording
   hold only metric values.

3. Open an issue with the "A tripwire misfired" or "Recorded run" template and attach
   `boot.log`, `before.snapshot.json`, `after.snapshot.json`, `run.series.jsonl`,
   `results.json` and, if there is one, `bench.json`. Say which vLLM version, GPU and
   model the run used.

Recorded runs go in `tests/fixtures/vllm-<version>/`, and `tests/test_live.py` pins the
numbers they produce.

## Work on the code

```bash
git clone https://github.com/radianvector/inferlint
cd inferlint
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -q                          # no GPU needed
ruff check src tests && ruff format --check src tests && pyright
```

GitHub Actions runs the same commands on every push and pull request.

### Where things are

| module | what it does |
|---|---|
| `bootlog` | boot log to structured facts; unknown formats are listed, never guessed |
| `prom`, `telemetry` | Prometheus text parsing; snapshots with timing; counter deltas |
| `metricnames` | every metric name inferlint reads, in one place |
| `series` | the gauge recording made during a run |
| `checks` | the tripwires; each returns a `CheckResult` |
| `catalog` | each tripwire's name and explanation, used by `explain` and the report |
| `probe`, `teardown`, `gpu` | live checks: real requests, stopping servers, GPU facts |
| `benchresult` | the load tool's result file (`vllm bench serve --save-result`) |
| `xray` | the whole sequence in one command |
| `report`, `svgchart` | the HTML report |
| `cli` | the `inferlint` command |

### Add a tripwire

A tripwire is accepted when it has evidence and has been seen to fail:

1. **Evidence.** A recorded file from a real run that shows the problem, in
   `tests/fixtures/`.
2. **A check** in `checks.py` that returns a `CheckResult`. It never raises on a finding.
   It returns `UNKNOWN` (Can't tell) when an input it needs is missing, never a pass.
3. **Both verdicts tested.** A test where it passes and a test where it fails, both on
   recorded evidence, plus a row in `tests/test_mutations.py` that changes one number or
   line and requires the verdict to flip.
4. **An entry** in `catalog.py` (code, name, what, why, command) and in
   `docs/tripwires.md`.

Codes are permanent: a new tripwire gets the next number.

### Wording

Output, `explain`, the README and `docs/tripwires.md` use precise technical terms. The
tutorial and the report's "What happened" section explain the same things in plain
words. Neither uses slogans: say what was measured and what it means.
