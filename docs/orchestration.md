# Orchestration traps

Two tripwires live in the shell that drives a benchmark, not in anything Python can
check from outside. Both produce plausible numbers instead of errors.

## T3: `pkill -f PATTERN` inside `bash -c` kills its own shell

```bash
bash -lc 'echo stopping; pkill -f "vllm serve"; echo done'   # exit 143; "done" never prints
```

Measured on Ubuntu under WSL2 with a pattern unique to the test: the one-liner is killed
by its own SIGTERM (exit status 143) and never reaches `echo done`. The bracketed pattern
below exits 0.

`pkill -f` matches against full command lines. The shell running the one-liner has the
pattern in its own command line, so it signals itself. Two fixes:

```bash
pkill -f "vllm[ ]serve"      # the regex still matches "vllm serve"; the literal text does not
inferlint teardown                # matches argv structurally and never targets its own ancestry
```

The bracket trick fixes T3 and still leaves T2. The renamed `VLLM::EngineCore` child
does not match either pattern.

## T13: a killed campaign's waiter adopts the replacement server

Sequence that voids a run:

1. Campaign A is waiting for its server: `wait_ready` polls `http://127.0.0.1:8000/health`.
2. A is sent SIGTERM. Its `trap cleanup EXIT` is **deferred**, because bash does not run
   traps while blocked in a foreground child.
3. Campaign B starts and boots its server on the same port.
4. A's `wait_ready` sees B's server answer, reports ready and returns. A's deferred trap
   now runs its cleanup and kills **B's** server a few seconds into B's run.

Rules that prevent it:

- After killing a campaign, confirm the **script** process is gone before launching the
  next one, not just that the GPU is free. `kill -9` on the script avoids the deferral
  entirely.
- Match scripts by their bare name. A script started with `exec bash "$DIR/x.sh"` shows
  an absolute path, so `pgrep -f "bash scripts/x.sh"` silently finds nothing.
- Give each campaign its own port, or have `wait_ready` check that the server it found
  is the one it launched (its PID, or a unique `--served-model-name`).

## A boot sequence that respects T2, T4, T7 and T11

```bash
inferlint gpu-clear || exit 1                      # refuse to boot on a dirty card (T2)
vllm serve "$MODEL" "${ARGS[@]}" > boot.log 2>&1 &
until grep -q "Application startup complete" boot.log; do sleep 5; done
inferlint check-log boot.log || exit 1              # backend, failure reason, block size
inferlint probe http://127.0.0.1:8000 || exit 1     # boot is not a gate (T7)
inferlint boot-facts --json boot.log > boot.json    # pool, block size, backend: stamp every result (T4, T11)
# ... run the benchmark with `inferlint watch` sampling alongside ...
inferlint teardown                                  # SIGTERM, SIGKILL survivors, consecutive clear readings (T2)
```
