# Security

## What inferlint does on your machine

- **Reads** a server's `/metrics`, `/health` and `/v1/models` over HTTP, boot logs and
  other files you name, `/proc` (process lists) and `nvidia-smi` output.
- **Sends requests** to the server you name: `probe` sends 9 small completion requests,
  and `xray` sends them too unless you pass `--no-probe`.
- **Stops processes.** `teardown`, and `xray --serve` when it finishes, send SIGTERM and
  then SIGKILL to processes it identifies as vLLM servers (`vllm serve`, `python -m
  vllm.entrypoints...`, and `VLLM::*` engine processes). It never signals itself or its
  own parent processes. `teardown --dry-run` lists the targets without signalling.
- **Runs the commands you give** to `xray`: the `--serve` command and the load command.

It opens no ports and sends nothing anywhere except to the server URL you give.

## Reporting a vulnerability

Please report security problems privately, through GitHub's
[private vulnerability reporting](https://github.com/radianvector/inferlint/security/advisories/new)
for this repository, or by email to hello@radianvector.com. Do not open a public issue.

Examples: `teardown` signalling a process that is not a vLLM server, a crafted log or
`/metrics` response that makes inferlint run code or write outside its output folder.

You will get a reply within a week. Fixes go into the latest release; there are no
backports to older versions while inferlint is below 1.0.
