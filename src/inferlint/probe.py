"""T7: a boot is not a gate. Send real requests before trusting a server.

Some configs start, announce they are ready, and die on the first request that reaches
a code path the boot never exercised (a kernel JIT-compiled lazily, for instance). The
probe sends one request, then a concurrent burst, then checks the server is still alive.
It is a smoke test, not a benchmark; use a load generator to measure.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .result import CheckResult, Status

__all__ = ["probe"]

PROMPT = "Write one sentence about the weather."


@dataclass(frozen=True)
class _Reply:
    ok: bool
    status: int | None
    seconds: float
    tokens: int
    error: str = ""


def _post(url: str, body: dict[str, Any], timeout: float) -> _Reply:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            doc: dict[str, Any] = json.loads(r.read())
            status = int(r.status)
    except urllib.error.HTTPError as e:
        return _Reply(False, e.code, time.monotonic() - t0, 0, e.reason)
    except (urllib.error.URLError, OSError, ValueError) as e:
        return _Reply(False, None, time.monotonic() - t0, 0, str(e))
    usage: dict[str, Any] = doc.get("usage") or {}
    tokens = int(usage.get("completion_tokens") or 0)
    choices: list[dict[str, Any]] = doc.get("choices") or [{}]
    text = str(choices[0].get("text") or "")
    ok = status == 200 and (tokens > 0 or bool(text))
    return _Reply(ok, status, time.monotonic() - t0, tokens, "" if ok else "empty completion")


def _get_json(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _alive(base: str, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(base + "/health", timeout=timeout) as r:
            return int(r.status) == 200
    except (urllib.error.URLError, OSError):
        return False


def probe(
    base_url: str,
    *,
    model: str | None = None,
    concurrency: int = 8,
    max_tokens: int = 16,
    timeout: float = 300.0,
) -> CheckResult:
    base = base_url.rstrip("/")
    try:
        model = model or str(_get_json(base + "/v1/models", 30.0)["data"][0]["id"])
    except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError) as e:
        return CheckResult("T7", Status.FAIL, f"cannot list models: {e}")
    body = {"model": model, "prompt": PROMPT, "max_tokens": max_tokens, "temperature": 0.0}
    url = base + "/v1/completions"

    first = _post(url, body, timeout)
    ev: dict[str, Any] = {"model": model, "first": first.__dict__}
    if not first.ok:
        ev["alive_after"] = _alive(base, 10.0)
        return CheckResult("T7", Status.FAIL, f"first request failed: {first.error}", ev)

    def one(_: int) -> _Reply:
        return _post(url, body, timeout)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        burst = list(pool.map(one, range(concurrency)))
    bad = [r for r in burst if not r.ok]
    ev["burst"] = {
        "concurrency": concurrency,
        "failed": len(bad),
        "max_seconds": max(r.seconds for r in burst),
    }
    alive = _alive(base, 10.0)
    ev["alive_after"] = alive
    if bad or not alive:
        why = (
            f"{len(bad)}/{concurrency} burst requests failed" if bad else "server died after burst"
        )
        return CheckResult("T7", Status.FAIL, why, ev)
    return CheckResult(
        "T7", Status.PASS, f"1 + {concurrency} concurrent requests served, server alive", ev
    )
