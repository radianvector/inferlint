"""inferlint xray, run end to end against a mock server whose counters are known."""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from inferlint import cli, xray
from inferlint.result import Status


class _Mock:
    """Counts like vLLM: every completion adds one finished request and its tokens."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.done = 0
        self.tokens = 0

    def metrics(self) -> str:
        with self.lock:
            return (
                'vllm:num_requests_running{model_name="m"} 0.0\n'
                'vllm:num_requests_waiting{model_name="m"} 0.0\n'
                'vllm:kv_cache_usage_perc{model_name="m"} 0.0\n'
                'vllm:num_preemptions_total{model_name="m"} 0.0\n'
                f'vllm:generation_tokens_total{{model_name="m"}} {self.tokens}.0\n'
                f'vllm:prompt_tokens_total{{model_name="m"}} {self.done * 8}.0\n'
                'vllm:request_success_total{finished_reason="length",model_name="m"} '
                f"{self.done}.0\n"
                'vllm:request_success_total{finished_reason="error",model_name="m"} 0.0\n'
            )


def _handler(m: _Mock) -> type[BaseHTTPRequestHandler]:
    class H(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, b"{}")
            elif self.path == "/v1/models":
                self._send(200, json.dumps({"data": [{"id": "m"}]}).encode())
            elif self.path == "/metrics":
                self._send(200, m.metrics().encode(), "text/plain")
            else:
                self._send(404, b"{}")

        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            toks = int(req.get("max_tokens", 4))
            with m.lock:
                m.done += 1
                m.tokens += toks
            body = {"choices": [{"text": "ok"}], "usage": {"completion_tokens": toks}}
            self._send(200, json.dumps(body).encode())

    return H


@pytest.fixture
def server() -> Iterator[tuple[str, _Mock]]:
    m = _Mock()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler(m))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", m
    finally:
        httpd.shutdown()


# A load tool: sends requests, then writes a result file in vllm bench serve's format.
_LOAD = """
import json, sys, urllib.request
url, n, out, claimed = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
for _ in range(n):
    body = json.dumps({"model": "m", "prompt": "hi", "max_tokens": 10}).encode()
    req = urllib.request.Request(url + "/v1/completions", body, {"Content-Type": "application/json"})
    urllib.request.urlopen(req).read()
json.dump({"completed": claimed, "failed": 0, "total_output_tokens": claimed * 10,
           "max_concurrency": 4, "duration": 1.0}, open(out, "w"))
print(f"sent {n} requests")
"""


def _load_cmd(tmp: Path, url: str, sent: int, claimed: int) -> list[str]:
    script = tmp / "load.py"
    script.write_text(_LOAD, encoding="utf-8")
    return [sys.executable, str(script), url, str(sent), str(tmp / "result.json"), str(claimed)]


def test_attach_runs_the_whole_sequence(tmp_path: Path, server: tuple[str, _Mock]) -> None:
    url, _ = server
    out = tmp_path / "run"
    plan = xray.Plan(
        out=out,
        load=_load_cmd(tmp_path, url, 6, 6),
        url=url,
        bench_result=tmp_path / "result.json",
        interval_s=0.05,
    )
    o = xray.run(plan, say=lambda _s: None)
    by = {r.tripwire: r for r in o.results}
    assert o.problem is None and o.load_exit == 0
    assert by["T7"].status is Status.PASS  # the probe
    assert by["T1"].status is Status.PASS and by["T10"].status is Status.PASS
    assert by["T15"].status is Status.PASS, by["T15"].message
    assert by["T15"].evidence["extra_requests"] == 0  # not vllm bench: no untimed requests
    for f in ("before.snapshot.json", "after.snapshot.json", "run.series.jsonl", "load.log"):
        assert (out / f).is_file(), f
    assert "sent 6 requests" in (out / "load.log").read_text(encoding="utf-8")
    assert o.report is not None and "T15" in o.report.read_text(encoding="utf-8")
    saved = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert {r["tripwire"] for r in saved} >= {"T1", "T7", "T10", "T15"}
    # The concurrency asked for comes from the result file (4); the mock never reports a
    # running request, so T8 fails, as it would for a server that queued everything.
    assert by["T8"].status is Status.FAIL and "requested 4" in by["T8"].message
    assert xray.exit_code(o) == 1


def test_other_traffic_fails_t15(tmp_path: Path, server: tuple[str, _Mock]) -> None:
    url, _ = server
    plan = xray.Plan(
        out=tmp_path / "run",
        load=_load_cmd(tmp_path, url, 7, 5),  # the server sees 7; the client reports 5
        url=url,
        bench_result=tmp_path / "result.json",
        probe=False,
        interval_s=0.05,
    )
    o = xray.run(plan, say=lambda _s: None)
    t15 = next(r for r in o.results if r.tripwire == "T15")
    assert t15.status is Status.FAIL
    assert "2 more requests than the load tool sent" in t15.message
    assert xray.exit_code(o) == 1


def test_no_server_is_a_problem_not_a_crash(tmp_path: Path) -> None:
    o = xray.run(
        xray.Plan(out=tmp_path, load=["true"], url="http://127.0.0.1:9"), say=lambda _s: None
    )
    assert o.problem is not None and "no server answers" in o.problem
    assert xray.exit_code(o) == 1 and o.report is None
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8")) == []


def test_the_load_command_is_read_and_completed(tmp_path: Path) -> None:
    bench = ["vllm", "bench", "serve", "--model", "M", "--max-concurrency", "32"]
    cmd, result = xray.prepare_load(bench, tmp_path)
    added = ["--save-result", "--result-dir", str(tmp_path), "--result-filename", "bench.json"]
    assert cmd == bench + added and result == tmp_path / "bench.json"
    assert xray.load_concurrency(bench) == 32
    assert xray.load_concurrency(["x", "--max-concurrency=8"]) == 8
    # a command that already saves its result keeps its own file name
    own = [*bench, "--save-result", "--result-dir", "r", "--result-filename", "b.json"]
    assert xray.prepare_load(own, tmp_path) == (own, Path("r") / "b.json")
    # other load tools are run as they are
    assert xray.prepare_load(["locust", "-u", "32"], tmp_path) == (["locust", "-u", "32"], None)
    # vllm bench serve's output says which requests it left out of its result file
    skipped = "Skipping endpoint ready check.\nStarting main benchmark run"
    tested = "Initial test run completed.\nStarting main benchmark run"
    warmed = "Initial test run completed.\nWarming up with 3 requests..."
    assert xray.untimed_requests(skipped) == 0
    assert xray.untimed_requests(tested) == 1
    assert xray.untimed_requests(warmed) == 4


def test_cli_xray(
    tmp_path: Path, server: tuple[str, _Mock], capsys: pytest.CaptureFixture[str]
) -> None:
    url, _ = server
    load = _load_cmd(tmp_path, url, 3, 3)
    out = tmp_path / "cli"
    code = cli.main(
        [
            "xray",
            "-o",
            str(out),
            "--url",
            url,
            "--bench-result",
            str(tmp_path / "result.json"),
            "--interval",
            "0.05",
            "--",
            *load,
        ]
    )
    text = capsys.readouterr().out
    assert "[T15] PASS" in text and f"report: {out / 'report.html'}" in text
    assert code == 1  # T8: the mock never reports a running request
    assert cli.main(["xray", "-o", str(out)]) == 2  # no load command


def test_serve_starts_and_always_stops_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--serve with a server that dies during boot: the boot checks run, and stop still runs."""
    from inferlint.result import CheckResult

    calls: list[str] = []
    monkeypatch.setattr(xray, "_gpu_free", lambda: CheckResult("T2", Status.PASS, "clear"))

    def stop(proc: object) -> CheckResult:
        calls.append("stop")
        return CheckResult("T2", Status.PASS, "2 consecutive clear readings")

    monkeypatch.setattr(xray, "_stop_server", stop)
    monkeypatch.setattr(xray, "_linux", lambda: True)
    # A "server" that prints a real recorded boot failure and exits.
    log = Path(__file__).parent / "fixtures" / "vllm-0.28" / "boot_fail_accelerator.log"
    script = tmp_path / "fake_server.py"
    script.write_text(
        f"import sys; sys.stdout.write(open({str(log)!r}, encoding='utf-8').read()); sys.exit(1)",
        encoding="utf-8",
    )
    plan = xray.Plan(
        out=tmp_path / "run",
        load=["true"],
        url="http://127.0.0.1:9",
        serve=f'"{sys.executable}" "{script}"',
        ready_timeout_s=20,
    )
    o = xray.run(plan, say=lambda _s: None)
    assert calls == ["stop"]
    assert o.problem is not None and "exited with code 1" in o.problem
    assert any(r.tripwire == "T12" and r.status is Status.FAIL for r in o.results)
    assert o.report is not None  # the boot log alone still makes a report
    assert xray.exit_code(o) == 1
