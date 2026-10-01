"""Naming the engine: --engine and INFERLINT_ENGINE, and what is told without them."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from inferlint import bootlog, engines, teardown, xray
from inferlint.cli import main
from inferlint.result import CheckResult, Status
from inferlint.teardown import Proc

FX = Path(__file__).parent / "fixtures"
SGLANG_RUN = FX / "sglang-0.5" / "qwen3-8b"
VLLM_RUN = FX / "vllm-0.30" / "qwen3-8b-start2"


@pytest.fixture(autouse=True)
def _no_engine_in_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(engines.ENV_VAR, raising=False)


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    got = capsys.readouterr()
    return code, got.out, got.err


# --------------------------------------------------------------------------- engines


@pytest.mark.parametrize(
    ("cmd", "want"),
    [
        ("vllm serve /models/m --max-num-seqs 32", engines.VLLM),
        ("/opt/venv/bin/vllm serve m", engines.VLLM),
        ("env CUDA_VISIBLE_DEVICES=0 vllm serve m", engines.VLLM),
        ("python -m vllm.entrypoints.openai.api_server --model m", engines.VLLM),
        ("python -m sglang.launch_server --model-path m --enable-metrics", engines.SGLANG),
        ("sglang serve --model-path m", engines.SGLANG),
        ("/venv/bin/trtllm-serve serve m --config x.yaml", engines.TRTLLM),
        ("python -m tensorrt_llm.commands.serve serve m", engines.TRTLLM),
        (["python3", "-m", "sglang.launch_server"], engines.SGLANG),
        ("python serve.py --model vllm", None),  # names vllm, starts none of them
        ("vllm bench serve --model m", None),  # the load tool, not a server
        ('vllm serve "unbalanced', None),
        ("", None),
        (None, None),
    ],
)
def test_engine_from_the_server_command(
    cmd: str | list[str] | None, want: engines.Engine | None
) -> None:
    assert engines.from_command(cmd) is want


def test_engine_names_a_user_may_type() -> None:
    assert engines.parse_key("sglang") is engines.SGLANG
    assert engines.parse_key(" TensorRT-LLM ") is engines.TRTLLM
    assert engines.parse_key("vLLM") is engines.VLLM
    assert engines.parse_key("tgi") is None


def test_metric_names_that_name_no_engine() -> None:
    assert engines.named_by(["process_cpu_seconds_total"]) is None
    assert engines.from_names(["process_cpu_seconds_total"]) is engines.VLLM
    assert engines.named_by(["sglang:num_running_reqs", "process_cpu_seconds_total"]) is (
        engines.SGLANG
    )


def test_a_log_that_names_no_engine_takes_the_named_one() -> None:
    text = "loading weights\nApplication startup complete.\n"
    assert bootlog.parse(text).engine is None
    assert bootlog.parse(text, "sglang").engine == "sglang"
    real = (SGLANG_RUN / "boot.log").read_text(encoding="utf-8")
    assert bootlog.parse(real, "vllm").engine == "sglang"  # the log's own word wins


# --------------------------------------------------------------------------- the CLI


def test_snapshots_from_another_engine_are_refused(capsys: pytest.CaptureFixture[str]) -> None:
    snaps = [str(SGLANG_RUN / "before.snapshot.json"), str(SGLANG_RUN / "after.snapshot.json")]
    code, out, err = run(capsys, "preemption", *snaps, "--engine", "vllm")
    assert code == 2 and out == ""
    assert "--engine vllm, but the snapshots came from SGLang" in err
    code, out, _ = run(capsys, "preemption", *snaps, "--engine", "sglang")
    assert code == 0 and "[ T1] PASS" in out
    assert run(capsys, "rate", *snaps, "--engine", "trtllm")[0] == 2


def test_the_engine_from_the_environment(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    snaps = [str(SGLANG_RUN / "before.snapshot.json"), str(SGLANG_RUN / "after.snapshot.json")]
    monkeypatch.setenv(engines.ENV_VAR, "SGLang")
    assert run(capsys, "preemption", *snaps)[0] == 0
    monkeypatch.setenv(engines.ENV_VAR, "trtllm")
    code, _, err = run(capsys, "preemption", *snaps)
    assert code == 2 and f"{engines.ENV_VAR}=trtllm, but the snapshots came from SGLang" in err
    # --engine wins over the environment
    assert run(capsys, "preemption", *snaps, "--engine", "sglang")[0] == 0
    monkeypatch.setenv(engines.ENV_VAR, "tgi")
    code, _, err = run(capsys, "preemption", *snaps)
    assert code == 2 and "'tgi' is not an engine" in err


def test_an_unknown_engine_flag_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        main(["explain", "--engine", "tgi"])
    assert e.value.code == 2
    assert "unknown engine 'tgi'" in capsys.readouterr().err


def test_series_and_report_from_another_engine_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = str(SGLANG_RUN / "run.series.jsonl")
    code, _, err = run(capsys, "series", rec, "--engine", "trtllm")
    assert code == 2 and "came from SGLang" in err
    out = tmp_path / "r.html"
    log = str(VLLM_RUN / "boot.log")
    code, _, err = run(capsys, "report", "-o", str(out), "--boot-log", log, "--engine", "sglang")
    assert code == 2 and "boot.log came from vLLM" in err and not out.exists()


def test_a_boot_log_that_names_no_engine(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "boot.log"
    log.write_text("loading weights\nApplication startup complete.\n", encoding="utf-8")
    code, _, err = run(capsys, "boot-facts", "--json", str(log))
    assert code == 0 and "read as vLLM's" in err
    code, out, err = run(capsys, "boot-facts", "--json", "--engine", "sglang", str(log))
    assert code == 0 and err == "" and json.loads(out)["engine"] == "sglang"


def test_explain_shows_the_named_engine_only(capsys: pytest.CaptureFixture[str]) -> None:
    _, out, _ = run(capsys, "explain", "T1")
    assert "On SGLang:" in out and "On TensorRT-LLM:" in out
    _, out, _ = run(capsys, "explain", "T1", "--engine", "sglang")
    assert "On SGLang:" in out and "On TensorRT-LLM:" not in out


# --------------------------------------------------------------------------- teardown

TABLE = [
    Proc(10, 1, ("vllm", "serve", "m"), "vllm"),
    Proc(11, 10, ("VLLM::EngineCore",), "VLLM::EngineCor"),
    Proc(20, 1, ("python", "-m", "sglang.launch_server", "--model-path", "m"), "python"),
    Proc(21, 20, ("sglang::scheduler",), "sglang::schedul"),
    Proc(30, 1, ("bash",), "bash"),
]


def test_teardown_of_one_engine_leaves_the_others() -> None:
    assert {p.pid for p in teardown.server_processes(TABLE, 30, "sglang")} == {20, 21}
    assert {p.pid for p in teardown.server_processes(TABLE, 30)} == {10, 11, 20, 21}
    alive = {p.pid for p in TABLE}
    sent: list[int] = []
    now = [0.0]

    def kill(pids: object, sig: int) -> list[int]:
        pl = list(pids)  # type: ignore[call-overload]
        sent.extend(pl)
        alive.difference_update(pl)
        return pl

    def sleep(s: float) -> None:
        now[0] += s

    targets, res = teardown.teardown(
        list_procs=lambda: [p for p in TABLE if p.pid in alive],
        kill=kill,
        read_used=lambda: [0],
        self_pid=30,
        timeout_s=5.0,
        sleep=sleep,
        clock=lambda: now[0],
        engine="sglang",
    )
    assert {p.pid for p in targets} == {20, 21} and set(sent) == {20, 21}
    assert not res.clear  # vLLM's server is still there, so the card is not clear


def test_teardown_command_says_what_it_leaves(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from inferlint import cli

    monkeypatch.setattr(cli, "_procs_readable", lambda: True)
    monkeypatch.setattr(teardown, "list_processes", lambda: list(TABLE))
    code, out, _ = run(capsys, "teardown", "--engine", "trtllm", "--dry-run")
    assert code == 0 and "no server processes found (TensorRT-LLM)" in out
    assert "left alone (--engine trtllm): 4 processes of SGLang and vLLM servers" in out
    _, out, _ = run(capsys, "teardown", "--engine", "sglang", "--dry-run")
    assert "target  pid=20 " in out and "target  pid=10 " not in out


# --------------------------------------------------------------------------- xray


def _server(metrics: str | None) -> tuple[ThreadingHTTPServer, str]:
    class H(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            if self.path == "/health":
                code, body = 200, b"{}"
            elif self.path == "/metrics" and metrics is not None:
                code, body = 200, metrics.encode()
            else:
                code, body = 404, b"{}"
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


@pytest.fixture
def no_metrics() -> Iterator[str]:
    httpd, url = _server(None)
    try:
        yield url
    finally:
        httpd.shutdown()


@pytest.fixture
def vllm_metrics() -> Iterator[str]:
    httpd, url = _server('vllm:num_requests_running{model_name="m"} 0.0\n')
    try:
        yield url
    finally:
        httpd.shutdown()


def test_snapshot_of_another_engine_or_no_metrics_is_not_saved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], vllm_metrics: str, no_metrics: str
) -> None:
    out = tmp_path / "s.json"
    code, _, err = run(capsys, "snapshot", vllm_metrics, "-o", str(out), "--engine", "sglang")
    assert code == 2 and not out.exists()
    assert f"--engine sglang, but the server at {vllm_metrics} runs vLLM" in err
    code, _, err = run(capsys, "snapshot", no_metrics, "-o", str(out), "--engine", "sglang")
    assert code == 2 and not out.exists() and err.rstrip().endswith("--enable-metrics")
    assert run(capsys, "snapshot", vllm_metrics, "-o", str(out))[0] == 0 and out.exists()


def test_xray_refuses_a_server_command_of_another_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[str] = []

    def gate() -> CheckResult:
        started.append("gate")
        return CheckResult("T2", Status.PASS, "clear")

    monkeypatch.setattr(xray, "_gpu_free", gate)
    plan = xray.Plan(out=tmp_path, load=["true"], serve="vllm serve m", engine="sglang")
    o = xray.run(plan, say=lambda _s: None)
    assert o.problem == "the engine named is SGLang, but the --serve command is vLLM's"
    assert started == [] and o.results == [] and xray.exit_code(o) == 1


def test_xray_says_how_to_turn_on_the_metrics(tmp_path: Path, no_metrics: str) -> None:
    o = xray.run(xray.Plan(out=tmp_path, load=["true"], url=no_metrics), say=lambda _s: None)
    assert o.problem is not None and "no vLLM, SGLang or TensorRT-LLM metrics" in o.problem
    assert "--enable-metrics" in o.problem and "return_perf_metrics" in o.problem
    plan = xray.Plan(out=tmp_path, load=["true"], url=no_metrics, engine="sglang")
    o = xray.run(plan, say=lambda _s: None)
    assert o.problem is not None and o.problem.endswith("start SGLang with --enable-metrics")
    assert "return_perf_metrics" not in o.problem and o.load_exit is None


def test_xray_tells_the_engine_and_refuses_another(tmp_path: Path, vllm_metrics: str) -> None:
    said: list[str] = []
    plan = xray.Plan(out=tmp_path / "a", load=["true"], url=vllm_metrics, probe=False)
    xray.run(plan, say=said.append)
    assert "== engine: vLLM (from its metrics)" in said
    plan = xray.Plan(out=tmp_path / "b", load=["true"], url=vllm_metrics, engine="trtllm")
    o = xray.run(plan, say=lambda _s: None)
    assert (
        o.problem == f"the engine named is TensorRT-LLM, but the server at {vllm_metrics} is vLLM's"
    )
