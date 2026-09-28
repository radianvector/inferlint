"""The probe, tested against a mock server whose behaviour is known."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from inferlint.probe import probe
from inferlint.result import Status


class _State:
    def __init__(self, mode: str) -> None:
        self.mode = mode  # "healthy" | "dies_on_first" | "dies_under_burst" | "empty"
        self.completions = 0
        self.dead = False
        self.lock = threading.Lock()


def _handler(state: _State) -> type[BaseHTTPRequestHandler]:
    class H(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def _send(self, code: int, doc: object) -> None:
            body = json.dumps(doc).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/v1/models":
                self._send(200, {"data": [{"id": "mock-model"}]})
            elif self.path == "/health":
                self._send(500 if state.dead else 200, {})
            else:
                self._send(404, {})

        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            with state.lock:
                state.completions += 1
                k = state.completions
                if state.mode == "dies_on_first" or (state.mode == "dies_under_burst" and k > 3):
                    state.dead = True
            if state.dead:
                self._send(500, {"error": "EngineDeadError"})
                return
            text = "" if state.mode == "empty" else "It is sunny."
            toks = 0 if state.mode == "empty" else min(4, req["max_tokens"])
            self._send(200, {"choices": [{"text": text}], "usage": {"completion_tokens": toks}})

    return H


@pytest.fixture
def server(request: pytest.FixtureRequest) -> Iterator[tuple[str, _State]]:
    state = _State(request.param)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", state
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.parametrize("server", ["healthy"], indirect=True)
def test_healthy(server: tuple[str, _State]) -> None:
    url, state = server
    r = probe(url, concurrency=6)
    assert r.status is Status.PASS, r.message
    assert state.completions == 7
    assert r.evidence["model"] == "mock-model"


@pytest.mark.parametrize("server", ["dies_on_first"], indirect=True)
def test_dies_on_first_request(server: tuple[str, _State]) -> None:
    url, _ = server
    r = probe(url)
    assert r.status is Status.FAIL
    assert r.evidence["alive_after"] is False


@pytest.mark.parametrize("server", ["dies_under_burst"], indirect=True)
def test_dies_under_burst(server: tuple[str, _State]) -> None:
    url, _ = server
    r = probe(url, concurrency=8)
    assert r.status is Status.FAIL
    assert r.evidence["burst"]["failed"] >= 1


@pytest.mark.parametrize("server", ["empty"], indirect=True)
def test_empty_completion_is_a_failure(server: tuple[str, _State]) -> None:
    url, _ = server
    assert probe(url).status is Status.FAIL


def test_nothing_listening() -> None:
    r = probe("http://127.0.0.1:9", concurrency=1)
    assert r.status is Status.FAIL
