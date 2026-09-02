"""A real OpenAI-compatible server, for testing the client against HTTP.

The LocalLLM client had never executed against an actual socket -- only against
an in-process fake, which cannot catch the things that break in practice: SSE
framing, chunked transfer, keep-alive reuse, malformed lines mid-stream,
connection drops.

This is a genuine HTTP server speaking the same wire protocol vLLM and Ollama
do, so the client is exercised for real without needing a GPU or a 40GB model.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


class StubConfig:
    def __init__(self) -> None:
        self.reply = "Structure just shifted. If that holds, continuation is the scenario."
        self.chunk_size = 7
        self.delay_s = 0.0
        self.status = 200
        #: Emit junk lines mid-stream. Real servers send keep-alive comments and
        #: the occasional non-JSON line; a client that dies on those is broken.
        self.inject_garbage = False
        #: Cut the connection mid-stream, as a server under load will.
        self.truncate_after = 0
        self.requests: list[dict] = []
        self.cached_tokens = 0
        #: Simulate a server that is up but has nothing loaded.
        self.no_models = False


def make_handler(cfg: StubConfig) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.endswith("/models"):
                data = [] if cfg.no_models else [{"id": "stub-model", "object": "model"}]
                self._json(200, {"data": data})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length) or b"{}")
            cfg.requests.append(request)

            if cfg.status != 200:
                self._json(cfg.status, {"error": {"message": "stub failure"}})
                return

            if request.get("stream"):
                self._stream()
            else:
                self._complete()

        def _complete(self) -> None:
            self._json(200, {
                "id": "stub", "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": cfg.reply},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 120, "completion_tokens": 24,
                    "total_tokens": 144,
                    "prompt_tokens_details": {"cached_tokens": cfg.cached_tokens},
                },
            })

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def emit(line: str) -> None:
                payload = f"{line}\n\n".encode()
                self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")
                self.wfile.flush()

            text = cfg.reply
            pieces = [
                text[i : i + cfg.chunk_size] for i in range(0, len(text), cfg.chunk_size)
            ]
            try:
                for n, piece in enumerate(pieces):
                    if cfg.truncate_after and n >= cfg.truncate_after:
                        return  # drop the connection mid-stream
                    if cfg.inject_garbage and n == 1:
                        emit(": keep-alive")
                        emit("data: {not valid json")
                    emit("data: " + json.dumps({
                        "choices": [{"delta": {"content": piece}, "index": 0}]
                    }))
                    if cfg.delay_s:
                        time.sleep(cfg.delay_s)
                emit("data: [DONE]")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


class StubLLMServer:
    """Context manager yielding (base_url, config)."""

    def __init__(self, cfg: StubConfig | None = None) -> None:
        self.cfg = cfg or StubConfig()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> tuple[str, StubConfig]:
        self._server = HTTPServer(("127.0.0.1", 0), make_handler(self.cfg))
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="stub-llm"
        )
        self._thread.start()
        port = self._server.server_address[1]
        return f"http://127.0.0.1:{port}/v1", self.cfg

    def __exit__(self, *exc) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
