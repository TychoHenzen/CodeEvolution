"""Routing proxy that delegates to codex or claude backends.

Starts both codex and claude proxies, then runs a lightweight HTTP
server that routes ``/v1/chat/completions`` requests to the correct
backend based on the ``model`` field in the request body.  OpenEvolve's
ensemble randomly samples between the two models, so requests naturally
distribute across backends.
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
from urllib.request import urlopen, Request

from codeevolve.base_proxy import BaseProxyHandler
from codeevolve.codex_proxy import CodexProxy
from codeevolve.claude_proxy import ClaudeProxy

logger = logging.getLogger(__name__)

MIXED_PROXY_PORT = 8083


class _RouterHandler(BaseProxyHandler):
    """Routes requests to codex or claude proxy based on model name."""

    proxy_name = "mixed-proxy"
    codex_port: int = 8081
    claude_port: int = 8082
    codex_model: str = "gpt-5.4-mini"
    claude_model: str = "haiku"

    def do_POST(self):
        if "/chat/completions" not in self.path:
            self.send_error(404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)
        body = json.loads(raw_body)

        model = body.get("model", "")
        if model == self.claude_model:
            port = self.claude_port
            backend = "claude"
        else:
            port = self.codex_port
            backend = "codex"

        logger.info("mixed-proxy: routing to %s (port %d, model=%s)", backend, port, model)

        url = f"http://localhost:{port}{self.path}"
        req = Request(url, data=raw_body, headers={"Content-Type": "application/json"})
        try:
            resp = urlopen(req, timeout=600)
            resp_body = resp.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
        except Exception as e:
            logger.error("mixed-proxy: backend %s failed: %s", backend, e)
            self._respond_json({
                "id": "mixed-proxy-error",
                "object": "chat.completion",
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "error",
                }],
            })

    def do_GET(self):
        if self.path in ("/health", "/v1/health"):
            self._respond_json({"status": "ok"})
        elif self.path == "/v1/models":
            self._respond_json({
                "data": [
                    {"id": self.codex_model, "object": "model"},
                    {"id": self.claude_model, "object": "model"},
                ],
            })
        else:
            self.send_error(404)

    def _invoke_cli(self, prompt: str) -> str:
        # Router never invokes CLI directly — it proxies to sub-proxies
        raise NotImplementedError


class MixedProxy:
    """Starts codex + claude proxies and a routing proxy in front."""

    def __init__(self, codex_config, claude_config):
        self._codex = CodexProxy(codex_config)
        self._claude = ClaudeProxy(claude_config)
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def api_base(self) -> str:
        return f"http://localhost:{MIXED_PROXY_PORT}/v1"

    def start(self) -> None:
        self._codex.start()
        self._claude.start()

        _RouterHandler.codex_port = self._codex._config.proxy_port
        _RouterHandler.claude_port = self._claude._config.proxy_port
        _RouterHandler.codex_model = self._codex._config.model
        _RouterHandler.claude_model = self._claude._config.model

        self._server = HTTPServer(("127.0.0.1", MIXED_PROXY_PORT), _RouterHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info(
            "Mixed proxy started on port %d (codex=%s@%d, claude=%s@%d)",
            MIXED_PROXY_PORT,
            self._codex._config.model, self._codex._config.proxy_port,
            self._claude._config.model, self._claude._config.proxy_port,
        )

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
            self._thread = None
        self._codex.stop()
        self._claude.stop()
        logger.info("Mixed proxy stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
