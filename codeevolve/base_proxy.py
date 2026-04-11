"""Shared base classes for CLI-to-OpenAI proxy servers.

CodexProxy and ClaudeProxy both translate OpenAI-compatible HTTP requests
into CLI invocations. This module provides the shared handler (do_GET,
_respond_json, _build_prompt) and proxy lifecycle (start/stop/context manager).
Subclass handlers override ``_invoke_cli`` to provide the actual CLI call.
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

logger = logging.getLogger(__name__)


class BaseProxyHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible HTTP handler that delegates CLI invocation to subclasses."""

    # Subclass must set this to identify the proxy in logs.
    proxy_name: str = "proxy"
    model: str = ""

    def log_message(self, format, *args):
        logger.debug("%s: %s", self.proxy_name, format % args)

    def do_POST(self):
        if "/chat/completions" not in self.path:
            self.send_error(404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))

        messages = body.get("messages", [])
        prompt = self._build_prompt(messages)

        response_text = self._invoke_cli(prompt)

        openai_response = {
            "id": f"{self.proxy_name}-0",
            "object": "chat.completion",
            "model": self.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

        self._respond_json(openai_response)

    def _invoke_cli(self, prompt: str) -> str:
        """Execute the CLI tool with the given prompt. Override in subclasses."""
        raise NotImplementedError

    def do_GET(self):
        if self.path in ("/health", "/v1/health"):
            self._respond_json({"status": "ok"})
        elif self.path == "/v1/models":
            self._respond_json({
                "data": [{"id": self.model, "object": "model"}],
            })
        else:
            self.send_error(404)

    def _respond_json(self, data: dict):
        payload = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    @staticmethod
    def _build_prompt(messages: list[dict]) -> str:
        """Combine an OpenAI messages array into a single prompt string."""
        system = ""
        user_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system = content
            elif role == "user":
                user_parts.append(content)
            elif role == "assistant":
                user_parts.append(f"Previous response:\n{content}")

        user_text = "\n\n".join(user_parts)

        return (
            f"{system}\n\n"
            f"{user_text}\n\n"
            "IMPORTANT: Output ONLY the code. Do NOT explain, do NOT ask "
            "questions, do NOT offer to help. Just output the code."
        )


class BaseProxy:
    """Shared proxy lifecycle: start/stop/context manager."""

    def __init__(self, config, handler_class: type[BaseProxyHandler]):
        self._config = config
        self._handler_class = handler_class
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def api_base(self) -> str:
        return f"http://localhost:{self._config.proxy_port}/v1"

    @property
    def model_name(self) -> str:
        return self._config.model

    def _configure_handler(self) -> None:
        """Set handler class attributes before accepting connections. Override in subclasses."""
        raise NotImplementedError

    def start(self) -> None:
        self._configure_handler()
        self._server = HTTPServer(
            ("127.0.0.1", self._config.proxy_port), self._handler_class,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
            self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
