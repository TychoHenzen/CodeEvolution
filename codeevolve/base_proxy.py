"""Shared base classes for CLI-to-OpenAI proxy servers.

CodexProxy and ClaudeProxy both translate OpenAI-compatible HTTP requests
into CLI invocations. This module provides the shared handler (do_GET,
_respond_json, _build_prompt) and proxy lifecycle (start/stop/context manager).
Subclass handlers override ``_invoke_cli`` to provide the actual CLI call.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

logger = logging.getLogger(__name__)

# On Windows, prevent each subprocess from spawning a conhost.exe.
# CLI proxies don't need a console window.
_CREATIONFLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


# ---------------------------------------------------------------------------
# Subprocess lifecycle tracking
# ---------------------------------------------------------------------------
# Thread-safe tracking of in-flight CLI subprocesses so they can be
# killed during proxy shutdown.  Without this, Ctrl+C or timeouts kill
# only the wsl.exe shim while the Linux-side process keeps running.

_active_children: set[subprocess.Popen] = set()
_children_lock = threading.Lock()


def _track(proc: subprocess.Popen) -> subprocess.Popen:
    """Register a subprocess for cleanup on proxy shutdown."""
    with _children_lock:
        _active_children.add(proc)
    return proc


def _untrack(proc: subprocess.Popen) -> None:
    """Remove a subprocess from cleanup tracking."""
    with _children_lock:
        _active_children.discard(proc)


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill a process and its entire tree.

    On Windows, ``taskkill /T`` kills the process tree — critical for
    wsl.exe subprocesses where a plain kill only terminates the shim.
    """
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        try:
            subprocess.call(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except Exception:
            proc.kill()
    else:
        proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _kill_all_children() -> None:
    """Kill all tracked child processes and their trees."""
    with _children_lock:
        children = list(_active_children)
    for proc in children:
        _kill_tree(proc)
    with _children_lock:
        _active_children.clear()


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
        request_model = body.get("model", "") or self.model
        prompt = self._build_prompt(messages)

        response_text = self._invoke_cli(prompt, model=request_model)

        openai_response = {
            "id": f"{self.proxy_name}-0",
            "object": "chat.completion",
            "model": request_model,
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

    def _invoke_cli(self, prompt: str, model: str = "") -> str:
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
            "IMPORTANT: Follow the formatting instructions above EXACTLY. "
            "Do NOT explain, do NOT ask questions, do NOT add commentary."
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
        _kill_all_children()
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
