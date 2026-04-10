"""Lightweight HTTP proxy that translates OpenAI API calls to Claude CLI.

Starts a local server that accepts ``/v1/chat/completions`` POST requests
(the same interface llama-server and OpenAI expose) and fulfils them by
shelling out to ``claude -p``.  This lets OpenEvolve and the evaluator
use Claude Code as a backend without any changes to their LLM client code.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

logger = logging.getLogger(__name__)


def _find_claude(configured_path: str) -> str:
    """Resolve the claude CLI binary path."""
    if "/" in configured_path or "\\" in configured_path:
        return configured_path
    found = shutil.which(configured_path)
    if found:
        return found
    if sys.platform == "win32":
        for ext in (".exe", ".cmd"):
            found = shutil.which(configured_path + ext)
            if found:
                return found
    return configured_path


class _ProxyHandler(BaseHTTPRequestHandler):
    """Translates a single OpenAI-compatible request into a ``claude -p`` call."""

    # Configured by ClaudeProxy before the server starts accepting.
    claude_path: str = "claude"
    model: str = "haiku"
    effort: str = "low"
    timeout: int = 300

    def log_message(self, format, *args):
        logger.debug("claude-proxy: %s", format % args)

    # ------------------------------------------------------------------
    # POST /v1/chat/completions
    # ------------------------------------------------------------------
    def do_POST(self):
        if "/chat/completions" not in self.path:
            self.send_error(404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))

        messages = body.get("messages", [])
        prompt = self._build_prompt(messages)

        logger.info("claude-proxy: calling claude -p (%d-char prompt)", len(prompt))

        try:
            # Pipe the prompt via stdin to avoid the Windows ~8191-char
            # command-line limit.  Claude CLI treats piped stdin as
            # user-provided context for the -p prompt.
            result = subprocess.run(
                [self.claude_path,
                 "--model", self.model,
                 "--no-session-persistence",
                 "--effort", self.effort,
                 "-p",
                 "Follow the instructions provided via stdin. "
                 "Output ONLY the code."],
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout,
            )
            response_text = result.stdout.strip()
            preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
            logger.info(
                "claude -p finished (rc=%d, %d-char response): %s",
                result.returncode, len(response_text), preview,
            )
            if not response_text:
                logger.warning(
                    "claude -p: empty response.\n"
                    "  stdout (%d bytes): %r\n"
                    "  stderr (%d bytes): %r",
                    len(result.stdout), result.stdout[:500],
                    len(result.stderr), result.stderr[:500],
                )
        except subprocess.TimeoutExpired:
            logger.warning("claude -p timed out after %ds", self.timeout)
            response_text = ""
        except Exception as e:
            logger.error("claude -p failed: %s", e)
            response_text = ""

        openai_response = {
            "id": "claude-proxy-0",
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

    # ------------------------------------------------------------------
    # GET /health, /v1/models
    # ------------------------------------------------------------------
    def do_GET(self):
        if self.path in ("/health", "/v1/health"):
            self._respond_json({"status": "ok"})
        elif self.path == "/v1/models":
            self._respond_json({
                "data": [{"id": self.model, "object": "model"}],
            })
        else:
            self.send_error(404)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _respond_json(self, data: dict):
        payload = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    @staticmethod
    def _build_prompt(messages: list[dict]) -> str:
        """Combine an OpenAI messages array into a direct task for claude -p.

        Claude CLI is invoked in non-interactive print mode (-p).  We
        combine system + user messages into a single prompt with an
        explicit output directive so Claude just produces code.
        """
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


class ClaudeProxy:
    """Local HTTP server that translates OpenAI API calls to ``claude -p``."""

    def __init__(self, config):
        self._config = config
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._claude_path = _find_claude(config.cli_path)

    @property
    def api_base(self) -> str:
        return f"http://localhost:{self._config.proxy_port}/v1"

    @property
    def model_name(self) -> str:
        return self._config.model

    def start(self) -> None:
        _ProxyHandler.claude_path = self._claude_path
        _ProxyHandler.model = self._config.model
        _ProxyHandler.effort = self._config.effort
        _ProxyHandler.timeout = self._config.timeout

        self._server = HTTPServer(
            ("127.0.0.1", self._config.proxy_port), _ProxyHandler,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True,
        )
        self._thread.start()
        logger.info(
            "Claude proxy started on port %d (model=%s, effort=%s, claude=%s)",
            self._config.proxy_port, self._config.model,
            self._config.effort, self._claude_path,
        )

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
            self._thread = None
            logger.info("Claude proxy stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
