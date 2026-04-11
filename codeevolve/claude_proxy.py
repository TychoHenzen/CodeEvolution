"""Lightweight HTTP proxy that translates OpenAI API calls to Claude CLI.

Starts a local server that accepts ``/v1/chat/completions`` POST requests
(the same interface llama-server and OpenAI expose) and fulfils them by
shelling out to ``claude -p``.  This lets OpenEvolve and the evaluator
use Claude Code as a backend without any changes to their LLM client code.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys

from codeevolve.base_proxy import BaseProxyHandler, BaseProxy

logger = logging.getLogger(__name__)


def _find_claude(configured_path: str) -> list[str]:
    """Resolve the claude CLI binary as a command list.

    On Windows, if the binary isn't found natively, invoke it through
    WSL since the Claude Code CLI is typically installed there.
    """
    if "/" in configured_path or "\\" in configured_path:
        return [configured_path]
    found = shutil.which(configured_path)
    if found:
        return [found]
    if sys.platform == "win32":
        for ext in (".exe", ".cmd"):
            found = shutil.which(configured_path + ext)
            if found:
                return [found]
        wsl = shutil.which("wsl")
        if wsl:
            try:
                result = subprocess.run(
                    [wsl, "bash", "-lc", f"which {configured_path}"],
                    capture_output=True, text=True, timeout=10,
                )
                linux_path = result.stdout.strip()
                if linux_path and result.returncode == 0:
                    return [wsl, linux_path]
            except Exception:
                pass
            return [wsl, configured_path]
    return [configured_path]


class _ProxyHandler(BaseProxyHandler):
    """Translates a single OpenAI-compatible request into a ``claude -p`` call."""

    proxy_name = "claude-proxy"
    claude_cmd: list[str] = ["claude"]
    model: str = "haiku"
    effort: str = "low"
    timeout: int = 300

    def _invoke_cli(self, prompt: str) -> str:
        logger.info("claude-proxy: calling claude -p (%d-char prompt)", len(prompt))
        try:
            result = subprocess.run(
                [*self.claude_cmd,
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
            return response_text
        except subprocess.TimeoutExpired:
            logger.warning("claude -p timed out after %ds", self.timeout)
            return ""
        except Exception as e:
            logger.error("claude -p failed: %s", e)
            return ""


class ClaudeProxy(BaseProxy):
    """Local HTTP server that translates OpenAI API calls to ``claude -p``."""

    def __init__(self, config):
        super().__init__(config, _ProxyHandler)
        self._claude_path = _find_claude(config.cli_path)

    def _configure_handler(self) -> None:
        _ProxyHandler.claude_cmd = self._claude_path
        _ProxyHandler.model = self._config.model
        _ProxyHandler.effort = self._config.effort
        _ProxyHandler.timeout = self._config.timeout

    def start(self) -> None:
        super().start()
        logger.info(
            "Claude proxy started on port %d (model=%s, effort=%s, claude=%s)",
            self._config.proxy_port, self._config.model,
            self._config.effort, " ".join(self._claude_path),
        )

    def stop(self) -> None:
        super().stop()
        logger.info("Claude proxy stopped")
