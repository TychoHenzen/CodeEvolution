"""Lightweight HTTP proxy that translates OpenAI API calls to codex exec.

Starts a local server that accepts ``/v1/chat/completions`` POST requests
(the same interface llama-server and OpenAI expose) and fulfils them by
shelling out to ``codex exec``.  This lets OpenEvolve and the evaluator
use Codex as a backend without any changes to their LLM client code.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys

from codeevolve.base_proxy import BaseProxyHandler, BaseProxy

logger = logging.getLogger(__name__)


def _find_codex(configured_path: str) -> str:
    """Resolve the codex CLI binary path."""
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


class _ProxyHandler(BaseProxyHandler):
    """Translates a single OpenAI-compatible request into a ``codex exec`` call."""

    proxy_name = "codex-proxy"
    codex_path: str = "codex"
    model: str = "gpt-5.4-mini"
    timeout: int = 300

    def _invoke_cli(self, prompt: str) -> str:
        logger.info("codex-proxy: calling codex exec (%d-char prompt)", len(prompt))
        try:
            result = subprocess.run(
                [self.codex_path, "exec",
                 "-m", self.model,
                 "--full-auto"],
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout,
            )
            response_text = result.stdout.strip()
            preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
            logger.info(
                "codex exec finished (rc=%d, %d-char response): %s",
                result.returncode, len(response_text), preview,
            )
            if not response_text:
                logger.warning(
                    "codex exec: parsed empty response.\n"
                    "  stdout (%d bytes): %r\n"
                    "  stderr (%d bytes): %r",
                    len(result.stdout), result.stdout[:500],
                    len(result.stderr), result.stderr[:500],
                )
            return response_text
        except subprocess.TimeoutExpired:
            logger.warning("codex exec timed out after %ds", self.timeout)
            return ""
        except Exception as e:
            logger.error("codex exec failed: %s", e)
            return ""


class CodexProxy(BaseProxy):
    """Local HTTP server that translates OpenAI API calls to ``codex exec``."""

    def __init__(self, config):
        super().__init__(config, _ProxyHandler)
        self._codex_path = _find_codex(config.cli_path)

    def _configure_handler(self) -> None:
        _ProxyHandler.codex_path = self._codex_path
        _ProxyHandler.model = self._config.model
        _ProxyHandler.timeout = self._config.timeout

    def start(self) -> None:
        super().start()
        logger.info(
            "Codex proxy started on port %d (model=%s, codex=%s)",
            self._config.proxy_port, self._config.model, self._codex_path,
        )

    def stop(self) -> None:
        super().stop()
        logger.info("Codex proxy stopped")
