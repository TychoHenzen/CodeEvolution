import json
from unittest.mock import patch, MagicMock

import pytest

from codeevolve.codex_proxy import CodexProxy, _ProxyHandler
from codeevolve.config import CodexConfig


class TestBuildPrompt:
    """Tests for combining OpenAI messages into a codex exec task."""

    def test_user_message_only(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = _ProxyHandler._build_prompt(messages)
        assert "Hello" in result
        assert "Output ONLY the code" in result

    def test_system_and_user(self):
        messages = [
            {"role": "system", "content": "You are an expert."},
            {"role": "user", "content": "Fix this code."},
        ]
        result = _ProxyHandler._build_prompt(messages)
        assert "You are an expert." in result
        assert "Fix this code." in result

    def test_multi_turn_conversation(self):
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second"},
        ]
        result = _ProxyHandler._build_prompt(messages)
        assert "Previous response:" in result
        assert "Second" in result


class TestCodexProxy:
    """Tests for the proxy server lifecycle."""

    def test_api_base_property(self):
        config = CodexConfig(proxy_port=9999)
        proxy = CodexProxy(config)
        assert proxy.api_base == "http://localhost:9999/v1"

    def test_model_name_property(self):
        config = CodexConfig(model="gpt-5.4-mini")
        proxy = CodexProxy(config)
        assert proxy.model_name == "gpt-5.4-mini"

    def test_start_stop(self):
        config = CodexConfig(proxy_port=18923)
        proxy = CodexProxy(config)
        proxy.start()
        try:
            # Server should be running
            import urllib.request
            resp = urllib.request.urlopen(
                f"http://localhost:{config.proxy_port}/health", timeout=2,
            )
            assert resp.status == 200
        finally:
            proxy.stop()

    def test_context_manager(self):
        config = CodexConfig(proxy_port=18924)
        with CodexProxy(config) as proxy:
            import urllib.request
            resp = urllib.request.urlopen(
                f"http://localhost:{config.proxy_port}/health", timeout=2,
            )
            assert resp.status == 200
