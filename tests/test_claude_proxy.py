from codeevolve.claude_proxy import ClaudeProxy, _ProxyHandler
from codeevolve.config import ClaudeConfig


class TestBuildPrompt:
    """Tests for combining OpenAI messages into a claude -p task."""

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


class TestClaudeProxy:
    """Tests for the proxy server lifecycle."""

    def test_api_base_property(self):
        config = ClaudeConfig(proxy_port=9999)
        proxy = ClaudeProxy(config)
        assert proxy.api_base == "http://localhost:9999/v1"

    def test_model_name_property(self):
        config = ClaudeConfig(model="haiku")
        proxy = ClaudeProxy(config)
        assert proxy.model_name == "haiku"

    def test_start_stop(self):
        config = ClaudeConfig(proxy_port=18925)
        proxy = ClaudeProxy(config)
        proxy.start()
        try:
            import urllib.request
            resp = urllib.request.urlopen(
                f"http://localhost:{config.proxy_port}/health", timeout=2,
            )
            assert resp.status == 200
        finally:
            proxy.stop()

    def test_context_manager(self):
        config = ClaudeConfig(proxy_port=18926)
        with ClaudeProxy(config) as proxy:
            import urllib.request
            resp = urllib.request.urlopen(
                f"http://localhost:{config.proxy_port}/health", timeout=2,
            )
            assert resp.status == 200
