import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Optional

import pytest

from codeevolve.config import load_config
from codeevolve.runner import (
    validate_ollama,
    build_openevolve_config_yaml,
    format_iteration_line,
    _normalize_llm_diffs,
)


@patch("codeevolve.runner.urlopen")
def test_validate_ollama_success(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"models": [
        {"name": "qwen2.5-coder:7b-instruct-q4_K_M"},
        {"name": "qwen2.5-coder:1.5b-instruct-q4_K_M"},
    ]}).encode()
    mock_urlopen.return_value = mock_resp
    config = load_config()
    errors = validate_ollama(config)
    assert errors == []


@patch("codeevolve.runner.urlopen")
def test_validate_ollama_missing_model(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"models": [
        {"name": "some-other-model:latest"},
    ]}).encode()
    mock_urlopen.return_value = mock_resp
    config = load_config()
    errors = validate_ollama(config)
    assert any("qwen2.5-coder:7b-instruct-q4_K_M" in e for e in errors)
    assert any("Available models" in e for e in errors)


@patch("codeevolve.runner.urlopen")
def test_validate_ollama_unreachable(mock_urlopen):
    from urllib.error import URLError
    mock_urlopen.side_effect = URLError("Connection refused")
    config = load_config()
    errors = validate_ollama(config)
    assert len(errors) == 1
    assert "Connection refused" in errors[0]


def test_build_openevolve_config_yaml(tmp_path: Path):
    config = load_config()
    yaml_path = build_openevolve_config_yaml(config, tmp_path)
    assert yaml_path.exists()
    content = yaml_path.read_text()
    assert "qwen2.5-coder:7b-instruct-q4_K_M" in content


def test_format_iteration_line_success():
    line = format_iteration_line(
        iteration=12,
        total=500,
        file_changed="src/lib.rs",
        diff_lines=47,
        build_ok=True,
        build_time=1.2,
        tests_ok=True,
        tests_passed=23,
        tests_failed=0,
        clippy_warnings=3,
        parent_clippy_warnings=7,
        binary_size=1_400_000,
        parent_binary_size=1_500_000,
        llm_ran=False,
        score=0.72,
        best_score=0.81,
    )
    assert "12/500" in line
    assert "src/lib.rs" in line
    assert "improved" in line.lower() or "3 warnings" in line


def test_format_iteration_line_build_failure():
    line = format_iteration_line(
        iteration=13,
        total=500,
        file_changed="src/utils.rs",
        diff_lines=12,
        build_ok=False,
        build_time=0.5,
        error="error[E0308]: mismatched types",
    )
    assert "FAILED" in line
    assert "0.00" in line


class TestNormalizeLlmDiffs:
    """Tests for _normalize_llm_diffs which rewrites markdown-style diffs."""

    def test_canonical_format_unchanged(self):
        """Already-correct format should pass through untouched."""
        text = (
            "<<<<<<< SEARCH\nfn foo() {}\n=======\n"
            "fn foo() { bar(); }\n>>>>>>> REPLACE"
        )
        assert _normalize_llm_diffs(text) == text

    def test_markdown_h4_with_rust_fences(self):
        """#### SEARCH / #### REPLACE with ```rust fences should be rewritten."""
        text = (
            "#### SEARCH\n"
            "```rust\n"
            "fn foo() {}\n"
            "```\n\n"
            "#### REPLACE\n"
            "```rust\n"
            "fn foo() { bar(); }\n"
            "```"
        )
        result = _normalize_llm_diffs(text)
        assert "<<<<<<< SEARCH" in result
        assert "=======" in result
        assert ">>>>>>> REPLACE" in result
        assert "fn foo() {}" in result
        assert "fn foo() { bar(); }" in result
        assert "```" not in result

    def test_markdown_h3_with_plain_fences(self):
        """### SEARCH / ### REPLACE with bare ``` fences."""
        text = (
            "### SEARCH\n"
            "```\n"
            "let x = 1;\n"
            "```\n\n"
            "### REPLACE\n"
            "```\n"
            "let x = 2;\n"
            "```"
        )
        result = _normalize_llm_diffs(text)
        assert "<<<<<<< SEARCH" in result
        assert "let x = 1;" in result
        assert "let x = 2;" in result

    def test_multiple_blocks(self):
        """Multiple markdown diff blocks should all be rewritten."""
        text = (
            "#### SEARCH\n```rust\nfn a() {}\n```\n\n"
            "#### REPLACE\n```rust\nfn a() { x(); }\n```\n\n"
            "Some explanation text\n\n"
            "#### SEARCH\n```rust\nfn b() {}\n```\n\n"
            "#### REPLACE\n```rust\nfn b() { y(); }\n```"
        )
        result = _normalize_llm_diffs(text)
        assert result.count("<<<<<<< SEARCH") == 2
        assert result.count(">>>>>>> REPLACE") == 2

    def test_no_diffs_returns_text_unchanged(self):
        """Plain text with no diff patterns should pass through."""
        text = "Just some commentary about the code."
        assert _normalize_llm_diffs(text) == text
