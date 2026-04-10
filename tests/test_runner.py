from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Optional

import pytest

from codeevolve.config import load_config
from codeevolve.runner import (
    validate_server,
    build_openevolve_config_yaml,
    format_iteration_line,
    _normalize_llm_diffs,
    _run_multi_file,
)


@patch("codeevolve.runner.urlopen")
def test_validate_server_success(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b'{"status":"ok"}'
    mock_urlopen.return_value = mock_resp
    config = load_config()
    errors = validate_server(config)
    assert errors == []


@patch("codeevolve.runner.urlopen")
def test_validate_server_unreachable(mock_urlopen):
    from urllib.error import URLError
    mock_urlopen.side_effect = URLError("Connection refused")
    config = load_config()
    errors = validate_server(config)
    assert len(errors) == 1
    assert "Cannot connect" in errors[0]


def test_build_openevolve_config_yaml(tmp_path: Path):
    config = load_config()
    yaml_path = build_openevolve_config_yaml(config, tmp_path)
    assert yaml_path.exists()
    content = yaml_path.read_text()
    assert "api_base" in content


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


def test_run_multi_file_uses_workspace_bundle(sample_workspace, tmp_path):
    """When a workspace is detected, _run_multi_file uses create_workspace_bundle."""
    config = load_config()
    config_path = sample_workspace / ".codeevolve" / "evolution.yaml"
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    source_files = sorted((sample_workspace / "crates").rglob("*.rs"))
    # Filter to only files with EVOLVE-BLOCK markers
    marked = [f for f in source_files if "EVOLVE-BLOCK-START" in f.read_text()]

    # Patch at SOURCE modules — _run_multi_file uses local imports, so
    # the from-import will pick up whatever is in the source module at
    # call time.
    with patch("openevolve.api.run_evolution") as mock_oe, \
         patch("codeevolve.bundler.create_workspace_bundle") as mock_ws_bundle, \
         patch("codeevolve.summary.summarize_files") as mock_summarize:
        mock_ws_bundle.return_value = "// bundle content"
        mock_summarize.return_value = {}
        mock_oe.return_value = MagicMock(best_code="pub fn improved() {}")

        # This should detect workspace and use create_workspace_bundle
        _run_multi_file(
            config, config_path, sample_workspace, marked,
            config_path.parent / "evaluator.py", output_dir,
        )
        mock_ws_bundle.assert_called_once()
