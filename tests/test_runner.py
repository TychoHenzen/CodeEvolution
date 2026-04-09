from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Optional

import pytest

from codeevolve.config import load_config
from codeevolve.runner import (
    validate_ollama,
    build_openevolve_config_yaml,
    format_iteration_line,
)


@patch("codeevolve.runner.OpenAI")
def test_validate_ollama_success(mock_openai_cls):
    mock_client = MagicMock()
    mock_client.models.list.return_value = MagicMock(
        data=[MagicMock(id="qwen2.5-coder:7b-instruct-q4_K_M"), MagicMock(id="qwen2.5-coder:1.5b-instruct-q4_K_M")]
    )
    mock_openai_cls.return_value = mock_client
    config = load_config()
    errors = validate_ollama(config)
    assert errors == []


@patch("codeevolve.runner.OpenAI")
def test_validate_ollama_missing_model(mock_openai_cls):
    mock_client = MagicMock()
    mock_client.models.list.return_value = MagicMock(
        data=[MagicMock(id="qwen2.5-coder:7b-instruct-q4_K_M")]
    )
    mock_openai_cls.return_value = mock_client
    config = load_config()
    errors = validate_ollama(config)
    assert len(errors) == 1
    assert "qwen2.5-coder:1.5b-instruct-q4_K_M" in errors[0]


@patch("codeevolve.runner.OpenAI")
def test_validate_ollama_unreachable(mock_openai_cls):
    mock_openai_cls.side_effect = Exception("Connection refused")
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
