import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeevolve.evaluator.llm_judge import (
    LlmJudgment,
    build_judgment_prompt,
    parse_judgment_response,
    judge_code,
    _normalize_score,
)


def test_build_judgment_prompt():
    diff = "- fn add(a: i32) -> i32 { a }\n+ fn add(a: i32, b: i32) -> i32 { a + b }"
    dimensions = ["readability", "rust_idiomaticity"]
    prompt = build_judgment_prompt(diff, dimensions)
    assert "readability" in prompt
    assert "rust_idiomaticity" in prompt
    assert "fn add" in prompt
    assert "-0.99 to +0.99" in prompt


def test_parse_judgment_response_valid():
    response = json.dumps({
        "readability": 0.5,
        "rust_idiomaticity": -0.3,
        "maintainability": 0.0,
        "design": 0.75,
    })
    scores = parse_judgment_response(response, ["readability", "rust_idiomaticity", "maintainability", "design"])
    assert scores == {"readability": 0.5, "rust_idiomaticity": -0.3, "maintainability": 0.0, "design": 0.75}


def test_parse_judgment_response_with_reasoning():
    response = """Here is my analysis...

```json
{"readability": 0.4, "rust_idiomaticity": -0.2}
```"""
    scores = parse_judgment_response(response, ["readability", "rust_idiomaticity"])
    assert scores == {"readability": 0.4, "rust_idiomaticity": -0.2}


def test_parse_judgment_response_invalid():
    scores = parse_judgment_response("not json at all", ["readability"])
    assert scores == {}


def test_parse_judgment_response_clamps_scores():
    response = json.dumps({"readability": 10, "design": -5})
    scores = parse_judgment_response(response, ["readability", "design"])
    assert scores["readability"] == 0.99
    assert scores["design"] == -0.99


def test_normalize_score():
    assert _normalize_score(0.0) == pytest.approx(0.5)
    assert _normalize_score(0.99) == pytest.approx(0.995)
    assert _normalize_score(-0.99) == pytest.approx(0.005)
    assert _normalize_score(0.5) == pytest.approx(0.75)
    assert _normalize_score(-0.5) == pytest.approx(0.25)


@patch("codeevolve.evaluator.llm_judge.get_git_diff")
@patch("codeevolve.evaluator.llm_judge._call_llm")
def test_judge_code_aggregates_runs(mock_call, mock_diff):
    """judge_code runs N times, takes medians of raw scores, then normalizes."""
    mock_diff.return_value = "- old\n+ new"
    mock_call.side_effect = [
        json.dumps({"readability": 0.4, "design": 0.6}),
        json.dumps({"readability": 0.8, "design": 0.6}),
        json.dumps({"readability": 0.5, "design": -0.2}),
    ]
    result = judge_code(
        file_path=Path("/fake/file.rs"),
        api_base="http://localhost:11434/v1",
        model="test-model",
        dimensions=["readability", "design"],
        num_runs=3,
    )
    # Raw medians: readability=0.5, design=0.6
    # Normalized: readability=(0.5+1)/2=0.75, design=(0.6+1)/2=0.8
    assert result.dimension_scores["readability"] == pytest.approx(0.75)
    assert result.dimension_scores["design"] == pytest.approx(0.8)
    # Combined: raw mean = (0.5+0.6)/2 = 0.55, normalized = (0.55+1)/2 = 0.775
    assert result.combined_score == pytest.approx(0.775)


@patch("codeevolve.evaluator.llm_judge.get_git_diff")
def test_judge_code_no_diff_returns_neutral(mock_diff):
    """No git diff means neutral 0.5 score."""
    mock_diff.return_value = ""
    result = judge_code(
        file_path=Path("/fake/file.rs"),
        api_base="http://localhost:11434/v1",
        model="test-model",
        dimensions=["readability", "design"],
    )
    assert result.combined_score == 0.5
    assert result.dimension_scores == {"readability": 0.5, "design": 0.5}
