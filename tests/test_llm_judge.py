import json
from unittest.mock import MagicMock, patch

import pytest

from codeevolve.evaluator.llm_judge import (
    LlmJudgment,
    build_judgment_prompt,
    parse_judgment_response,
    judge_code,
)


def test_build_judgment_prompt():
    code = "fn add(a: i32, b: i32) -> i32 { a + b }"
    dimensions = ["readability", "rust_idiomaticity"]
    prompt = build_judgment_prompt(code, dimensions)
    assert "readability" in prompt
    assert "rust_idiomaticity" in prompt
    assert "fn add" in prompt
    assert "0.0 to 1.0" in prompt


def test_parse_judgment_response_valid():
    response = json.dumps({
        "readability": 0.8,
        "rust_idiomaticity": 0.9,
        "maintainability": 0.6,
        "design": 0.75,
    })
    scores = parse_judgment_response(response, ["readability", "rust_idiomaticity", "maintainability", "design"])
    assert scores == {"readability": 0.8, "rust_idiomaticity": 0.9, "maintainability": 0.6, "design": 0.75}


def test_parse_judgment_response_with_reasoning():
    response = """Here is my analysis...

```json
{"readability": 0.8, "rust_idiomaticity": 0.6}
```"""
    scores = parse_judgment_response(response, ["readability", "rust_idiomaticity"])
    assert scores == {"readability": 0.8, "rust_idiomaticity": 0.6}


def test_parse_judgment_response_invalid():
    scores = parse_judgment_response("not json at all", ["readability"])
    assert scores == {}


def test_parse_judgment_response_clamps_scores():
    response = json.dumps({"readability": 10, "design": -1})
    scores = parse_judgment_response(response, ["readability", "design"])
    assert scores["readability"] == 1.0
    assert scores["design"] == 0.0


@patch("codeevolve.evaluator.llm_judge._call_ollama")
def test_judge_code_aggregates_runs(mock_call):
    """judge_code runs N times and takes medians."""
    mock_call.side_effect = [
        json.dumps({"readability": 0.6, "design": 0.8}),
        json.dumps({"readability": 0.9, "design": 0.8}),
        json.dumps({"readability": 0.7, "design": 0.4}),
    ]
    result = judge_code(
        code="fn main() {}",
        api_base="http://localhost:11434/v1",
        model="test-model",
        dimensions=["readability", "design"],
        num_runs=3,
    )
    assert result.dimension_scores["readability"] == 0.7  # median of [0.6, 0.9, 0.7]
    assert result.dimension_scores["design"] == 0.8  # median of [0.8, 0.8, 0.4]
    assert result.combined_score == 0.75  # mean of [0.7, 0.8]
