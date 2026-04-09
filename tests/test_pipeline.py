from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeevolve.config import load_config
from codeevolve.evaluator.pipeline import EvaluationPipeline, EvaluationResult


@pytest.fixture
def pipeline():
    config = load_config()
    return EvaluationPipeline(config)


def test_evaluation_result_fields():
    r = EvaluationResult(
        passed_gates=True,
        combined_score=0.75,
        static_score=-5,
        perf_score=0.8,
        llm_score=0.0,
    )
    assert r.combined_score == 0.75


@patch("codeevolve.evaluator.pipeline.run_cargo_build")
def test_pipeline_build_failure_returns_zero(mock_build, pipeline):
    mock_build.return_value = MagicMock(success=False, error_output="error", elapsed_seconds=1.0)
    result = pipeline.evaluate("/fake/path")
    assert result.passed_gates is False
    assert result.combined_score == 0.0


@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_build")
def test_pipeline_test_failure_returns_zero(mock_build, mock_test, pipeline):
    mock_build.return_value = MagicMock(success=True, elapsed_seconds=1.0)
    mock_test.return_value = MagicMock(success=False, error_output="test failed", tests_passed=0, tests_failed=1, elapsed_seconds=1.0)
    result = pipeline.evaluate("/fake/path")
    assert result.passed_gates is False
    assert result.combined_score == 0.0


@patch("codeevolve.evaluator.pipeline.judge_code")
@patch("codeevolve.evaluator.pipeline.measure_binary_size")
@patch("codeevolve.evaluator.pipeline.measure_compile_time")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_build")
@patch("pathlib.Path.read_text", return_value="fn main() {}")
def test_pipeline_full_pass(mock_read_text, mock_build, mock_test, mock_clippy, mock_compile_time, mock_binary_size, mock_judge, pipeline):
    mock_build.return_value = MagicMock(success=True, elapsed_seconds=1.0)
    mock_test.return_value = MagicMock(success=True, tests_passed=5, tests_failed=0, elapsed_seconds=1.0)
    mock_clippy.return_value = MagicMock(success=True, warnings=[], warning_counts={}, elapsed_seconds=0.5)
    mock_compile_time.return_value = 2.5
    mock_binary_size.return_value = 1_000_000
    mock_judge.return_value = MagicMock(combined_score=4.0, dimension_scores={"readability": 4})

    # Force top-quartile by setting low history
    pipeline._score_history = [0.1, 0.2]

    result = pipeline.evaluate("/fake/path")
    assert result.passed_gates is True
    assert result.combined_score > 0


@patch("codeevolve.evaluator.pipeline.measure_binary_size")
@patch("codeevolve.evaluator.pipeline.measure_compile_time")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_build")
@patch("pathlib.Path.read_text", return_value="fn main() {}")
def test_pipeline_skips_llm_if_not_top_quartile(mock_read_text, mock_build, mock_test, mock_clippy, mock_compile_time, mock_binary_size, pipeline):
    mock_build.return_value = MagicMock(success=True, elapsed_seconds=1.0)
    mock_test.return_value = MagicMock(success=True, tests_passed=5, tests_failed=0, elapsed_seconds=1.0)
    mock_clippy.return_value = MagicMock(success=True, warnings=[{"code": "clippy::style"}] * 20, warning_counts={"style": 20}, elapsed_seconds=0.5)
    mock_compile_time.return_value = 10.0
    mock_binary_size.return_value = 5_000_000

    # Fill history with high scores so this one won't be top quartile
    pipeline._score_history = [0.9, 0.95, 0.85, 0.88, 0.92, 0.87, 0.91, 0.86]

    result = pipeline.evaluate("/fake/path")
    assert result.passed_gates is True
    assert result.llm_score == 0.0
