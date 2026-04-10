from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeevolve.config import load_config
from codeevolve.evaluator.pipeline import (
    EvaluationPipeline,
    EvaluationResult,
    parse_evolve_block,
    splice_evolve_block,
)


@pytest.fixture
def pipeline(tmp_path):
    config = load_config()
    source_file = tmp_path / "lib.rs"
    source_file.write_text("fn main() {}")
    return EvaluationPipeline(config, tmp_path, source_file)


@pytest.fixture
def candidate_file(tmp_path):
    """A temp file simulating what OpenEvolve passes to the evaluator.

    Uses different code than the source fixture so dedup doesn't trigger.
    """
    f = tmp_path / "candidate.rs"
    f.write_text("fn main() { println!(\"hello\"); }")
    return f


def test_evaluation_result_fields():
    r = EvaluationResult(
        passed_gates=True,
        combined_score=0.75,
        static_score=0.8,
        perf_score=1.05,
        llm_score=0.0,
    )
    assert r.combined_score == 0.75


@patch("codeevolve.evaluator.pipeline.attempt_fix", return_value=None)
@patch("codeevolve.evaluator.pipeline.run_cargo_build")
def test_pipeline_build_failure_returns_zero(mock_build, mock_fix, pipeline, candidate_file):
    mock_build.return_value = MagicMock(success=False, error_output="error", elapsed_seconds=1.0)
    result = pipeline.evaluate(str(candidate_file))
    assert result.passed_gates is False
    assert result.combined_score == 0.0


@patch("codeevolve.evaluator.pipeline.attempt_fix", return_value=None)
@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_build")
def test_pipeline_test_failure_returns_zero(mock_build, mock_test, mock_fix, pipeline, candidate_file):
    mock_build.return_value = MagicMock(success=True, elapsed_seconds=1.0)
    mock_test.return_value = MagicMock(success=False, error_output="test failed", tests_passed=0, tests_failed=1, elapsed_seconds=1.0)
    result = pipeline.evaluate(str(candidate_file))
    assert result.passed_gates is False
    assert result.combined_score == 0.0


@patch("codeevolve.evaluator.pipeline.judge_code")
@patch("codeevolve.evaluator.pipeline.measure_binary_size")
@patch("codeevolve.evaluator.pipeline.measure_compile_time")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_build")
def test_pipeline_full_pass(mock_build, mock_test, mock_clippy, mock_compile_time, mock_binary_size, mock_judge, pipeline, candidate_file):
    mock_build.return_value = MagicMock(success=True, elapsed_seconds=1.0)
    mock_test.return_value = MagicMock(success=True, tests_passed=5, tests_failed=0, elapsed_seconds=1.0)
    mock_clippy.return_value = MagicMock(success=True, warnings=[], warning_counts={}, elapsed_seconds=0.5)
    mock_compile_time.return_value = 2.5
    mock_binary_size.return_value = 1_000_000
    mock_judge.return_value = MagicMock(combined_score=0.7)

    result = pipeline.evaluate(str(candidate_file))
    assert result.passed_gates is True
    assert result.combined_score > 0
    assert result.static_score == 1.0  # no clippy warnings
    assert result.perf_score == 1.0  # baseline ratio


@patch("codeevolve.evaluator.pipeline.measure_binary_size")
@patch("codeevolve.evaluator.pipeline.measure_compile_time")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_build")
def test_pipeline_skips_llm_if_not_top_quartile(mock_build, mock_test, mock_clippy, mock_compile_time, mock_binary_size, pipeline, candidate_file):
    # Enable top_quartile_only for this test
    pipeline.config.llm_judgment.top_quartile_only = True

    mock_build.return_value = MagicMock(success=True, elapsed_seconds=1.0)
    mock_test.return_value = MagicMock(success=True, tests_passed=5, tests_failed=0, elapsed_seconds=1.0)
    mock_clippy.return_value = MagicMock(success=True, warnings=[{"code": "clippy::style"}] * 20, warning_counts={"style": 20}, elapsed_seconds=0.5)
    mock_compile_time.return_value = 10.0
    mock_binary_size.return_value = 5_000_000

    # Fill history with high scores so this one won't be top quartile
    pipeline._score_history = [0.9, 0.95, 0.85, 0.88, 0.92, 0.87, 0.91, 0.86]

    result = pipeline.evaluate(str(candidate_file))
    assert result.passed_gates is True
    assert result.llm_score == 0.0


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------

def test_pipeline_rejects_candidate_identical_to_original(pipeline, tmp_path):
    """Candidate with same code as the source file is rejected as duplicate."""
    dup_file = tmp_path / "dup.rs"
    dup_file.write_text("fn main() {}")  # same as source_file fixture
    result = pipeline.evaluate(str(dup_file))
    assert result.passed_gates is False
    assert result.combined_score == 0.0
    assert "identical to original" in result.error


@patch("codeevolve.evaluator.pipeline.attempt_fix", return_value=None)
@patch("codeevolve.evaluator.pipeline.run_cargo_build")
def test_pipeline_rejects_repeated_candidate(mock_build, mock_fix, pipeline, tmp_path):
    """A candidate that was already seen in a prior iteration is rejected."""
    mock_build.return_value = MagicMock(success=False, error_output="err", elapsed_seconds=0.5)

    # First evaluation of novel candidate (will fail build, but still recorded)
    f1 = tmp_path / "v1.rs"
    f1.write_text("fn main() { let x = 1; }")
    pipeline.evaluate(str(f1))

    # Second evaluation with identical code should be dedup-rejected
    f2 = tmp_path / "v2.rs"
    f2.write_text("fn main() { let x = 1; }")
    result = pipeline.evaluate(str(f2))
    assert result.passed_gates is False
    assert "duplicate of previous candidate" in result.error


@patch("codeevolve.evaluator.pipeline.attempt_fix", return_value=None)
@patch("codeevolve.evaluator.pipeline.run_cargo_build")
def test_pipeline_accepts_novel_candidates(mock_build, mock_fix, pipeline, tmp_path):
    """Distinct candidates are accepted (not rejected by dedup)."""
    mock_build.return_value = MagicMock(success=False, error_output="err", elapsed_seconds=0.5)

    for i in range(5):
        f = tmp_path / f"v{i}.rs"
        f.write_text(f"fn main() {{ let x = {i}; }}")
        result = pipeline.evaluate(str(f))
        # Should NOT be rejected as duplicate (build fails, but not a dedup issue)
        assert "duplicate" not in (result.error or "")


def test_pipeline_tracks_consecutive_duplicates(pipeline, tmp_path):
    """Consecutive duplicate counter increments correctly."""
    dup_file = tmp_path / "dup.rs"
    dup_file.write_text("fn main() {}")  # same as original

    for i in range(3):
        pipeline.evaluate(str(dup_file))

    assert pipeline._consecutive_duplicates == 3
    assert pipeline._total_duplicates == 3


# ---------------------------------------------------------------------------
# EVOLVE-BLOCK parsing / splicing tests
# ---------------------------------------------------------------------------

_SAMPLE_FILE = """\
use std::io;
// EVOLVE-BLOCK-START
fn foo() -> i32 {
    42
}
// EVOLVE-BLOCK-END
#[cfg(test)]
mod tests;
"""


def test_parse_evolve_block_basic():
    parsed = parse_evolve_block(_SAMPLE_FILE)
    assert parsed is not None
    prefix, content, suffix = parsed
    assert "EVOLVE-BLOCK-START" in prefix
    assert "fn foo()" in content
    assert "mod tests;" in suffix
    assert "EVOLVE-BLOCK-END" in suffix


def test_parse_evolve_block_no_markers():
    assert parse_evolve_block("fn main() {}") is None


def test_splice_roundtrip():
    """Parsing then splicing with same content reproduces original."""
    parsed = parse_evolve_block(_SAMPLE_FILE)
    assert parsed is not None
    prefix, content, suffix = parsed
    result = splice_evolve_block(prefix, content, suffix)
    assert result == _SAMPLE_FILE


def test_splice_replaces_content():
    """Splicing with new content preserves prefix/suffix."""
    parsed = parse_evolve_block(_SAMPLE_FILE)
    assert parsed is not None
    prefix, _, suffix = parsed
    result = splice_evolve_block(prefix, "fn bar() -> i32 {\n    99\n}\n", suffix)
    assert "fn bar()" in result
    assert "fn foo()" not in result
    assert "mod tests;" in result
    assert "use std::io;" in result
    assert "EVOLVE-BLOCK-START" in result
    assert "EVOLVE-BLOCK-END" in result


@patch("codeevolve.evaluator.pipeline.attempt_fix", return_value=None)
@patch("codeevolve.evaluator.pipeline.run_cargo_build")
def test_pipeline_enforces_evolve_block(mock_build, mock_fix, tmp_path):
    """LLM output that removes markers still gets spliced correctly."""
    config = load_config()
    source_file = tmp_path / "lib.rs"
    source_file.write_text(
        "// EVOLVE-BLOCK-START\nfn foo() { 1 }\n// EVOLVE-BLOCK-END\nmod tests;\n"
    )
    pipeline = EvaluationPipeline(config, tmp_path, source_file)

    mock_build.return_value = MagicMock(success=False, error_output="err", elapsed_seconds=0.5)

    # Candidate has NO markers — LLM stripped them
    candidate = tmp_path / "candidate.rs"
    candidate.write_text("fn foo() { 2 }\nfn evil_extra() {}")

    pipeline.evaluate(str(candidate))

    # Source should have been restored, but during eval it should have
    # had markers + suffix intact. Verify the pipeline stored the structure.
    assert pipeline._evolve_prefix is not None
    assert "EVOLVE-BLOCK-START" in pipeline._evolve_prefix
    assert "mod tests;" in pipeline._evolve_suffix
