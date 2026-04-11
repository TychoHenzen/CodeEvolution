from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeevolve.config import load_config
from codeevolve.evaluator.pipeline import (
    EvaluationPipeline,
    EvaluationResult,
    _format_clippy_diagnostics,
    _truncate_artifact,
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
        perf_score=0.75,
        llm_score=0.0,
    )
    assert r.combined_score == 0.75


@patch("codeevolve.evaluator.pipeline.attempt_fix", return_value=None)
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_pipeline_build_failure_returns_zero(mock_clean, mock_clippy, mock_fix, pipeline, candidate_file):
    mock_clippy.return_value = MagicMock(success=False, error_output="error", elapsed_seconds=1.0)
    result = pipeline.evaluate(str(candidate_file))
    assert result.passed_gates is False
    assert result.combined_score == 0.0


@patch("codeevolve.evaluator.pipeline.attempt_fix", return_value=None)
@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_pipeline_test_failure_returns_zero(mock_clean, mock_clippy, mock_test, mock_fix, pipeline, candidate_file):
    mock_clippy.return_value = MagicMock(success=True, warnings=[], warning_counts={}, elapsed_seconds=1.0)
    mock_test.return_value = MagicMock(success=False, error_output="test failed", tests_passed=0, tests_failed=1, failed_test_names=["tests::test_something"], elapsed_seconds=1.0)
    result = pipeline.evaluate(str(candidate_file))
    assert result.passed_gates is False
    assert result.combined_score == 0.0


@patch("codeevolve.evaluator.pipeline.judge_code")
@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_pipeline_full_pass(mock_clean, mock_clippy, mock_test, mock_judge, pipeline, candidate_file):
    mock_clippy.return_value = MagicMock(success=True, warnings=[], warning_counts={}, elapsed_seconds=2.5)
    mock_test.return_value = MagicMock(success=True, tests_passed=5, tests_failed=0, elapsed_seconds=1.0)
    mock_judge.return_value = MagicMock(combined_score=0.7)

    # Disable binary size (no binary_package set by default)
    pipeline.config.benchmarks.binary_package = None
    pipeline.config.benchmarks.custom_command = None

    result = pipeline.evaluate(str(candidate_file))
    assert result.passed_gates is True
    assert result.combined_score > 0
    assert result.static_score == 1.0  # no clippy warnings
    assert result.perf_score == 0.5  # baseline: norm_perf = 1.0/2.0 = 0.5


@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_pipeline_skips_llm_if_not_top_quartile(mock_clean, mock_clippy, mock_test, pipeline, candidate_file):
    # Enable top_quartile_only for this test
    pipeline.config.llm_judgment.top_quartile_only = True
    pipeline.config.benchmarks.binary_package = None
    pipeline.config.benchmarks.custom_command = None

    mock_clippy.return_value = MagicMock(
        success=True,
        warnings=[{"code": "clippy::style"}] * 20,
        warning_counts={"style": 20},
        elapsed_seconds=2.5,
    )
    mock_test.return_value = MagicMock(success=True, tests_passed=5, tests_failed=0, elapsed_seconds=1.0)

    # Fill history with high scores so this one won't be top quartile
    pipeline._score_history = [0.9, 0.95, 0.85, 0.88, 0.92, 0.87, 0.91, 0.86]

    result = pipeline.evaluate(str(candidate_file))
    assert result.passed_gates is True
    assert result.llm_score == 0.0


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------

@patch("codeevolve.evaluator.pipeline.attempt_fix", return_value=None)
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_pipeline_initial_program_not_dedup_rejected(mock_clean, mock_clippy, mock_fix, pipeline, tmp_path):
    """The first evaluation (initial/seed program, identical to source) must not
    be rejected as a duplicate — OpenEvolve needs a real baseline score."""
    mock_clippy.return_value = MagicMock(success=False, error_output="err", elapsed_seconds=0.5)

    dup_file = tmp_path / "initial.rs"
    dup_file.write_text("fn main() {}")  # same as source_file fixture
    result = pipeline.evaluate(str(dup_file))
    assert "duplicate" not in (result.error or "")


@patch("codeevolve.evaluator.pipeline.attempt_fix", return_value=None)
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_pipeline_rejects_candidate_identical_to_original(mock_clean, mock_clippy, mock_fix, pipeline, tmp_path):
    """After the initial evaluation, a candidate identical to the original is
    rejected as a duplicate."""
    mock_clippy.return_value = MagicMock(success=False, error_output="err", elapsed_seconds=0.5)

    # First call — the initial/seed evaluation (must succeed, not dedup)
    initial = tmp_path / "initial.rs"
    initial.write_text("fn main() {}")
    pipeline.evaluate(str(initial))

    # Second call with identical code — should be rejected
    dup_file = tmp_path / "dup.rs"
    dup_file.write_text("fn main() {}")
    result = pipeline.evaluate(str(dup_file))
    assert result.passed_gates is False
    assert result.combined_score == 0.0
    assert "identical to original" in result.error


@patch("codeevolve.evaluator.pipeline.attempt_fix", return_value=None)
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_pipeline_rejects_repeated_candidate(mock_clean, mock_clippy, mock_fix, pipeline, tmp_path):
    """A candidate that was already seen in a prior iteration is rejected."""
    mock_clippy.return_value = MagicMock(success=False, error_output="err", elapsed_seconds=0.5)

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
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_pipeline_accepts_novel_candidates(mock_clean, mock_clippy, mock_fix, pipeline, tmp_path):
    """Distinct candidates are accepted (not rejected by dedup)."""
    mock_clippy.return_value = MagicMock(success=False, error_output="err", elapsed_seconds=0.5)

    for i in range(5):
        f = tmp_path / f"v{i}.rs"
        f.write_text(f"fn main() {{ let x = {i}; }}")
        result = pipeline.evaluate(str(f))
        # Should NOT be rejected as duplicate (build fails, but not a dedup issue)
        assert "duplicate" not in (result.error or "")


@patch("codeevolve.evaluator.pipeline.attempt_fix", return_value=None)
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_pipeline_tracks_consecutive_duplicates(mock_clean, mock_clippy, mock_fix, pipeline, tmp_path):
    """Consecutive duplicate counter increments correctly."""
    mock_clippy.return_value = MagicMock(success=False, error_output="err", elapsed_seconds=0.5)

    dup_file = tmp_path / "dup.rs"
    dup_file.write_text("fn main() {}")  # same as original

    # First call is the initial/seed evaluation (not a duplicate).
    # The subsequent 3 calls are all duplicates.
    for _ in range(4):
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
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_pipeline_enforces_evolve_block(mock_clean, mock_clippy, mock_fix, tmp_path):
    """LLM output that removes markers still gets spliced correctly."""
    config = load_config()
    source_file = tmp_path / "lib.rs"
    source_file.write_text(
        "// EVOLVE-BLOCK-START\nfn foo() { 1 }\n// EVOLVE-BLOCK-END\nmod tests;\n"
    )
    pipeline = EvaluationPipeline(config, tmp_path, source_file)

    mock_clippy.return_value = MagicMock(success=False, error_output="err", elapsed_seconds=0.5)

    # Candidate has NO markers — LLM stripped them
    candidate = tmp_path / "candidate.rs"
    candidate.write_text("fn foo() { 2 }\nfn evil_extra() {}")

    pipeline.evaluate(str(candidate))

    # Source should have been restored, but during eval it should have
    # had markers + suffix intact. Verify the pipeline stored the structure.
    assert pipeline._evolve_prefix is not None
    assert "EVOLVE-BLOCK-START" in pipeline._evolve_prefix
    assert "mod tests;" in pipeline._evolve_suffix


# ---------------------------------------------------------------------------
# Metrics normalization tests
# ---------------------------------------------------------------------------

@patch("codeevolve.evaluator.pipeline.judge_code")
@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_perf_score_is_norm_perf_not_raw_ratio(mock_clean, mock_clippy, mock_test, mock_judge, pipeline, candidate_file):
    """perf_score should be norm_perf (clamped ratio/2), not the raw perf_ratio."""
    mock_clippy.return_value = MagicMock(success=True, warnings=[], warning_counts={}, elapsed_seconds=2.0)
    mock_test.return_value = MagicMock(success=True, tests_passed=3, tests_failed=0, elapsed_seconds=0.5)
    mock_judge.return_value = MagicMock(combined_score=0.0)

    pipeline.config.benchmarks.binary_package = None
    pipeline.config.benchmarks.custom_command = None

    result = pipeline.evaluate(str(candidate_file))
    # At baseline: perf_ratio=1.0, norm_perf=max(0, min(1, 1.0/2.0))=0.5
    assert result.perf_score == 0.5
    # NOT 1.0 (which would be the raw perf_ratio)
    assert result.perf_ratio == 1.0


@patch("codeevolve.evaluator.pipeline.judge_code")
@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_all_scores_bounded_zero_one(mock_clean, mock_clippy, mock_test, mock_judge, pipeline, candidate_file):
    """All score fields in EvaluationResult should be in [0, 1]."""
    mock_clippy.return_value = MagicMock(success=True, warnings=[], warning_counts={}, elapsed_seconds=2.0)
    mock_test.return_value = MagicMock(success=True, tests_passed=5, tests_failed=0, elapsed_seconds=1.0)
    mock_judge.return_value = MagicMock(combined_score=0.9)

    pipeline.config.benchmarks.binary_package = None
    pipeline.config.benchmarks.custom_command = None

    result = pipeline.evaluate(str(candidate_file))
    assert 0.0 <= result.static_score <= 1.0
    assert 0.0 <= result.perf_score <= 1.0
    assert 0.0 <= result.llm_score <= 1.0


@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_static_score_perfect_when_no_warnings(mock_clean, mock_clippy, mock_test, pipeline, candidate_file):
    """Zero clippy warnings should yield static_score=1.0 (perfect)."""
    mock_clippy.return_value = MagicMock(success=True, warnings=[], warning_counts={}, elapsed_seconds=1.0)
    mock_test.return_value = MagicMock(success=True, tests_passed=1, tests_failed=0, elapsed_seconds=0.5)

    pipeline.config.benchmarks.binary_package = None
    pipeline.config.benchmarks.custom_command = None
    pipeline.config.llm_judgment.enabled = False  # test doesn't need the judge

    result = pipeline.evaluate(str(candidate_file))
    assert result.static_score == 1.0


@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_static_score_decreases_with_warnings(mock_clean, mock_clippy, mock_test, pipeline, candidate_file):
    """More clippy warnings should lower static_score."""
    mock_clippy.return_value = MagicMock(
        success=True,
        warnings=[{"code": "clippy::style"}] * 5,
        warning_counts={"style": 5},
        elapsed_seconds=1.0,
    )
    mock_test.return_value = MagicMock(success=True, tests_passed=1, tests_failed=0, elapsed_seconds=0.5)

    pipeline.config.benchmarks.binary_package = None
    pipeline.config.benchmarks.custom_command = None
    pipeline.config.llm_judgment.enabled = False  # test doesn't need the judge

    result = pipeline.evaluate(str(candidate_file))
    assert 0.0 < result.static_score < 1.0


@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_result_keeps_raw_counts_for_internal_use(mock_clean, mock_clippy, mock_test, pipeline, candidate_file):
    """EvaluationResult still carries raw counts (tests_passed, clippy_warnings)
    for internal logging, even though these should not go to OpenEvolve."""
    mock_clippy.return_value = MagicMock(
        success=True,
        warnings=[{"code": "clippy::style"}] * 3,
        warning_counts={"style": 3},
        elapsed_seconds=1.5,
    )
    mock_test.return_value = MagicMock(success=True, tests_passed=7, tests_failed=0, elapsed_seconds=0.5)

    pipeline.config.benchmarks.binary_package = None
    pipeline.config.benchmarks.custom_command = None
    pipeline.config.llm_judgment.enabled = False  # test doesn't need the judge

    result = pipeline.evaluate(str(candidate_file))
    # Raw counts preserved in the dataclass
    assert result.tests_passed == 7
    assert result.tests_failed == 0
    assert result.clippy_warnings == 3
    assert result.build_time == 1.5


# ---------------------------------------------------------------------------
# Fixer writeback tests
# ---------------------------------------------------------------------------


@patch("codeevolve.evaluator.pipeline.judge_code")
@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_fixer_writeback_updates_program_path(mock_clean, mock_clippy, mock_test, mock_judge, tmp_path):
    """When the fixer changes code during evaluation, program_path is updated."""
    config = load_config()
    source_file = tmp_path / "lib.rs"
    source_file.write_text("fn main() { original(); }", encoding="utf-8")
    pipeline = EvaluationPipeline(config, tmp_path, source_file)

    pipeline.config.benchmarks.binary_package = None
    pipeline.config.benchmarks.custom_command = None

    # First call to clippy fails (triggers fixer), second succeeds
    clippy_fail = MagicMock(success=False, error_output="compile error", elapsed_seconds=1.0)
    clippy_pass = MagicMock(success=True, warnings=[], warning_counts={}, elapsed_seconds=1.0)
    mock_clippy.side_effect = [clippy_fail, clippy_pass]
    mock_test.return_value = MagicMock(success=True, tests_passed=1, tests_failed=0, elapsed_seconds=0.5)
    mock_judge.return_value = MagicMock(combined_score=0.5)

    # The fixer modifies the focus file on disk
    fixed_code = "fn main() { fixed(); }"
    def fake_attempt_fix(code, err_type, err_output, api_base, model, **kwargs):
        return fixed_code

    candidate = tmp_path / "candidate.rs"
    candidate.write_text("fn main() { broken(); }", encoding="utf-8")
    original_candidate = candidate.read_text(encoding="utf-8")

    with patch("codeevolve.evaluator.pipeline.attempt_fix", side_effect=fake_attempt_fix):
        result = pipeline.evaluate(str(candidate))

    assert result.passed_gates is True
    # program_path should have been updated with the fixed code
    updated = candidate.read_text(encoding="utf-8")
    assert updated != original_candidate
    assert "fixed" in updated
    # Source file should be restored to original
    assert source_file.read_text(encoding="utf-8") == "fn main() { original(); }"


@patch("codeevolve.evaluator.pipeline.judge_code")
@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_no_writeback_when_code_unchanged(mock_clean, mock_clippy, mock_test, mock_judge, tmp_path):
    """When the fixer is not invoked, program_path remains unchanged."""
    config = load_config()
    source_file = tmp_path / "lib.rs"
    source_file.write_text("fn main() { original(); }", encoding="utf-8")
    pipeline = EvaluationPipeline(config, tmp_path, source_file)

    pipeline.config.benchmarks.binary_package = None
    pipeline.config.benchmarks.custom_command = None

    # Clippy passes first try (no fixer needed)
    mock_clippy.return_value = MagicMock(success=True, warnings=[], warning_counts={}, elapsed_seconds=1.0)
    mock_test.return_value = MagicMock(success=True, tests_passed=1, tests_failed=0, elapsed_seconds=0.5)
    mock_judge.return_value = MagicMock(combined_score=0.5)

    candidate = tmp_path / "candidate.rs"
    candidate.write_text("fn main() { improved(); }", encoding="utf-8")
    original_candidate = candidate.read_text(encoding="utf-8")

    result = pipeline.evaluate(str(candidate))

    assert result.passed_gates is True
    # program_path should be unchanged
    assert candidate.read_text(encoding="utf-8") == original_candidate


@patch("codeevolve.evaluator.pipeline.judge_code")
@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_fixer_writeback_no_update_on_gate_failure(mock_clean, mock_clippy, mock_test, mock_judge, tmp_path):
    """When passed_gates is False, no writeback happens even if fixer ran."""
    config = load_config()
    source_file = tmp_path / "lib.rs"
    source_file.write_text("fn main() { original(); }", encoding="utf-8")
    pipeline = EvaluationPipeline(config, tmp_path, source_file)

    pipeline.config.benchmarks.binary_package = None
    pipeline.config.benchmarks.custom_command = None

    # All attempts fail (fixer doesn't help enough)
    mock_clippy.return_value = MagicMock(success=False, error_output="compile error", elapsed_seconds=1.0)

    candidate = tmp_path / "candidate.rs"
    candidate.write_text("fn main() { broken(); }", encoding="utf-8")
    original_candidate = candidate.read_text(encoding="utf-8")

    with patch("codeevolve.evaluator.pipeline.attempt_fix", return_value=None):
        result = pipeline.evaluate(str(candidate))

    assert result.passed_gates is False
    # program_path should NOT be updated
    assert candidate.read_text(encoding="utf-8") == original_candidate


@patch("codeevolve.evaluator.pipeline.judge_code")
@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_fixer_writeback_with_evolve_block(mock_clean, mock_clippy, mock_test, mock_judge, tmp_path):
    """Writeback works correctly when focus file has EVOLVE-BLOCK markers."""
    config = load_config()
    source_file = tmp_path / "lib.rs"
    source_file.write_text(
        "use std::io;\n"
        "// EVOLVE-BLOCK-START\n"
        "fn foo() { 1 }\n"
        "// EVOLVE-BLOCK-END\n"
        "mod tests;\n",
        encoding="utf-8",
    )
    pipeline = EvaluationPipeline(config, tmp_path, source_file)

    pipeline.config.benchmarks.binary_package = None
    pipeline.config.benchmarks.custom_command = None

    # First clippy fails, second passes (fixer succeeds)
    clippy_fail = MagicMock(success=False, error_output="error", elapsed_seconds=1.0)
    clippy_pass = MagicMock(success=True, warnings=[], warning_counts={}, elapsed_seconds=1.0)
    mock_clippy.side_effect = [clippy_fail, clippy_pass]
    mock_test.return_value = MagicMock(success=True, tests_passed=1, tests_failed=0, elapsed_seconds=0.5)
    mock_judge.return_value = MagicMock(combined_score=0.5)

    def fake_attempt_fix(code, err_type, err_output, api_base, model, **kwargs):
        return "fn foo() { 42 }"

    candidate = tmp_path / "candidate.rs"
    candidate.write_text("fn foo() { broken }", encoding="utf-8")

    with patch("codeevolve.evaluator.pipeline.attempt_fix", side_effect=fake_attempt_fix):
        result = pipeline.evaluate(str(candidate))

    assert result.passed_gates is True
    # The candidate file should contain the fixed evolve content
    updated = candidate.read_text(encoding="utf-8")
    assert "fn foo() { 42 }" in updated
    # The source file should be restored
    restored = source_file.read_text(encoding="utf-8")
    assert "fn foo() { 1 }" in restored


@patch("codeevolve.evaluator.pipeline.judge_code")
@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_fixer_writeback_bundle_candidate(mock_clean, mock_clippy, mock_test, mock_judge, tmp_path):
    """Writeback correctly reconstructs the bundle when candidate is a bundle."""
    config = load_config()
    source_file = tmp_path / "lib.rs"
    source_file.write_text("fn main() { original(); }", encoding="utf-8")
    pipeline = EvaluationPipeline(config, tmp_path, source_file)

    pipeline.config.benchmarks.binary_package = None
    pipeline.config.benchmarks.custom_command = None

    # First clippy fails, second passes
    clippy_fail = MagicMock(success=False, error_output="error", elapsed_seconds=1.0)
    clippy_pass = MagicMock(success=True, warnings=[], warning_counts={}, elapsed_seconds=1.0)
    mock_clippy.side_effect = [clippy_fail, clippy_pass]
    mock_test.return_value = MagicMock(success=True, tests_passed=1, tests_failed=0, elapsed_seconds=0.5)
    mock_judge.return_value = MagicMock(combined_score=0.5)

    def fake_attempt_fix(code, err_type, err_output, api_base, model, **kwargs):
        return "fn main() { fixed(); }"

    bundle_content = (
        "// === CONTEXT (read-only \u2014 do NOT modify) ===\n"
        "// file: other.rs\n"
        "pub fn other() {}\n"
        "// === END CONTEXT ===\n\n"
        "// === FOCUS: src/lib.rs ===\n"
        "// (This is the file you should improve. Output your improved version below.)\n"
        "fn main() { broken(); }\n"
        "// === END FOCUS ===\n"
    )
    candidate = tmp_path / "bundle_candidate.rs"
    candidate.write_text(bundle_content, encoding="utf-8")

    with patch("codeevolve.evaluator.pipeline.attempt_fix", side_effect=fake_attempt_fix):
        result = pipeline.evaluate(str(candidate))

    assert result.passed_gates is True
    # The bundle should be updated with fixed focus content
    updated = candidate.read_text(encoding="utf-8")
    assert "// === FOCUS:" in updated
    assert "// === END FOCUS ===" in updated
    assert "fn main() { fixed(); }" in updated
    assert "fn main() { broken(); }" not in updated
    # Context section should be preserved
    assert "pub fn other() {}" in updated


# ---------------------------------------------------------------------------
# Artifact feedback channel tests
# ---------------------------------------------------------------------------


def test_evaluation_result_has_artifacts_field():
    """EvaluationResult dataclass has an artifacts dict, defaulting to empty."""
    r = EvaluationResult(passed_gates=True, combined_score=0.5)
    assert r.artifacts == {}

    r2 = EvaluationResult(
        passed_gates=True, combined_score=0.5,
        artifacts={"clippy_diagnostics": "some warning"},
    )
    assert r2.artifacts["clippy_diagnostics"] == "some warning"


def test_format_clippy_diagnostics_empty():
    assert _format_clippy_diagnostics([]) == ""


def test_format_clippy_diagnostics_formats_warnings():
    warnings = [
        {"code": "clippy::needless_return", "message": "unneeded return", "level": "warning", "file": "src/lib.rs", "line": 42},
        {"code": "clippy::redundant_clone", "message": "redundant clone", "level": "warning", "file": "src/main.rs", "line": 10},
    ]
    result = _format_clippy_diagnostics(warnings)
    assert "clippy::needless_return" in result
    assert "unneeded return" in result
    assert "src/lib.rs:42" in result
    assert "clippy::redundant_clone" in result
    assert "src/main.rs:10" in result


def test_truncate_artifact_short_string():
    short = "hello world"
    assert _truncate_artifact(short) == short


def test_truncate_artifact_long_string():
    long_text = "x" * 10_000
    result = _truncate_artifact(long_text, limit=500)
    assert len(result.encode("utf-8")) <= 500
    assert "truncated" in result


@patch("codeevolve.evaluator.pipeline.attempt_fix", return_value=None)
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_build_failure_includes_build_errors_artifact(mock_clean, mock_clippy, mock_fix, pipeline, candidate_file):
    """Build failure should include build_errors artifact."""
    mock_clippy.return_value = MagicMock(
        success=False, error_output="error[E0308]: mismatched types",
        elapsed_seconds=1.0,
    )
    result = pipeline.evaluate(str(candidate_file))
    assert result.passed_gates is False
    assert "build_errors" in result.artifacts
    assert "mismatched types" in result.artifacts["build_errors"]


@patch("codeevolve.evaluator.pipeline.attempt_fix", return_value=None)
@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_test_failure_includes_artifacts(mock_clean, mock_clippy, mock_test, mock_fix, pipeline, candidate_file):
    """Test failure should include test_failures and clippy_diagnostics artifacts."""
    mock_clippy.return_value = MagicMock(
        success=True,
        warnings=[{"code": "clippy::needless_return", "message": "unneeded return", "level": "warning", "file": "src/lib.rs", "line": 5}],
        warning_counts={"style": 1},
        elapsed_seconds=1.0,
    )
    mock_test.return_value = MagicMock(
        success=False,
        error_output="thread 'tests::it_works' panicked at 'assertion failed'",
        tests_passed=0, tests_failed=1,
        failed_test_names=["tests::it_works"],
        elapsed_seconds=1.0,
    )
    result = pipeline.evaluate(str(candidate_file))
    assert result.passed_gates is False
    assert "test_failures" in result.artifacts
    assert "assertion failed" in result.artifacts["test_failures"]
    assert "clippy_diagnostics" in result.artifacts
    assert "needless_return" in result.artifacts["clippy_diagnostics"]


@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_passing_candidate_includes_clippy_artifacts(mock_clean, mock_clippy, mock_test, pipeline, candidate_file):
    """Passing candidate with clippy warnings should include clippy_diagnostics artifact."""
    mock_clippy.return_value = MagicMock(
        success=True,
        warnings=[
            {"code": "clippy::redundant_clone", "message": "redundant clone", "level": "warning", "file": "src/lib.rs", "line": 10},
        ],
        warning_counts={"perf": 1},
        elapsed_seconds=1.0,
    )
    mock_test.return_value = MagicMock(success=True, tests_passed=3, tests_failed=0, elapsed_seconds=0.5)

    pipeline.config.benchmarks.binary_package = None
    pipeline.config.benchmarks.custom_command = None
    pipeline.config.llm_judgment.enabled = False  # test doesn't need the judge

    result = pipeline.evaluate(str(candidate_file))
    assert result.passed_gates is True
    assert "clippy_diagnostics" in result.artifacts
    assert "redundant_clone" in result.artifacts["clippy_diagnostics"]


@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_passing_candidate_no_warnings_has_empty_artifacts(mock_clean, mock_clippy, mock_test, pipeline, candidate_file):
    """Passing candidate with zero warnings should have empty artifacts dict."""
    mock_clippy.return_value = MagicMock(
        success=True, warnings=[], warning_counts={}, elapsed_seconds=1.0,
    )
    mock_test.return_value = MagicMock(success=True, tests_passed=3, tests_failed=0, elapsed_seconds=0.5)

    pipeline.config.benchmarks.binary_package = None
    pipeline.config.benchmarks.custom_command = None
    pipeline.config.llm_judgment.enabled = False  # test doesn't need the judge

    result = pipeline.evaluate(str(candidate_file))
    assert result.passed_gates is True
    assert result.artifacts == {}


@patch("codeevolve.evaluator.pipeline.attempt_fix", return_value=None)
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_duplicate_candidate_has_empty_artifacts(mock_clean, mock_clippy, mock_fix, pipeline, tmp_path):
    """Duplicate candidate rejection should have empty artifacts."""
    mock_clippy.return_value = MagicMock(success=False, error_output="err", elapsed_seconds=0.5)

    # Initial eval to register the hash
    initial = tmp_path / "initial.rs"
    initial.write_text("fn main() {}")
    pipeline.evaluate(str(initial))

    # Duplicate should be rejected with empty artifacts
    dup = tmp_path / "dup.rs"
    dup.write_text("fn main() {}")  # same as source_file fixture
    result = pipeline.evaluate(str(dup))
    assert result.passed_gates is False
    assert result.artifacts == {}
