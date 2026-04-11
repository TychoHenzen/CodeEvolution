"""Tests for workspace-level evaluation pipeline features (Task 5).

Tests bundle handling, focus_file parameter, crate root discovery,
release binary size measurement, and backward compatibility.
"""
import platform
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeevolve.config import load_config
from codeevolve.evaluator.pipeline import EvaluationPipeline, EvaluationResult


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------

def test_pipeline_accepts_focus_file_positional(tmp_path):
    """Positional focus_file parameter works."""
    config = load_config()
    src = tmp_path / "lib.rs"
    src.write_text("fn main() {}")
    p = EvaluationPipeline(config, tmp_path, src)
    assert p.focus_file == src


def test_pipeline_accepts_focus_file_kwarg(tmp_path):
    """focus_file keyword argument works."""
    config = load_config()
    src = tmp_path / "lib.rs"
    src.write_text("fn main() {}")
    p = EvaluationPipeline(config, tmp_path, focus_file=src)
    assert p.focus_file == src


def test_pipeline_requires_focus_file(tmp_path):
    """Omitting focus_file raises ValueError."""
    config = load_config()
    with pytest.raises(ValueError, match="focus_file must be provided"):
        EvaluationPipeline(config, tmp_path)


# ---------------------------------------------------------------------------
# Bundle detection
# ---------------------------------------------------------------------------

def test_is_bundle_positive():
    """Text with FOCUS markers is detected as a bundle."""
    bundle = (
        "// === CONTEXT (read-only) ===\n"
        "// === END CONTEXT ===\n\n"
        "// === FOCUS: src/lib.rs ===\n"
        "// (This is the file you should improve. Output your improved version below.)\n"
        "fn main() {}\n"
        "// === END FOCUS ===\n"
    )
    assert EvaluationPipeline._is_bundle(bundle) is True


def test_is_bundle_negative():
    """Regular Rust code is not detected as a bundle."""
    assert EvaluationPipeline._is_bundle("fn main() {}") is False


def test_is_bundle_partial_markers():
    """Only FOCUS start without END is not a bundle."""
    text = "// === FOCUS: src/lib.rs ===\nfn main() {}"
    assert EvaluationPipeline._is_bundle(text) is False


# ---------------------------------------------------------------------------
# Bundle evaluation
# ---------------------------------------------------------------------------

@patch("codeevolve.evaluator.pipeline.attempt_fix", return_value=None)
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_pipeline_evaluates_bundle_candidate(mock_clean, mock_clippy, mock_fix, tmp_path):
    """Pipeline extracts focus content from a bundle and evaluates it."""
    config = load_config()
    source_file = tmp_path / "lib.rs"
    source_file.write_text("fn main() { original(); }", encoding="utf-8")
    pipeline = EvaluationPipeline(config, tmp_path, source_file)

    mock_clippy.return_value = MagicMock(
        success=False, error_output="err", elapsed_seconds=0.5,
    )

    # Create a bundle-format candidate
    bundle_content = (
        "// === CONTEXT (read-only \u2014 do NOT modify) ===\n"
        "// file: other.rs\n"
        "pub fn other() {}\n"
        "// === END CONTEXT ===\n\n"
        "// === FOCUS: src/lib.rs ===\n"
        "// (This is the file you should improve. Output your improved version below.)\n"
        "fn main() { improved(); }\n"
        "// === END FOCUS ===\n"
    )
    candidate = tmp_path / "bundle_candidate.rs"
    candidate.write_text(bundle_content, encoding="utf-8")

    result = pipeline.evaluate(str(candidate))
    # Build fails, but that's fine -- we just want to verify the pipeline
    # didn't crash and properly handled the bundle.
    assert result.passed_gates is False

    # Source file should be restored after evaluation
    assert source_file.read_text(encoding="utf-8") == "fn main() { original(); }"


@patch("codeevolve.evaluator.pipeline.attempt_fix", return_value=None)
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_pipeline_bundle_with_evolve_block(mock_clean, mock_clippy, mock_fix, tmp_path):
    """Bundle focus content with EVOLVE-BLOCK markers is handled correctly."""
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

    mock_clippy.return_value = MagicMock(
        success=False, error_output="err", elapsed_seconds=0.5,
    )

    # Bundle where focus content has EVOLVE-BLOCK markers
    bundle_content = (
        "// === CONTEXT (read-only \u2014 do NOT modify) ===\n"
        "// === END CONTEXT ===\n\n"
        "// === FOCUS: src/lib.rs ===\n"
        "// (This is the file you should improve. Output your improved version below.)\n"
        "fn foo() { 2 }\n"
        "// === END FOCUS ===\n"
    )
    candidate = tmp_path / "bundle.rs"
    candidate.write_text(bundle_content, encoding="utf-8")

    pipeline.evaluate(str(candidate))

    # Verify EVOLVE-BLOCK structure was preserved
    assert pipeline._evolve_prefix is not None
    assert "EVOLVE-BLOCK-START" in pipeline._evolve_prefix
    assert "mod tests;" in pipeline._evolve_suffix

    # Source restored
    assert "fn foo() { 1 }" in source_file.read_text(encoding="utf-8")


def test_pipeline_bundle_empty_focus_rejected(tmp_path):
    """Bundle with empty focus content is rejected."""
    config = load_config()
    source_file = tmp_path / "lib.rs"
    source_file.write_text("fn main() {}", encoding="utf-8")
    pipeline = EvaluationPipeline(config, tmp_path, source_file)

    # Bundle with no content between FOCUS markers
    bundle_content = (
        "// === CONTEXT (read-only \u2014 do NOT modify) ===\n"
        "// === END CONTEXT ===\n\n"
        "// === FOCUS: src/lib.rs ===\n"
        "// (This is the file you should improve. Output your improved version below.)\n"
        "// === END FOCUS ===\n"
    )
    candidate = tmp_path / "empty_bundle.rs"
    candidate.write_text(bundle_content, encoding="utf-8")

    result = pipeline.evaluate(str(candidate))
    assert result.passed_gates is False
    assert "empty focus" in result.error


# ---------------------------------------------------------------------------
# Crate root discovery
# ---------------------------------------------------------------------------

def test_find_crate_root_single_crate(tmp_path):
    """For a single crate, crate root is the project path."""
    config = load_config()
    (tmp_path / "Cargo.toml").write_text("[package]\nname = \"test\"")
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    lib_rs = src_dir / "lib.rs"
    lib_rs.write_text("fn main() {}")

    pipeline = EvaluationPipeline(config, tmp_path, lib_rs)
    assert pipeline._find_crate_root() == tmp_path


def test_find_crate_root_workspace_member(tmp_path):
    """For a workspace member, crate root is the member directory."""
    config = load_config()
    # Workspace root
    (tmp_path / "Cargo.toml").write_text(
        "[workspace]\nmembers = [\"crates/engine\"]\n"
    )
    # Member crate
    member = tmp_path / "crates" / "engine"
    member.mkdir(parents=True)
    (member / "Cargo.toml").write_text("[package]\nname = \"engine\"")
    src_dir = member / "src"
    src_dir.mkdir()
    lib_rs = src_dir / "lib.rs"
    lib_rs.write_text("fn main() {}")

    pipeline = EvaluationPipeline(config, tmp_path, lib_rs)
    assert pipeline._find_crate_root() == member


def test_find_crate_root_no_cargo_toml(tmp_path):
    """Falls back to project_path when no Cargo.toml is found."""
    config = load_config()
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    lib_rs = src_dir / "lib.rs"
    lib_rs.write_text("fn main() {}")

    pipeline = EvaluationPipeline(config, tmp_path, lib_rs)
    assert pipeline._find_crate_root() == tmp_path


# ---------------------------------------------------------------------------
# Test source collection for workspace
# ---------------------------------------------------------------------------

def test_collect_test_sources_workspace_member(tmp_path):
    """_collect_test_sources finds tests relative to the member crate, not project root."""
    config = load_config()
    # Workspace root has no tests
    (tmp_path / "Cargo.toml").write_text("[workspace]\nmembers = [\"crates/my_crate\"]")

    # Member crate with tests
    member = tmp_path / "crates" / "my_crate"
    member.mkdir(parents=True)
    (member / "Cargo.toml").write_text("[package]\nname = \"my_crate\"")
    src_dir = member / "src"
    src_dir.mkdir()
    lib_rs = src_dir / "lib.rs"
    lib_rs.write_text("fn lib_fn() {}")

    tests_dir = member / "tests"
    tests_dir.mkdir()
    (tests_dir / "integration.rs").write_text("fn test_integration() {}")

    pipeline = EvaluationPipeline(config, tmp_path, lib_rs)
    sources = pipeline._collect_test_sources()
    assert "tests/integration.rs" in sources
    assert "fn test_integration()" in sources["tests/integration.rs"]


# ---------------------------------------------------------------------------
# Release binary size measurement (find_release_binary_size)
# ---------------------------------------------------------------------------

from codeevolve.evaluator.benchmark import find_release_binary_size


def test_find_release_binary_size_success(tmp_path):
    """Returns the binary file size when the binary exists."""
    release_dir = tmp_path / "target" / "release"
    release_dir.mkdir(parents=True)

    is_windows = platform.system() == "Windows"
    binary_name = "my_app.exe" if is_windows else "my_app"
    binary_path = release_dir / binary_name
    binary_path.write_bytes(b"x" * 5000)

    size = find_release_binary_size(tmp_path, "my_app")
    assert size == 5000


def test_find_release_binary_size_not_found(tmp_path):
    """Returns 0 when the binary doesn't exist."""
    (tmp_path / "target" / "release").mkdir(parents=True)
    size = find_release_binary_size(tmp_path, "my_app")
    assert size == 0


def test_find_release_binary_size_with_upx(tmp_path):
    """UPX compression is applied when upx_path is provided."""
    release_dir = tmp_path / "target" / "release"
    release_dir.mkdir(parents=True)

    is_windows = platform.system() == "Windows"
    binary_name = "my_app.exe" if is_windows else "my_app"
    binary_path = release_dir / binary_name
    binary_path.write_bytes(b"x" * 5000)

    calls = []

    def mock_run_side_effect(cmd, **kwargs):
        calls.append(cmd)
        return MagicMock(returncode=0, stderr="")

    with patch("subprocess.run", side_effect=mock_run_side_effect):
        size = find_release_binary_size(
            tmp_path, "my_app",
            upx_path="/usr/bin/upx",
            upx_args=["--best"],
        )

    assert size == 5000
    # Verify UPX was called
    assert len(calls) == 1
    upx_call = calls[0]
    assert upx_call[0] == "/usr/bin/upx"
    assert "--best" in upx_call


def test_find_release_binary_size_custom_target_dir(tmp_path):
    """Custom target_dir is used for finding binaries."""
    custom_target = tmp_path / "custom_target"
    release_dir = custom_target / "release"
    release_dir.mkdir(parents=True)

    is_windows = platform.system() == "Windows"
    binary_name = "my_app.exe" if is_windows else "my_app"
    (release_dir / binary_name).write_bytes(b"x" * 3000)

    size = find_release_binary_size(
        tmp_path, "my_app", target_dir=str(custom_target),
    )
    assert size == 3000


# ---------------------------------------------------------------------------
# Pipeline wiring: binary_package triggers release binary measurement
# ---------------------------------------------------------------------------

@patch("codeevolve.evaluator.pipeline.judge_code")
@patch("codeevolve.evaluator.pipeline.find_release_binary_size")
@patch("codeevolve.evaluator.pipeline.run_cargo_build")
@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_pipeline_uses_release_binary_when_package_set(
    mock_clean, mock_clippy, mock_test, mock_build,
    mock_release_size, mock_judge, tmp_path,
):
    """When binary_package is set in config, a release link + find_release_binary_size is used."""
    config = load_config()
    config.benchmarks.binary_package = "my_app"
    config.benchmarks.custom_command = None

    source_file = tmp_path / "lib.rs"
    source_file.write_text("fn main() {}")
    pipeline = EvaluationPipeline(config, tmp_path, source_file)

    mock_clippy.return_value = MagicMock(
        success=True, warnings=[], warning_counts={}, elapsed_seconds=2.5,
    )
    mock_test.return_value = MagicMock(
        success=True, tests_passed=5, tests_failed=0, elapsed_seconds=1.0,
    )
    mock_build.return_value = MagicMock(success=True, elapsed_seconds=0.5)
    mock_release_size.return_value = 500_000
    mock_judge.return_value = MagicMock(combined_score=0.7)

    candidate = tmp_path / "candidate.rs"
    candidate.write_text("fn main() { println!(\"hi\"); }")

    result = pipeline.evaluate(str(candidate))
    assert result.passed_gates is True

    # Verify release build (link step) and binary size measurement
    mock_build.assert_called_once()
    assert mock_build.call_args[1].get("release") or mock_build.call_args[0][3] is True
    mock_release_size.assert_called_once()
    assert mock_release_size.call_args[0][1] == "my_app"


@patch("codeevolve.evaluator.pipeline.judge_code")
@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_pipeline_skips_binary_when_no_package(
    mock_clean, mock_clippy, mock_test, mock_judge, tmp_path,
):
    """When binary_package is not set, binary size measurement is skipped."""
    config = load_config()
    config.benchmarks.binary_package = None
    config.benchmarks.custom_command = None

    source_file = tmp_path / "lib.rs"
    source_file.write_text("fn main() {}")
    pipeline = EvaluationPipeline(config, tmp_path, source_file)

    mock_clippy.return_value = MagicMock(
        success=True, warnings=[], warning_counts={}, elapsed_seconds=2.5,
    )
    mock_test.return_value = MagicMock(
        success=True, tests_passed=5, tests_failed=0, elapsed_seconds=1.0,
    )
    mock_judge.return_value = MagicMock(combined_score=0.7)

    candidate = tmp_path / "candidate.rs"
    candidate.write_text("fn main() { println!(\"hi\"); }")

    result = pipeline.evaluate(str(candidate))
    assert result.passed_gates is True


# ---------------------------------------------------------------------------
# Verify custom_command flows through (Task 5.4)
# ---------------------------------------------------------------------------

@patch("codeevolve.evaluator.pipeline.judge_code")
@patch("codeevolve.evaluator.pipeline.run_user_benchmark")
@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_clean")
def test_pipeline_custom_command_flows_through(
    mock_clean, mock_clippy, mock_test, mock_bench, mock_judge, tmp_path,
):
    """cfg.benchmarks.custom_command is passed to run_user_benchmark."""
    config = load_config()
    config.benchmarks.custom_command = "cargo bench"
    config.benchmarks.custom_command_score_regex = r"time:\s+([\d.]+) (ms|us|ns)"
    config.benchmarks.binary_package = None

    source_file = tmp_path / "lib.rs"
    source_file.write_text("fn main() {}")
    pipeline = EvaluationPipeline(config, tmp_path, source_file)

    mock_clippy.return_value = MagicMock(
        success=True, warnings=[], warning_counts={}, elapsed_seconds=2.5,
    )
    mock_test.return_value = MagicMock(
        success=True, tests_passed=5, tests_failed=0, elapsed_seconds=1.0,
    )
    mock_bench.return_value = MagicMock(success=True, score=42.0, output="time: 42 ms")
    mock_judge.return_value = MagicMock(combined_score=0.7)

    candidate = tmp_path / "candidate.rs"
    candidate.write_text("fn main() { println!(\"hi\"); }")

    result = pipeline.evaluate(str(candidate))
    assert result.passed_gates is True

    # Verify run_user_benchmark was called with the custom command
    mock_bench.assert_called_once_with(
        "cargo bench",
        tmp_path,
        r"time:\s+([\d.]+) (ms|us|ns)",
    )
