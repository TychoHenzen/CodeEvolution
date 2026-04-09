import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codeevolve.evaluator.cargo import (
    CargoResult,
    parse_clippy_json,
    run_cargo_build,
    run_cargo_clippy,
    run_cargo_test,
    categorize_lint,
    compute_clippy_score,
)
from codeevolve.config import ClippyWeights


# --- Clippy JSON parsing ---

def test_parse_clippy_json(clippy_output_json: str):
    warnings = parse_clippy_json(clippy_output_json)
    assert len(warnings) == 3
    assert warnings[0]["code"] == "clippy::needless_return"
    assert warnings[1]["code"] == "clippy::cast_possible_truncation"
    assert warnings[2]["code"] == "clippy::unwrap_used"


def test_parse_clippy_json_empty():
    warnings = parse_clippy_json("[]")
    assert warnings == []


# --- Lint categorization ---

def test_categorize_lint_style():
    assert categorize_lint("clippy::needless_return") == "style"


def test_categorize_lint_correctness():
    assert categorize_lint("clippy::wrong_self_convention") == "correctness"


def test_categorize_lint_suspicious():
    assert categorize_lint("clippy::cast_possible_truncation") == "suspicious"


def test_categorize_lint_unknown_defaults_to_style():
    assert categorize_lint("clippy::some_unknown_lint") == "style"


# --- Clippy score computation ---

def test_compute_clippy_score_no_warnings():
    score = compute_clippy_score({}, ClippyWeights())
    assert score == 0


def test_compute_clippy_score_weighted():
    counts = {"correctness": 1, "suspicious": 2, "style": 3}
    weights = ClippyWeights()
    # -(5*1 + 3*2 + 1*3) = -(5 + 6 + 3) = -14
    assert compute_clippy_score(counts, weights) == -14


# --- Cargo subprocess calls (using real cargo if available) ---

def test_run_cargo_build_success(sample_crate: Path):
    result = run_cargo_build(sample_crate)
    assert result.success is True
    assert result.elapsed_seconds > 0


def test_run_cargo_build_failure(tmp_path: Path):
    bad_crate = tmp_path / "bad"
    bad_crate.mkdir()
    (bad_crate / "Cargo.toml").write_text('[package]\nname = "bad"\nversion = "0.1.0"\nedition = "2021"')
    (bad_crate / "src").mkdir()
    (bad_crate / "src" / "lib.rs").write_text("fn broken( {}")  # syntax error
    result = run_cargo_build(bad_crate)
    assert result.success is False
    assert result.error_output != ""


def test_run_cargo_test_success(sample_crate: Path):
    run_cargo_build(sample_crate)  # must build first
    result = run_cargo_test(sample_crate)
    assert result.success is True
    assert result.tests_passed >= 1
    assert result.tests_failed == 0


def test_run_cargo_clippy_returns_warnings(sample_crate: Path):
    result = run_cargo_clippy(sample_crate)
    assert result.success is True
    assert isinstance(result.warnings, list)
    # sample_crate is clean, so few/no warnings expected
