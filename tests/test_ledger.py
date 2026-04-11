"""Tests for codeevolve.ledger — parse_ledger() and LedgerEntry."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeevolve.ledger import LedgerEntry, parse_ledger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_LEDGER = """\
## Tech Debt Summary

| File Path | Type | Structural | Semantic | Combined | Top Issue | Last Reviewed | Trend |
|-----------|------|-----------|----------|----------|-----------|---------------|-------|
| tools/img-to-shape/tests/suite/core_lib.rs | test | 116.00 | 0 | 116.00 | duplicate_blocks (173) | 2026-04-02 | — |
| crates/axiom2d/src/splash/render.rs | prod | 57.70 | 0 | 57.70 | magic_literals (320) | 2026-04-02 | — |
| crates/axiom2d/src/audio/mixer.rs | prod | 32.50 | 12.0 | 44.50 | long_function (5) | 2026-04-02 | — |
| crates/axiom2d/src/utils/helpers.rs | test | 20.00 | 0 | 20.00 | dead_code (3) | 2026-04-02 | — |
"""


def _write(tmp_path: Path, content: str, filename: str = "TECH_DEBT_LEDGER.md") -> Path:
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests: valid ledger, prod_only=True (default)
# ---------------------------------------------------------------------------

def test_parse_ledger_returns_prod_entries_by_default(tmp_path: Path):
    path = _write(tmp_path, _VALID_LEDGER)
    entries = parse_ledger(path)
    types = {e.file_type for e in entries}
    assert types == {"prod"}, "Default prod_only=True should exclude test entries"


def test_parse_ledger_prod_only_correct_count(tmp_path: Path):
    path = _write(tmp_path, _VALID_LEDGER)
    entries = parse_ledger(path)
    assert len(entries) == 2  # render.rs and mixer.rs


def test_parse_ledger_prod_only_correct_paths(tmp_path: Path):
    path = _write(tmp_path, _VALID_LEDGER)
    entries = parse_ledger(path)
    paths = {e.file_path for e in entries}
    assert "crates/axiom2d/src/splash/render.rs" in paths
    assert "crates/axiom2d/src/audio/mixer.rs" in paths


def test_parse_ledger_prod_only_correct_scores(tmp_path: Path):
    path = _write(tmp_path, _VALID_LEDGER)
    entries = parse_ledger(path)
    score_map = {e.file_path: e.combined_score for e in entries}
    assert score_map["crates/axiom2d/src/splash/render.rs"] == pytest.approx(57.70)
    assert score_map["crates/axiom2d/src/audio/mixer.rs"] == pytest.approx(44.50)


# ---------------------------------------------------------------------------
# Tests: prod_only=False returns all entries
# ---------------------------------------------------------------------------

def test_parse_ledger_prod_only_false_returns_all(tmp_path: Path):
    path = _write(tmp_path, _VALID_LEDGER)
    entries = parse_ledger(path, prod_only=False)
    assert len(entries) == 4


def test_parse_ledger_prod_only_false_includes_test_type(tmp_path: Path):
    path = _write(tmp_path, _VALID_LEDGER)
    entries = parse_ledger(path, prod_only=False)
    types = {e.file_type for e in entries}
    assert "test" in types
    assert "prod" in types


# ---------------------------------------------------------------------------
# Tests: sorting
# ---------------------------------------------------------------------------

def test_parse_ledger_sorted_descending_by_default(tmp_path: Path):
    path = _write(tmp_path, _VALID_LEDGER)
    entries = parse_ledger(path)
    scores = [e.combined_score for e in entries]
    assert scores == sorted(scores, reverse=True)


def test_parse_ledger_all_sorted_descending(tmp_path: Path):
    path = _write(tmp_path, _VALID_LEDGER)
    entries = parse_ledger(path, prod_only=False)
    scores = [e.combined_score for e in entries]
    assert scores == sorted(scores, reverse=True)
    # Highest should be the test entry with score 116.00
    assert entries[0].combined_score == pytest.approx(116.00)
    assert entries[0].file_type == "test"


# ---------------------------------------------------------------------------
# Tests: missing file
# ---------------------------------------------------------------------------

def test_parse_ledger_missing_file_returns_empty(tmp_path: Path):
    path = tmp_path / "TECH_DEBT_LEDGER.md"
    assert not path.exists()
    result = parse_ledger(path)
    assert result == []


def test_parse_ledger_missing_file_prod_only_false_returns_empty(tmp_path: Path):
    path = tmp_path / "TECH_DEBT_LEDGER.md"
    result = parse_ledger(path, prod_only=False)
    assert result == []


# ---------------------------------------------------------------------------
# Tests: malformed / empty file
# ---------------------------------------------------------------------------

def test_parse_ledger_empty_file_returns_empty(tmp_path: Path):
    path = _write(tmp_path, "")
    result = parse_ledger(path)
    assert result == []


def test_parse_ledger_no_table_returns_empty(tmp_path: Path):
    content = "# Tech Debt Ledger\n\nNo table here, just prose.\n"
    path = _write(tmp_path, content)
    result = parse_ledger(path)
    assert result == []


def test_parse_ledger_only_header_row_returns_empty(tmp_path: Path):
    content = (
        "| File Path | Type | Structural | Semantic | Combined | Top Issue | Last Reviewed | Trend |\n"
        "|-----------|------|-----------|----------|----------|-----------|---------------|-------|\n"
    )
    path = _write(tmp_path, content)
    result = parse_ledger(path)
    assert result == []


def test_parse_ledger_row_with_non_numeric_combined_skipped(tmp_path: Path):
    content = (
        "| File Path | Type | Structural | Semantic | Combined | Top Issue | Last Reviewed | Trend |\n"
        "|-----------|------|-----------|----------|----------|-----------|---------------|-------|\n"
        "| crates/foo/src/lib.rs | prod | 10.0 | 0 | N/A | foo | 2026-04-02 | — |\n"
        "| crates/bar/src/lib.rs | prod | 20.0 | 0 | 20.0 | bar | 2026-04-02 | — |\n"
    )
    path = _write(tmp_path, content)
    result = parse_ledger(path)
    assert len(result) == 1
    assert result[0].file_path == "crates/bar/src/lib.rs"


def test_parse_ledger_row_with_too_few_columns_skipped(tmp_path: Path):
    content = (
        "| File Path | Type | Combined |\n"
        "|-----------|------|----------|\n"
        "| crates/foo/src/lib.rs | prod | 10.0 |\n"
    )
    path = _write(tmp_path, content)
    # Row has only 3 cells; _MIN_COLS=5 so it should be skipped
    result = parse_ledger(path)
    assert result == []


# ---------------------------------------------------------------------------
# Tests: LedgerEntry dataclass
# ---------------------------------------------------------------------------

def test_ledger_entry_fields():
    entry = LedgerEntry(
        file_path="crates/foo/src/lib.rs",
        file_type="prod",
        combined_score=42.5,
    )
    assert entry.file_path == "crates/foo/src/lib.rs"
    assert entry.file_type == "prod"
    assert entry.combined_score == pytest.approx(42.5)


def test_ledger_entry_is_dataclass():
    """LedgerEntry should support equality comparison via dataclass."""
    a = LedgerEntry("a.rs", "prod", 10.0)
    b = LedgerEntry("a.rs", "prod", 10.0)
    assert a == b
