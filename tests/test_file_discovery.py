from __future__ import annotations

from pathlib import Path

import pytest

from codeevolve.file_discovery import discover_rs_files


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_files(base: Path, paths: list[str], content: str = "") -> list[Path]:
    """Create files under base, returning their absolute Paths."""
    created = []
    for rel in paths:
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        created.append(p.resolve())
    return created


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_include_returns_rs_files(tmp_path: Path):
    """A simple include glob matches .rs files under src/."""
    _make_files(tmp_path, ["src/main.rs", "src/lib.rs"])
    result = discover_rs_files(tmp_path, ["src/**/*.rs"], [])
    names = {p.name for p in result}
    assert names == {"main.rs", "lib.rs"}


def test_exclude_pattern_filters_files(tmp_path: Path):
    """Files matching an exclude glob are removed from results."""
    _make_files(tmp_path, ["src/main.rs", "src/generated/auto.rs"])
    result = discover_rs_files(tmp_path, ["src/**/*.rs"], ["src/generated/**"])
    names = {p.name for p in result}
    assert names == {"main.rs"}
    assert "auto.rs" not in names


def test_multiple_include_patterns_combine(tmp_path: Path):
    """Multiple include globs union their results without duplicates."""
    _make_files(tmp_path, ["crates/alpha/src/lib.rs", "crates/beta/src/lib.rs"])
    result = discover_rs_files(
        tmp_path,
        ["crates/alpha/src/**/*.rs", "crates/beta/src/**/*.rs"],
        [],
    )
    crate_names = {p.parent.parent.name for p in result}
    assert crate_names == {"alpha", "beta"}
    # Both lib.rs files found
    assert len(result) == 2


def test_multiple_exclude_patterns_all_apply(tmp_path: Path):
    """Each exclude glob is applied; all matching files are removed."""
    _make_files(
        tmp_path,
        [
            "src/main.rs",
            "src/generated/auto.rs",
            "src/vendor/third_party.rs",
            "src/utils.rs",
        ],
    )
    result = discover_rs_files(
        tmp_path,
        ["src/**/*.rs"],
        ["src/generated/**", "src/vendor/**"],
    )
    names = {p.name for p in result}
    assert names == {"main.rs", "utils.rs"}


def test_non_rs_files_excluded_even_if_glob_matches(tmp_path: Path):
    """A broad glob that also matches non-.rs files should return only .rs files."""
    _make_files(tmp_path, ["src/main.rs", "src/readme.md", "src/data.txt"])
    result = discover_rs_files(tmp_path, ["src/**/*"], [])
    assert all(p.suffix == ".rs" for p in result)
    assert len(result) == 1


def test_empty_directory_returns_empty(tmp_path: Path):
    """Globbing an empty tree returns an empty list."""
    result = discover_rs_files(tmp_path, ["**/*.rs"], [])
    assert result == []


def test_workspace_style_directory_structure(tmp_path: Path):
    """Workspace-style crates/*/src/**/*.rs pattern finds files across crates."""
    _make_files(
        tmp_path,
        [
            "crates/physics/src/lib.rs",
            "crates/physics/src/collision.rs",
            "crates/render/src/lib.rs",
            "crates/render/src/generated/shaders.rs",  # should be excluded
        ],
    )
    result = discover_rs_files(
        tmp_path,
        ["crates/*/src/**/*.rs"],
        ["**/generated/**"],
    )
    names = {p.name for p in result}
    assert "lib.rs" in names
    assert "collision.rs" in names
    assert "shaders.rs" not in names
    assert len(result) == 3


def test_results_are_sorted(tmp_path: Path):
    """Returned list is in sorted (lexicographic) order."""
    _make_files(tmp_path, ["src/z.rs", "src/a.rs", "src/m.rs"])
    result = discover_rs_files(tmp_path, ["src/**/*.rs"], [])
    assert result == sorted(result)


def test_results_are_absolute_paths(tmp_path: Path):
    """Every returned path is an absolute path."""
    _make_files(tmp_path, ["src/main.rs"])
    result = discover_rs_files(tmp_path, ["src/**/*.rs"], [])
    assert all(p.is_absolute() for p in result)


def test_no_duplicates_when_globs_overlap(tmp_path: Path):
    """Overlapping include patterns do not produce duplicate entries."""
    _make_files(tmp_path, ["src/lib.rs"])
    result = discover_rs_files(
        tmp_path,
        ["src/**/*.rs", "src/*.rs"],  # both match src/lib.rs
        [],
    )
    assert len(result) == 1


def test_include_glob_with_no_matches_returns_empty(tmp_path: Path):
    """A non-matching include glob returns an empty list."""
    _make_files(tmp_path, ["src/main.rs"])
    result = discover_rs_files(tmp_path, ["nonexistent/**/*.rs"], [])
    assert result == []
