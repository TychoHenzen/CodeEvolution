"""Tests for codeevolve.bundler — multi-file bundle format and focus rotation."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeevolve.bundler import (
    create_bundle,
    create_focus_rotation,
    extract_focus,
    extract_focus_path,
    FocusRotation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, name: str, content: str) -> Path:
    """Write a file under tmp_path and return its absolute Path."""
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p.resolve()


_EVOLVE_BLOCK_TEMPLATE = """\
use std::io;

// EVOLVE-BLOCK-START
{content}
// EVOLVE-BLOCK-END

#[cfg(test)]
mod tests {{
    #[test]
    fn it_works() {{ assert!(true); }}
}}
"""


def _write_evolve(tmp_path: Path, name: str, content: str) -> Path:
    """Write a .rs file with EVOLVE-BLOCK markers around *content*."""
    return _write(tmp_path, name, _EVOLVE_BLOCK_TEMPLATE.format(content=content))


# ---------------------------------------------------------------------------
# create_bundle — basic format
# ---------------------------------------------------------------------------

class TestCreateBundle:
    def test_bundle_has_context_and_focus_sections(self, tmp_path: Path):
        focus = _write_evolve(tmp_path, "src/a.rs", "pub fn a() -> i32 { 1 }")
        other = _write(tmp_path, "src/b.rs", "pub fn b() -> i32 { 2 }\n")
        summaries = {
            other: "// file: src/b.rs\npub fn b() -> i32",
        }
        bundle = create_bundle(focus, [focus, other], summaries, tmp_path)

        assert "// === CONTEXT (read-only" in bundle
        assert "// === END CONTEXT ===" in bundle
        assert "// === FOCUS:" in bundle
        assert "// === END FOCUS ===" in bundle

    def test_context_excludes_focus_file_summary(self, tmp_path: Path):
        focus = _write_evolve(tmp_path, "src/a.rs", "pub fn a() {}")
        other = _write(tmp_path, "src/b.rs", "pub fn b() {}\n")
        summaries = {
            focus: "// file: src/a.rs\npub fn a()",
            other: "// file: src/b.rs\npub fn b()",
        }
        bundle = create_bundle(focus, [focus, other], summaries, tmp_path)

        # The context section should include b but NOT a
        context_section = bundle.split("// === END CONTEXT ===")[0]
        assert "src/b.rs" in context_section
        assert "pub fn a()" not in context_section

    def test_focus_contains_evolve_block_content_only(self, tmp_path: Path):
        inner = "pub fn a() -> i32 { 42 }"
        focus = _write_evolve(tmp_path, "src/a.rs", inner)
        bundle = create_bundle(focus, [focus], {}, tmp_path)

        # Focus section should contain the evolve-block content
        extracted = extract_focus(bundle)
        assert "pub fn a() -> i32 { 42 }" in extracted
        # But NOT the prefix/suffix outside the markers
        assert "use std::io" not in extracted
        assert "mod tests" not in extracted
        assert "EVOLVE-BLOCK-START" not in extracted

    def test_focus_header_has_relative_path(self, tmp_path: Path):
        focus = _write_evolve(tmp_path, "src/keyboard.rs", "pub fn handle() {}")
        bundle = create_bundle(focus, [focus], {}, tmp_path)

        assert "// === FOCUS: src/keyboard.rs ===" in bundle

    def test_bundle_includes_correct_relative_paths(self, tmp_path: Path):
        focus = _write_evolve(
            tmp_path, "crates/input/src/keyboard.rs", "pub fn handle() {}"
        )
        other = _write(tmp_path, "crates/core/src/lib.rs", "pub struct Engine {}\n")
        summaries = {
            other: "// file: crates/core/src/lib.rs\npub struct Engine",
        }
        bundle = create_bundle(focus, [focus, other], summaries, tmp_path)

        assert "// === FOCUS: crates/input/src/keyboard.rs ===" in bundle
        assert "crates/core/src/lib.rs" in bundle

    def test_bundle_with_no_other_files(self, tmp_path: Path):
        """A single-file workspace produces an empty context section."""
        focus = _write_evolve(tmp_path, "src/main.rs", "fn main() {}")
        bundle = create_bundle(focus, [focus], {}, tmp_path)

        # Context section should exist but have no file summaries
        context_start = bundle.index("// === CONTEXT")
        context_end = bundle.index("// === END CONTEXT ===")
        between = bundle[context_start:context_end]
        assert "// file:" not in between

    def test_bundle_with_multiple_context_files(self, tmp_path: Path):
        focus = _write_evolve(tmp_path, "src/a.rs", "pub fn a() {}")
        b = _write(tmp_path, "src/b.rs", "pub fn b() {}\n")
        c = _write(tmp_path, "src/c.rs", "pub fn c() {}\n")
        summaries = {
            b: "// file: src/b.rs\npub fn b()",
            c: "// file: src/c.rs\npub fn c()",
        }
        bundle = create_bundle(focus, [focus, b, c], summaries, tmp_path)

        context_section = bundle.split("// === END CONTEXT ===")[0]
        assert "src/b.rs" in context_section
        assert "src/c.rs" in context_section

    def test_file_without_evolve_markers_uses_full_content(self, tmp_path: Path):
        """If the focus file has no EVOLVE-BLOCK markers, use entire content."""
        content = "pub fn no_markers() -> bool { true }\n"
        focus = _write(tmp_path, "src/plain.rs", content)
        bundle = create_bundle(focus, [focus], {}, tmp_path)

        extracted = extract_focus(bundle)
        assert "pub fn no_markers() -> bool { true }" in extracted


# ---------------------------------------------------------------------------
# extract_focus
# ---------------------------------------------------------------------------

class TestExtractFocus:
    def test_roundtrip(self, tmp_path: Path):
        """Content survives create_bundle -> extract_focus roundtrip."""
        inner = "pub fn roundtrip() -> u32 { 99 }"
        focus = _write_evolve(tmp_path, "src/rt.rs", inner)
        bundle = create_bundle(focus, [focus], {}, tmp_path)
        extracted = extract_focus(bundle)
        assert inner in extracted

    def test_no_markers_returns_empty(self):
        assert extract_focus("some random text\nwith no markers") == ""

    def test_empty_focus_section(self):
        bundle = (
            "// === CONTEXT (read-only \u2014 do NOT modify) ===\n"
            "// === END CONTEXT ===\n"
            "\n"
            "// === FOCUS: src/empty.rs ===\n"
            "// (This is the file you should improve. Output your improved version below.)\n"
            "\n"
            "// === END FOCUS ===\n"
        )
        # The focus section contains only a blank line
        result = extract_focus(bundle)
        assert result == ""

    def test_multiline_focus_content(self):
        bundle = (
            "// === CONTEXT (read-only \u2014 do NOT modify) ===\n"
            "// === END CONTEXT ===\n"
            "\n"
            "// === FOCUS: src/multi.rs ===\n"
            "// (This is the file you should improve. Output your improved version below.)\n"
            "pub fn line_one() {}\n"
            "pub fn line_two() {}\n"
            "pub fn line_three() {}\n"
            "// === END FOCUS ===\n"
        )
        result = extract_focus(bundle)
        assert "pub fn line_one() {}" in result
        assert "pub fn line_two() {}" in result
        assert "pub fn line_three() {}" in result

    def test_focus_excludes_markers(self):
        bundle = (
            "// === FOCUS: src/x.rs ===\n"
            "// (This is the file you should improve. Output your improved version below.)\n"
            "content here\n"
            "// === END FOCUS ===\n"
        )
        result = extract_focus(bundle)
        assert "=== FOCUS" not in result
        assert "=== END FOCUS" not in result
        assert "content here" in result


# ---------------------------------------------------------------------------
# extract_focus_path
# ---------------------------------------------------------------------------

class TestExtractFocusPath:
    def test_extracts_path(self, tmp_path: Path):
        inner = "pub fn foo() {}"
        focus = _write_evolve(tmp_path, "src/foo.rs", inner)
        bundle = create_bundle(focus, [focus], {}, tmp_path)
        assert extract_focus_path(bundle) == "src/foo.rs"

    def test_no_focus_returns_none(self):
        assert extract_focus_path("no markers here") is None

    def test_nested_path(self, tmp_path: Path):
        inner = "pub fn bar() {}"
        focus = _write_evolve(tmp_path, "crates/core/src/bar.rs", inner)
        bundle = create_bundle(focus, [focus], {}, tmp_path)
        assert extract_focus_path(bundle) == "crates/core/src/bar.rs"


# ---------------------------------------------------------------------------
# FocusRotation
# ---------------------------------------------------------------------------

class TestFocusRotation:
    def test_single_file_rotation(self, tmp_path: Path):
        f = tmp_path / "src/a.rs"
        rot = FocusRotation([f])
        assert rot.current() == f
        assert rot.next() == f
        assert rot.next() == f  # wraps around

    def test_cycles_through_files(self, tmp_path: Path):
        files = [tmp_path / f"src/{c}.rs" for c in "abc"]
        rot = FocusRotation(files)

        # Should cycle through all files in sorted order
        seen = [rot.next() for _ in range(3)]
        assert len(set(seen)) == 3

    def test_wraps_around(self, tmp_path: Path):
        files = [tmp_path / f"src/{c}.rs" for c in "ab"]
        rot = FocusRotation(files)

        first_cycle = [rot.next() for _ in range(2)]
        second_cycle = [rot.next() for _ in range(2)]
        assert first_cycle == second_cycle

    def test_current_does_not_advance(self, tmp_path: Path):
        files = [tmp_path / f"src/{c}.rs" for c in "abc"]
        rot = FocusRotation(files)

        c1 = rot.current()
        c2 = rot.current()
        assert c1 == c2

    def test_next_advances_then_current_matches(self, tmp_path: Path):
        files = [tmp_path / f"src/{c}.rs" for c in "abc"]
        rot = FocusRotation(files)

        rot.next()  # advance past first
        current = rot.current()
        next_val = rot.next()
        assert current == next_val

    def test_deterministic_order(self, tmp_path: Path):
        """Same files in different input order produce same rotation."""
        a = tmp_path / "src/a.rs"
        b = tmp_path / "src/b.rs"
        c = tmp_path / "src/c.rs"

        rot1 = FocusRotation([c, a, b])
        rot2 = FocusRotation([b, c, a])

        seq1 = [rot1.next() for _ in range(6)]
        seq2 = [rot2.next() for _ in range(6)]
        assert seq1 == seq2

    def test_empty_files_raises(self):
        with pytest.raises(ValueError, match="at least one file"):
            FocusRotation([])

    def test_len(self, tmp_path: Path):
        files = [tmp_path / f"src/{c}.rs" for c in "abcd"]
        rot = FocusRotation(files)
        assert len(rot) == 4

    def test_repr(self, tmp_path: Path):
        files = [tmp_path / "src/main.rs"]
        rot = FocusRotation(files)
        r = repr(rot)
        assert "FocusRotation" in r
        assert "main.rs" in r


# ---------------------------------------------------------------------------
# create_focus_rotation convenience function
# ---------------------------------------------------------------------------

class TestCreateFocusRotation:
    def test_returns_focus_rotation(self, tmp_path: Path):
        files = [tmp_path / "src/a.rs", tmp_path / "src/b.rs"]
        rot = create_focus_rotation(files)
        assert isinstance(rot, FocusRotation)
        assert len(rot) == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            create_focus_rotation([])
