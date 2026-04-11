"""Tests for codeevolve.bundler — multi-file bundle format."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeevolve.bundler import (
    create_bundle,
    create_workspace_bundle,
    extract_focus,
    extract_focus_path,
    replace_focus,
)
from codeevolve.crate_graph import CrateGraph


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
# Helpers for workspace bundle tests
# ---------------------------------------------------------------------------

def _setup_workspace(tmp_path: Path) -> tuple[CrateGraph, dict[str, Path]]:
    """Create a 3-crate workspace layout and return (graph, file_map).

    Crates: engine_core (no deps), engine_render (deps: engine_core),
    game (deps: engine_core, engine_render).
    """
    graph = CrateGraph(
        deps={
            "engine_core": [],
            "engine_render": ["engine_core"],
            "game": ["engine_core", "engine_render"],
        },
        crate_roots={
            "engine_core": tmp_path / "crates" / "engine_core",
            "engine_render": tmp_path / "crates" / "engine_render",
            "game": tmp_path / "crates" / "game",
        },
    )
    files = {
        "core_lib": _write_evolve(tmp_path, "crates/engine_core/src/lib.rs", "pub fn core() {}"),
        "render_lib": _write_evolve(tmp_path, "crates/engine_render/src/lib.rs", "pub fn render() {}"),
        "render_pipe": _write_evolve(tmp_path, "crates/engine_render/src/pipeline.rs", "pub fn pipe() {}"),
        "game_lib": _write_evolve(tmp_path, "crates/game/src/lib.rs", "pub fn game() {}"),
    }
    return graph, files


# ---------------------------------------------------------------------------
# create_workspace_bundle
# ---------------------------------------------------------------------------

class TestCreateWorkspaceBundle:
    def test_filters_context_to_relevant_crates(self, tmp_path: Path):
        """Focus in engine_render: context includes engine_core, excludes game."""
        graph, files = _setup_workspace(tmp_path)
        all_files = list(files.values())
        summaries = {
            files["core_lib"]: "// file: crates/engine_core/src/lib.rs\npub fn core()",
            files["render_pipe"]: "// file: crates/engine_render/src/pipeline.rs\npub fn pipe()",
            files["game_lib"]: "// file: crates/game/src/lib.rs\npub fn game()",
        }
        bundle = create_workspace_bundle(
            files["render_lib"], all_files, summaries, tmp_path, graph,
        )
        context = bundle.split("// === END CONTEXT ===")[0]
        # engine_core is a dep of engine_render -> included
        assert "engine_core/src/lib.rs" in context
        # engine_render/pipeline.rs is a sibling -> included
        assert "engine_render/src/pipeline.rs" in context
        # game is NOT a dep of engine_render -> excluded
        assert "game/src/lib.rs" not in context

    def test_leaf_crate_has_minimal_context(self, tmp_path: Path):
        """Focus in engine_core (no deps): only sibling files in context."""
        graph, files = _setup_workspace(tmp_path)
        all_files = list(files.values())
        summaries = {
            files["render_lib"]: "// file: crates/engine_render/src/lib.rs\npub fn render()",
            files["game_lib"]: "// file: crates/game/src/lib.rs\npub fn game()",
        }
        bundle = create_workspace_bundle(
            files["core_lib"], all_files, summaries, tmp_path, graph,
        )
        context = bundle.split("// === END CONTEXT ===")[0]
        assert "engine_render" not in context
        assert "game" not in context

    def test_high_dep_crate_includes_all_deps(self, tmp_path: Path):
        """Focus in game (deps: engine_core, engine_render): both deps in context."""
        graph, files = _setup_workspace(tmp_path)
        all_files = list(files.values())
        summaries = {
            files["core_lib"]: "// file: crates/engine_core/src/lib.rs\npub fn core()",
            files["render_lib"]: "// file: crates/engine_render/src/lib.rs\npub fn render()",
            files["render_pipe"]: "// file: crates/engine_render/src/pipeline.rs\npub fn pipe()",
        }
        bundle = create_workspace_bundle(
            files["game_lib"], all_files, summaries, tmp_path, graph,
        )
        context = bundle.split("// === END CONTEXT ===")[0]
        assert "engine_core/src/lib.rs" in context
        assert "engine_render/src/lib.rs" in context
        assert "engine_render/src/pipeline.rs" in context

    def test_focus_content_is_correct(self, tmp_path: Path):
        """Focus file content is included regardless of graph filtering."""
        graph, files = _setup_workspace(tmp_path)
        all_files = list(files.values())
        bundle = create_workspace_bundle(
            files["render_lib"], all_files, {}, tmp_path, graph,
        )
        extracted = extract_focus(bundle)
        assert "pub fn render() {}" in extracted

    def test_focus_file_excluded_from_context(self, tmp_path: Path):
        """The focus file's own summary should not appear in context."""
        graph, files = _setup_workspace(tmp_path)
        all_files = list(files.values())
        summaries = {
            files["render_lib"]: "// file: crates/engine_render/src/lib.rs\npub fn render()",
            files["core_lib"]: "// file: crates/engine_core/src/lib.rs\npub fn core()",
        }
        bundle = create_workspace_bundle(
            files["render_lib"], all_files, summaries, tmp_path, graph,
        )
        context = bundle.split("// === END CONTEXT ===")[0]
        # The focus file's summary should not be in context
        assert "pub fn render()" not in context


# ---------------------------------------------------------------------------
# replace_focus
# ---------------------------------------------------------------------------

class TestReplaceFocus:
    def test_replaces_focus_content(self):
        bundle = (
            "// === CONTEXT (read-only \u2014 do NOT modify) ===\n"
            "// === END CONTEXT ===\n"
            "\n"
            "// === FOCUS: src/lib.rs ===\n"
            "// (This is the file you should improve. Output your improved version below.)\n"
            "fn original() {}\n"
            "// === END FOCUS ===\n"
        )
        result = replace_focus(bundle, "fn fixed() {}")
        extracted = extract_focus(result)
        assert "fn fixed() {}" in extracted
        assert "fn original() {}" not in extracted

    def test_preserves_context_section(self):
        bundle = (
            "// === CONTEXT (read-only \u2014 do NOT modify) ===\n"
            "// file: src/other.rs\n"
            "pub fn other() {}\n"
            "// === END CONTEXT ===\n"
            "\n"
            "// === FOCUS: src/lib.rs ===\n"
            "// (This is the file you should improve. Output your improved version below.)\n"
            "fn original() {}\n"
            "// === END FOCUS ===\n"
        )
        result = replace_focus(bundle, "fn fixed() {}")
        assert "// file: src/other.rs" in result
        assert "pub fn other() {}" in result

    def test_preserves_focus_path(self):
        bundle = (
            "// === FOCUS: crates/core/src/lib.rs ===\n"
            "// (This is the file you should improve. Output your improved version below.)\n"
            "fn original() {}\n"
            "// === END FOCUS ===\n"
        )
        result = replace_focus(bundle, "fn fixed() {}")
        assert extract_focus_path(result) == "crates/core/src/lib.rs"

    def test_roundtrip_with_create_bundle(self, tmp_path: Path):
        """replace_focus + extract_focus roundtrip works on real bundles."""
        inner = "pub fn old() -> i32 { 1 }"
        focus = _write_evolve(tmp_path, "src/lib.rs", inner)
        bundle = create_bundle(focus, [focus], {}, tmp_path)

        new_content = "pub fn new() -> i32 { 2 }"
        replaced = replace_focus(bundle, new_content)
        extracted = extract_focus(replaced)
        assert new_content in extracted
        assert inner not in extracted

    def test_raises_on_no_markers(self):
        with pytest.raises(ValueError, match="FOCUS markers"):
            replace_focus("no markers here", "anything")

    def test_multiline_replacement(self):
        bundle = (
            "// === FOCUS: src/lib.rs ===\n"
            "// (This is the file you should improve. Output your improved version below.)\n"
            "fn original() {}\n"
            "// === END FOCUS ===\n"
        )
        new_content = "fn line_one() {}\nfn line_two() {}\nfn line_three() {}"
        result = replace_focus(bundle, new_content)
        extracted = extract_focus(result)
        assert "fn line_one() {}" in extracted
        assert "fn line_two() {}" in extracted
        assert "fn line_three() {}" in extracted
