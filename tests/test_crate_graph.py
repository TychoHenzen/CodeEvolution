"""Tests for codeevolve.crate_graph — workspace detection and dependency graph."""
from __future__ import annotations

from pathlib import Path

import pytest

from codeevolve.crate_graph import detect_workspace, CrateGraph


class TestDetectWorkspace:
    def test_returns_none_for_single_crate(self, sample_crate: Path):
        result = detect_workspace(sample_crate)
        assert result is None

    def test_detects_workspace(self, sample_workspace: Path):
        result = detect_workspace(sample_workspace)
        assert result is not None
        assert "crates/*/src/**/*.rs" in result.include_globs

    def test_workspace_has_crate_graph(self, sample_workspace: Path):
        result = detect_workspace(sample_workspace)
        assert result.crate_graph is not None
        assert "engine_core" in result.crate_graph.crate_roots
        assert "engine_render" in result.crate_graph.crate_roots
        assert "game" in result.crate_graph.crate_roots

    def test_detects_generated_dirs(self, sample_workspace: Path):
        result = detect_workspace(sample_workspace)
        assert any("generated" in g for g in result.exclude_globs)


class TestCrateGraph:
    def test_direct_deps(self, sample_workspace: Path):
        info = detect_workspace(sample_workspace)
        graph = info.crate_graph
        assert graph.direct_deps("engine_core") == []
        assert "engine_core" in graph.direct_deps("engine_render")
        deps = graph.direct_deps("game")
        assert "engine_core" in deps
        assert "engine_render" in deps

    def test_relevant_crates_includes_self(self, sample_workspace: Path):
        info = detect_workspace(sample_workspace)
        graph = info.crate_graph
        relevant = graph.relevant_crates("engine_core")
        assert "engine_core" in relevant

    def test_relevant_crates_includes_direct_deps(self, sample_workspace: Path):
        info = detect_workspace(sample_workspace)
        graph = info.crate_graph
        relevant = graph.relevant_crates("game")
        assert "game" in relevant
        assert "engine_core" in relevant
        assert "engine_render" in relevant

    def test_relevant_crates_excludes_reverse_deps(self, sample_workspace: Path):
        info = detect_workspace(sample_workspace)
        graph = info.crate_graph
        # engine_core doesn't depend on game
        relevant = graph.relevant_crates("engine_core")
        assert "game" not in relevant

    def test_crate_for_file(self, sample_workspace: Path):
        info = detect_workspace(sample_workspace)
        graph = info.crate_graph
        lib_rs = sample_workspace / "crates" / "engine_render" / "src" / "lib.rs"
        assert graph.crate_for_file(lib_rs) == "engine_render"

    def test_crate_for_file_returns_none_for_unknown(self, sample_workspace: Path):
        info = detect_workspace(sample_workspace)
        graph = info.crate_graph
        unknown = sample_workspace / "random" / "file.rs"
        assert graph.crate_for_file(unknown) is None
