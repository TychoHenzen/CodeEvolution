"""Tests for codeevolve.import_graph — build_reverse_deps()."""
from __future__ import annotations

from pathlib import Path

import pytest

from codeevolve.crate_graph import CrateGraph
from codeevolve.import_graph import build_reverse_deps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_rs(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tests: use crate:: detection
# ---------------------------------------------------------------------------

class TestUseCrate:
    def test_use_crate_module_rs(self, tmp_path: Path):
        """use crate::utils should resolve to src/utils.rs and count as reverse dep."""
        _write_rs(tmp_path / "src" / "utils.rs", "pub fn helper() {}")
        _write_rs(tmp_path / "src" / "main.rs", "use crate::utils;\nfn main() {}")
        # Add Cargo.toml so crate root detection works
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "test"\n', encoding="utf-8")

        files = [tmp_path / "src" / "utils.rs", tmp_path / "src" / "main.rs"]
        result = build_reverse_deps(tmp_path, files)

        assert result["src/utils.rs"] == 1
        assert result["src/main.rs"] == 0

    def test_use_crate_module_mod_rs(self, tmp_path: Path):
        """use crate::utils should resolve to src/utils/mod.rs when it exists."""
        _write_rs(tmp_path / "src" / "utils" / "mod.rs", "pub fn helper() {}")
        _write_rs(tmp_path / "src" / "main.rs", "use crate::utils;\nfn main() {}")
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "test"\n', encoding="utf-8")

        files = [tmp_path / "src" / "utils" / "mod.rs", tmp_path / "src" / "main.rs"]
        result = build_reverse_deps(tmp_path, files)

        assert result["src/utils/mod.rs"] == 1
        assert result["src/main.rs"] == 0

    def test_use_crate_multiple_importers(self, tmp_path: Path):
        """Multiple files importing from the same module should sum up."""
        _write_rs(tmp_path / "src" / "core.rs", "pub fn core_fn() {}")
        _write_rs(tmp_path / "src" / "main.rs", "use crate::core;\nfn main() {}")
        _write_rs(tmp_path / "src" / "lib.rs", "use crate::core;\npub fn lib_fn() {}")
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "test"\n', encoding="utf-8")

        files = [
            tmp_path / "src" / "core.rs",
            tmp_path / "src" / "main.rs",
            tmp_path / "src" / "lib.rs",
        ]
        result = build_reverse_deps(tmp_path, files)

        assert result["src/core.rs"] == 2


# ---------------------------------------------------------------------------
# Tests: use super:: detection
# ---------------------------------------------------------------------------

class TestUseSuper:
    def test_use_super_resolves_to_parent_mod(self, tmp_path: Path):
        """use super::item in sub/child.rs should count sub.rs as a reverse dep."""
        _write_rs(tmp_path / "src" / "sub.rs", "pub fn parent_fn() {}")
        _write_rs(tmp_path / "src" / "sub" / "child.rs", "use super::parent_fn;")
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "test"\n', encoding="utf-8")

        files = [tmp_path / "src" / "sub.rs", tmp_path / "src" / "sub" / "child.rs"]
        result = build_reverse_deps(tmp_path, files)

        assert result["src/sub.rs"] == 1

    def test_use_super_resolves_to_lib_rs(self, tmp_path: Path):
        """use super::item in src/module.rs should resolve to src/lib.rs."""
        _write_rs(tmp_path / "src" / "lib.rs", "pub mod module;")
        _write_rs(tmp_path / "src" / "module.rs", "use super::something;")
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "test"\n', encoding="utf-8")

        files = [tmp_path / "src" / "lib.rs", tmp_path / "src" / "module.rs"]
        result = build_reverse_deps(tmp_path, files)

        assert result["src/lib.rs"] == 1


# ---------------------------------------------------------------------------
# Tests: mod declaration detection
# ---------------------------------------------------------------------------

class TestModDecl:
    def test_mod_decl_resolves_to_file(self, tmp_path: Path):
        """mod utils; should resolve to utils.rs and count as reverse dep."""
        _write_rs(tmp_path / "src" / "lib.rs", "mod utils;")
        _write_rs(tmp_path / "src" / "utils.rs", "pub fn helper() {}")
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "test"\n', encoding="utf-8")

        files = [tmp_path / "src" / "lib.rs", tmp_path / "src" / "utils.rs"]
        result = build_reverse_deps(tmp_path, files)

        assert result["src/utils.rs"] == 1

    def test_mod_decl_resolves_to_mod_rs(self, tmp_path: Path):
        """mod utils; should resolve to utils/mod.rs when directory exists."""
        _write_rs(tmp_path / "src" / "lib.rs", "mod utils;")
        _write_rs(tmp_path / "src" / "utils" / "mod.rs", "pub fn helper() {}")
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "test"\n', encoding="utf-8")

        files = [tmp_path / "src" / "lib.rs", tmp_path / "src" / "utils" / "mod.rs"]
        result = build_reverse_deps(tmp_path, files)

        assert result["src/utils/mod.rs"] == 1

    def test_mod_with_body_not_detected(self, tmp_path: Path):
        """mod utils { ... } should NOT be treated as an import."""
        _write_rs(tmp_path / "src" / "lib.rs", "mod utils {\n    pub fn inline() {}\n}")
        _write_rs(tmp_path / "src" / "utils.rs", "pub fn helper() {}")
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "test"\n', encoding="utf-8")

        files = [tmp_path / "src" / "lib.rs", tmp_path / "src" / "utils.rs"]
        result = build_reverse_deps(tmp_path, files)

        # utils.rs should NOT be counted because `mod utils { }` is inline, not a declaration
        assert result["src/utils.rs"] == 0


# ---------------------------------------------------------------------------
# Tests: reverse dependency counting
# ---------------------------------------------------------------------------

class TestReverseDeps:
    def test_file_with_no_importers_has_zero(self, tmp_path: Path):
        _write_rs(tmp_path / "src" / "orphan.rs", "fn lonely() {}")
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "test"\n', encoding="utf-8")

        files = [tmp_path / "src" / "orphan.rs"]
        result = build_reverse_deps(tmp_path, files)

        assert result["src/orphan.rs"] == 0

    def test_self_reference_not_counted(self, tmp_path: Path):
        """A file importing from itself should not inflate its own count."""
        _write_rs(tmp_path / "src" / "lib.rs", "use crate::lib;\nmod lib;")
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "test"\n', encoding="utf-8")

        files = [tmp_path / "src" / "lib.rs"]
        result = build_reverse_deps(tmp_path, files)

        assert result["src/lib.rs"] == 0

    def test_combined_imports(self, tmp_path: Path):
        """File with both use crate:: and mod declarations should count both targets."""
        _write_rs(tmp_path / "src" / "core.rs", "pub fn core_fn() {}")
        _write_rs(tmp_path / "src" / "utils.rs", "pub fn util_fn() {}")
        _write_rs(tmp_path / "src" / "main.rs", "use crate::core;\nmod utils;")
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "test"\n', encoding="utf-8")

        files = [
            tmp_path / "src" / "core.rs",
            tmp_path / "src" / "utils.rs",
            tmp_path / "src" / "main.rs",
        ]
        result = build_reverse_deps(tmp_path, files)

        assert result["src/core.rs"] == 1
        assert result["src/utils.rs"] == 1
        assert result["src/main.rs"] == 0


# ---------------------------------------------------------------------------
# Tests: cross-crate impact via CrateGraph
# ---------------------------------------------------------------------------

class TestCrossCrate:
    def test_cross_crate_adds_reverse_deps(self, tmp_path: Path):
        """Files in a crate depended on by others should get extra reverse dep counts."""
        # Set up two crates: core (depended on by app) and app
        core_dir = tmp_path / "crates" / "core"
        app_dir = tmp_path / "crates" / "app"

        _write_rs(core_dir / "src" / "lib.rs", "pub fn core_fn() {}")
        _write_rs(app_dir / "src" / "main.rs", "fn main() {}")

        graph = CrateGraph(
            deps={"core": [], "app": ["core"]},
            crate_roots={"core": core_dir, "app": app_dir},
        )

        files = [core_dir / "src" / "lib.rs", app_dir / "src" / "main.rs"]
        result = build_reverse_deps(tmp_path, files, crate_graph=graph)

        # core is depended on by app -> +1 cross-crate
        assert result["crates/core/src/lib.rs"] == 1
        # app has no reverse crate deps
        assert result["crates/app/src/main.rs"] == 0

    def test_cross_crate_multiple_dependents(self, tmp_path: Path):
        """Crate depended on by 3 others should get +3 for each file."""
        base_dir = tmp_path / "crates" / "base"
        _write_rs(base_dir / "src" / "lib.rs", "pub fn base_fn() {}")

        graph = CrateGraph(
            deps={
                "base": [],
                "a": ["base"],
                "b": ["base"],
                "c": ["base"],
            },
            crate_roots={
                "base": base_dir,
                "a": tmp_path / "crates" / "a",
                "b": tmp_path / "crates" / "b",
                "c": tmp_path / "crates" / "c",
            },
        )

        files = [base_dir / "src" / "lib.rs"]
        result = build_reverse_deps(tmp_path, files, crate_graph=graph)

        assert result["crates/base/src/lib.rs"] == 3

    def test_cross_crate_stacks_with_intra_crate(self, tmp_path: Path):
        """Cross-crate and intra-crate deps should add together."""
        core_dir = tmp_path / "crates" / "core"
        _write_rs(core_dir / "src" / "lib.rs", "mod utils;")
        _write_rs(core_dir / "src" / "utils.rs", "pub fn u() {}")

        graph = CrateGraph(
            deps={"core": [], "app": ["core"]},
            crate_roots={
                "core": core_dir,
                "app": tmp_path / "crates" / "app",
            },
        )

        files = [core_dir / "src" / "lib.rs", core_dir / "src" / "utils.rs"]
        result = build_reverse_deps(tmp_path, files, crate_graph=graph)

        # utils.rs: 1 intra-crate (from lib.rs mod decl) + 1 cross-crate (app depends on core)
        assert result["crates/core/src/utils.rs"] == 2
        # lib.rs: 0 intra-crate + 1 cross-crate
        assert result["crates/core/src/lib.rs"] == 1


# ---------------------------------------------------------------------------
# Tests: empty/missing files handled gracefully
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_file_list(self, tmp_path: Path):
        result = build_reverse_deps(tmp_path, [])
        assert result == {}

    def test_missing_file_skipped(self, tmp_path: Path):
        """A file path that doesn't exist on disk should be handled gracefully."""
        missing = tmp_path / "src" / "gone.rs"
        result = build_reverse_deps(tmp_path, [missing])
        # The file can't be read, so it just gets 0 or is skipped
        assert isinstance(result, dict)

    def test_empty_file_content(self, tmp_path: Path):
        """An empty .rs file should produce zero reverse deps."""
        _write_rs(tmp_path / "src" / "empty.rs", "")
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "test"\n', encoding="utf-8")

        files = [tmp_path / "src" / "empty.rs"]
        result = build_reverse_deps(tmp_path, files)

        assert result["src/empty.rs"] == 0

    def test_unresolvable_import_ignored(self, tmp_path: Path):
        """use crate::nonexistent should not crash, just be ignored."""
        _write_rs(tmp_path / "src" / "main.rs", "use crate::nonexistent;\nfn main() {}")
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "test"\n', encoding="utf-8")

        files = [tmp_path / "src" / "main.rs"]
        result = build_reverse_deps(tmp_path, files)

        assert result["src/main.rs"] == 0


# ---------------------------------------------------------------------------
# Tests: relative path output matches ledger paths
# ---------------------------------------------------------------------------

class TestRelativePaths:
    def test_paths_are_posix_relative(self, tmp_path: Path):
        """Output paths should be POSIX-style relative to project_path."""
        _write_rs(tmp_path / "src" / "lib.rs", "mod utils;")
        _write_rs(tmp_path / "src" / "utils.rs", "pub fn u() {}")
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "test"\n', encoding="utf-8")

        files = [tmp_path / "src" / "lib.rs", tmp_path / "src" / "utils.rs"]
        result = build_reverse_deps(tmp_path, files)

        for path in result:
            assert "/" in path or path.count("/") == 0  # POSIX separators
            assert not path.startswith("/")  # relative, not absolute
            assert "\\" not in path  # no Windows backslashes

    def test_workspace_paths_match_ledger_format(self, tmp_path: Path):
        """Paths like crates/foo/src/lib.rs should match ledger entry format."""
        crate_dir = tmp_path / "crates" / "foo"
        _write_rs(crate_dir / "src" / "lib.rs", "pub fn f() {}")

        graph = CrateGraph(
            deps={"foo": []},
            crate_roots={"foo": crate_dir},
        )

        files = [crate_dir / "src" / "lib.rs"]
        result = build_reverse_deps(tmp_path, files, crate_graph=graph)

        assert "crates/foo/src/lib.rs" in result
