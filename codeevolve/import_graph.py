"""Import graph analysis for Rust source files.

Scans .rs files for import statements (use crate::, use super::, mod declarations)
and builds reverse dependency counts to identify high-impact files.
"""
from __future__ import annotations

import re
from pathlib import Path

from codeevolve.crate_graph import CrateGraph

# Patterns to detect Rust imports
_USE_CRATE_RE = re.compile(r"^\s*use\s+crate::(\w+)")
_USE_SUPER_RE = re.compile(r"^\s*use\s+super::(\w+)")
_MOD_DECL_RE = re.compile(r"^\s*mod\s+(\w+)\s*;")


def _resolve_use_crate(crate_root: Path, project_path: Path, module_name: str) -> str | None:
    """Resolve `use crate::module_name` within a crate root to a relative file path."""
    src_dir = crate_root / "src"
    candidate = src_dir / f"{module_name}.rs"
    if candidate.exists():
        return candidate.relative_to(project_path).as_posix()
    candidate = src_dir / module_name / "mod.rs"
    if candidate.exists():
        return candidate.relative_to(project_path).as_posix()
    return None


def _resolve_use_super(file_path: Path, project_path: Path) -> str | None:
    """Resolve `use super::item` to the parent module file."""
    parent_dir = file_path.parent

    # For files like src/module.rs, super is the crate root (lib.rs/main.rs in same dir)
    if file_path.name != "mod.rs":
        for root_name in ("lib.rs", "main.rs", "mod.rs"):
            candidate = parent_dir / root_name
            if candidate.exists() and candidate.resolve() != file_path.resolve():
                return candidate.relative_to(project_path).as_posix()

    # For files inside a module directory (e.g. module_name/mod.rs or module_name/sub.rs),
    # super refers to the parent module
    grandparent = parent_dir.parent
    parent_mod = grandparent / f"{parent_dir.name}.rs"
    if parent_mod.exists():
        return parent_mod.relative_to(project_path).as_posix()
    for root_name in ("mod.rs", "lib.rs", "main.rs"):
        candidate = grandparent / root_name
        if candidate.exists():
            return candidate.relative_to(project_path).as_posix()
    return None


def _resolve_mod_decl(file_path: Path, project_path: Path, module_name: str) -> str | None:
    """Resolve `mod module_name;` to the child module file."""
    parent_dir = file_path.parent
    # Check sibling: parent_dir/module_name.rs
    candidate = parent_dir / f"{module_name}.rs"
    if candidate.exists():
        return candidate.relative_to(project_path).as_posix()
    # Check directory: parent_dir/module_name/mod.rs
    candidate = parent_dir / module_name / "mod.rs"
    if candidate.exists():
        return candidate.relative_to(project_path).as_posix()
    return None


def _find_crate_root_for_file(file_path: Path, project_path: Path, crate_graph: CrateGraph | None) -> Path | None:
    """Find the crate root directory for a given file."""
    if crate_graph is not None:
        crate_name = crate_graph.crate_for_file(file_path)
        if crate_name is not None:
            return crate_graph.crate_roots[crate_name]
    # Fallback: walk up looking for Cargo.toml
    current = file_path.parent
    while current != project_path.parent and current != current.parent:
        if (current / "Cargo.toml").exists():
            return current
        current = current.parent
    return None


def build_reverse_deps(
    project_path: Path,
    rs_files: list[Path],
    crate_graph: CrateGraph | None = None,
) -> dict[str, int]:
    """Build reverse dependency counts for Rust source files.

    Scans each file for import statements and counts how many other files
    depend on each file. When a CrateGraph is provided, also counts
    cross-crate dependents.

    Args:
        project_path: Root directory of the project.
        rs_files: List of absolute paths to .rs files to scan.
        crate_graph: Optional crate dependency graph for cross-crate impact.

    Returns:
        Dict mapping relative file path (POSIX) -> reverse dependency count.
    """
    project_path = project_path.resolve()

    # Initialize counts for all files
    rel_paths = {}
    for f in rs_files:
        try:
            rel = f.resolve().relative_to(project_path).as_posix()
            rel_paths[f.resolve()] = rel
        except ValueError:
            continue
    reverse_counts: dict[str, int] = {rel: 0 for rel in rel_paths.values()}

    # Scan each file for imports
    for f in rs_files:
        resolved = f.resolve()
        if resolved not in rel_paths:
            continue

        try:
            content = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        crate_root = _find_crate_root_for_file(resolved, project_path, crate_graph)
        deps_of_this_file: set[str] = set()

        for line in content.splitlines():
            # use crate::module
            m = _USE_CRATE_RE.match(line)
            if m:
                module_name = m.group(1)
                if crate_root is not None:
                    target = _resolve_use_crate(crate_root, project_path, module_name)
                    if target is not None:
                        deps_of_this_file.add(target)
                continue

            # use super::item
            m = _USE_SUPER_RE.match(line)
            if m:
                target = _resolve_use_super(resolved, project_path)
                if target is not None:
                    deps_of_this_file.add(target)
                continue

            # mod module_name;
            m = _MOD_DECL_RE.match(line)
            if m:
                module_name = m.group(1)
                target = _resolve_mod_decl(resolved, project_path, module_name)
                if target is not None:
                    deps_of_this_file.add(target)
                continue

        # Don't count self-references
        self_rel = rel_paths[resolved]
        deps_of_this_file.discard(self_rel)

        for dep in deps_of_this_file:
            if dep in reverse_counts:
                reverse_counts[dep] += 1

    # Add cross-crate impact
    if crate_graph is not None:
        # Build reverse crate deps: for each crate, how many other crates depend on it
        reverse_crate_deps: dict[str, int] = {name: 0 for name in crate_graph.deps}
        for crate_name, dep_list in crate_graph.deps.items():
            for dep in dep_list:
                if dep in reverse_crate_deps:
                    reverse_crate_deps[dep] += 1

        for f in rs_files:
            resolved = f.resolve()
            if resolved not in rel_paths:
                continue
            rel = rel_paths[resolved]
            crate_name = crate_graph.crate_for_file(resolved)
            if crate_name is not None and crate_name in reverse_crate_deps:
                reverse_counts[rel] += reverse_crate_deps[crate_name]

    return reverse_counts
