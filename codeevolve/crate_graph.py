"""Workspace detection and crate dependency graph.

Parses workspace Cargo.toml files to build a directed graph of local
path dependencies, and detects workspace-appropriate include/exclude
glob patterns for file discovery.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from glob import glob as stdlib_glob
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python < 3.11 fallback


@dataclass
class CrateGraph:
    """Directed dependency graph of local crates in a workspace."""

    # crate_name -> list of local dependency crate names
    deps: dict[str, list[str]] = field(default_factory=dict)
    # crate_name -> absolute Path to crate root directory
    crate_roots: dict[str, Path] = field(default_factory=dict)

    def direct_deps(self, crate_name: str) -> list[str]:
        """Return direct local dependencies of a crate."""
        return self.deps.get(crate_name, [])

    def relevant_crates(self, crate_name: str) -> list[str]:
        """Return the crate itself plus its direct dependencies."""
        return [crate_name] + self.direct_deps(crate_name)

    def crate_for_file(self, file_path: Path) -> str | None:
        """Find which crate a file belongs to by checking crate roots."""
        resolved = file_path.resolve()
        for name, root in self.crate_roots.items():
            try:
                resolved.relative_to(root.resolve())
                return name
            except ValueError:
                continue
        return None


@dataclass
class WorkspaceInfo:
    """Result of workspace detection."""

    include_globs: list[str]
    exclude_globs: list[str]
    crate_names: list[str]
    crate_graph: CrateGraph


def _parse_workspace_members(cargo_toml_path: Path) -> list[str] | None:
    """Parse [workspace].members from a Cargo.toml. Returns None if not a workspace."""
    with open(cargo_toml_path, "rb") as f:
        data = tomllib.load(f)
    workspace = data.get("workspace")
    if workspace is None:
        return None
    return workspace.get("members", [])


def _resolve_member_dirs(project_path: Path, member_patterns: list[str]) -> list[Path]:
    """Expand workspace member glob patterns to actual directories."""
    dirs: list[Path] = []
    for pattern in member_patterns:
        matches = sorted(project_path.glob(pattern))
        for m in matches:
            if m.is_dir() and (m / "Cargo.toml").exists():
                dirs.append(m)
    return dirs


def _parse_local_deps(cargo_toml_path: Path) -> list[str]:
    """Extract local path dependency crate names from a Cargo.toml."""
    with open(cargo_toml_path, "rb") as f:
        data = tomllib.load(f)
    deps = data.get("dependencies", {})
    local = []
    for name, spec in deps.items():
        if isinstance(spec, dict) and "path" in spec:
            dep_path = cargo_toml_path.parent / spec["path"]
            if dep_path.exists():
                dep_cargo = dep_path / "Cargo.toml"
                if dep_cargo.exists():
                    with open(dep_cargo, "rb") as f:
                        dep_data = tomllib.load(f)
                    pkg_name = dep_data.get("package", {}).get("name", dep_path.name)
                    local.append(pkg_name)
                else:
                    local.append(dep_path.name)
    return local


def _build_crate_graph(project_path: Path, member_dirs: list[Path]) -> CrateGraph:
    """Build a CrateGraph from resolved workspace member directories."""
    graph = CrateGraph()
    for member_dir in member_dirs:
        cargo_toml = member_dir / "Cargo.toml"
        with open(cargo_toml, "rb") as f:
            data = tomllib.load(f)
        crate_name = data.get("package", {}).get("name", member_dir.name)
        graph.crate_roots[crate_name] = member_dir
        graph.deps[crate_name] = _parse_local_deps(cargo_toml)
    return graph


def _detect_generated_dirs(project_path: Path, rs_files: list[Path]) -> list[str]:
    """Scan file paths for directories named 'generated/' and return exclude globs."""
    generated_parents: set[str] = set()
    for f in rs_files:
        parts = f.resolve().relative_to(project_path.resolve()).parts
        for i, part in enumerate(parts):
            if part == "generated":
                parent_pattern = "/".join(parts[: i + 1]) + "/**"
                generated_parents.add(parent_pattern)
                break
    return sorted(generated_parents)


def detect_workspace(project_path: Path) -> WorkspaceInfo | None:
    """Detect if project_path is a Cargo workspace and return workspace info.

    Returns None if the project is a single crate (no [workspace] section).
    """
    cargo_toml = project_path / "Cargo.toml"
    if not cargo_toml.exists():
        return None

    member_patterns = _parse_workspace_members(cargo_toml)
    if member_patterns is None:
        return None

    member_dirs = _resolve_member_dirs(project_path, member_patterns)
    if not member_dirs:
        return None

    graph = _build_crate_graph(project_path, member_dirs)

    include_globs = [f"{pattern}/src/**/*.rs" for pattern in member_patterns]

    all_rs: list[Path] = []
    for pattern in include_globs:
        all_rs.extend(project_path.glob(pattern))

    exclude_globs = _detect_generated_dirs(project_path, all_rs)

    return WorkspaceInfo(
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        crate_names=sorted(graph.crate_roots.keys()),
        crate_graph=graph,
    )
