# Workspace-Aware Mutator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CodeEvolution auto-detect and correctly handle multi-crate Rust workspaces (specifically axiom2d's 14-crate layout), including smart context scoping in bundles.

**Architecture:** New `crate_graph.py` module handles workspace detection and dependency parsing. Init flow uses it to set workspace-appropriate globs. Bundle creation uses the dependency graph to scope context to the focus crate + direct deps. All existing single-crate behavior is preserved via fallback paths.

**Tech Stack:** Python 3.13, toml (stdlib `tomllib`), Click, pytest, existing CodeEvolution modules.

---

## File Structure

**New files:**
- `codeevolve/crate_graph.py` — Workspace detection, Cargo.toml dependency parsing, `CrateGraph` and `WorkspaceInfo` dataclasses
- `tests/test_crate_graph.py` — Tests for workspace detection and dependency graph
- `tests/fixtures/sample_workspace/` — Fixture: 3-crate workspace for testing

**Modified files:**
- `codeevolve/init_project.py` — `generate_codeevolve_dir` accepts detected globs; new `detect_generated_dirs()`
- `codeevolve/cli.py` — Init command calls `detect_workspace()`, prints workspace summary, passes globs
- `codeevolve/bundler.py` — New `create_workspace_bundle()` that filters by crate graph
- `codeevolve/runner.py` — `_run_multi_file` builds CrateGraph, uses workspace bundle
- `tests/test_init_project.py` — Tests for workspace-aware init
- `tests/test_bundler.py` — Tests for workspace bundle filtering
- `tests/test_cli.py` — Tests for workspace init CLI output

---

### Task 1: Create sample_workspace test fixture

**Files:**
- Create: `tests/fixtures/sample_workspace/Cargo.toml`
- Create: `tests/fixtures/sample_workspace/crates/engine_core/Cargo.toml`
- Create: `tests/fixtures/sample_workspace/crates/engine_core/src/lib.rs`
- Create: `tests/fixtures/sample_workspace/crates/engine_render/Cargo.toml`
- Create: `tests/fixtures/sample_workspace/crates/engine_render/src/lib.rs`
- Create: `tests/fixtures/sample_workspace/crates/engine_render/src/pipeline.rs`
- Create: `tests/fixtures/sample_workspace/crates/game/Cargo.toml`
- Create: `tests/fixtures/sample_workspace/crates/game/src/lib.rs`
- Create: `tests/fixtures/sample_workspace/crates/game/src/card/mod.rs`
- Create: `tests/fixtures/sample_workspace/crates/game/src/card/generated/art.rs`

A minimal 3-crate workspace: `engine_core` (no local deps), `engine_render` (depends on engine_core), `game` (depends on engine_core + engine_render). Includes a `generated/` directory in `game` for exclusion testing.

- [ ] **Step 1: Create the fixture files**

`tests/fixtures/sample_workspace/Cargo.toml`:
```toml
[workspace]
members = ["crates/*"]
resolver = "2"
```

`tests/fixtures/sample_workspace/crates/engine_core/Cargo.toml`:
```toml
[package]
name = "engine_core"
version = "0.1.0"
edition = "2021"
```

`tests/fixtures/sample_workspace/crates/engine_core/src/lib.rs`:
```rust
// EVOLVE-BLOCK-START
pub fn core_add(a: i32, b: i32) -> i32 {
    a + b
}
// EVOLVE-BLOCK-END

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_core_add() { assert_eq!(core_add(1, 2), 3); }
}
```

`tests/fixtures/sample_workspace/crates/engine_render/Cargo.toml`:
```toml
[package]
name = "engine_render"
version = "0.1.0"
edition = "2021"

[dependencies]
engine_core = { path = "../engine_core" }
```

`tests/fixtures/sample_workspace/crates/engine_render/src/lib.rs`:
```rust
// EVOLVE-BLOCK-START
pub struct Renderer {
    pub width: u32,
    pub height: u32,
}
// EVOLVE-BLOCK-END
```

`tests/fixtures/sample_workspace/crates/engine_render/src/pipeline.rs`:
```rust
// EVOLVE-BLOCK-START
pub fn create_pipeline() -> bool {
    true
}
// EVOLVE-BLOCK-END
```

`tests/fixtures/sample_workspace/crates/game/Cargo.toml`:
```toml
[package]
name = "game"
version = "0.1.0"
edition = "2021"

[dependencies]
engine_core = { path = "../engine_core" }
engine_render = { path = "../engine_render" }
```

`tests/fixtures/sample_workspace/crates/game/src/lib.rs`:
```rust
// EVOLVE-BLOCK-START
pub fn run_game() -> bool {
    true
}
// EVOLVE-BLOCK-END
```

`tests/fixtures/sample_workspace/crates/game/src/card/mod.rs`:
```rust
pub mod generated;
```

`tests/fixtures/sample_workspace/crates/game/src/card/generated/art.rs`:
```rust
// This is auto-generated, should be excluded from evolution
pub const CARD_ART: &str = "generated";
```

- [ ] **Step 2: Add conftest fixture for sample_workspace**

Add to `tests/conftest.py`:
```python
@pytest.fixture
def sample_workspace(tmp_path: Path) -> Path:
    """Copy the sample workspace to a temp directory so tests can modify it."""
    dest = tmp_path / "sample_workspace"
    shutil.copytree(FIXTURES_DIR / "sample_workspace", dest)
    return dest
```

- [ ] **Step 3: Verify fixture is loadable**

Run: `.venv/Scripts/python.exe -m pytest tests/test_init_project.py -v --collect-only`
Expected: All existing tests collected, no import errors.

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/sample_workspace/ tests/conftest.py
git commit -m "test: add sample_workspace fixture for workspace-aware tests"
```

---

### Task 2: Implement CrateGraph and detect_workspace

**Files:**
- Create: `codeevolve/crate_graph.py`
- Create: `tests/test_crate_graph.py`

- [ ] **Step 1: Write failing tests for detect_workspace**

Create `tests/test_crate_graph.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_crate_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'codeevolve.crate_graph'`

- [ ] **Step 3: Write failing tests for CrateGraph**

Append to `tests/test_crate_graph.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_crate_graph.py -v`
Expected: FAIL (module not found)

- [ ] **Step 5: Implement crate_graph.py**

Create `codeevolve/crate_graph.py`:
```python
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
        # Cargo workspace members use forward slashes and glob syntax
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
            # Resolve the path to find the actual crate directory name
            dep_path = cargo_toml_path.parent / spec["path"]
            if dep_path.exists():
                # Read the dependency's Cargo.toml to get its package name
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
                # Build a glob pattern up to and including the generated dir
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

    # Build include_globs from member patterns
    include_globs = [f"{pattern}/src/**/*.rs" for pattern in member_patterns]

    # Discover all .rs files to detect generated dirs
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_crate_graph.py -v`
Expected: All 10 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add codeevolve/crate_graph.py tests/test_crate_graph.py
git commit -m "feat: add crate_graph module for workspace detection and dependency parsing"
```

---

### Task 3: Wire workspace detection into init

**Files:**
- Modify: `codeevolve/init_project.py:100-182` (`generate_codeevolve_dir`)
- Modify: `codeevolve/cli.py:54-95` (`init` command)
- Test: `tests/test_init_project.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing test for generate_codeevolve_dir with workspace globs**

Add to `tests/test_init_project.py`:
```python
import yaml


def test_generate_codeevolve_dir_with_workspace_globs(sample_workspace: Path):
    """Workspace globs are written into evolution.yaml when provided."""
    rs_files = list((sample_workspace / "crates").rglob("*.rs"))
    generate_codeevolve_dir(
        project_path=sample_workspace,
        rs_files=rs_files,
        include_globs=["crates/*/src/**/*.rs"],
        exclude_globs=["crates/game/src/card/generated/**"],
    )
    config_path = sample_workspace / ".codeevolve" / "evolution.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    assert config["include_globs"] == ["crates/*/src/**/*.rs"]
    assert "crates/game/src/card/generated/**" in config["exclude_globs"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_init_project.py::test_generate_codeevolve_dir_with_workspace_globs -v`
Expected: FAIL with `TypeError: generate_codeevolve_dir() got an unexpected keyword argument 'include_globs'`

- [ ] **Step 3: Modify generate_codeevolve_dir to accept workspace globs**

In `codeevolve/init_project.py`, change the `generate_codeevolve_dir` signature and body. Replace the function starting at line 100:

```python
def generate_codeevolve_dir(
    project_path: Path,
    rs_files: list[Path],
    custom_benchmark: Optional[str] = None,
    custom_benchmark_regex: Optional[str] = None,
    include_globs: Optional[list[str]] = None,
    exclude_globs: Optional[list[str]] = None,
) -> Path:
    """Generate the .codeevolve/ directory with config, evaluator, and README."""
    codeevolve_dir = project_path / ".codeevolve"
    codeevolve_dir.mkdir(exist_ok=True)

    # --- Generate evolution.yaml ---
    with open(_DEFAULTS_DIR / "evolution.yaml") as f:
        config_data = yaml.safe_load(f)

    if custom_benchmark:
        config_data["benchmarks"]["custom_command"] = custom_benchmark
    if custom_benchmark_regex:
        config_data["benchmarks"]["custom_command_score_regex"] = _SingleQuotedStr(
            custom_benchmark_regex
        )

    # Override globs if workspace detection provided them
    if include_globs is not None:
        config_data["include_globs"] = include_globs
    if exclude_globs is not None:
        config_data["exclude_globs"] = exclude_globs

    config_path = codeevolve_dir / "evolution.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f, Dumper=_ConfigDumper, default_flow_style=False, sort_keys=False)

    # --- Generate evaluator.py ---
    regenerate_evaluator(
        project_path, config_path,
        source_files=rs_files,
        focus_file=rs_files[0],
    )

    # --- Generate README.md ---
    file_list = "\n".join(f"  - {f.relative_to(project_path)}" for f in rs_files)
    readme = f"""# CodeEvolution Setup

This directory was generated by `codeevolve init`. Here's what each file does:

## Files

- **evolution.yaml** — Configuration for the evolutionary optimizer. Controls which
  model to use, how many iterations to run, fitness weights, and more.
  Edit this to tune the evolution. All fields have sensible defaults.

- **evaluator.py** — The fitness function that scores each code candidate. This is
  called automatically by the optimizer. You generally don't need to edit this.

## Evolved Files

The following source files have been marked for evolution:
{file_list}

Each file has `// EVOLVE-BLOCK-START` and `// EVOLVE-BLOCK-END` markers around
the code that the optimizer is allowed to change. You can move these markers to
control which parts of your code get evolved.

## How to Run

1. Download the model GGUF if you haven't already
2. Set `server_path` and `model_path` in `evolution.yaml` to point to your
   llama-server binary and .gguf model file
3. Start the evolution: `codeevolve run`

The server is started and stopped automatically by `codeevolve run`.

The optimizer will try hundreds of variations of your code, keeping only the ones
that compile, pass tests, and score well on code quality metrics.

Results are saved to `.codeevolve/output/`. The best version is in `output/best/`.

## How It Works

Each candidate goes through 4 checks:
1. **Build + Tests** — Must compile and pass all tests (hard requirement)
2. **Clippy Analysis** — Fewer warnings = better score
3. **Performance** — Faster compile time and smaller binary = better score
4. **AI Review** — An AI model rates code quality (only for top candidates)

Press Ctrl+C at any time to stop. Your best result so far is always saved.
"""
    (codeevolve_dir / "README.md").write_text(readme)

    return codeevolve_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_init_project.py -v`
Expected: All tests PASS (including existing ones — the new params are optional).

- [ ] **Step 5: Write failing test for CLI workspace init**

Add to `tests/test_cli.py`:
```python
def test_init_workspace_detects_crates(cli_runner, sample_workspace):
    result = cli_runner.invoke(main, ["init", "--path", str(sample_workspace)])
    assert result.exit_code == 0
    assert "Detected workspace" in result.output
    assert "engine_core" in result.output
    assert "engine_render" in result.output
    assert "game" in result.output


def test_init_workspace_excludes_generated(cli_runner, sample_workspace):
    result = cli_runner.invoke(main, ["init", "--path", str(sample_workspace)])
    assert result.exit_code == 0
    assert "generated" in result.output.lower()
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py::test_init_workspace_detects_crates tests/test_cli.py::test_init_workspace_excludes_generated -v`
Expected: FAIL (no "Detected workspace" in output because CLI doesn't call detect_workspace yet)

- [ ] **Step 7: Modify CLI init to use workspace detection**

Replace the `init` command in `codeevolve/cli.py` (lines 54-95):

```python
@main.command()
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".", help="Path to Rust project (default: current directory)")
def init(path: Path):
    """Set up a Rust project for evolutionary optimization.

    Scans your project for Rust source files using glob patterns, marks
    them for evolution, and generates configuration files in a .codeevolve/
    directory.
    """
    from codeevolve.crate_graph import detect_workspace

    path = path.resolve()

    try:
        find_cargo_toml(path)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Detect workspace structure
    workspace_info = detect_workspace(path)
    if workspace_info is not None:
        include_globs = workspace_info.include_globs
        exclude_globs = workspace_info.exclude_globs

        click.echo(f"\n  Detected workspace with {len(workspace_info.crate_names)} crates:")
        # Count files per crate
        crate_counts = []
        for crate_name in workspace_info.crate_names:
            crate_root = workspace_info.crate_graph.crate_roots[crate_name]
            rs_count = len(list((crate_root / "src").rglob("*.rs"))) if (crate_root / "src").exists() else 0
            crate_counts.append(f"{crate_name} ({rs_count})")
        click.echo("    " + ", ".join(crate_counts))

        if exclude_globs:
            click.echo("\n  Auto-excluded generated directories:")
            for g in exclude_globs:
                click.echo(f"    - {g}")
    else:
        # Single-crate fallback: use defaults
        defaults = load_config()
        include_globs = defaults.include_globs
        exclude_globs = defaults.exclude_globs

    rs_files = discover_rs_files(path, include_globs, exclude_globs)
    if not rs_files:
        click.echo("Error: No .rs files matched include globs", err=True)
        sys.exit(1)

    click.echo(f"\n  Discovered {len(rs_files)} Rust source file(s)")

    click.echo("\n  Marking files for evolution...")
    for f in rs_files:
        insert_evolve_markers(f)

    codeevolve_dir = generate_codeevolve_dir(
        project_path=path,
        rs_files=rs_files,
        include_globs=include_globs if workspace_info is not None else None,
        exclude_globs=exclude_globs if workspace_info is not None else None,
    )

    click.echo(f"\n  Setup complete! Files created in {codeevolve_dir.relative_to(path)}/")
    click.echo("\n  Next steps:")
    click.echo("    1. Edit .codeevolve/evolution.yaml to set provider, binary_package, etc.")
    click.echo("    2. Start evolving:  codeevolve run")
```

- [ ] **Step 8: Run all tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py tests/test_init_project.py -v`
Expected: All tests PASS.

- [ ] **Step 9: Commit**

```bash
git add codeevolve/init_project.py codeevolve/cli.py tests/test_init_project.py tests/test_cli.py
git commit -m "feat: wire workspace detection into codeevolve init"
```

---

### Task 4: Implement workspace-aware bundling

**Files:**
- Modify: `codeevolve/bundler.py`
- Test: `tests/test_bundler.py`

- [ ] **Step 1: Write failing tests for create_workspace_bundle**

Add to `tests/test_bundler.py`:
```python
from codeevolve.crate_graph import CrateGraph
from codeevolve.bundler import create_workspace_bundle


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_bundler.py::TestCreateWorkspaceBundle -v`
Expected: FAIL with `ImportError: cannot import name 'create_workspace_bundle'`

- [ ] **Step 3: Implement create_workspace_bundle**

Add to `codeevolve/bundler.py` after the existing `create_bundle` function:

```python
def create_workspace_bundle(
    focus_file: Path,
    all_files: list[Path],
    summaries: dict[Path, str],
    project_path: Path,
    crate_graph: "CrateGraph",
) -> str:
    """Create a bundled program string scoped to relevant crates.

    Like ``create_bundle``, but filters summaries to only include files
    from the focus file's crate and its direct dependencies (one hop).

    Args:
        focus_file: Absolute path to the file being evolved.
        all_files: All .rs files in the workspace (absolute paths).
        summaries: Pre-computed summaries keyed by absolute path.
        project_path: Project root for computing relative paths.
        crate_graph: Dependency graph from ``crate_graph.build_crate_graph``.

    Returns:
        A single string that OpenEvolve treats as the "initial program."
    """
    focus_crate = crate_graph.crate_for_file(focus_file)

    if focus_crate is None:
        # Fallback: can't determine crate, include everything
        return create_bundle(focus_file, all_files, summaries, project_path)

    relevant = set(crate_graph.relevant_crates(focus_crate))

    # Filter files to only those in relevant crates
    filtered_files = [
        f for f in all_files
        if crate_graph.crate_for_file(f) in relevant
    ]

    # Filter summaries to only relevant files
    filtered_summaries = {
        f: s for f, s in summaries.items()
        if crate_graph.crate_for_file(f) in relevant
    }

    return create_bundle(focus_file, filtered_files, filtered_summaries, project_path)
```

Add the import at the top of `bundler.py` (with the other imports — use TYPE_CHECKING to avoid circular imports):

```python
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeevolve.crate_graph import CrateGraph

from codeevolve.evaluator.pipeline import parse_evolve_block
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_bundler.py -v`
Expected: All tests PASS (existing + new).

- [ ] **Step 5: Commit**

```bash
git add codeevolve/bundler.py tests/test_bundler.py
git commit -m "feat: add create_workspace_bundle with crate-scoped context filtering"
```

---

### Task 5: Wire workspace bundling into runner

**Files:**
- Modify: `codeevolve/runner.py:338-397` (`_run_multi_file`)
- Test: `tests/test_runner.py`

- [ ] **Step 1: Read current test_runner.py to understand test patterns**

Read `tests/test_runner.py` to understand existing test setup before writing new tests.

- [ ] **Step 2: Write failing test for workspace-aware runner**

Add to `tests/test_runner.py`:
```python
from unittest.mock import patch, MagicMock
from codeevolve.runner import _run_multi_file
from codeevolve.config import load_config


def test_run_multi_file_uses_workspace_bundle(sample_workspace, tmp_path):
    """When a workspace is detected, _run_multi_file uses create_workspace_bundle."""
    config = load_config()
    config_path = sample_workspace / ".codeevolve" / "evolution.yaml"
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    source_files = sorted((sample_workspace / "crates").rglob("*.rs"))
    # Filter to only files with EVOLVE-BLOCK markers
    marked = [f for f in source_files if "EVOLVE-BLOCK-START" in f.read_text()]

    # Patch at SOURCE modules — _run_multi_file uses local imports, so
    # the from-import will pick up whatever is in the source module at
    # call time.
    with patch("openevolve.api.run_evolution") as mock_oe, \
         patch("codeevolve.bundler.create_workspace_bundle") as mock_ws_bundle, \
         patch("codeevolve.summary.summarize_files") as mock_summarize:
        mock_ws_bundle.return_value = "// bundle content"
        mock_summarize.return_value = {}
        mock_oe.return_value = MagicMock(best_code="pub fn improved() {}")

        # This should detect workspace and use create_workspace_bundle
        _run_multi_file(
            config, config_path, sample_workspace, marked,
            config_path.parent / "evaluator.py", output_dir,
        )
        mock_ws_bundle.assert_called_once()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_runner.py::test_run_multi_file_uses_workspace_bundle -v`
Expected: FAIL (no `create_workspace_bundle` import in runner.py)

- [ ] **Step 4: Modify _run_multi_file to use workspace bundling**

In `codeevolve/runner.py`, modify the `_run_multi_file` function. Replace lines 338-397:

```python
def _run_multi_file(
    config: CodeEvolveConfig,
    config_path: Path,
    project_path: Path,
    source_files: list[Path],
    evaluator_path: Path,
    output_dir: Path,
):
    """Multi-file evolution flow using bundles and summaries."""
    from openevolve.api import run_evolution as oe_run_evolution
    from codeevolve.summary import summarize_files
    from codeevolve.bundler import create_bundle, create_workspace_bundle
    from codeevolve.crate_graph import detect_workspace

    # Backup ALL source files
    backups: dict[Path, Path] = {}
    for f in source_files:
        backup_path = output_dir / f"{f.name}.backup"
        backup_path.write_text(f.read_text())
        backups[f] = backup_path
    logger.info("Saved %d source backups", len(backups))

    # Generate summaries of all files for context
    summaries = summarize_files(source_files, project_path)
    logger.info("Generated summaries for %d files", len(summaries))

    # Detect workspace for smart context scoping
    workspace_info = detect_workspace(project_path)

    # Create bundle with first file as focus (v1: no rotation)
    focus_file = source_files[0]
    if workspace_info is not None:
        bundle = create_workspace_bundle(
            focus_file, source_files, summaries, project_path,
            workspace_info.crate_graph,
        )
        logger.info("Created workspace bundle (focus=%s)", focus_file.name)
    else:
        bundle = create_bundle(focus_file, source_files, summaries, project_path)
        logger.info("Created bundle (focus=%s)", focus_file.name)

    bundle_path = output_dir / "initial_bundle.rs"
    bundle_path.write_text(bundle)
    logger.info("Bundle written (%d chars)", len(bundle))

    # For the bundle path, frozen context is not needed separately --
    # the bundle's CONTEXT section provides read-only context to the LLM.
    oe_config_path = build_openevolve_config_yaml(config, output_dir, frozen_context="")

    try:
        result = oe_run_evolution(
            initial_program=str(bundle_path),
            evaluator=str(evaluator_path),
            config=str(oe_config_path),
            iterations=config.evolution.max_iterations,
            output_dir=str(output_dir),
            cleanup=False,
        )
    finally:
        # Restore ALL original source files
        for f, backup in backups.items():
            f.write_text(backup.read_text())
        logger.info("Restored %d source files from backups", len(backups))

    best_dir = output_dir / "best"
    best_dir.mkdir(exist_ok=True)
    if result.best_code:
        (best_dir / focus_file.name).write_text(result.best_code)

    return result
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_runner.py -v`
Expected: All tests PASS.

- [ ] **Step 6: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: All tests PASS. No regressions.

- [ ] **Step 7: Commit**

```bash
git add codeevolve/runner.py tests/test_runner.py
git commit -m "feat: use workspace-aware bundling in multi-file evolution"
```

---

### Task 6: End-to-end validation with axiom2d

**Files:**
- No code changes — validation only

- [ ] **Step 1: Run codeevolve init on axiom2d (dry run)**

Run: `.venv/Scripts/python.exe -c "from codeevolve.crate_graph import detect_workspace; from pathlib import Path; info = detect_workspace(Path('D:/rust-target/axiom2d')); print(f'Crates: {info.crate_names}'); print(f'Include: {info.include_globs}'); print(f'Exclude: {info.exclude_globs}'); print(f'Graph deps:'); [print(f'  {k}: {v}') for k,v in info.crate_graph.deps.items()]"`

Expected output should show:
- 14 crate names
- Include glob: `crates/*/src/**/*.rs`
- Exclude glob containing `generated`
- Correct dependency graph matching the axiom2d Cargo.toml structure

- [ ] **Step 2: Test file discovery with workspace globs**

Run: `.venv/Scripts/python.exe -c "from codeevolve.file_discovery import discover_rs_files; from pathlib import Path; files = discover_rs_files(Path('D:/rust-target/axiom2d'), ['crates/*/src/**/*.rs'], ['crates/card_game/src/card/art/generated/**']); print(f'{len(files)} files discovered'); print(f'Sample: {[f.name for f in files[:5]]}')"` 

Expected: ~241 files discovered (602 total minus 361 generated).

- [ ] **Step 3: Test workspace bundle context scoping**

Run: `.venv/Scripts/python.exe -c "
from pathlib import Path
from codeevolve.crate_graph import detect_workspace
from codeevolve.summary import summarize_files
from codeevolve.file_discovery import discover_rs_files
from codeevolve.bundler import create_workspace_bundle

project = Path('D:/rust-target/axiom2d')
info = detect_workspace(project)
files = discover_rs_files(project, info.include_globs, info.exclude_globs)
summaries = summarize_files(files, project)

# Test with a focus file in engine_core (leaf crate, no deps)
core_files = [f for f in files if 'engine_core' in str(f)]
focus = core_files[0]
bundle = create_workspace_bundle(focus, files, summaries, project, info.crate_graph)
context = bundle.split('// === END CONTEXT ===')[0]
print(f'Focus: {focus.name} (engine_core)')
print(f'Bundle length: {len(bundle)} chars')
print(f'Context crates mentioned:')
for crate in info.crate_names:
    if crate in context:
        print(f'  - {crate}')
"`

Expected: Only `engine_core` files in context (no other crates since engine_core has no deps).

- [ ] **Step 4: Run the full test suite one final time**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit any test fixes**

Only if Step 4 revealed issues. Otherwise skip.
