# Workspace-Aware Mutator Setup for axiom2d

**Date:** 2026-04-10
**Status:** Approved
**Approach:** B (Workspace-Aware Init + Smart Bundling)

## Overview

Make CodeEvolution work correctly with multi-crate Rust workspaces, specifically targeting axiom2d (14 crates, ~240 non-generated .rs files). The current codebase assumes single-crate projects with `src/**/*.rs` layout. This spec covers auto-detecting workspace structure during init, building a crate dependency graph, and scoping bundle context to relevant crates during evolution.

## Problem

Running `codeevolve init` on axiom2d today discovers **zero files** because:
- Default `include_globs` is `["src/**/*.rs"]` but axiom2d uses `crates/*/src/**/*.rs`
- No workspace detection — init doesn't know it's looking at a workspace
- 361 generated files in `crates/card_game/src/card/art/generated/` would be included with no auto-exclusion
- Bundle context would include summaries from all 14 crates regardless of relevance, wasting context on unrelated code

## Design

### 1. Workspace Detection (init time)

When `codeevolve init --path <project>` runs, before file discovery:

1. Read `Cargo.toml` at project root
2. Check for `[workspace]` section with `members` key
3. If found, parse `members` patterns (e.g. `["crates/*"]`) and convert to include_globs by appending `src/**/*.rs` to each pattern
4. Scan discovered file paths for directories named `generated/` — add matching glob patterns to `exclude_globs`

For axiom2d this produces:
```yaml
include_globs:
  - "crates/*/src/**/*.rs"
exclude_globs:
  - "crates/card_game/src/card/art/generated/**"
```

The detection is a helper function, not a separate command. It returns a `WorkspaceInfo` dataclass (or `None` for single-crate projects) that the init flow uses to override defaults.

### 2. Crate Dependency Graph

New module `crate_graph.py` that parses workspace Cargo.toml files to build a directed graph of local path dependencies.

**Input:** Project root path + list of workspace member directories.

**Parsing:** For each member's `Cargo.toml`, extract `[dependencies]` entries with `path = "../<crate>"` keys. Only local path deps are tracked — external registry deps are ignored.

**Graph for axiom2d:**
```
engine_core      → (no local deps)
engine_ecs       → (no local deps)
engine_assets    → (no local deps)
engine_scene     → engine_core
engine_input     → engine_core
engine_physics   → engine_core
engine_render    → engine_core, engine_scene
engine_audio     → engine_core, engine_scene
engine_ui        → engine_core, engine_input, engine_render, engine_scene
engine_app       → engine_core, engine_ecs, engine_input, engine_render
axiom2d          → engine_core, engine_ecs, engine_app, engine_assets, engine_input, engine_audio, engine_physics, engine_render, engine_scene, engine_ui
card_game        → engine_app, engine_core, engine_input, engine_render, engine_physics, engine_scene, engine_ui
card_game_bin    → axiom2d, card_game
demo             → axiom2d
```

**API:**
```python
@dataclass
class CrateGraph:
    # crate_name -> list of local dependency crate names
    deps: dict[str, list[str]]
    # crate_name -> absolute Path to crate root directory
    crate_roots: dict[str, Path]

    def crate_for_file(self, file_path: Path) -> str | None
    def direct_deps(self, crate_name: str) -> list[str]
    def relevant_crates(self, crate_name: str) -> list[str]
        """Returns [crate_name] + direct_deps(crate_name)."""
```

### 3. Smart Bundling

**Context scoping rule:** When building a bundle for a focus file in crate X, include summaries only from:
1. Files in crate X itself (sibling files)
2. Files in crates that X directly depends on (one hop only)

No transitive deps, no reverse deps, no budget cap (Codex has ample context).

**Example:** Focus in `engine_ui` → summaries from engine_ui + engine_core + engine_input + engine_render + engine_scene (~84 files).

**Example:** Focus in `engine_core` → summaries from engine_core only (~10 files).

**Implementation:** New function `create_workspace_bundle(focus_file, all_files, summaries, project_path, crate_graph)` in `bundler.py`. Filters `all_files` and `summaries` to only files belonging to relevant crates before building the bundle. The existing `create_bundle` stays unchanged for single-crate backward compatibility.

**Integration:** `runner.py`'s `_run_multi_file` builds the `CrateGraph` once at startup, passes it to the new bundle function.

### 4. Init Output

```
$ codeevolve init --path D:\rust-target\axiom2d

  Detected workspace with 14 crates:
    axiom2d (8), card_game (458), card_game_bin (2), demo (5),
    engine_app (6), engine_assets (4), engine_audio (18),
    engine_core (11), engine_ecs (3), engine_input (14),
    engine_physics (13), engine_render (32), engine_scene (8),
    engine_ui (20)

  Auto-excluded generated directories:
    - crates/card_game/src/card/art/generated/** (361 files)

  Discovered 241 Rust source files after exclusions

  Marking files for evolution...
    Marked: crates/axiom2d/src/lib.rs
    ... (241 files)

  Setup complete! Files created in .codeevolve/

  Next steps:
    1. Edit .codeevolve/evolution.yaml to set provider, binary_package, etc.
    2. Start evolving:  codeevolve run
```

### 5. Config Generation

`generate_codeevolve_dir` accepts optional detected globs from workspace detection and writes them into `evolution.yaml` instead of defaults. All other config values (provider, fitness weights, benchmarks, llm_judgment) stay as current defaults — the user tunes post-init.

## Files Changed

**New file:**
- `codeevolve/crate_graph.py` — Workspace detection, Cargo.toml dependency parsing, `CrateGraph` dataclass, `detect_workspace()` entry point.

**Modified files:**
- `codeevolve/init_project.py` — `generate_codeevolve_dir` accepts detected globs, writes workspace-appropriate evolution.yaml. New `detect_generated_dirs()` helper.
- `codeevolve/cli.py` — `init` command calls `detect_workspace()` before file discovery, uses returned globs, prints workspace summary.
- `codeevolve/bundler.py` — New `create_workspace_bundle()` that filters summaries to focus crate + direct deps via `CrateGraph`.
- `codeevolve/runner.py` — `_run_multi_file` builds `CrateGraph` once, passes it to workspace bundle function.

**Not changed (already workspace-compatible):**
- `config.py`, `file_discovery.py`, `summary.py`, `evaluator/pipeline.py`, `evaluator/cargo.py`, `evaluator/benchmark.py`, `evaluator/llm_judge.py`, `evaluator/llm_fixer.py`, `llama_server.py`, `codex_proxy.py`, `claude_proxy.py`, `defaults/evolution.yaml`, `templates/evaluator.py.j2`.

## Scope Boundaries

**Not in this spec:**
- Focus file rotation across crates (v1: first file stays focus)
- Per-crate `cargo build -p` optimization (full workspace build is correct)
- Per-crate test isolation
- Interactive crate/file selection during init
- Workspace member `exclude` patterns from Cargo.toml (only `members` is parsed)
