# Workspace-Level Evolution Support — Requirements Spec

> **For Claude:** This spec was produced by /interview. Use /writing-plans to expand into an implementation plan, or /tdd to implement directly.

**Goal:** Extend CodeEvolution to support workspace-level evolution of Rust projects, using glob-based file selection, a summary+focus-file context strategy, and workspace-level evaluation including post-UPX binary size measurement.

**Date:** 2026-04-10

---

## Requirements

### Config & File Selection

- Add `include_globs: list[str]` and `exclude_globs: list[str]` to config dataclass and YAML
- Example: `include: ["crates/**/*.rs"]`, `exclude: ["**/generated/**", "**/target/**"]`
- Remove the workspace rejection in `init_project.py` — workspaces are now first-class
- `codeevolve init` scans the workspace using globs, inserts EVOLVE-BLOCK markers on all matched files
- Set `cargo bench` as the default `benchmarks.custom_command` in `codeevolve/defaults/evolution.yaml`

### Multi-File Evolution (Summary + Focus File)

OpenEvolve is fundamentally single-file (`code: str`). To evolve a workspace:

- Each iteration, **one** matched file is the "focus" (fully editable by the LLM)
- All other matched files are presented as **read-only summaries** (pub fn signatures, struct/enum defs, trait impls, key types) — providing cross-crate context without overwhelming the LLM
- The focus file rotates across iterations so all files get evolved over time
- The evaluator receives the single "program" from OpenEvolve, extracts the focus file edit, writes it back to disk, and runs workspace-level evaluation
- The bundle format must clearly delimit the editable section from read-only context

### Summary Generation

- Must be fast — static text parsing, not cargo-based (no `cargo doc` or `rust-analyzer`)
- Extract: `pub fn` signatures, `pub struct`/`pub enum` definitions, `pub trait` definitions, `impl` block headers
- Include the file path as context so the LLM understands cross-crate relationships
- Summaries are regenerated at init time and cached; refreshed when files change

### Evaluation Pipeline (Workspace-Level)

All evaluation runs at the workspace root to catch cross-crate breakage:

- **Layer 1 (hard gate):** `cargo build` + `cargo test` at workspace root
- **Layer 2 (static analysis):** `cargo clippy` at workspace root
- **Layer 3 (benchmarks):**
  - Compile time: `cargo build` duration
  - Binary size: `cargo build --release -p <binary_package>` + UPX compression, measure post-UPX size
  - LOC measurement across all focus files
  - Custom benchmark: `cargo bench` (now the default)
- **Layer 4 (LLM judgment):** Unchanged from current single-file implementation

### Binary Size Measurement

- New config fields in `BenchmarksConfig`:
  - `binary_package: Optional[str]` — e.g. `"card_game_bin"` (which `-p` to build)
  - `upx_path: Optional[str]` — e.g. `"upx.exe"` or absolute path
  - `upx_args: list[str]` — e.g. `["--best", "--force"]`
- Always measured for every candidate (release build + UPX)
- Measures post-UPX compressed binary size
- If `upx_path` is null, measures pre-UPX release binary size
- Binary path derived from: `target/release/<binary_package>.exe` (Windows)

### Default Configuration for axiom2d

The primary target project. Expected `.codeevolve/evolution.yaml`:

```yaml
provider: "codex"

include_globs:
  - "crates/**/*.rs"
exclude_globs:
  - "**/generated/**"
  - "**/target/**"
  - "**/tests/**"       # optional, exclude test files from evolution

evolution:
  diff_based_evolution: true
  population_size: 10
  num_islands: 3

rust:
  cargo_path: "cargo"
  target_dir: null

benchmarks:
  measure_compile_time: true
  measure_binary_size: true
  binary_package: "card_game_bin"
  upx_path: "upx.exe"
  upx_args: ["--best", "--force"]
  custom_command: "cargo bench"

fitness:
  static_analysis_weight: 0.35
  performance_weight: 0.35
  llm_judgment_weight: 0.30
```

### Scope Boundaries (NOT included)

- Multi-workspace support (still one workspace at a time)
- Modifying OpenEvolve itself (works within its single-program API)
- Interactive file picking UX (globs replace the old comma-separated index picker)
- Evolving generated files, test files, or build scripts
- Rust-analyzer or LSP integration for summaries

### Constraints

- Runs on Windows Python 3.13 via WSL `.venv/Scripts/python.exe`
- UPX and cargo are native Windows binaries (no WSL interop needed)
- Must stay compatible with OpenEvolve's single-`code: str` program model
- Summary generation must be fast (static regex/parsing, not cargo-based)
- EVOLVE-BLOCK markers remain the mechanism for identifying evolvable code

---

## Subtask Checklist

### Phase 1: Config & Data Model

- [ ] **1.1** Add `include_globs`, `exclude_globs` fields to config dataclass (`config.py`)
- [ ] **1.2** Add `binary_package`, `upx_path`, `upx_args` fields to `BenchmarksConfig` (`config.py`)
- [ ] **1.3** Update `codeevolve/defaults/evolution.yaml` with new fields and `cargo bench` as default `custom_command`
- [ ] **1.4** Update config loading/merging to handle the new list fields

### Phase 2: Glob-Based File Discovery

- [ ] **2.1** Create `file_discovery.py` module: given `project_path`, `include_globs`, `exclude_globs`, return list of matched `.rs` file paths
- [ ] **2.2** Remove workspace rejection from `init_project.py:find_cargo_toml()`
- [ ] **2.3** Replace `scan_rs_files()` with glob-based discovery using new module
- [ ] **2.4** Update `codeevolve init` CLI to use glob discovery and show matched file count/list

### Phase 3: Rust Summary Generator

- [ ] **3.1** Create `summary.py` module: parse a `.rs` file and extract pub signatures, struct/enum/trait defs, impl headers
- [ ] **3.2** Output format: compact text with file path header, one line per item
- [ ] **3.3** Cache summaries to `.codeevolve/summaries/` keyed by file content hash
- [ ] **3.4** Add summary regeneration to init and as a pre-run step

### Phase 4: Multi-File Bundle Format & Focus Rotation

- [ ] **4.1** Design bundle format: editable focus section + read-only summary section with clear delimiters
- [ ] **4.2** Create `bundler.py` module: `create_bundle(focus_file, all_files, summaries) -> str` and `extract_focus(bundle_text) -> str`
- [ ] **4.3** Implement focus file rotation strategy (round-robin or fitness-weighted)
- [ ] **4.4** Integrate bundler into `runner.py` — generate bundle as the "initial program" for OpenEvolve

### Phase 5: Workspace-Level Evaluation

- [ ] **5.1** Update `pipeline.py` to accept multiple source files and track which is the current focus
- [ ] **5.2** Update `cargo.py` to run build/test/clippy at workspace root (remove any `-p` scoping for hard gates)
- [ ] **5.3** Add release build + UPX binary size measurement to `benchmark.py` (`binary_package`, `upx_path`, `upx_args` config)
- [ ] **5.4** Update LOC measurement to cover focus file (or all evolved files)
- [ ] **5.5** Wire `cargo bench` as the default custom benchmark command

### Phase 6: CLI & Runner Updates

- [ ] **6.1** Update `cli.py` init command: remove interactive file picker, show glob matches, generate multi-file evaluator
- [ ] **6.2** Update `cli.py` run command: find all marked files (not just first), pass to runner
- [ ] **6.3** Update `cli.py` reinit command: same multi-file support
- [ ] **6.4** Update `runner.py`: orchestrate bundle creation, focus rotation per iteration, pass bundle to OpenEvolve
- [ ] **6.5** Update evaluator template (`evaluator.py.j2`) to handle multi-file pipeline

### Phase 7: Tests

- [ ] **7.1** Tests for glob-based file discovery (include/exclude patterns, workspace structure)
- [ ] **7.2** Tests for Rust summary generator (various pub item types, edge cases)
- [ ] **7.3** Tests for bundle format (create/extract round-trip, delimiter handling in code)
- [ ] **7.4** Tests for focus rotation logic
- [ ] **7.5** Tests for workspace-level pipeline (multi-file evaluate, binary size with mock UPX)
- [ ] **7.6** Integration test: init + run on a mock workspace project

---

## Research Notes

### OpenEvolve API (single-file constraint)

- `run_evolution(initial_program, evaluator, config)` — `initial_program` is `Union[str, Path, List[str]]` but always becomes a single temp file
- `Program` dataclass has one `code: str` field
- The bundle must be a single string that OpenEvolve passes through unchanged
- The evaluator receives `program_path: str` pointing to a temp file with the bundle content

### axiom2d Project Structure

- 14 crates in workspace at `D:\rust-target\axiom2d`
- 756 total .rs files, 361 generated in `crates/card_game/src/card/art/generated/`
- 395 hand-written files across engine_core, engine_render, engine_ui, etc.
- `release.ps1` builds `card_game_bin` with `--release` then UPX `--best --force`
- Post-UPX binary is what ships; this is the size metric
- No existing benches/ directory (cargo bench is a default for projects that have them)
- Windows Python 3.13 in `.venv/`, cargo/UPX are native Windows binaries

### Key File Paths in CodeEvolution

- Config dataclass: `codeevolve/config.py`
- Init logic: `codeevolve/init_project.py`
- CLI commands: `codeevolve/cli.py`
- Runner (starts LLM, calls OpenEvolve): `codeevolve/runner.py`
- Eval pipeline: `codeevolve/evaluator/pipeline.py`
- Cargo helpers: `codeevolve/evaluator/cargo.py`
- Benchmark helpers: `codeevolve/evaluator/benchmark.py`
- LLM judge: `codeevolve/evaluator/llm_judge.py`
- Evaluator template: `codeevolve/templates/evaluator.py.j2`
- Defaults: `codeevolve/defaults/evolution.yaml`
- Tests: `tests/`

### Benchmark Reference (optimizationTest)

- Located at `C:\Users\siriu\RustroverProjects\optimizationTest`
- Uses Criterion 0.5 with `harness = false`
- Has timeout probe mechanism (5s) for slow implementations
- Small (100-2000 points) and large (10K-100K) benchmark groups
- Pattern to follow for any future axiom2d benchmarks

---

## Open Questions

- **Focus rotation strategy:** Round-robin is simplest, but fitness-weighted (spend more time on files with lower scores) could be more effective. Defer to implementation.
- **Summary staleness:** If a focus file edit changes pub signatures, other files' summaries may become stale within the same run. Acceptable for v1 since summaries refresh between runs.
- **UPX caching:** Could cache the release build between candidates if only the focus file changed and it's not in the card_game_bin dependency tree. Optimization for later.
