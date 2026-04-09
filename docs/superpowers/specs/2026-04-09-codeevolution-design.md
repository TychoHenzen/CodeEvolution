# CodeEvolution Design Spec

**Date:** 2026-04-09
**Status:** Approved

## Overview

CodeEvolution is a CLI tool that makes evolutionary code optimization accessible for Rust projects. It wraps OpenEvolve (the open-source AlphaEvolve reproduction) and provides a batteries-included experience where users go from zero to evolving their code in two commands. All inference runs locally via Ollama — no API keys, no cloud dependencies.

## Goals

- **Two-command UX**: `codeevolve init` then `codeevolve run` with no knowledge of OpenEvolve required.
- **Batteries-included for Rust**: the tool automatically runs `cargo build`, `cargo test`, `cargo clippy`, measures compile time and binary size, and runs LLM-based quality judgment — all preconfigured.
- **Beginner-friendly output**: every terminal line is human-readable. Failures are expected, not errors. A summary tells the story of what improved.
- **Fully local**: Ollama-only. Qwen2.5-Coder-7B as the mutator, Qwen2.5-Coder-1.5B as the evaluator. Both fit in 8GB VRAM simultaneously.

## Tech Stack

- **Python 3.13** with venv
- **Click** for CLI
- **OpenEvolve** (`pip install openevolve`) as the evolutionary engine
- **Jinja2** for evaluator template generation
- **Ollama** for all LLM inference (OpenAI-compatible API at `http://localhost:11434/v1`)

## Project Structure

```
codeevolve/
├── __init__.py
├── cli.py              # Click-based CLI entry point
├── config.py           # Config dataclass + YAML loading
├── init_project.py     # codeevolve init — scans Rust project, generates files
├── runner.py           # codeevolve run — calls OpenEvolve's run_evolution()
├── evaluator/
│   ├── __init__.py
│   ├── pipeline.py     # Gated 4-layer evaluation orchestrator
│   ├── cargo.py        # Layer 1+3: cargo build, test, clippy, timings, binary size
│   ├── benchmark.py    # Layer 3: optional user benchmark command
│   └── llm_judge.py    # Layer 4: Ollama-based quality judgment
├── templates/
│   └── evaluator.py.j2 # Template for the generated evaluator.py OpenEvolve calls
└── defaults/
    └── evolution.yaml   # Default OpenEvolve config tuned for Rust + Ollama
```

## CLI Commands

### `codeevolve init [--path ./my-rust-project]`

Walks the user through setting up evolution for a Rust project:

1. **Find the project** — looks for `Cargo.toml` in the current directory (or `--path`). Fails with a clear message if not found. V1 supports single-crate projects only — if a workspace `Cargo.toml` is detected (contains `[workspace]`), the user must point `--path` at a specific member crate.
2. **Scan source files** — walks `src/` and lists all `.rs` files, printed numbered.
3. **Interactive selection** — asks the user:
   - "Which files should be evolved?" (multi-select, defaults to all)
   - "Want to provide a custom benchmark command?" (e.g., `cargo bench`, a script path). If no, skips Layer 3 runtime benchmarks.
4. **Insert EVOLVE-BLOCK markers** — for each selected file, wraps content in `// EVOLVE-BLOCK-START` / `// EVOLVE-BLOCK-END` comments.
5. **Generate `.codeevolve/` directory** containing:
   - `evolution.yaml` — pre-configured for Ollama with the 7B mutator + 1.5B evaluator.
   - `evaluator.py` — generated from the Jinja template, hardcoded to this project's paths and the user's benchmark choice.
   - `README.md` — ELI5 explanation: what these files are, how to tweak them, how to run.
6. **Print next steps** — tells the user exactly what to do next.

### `codeevolve run [--config .codeevolve/evolution.yaml]`

Runs the evolutionary loop:

1. Loads config from YAML.
2. Validates Ollama is reachable and both models are available.
3. Calls `openevolve.api.run_evolution()` with the config and evaluator.
4. Streams progress to the terminal in a human-readable format.
5. On completion (or Ctrl+C), writes the best candidate to `.codeevolve/output/best/` and a summary.

## Evaluation Pipeline

The evaluator is a 4-layer gated pipeline. Each layer is a gate — failing an earlier layer skips all later layers.

```
Candidate code
    │
    ▼
┌─────────────────────────┐
│ Layer 1: Hard Gates      │  cargo build + cargo test
│ Pass/Fail                │  Fail = score 0, stop immediately
└─────────┬───────────────┘
          │ pass
          ▼
┌─────────────────────────┐
│ Layer 2: Static Analysis │  cargo clippy --message-format=json
│ Weighted penalty score   │  5*correctness + 3*suspicious
└─────────┬───────────────┘  + 2*complexity + 2*perf + 1*style
          │
          ▼
┌─────────────────────────┐
│ Layer 3: Performance     │  Compile time (cargo build --timings)
│ Measured metrics         │  Binary size (ls target/)
└─────────┬───────────────┘  Optional: user benchmark command
          │
          ▼
┌─────────────────────────┐
│ Layer 4: LLM Judgment    │  Only for top-quartile candidates
│ Ollama 1.5B evaluator   │  3 runs, median, 1-5 Likert scale
└─────────┬───────────────┘  Dimensions: readability, idiomaticity,
          │                  maintainability, design
          ▼
    Combined score
```

### Scoring Formula

```
combined_score =
    0.00                          if hard gates fail
    0.35 * normalize(static)
  + 0.35 * normalize(perf)
  + 0.30 * normalize(llm)        llm = 0 if not top-quartile
```

All metrics are normalized to 0-1 against the best and worst seen so far in the current run.

### Layer Details

**Layer 1 — Hard Gates:** Runs `cargo build` and `cargo test`. Either passes or the candidate gets score 0 and is discarded. No partial credit.

**Layer 2 — Clippy Static Analysis:** Runs `cargo clippy --message-format=json` and categorizes each warning by lint group. Weighted sum formula:

```
static_score = -(5*correctness + 3*suspicious + 2*complexity + 2*perf + 1*style)
```

**Layer 3 — Performance Metrics:**
- Compile time: measured via `cargo build --timings`, parsed from output.
- Binary size: size of the compiled binary in `target/`.
- User benchmark (optional): runs the user-provided command, extracts a numeric score via a user-provided regex (one capture group, e.g., `"time: ([\d.]+)ms"`). If no regex, uses exit code (0 = pass, nonzero = penalty).

**Layer 4 — LLM Quality Judgment:**
- Only invoked for candidates scoring in the top 25% of Layers 2+3 combined (rolling distribution).
- Uses the Ollama 1.5B evaluator model.
- Prompt asks the model to evaluate on 4 dimensions (readability, Rust idiomaticity, maintainability, design), each scored 1-5, with chain-of-thought reasoning before scoring.
- Runs 3 times, takes the median per dimension.
- Final LLM score = mean of the 4 dimension medians, normalized to 0-1.

## Configuration

Single YAML file at `.codeevolve/evolution.yaml`:

```yaml
# --- Ollama Models ---
ollama:
  api_base: "http://localhost:11434/v1"
  mutator_model: "qwen2.5-coder:7b-instruct-q4_K_M"
  evaluator_model: "qwen2.5-coder:1.5b-instruct-q4_K_M"

# --- Evolution Settings (passed to OpenEvolve) ---
evolution:
  max_iterations: 500
  population_size: 100
  num_islands: 3
  migration_interval: 20
  context_window: 4096
  diff_based_evolution: true

# --- Rust Project ---
rust:
  cargo_path: "cargo"
  target_dir: null              # defaults to project's target/
  test_args: []                 # extra args for cargo test
  clippy_args: []               # extra args for cargo clippy

# --- Fitness Weights ---
fitness:
  static_analysis_weight: 0.35
  performance_weight: 0.35
  llm_judgment_weight: 0.30
  clippy_weights:
    correctness: 5
    suspicious: 3
    complexity: 2
    perf: 2
    style: 1

# --- Benchmarks ---
benchmarks:
  measure_compile_time: true
  measure_binary_size: true
  custom_command: null
  custom_command_score_regex: null

# --- LLM Judgment ---
llm_judgment:
  enabled: true
  top_quartile_only: true
  num_runs: 3
  dimensions:
    - readability
    - rust_idiomaticity
    - maintainability
    - design
```

All fields have sensible defaults. Works out of the box with zero edits if Ollama is running with the expected models pulled.

## CLI Output

Terminal output is designed for users who know nothing about evolutionary algorithms:

```
$ codeevolve run

  Loading config from .codeevolve/evolution.yaml
  Found Rust project: my-project (src/lib.rs, src/utils.rs)
  Connected to Ollama (qwen2.5-coder:7b, qwen2.5-coder:1.5b)
  Starting evolution (500 iterations, population 100)

-- Iteration 12/500 ------------------------------------
  Mutating src/lib.rs ... generated 47-line diff
  |- Build:    pass, compiled (1.2s)
  |- Tests:    pass, 23/23 passed
  |- Clippy:   3 warnings (was 7) - improved
  |- Size:     1.4 MB (was 1.5 MB) - improved
  |- Bench:    skipped (not top quartile)
  |- LLM:      skipped (not top quartile)
  '- Score:    0.72 (best so far: 0.81)

-- Iteration 13/500 ------------------------------------
  Mutating src/utils.rs ... generated 12-line diff
  |- Build:    FAILED (error[E0308]: mismatched types)
  '- Score:    0.00 (discarded)

-- Summary ---------------------------------------------
  Best score:      0.91 (iteration 347)
  Improvements:    Clippy 14->2 warnings, binary 1.5->1.3 MB
  Best candidate:  .codeevolve/output/best/
  All candidates:  .codeevolve/output/history/
  Run time:        2h 14m (482 evaluations)
```

**Principles:** failures are normal ("discarded", not "ERROR"), directional arrows show improvement, summary tells the story.

## Output Directory

```
.codeevolve/output/
├── best/                  # Best-scoring candidate's evolved source files
├── history/
│   ├── iter_012/
│   │   ├── src/           # The mutated source files
│   │   ├── diff.patch     # What changed from parent
│   │   └── scores.json    # All fitness scores
│   └── ...
└── run_log.json           # Full evolution history for analysis
```

## Scope Boundaries (NOT in v1)

- No multi-language support (Rust only)
- No external API support (Ollama only)
- No GUI or web dashboard
- No distributed/multi-machine evolution
- No meta-prompting or prompt co-evolution
- No ShinkaEvolve-style novelty rejection
