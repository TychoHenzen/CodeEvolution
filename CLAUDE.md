# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CodeEvolution is a CLI tool that wraps OpenEvolve to provide batteries-included evolutionary code optimization for Rust projects. Users run `codeevolve init` in a Rust project, then `codeevolve run` to evolve their code using one of three LLM backends: local llama-server (llama.cpp), OpenAI Codex CLI, or Claude Code CLI.

See `Reference.md` for the design rationale and `docs/superpowers/specs/2026-04-09-codeevolution-design.md` for the full spec.

## Tech Stack

- Python 3.13, Click (CLI), OpenEvolve (evolutionary engine), Jinja2 (templates), PyYAML, openai (LLM client)
- Four LLM backends (`provider` setting in config): `local` (llama-server/llama.cpp subprocess), `codex` (OpenAI Codex CLI via `codex_proxy.py`), `claude` (Claude Code CLI via `claude_proxy.py`), `mixed` (codex + claude behind a routing proxy via `mixed_proxy.py`)
- Codex/Claude proxies are lightweight HTTP servers that translate OpenAI API calls to CLI invocations, sharing `base_proxy.py` for handler and lifecycle code
- Mixed proxy routes requests by model name to the correct backend, with automatic fallback if one backend returns empty/error
- Default provider is `codex` (gpt-5.4-mini)

## Commands

```bash
# Install in dev mode (Windows Python venv in WSL)
pip install -e ".[dev]"

# Run all tests (MUST use venv Python, not system python3)
.venv/Scripts/python.exe -m pytest tests/ -v

# Run a single test
.venv/Scripts/python.exe -m pytest tests/test_cargo.py::test_parse_clippy_json -v

# Run the CLI
.venv/Scripts/python.exe -c "from codeevolve.cli import main; main()"
codeevolve init --path /path/to/rust/project
codeevolve run --config .codeevolve/evolution.yaml
```

## Architecture

The system is a thin wrapper over OpenEvolve with two CLI commands:

1. **`codeevolve init`** (`init_project.py`) — scans a Rust crate, inserts EVOLVE-BLOCK markers, generates `.codeevolve/` with config YAML + evaluator.
2. **`codeevolve reinit`** — re-generates evaluator from existing config + syncs config with latest defaults (adds new sections like `tiers`).
3. **`codeevolve run`** (`runner.py` + `llama_server.py`/`codex_proxy.py`/`claude_proxy.py`/`mixed_proxy.py`) — starts the configured LLM backend, builds OpenEvolve config, calls `run_evolution()`, stops backend on exit.

The core value-add is the **3-layer evaluation pipeline** (`evaluator/pipeline.py`):
- Layer 1: `cargo.py` — hard gates (cargo build + cargo clippy zero-warnings + cargo test), with LLM fixer retry loop
- Layer 2: `benchmark.py` — LoC, compile time, binary size, optional user benchmark
- Layer 3: `llm_judge.py` — LLM quality judgment via llama-server (top-quartile only, 3-run median)

Failed-gate candidates trigger **regeneration retries** (`max_gate_retries`): the LLM generates a fresh improvement from the original working code (via `llm_fixer.attempt_regenerate`) rather than wasting the iteration with score=0. Fitness weights are 50/50 performance + LLM judgment (no separate static analysis score).

Config is a single dataclass hierarchy (`config.py`) loaded from YAML, with defaults in `codeevolve/defaults/evolution.yaml`.

**Model tier escalation**: The evaluator uses three quality tiers (`low`/`mid`/`high`) to balance cost vs. quality. Low (provider default) handles generation + first fixer attempt; mid (sonnet/gpt-5.3-codex) handles LLM judging + middle fixer attempts; high (opus/gpt-5.4) handles the last fixer attempt. Tier config lives in the `tiers` YAML section and `ModelTiers` dataclass.

## Key Constraints

- Rust-only in v1 (no multi-language)
- Single-crate only (workspace members must be targeted individually via --path)
- Local provider uses a single 14B model with partial GPU offload (~30/48 layers) to fit in 8GB VRAM
- The `.venv/` uses Windows Python 3.13 — always run tests with `.venv/Scripts/python.exe -m pytest`, not system `python3`
