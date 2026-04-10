# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CodeEvolution is a CLI tool that wraps OpenEvolve to provide batteries-included evolutionary code optimization for Rust projects. Users run `codeevolve init` in a Rust project, then `codeevolve run` to evolve their code using a local LLM via llama-server (llama.cpp).

See `Reference.md` for the design rationale and `docs/superpowers/specs/2026-04-09-codeevolution-design.md` for the full spec.

## Tech Stack

- Python 3.13, Click (CLI), OpenEvolve (evolutionary engine), Jinja2 (templates), PyYAML, openai (llama-server client)
- llama-server (llama.cpp) for local LLM inference, started internally as a subprocess
- Target model: Qwen2.5-Coder-14B-Instruct Q4_K_M via llama-server with partial GPU offload

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
2. **`codeevolve run`** (`runner.py` + `llama_server.py`) — starts llama-server, builds OpenEvolve config, calls `run_evolution()`, stops server on exit.

The core value-add is the **4-layer evaluation pipeline** (`evaluator/pipeline.py`):
- Layer 1: `cargo.py` — hard gates (cargo build + cargo test)
- Layer 2: `cargo.py` — Clippy static analysis with weighted lint scoring
- Layer 3: `benchmark.py` — compile time, binary size, optional user benchmark
- Layer 4: `llm_judge.py` — LLM quality judgment via llama-server (top-quartile only, 3-run median)

Config is a single dataclass hierarchy (`config.py`) loaded from YAML, with defaults in `codeevolve/defaults/evolution.yaml`.

## Key Constraints

- Rust-only in v1 (no multi-language)
- llama-server only (no external API support)
- Single-crate only (workspace members must be targeted individually via --path)
- Single 14B model with partial GPU offload (~30/48 layers) to fit in 8GB VRAM
- The `.venv/` uses Windows Python 3.13 — always run tests with `.venv/Scripts/python.exe -m pytest`, not system `python3`
