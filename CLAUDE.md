# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CodeEvolution is a local evolutionary code optimization system inspired by Google DeepMind's AlphaEvolve. It uses an LLM-driven evolutionary loop to iteratively mutate and improve code, with a multi-layered fitness evaluation pipeline. The target hardware is an RTX 4060 (8GB VRAM) running models via Ollama.

The system evolves code by: sampling a parent program from a population database, building a prompt with inspirations, generating a diff via LLM, applying it, evaluating the result, and storing it back. See `Reference.md` for the full design rationale.

## Tech Stack

- **Python 3.13** with venv at `.venv/`
- **Ollama** for local LLM inference (OpenAI-compatible API at `http://localhost:11434/v1`)
- **Target models**: Qwen2.5-Coder-7B Q4_K_M (mutator, ~5GB) + Qwen2.5-Coder-1.5B Q4_K_M (evaluator, ~1.5GB)
- **Frameworks**: OpenEvolve (`pip install openevolve`) or ShinkaEvolve for the evolutionary loop

## Architecture

The system has five core components:

1. **Program Database** — MAP-Elites grid + island-based evolution. Maintains population diversity by mapping solutions onto feature dimensions and running independent populations with periodic migration.

2. **Prompt Sampler** — Constructs rich prompts: system instruction, high-scoring "inspiration" programs from the database, the parent program with fitness scores, and SEARCH/REPLACE diff format instructions.

3. **LLM Ensemble** — Two-tier model setup. The 7B mutator generates the bulk of candidates (breadth). Optionally, a larger model provides occasional "Pro-tier" mutations (depth). Both run concurrently in Ollama.

4. **Evaluator Pipeline** — Four-layer gated evaluation:
   - **Layer 1**: Hard gates — compilation + tests (fail = fitness zero)
   - **Layer 2**: Static analysis — Clippy lints weighted by category
   - **Layer 3**: Performance benchmarks — frame time, compile time, binary size, memory
   - **Layer 4**: LLM quality judgment — 3-5x median aggregation on 1-5 Likert scales (only for top-quartile candidates)

5. **Evolution Controller** — Async orchestrator running the generate-evaluate loop. Target: 4-8 cycles/minute.

## Key Design Constraints

- Both models must fit in VRAM simultaneously (6.5GB of 8GB). No 14B models.
- Context windows kept short (2K-4K tokens) via Ollama `num_ctx`.
- LLM evaluation reserved for top-quartile candidates only to manage compute budget.
- ShinkaEvolve's sample-efficient algorithms preferred (~150 evaluations vs thousands).

## Commands

```bash
# Activate venv (Windows paths, WSL environment)
source .venv/bin/activate  # or: .venv/Scripts/activate

# Install dependencies (once frameworks are added)
pip install -r requirements.txt

# Run Ollama with flash attention
OLLAMA_FLASH_ATTENTION=1 ollama serve

# Pull required models
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
ollama pull qwen2.5-coder:1.5b-instruct-q4_K_M
```
