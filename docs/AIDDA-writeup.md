# CodeEvolution: Evolutionary Code Optimization for Rust via OpenEvolve

## What we're building

CodeEvolution is a CLI tool that wraps [OpenEvolve](https://github.com/codelion/openevolve) — the most popular open-source reproduction of Google DeepMind's AlphaEvolve — to explore evolutionary code optimization for Rust projects. The goal is a two-command workflow:

```
codeevolve init --path ./my-project
codeevolve run
```

The `init` command scans a Rust crate, identifies evolvable code regions, inserts `EVOLVE-BLOCK` markers, and generates a configuration directory. The `run` command starts an LLM backend, wires it to OpenEvolve, and begins the evolutionary loop. In practice, the system has been developed and tested primarily against a single 14-crate Rust workspace (axiom2d), so there are likely assumptions in the init/evaluator flow that would need adjustment for other projects.

## What we add on top of OpenEvolve

OpenEvolve provides the evolutionary engine: MAP-Elites population database with island-based evolution, prompt sampling with inspiration from top programs, and an async controller. CodeEvolution contributes the domain-specific layers that make this engine useful for real Rust codebases.

### Multi-backend LLM support

Rather than requiring a single OpenAI-compatible API, CodeEvolution supports four provider modes:

- **Local** — manages a llama-server (llama.cpp) subprocess with tuned GPU offloading for consumer hardware (14B model with partial offload on 8GB VRAM). In practice, small local models (7B–14B) produce code that fails to compile or pass tests roughly 95% of the time, making this backend largely non-viable for meaningful evolution. We tried both OpenEvolve's SEARCH/REPLACE diff format and whole-file replacement — neither helped. The models struggle with diff syntax compliance, but even when given the simpler task of rewriting an entire function, they consistently introduce type errors, missing imports, or logic bugs that fail the hard gates. The problem is that models at this scale lack the Rust-specific reasoning needed to make non-trivial changes to real code while preserving correctness
- **Codex** — routes through a lightweight HTTP proxy that translates OpenAI API calls into OpenAI Codex CLI invocations
- **Claude** — same proxy pattern for Anthropic's Claude Code CLI
- **Mixed** — a routing proxy that dispatches requests to both Codex and Claude backends by model name, with configurable weighting (default 70% Claude / 30% Codex) and automatic fallback

The proxy architecture is intentionally thin. Each proxy is ~200 lines of Python that translates a `/v1/chat/completions` request into a CLI subprocess call, letting OpenEvolve treat any backend as a standard API endpoint.

### 3-layer evaluation pipeline

This is the core value-add. AlphaEvolve's published successes rely on problems where fitness is trivially measurable (runtime, mathematical optimality). Evaluating *code quality* is harder. CodeEvolution implements a layered pipeline that combines deterministic signals with LLM judgment:

**Layer 1: Hard gates** (`cargo.py`) — `cargo build` + `cargo clippy` (zero warnings) + `cargo test`. Any failure means fitness zero. This is non-negotiable: correctness dominates all other objectives.

**Layer 2: Performance benchmarks** (`benchmark.py`) — lines of code (less is better), compile time, binary size, and an optional user-defined benchmark command with regex-based score extraction.

**Layer 3: Diff-based LLM judgment** (`llm_judge.py`) — instead of asking an LLM to score code in isolation (which is noisy and expensive), we show it the *diff* of what changed and ask: "Is this an improvement or a regression?" Scores range from -0.99 (clear regression) to +0.99 (clear improvement) across dimensions like readability, Rust idiomaticity, maintainability, and design. The diff-based approach is cheaper (smaller prompts) and relative judgments are easier than absolute ones, but it is not reliable. Manual post-processing review suggests that most changes the judge labels as improvements *are* improvements, but there are clear cases where it labels regressions as improvements — particularly subtle ones like unnecessary abstraction, gratuitous refactoring, or changes that look cleaner in the diff but hurt readability in context. The judge is a useful signal, not a trustworthy one.

Final fitness is a 50/50 blend of performance metrics and LLM judgment.

### Regeneration retries

A naive evolutionary loop wastes iterations when candidates fail hard gates — they score zero and contribute nothing. CodeEvolution introduces *regeneration retries*: when a candidate fails compilation or tests, the system asks the LLM to generate a fresh improvement from the original working code rather than trying to patch the broken candidate. This is distinct from the LLM fixer (which attempts to fix build errors) and the gate retries (which re-attempt the whole generation). The temperature escalates across retry attempts (0.3 → 0.5 → 0.7) to increase diversity when conservative fixes fail.

### Candidate merge

After each evolution slot, the system doesn't just take the winner — it extracts non-conflicting improvements from the top K runners-up (default K=5) and layers them onto the winning candidate using diff-based patch extraction. This means independent gains discovered by different evolutionary branches aren't lost when a single winner is selected.

### Model tier escalation

To balance cost and quality, the evaluator uses a tiered model system. The low tier (provider's default model — e.g., Haiku for Claude, gpt-5.4-mini for Codex) handles the bulk of work: code generation, LLM judging, and early fixer attempts. The mid tier (Sonnet / gpt-5.3-codex) kicks in for the final fixer attempts where a stronger model is more likely to resolve stubborn compilation errors. This avoids spending premium model tokens on routine mutations while reserving them for the hardest recovery scenarios.

### Schedule shuffling

When evolving multiple files, a deterministic schedule means restarts always visit the same high-priority files first — introducing restart bias. CodeEvolution supports weighted random permutation of the file schedule, so each run explores files in a different order while still respecting priority weights derived from tech debt scores and file length.

### EVOLVE-BLOCK isolation

The system carefully separates evolvable code from frozen context. The `init` command identifies function bodies as evolvable regions and wraps them in `EVOLVE-BLOCK-START`/`EVOLVE-BLOCK-END` markers. During evolution, the LLM receives *only* the evolvable section, with the frozen context (struct definitions, imports, test modules) provided as read-only reference. This prevents the LLM from accidentally redefining types or duplicating imports — a common failure mode when LLMs are given entire files to rewrite.

## Architecture

```
codeevolve init
  └─ Scans Cargo.toml + src/**/*.rs
  └─ Inserts EVOLVE-BLOCK markers
  └─ Generates .codeevolve/evolution.yaml + evaluator

codeevolve run
  ├─ Starts LLM backend (llama-server / codex proxy / claude proxy / mixed proxy)
  ├─ Builds OpenEvolve config from evolution.yaml
  ├─ For each file in schedule:
  │   ├─ OpenEvolve evolutionary loop (MAP-Elites + island migration)
  │   │   ├─ Sample parent + inspirations from population
  │   │   ├─ LLM generates candidate mutation
  │   │   ├─ 3-layer evaluation pipeline scores candidate
  │   │   │   ├─ cargo build / clippy / test (hard gates)
  │   │   │   ├─ LoC + compile time + binary size
  │   │   │   └─ Diff-based LLM judgment
  │   │   ├─ If gates fail → regeneration retry (fresh from original)
  │   │   └─ Add scored candidate to population database
  │   └─ Merge non-conflicting improvements from top K candidates
  └─ Stop LLM backend
```

## Current status and honest assessment

The system is functional and being tested against a single 14-crate Rust workspace (axiom2d). The evaluation pipeline, all four LLM backends, candidate merge, schedule shuffling, tier escalation, and regeneration retries are implemented with unit test coverage. The codebase is ~3,000 lines of Python across 20 modules.

**What's working:** The Codex and Claude backends produce compilable, test-passing candidates at a reasonable rate. The hard-gate layer (build + clippy + tests) is reliable and catches genuinely broken candidates. The regeneration retry system recovers useful iterations that would otherwise be wasted. Candidate merge and schedule shuffling work mechanically as designed.

**What's uncertain:** Whether the system actually produces net-positive improvements over enough iterations. Post-processing review of evolved code suggests many changes *are* improvements, but we don't yet have rigorous before/after benchmarks that definitively demonstrate value. The LLM judge mislabels regressions as improvements often enough to erode confidence in the fitness signal. The system's generality beyond the single workspace it was developed against is untested.

**What's not working:** The local llama.cpp backend. Small models (7B–14B) fail to produce compilable, test-passing code roughly 95% of the time — whether using diff-based or whole-file replacement. This was the original motivation for the project (run AlphaEvolve locally on consumer hardware), so its failure is significant — the viable backends all require cloud API access.

## Why this matters (and where it falls short)

AlphaEvolve demonstrated that LLM-guided evolutionary search can discover improvements that neither humans nor LLMs find on their own. But AlphaEvolve is a closed system, and the open-source reproductions (OpenEvolve, ShinkaEvolve) provide the engine without the domain-specific tooling needed to apply it to real projects. CodeEvolution is an attempt to bridge that gap for Rust — handling the LLM infrastructure, evaluation design, and iteration management.

The honest takeaway so far: the infrastructure works, but the evidence that evolutionary search produces reliably better code is still thin. The hard-gate layer is the unambiguous win — it prevents regressions. The LLM judge and fitness blending are more speculative; they *might* be guiding evolution in a useful direction, but the signal-to-noise ratio is low enough that manual review of every accepted change is still necessary. The original vision of running this entirely on local hardware didn't pan out — the models that fit on consumer GPUs can't produce working Rust code reliably enough to participate in evolution, regardless of whether they're asked to emit diffs or rewrite whole files.
