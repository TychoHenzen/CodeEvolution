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

### LLM fixer

When a candidate fails a hard gate (build, clippy, or tests), the system doesn't immediately discard it — it asks the LLM to fix the broken code. The fixer gets the broken candidate, the error output, and the source code of failing tests and frozen context (struct definitions, imports outside the evolvable region), so it understands what it can and can't change. Temperature escalates across retry attempts (0.3 → 0.5 → 0.7) to explore more diverse fixes when conservative ones fail. The system tracks previous failed fix attempts and includes them in the prompt ("these didn't work, try something different"), and detects when the fixer produces repeated output and gives up. Model tier escalation applies here: the first N-2 fix attempts use the cheap low-tier model, and the last 2 use the stronger mid-tier model.

### Regeneration retries

If a candidate fails all gates even after fixer attempts, the system has a second recovery mechanism: instead of trying to patch the broken candidate further, it goes back to the *original working code* and asks the LLM to generate an entirely new improvement, using the previous failure as guidance on what to avoid. This is a fresh start rather than an incremental fix. Regeneration retries are configured separately (`max_gate_retries`) and also use tier escalation.

### Candidate merge

After each evolution slot, the system doesn't just take the winner — it extracts non-conflicting improvements from the top K runners-up (default K=5) and layers them onto the winning candidate using diff-based patch extraction. This means independent gains discovered by different evolutionary branches aren't lost when a single winner is selected.

### Model tier escalation

To balance cost and quality, the evaluator uses a tiered model system. The low tier (provider's default model — e.g., Haiku for Claude, gpt-5.4-mini for Codex) handles the bulk of work: code generation, LLM judging, and early fixer attempts. The mid tier (Sonnet / gpt-5.3-codex) kicks in for the final fixer attempts where a stronger model is more likely to resolve stubborn compilation errors. This avoids spending premium model tokens on routine mutations while reserving them for the hardest recovery scenarios.

### Iteration scheduling

When evolving a workspace with many files, the system needs to decide how many evolution iterations to spend on each file and in what order. The scheduler allocates iterations proportionally to *tech debt scores* — per-file quality scores from a `TECH_DEBT_LEDGER.md` file, if one exists in the project. This ledger is produced externally (by a separate static analysis tool) and contains structural/semantic quality scores for each source file. Files with higher debt get more iterations. File length provides a secondary bias (longer files get a gentle boost) regardless of whether a ledger exists. When no ledger is present, all files start with equal base scores but length bias still differentiates them, so longer files get somewhat more iterations.

A deterministic schedule means restarts always visit the same high-priority files first — introducing restart bias. CodeEvolution supports weighted random permutation of the schedule, so each run explores files in a different order while still respecting priority weights.

### EVOLVE-BLOCK isolation and context bundling

The system carefully separates evolvable code from frozen context. The `init` command identifies function bodies as evolvable regions and wraps them in `EVOLVE-BLOCK-START`/`EVOLVE-BLOCK-END` markers. OpenEvolve's model is single-file — it sees one "program" string per iteration. We bridge this to multi-file workspaces using a bundle format that packs two things into that string:

1. **FOCUS section** — the full EVOLVE-BLOCK content from the file being evolved. This is the only part the LLM should edit.
2. **CONTEXT section** — read-only API summaries of other files in the workspace. A regex-based summarizer (`summary.py`) extracts pub fn signatures, struct/enum definitions, impl block headers, trait definitions, and type aliases — enough for the LLM to understand what types and functions exist without seeing full implementations. For workspaces, the context is scoped to the focus file's crate and its direct dependencies (one hop in the crate dependency graph) to keep prompt size manageable.

This prevents the LLM from accidentally redefining types or duplicating imports (a common failure mode when LLMs are given entire files to rewrite), while giving it enough context to make changes that are compatible with the surrounding code.

## Architecture

```
codeevolve init
  └─ Scans Cargo.toml + src/**/*.rs
  └─ Inserts EVOLVE-BLOCK markers
  └─ Generates .codeevolve/evolution.yaml + evaluator

codeevolve run
  ├─ Starts LLM backend (llama-server / codex proxy / claude proxy / mixed proxy)
  ├─ Builds OpenEvolve config from evolution.yaml
  ├─ Summarize all .rs files (pub API signatures only)
  ├─ For each file in schedule:
  │   ├─ Bundle focus file (full EVOLVE-BLOCK) + context summaries (read-only)
  │   ├─ OpenEvolve evolutionary loop (MAP-Elites + island migration)
  │   │   ├─ Sample parent + inspirations from population
  │   │   ├─ LLM generates candidate mutation
  │   │   ├─ 3-layer evaluation pipeline scores candidate
  │   │   │   ├─ cargo build / clippy / test (hard gates)
  │   │   │   ├─ If gate fails → LLM fixer (escalating temp + tier)
  │   │   │   ├─ LoC + compile time + binary size
  │   │   │   └─ Diff-based LLM judgment
  │   │   ├─ If gates still fail → regeneration retry (fresh from original)
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

The honest takeaway so far: the infrastructure works, but the evidence that evolutionary search produces reliably better code is still thin. The hard-gate layer is the unambiguous win — it prevents regressions to the extent that test coverage finds the problem. The LLM judge and fitness blending are more speculative; they *might* be guiding evolution in a useful direction, but the signal-to-noise ratio is low enough that manual review of every accepted change is still necessary, especially given the short time testing and working with this. The original vision of running this entirely on local hardware didn't pan out — the models that fit on consumer GPUs can't produce working Rust code reliably enough to participate in evolution, regardless of whether they're asked to emit diffs or rewrite whole files.
