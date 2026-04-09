# Building a local AlphaEvolve on an RTX 4060

**You can build a working evolutionary code optimization system locally today.** OpenEvolve and ShinkaEvolve provide mature open-source frameworks that replicate AlphaEvolve's core architecture, both supporting local models via Ollama. On an RTX 4060's **8GB VRAM** (note: the standard 4060 ships with 8GB, not 4GB), you can run a Qwen2.5-Coder-7B mutator and a 1.5B evaluator simultaneously, achieving 4–8 full generate-evaluate cycles per minute — enough for meaningful evolutionary search. The real engineering challenge isn't the evolutionary loop itself; it's designing fitness functions that capture code quality beyond test-pass rates.

This report covers AlphaEvolve's internals, the open-source ecosystem that has sprung up around it, practical hardware configurations, and a concrete fitness function architecture for a Rust game engine.

---

## How AlphaEvolve actually works inside

AlphaEvolve, published by Google DeepMind in June 2025, is a closed-loop evolutionary coding agent. The core loop is deceptively simple — five lines of pseudocode from the paper's Figure 2 capture it:

```
parent_program, inspirations = database.sample()
prompt = prompt_sampler.build(parent_program, inspirations)
diff = llm.generate(prompt)
child_program = apply_diff(parent_program, diff)
results = evaluator.execute(child_program)
database.add(child_program, results)
```

The system marks evolvable code regions with `EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END` comments, then iteratively mutates them. Unlike its predecessor FunSearch (which evolved single functions of 10–20 lines), AlphaEvolve evolves **entire codebases up to hundreds of lines** and works in any programming language. Ablation studies confirmed that full-file evolution significantly outperforms single-function evolution because it enables coordinated changes across multiple components.

**The multi-LLM ensemble** is central to the design. AlphaEvolve pairs Gemini Flash (high throughput, fast iteration) with Gemini Pro (fewer but higher-quality suggestions). Flash generates the bulk of candidates — maximizing the volume of ideas explored per compute-hour — while Pro occasionally produces breakthrough mutations that advance the search past local optima. Ablations show that removing either model degrades results: Flash alone lacks depth, Pro alone lacks breadth.

**The program database** combines two ideas from evolutionary computation. MAP-Elites maps solutions onto a multi-dimensional feature grid, retaining the best individual in each cell to maintain both quality and diversity. Island-based evolution runs multiple independent populations that periodically exchange top solutions, preventing premature convergence. Together, these mechanisms ensure the system explores diverse solution strategies rather than collapsing onto a single approach.

**Prompt construction** is surprisingly rich. Each prompt includes a system instruction ("Act as an expert software developer"), multiple high-scoring programs sampled from the database as "inspiration," the parent program to mutate with its fitness scores, and instructions to output changes in a SEARCH/REPLACE diff format. AlphaEvolve also co-evolves the prompts themselves through meta-prompting — a separate evolutionary database optimizes the instructions given to the LLM mutators. This delivered measurable gains in the ablation studies, though less dramatic than the evolutionary loop or rich context.

A critical efficiency gain over FunSearch: AlphaEvolve needs only **thousands of LLM samples** where FunSearch required millions. This higher sample efficiency comes from richer prompts, stronger models, and full-file evolution, and it makes the system practical for problems where evaluation is expensive (hours on accelerators rather than seconds on a CPU).

---

## The open-source ecosystem has matured fast

Within months of AlphaEvolve's announcement, several high-quality open-source reproductions emerged. Two stand out as production-ready.

**OpenEvolve** (by Asankhaya Sharma, ~5,900 GitHub stars, Apache 2.0) is the most popular and accessible implementation. It faithfully reproduces AlphaEvolve's architecture: prompt sampler, LLM ensemble, evaluator pool, MAP-Elites database with island-based evolution, and an async controller. It ships as a pip package (`pip install openevolve`), includes Docker images, and supports any OpenAI-compatible API — meaning it works directly with Ollama at `http://localhost:11434/v1`. It successfully replicated AlphaEvolve's circle-packing result at **99.97% accuracy** and includes examples for function minimization, GPU kernel optimization, and prompt evolution.

**ShinkaEvolve** (by Sakana AI, accepted at ICLR 2026) takes a more research-driven approach. Its key innovation is **orders-of-magnitude better sample efficiency** — achieving state-of-the-art circle packing in roughly 150 evaluations versus thousands. It uses adaptive parent sampling, embedding-based novelty rejection to avoid wasting evaluations on duplicate solutions, and a UCB1 bandit algorithm that automatically routes mutations to whichever LLM in the ensemble is performing best. It also supports local models via OpenAI-compatible endpoints.

**CodeEvolve** (by Inter&Co, 65 stars) is a smaller but academically rigorous project that surpassed AlphaEvolve on 4 of 13 mathematical benchmarks using the open-weight Qwen3-Coder-30B model, demonstrating that open models can match closed-source performance. Several other projects round out the ecosystem: OpenAlpha_Evolve (educational/modular), OpenELM (CarperAI's pre-AlphaEvolve ELM framework), and multiple community FunSearch implementations that support local models including DeepSeek-Coder at 1.3B–6.7B parameters.

---

## What actually fits on 8GB VRAM and how fast it runs

First, a hardware clarification: **the RTX 4060 ships with 8GB GDDR6**, not 4GB. No 4GB variant exists on the consumer market. This is meaningfully better news for local evolutionary search.

The sweet spot for code generation on this card is **Qwen2.5-Coder-7B at Q4_K_M quantization**, consuming roughly **5GB VRAM** and generating at **28–35 tokens/second**. This model punches well above its weight class, outperforming CodeStral-22B and DeepSeek-Coder-33B on multiple code benchmarks despite being a fraction of their size. For the evaluator role, Qwen2.5-Coder-1.5B at Q4_K_M uses only ~1.5GB and runs at 50+ tokens/second.

The critical insight is that **both models fit in VRAM simultaneously** (5GB + 1.5GB = 6.5GB out of 8GB available). Ollama supports concurrent model loading, eliminating the 3–10 second swap overhead that would otherwise dominate an evolutionary loop. The recommended architecture:

```
Ollama Server (persistent, 6.5GB / 8GB VRAM)
├── Qwen2.5-Coder-7B Q4_K_M  [Mutator]  ~5.0GB, 30 t/s
└── Qwen2.5-Coder-1.5B Q4_K_M [Evaluator] ~1.5GB, 50 t/s

Evolution Controller (Python, via API)
├── Generate code variant → 7B model (~6-7s per 200 tokens)
├── Evaluate fitness     → 1.5B model (~1s per 50 tokens)
└── Update population database
```

This yields roughly **4–8 complete generate-evaluate cycles per minute**. Over a 3-hour session, that's 720–1,440 evaluations — well within the range where AlphaEvolve-style systems produce meaningful results, especially with ShinkaEvolve's sample-efficient algorithms.

**14B models do not fit.** At Q4_K_M, a 14B model needs 8–10GB, forcing CPU offloading that drops throughput to roughly 8–15 tokens/second — a 4x slowdown that compounds across hundreds of evolutionary iterations. Stick with 7B. If you need stronger reasoning for occasional "Pro-tier" mutations, consider offloading a single 14B evaluation to CPU every N iterations while the 7B handles the bulk.

Key optimization tips: keep context windows short (2K–4K tokens, configured via `num_ctx` in Ollama), enable flash attention (`OLLAMA_FLASH_ATTENTION=1`), and quantize the KV cache (`--cache-type-k q8_0` in llama.cpp) to halve cache VRAM with minimal quality loss. For maximum raw speed, llama.cpp server mode is 10–20% faster than Ollama, though Ollama's model management is more convenient.

---

## Designing fitness functions that measure more than test results

AlphaEvolve's published successes — matrix multiplication speedups, data center scheduling, kernel optimization — share a trait: **fitness is objectively measurable** (runtime, correctness, mathematical optimality). Evaluating code *quality* is fundamentally harder because readability and architectural elegance lack ground truth. The solution is a layered approach combining deterministic signals with calibrated LLM judgment.

**Layer 1: Hard gates (deterministic).** Compilation and test passage are non-negotiable prerequisites. Any candidate that fails `cargo build` or drops below 100% test passage gets fitness zero. This is the lexicographic ordering principle — correctness dominates all other objectives.

**Layer 2: Static analysis signals (deterministic).** Rust's Clippy provides **800+ lints** organized into categories that map directly to fitness dimensions. Correctness lints (deny-by-default) serve as additional hard gates. Complexity lints, performance lints, and style lints each become weighted penalty terms. Cyclomatic complexity per function, nesting depth, and code volume provide complementary automated quality signals. A practical formula:

```
static_score = -(5 × correctness_lints + 3 × suspicious_lints 
                + 2 × complexity_lints + 2 × perf_lints + 1 × style_lints)
```

**Layer 3: Performance benchmarks (deterministic).** Frame time (mean and p99), compile time, binary size, peak memory, and startup time — all measurable via headless benchmark scenes and `cargo build --timings`.

**Layer 4: LLM quality judgment (stochastic, calibrated).** This is where the research gets interesting. The CodeJudge paper demonstrated that even Llama-3-8B-Instruct outperforms GPT-3.5-based code evaluation methods, suggesting 7B models are viable judges. However, individual LLM evaluations are noisy — studies show up to **15% accuracy variation** across runs even at temperature zero. The mitigation is aggregation: run the evaluation 3–5 times, take the median, and use coarse 1–5 Likert scales rather than fine-grained scores. Structured JSON output with chain-of-thought reasoning ("explain before scoring") produces the most consistent results.

The Prometheus line of research shows that small models fine-tuned on evaluation tasks can achieve **0.897 Pearson correlation** with human judges, rivaling GPT-4. An even more radical approach from the INSPECTOR paper (January 2026) uses hidden-state probing on 1.7B-parameter models — bypassing text generation entirely and reading evaluative signals directly from internal representations.

**Combining layers:** Use the gated hierarchical approach. Only candidates passing gates 1–2 get benchmarked (layer 3). Only candidates in the top quartile of automated metrics get the expensive LLM evaluation (layer 4). This keeps LLM calls to roughly 25% of the population per generation — manageable on local hardware.

---

## A concrete fitness architecture for a Bevy game engine

For a Rust 2D game engine built on bevy_ecs and wgpu, fitness functions must capture ECS-specific architectural qualities alongside standard metrics. Here's a concrete design:

**Tier 1 — hard gates** require compilation, all tests passing, and zero Clippy correctness lints. These are boolean: fail any, get zero fitness.

**Tier 2 — automated performance (weight: 0.35)** includes p99 frame time targeting sub-16.67ms (60fps), incremental compile time targeting under 3 seconds, binary size, and peak memory. These come from headless benchmark scenes that exercise the engine's core systems.

**Tier 3 — automated code quality (weight: 0.35)** measures total Clippy warnings, average cyclomatic complexity per function (target <10), documentation coverage via `#![warn(missing_docs)]`, and three ECS-specific metrics:

- **Average system parameter count** (target <5 Query/Res/ResMut params per system — more suggests the system is doing too much)
- **Average component field count** (target <5 fields per Component struct — components should be small, focused data containers)
- **System ordering dependency count** (fewer `.before()`/`.after()` constraints means more parallelizable, better-decomposed architecture)

**Tier 4 — LLM architectural judgment (weight: 0.30)** evaluates readability, ECS design soundness, Rust idiomaticity, and maintainability on a 1–5 scale. The prompt should reference Bevy conventions explicitly: "Are components focused? Are systems small? Does the code follow Bevy's plugin organization patterns?"

This composite fitness function can be implemented in a single `evaluate()` function that OpenEvolve or ShinkaEvolve calls automatically. The hard gates run in milliseconds (compilation/tests), automated metrics in seconds (Clippy + benchmarks), and LLM evaluation in 5–10 seconds on local hardware. A full evaluation costs roughly 15–30 seconds per candidate — workable at 4–8 candidates per minute when generation and evaluation overlap.

---

## Conclusion

Building a local AlphaEvolve is no longer a theoretical exercise. OpenEvolve and ShinkaEvolve provide the evolutionary scaffolding; Qwen2.5-Coder-7B provides surprisingly capable mutation on consumer hardware; and the layered fitness function approach — deterministic gates, static analysis, benchmarks, then LLM judgment — makes code quality measurable enough for evolutionary selection.

The key architectural decisions for a constrained setup: use ShinkaEvolve's sample-efficient algorithms (150 evaluations vs. thousands), keep both mutator and evaluator models loaded simultaneously in VRAM, reserve LLM-based quality evaluation for top-quartile candidates only, and design ECS-specific metrics (system parameter counts, component granularity, ordering dependencies) that capture Bevy architectural quality without relying on the LLM judge for everything. The bottleneck isn't compute — it's designing fitness functions that genuinely correlate with the code qualities you care about.