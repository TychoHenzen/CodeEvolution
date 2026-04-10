from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

_DEFAULTS_DIR = Path(__file__).parent / "defaults"
_PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class LlamaServerConfig:
    server_path: str = "llama-server"
    model_path: str = "qwen2.5-coder-14b-instruct-q4_k_m.gguf"
    port: int = 8080
    gpu_layers: int = 30
    context_size: int = 8192
    threads: int = 8
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    flash_attn: bool = True

    @property
    def api_base(self) -> str:
        return f"http://localhost:{self.port}/v1"

    @property
    def model_name(self) -> str:
        return Path(self.model_path).stem


@dataclass
class EvolutionConfig:
    max_iterations: int = 500
    population_size: int = 10
    num_islands: int = 3
    migration_interval: int = 20
    diff_based_evolution: bool = False
    max_fix_attempts: int = 4  # LLM retry attempts on build/test failure


@dataclass
class RustConfig:
    cargo_path: str = "cargo"
    target_dir: Optional[str] = None
    test_args: list[str] = field(default_factory=list)
    clippy_args: list[str] = field(default_factory=list)


@dataclass
class ClippyWeights:
    correctness: int = 5
    suspicious: int = 3
    complexity: int = 2
    perf: int = 2
    style: int = 1


@dataclass
class FitnessConfig:
    static_analysis_weight: float = 0.35
    performance_weight: float = 0.35
    llm_judgment_weight: float = 0.30
    clippy_weights: ClippyWeights = field(default_factory=ClippyWeights)


@dataclass
class BenchmarksConfig:
    measure_compile_time: bool = True
    measure_binary_size: bool = True
    custom_command: Optional[str] = "cargo bench"
    custom_command_score_regex: Optional[str] = None
    binary_package: Optional[str] = None
    upx_path: Optional[str] = None
    upx_args: list[str] = field(default_factory=lambda: ["--best", "--force"])


@dataclass
class LlmJudgmentConfig:
    enabled: bool = True
    top_quartile_only: bool = False
    num_runs: int = 1
    dimensions: list[str] = field(
        default_factory=lambda: ["readability", "rust_idiomaticity", "maintainability", "design"]
    )


@dataclass
class CodexConfig:
    cli_path: str = "codex"
    model: str = "gpt-5.4-mini"
    proxy_port: int = 8081
    timeout: int = 300


@dataclass
class ClaudeConfig:
    cli_path: str = "claude"
    model: str = "haiku"
    effort: str = "low"
    proxy_port: int = 8082
    timeout: int = 300


@dataclass
class CodeEvolveConfig:
    provider: str = "local"  # "local" (llama-server), "codex", or "claude"
    llama_server: LlamaServerConfig = field(default_factory=LlamaServerConfig)
    codex: CodexConfig = field(default_factory=CodexConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    rust: RustConfig = field(default_factory=RustConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    benchmarks: BenchmarksConfig = field(default_factory=BenchmarksConfig)
    llm_judgment: LlmJudgmentConfig = field(default_factory=LlmJudgmentConfig)
    include_globs: list[str] = field(default_factory=lambda: ["src/**/*.rs"])
    exclude_globs: list[str] = field(default_factory=list)

    @property
    def api_base(self) -> str:
        if self.provider == "codex":
            return f"http://localhost:{self.codex.proxy_port}/v1"
        if self.provider == "claude":
            return f"http://localhost:{self.claude.proxy_port}/v1"
        return self.llama_server.api_base

    @property
    def model_name(self) -> str:
        if self.provider == "codex":
            return self.codex.model
        if self.provider == "claude":
            return self.claude.model
        return self.llama_server.model_name

    def to_openevolve_dict(self, frozen_context: str = "") -> dict:
        """Convert to a dict compatible with OpenEvolve's Config.from_dict().

        Parameters
        ----------
        frozen_context:
            Code that exists outside the EVOLVE-BLOCK (struct definitions,
            imports, test modules).  Included in the system message so the
            LLM knows what exists but cannot modify it.
        """
        diff_mode = self.evolution.diff_based_evolution

        frozen_block = ""
        if frozen_context:
            frozen_block = (
                "\n\nThe code you receive is ONLY the evolvable section. "
                "The following code exists OUTSIDE your control — do NOT "
                "redefine, duplicate, or re-import anything shown here:\n\n"
                f"```rust\n{frozen_context}\n```\n\n"
                "Do NOT add struct/enum definitions, type aliases, use "
                "statements, or #[cfg(test)] modules that duplicate the above."
            )

        if diff_mode:
            system_msg = (
                "You are an expert Rust developer tasked with iteratively "
                "improving a Rust codebase. Your goal is to maximize the "
                "FITNESS SCORE while exploring diverse solutions. The code "
                "MUST compile with cargo build and pass all tests. Pay close "
                "attention to Rust's ownership, borrowing, and lifetime rules. "
                "Make small, targeted changes — do not rewrite entire files.\n\n"
                "CRITICAL FORMAT RULE: You MUST output changes using EXACTLY "
                "the <<<<<<< SEARCH / ======= / >>>>>>> REPLACE markers. "
                "Do NOT use markdown code fences (```), do NOT use #### SEARCH "
                "or #### REPLACE headers. Only use the exact markers shown in "
                "the examples.\n\n"
                "IMPORTANT: You MUST make at least one meaningful change. "
                "Do NOT output the original code unchanged. If the SEARCH and "
                "REPLACE sections are identical, your submission is worthless. "
                "Always change something — try a different algorithm, data "
                "structure, API usage, or code pattern. do NOT add tests."
                + frozen_block
            )
        else:
            system_msg = (
                "You are an expert Rust developer tasked with iteratively "
                "improving a Rust codebase. Your goal is to maximize the "
                "FITNESS SCORE while exploring diverse solutions. The code "
                "MUST compile with cargo build and pass all tests. Pay close "
                "attention to Rust's ownership, borrowing, and lifetime rules.\n\n"
                "You will receive ONLY the evolvable section of code. Output "
                "an improved version inside a single ```rust code block. "
                "Do NOT add struct definitions, imports, or test modules — "
                "those exist outside your control.\n\n"
                "IMPORTANT: You MUST make at least one meaningful change. "
                "Do NOT output the original code unchanged. "
                "Always improve something — try a different algorithm, data "
                "structure, API usage, or code pattern. do NOT add tests."
                + frozen_block
            )

        return {
            "max_iterations": self.evolution.max_iterations,
            "diff_based_evolution": diff_mode,
            "max_code_length": 200000,
            "file_suffix": ".rs",
            # Null seed: let each LLM call use a fresh random state.
            # A fixed seed (OpenEvolve defaults to 42) makes the model
            # produce identical outputs across iterations.
            "random_seed": None,
            "llm": {
                "api_base": self.api_base,
                "api_key": "no-key",
                "models": [
                    {"name": self.model_name, "weight": 1.0},
                ],
                "temperature": 0.7,
                "max_tokens": 16384,
                "timeout": 300,
            },
            "prompt": {
                "system_message": system_msg,
                "template_dir": str(_PROMPTS_DIR),
                "num_top_programs": 3,
                "num_diverse_programs": 2,
                "use_template_stochasticity": True,
            },
            "database": {
                "population_size": self.evolution.population_size,
                "num_islands": self.evolution.num_islands,
                "migration_interval": self.evolution.migration_interval,
                "feature_dimensions": ["complexity", "diversity"],
                "log_prompts": True,
            },
            "evaluator": {
                "timeout": 900,
                "parallel_evaluations": 1,
            },
            "early_stopping_patience": 60,
            "convergence_threshold": 0.001,
            "evolution_trace": {
                "enabled": True,
                "format": "jsonl",
                "include_code": False,
                "include_prompts": True,
            },
        }


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dict_to_config(data: dict) -> CodeEvolveConfig:
    """Build a CodeEvolveConfig from a flat dict (parsed YAML)."""
    provider = data.get("provider", "local")
    llama_server = LlamaServerConfig(**data.get("llama_server", {}))
    codex = CodexConfig(**data.get("codex", {}))
    claude = ClaudeConfig(**data.get("claude", {}))
    evolution = EvolutionConfig(**data.get("evolution", {}))
    rust_data = data.get("rust", {})
    rust = RustConfig(**{k: v for k, v in rust_data.items() if v is not None or k != "target_dir"})
    fitness_data = data.get("fitness", {})
    clippy_data = fitness_data.pop("clippy_weights", {})
    clippy_weights = ClippyWeights(**clippy_data)
    fitness = FitnessConfig(**fitness_data, clippy_weights=clippy_weights)
    benchmarks_data = data.get("benchmarks", {})
    # YAML null becomes None for list fields; coerce to let dataclass defaults apply
    for _list_key in ("upx_args",):
        if benchmarks_data.get(_list_key) is None and _list_key in benchmarks_data:
            del benchmarks_data[_list_key]
    benchmarks = BenchmarksConfig(**benchmarks_data)
    llm_judgment = LlmJudgmentConfig(**data.get("llm_judgment", {}))
    include_globs = data.get("include_globs") or ["src/**/*.rs"]
    exclude_globs = data.get("exclude_globs") or []
    return CodeEvolveConfig(
        provider=provider,
        llama_server=llama_server,
        codex=codex,
        claude=claude,
        evolution=evolution,
        rust=rust,
        fitness=fitness,
        benchmarks=benchmarks,
        llm_judgment=llm_judgment,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
    )


def load_config(path: Optional[Path] = None) -> CodeEvolveConfig:
    """Load config from YAML, merging with defaults. No path = pure defaults."""
    defaults_path = _DEFAULTS_DIR / "evolution.yaml"
    with open(defaults_path) as f:
        base_data = yaml.safe_load(f)

    if path is None:
        return _dict_to_config(base_data)

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        override_data = yaml.safe_load(f) or {}

    merged = _deep_merge(base_data, override_data)
    return _dict_to_config(merged)
