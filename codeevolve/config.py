from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

_DEFAULTS_DIR = Path(__file__).parent / "defaults"
_PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class OllamaConfig:
    api_base: str = "http://localhost:11434/v1"
    mutator_model: str = "qwen2.5-coder:7b-instruct-q4_K_M"
    evaluator_model: str = "qwen2.5-coder:7b-instruct-q4_K_M"


@dataclass
class EvolutionConfig:
    max_iterations: int = 500
    population_size: int = 10
    num_islands: int = 3
    migration_interval: int = 20
    context_window: int = 16384
    diff_based_evolution: bool = False
    max_fix_attempts: int = 2  # LLM retry attempts on build/test failure


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
    custom_command: Optional[str] = None
    custom_command_score_regex: Optional[str] = None


@dataclass
class LlmJudgmentConfig:
    enabled: bool = True
    top_quartile_only: bool = False
    num_runs: int = 3
    dimensions: list[str] = field(
        default_factory=lambda: ["readability", "rust_idiomaticity", "maintainability", "design"]
    )


@dataclass
class CodeEvolveConfig:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    rust: RustConfig = field(default_factory=RustConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    benchmarks: BenchmarksConfig = field(default_factory=BenchmarksConfig)
    llm_judgment: LlmJudgmentConfig = field(default_factory=LlmJudgmentConfig)

    def to_openevolve_dict(self) -> dict:
        """Convert to a dict compatible with OpenEvolve's Config.from_dict()."""
        diff_mode = self.evolution.diff_based_evolution
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
            )
        else:
            system_msg = (
                "You are an expert Rust developer tasked with iteratively "
                "improving a Rust codebase. Your goal is to maximize the "
                "FITNESS SCORE while exploring diverse solutions. The code "
                "MUST compile with cargo build and pass all tests. Pay close "
                "attention to Rust's ownership, borrowing, and lifetime rules.\n\n"
                "You will receive the current program and must output a COMPLETE "
                "rewritten version with improvements. Output the full program "
                "inside a single ```rust code block.\n\n"
                "IMPORTANT: You MUST make at least one meaningful change. "
                "Do NOT output the original code unchanged. "
                "Always improve something — try a different algorithm, data "
                "structure, API usage, or code pattern. do NOT add tests."
            )

        return {
            "max_iterations": self.evolution.max_iterations,
            "diff_based_evolution": diff_mode,
            "file_suffix": ".rs",
            # Null seed: let each LLM call use a fresh random state.
            # A fixed seed (OpenEvolve defaults to 42) makes the model
            # produce identical outputs across iterations.
            "random_seed": None,
            "llm": {
                "api_base": self.ollama.api_base,
                "api_key": "ollama",
                "models": [
                    {"name": self.ollama.mutator_model, "weight": 1.0},
                ],
                "temperature": 1.0,
                "max_tokens": 16384,
                "timeout": 300,
            },
            "prompt": {
                "system_message": system_msg,
                "template_dir": str(_PROMPTS_DIR),
                "num_top_programs": 0,
                "num_diverse_programs": 0,
                "use_template_stochasticity": True,
            },
            "database": {
                "population_size": self.evolution.population_size,
                "num_islands": self.evolution.num_islands,
                "migration_interval": self.evolution.migration_interval,
                "feature_dimensions": ["llm_score", "perf_score", "loc"],
                "log_prompts": True,
            },
            "evaluator": {
                "timeout": 600,
                "parallel_evaluations": 1,
            },
            "early_stopping_patience": 40,
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
    ollama = OllamaConfig(**data.get("ollama", {}))
    evolution = EvolutionConfig(**data.get("evolution", {}))
    rust_data = data.get("rust", {})
    rust = RustConfig(**{k: v for k, v in rust_data.items() if v is not None or k != "target_dir"})
    fitness_data = data.get("fitness", {})
    clippy_data = fitness_data.pop("clippy_weights", {})
    clippy_weights = ClippyWeights(**clippy_data)
    fitness = FitnessConfig(**fitness_data, clippy_weights=clippy_weights)
    benchmarks = BenchmarksConfig(**data.get("benchmarks", {}))
    llm_judgment = LlmJudgmentConfig(**data.get("llm_judgment", {}))
    return CodeEvolveConfig(
        ollama=ollama,
        evolution=evolution,
        rust=rust,
        fitness=fitness,
        benchmarks=benchmarks,
        llm_judgment=llm_judgment,
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
