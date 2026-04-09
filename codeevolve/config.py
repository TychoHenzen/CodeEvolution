from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

_DEFAULTS_DIR = Path(__file__).parent / "defaults"


@dataclass
class OllamaConfig:
    api_base: str = "http://localhost:11434/v1"
    mutator_model: str = "qwen2.5-coder:7b-instruct-q4_K_M"
    evaluator_model: str = "qwen2.5-coder:1.5b-instruct-q4_K_M"


@dataclass
class EvolutionConfig:
    max_iterations: int = 500
    population_size: int = 100
    num_islands: int = 3
    migration_interval: int = 20
    context_window: int = 4096
    diff_based_evolution: bool = True


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
    top_quartile_only: bool = True
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
        return {
            "max_iterations": self.evolution.max_iterations,
            "diff_based_evolution": self.evolution.diff_based_evolution,
            "file_suffix": ".rs",
            "llm": {
                "api_base": self.ollama.api_base,
                "api_key": "ollama",
                "models": [
                    {"name": self.ollama.mutator_model, "weight": 1.0},
                ],
                "temperature": 0.7,
                "max_tokens": self.evolution.context_window,
            },
            "database": {
                "population_size": self.evolution.population_size,
                "num_islands": self.evolution.num_islands,
                "migration_interval": self.evolution.migration_interval,
            },
            "evaluator": {
                "timeout": 120,
                "parallel_evaluations": 1,
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
