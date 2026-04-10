from pathlib import Path

import pytest

from codeevolve.config import CodeEvolveConfig, load_config


def test_load_default_config():
    """Loading with no path returns defaults."""
    config = load_config()
    assert config.ollama.api_base == "http://localhost:11434/v1"
    assert config.ollama.mutator_model == "qwen2.5-coder:7b-instruct-q4_K_M"
    assert config.ollama.evaluator_model == "qwen2.5-coder:7b-instruct-q4_K_M"
    assert config.evolution.max_iterations == 500
    assert config.fitness.static_analysis_weight == 0.35


def test_load_config_from_yaml(tmp_path: Path):
    """Loading from a YAML file overrides defaults."""
    yaml_content = """
ollama:
  api_base: "http://custom:1234/v1"
evolution:
  max_iterations: 100
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)

    config = load_config(config_path)
    assert config.ollama.api_base == "http://custom:1234/v1"
    assert config.evolution.max_iterations == 100
    # non-overridden fields keep defaults
    assert config.ollama.mutator_model == "qwen2.5-coder:7b-instruct-q4_K_M"


def test_load_config_missing_file():
    """Loading a nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.yaml"))


def test_config_clippy_weights_default():
    config = load_config()
    assert config.fitness.clippy_weights.correctness == 5
    assert config.fitness.clippy_weights.suspicious == 3
    assert config.fitness.clippy_weights.complexity == 2
    assert config.fitness.clippy_weights.perf == 2
    assert config.fitness.clippy_weights.style == 1


def test_config_to_openevolve_dict():
    """Config converts to an OpenEvolve-compatible dict."""
    config = load_config()
    oe_dict = config.to_openevolve_dict()
    assert oe_dict["max_iterations"] == 500
    assert oe_dict["diff_based_evolution"] is False
    assert oe_dict["llm"]["api_base"] == "http://localhost:11434/v1"
    assert oe_dict["llm"]["models"][0]["name"] == "qwen2.5-coder:7b-instruct-q4_K_M"
