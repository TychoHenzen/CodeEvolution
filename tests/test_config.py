from pathlib import Path

import pytest

from codeevolve.config import CodeEvolveConfig, load_config


def test_load_default_config():
    """Loading with no path returns defaults."""
    config = load_config()
    assert "llama-server" in config.llama_server.server_path
    assert config.llama_server.model_path.endswith(".gguf")
    assert config.llama_server.port == 8080
    assert config.llama_server.gpu_layers == 30
    assert config.llama_server.context_size == 4096
    assert config.llama_server.threads == 8
    assert config.llama_server.flash_attn is True
    assert config.evolution.max_iterations == 500
    assert config.fitness.static_analysis_weight == 0.35


def test_load_config_from_yaml(tmp_path: Path):
    """Loading from a YAML file overrides defaults."""
    yaml_content = """
llama_server:
  port: 9090
  gpu_layers: 20
evolution:
  max_iterations: 100
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)

    config = load_config(config_path)
    assert config.llama_server.port == 9090
    assert config.llama_server.gpu_layers == 20
    assert config.evolution.max_iterations == 100
    # non-overridden fields keep defaults
    assert config.llama_server.model_path.endswith(".gguf")


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
    assert oe_dict["llm"]["api_base"] == "http://localhost:8080/v1"
    assert oe_dict["llm"]["models"][0]["name"]  # model name is non-empty


def test_config_api_base_property():
    """api_base is derived from port."""
    config = load_config()
    assert config.llama_server.api_base == "http://localhost:8080/v1"


def test_config_model_name_property():
    """model_name is derived from model_path stem."""
    config = load_config()
    assert config.llama_server.model_name  # derived from model_path stem
