import logging
from pathlib import Path

import pytest

from codeevolve.config import CodeEvolveConfig, CodexConfig, EvolutionConfig, load_config


def test_load_default_config():
    """Loading with no path returns defaults."""
    config = load_config()
    assert "llama-server" in config.llama_server.server_path
    assert config.llama_server.model_path.endswith(".gguf")
    assert config.llama_server.port == 8080
    assert config.llama_server.gpu_layers == 30
    assert config.llama_server.context_size == 8192
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
    assert oe_dict["diff_based_evolution"] is True
    assert oe_dict["llm"]["api_base"] == config.api_base
    assert oe_dict["llm"]["models"][0]["name"]  # model name is non-empty


def test_config_api_base_property():
    """api_base is derived from port."""
    config = load_config()
    assert config.llama_server.api_base == "http://localhost:8080/v1"


def test_config_model_name_property():
    """model_name is derived from model_path stem."""
    config = load_config()
    assert config.llama_server.model_name  # derived from model_path stem


def test_default_provider_is_codex():
    config = load_config()
    assert config.provider == "codex"


def test_codex_config_defaults():
    config = load_config()
    assert config.codex.cli_path == "codex"
    assert config.codex.model == "gpt-5.4-mini"
    assert config.codex.proxy_port == 8081
    assert config.codex.timeout == 300


def test_convenience_properties_local(tmp_path: Path):
    """api_base and model_name delegate to llama_server when provider=local."""
    yaml_content = 'provider: "local"\n'
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)
    config = load_config(config_path)
    assert config.api_base == config.llama_server.api_base
    assert config.model_name == config.llama_server.model_name


def test_convenience_properties_codex(tmp_path: Path):
    """api_base and model_name delegate to codex when provider=codex."""
    yaml_content = """
provider: "codex"
codex:
  model: "gpt-5.4-mini"
  proxy_port: 9090
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)
    config = load_config(config_path)
    assert config.provider == "codex"
    assert config.api_base == "http://localhost:9090/v1"
    assert config.model_name == "gpt-5.4-mini"


def test_openevolve_dict_uses_codex_when_configured(tmp_path: Path):
    """to_openevolve_dict should use codex api_base when provider=codex."""
    yaml_content = """
provider: "codex"
codex:
  proxy_port: 9090
  model: "gpt-5.4-mini"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)
    config = load_config(config_path)
    oe_dict = config.to_openevolve_dict()
    assert oe_dict["llm"]["api_base"] == "http://localhost:9090/v1"
    assert oe_dict["llm"]["models"][0]["name"] == "gpt-5.4-mini"


# ---------------------------------------------------------------------------
# New workspace evolution fields
# ---------------------------------------------------------------------------


def test_include_exclude_globs_defaults():
    """include_globs defaults to ['src/**/*.rs'] and exclude_globs to []."""
    config = load_config()
    assert config.include_globs == ["src/**/*.rs"]
    assert config.exclude_globs == []


def test_include_exclude_globs_override(tmp_path: Path):
    """include_globs and exclude_globs can be overridden via YAML."""
    yaml_content = """
include_globs:
  - "src/**/*.rs"
  - "benches/**/*.rs"
exclude_globs:
  - "src/generated/**"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)
    config = load_config(config_path)
    assert config.include_globs == ["src/**/*.rs", "benches/**/*.rs"]
    assert config.exclude_globs == ["src/generated/**"]


def test_benchmarks_new_fields_defaults():
    """BenchmarksConfig new fields have correct defaults."""
    config = load_config()
    assert config.benchmarks.custom_command == "cargo bench"
    assert config.benchmarks.binary_package is None
    assert config.benchmarks.upx_path is None
    assert config.benchmarks.upx_args == ["--best", "--force"]


def test_benchmarks_new_fields_override(tmp_path: Path):
    """BenchmarksConfig new fields can be overridden via YAML."""
    yaml_content = """
benchmarks:
  binary_package: "my_bin"
  upx_path: "/usr/bin/upx"
  upx_args:
    - "--ultra-brute"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)
    config = load_config(config_path)
    assert config.benchmarks.binary_package == "my_bin"
    assert config.benchmarks.upx_path == "/usr/bin/upx"
    assert config.benchmarks.upx_args == ["--ultra-brute"]


def test_benchmarks_upx_args_null_in_yaml(tmp_path: Path):
    """upx_args set to null in YAML falls back to dataclass default."""
    yaml_content = """
benchmarks:
  upx_args: null
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)
    config = load_config(config_path)
    assert config.benchmarks.upx_args == ["--best", "--force"]


def test_include_globs_null_in_yaml(tmp_path: Path):
    """include_globs/exclude_globs set to null in YAML fall back to defaults."""
    yaml_content = """
include_globs: null
exclude_globs: null
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)
    config = load_config(config_path)
    assert config.include_globs == ["src/**/*.rs"]
    assert config.exclude_globs == []


# ---------------------------------------------------------------------------
# Checkpoint/resume and tech-debt-weighted evolution fields
# ---------------------------------------------------------------------------


def test_checkpoint_and_tech_debt_defaults():
    """New EvolutionConfig fields have correct default values."""
    config = load_config()
    assert config.evolution.checkpoint_interval == 10
    assert config.evolution.tech_debt_ledger == ""
    assert config.evolution.top_n_files == 20
    assert config.evolution.prod_only is True


def test_checkpoint_and_tech_debt_override(tmp_path: Path):
    """New evolution fields can be overridden via YAML."""
    yaml_content = """
evolution:
  checkpoint_interval: 25
  tech_debt_ledger: "TECH_DEBT_LEDGER.md"
  top_n_files: 10
  prod_only: false
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)
    config = load_config(config_path)
    assert config.evolution.checkpoint_interval == 25
    assert config.evolution.tech_debt_ledger == "TECH_DEBT_LEDGER.md"
    assert config.evolution.top_n_files == 10
    assert config.evolution.prod_only is False
    # non-overridden evolution fields keep their defaults
    assert config.evolution.max_iterations == 500


def test_checkpoint_interval_in_openevolve_dict():
    """checkpoint_interval is included in the OpenEvolve dict."""
    config = load_config()
    oe_dict = config.to_openevolve_dict()
    assert "checkpoint_interval" in oe_dict
    assert oe_dict["checkpoint_interval"] == 10


def test_tech_debt_fields_not_in_openevolve_dict():
    """tech_debt_ledger, top_n_files, prod_only are NOT passed to OpenEvolve."""
    config = load_config()
    oe_dict = config.to_openevolve_dict()
    assert "tech_debt_ledger" not in oe_dict
    assert "top_n_files" not in oe_dict
    assert "prod_only" not in oe_dict


def test_backward_compat_yaml_without_new_fields(tmp_path: Path):
    """YAML that omits the new evolution fields still loads with correct defaults."""
    yaml_content = """
evolution:
  max_iterations: 200
  population_size: 50
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)
    config = load_config(config_path)
    # Explicitly set fields respected
    assert config.evolution.max_iterations == 200
    assert config.evolution.population_size == 50
    # New fields fall back to dataclass defaults
    assert config.evolution.checkpoint_interval == 10
    assert config.evolution.tech_debt_ledger == ""
    assert config.evolution.top_n_files == 20
    assert config.evolution.prod_only is True


# ---------------------------------------------------------------------------
# Meta-prompting and exploration/exploitation/temperature fields
# ---------------------------------------------------------------------------


def test_new_evolution_fields_defaults():
    """New EvolutionConfig fields have correct default values."""
    config = load_config()
    assert config.evolution.changes_description is False
    assert config.evolution.exploration_ratio == 0.2
    assert config.evolution.exploitation_ratio == 0.7
    assert config.evolution.temperature == 0.7


def test_exploration_exploitation_in_openevolve_dict():
    """exploration_ratio and exploitation_ratio appear in the 'database' section."""
    config = load_config()
    oe_dict = config.to_openevolve_dict()
    assert oe_dict["database"]["exploration_ratio"] == 0.2
    assert oe_dict["database"]["exploitation_ratio"] == 0.7


def test_temperature_custom_value_in_openevolve_dict():
    """Custom temperature flows through to OpenEvolve dict."""
    config = load_config()
    config.evolution.temperature = 0.9
    oe_dict = config.to_openevolve_dict()
    assert oe_dict["llm"]["temperature"] == 0.9


def test_temperature_default_in_openevolve_dict():
    """Default temperature of 0.7 is forwarded to OpenEvolve dict."""
    config = load_config()
    oe_dict = config.to_openevolve_dict()
    assert oe_dict["llm"]["temperature"] == 0.7


def test_changes_description_false_no_prompt_keys():
    """When changes_description=False, meta-prompting keys are absent."""
    config = load_config()
    config.evolution.changes_description = False
    oe_dict = config.to_openevolve_dict()
    assert "programs_as_changes_description" not in oe_dict["prompt"]
    assert "initial_changes_description" not in oe_dict["prompt"]


def test_changes_description_true_adds_prompt_keys():
    """When changes_description=True, meta-prompting keys are present and diff is forced."""
    config = load_config()
    config.evolution.changes_description = True
    config.evolution.diff_based_evolution = True
    oe_dict = config.to_openevolve_dict()
    assert oe_dict["prompt"]["programs_as_changes_description"] is True
    assert "initial_changes_description" in oe_dict["prompt"]
    assert oe_dict["diff_based_evolution"] is True


def test_changes_description_forces_diff_mode(caplog):
    """changes_description=True forces diff_based_evolution=True even when config has it False."""
    config = load_config()
    config.evolution.changes_description = True
    config.evolution.diff_based_evolution = False

    with caplog.at_level(logging.WARNING, logger="codeevolve.config"):
        oe_dict = config.to_openevolve_dict()

    assert oe_dict["diff_based_evolution"] is True
    assert oe_dict["prompt"]["programs_as_changes_description"] is True
    assert any("forcing diff_based_evolution" in msg for msg in caplog.messages)


def test_new_evolution_fields_override_via_yaml(tmp_path: Path):
    """New evolution fields can be overridden via YAML."""
    yaml_content = """
evolution:
  changes_description: true
  exploration_ratio: 0.3
  exploitation_ratio: 0.6
  temperature: 0.5
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)
    config = load_config(config_path)
    assert config.evolution.changes_description is True
    assert config.evolution.exploration_ratio == 0.3
    assert config.evolution.exploitation_ratio == 0.6
    assert config.evolution.temperature == 0.5


def test_new_fields_in_openevolve_dict_after_yaml_override(tmp_path: Path):
    """YAML-overridden fields flow through to the OE dict."""
    yaml_content = """
evolution:
  changes_description: true
  exploration_ratio: 0.15
  exploitation_ratio: 0.80
  temperature: 0.4
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)
    config = load_config(config_path)
    oe_dict = config.to_openevolve_dict()
    assert oe_dict["database"]["exploration_ratio"] == 0.15
    assert oe_dict["database"]["exploitation_ratio"] == 0.80
    assert oe_dict["llm"]["temperature"] == 0.4
    assert oe_dict["prompt"]["programs_as_changes_description"] is True
    assert "initial_changes_description" in oe_dict["prompt"]
