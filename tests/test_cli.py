from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from codeevolve.cli import main


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_cli_help(cli_runner):
    result = cli_runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "init" in result.output
    assert "run" in result.output


def test_init_help(cli_runner):
    result = cli_runner.invoke(main, ["init", "--help"])
    assert result.exit_code == 0
    assert "--path" in result.output


def test_run_help(cli_runner):
    result = cli_runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output


def test_init_no_cargo_toml(cli_runner, tmp_path):
    result = cli_runner.invoke(main, ["init", "--path", str(tmp_path)])
    assert result.exit_code != 0
    assert "Cargo.toml" in result.output


@patch("codeevolve.cli.generate_codeevolve_dir")
@patch("codeevolve.cli.insert_evolve_markers")
@patch("codeevolve.cli.scan_rs_files")
@patch("codeevolve.cli.find_cargo_toml")
def test_init_success(mock_find, mock_scan, mock_markers, mock_generate, cli_runner, sample_crate):
    mock_find.return_value = sample_crate / "Cargo.toml"
    mock_scan.return_value = [sample_crate / "src" / "lib.rs"]
    mock_generate.return_value = sample_crate / ".codeevolve"
    result = cli_runner.invoke(main, ["init", "--path", str(sample_crate)], input="\nn\n")
    assert result.exit_code == 0


@patch("codeevolve.cli.LlamaServer")
def test_run_server_start_fails(mock_server_cls, cli_runner, tmp_path):
    mock_server_cls.return_value.start.side_effect = RuntimeError("model not found")
    config_path = tmp_path / "evolution.yaml"
    config_path.write_text("provider: local\nllama_server:\n  port: 8080\n")
    result = cli_runner.invoke(main, ["run", "--config", str(config_path)])
    assert result.exit_code != 0
    assert "model not found" in result.output
