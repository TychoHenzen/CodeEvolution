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
@patch("codeevolve.cli.discover_rs_files")
@patch("codeevolve.cli.find_cargo_toml")
def test_init_success(mock_find, mock_discover, mock_markers, mock_generate, cli_runner, sample_crate):
    mock_find.return_value = sample_crate / "Cargo.toml"
    mock_discover.return_value = [sample_crate / "src" / "lib.rs"]
    mock_generate.return_value = sample_crate / ".codeevolve"
    result = cli_runner.invoke(main, ["init", "--path", str(sample_crate)])
    assert result.exit_code == 0


def test_init_workspace_detects_crates(cli_runner, sample_workspace):
    result = cli_runner.invoke(main, ["init", "--path", str(sample_workspace)])
    assert result.exit_code == 0
    assert "Detected workspace" in result.output
    assert "engine_core" in result.output
    assert "engine_render" in result.output
    assert "game" in result.output


def test_init_workspace_excludes_generated(cli_runner, sample_workspace):
    result = cli_runner.invoke(main, ["init", "--path", str(sample_workspace)])
    assert result.exit_code == 0
    assert "generated" in result.output.lower()


@patch("codeevolve.cli.LlamaServer")
def test_run_server_start_fails(mock_server_cls, cli_runner, tmp_path):
    mock_server_cls.return_value.start.side_effect = RuntimeError("model not found")
    config_path = tmp_path / "evolution.yaml"
    config_path.write_text("provider: local\nllama_server:\n  port: 8080\n")
    result = cli_runner.invoke(main, ["run", "--config", str(config_path)])
    assert result.exit_code != 0
    assert "model not found" in result.output


def test_run_fresh_flag_in_help(cli_runner):
    result = cli_runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "--fresh" in result.output


def test_run_fresh_deletes_checkpoints(cli_runner, tmp_path):
    """--fresh should delete the checkpoints directory before starting."""
    # Set up a fake .codeevolve structure with an existing checkpoints dir
    codeevolve_dir = tmp_path / ".codeevolve"
    codeevolve_dir.mkdir()
    config_path = codeevolve_dir / "evolution.yaml"
    config_path.write_text("provider: local\nllama_server:\n  port: 8080\n")

    checkpoints_dir = codeevolve_dir / "output" / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    sentinel = checkpoints_dir / "checkpoint_001.json"
    sentinel.write_text("{}")

    # Stop execution after checkpoint deletion by making the server fail
    with patch("codeevolve.cli.LlamaServer") as mock_server_cls:
        mock_server_cls.return_value.start.side_effect = RuntimeError("stop here")
        result = cli_runner.invoke(main, ["run", "--config", str(config_path), "--fresh"])

    assert "Clearing existing checkpoints" in result.output
    assert not checkpoints_dir.exists(), "checkpoints dir should have been deleted"


def test_run_no_fresh_preserves_checkpoints(cli_runner, tmp_path):
    """Without --fresh, the checkpoints directory must not be touched."""
    codeevolve_dir = tmp_path / ".codeevolve"
    codeevolve_dir.mkdir()
    config_path = codeevolve_dir / "evolution.yaml"
    config_path.write_text("provider: local\nllama_server:\n  port: 8080\n")

    checkpoints_dir = codeevolve_dir / "output" / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    sentinel = checkpoints_dir / "checkpoint_001.json"
    sentinel.write_text("{}")

    with patch("codeevolve.cli.LlamaServer") as mock_server_cls:
        mock_server_cls.return_value.start.side_effect = RuntimeError("stop here")
        result = cli_runner.invoke(main, ["run", "--config", str(config_path)])

    assert "Clearing existing checkpoints" not in result.output
    assert checkpoints_dir.exists(), "checkpoints dir should still exist"
    assert sentinel.exists(), "checkpoint file should still exist"


def test_run_fresh_no_checkpoints_dir(cli_runner, tmp_path):
    """--fresh with no existing checkpoints dir should not raise an error."""
    codeevolve_dir = tmp_path / ".codeevolve"
    codeevolve_dir.mkdir()
    config_path = codeevolve_dir / "evolution.yaml"
    config_path.write_text("provider: local\nllama_server:\n  port: 8080\n")

    # No checkpoints dir created — should succeed silently
    with patch("codeevolve.cli.LlamaServer") as mock_server_cls:
        mock_server_cls.return_value.start.side_effect = RuntimeError("stop here")
        result = cli_runner.invoke(main, ["run", "--config", str(config_path), "--fresh"])

    assert "Clearing existing checkpoints" in result.output
    # Exit should be non-zero only due to the mocked server failure, not due to missing dir
    assert "FileNotFoundError" not in result.output
