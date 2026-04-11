from pathlib import Path

import pytest
import yaml

from codeevolve.init_project import (
    find_cargo_toml,
    insert_evolve_markers,
    generate_codeevolve_dir,
)


def test_find_cargo_toml(sample_crate: Path):
    result = find_cargo_toml(sample_crate)
    assert result == sample_crate / "Cargo.toml"


def test_find_cargo_toml_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Cargo.toml"):
        find_cargo_toml(tmp_path)


def test_find_cargo_toml_workspace_accepted(tmp_path: Path):
    """Workspace Cargo.toml is now accepted (workspace evolution is supported)."""
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text('[workspace]\nmembers = ["crate_a"]')
    result = find_cargo_toml(tmp_path)
    assert result == cargo


def test_insert_evolve_markers_wraps_whole_file(tmp_path: Path):
    rs_file = tmp_path / "test.rs"
    rs_file.write_text("fn hello() {}\n")
    insert_evolve_markers(rs_file)
    content = rs_file.read_text()
    assert content.startswith("// EVOLVE-BLOCK-START\n")
    assert content.endswith("// EVOLVE-BLOCK-END\n")
    assert "fn hello() {}" in content


def test_insert_evolve_markers_skips_if_already_marked(tmp_path: Path):
    rs_file = tmp_path / "test.rs"
    original = "// EVOLVE-BLOCK-START\nfn hello() {}\n// EVOLVE-BLOCK-END\n"
    rs_file.write_text(original)
    insert_evolve_markers(rs_file)
    assert rs_file.read_text() == original


def test_generate_codeevolve_dir(sample_crate: Path):
    generate_codeevolve_dir(
        project_path=sample_crate,
        rs_files=[sample_crate / "src" / "lib.rs"],
        custom_benchmark=None,
    )
    codeevolve_dir = sample_crate / ".codeevolve"
    assert codeevolve_dir.exists()
    assert (codeevolve_dir / "evolution.yaml").exists()
    assert (codeevolve_dir / "evaluator.py").exists()
    assert (codeevolve_dir / "README.md").exists()
    evaluator_code = (codeevolve_dir / "evaluator.py").read_text()
    compile(evaluator_code, "evaluator.py", "exec")


def test_generate_codeevolve_dir_with_workspace_globs(sample_workspace: Path):
    """Workspace globs are written into evolution.yaml when provided."""
    rs_files = list((sample_workspace / "crates").rglob("*.rs"))
    generate_codeevolve_dir(
        project_path=sample_workspace,
        rs_files=rs_files,
        include_globs=["crates/*/src/**/*.rs"],
        exclude_globs=["crates/game/src/card/generated/**"],
    )
    config_path = sample_workspace / ".codeevolve" / "evolution.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    assert config["include_globs"] == ["crates/*/src/**/*.rs"]
    assert "crates/game/src/card/generated/**" in config["exclude_globs"]
