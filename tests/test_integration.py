"""Integration test: init a real Rust crate and verify everything is generated."""
from pathlib import Path

from click.testing import CliRunner

from codeevolve.cli import main


def test_full_init_flow(sample_crate: Path):
    """End-to-end: codeevolve init on a real Rust crate."""
    runner = CliRunner()

    # Run init with default selection (all files) and no benchmark
    result = runner.invoke(
        main,
        ["init", "--path", str(sample_crate)],
        input="\nn\n",  # Enter for all files, 'n' for no benchmark
    )
    assert result.exit_code == 0, f"Init failed: {result.output}"

    # Verify .codeevolve/ was created
    ce_dir = sample_crate / ".codeevolve"
    assert ce_dir.exists()
    assert (ce_dir / "evolution.yaml").exists()
    assert (ce_dir / "evaluator.py").exists()
    assert (ce_dir / "README.md").exists()

    # Verify EVOLVE-BLOCK markers are present (sample_crate already has them)
    lib_rs = sample_crate / "src" / "lib.rs"
    content = lib_rs.read_text()
    assert "// EVOLVE-BLOCK-START" in content
    assert "// EVOLVE-BLOCK-END" in content

    # Verify evaluator.py is valid Python
    evaluator_code = (ce_dir / "evaluator.py").read_text()
    compile(evaluator_code, "evaluator.py", "exec")

    # Verify config has expected structure
    import yaml
    with open(ce_dir / "evolution.yaml") as f:
        config = yaml.safe_load(f)
    assert config["ollama"]["api_base"] == "http://localhost:11434/v1"
    assert config["evolution"]["max_iterations"] == 500
