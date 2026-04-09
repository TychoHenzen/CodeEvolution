from __future__ import annotations

import sys
from pathlib import Path

import click

from codeevolve.config import load_config
from codeevolve.init_project import (
    find_cargo_toml,
    generate_codeevolve_dir,
    insert_evolve_markers,
    scan_rs_files,
)
from codeevolve.runner import validate_ollama, run_evolution


@click.group()
@click.version_option(package_name="codeevolve")
def main():
    """CodeEvolution — evolutionary code optimization for Rust projects.

    Evolve your Rust code using AI-powered mutations. The optimizer tries
    hundreds of variations of your code, keeping only the ones that compile,
    pass tests, and score well on code quality metrics.

    Quick start:
      1. cd into your Rust project
      2. codeevolve init
      3. codeevolve run
    """
    pass


@main.command()
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".", help="Path to Rust project (default: current directory)")
def init(path: Path):
    """Set up a Rust project for evolutionary optimization.

    Scans your project for Rust source files, marks them for evolution,
    and generates configuration files in a .codeevolve/ directory.
    """
    path = path.resolve()

    try:
        find_cargo_toml(path)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    rs_files = scan_rs_files(path)
    if not rs_files:
        click.echo("Error: No .rs files found in src/", err=True)
        sys.exit(1)

    click.echo("\nFound Rust source files:")
    for i, f in enumerate(rs_files, 1):
        click.echo(f"  {i}. {f.relative_to(path)}")

    click.echo(f"\nWhich files should be evolved? (comma-separated numbers, or Enter for all)")
    selection = click.prompt("Selection", default="", show_default=False)

    if selection.strip():
        indices = [int(x.strip()) - 1 for x in selection.split(",")]
        selected = [rs_files[i] for i in indices if 0 <= i < len(rs_files)]
    else:
        selected = rs_files

    if not selected:
        click.echo("Error: No files selected", err=True)
        sys.exit(1)

    custom_benchmark = None
    custom_regex = None
    click.echo("\nBy default, the optimizer measures compile time, binary size, and lines of code.")
    if click.confirm("Do you also want to measure runtime performance? (requires a benchmark command like 'cargo bench')", default=False):
        custom_benchmark = click.prompt("Benchmark command")
        custom_regex = click.prompt(
            "Regex to extract score from output (or Enter to use exit code)",
            default="",
            show_default=False,
        )
        custom_regex = custom_regex if custom_regex else None

    click.echo("\nMarking files for evolution...")
    for f in selected:
        insert_evolve_markers(f)
        click.echo(f"  Marked: {f.relative_to(path)}")

    codeevolve_dir = generate_codeevolve_dir(
        project_path=path,
        rs_files=selected,
        custom_benchmark=custom_benchmark,
        custom_benchmark_regex=custom_regex,
    )

    click.echo(f"\nSetup complete! Files created in {codeevolve_dir.relative_to(path)}/")
    click.echo("\nNext steps:")
    click.echo("  1. Make sure Ollama is running:  ollama serve")
    click.echo("  2. Pull models (first time):")
    click.echo("       ollama pull qwen2.5-coder:7b-instruct-q4_K_M")
    click.echo("       ollama pull qwen2.5-coder:1.5b-instruct-q4_K_M")
    click.echo("  3. Start evolving:  codeevolve run")


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), default=".codeevolve/evolution.yaml", help="Path to config file")
def run(config_path: Path):
    """Run the evolutionary optimizer.

    Reads your configuration and starts evolving your code. Each iteration:
    1. The AI suggests a code change
    2. The change is tested (build, tests, linting, benchmarks)
    3. Good changes are kept, bad ones are discarded
    4. Over time, your code gets better

    Press Ctrl+C to stop at any time. Your best result is always saved.
    """
    config_path = config_path.resolve()
    project_path = config_path.parent.parent

    try:
        config = load_config(config_path)
    except FileNotFoundError:
        click.echo(f"Error: Config not found at {config_path}", err=True)
        click.echo("Run 'codeevolve init' first to set up your project.", err=True)
        sys.exit(1)

    click.echo(f"  Loading config from {config_path.relative_to(project_path.parent)}")

    errors = validate_ollama(config)
    if errors:
        for e in errors:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"  Connected to Ollama ({config.ollama.mutator_model}, {config.ollama.evaluator_model})")
    click.echo(f"  Starting evolution ({config.evolution.max_iterations} iterations, population {config.evolution.population_size})")
    click.echo()

    rs_files = scan_rs_files(project_path)
    initial = None
    for f in rs_files:
        if "EVOLVE-BLOCK-START" in f.read_text():
            initial = f
            break

    if not initial:
        click.echo("Error: No files with EVOLVE-BLOCK markers found. Run 'codeevolve init' first.", err=True)
        sys.exit(1)

    evaluator_path = config_path.parent / "evaluator.py"
    if not evaluator_path.exists():
        click.echo("Error: evaluator.py not found in .codeevolve/. Run 'codeevolve init' first.", err=True)
        sys.exit(1)

    try:
        result = run_evolution(config_path, project_path, initial, evaluator_path)
        click.echo("\n-- Summary " + "-" * 45)
        click.echo(f"  Best score:      {result.best_score:.2f}")
        click.echo(f"  Best candidate:  .codeevolve/output/best/")
        click.echo(f"  All candidates:  .codeevolve/output/")
    except KeyboardInterrupt:
        click.echo("\n\nStopped by user. Best result saved to .codeevolve/output/best/")
        sys.exit(0)
