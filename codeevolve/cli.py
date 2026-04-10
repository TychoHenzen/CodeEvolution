from __future__ import annotations

import sys
from pathlib import Path

import click

# Force UTF-8 on stdout/stderr so OpenEvolve's Unicode log messages
# (arrows, emojis) don't crash on Windows cp1252 consoles.
# reconfigure() alone doesn't reliably stick on Windows .exe entry
# points, so replace the streams entirely before any logging handler
# captures a reference.  Skip when pytest captures streams.
if sys.platform == "win32" and "pytest" not in sys.modules:
    import io
    for _attr in ("stdout", "stderr"):
        _old = getattr(sys, _attr)
        if hasattr(_old, "buffer"):
            setattr(sys, _attr, io.TextIOWrapper(
                _old.buffer, encoding="utf-8", errors="replace",
                line_buffering=_old.line_buffering,
            ))

from codeevolve.config import load_config
from codeevolve.file_discovery import discover_rs_files
from codeevolve.init_project import (
    find_cargo_toml,
    generate_codeevolve_dir,
    insert_evolve_markers,
    regenerate_evaluator,
    scan_rs_files,
)
from codeevolve.runner import validate_server, run_evolution
from codeevolve.llama_server import LlamaServer
from codeevolve.codex_proxy import CodexProxy
from codeevolve.claude_proxy import ClaudeProxy


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

    Scans your project for Rust source files using glob patterns, marks
    them for evolution, and generates configuration files in a .codeevolve/
    directory.
    """
    path = path.resolve()

    try:
        find_cargo_toml(path)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Load default config to get include/exclude globs
    defaults = load_config()
    rs_files = discover_rs_files(path, defaults.include_globs, defaults.exclude_globs)
    if not rs_files:
        click.echo("Error: No .rs files matched include globs", err=True)
        sys.exit(1)

    click.echo(f"\nDiscovered {len(rs_files)} Rust source file(s):")
    for f in rs_files:
        click.echo(f"  - {f.relative_to(path)}")

    click.echo("\nMarking files for evolution...")
    for f in rs_files:
        insert_evolve_markers(f)
        click.echo(f"  Marked: {f.relative_to(path)}")

    codeevolve_dir = generate_codeevolve_dir(
        project_path=path,
        rs_files=rs_files,
    )

    click.echo(f"\nSetup complete! Files created in {codeevolve_dir.relative_to(path)}/")
    click.echo("\nNext steps:")
    click.echo("  1. Edit .codeevolve/evolution.yaml (set include/exclude globs, provider, etc.)")
    click.echo("  2. Start evolving:  codeevolve run")


@main.command()
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".", help="Path to Rust project (default: current directory)")
def reinit(path: Path):
    """Regenerate evaluator.py without interactive prompts.

    Use this after updating codeevolve to pick up evaluator changes
    without re-entering benchmark config or file selections.
    """
    path = path.resolve()
    config_path = path / ".codeevolve" / "evolution.yaml"
    if not config_path.exists():
        click.echo("Error: .codeevolve/evolution.yaml not found. Run 'codeevolve init' first.", err=True)
        sys.exit(1)

    config = load_config(config_path)
    rs_files = discover_rs_files(path, config.include_globs, config.exclude_globs)
    marked_files = [f for f in rs_files if "EVOLVE-BLOCK-START" in f.read_text()]

    if not marked_files:
        click.echo("Error: No files with EVOLVE-BLOCK markers found. Run 'codeevolve init' first.", err=True)
        sys.exit(1)

    regenerate_evaluator(
        path, config_path,
        source_files=marked_files,
        focus_file=marked_files[0],
    )
    click.echo(f"  Regenerated .codeevolve/evaluator.py ({len(marked_files)} source file(s))")


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

    # Start the LLM backend based on provider setting.
    backend = None
    if config.provider == "codex":
        click.echo(f"  Starting Codex proxy (model: {config.codex.model})...")
        try:
            backend = CodexProxy(config.codex)
            backend.start()
        except Exception as e:
            click.echo(f"Error starting Codex proxy: {e}", err=True)
            sys.exit(1)
        click.echo(f"  Codex proxy ready on port {config.codex.proxy_port}")
    elif config.provider == "claude":
        click.echo(f"  Starting Claude proxy (model: {config.claude.model}, effort: {config.claude.effort})...")
        try:
            backend = ClaudeProxy(config.claude)
            backend.start()
        except Exception as e:
            click.echo(f"Error starting Claude proxy: {e}", err=True)
            sys.exit(1)
        click.echo(f"  Claude proxy ready on port {config.claude.proxy_port}")
    else:
        model_name = Path(config.llama_server.model_path).name
        click.echo(f"  Starting llama-server ({model_name})...")
        click.echo(f"  Loading model into GPU ({config.llama_server.gpu_layers} layers offloaded)... this may take a minute")
        try:
            backend = LlamaServer(config.llama_server)
            backend.start()
        except (RuntimeError, TimeoutError) as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        click.echo(f"  llama-server ready on port {config.llama_server.port}")

    click.echo(f"  Starting evolution ({config.evolution.max_iterations} iterations, population {config.evolution.population_size})")
    click.echo()

    rs_files = discover_rs_files(project_path, config.include_globs, config.exclude_globs)
    marked_files = [f for f in rs_files if "EVOLVE-BLOCK-START" in f.read_text()]

    if not marked_files:
        click.echo("Error: No files with EVOLVE-BLOCK markers found. Run 'codeevolve init' first.", err=True)
        sys.exit(1)

    click.echo(f"  Source files: {len(marked_files)} with EVOLVE-BLOCK markers")

    evaluator_path = config_path.parent / "evaluator.py"
    if not evaluator_path.exists():
        click.echo("Error: evaluator.py not found in .codeevolve/. Run 'codeevolve init' first.", err=True)
        sys.exit(1)

    csv_path = config_path.parent / "output" / "metrics.csv"
    click.echo(f"  Metrics CSV: {csv_path.relative_to(project_path)}")
    click.echo()

    try:
        result = run_evolution(config_path, project_path, marked_files, evaluator_path)
        click.echo("\n-- Summary " + "-" * 45)
        click.echo(f"  Best score:      {result.best_score:.2f}")
        click.echo(f"  Best candidate:  .codeevolve/output/best/")
        click.echo(f"  Metrics CSV:     .codeevolve/output/metrics.csv")
        click.echo(f"  All candidates:  .codeevolve/output/")
    except KeyboardInterrupt:
        click.echo("\n\nStopped by user. Best result saved to .codeevolve/output/best/")
    finally:
        backend.stop()
