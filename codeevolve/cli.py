from __future__ import annotations

import shutil
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
    sync_project_config,
)
from codeevolve.runner import run_evolution, run_evolution_with_rotation, find_latest_checkpoint
from codeevolve.llama_server import LlamaServer
from codeevolve.codex_proxy import CodexProxy
from codeevolve.claude_proxy import ClaudeProxy
from codeevolve.mixed_proxy import MixedProxy


def _read_text_and_line_count(path: Path) -> tuple[str, int]:
    """Read a text file and return its content plus a minimum-1 line count."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text, max(1, len(text.splitlines()))


# Minimum iterations a file must receive for evolution to be useful.
# Below this the population can't fill and selection pressure is meaningless.
_MIN_ITERS_PER_FILE = 50


def _cap_files_for_budget(
    n_files: int,
    max_iterations: int,
    population_size: int,
    chunk_size: int,
) -> int:
    """Return the max number of files that fit the iteration budget.

    Each file needs at least ``_MIN_ITERS_PER_FILE`` or ``population_size``
    iterations (whichever is larger), rounded up to the next ``chunk_size``
    boundary.
    """
    min_iters = max(_MIN_ITERS_PER_FILE, population_size)
    # Round up to chunk boundary so the scheduler doesn't trim further
    if chunk_size > 0:
        min_iters = ((min_iters + chunk_size - 1) // chunk_size) * chunk_size
    max_files = max(1, max_iterations // min_iters)
    return min(n_files, max_files)


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
    from codeevolve.crate_graph import detect_workspace

    path = path.resolve()

    try:
        find_cargo_toml(path)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Detect workspace structure
    workspace_info = detect_workspace(path)
    if workspace_info is not None:
        include_globs = workspace_info.include_globs
        exclude_globs = workspace_info.exclude_globs

        click.echo(f"\n  Detected workspace with {len(workspace_info.crate_names)} crates:")
        # Count files per crate
        crate_counts = []
        for crate_name in workspace_info.crate_names:
            crate_root = workspace_info.crate_graph.crate_roots[crate_name]
            rs_count = len(list((crate_root / "src").rglob("*.rs"))) if (crate_root / "src").exists() else 0
            crate_counts.append(f"{crate_name} ({rs_count})")
        click.echo("    " + ", ".join(crate_counts))

        if exclude_globs:
            click.echo("\n  Auto-excluded generated directories:")
            for g in exclude_globs:
                click.echo(f"    - {g}")
    else:
        # Single-crate fallback: use defaults
        defaults = load_config()
        include_globs = defaults.include_globs
        exclude_globs = defaults.exclude_globs

    rs_files = discover_rs_files(path, include_globs, exclude_globs)
    if not rs_files:
        click.echo("Error: No .rs files matched include globs", err=True)
        sys.exit(1)

    click.echo(f"\n  Discovered {len(rs_files)} Rust source file(s)")

    click.echo("\n  Marking files for evolution...")
    for f in rs_files:
        insert_evolve_markers(f)

    codeevolve_dir = generate_codeevolve_dir(
        project_path=path,
        rs_files=rs_files,
        include_globs=include_globs if workspace_info is not None else None,
        exclude_globs=exclude_globs if workspace_info is not None else None,
    )

    click.echo(f"\n  Setup complete! Files created in {codeevolve_dir.relative_to(path)}/")
    click.echo("\n  Next steps:")
    click.echo("    1. Edit .codeevolve/evolution.yaml to set provider, binary_package, etc.")
    click.echo("    2. Start evolving:  codeevolve run")


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

    # Sync config with latest defaults (adds new sections, preserves user values)
    added_keys = sync_project_config(config_path)
    if added_keys:
        click.echo(f"  Config synced: added new sections: {', '.join(added_keys)}")
    else:
        click.echo("  Config up to date (no new sections)")

    config = load_config(config_path)
    rs_files = discover_rs_files(path, config.include_globs, config.exclude_globs)
    marked_files = [f for f in rs_files if "EVOLVE-BLOCK-START" in f.read_text(encoding="utf-8")]

    if not marked_files:
        click.echo("Error: No files with EVOLVE-BLOCK markers found. Run 'codeevolve init' first.", err=True)
        sys.exit(1)

    regenerate_evaluator(path, config_path, focus_file=marked_files[0])
    click.echo(f"  Regenerated .codeevolve/evaluator.py ({len(marked_files)} source file(s))")


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), default=".codeevolve/evolution.yaml", help="Path to config file")
@click.option("--fresh", is_flag=True, default=False, help="Start fresh, ignoring any existing checkpoints")
def run(config_path: Path, fresh: bool):
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

    if fresh:
        output_dir = project_path / ".codeevolve" / "output"
        checkpoints_dir = output_dir / "checkpoints"
        rotation_state_path = output_dir / "rotation_state.json"
        click.echo("  Clearing existing checkpoints and rotation state (--fresh)...")
        try:
            shutil.rmtree(checkpoints_dir)
        except FileNotFoundError:
            pass
        try:
            rotation_state_path.unlink()
        except FileNotFoundError:
            pass

    try:
        config = load_config(config_path)
    except FileNotFoundError:
        click.echo(f"Error: Config not found at {config_path}", err=True)
        click.echo("Run 'codeevolve init' first to set up your project.", err=True)
        sys.exit(1)

    click.echo(f"  Loading config from {config_path.relative_to(project_path.parent)}")

    # Ensure all CLI subprocesses (wsl.exe, codex, claude) are killed on exit,
    # even if the process is terminated without reaching the finally block.
    import atexit
    from codeevolve.base_proxy import _kill_all_children
    atexit.register(_kill_all_children)

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
    elif config.provider == "mixed":
        click.echo(f"  Starting mixed proxy (codex: {config.codex.model}, claude: {config.claude.model})...")
        try:
            backend = MixedProxy(config.codex, config.claude, config.tiers)
            backend.start()
        except Exception as e:
            click.echo(f"Error starting mixed proxy: {e}", err=True)
            sys.exit(1)
        click.echo(f"  Mixed proxy ready (codex@{config.codex.proxy_port} + claude@{config.claude.proxy_port})")
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
    marked_files = []
    marked_file_lengths: dict[str, int] = {}
    for f in rs_files:
        text, line_count = _read_text_and_line_count(f)
        if "EVOLVE-BLOCK-START" in text:
            marked_files.append(f)
            marked_file_lengths[f.relative_to(project_path).as_posix()] = line_count

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

    checkpoint_path = None
    if not fresh:
        checkpoint_path = find_latest_checkpoint(project_path / ".codeevolve" / "output")
        if checkpoint_path:
            iter_num = Path(checkpoint_path).name.split("_")[-1]
            click.echo(f"  Resuming from checkpoint at iteration {iter_num}")

    # Check if rotation is configured via tech debt ledger
    schedule = None
    if config.evolution.tech_debt_ledger:
        from codeevolve.crate_graph import detect_workspace
        from codeevolve.import_graph import build_reverse_deps
        from codeevolve.ledger import LedgerEntry, parse_ledger
        from codeevolve.scheduler import build_schedule

        ledger_path = project_path / config.evolution.tech_debt_ledger
        entries = parse_ledger(ledger_path, prod_only=config.evolution.prod_only)
        if entries:
            # Build import graph for impact weighting
            workspace_info = detect_workspace(project_path)
            crate_graph = workspace_info.crate_graph if workspace_info is not None else None
            reverse_deps = build_reverse_deps(project_path, rs_files, crate_graph)

            # Apply impact weighting: priority = debt_score * (1 + reverse_dep_count)
            weighted_entries = []
            weighted_lengths: dict[str, int] = {}
            for entry in entries:
                dep_count = reverse_deps.get(entry.file_path, 0)
                weighted_score = entry.combined_score * (1 + dep_count)
                weighted_entries.append(
                    LedgerEntry(
                        file_path=entry.file_path,
                        file_type=entry.file_type,
                        combined_score=weighted_score,
                    )
                )
            weighted_entries.sort(key=lambda e: e.combined_score, reverse=True)

            top_entries = weighted_entries[:config.evolution.top_n_files]
            # Filter to only entries that exist on disk AND have EVOLVE-BLOCK markers
            valid_entries = []
            for entry in top_entries:
                full_path = project_path / entry.file_path
                if full_path.exists():
                    try:
                        text, line_count = _read_text_and_line_count(full_path)
                        if "EVOLVE-BLOCK-START" in text:
                            valid_entries.append(entry)
                            weighted_lengths[entry.file_path] = line_count
                    except OSError:
                        pass
            if valid_entries:
                schedule = build_schedule(
                    valid_entries,
                    total_iterations=config.evolution.max_iterations,
                    chunk_size=config.evolution.checkpoint_interval,
                    file_lengths=weighted_lengths,
                    shuffle=config.evolution.shuffle_schedule,
                )

    try:
        if schedule:
            click.echo(f"  Rotation schedule: {len(schedule)} slots across {len(set(s.file_path for s in schedule))} files")
            results = run_evolution_with_rotation(
                config_path, project_path, schedule, marked_files,
                evaluator_path, checkpoint_path=checkpoint_path,
            )
            click.echo("\n-- Summary " + "-" * 45)
            for file_path, result in results.items():
                click.echo(f"  {file_path}: score {result.best_score:.2f}")
            click.echo(f"  All best candidates: .codeevolve/output/best/")
        else:
            # No ledger configured — use round-robin if multiple files
            if len(marked_files) > 1:
                from codeevolve.scheduler import build_roundrobin_schedule

                # Rank by LoC (descending) and apply top_n_files cap
                top_n = config.evolution.top_n_files
                ranked = sorted(
                    marked_files,
                    key=lambda f: marked_file_lengths.get(
                        f.relative_to(project_path).as_posix(), 0
                    ),
                    reverse=True,
                )[:top_n]
                ranked_rel = [f.relative_to(project_path).as_posix() for f in ranked]
                ranked_lengths = {k: v for k, v in marked_file_lengths.items() if k in ranked_rel}

                if len(marked_files) > top_n:
                    click.echo(f"  Capped to top {top_n} files by LoC (from {len(marked_files)} marked)")

                rr_schedule = build_roundrobin_schedule(
                    ranked_rel,
                    total_iterations=config.evolution.max_iterations,
                    chunk_size=config.evolution.checkpoint_interval,
                    file_lengths=ranked_lengths,
                    shuffle=config.evolution.shuffle_schedule,
                )
                if rr_schedule:
                    click.echo(f"  Round-robin schedule: {len(rr_schedule)} slots across {len(ranked)} files")
                    results = run_evolution_with_rotation(
                        config_path, project_path, rr_schedule, ranked,
                        evaluator_path, checkpoint_path=checkpoint_path,
                    )
                    click.echo("\n-- Summary " + "-" * 45)
                    for file_path, result in results.items():
                        click.echo(f"  {file_path}: score {result.best_score:.2f}")
                    click.echo(f"  All best candidates: .codeevolve/output/best/")
                else:
                    # Not enough iterations for even one chunk — single file fallback
                    result = run_evolution(config_path, project_path, marked_files, evaluator_path, checkpoint_path=checkpoint_path)
                    click.echo("\n-- Summary " + "-" * 45)
                    click.echo(f"  Best score:      {result.best_score:.2f}")
                    click.echo(f"  Best candidate:  .codeevolve/output/best/")
                    click.echo(f"  Metrics CSV:     .codeevolve/output/metrics.csv")
                    click.echo(f"  All candidates:  .codeevolve/output/")
            else:
                # Single file — use original path
                result = run_evolution(config_path, project_path, marked_files, evaluator_path, checkpoint_path=checkpoint_path)
                click.echo("\n-- Summary " + "-" * 45)
                click.echo(f"  Best score:      {result.best_score:.2f}")
                click.echo(f"  Best candidate:  .codeevolve/output/best/")
                click.echo(f"  Metrics CSV:     .codeevolve/output/metrics.csv")
                click.echo(f"  All candidates:  .codeevolve/output/")
    except KeyboardInterrupt:
        click.echo("\n\nStopped by user. Best result saved to .codeevolve/output/best/")
        click.echo("  Resume with: codeevolve run")
    finally:
        if backend is not None:
            backend.stop()
