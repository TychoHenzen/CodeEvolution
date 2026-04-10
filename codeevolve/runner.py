from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.request import urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)


def _normalize_llm_diffs(text: str) -> str:
    """Rewrite common malformed diff formats into the canonical markers.

    Small LLMs (e.g. Qwen-7B) frequently emit markdown-style diffs instead of
    the ``<<<<<<< SEARCH`` / ``=======`` / ``>>>>>>> REPLACE`` format that
    OpenEvolve expects.  This function rewrites the two most common variants
    so that ``extract_diffs`` can parse them.

    Handled variants
    ----------------
    1. ``#### SEARCH`` … ````rust`` code ```` … ``#### REPLACE`` … ````rust`` code ````
    2. ``SEARCH`` / ``REPLACE`` as standalone lines with fenced code blocks
    """
    # Pattern: markdown headers with fenced code blocks
    # e.g.  #### SEARCH\n```rust\ncode\n```\n\n#### REPLACE\n```rust\ncode\n```
    md_pattern = re.compile(
        r"#{1,6}\s*SEARCH\s*\n"       # #### SEARCH
        r"```\w*\n"                    # ```rust  (opening fence)
        r"(.*?)"                       # captured search body
        r"\n```\s*\n+"                 # ``` (closing fence)
        r"#{1,6}\s*REPLACE\s*\n"      # #### REPLACE
        r"```\w*\n"                    # ```rust  (opening fence)
        r"(.*?)"                       # captured replace body
        r"\n```",                      # ``` (closing fence)
        re.DOTALL,
    )

    def _rewrite(m: re.Match) -> str:
        search_body = m.group(1)
        replace_body = m.group(2)
        return (
            f"<<<<<<< SEARCH\n{search_body}\n=======\n{replace_body}\n>>>>>>> REPLACE"
        )

    return md_pattern.sub(_rewrite, text)

import yaml

from codeevolve.config import CodeEvolveConfig, load_config


def validate_server(config: CodeEvolveConfig) -> list[str]:
    """Check that the LLM backend is reachable."""
    if config.provider == "codex":
        port = config.codex.proxy_port
        label = "codex proxy"
    else:
        port = config.llama_server.port
        label = "llama-server"

    url = f"http://localhost:{port}/health"
    errors = []
    try:
        resp = urlopen(url, timeout=10)
        if resp.status != 200:
            errors.append(
                f"{label} health check failed (HTTP {resp.status}) on port {port}"
            )
    except (URLError, OSError) as e:
        errors.append(f"Cannot connect to {label} on port {port}: {e}")
    return errors


def build_openevolve_config_yaml(
    config: CodeEvolveConfig, output_dir: Path, frozen_context: str = "",
) -> Path:
    """Write an OpenEvolve-compatible config YAML and return its path."""
    oe_dict = config.to_openevolve_dict(frozen_context=frozen_context)
    yaml_path = output_dir / "openevolve_config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(oe_dict, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def format_iteration_line(
    iteration: int,
    total: int,
    file_changed: str = "",
    diff_lines: int = 0,
    build_ok: bool = False,
    build_time: float = 0.0,
    tests_ok: bool = False,
    tests_passed: int = 0,
    tests_failed: int = 0,
    clippy_warnings: int = 0,
    parent_clippy_warnings: Optional[int] = None,
    binary_size: int = 0,
    parent_binary_size: Optional[int] = None,
    loc: int = 0,
    parent_loc: Optional[int] = None,
    llm_ran: bool = False,
    llm_score: float = 0.0,
    score: float = 0.0,
    best_score: float = 0.0,
    error: str = "",
) -> str:
    """Format a single iteration's results for terminal display."""
    lines = []
    lines.append(f"-- Iteration {iteration}/{total} " + "-" * 40)
    lines.append(f"  Mutating {file_changed} ... generated {diff_lines}-line diff")

    if not build_ok:
        error_short = error.split("\n")[0][:80] if error else "compilation error"
        lines.append(f"  |- Build:    FAILED ({error_short})")
        lines.append(f"  '- Score:    0.00 (discarded)")
        return "\n".join(lines)

    lines.append(f"  |- Build:    pass, compiled ({build_time:.1f}s)")

    if not tests_ok:
        lines.append(f"  |- Tests:    FAILED ({tests_passed} passed, {tests_failed} failed)")
        lines.append(f"  '- Score:    0.00 (discarded)")
        return "\n".join(lines)

    lines.append(f"  |- Tests:    pass, {tests_passed}/{tests_passed + tests_failed} passed")

    clippy_delta = ""
    if parent_clippy_warnings is not None:
        if clippy_warnings < parent_clippy_warnings:
            clippy_delta = f" (was {parent_clippy_warnings}) - improved"
        elif clippy_warnings > parent_clippy_warnings:
            clippy_delta = f" (was {parent_clippy_warnings}) - regressed"
    lines.append(f"  |- Clippy:   {clippy_warnings} warnings{clippy_delta}")

    size_mb = binary_size / 1_000_000
    size_delta = ""
    if parent_binary_size is not None:
        parent_mb = parent_binary_size / 1_000_000
        if binary_size < parent_binary_size:
            size_delta = f" (was {parent_mb:.1f} MB) - improved"
        elif binary_size > parent_binary_size:
            size_delta = f" (was {parent_mb:.1f} MB) - regressed"
    lines.append(f"  |- Size:     {size_mb:.1f} MB{size_delta}")

    loc_delta = ""
    if parent_loc is not None:
        if loc < parent_loc:
            loc_delta = f" (was {parent_loc}) - improved"
        elif loc > parent_loc:
            loc_delta = f" (was {parent_loc}) - regressed"
    lines.append(f"  |- LoC:      {loc}{loc_delta}")

    if llm_ran:
        lines.append(f"  |- LLM:      {llm_score:.2f}")
    else:
        lines.append(f"  |- LLM:      skipped (not top quartile)")

    lines.append(f"  '- Score:    {score:.2f} (best so far: {best_score:.2f})")
    return "\n".join(lines)


def _patch_logging_utf8() -> None:
    """Force UTF-8 encoding for logging FileHandlers on Windows.

    OpenEvolve creates a ``FileHandler`` without specifying encoding.
    On Windows this defaults to the locale codec (cp1252) which cannot
    represent emojis used in OpenEvolve's log messages (e.g. U+1F31F).
    """
    if sys.platform != "win32":
        return
    _orig_fh_init = logging.FileHandler.__init__

    def _utf8_fh_init(self, filename, mode="a", encoding=None, delay=False,
                      errors=None):
        if encoding is None:
            encoding = "utf-8"
        _orig_fh_init(self, filename, mode, encoding, delay, errors)

    logging.FileHandler.__init__ = _utf8_fh_init


def _patch_feature_dimension_defaults() -> None:
    """Ensure feature dimension metrics always exist, even on timeout.

    OpenEvolve replaces metrics with ``{error, timeout}`` when an evaluation
    times out.  If the configured feature_dimensions reference evaluator
    metrics (e.g. ``llm_score``), the database crashes with ValueError.
    This patch injects 0.0 defaults for any missing feature dimensions
    before the coordinate calculation runs.
    """
    from openevolve import database as db_mod

    _original = db_mod.ProgramDatabase._calculate_feature_coords

    def _safe_feature_coords(self, program):
        for dim in self.config.feature_dimensions:
            if dim not in ("complexity", "diversity", "score"):
                program.metrics.setdefault(dim, 0.0)
        return _original(self, program)

    db_mod.ProgramDatabase._calculate_feature_coords = _safe_feature_coords


def _patch_extract_diffs() -> None:
    """Monkey-patch OpenEvolve's extract_diffs to normalise markdown diffs."""
    from openevolve.utils import code_utils

    _original = code_utils.extract_diffs

    def _patched(diff_text: str, diff_pattern: str = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE") -> List[Tuple[str, str]]:
        # Try the original pattern first.
        result = _original(diff_text, diff_pattern)
        if result:
            return result
        # Normalise markdown-style diffs and retry.
        normalised = _normalize_llm_diffs(diff_text)
        return _original(normalised, diff_pattern)

    code_utils.extract_diffs = _patched


def run_evolution(
    config_path: Path,
    project_path: Path,
    source_files: Path | list[Path],
    evaluator_path: Path,
):
    """Run the evolutionary loop via OpenEvolve. Returns EvolutionResult.

    Args:
        config_path: Path to evolution.yaml.
        project_path: Root of the Rust project.
        source_files: A single Path (legacy) or list of Paths with EVOLVE-BLOCK
            markers. When a single file is given, the existing single-file flow
            is used for backward compatibility. When multiple files are given,
            summaries are generated and a bundle is created with the first file
            as focus.
        evaluator_path: Path to the generated evaluator.py.
    """
    from openevolve.api import run_evolution as oe_run_evolution
    from codeevolve.evaluator.pipeline import parse_evolve_block, _MARKER_START, _MARKER_END

    _patch_extract_diffs()
    _patch_feature_dimension_defaults()
    _patch_logging_utf8()

    config = load_config(config_path)

    output_dir = project_path / ".codeevolve" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize to list
    if isinstance(source_files, Path):
        source_files = [source_files]

    # ---- Single-file backward-compatible path ----
    if len(source_files) == 1:
        return _run_single_file(
            config, config_path, project_path, source_files[0],
            evaluator_path, output_dir,
        )

    # ---- Multi-file bundle path ----
    return _run_multi_file(
        config, config_path, project_path, source_files,
        evaluator_path, output_dir,
    )


def _run_single_file(
    config: CodeEvolveConfig,
    config_path: Path,
    project_path: Path,
    initial_program: Path,
    evaluator_path: Path,
    output_dir: Path,
):
    """Single-file evolution flow (backward compatible)."""
    from openevolve.api import run_evolution as oe_run_evolution
    from codeevolve.evaluator.pipeline import parse_evolve_block, _MARKER_START, _MARKER_END

    # Save a backup of the original source before evolution starts.
    backup_path = output_dir / f"{initial_program.name}.backup"
    original_code = initial_program.read_text()
    backup_path.write_text(original_code)
    logger.info("Saved source backup to %s", backup_path)

    # Give OpenEvolve only the EVOLVE-BLOCK content so the LLM cannot
    # duplicate struct definitions, imports, or test modules that live
    # outside the evolvable region.
    frozen_context = ""
    evolve_program = initial_program
    parsed = parse_evolve_block(original_code)
    if parsed:
        prefix, evolve_content, suffix = parsed
        frozen_prefix = prefix.replace(_MARKER_START + "\n", "").strip()
        frozen_suffix = suffix.replace(_MARKER_END, "").strip()
        frozen_parts = [p for p in [frozen_prefix, frozen_suffix] if p]
        frozen_context = "\n\n".join(frozen_parts)

        evolve_program = output_dir / f"{initial_program.stem}_evolve_block.rs"
        evolve_program.write_text(evolve_content)
        logger.info(
            "Stripped initial program to EVOLVE-BLOCK only (%d chars, frozen=%d chars)",
            len(evolve_content), len(frozen_context),
        )

    oe_config_path = build_openevolve_config_yaml(
        config, output_dir, frozen_context=frozen_context,
    )

    try:
        result = oe_run_evolution(
            initial_program=str(evolve_program),
            evaluator=str(evaluator_path),
            config=str(oe_config_path),
            iterations=config.evolution.max_iterations,
            output_dir=str(output_dir),
            cleanup=False,
        )
    finally:
        # Restore the original source file so the project is never left
        # with a candidate's code after the run ends (success or crash).
        initial_program.write_text(backup_path.read_text())
        logger.info("Restored original source from backup")

    best_dir = output_dir / "best"
    best_dir.mkdir(exist_ok=True)
    if result.best_code:
        (best_dir / initial_program.name).write_text(result.best_code)

    return result


def _run_multi_file(
    config: CodeEvolveConfig,
    config_path: Path,
    project_path: Path,
    source_files: list[Path],
    evaluator_path: Path,
    output_dir: Path,
):
    """Multi-file evolution flow using bundles and summaries."""
    from openevolve.api import run_evolution as oe_run_evolution
    from codeevolve.summary import summarize_files
    from codeevolve.bundler import create_bundle

    # Backup ALL source files
    backups: dict[Path, Path] = {}
    for f in source_files:
        backup_path = output_dir / f"{f.name}.backup"
        backup_path.write_text(f.read_text())
        backups[f] = backup_path
    logger.info("Saved %d source backups", len(backups))

    # Generate summaries of all files for context
    summaries = summarize_files(source_files, project_path)
    logger.info("Generated summaries for %d files", len(summaries))

    # Create bundle with first file as focus (v1: no rotation)
    focus_file = source_files[0]
    bundle = create_bundle(focus_file, source_files, summaries, project_path)
    bundle_path = output_dir / "initial_bundle.rs"
    bundle_path.write_text(bundle)
    logger.info(
        "Created initial bundle (%d chars, focus=%s)",
        len(bundle), focus_file.name,
    )

    # For the bundle path, frozen context is not needed separately --
    # the bundle's CONTEXT section provides read-only context to the LLM.
    oe_config_path = build_openevolve_config_yaml(config, output_dir, frozen_context="")

    try:
        result = oe_run_evolution(
            initial_program=str(bundle_path),
            evaluator=str(evaluator_path),
            config=str(oe_config_path),
            iterations=config.evolution.max_iterations,
            output_dir=str(output_dir),
            cleanup=False,
        )
    finally:
        # Restore ALL original source files
        for f, backup in backups.items():
            f.write_text(backup.read_text())
        logger.info("Restored %d source files from backups", len(backups))

    best_dir = output_dir / "best"
    best_dir.mkdir(exist_ok=True)
    if result.best_code:
        (best_dir / focus_file.name).write_text(result.best_code)

    return result
