from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from codeevolve.scheduler import ScheduleSlot

logger = logging.getLogger(__name__)


def _normalize_llm_diffs(text: str) -> str:
    """Rewrite common malformed diff formats into the canonical markers.

    LLMs frequently emit variations of the ``<<<<<<< SEARCH`` / ``=======`` /
    ``>>>>>>> REPLACE`` format that OpenEvolve expects.  This function
    normalises the most common deviations so that ``extract_diffs`` can parse
    them.

    Handled variants
    ----------------
    1. Outer backtick wrapping — entire response fenced in ```` ``` … ``` ````
    2. Inner backtick fences — ```` ```rust ```` / ```` ``` ```` immediately
       inside SEARCH/REPLACE blocks
    3. Markdown-heading markers — ``#### SEARCH`` … ``#### REPLACE`` with
       fenced code blocks
    """

    # --- Pass 1: strip outer backtick fence wrapping the whole response ------
    # e.g.  ```\n<<<<<<< SEARCH\n…\n>>>>>>> REPLACE\n```
    outer = re.match(r"^```\w*\n(.*)\n```\s*$", text, re.DOTALL)
    if outer and "<<<<<<< SEARCH" in outer.group(1):
        text = outer.group(1)

    # --- Pass 2: strip backtick fences immediately adjacent to markers -------
    # Handles LLMs that put ```rust / ``` inside the SEARCH/REPLACE blocks.
    lines = text.split("\n")
    filtered: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        is_fence = bool(re.match(r"^```\w*$", stripped))

        if is_fence:
            # Opening fence right after <<<<<<< SEARCH or =======
            prev = filtered[-1].strip() if filtered else ""
            if prev in ("<<<<<<< SEARCH", "======="):
                i += 1
                continue
            # Closing fence right before ======= or >>>>>>> REPLACE
            next_s = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if next_s in ("=======", ">>>>>>> REPLACE"):
                i += 1
                continue

        filtered.append(lines[i])
        i += 1
    text = "\n".join(filtered)

    # --- Pass 3: rewrite markdown-heading style diffs ------------------------
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
from codeevolve.init_project import regenerate_evaluator


def build_openevolve_config_yaml(
    config: CodeEvolveConfig, output_dir: Path, frozen_context: str = "",
) -> Path:
    """Write an OpenEvolve-compatible config YAML and return its path."""
    oe_dict = config.to_openevolve_dict(frozen_context=frozen_context)
    yaml_path = output_dir / "openevolve_config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(oe_dict, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def _clear_root_handlers() -> None:
    """Remove all handlers from the root logger.

    OpenEvolve's ``_setup_logging`` adds a ``FileHandler`` and a
    ``StreamHandler`` to the root logger each time a controller is
    created.  When the rotation loop creates a new controller per slot,
    handlers accumulate and every log line is printed N times by slot N.
    Calling this before each slot resets the root logger to a clean state.
    """
    root = logging.getLogger()
    for handler in root.handlers[:]:
        handler.close()
        root.removeHandler(handler)


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
    """Monkey-patch OpenEvolve's extract_diffs to normalise LLM diffs.

    Always normalises before extraction — backtick fences inside canonical
    markers produce regex matches whose content doesn't match the actual
    source, so "try original first" is not safe.
    """
    from openevolve.utils import code_utils

    _original = code_utils.extract_diffs

    def _patched(diff_text: str, diff_pattern: str = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE") -> List[Tuple[str, str]]:
        normalised = _normalize_llm_diffs(diff_text)
        return _original(normalised, diff_pattern)

    code_utils.extract_diffs = _patched


_patches_applied = False


def _apply_patches() -> None:
    """Apply all OpenEvolve monkey patches once."""
    global _patches_applied
    if _patches_applied:
        return
    _patch_extract_diffs()
    _patch_feature_dimension_defaults()
    _patch_logging_utf8()
    _patches_applied = True


def find_latest_checkpoint(output_dir: Path) -> str | None:
    """Return the path to the latest valid checkpoint directory, or None.

    Looks in ``output_dir / "checkpoints"`` for directories named
    ``checkpoint_N`` where N is an integer.  Sorts by N descending and
    returns the first one that contains a ``metadata.json`` file.

    Args:
        output_dir: The ``.codeevolve/output`` directory for the project.

    Returns:
        Absolute path string of the latest valid checkpoint, or ``None``.
    """
    checkpoints_dir = output_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        return None

    candidates: list[tuple[int, Path]] = []
    for entry in checkpoints_dir.iterdir():
        if not entry.is_dir():
            continue
        parts = entry.name.split("_")
        if len(parts) == 2 and parts[0] == "checkpoint" and parts[1].isdigit():
            candidates.append((int(parts[1]), entry))

    # Sort descending by iteration number
    candidates.sort(key=lambda x: x[0], reverse=True)

    for _n, path in candidates:
        if (path / "metadata.json").exists():
            return str(path)

    return None


def run_evolution(
    config_path: Path,
    project_path: Path,
    source_files: list[Path],
    evaluator_path: Path,
    checkpoint_path: str | None = None,
):
    """Run the evolutionary loop via OpenEvolve. Returns EvolutionResult.

    Args:
        config_path: Path to evolution.yaml.
        project_path: Root of the Rust project.
        source_files: List of Paths with EVOLVE-BLOCK markers. When a single
            file is given, the single-file flow is used. When multiple files
            are given, summaries are generated and a bundle is created.
        evaluator_path: Path to the generated evaluator.py.
        checkpoint_path: Optional path to a checkpoint directory to resume from.
    """
    from codeevolve.evaluator.pipeline import parse_evolve_block, _MARKER_START, _MARKER_END

    _apply_patches()

    config = load_config(config_path)

    output_dir = project_path / ".codeevolve" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(source_files) == 1:
        return _run_single_file(
            config, project_path, source_files[0],
            evaluator_path, output_dir,
            checkpoint_path=checkpoint_path,
        )

    return _run_multi_file(
        config, project_path, source_files,
        evaluator_path, output_dir,
        checkpoint_path=checkpoint_path,
    )


def _run_single_file(
    config: CodeEvolveConfig,
    project_path: Path,
    initial_program: Path,
    evaluator_path: Path,
    output_dir: Path,
    checkpoint_path: str | None = None,
    owns_backup: bool = True,
):
    """Single-file evolution flow.

    Args:
        owns_backup: If True (default), back up and restore the source file.
            Set to False when the caller (e.g. run_evolution_with_rotation)
            manages backups externally.
    """
    from openevolve.controller import OpenEvolve
    from openevolve.config import load_config as oe_load_config
    from openevolve.api import EvolutionResult
    from codeevolve.evaluator.pipeline import parse_evolve_block, _MARKER_START, _MARKER_END

    original_code = initial_program.read_text(encoding="utf-8")
    backup_path = None
    if owns_backup:
        backup_path = output_dir / f"{initial_program.name}.backup"
        backup_path.write_text(original_code, encoding="utf-8")
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
        evolve_program.write_text(evolve_content, encoding="utf-8")
        logger.info(
            "Stripped initial program to EVOLVE-BLOCK only (%d chars, frozen=%d chars)",
            len(evolve_content), len(frozen_context),
        )

    oe_config_path = build_openevolve_config_yaml(
        config, output_dir, frozen_context=frozen_context,
    )

    # Load OE config and create controller directly (instead of via
    # openevolve.api.run_evolution) so we can pass checkpoint_path.
    oe_config = oe_load_config(str(oe_config_path))

    # Auto-disable cascade evaluation if the evaluator lacks stage functions
    if oe_config.evaluator.cascade_evaluation:
        with open(str(evaluator_path), "r") as f:
            if "evaluate_stage1" not in f.read():
                oe_config.evaluator.cascade_evaluation = False

    try:
        controller = OpenEvolve(
            initial_program_path=str(evolve_program),
            evaluation_file=str(evaluator_path),
            config=oe_config,
            output_dir=str(output_dir),
        )

        best_program = asyncio.run(controller.run(
            iterations=config.evolution.max_iterations,
            checkpoint_path=checkpoint_path,
        ))

        # Always save a final checkpoint so resume can pick up from here
        if hasattr(controller, '_save_checkpoint') and hasattr(controller, 'database'):
            try:
                controller._save_checkpoint(controller.database.last_iteration)
                logger.info("Saved final checkpoint at iteration %d", controller.database.last_iteration)
            except Exception:
                logger.warning("Failed to save final checkpoint", exc_info=True)

        # Build result
        best_score = 0.0
        metrics: dict = {}
        best_code = ""

        if best_program:
            best_code = best_program.code
            metrics = best_program.metrics or {}
            if "combined_score" in metrics:
                best_score = metrics["combined_score"]
            elif metrics:
                numeric_metrics = [v for v in metrics.values() if isinstance(v, (int, float))]
                if numeric_metrics:
                    best_score = sum(numeric_metrics) / len(numeric_metrics)

        result = EvolutionResult(
            best_program=best_program,
            best_score=best_score,
            best_code=best_code,
            metrics=metrics,
            output_dir=str(output_dir),
        )
    finally:
        if owns_backup and backup_path is not None:
            initial_program.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
            logger.info("Restored original source from backup")

    best_dir = output_dir / "best"
    best_dir.mkdir(exist_ok=True)
    if result.best_code:
        (best_dir / initial_program.name).write_text(result.best_code, encoding="utf-8")

    return result


def _run_multi_file(
    config: CodeEvolveConfig,
    project_path: Path,
    source_files: list[Path],
    evaluator_path: Path,
    output_dir: Path,
    checkpoint_path: str | None = None,
):
    """Multi-file evolution flow using bundles and summaries."""
    from openevolve.controller import OpenEvolve
    from openevolve.config import load_config as oe_load_config
    from openevolve.api import EvolutionResult
    from codeevolve.summary import summarize_files
    from codeevolve.bundler import create_bundle, create_workspace_bundle, extract_focus
    from codeevolve.crate_graph import detect_workspace

    # Backup ALL source files (use relative path as name to avoid collisions
    # when multiple crates have files with the same basename like lib.rs)
    backups: dict[Path, Path] = {}
    for f in source_files:
        rel = f.relative_to(project_path)
        backup_name = rel.as_posix().replace("/", "__") + ".backup"
        backup_path = output_dir / backup_name
        backup_path.write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
        backups[f] = backup_path
    logger.info("Saved %d source backups", len(backups))

    # Generate summaries of all files for context
    summaries = summarize_files(source_files, project_path)
    logger.info("Generated summaries for %d files", len(summaries))

    # Detect workspace for smart context scoping
    workspace_info = detect_workspace(project_path)

    # Create bundle with first file as focus (v1: no rotation)
    focus_file = source_files[0]
    if workspace_info is not None:
        bundle = create_workspace_bundle(
            focus_file, source_files, summaries, project_path,
            workspace_info.crate_graph,
        )
        logger.info("Created workspace bundle (focus=%s)", focus_file.name)
    else:
        bundle = create_bundle(focus_file, source_files, summaries, project_path)
        logger.info("Created bundle (focus=%s)", focus_file.name)

    bundle_path = output_dir / "initial_bundle.rs"
    bundle_path.write_text(bundle, encoding="utf-8")
    logger.info("Bundle written (%d chars)", len(bundle))

    # For the bundle path, frozen context is not needed separately --
    # the bundle's CONTEXT section provides read-only context to the LLM.
    oe_config_path = build_openevolve_config_yaml(config, output_dir, frozen_context="")

    # Load OE config and create controller directly (instead of via
    # openevolve.api.run_evolution) so we can pass checkpoint_path.
    oe_config = oe_load_config(str(oe_config_path))

    # Auto-disable cascade evaluation if the evaluator lacks stage functions
    if oe_config.evaluator.cascade_evaluation:
        with open(str(evaluator_path), "r") as f:
            if "evaluate_stage1" not in f.read():
                oe_config.evaluator.cascade_evaluation = False

    try:
        controller = OpenEvolve(
            initial_program_path=str(bundle_path),
            evaluation_file=str(evaluator_path),
            config=oe_config,
            output_dir=str(output_dir),
        )

        best_program = asyncio.run(controller.run(
            iterations=config.evolution.max_iterations,
            checkpoint_path=checkpoint_path,
        ))

        # Always save a final checkpoint so resume can pick up from here
        if hasattr(controller, '_save_checkpoint') and hasattr(controller, 'database'):
            try:
                controller._save_checkpoint(controller.database.last_iteration)
                logger.info("Saved final checkpoint at iteration %d", controller.database.last_iteration)
            except Exception:
                logger.warning("Failed to save final checkpoint", exc_info=True)

        # Build result
        best_score = 0.0
        metrics: dict = {}
        best_code = ""

        if best_program:
            best_code = best_program.code
            metrics = best_program.metrics or {}
            if "combined_score" in metrics:
                best_score = metrics["combined_score"]
            elif metrics:
                numeric_metrics = [v for v in metrics.values() if isinstance(v, (int, float))]
                if numeric_metrics:
                    best_score = sum(numeric_metrics) / len(numeric_metrics)

        result = EvolutionResult(
            best_program=best_program,
            best_score=best_score,
            best_code=best_code,
            metrics=metrics,
            output_dir=str(output_dir),
        )
    finally:
        # Restore ALL original source files
        for f, backup in backups.items():
            f.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info("Restored %d source files from backups", len(backups))

    best_dir = output_dir / "best"
    best_dir.mkdir(exist_ok=True)
    if result.best_code:
        # Extract focus content from bundle format if needed
        focus_content = extract_focus(result.best_code)
        best_content = focus_content if focus_content else result.best_code
        (best_dir / focus_file.name).write_text(best_content, encoding="utf-8")

    return result


def run_evolution_with_rotation(
    config_path: Path,
    project_path: Path,
    schedule: list[ScheduleSlot],
    all_source_files: list[Path],
    evaluator_path: Path,
    checkpoint_path: str | None = None,
) -> dict[str, EvolutionResult]:
    """Run evolution across multiple files following a rotation schedule.

    Each ScheduleSlot assigns a file a contiguous block of iterations.
    Each slot creates a fresh OpenEvolve population for that file.
    Results and checkpoints are saved per-slot.

    Args:
        config_path: Path to evolution.yaml
        project_path: Root of the Rust project
        schedule: List of ScheduleSlot from build_schedule()
        all_source_files: All marked files (for backup/restore and bundling context)
        evaluator_path: Path to evaluator.py
        checkpoint_path: Path to checkpoint to resume from (used to find rotation
            state only; each slot starts a fresh OpenEvolve run)

    Returns:
        Dict mapping file_path to the best EvolutionResult for that file
    """
    from openevolve.api import EvolutionResult

    _apply_patches()

    config = load_config(config_path)

    output_dir = project_path / ".codeevolve" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine starting slot from rotation state
    rotation_state_path = output_dir / "rotation_state.json"
    start_slot_index = 0

    if checkpoint_path is not None and rotation_state_path.exists():
        try:
            state = json.loads(rotation_state_path.read_text(encoding="utf-8"))
            start_slot_index = state.get("current_slot_index", 0)
            logger.info("Resuming rotation from slot %d", start_slot_index)
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read rotation state, starting from slot 0")

    # Backup ALL source files once at the start
    backups: dict[Path, Path] = {}
    for f in all_source_files:
        rel = f.relative_to(project_path)
        backup_name = rel.as_posix().replace("/", "__") + ".backup"
        backup_path = output_dir / backup_name
        backup_path.write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
        backups[f] = backup_path
    logger.info("Saved %d source backups for rotation", len(backups))

    results: dict[str, EvolutionResult] = {}
    best_dir = output_dir / "best"
    best_dir.mkdir(exist_ok=True)

    try:
        for i, slot in enumerate(schedule):
            if i < start_slot_index:
                logger.info("Skipping completed slot %d/%d (%s)", i + 1, len(schedule), slot.file_path)
                continue

            slot_iterations = slot.end_iter - slot.start_iter
            logger.info(
                "  [Slot %d/%d] Evolving %s (iterations %d-%d)",
                i + 1, len(schedule), slot.file_path, slot.start_iter, slot.end_iter,
            )

            # Find the actual Path object for slot.file_path from all_source_files
            source_file = None
            for f in all_source_files:
                rel = f.relative_to(project_path)
                if rel.as_posix() == slot.file_path or str(rel) == slot.file_path:
                    source_file = f
                    break

            if source_file is None:
                logger.warning("Could not find source file for slot: %s, skipping", slot.file_path)
                continue

            # Create a per-slot output directory
            slot_output_dir = output_dir / f"slot_{i}"
            slot_output_dir.mkdir(parents=True, exist_ok=True)

            # Create a copy of config with modified max_iterations for this slot
            slot_config = copy.deepcopy(config)
            slot_config.evolution.max_iterations = slot_iterations

            # Regenerate evaluator.py so the pipeline focus file matches the
            # source file assigned to this slot.
            regenerate_evaluator(project_path, config_path, focus_file=source_file)

            # Clear root logger handlers to prevent duplication.
            # OpenEvolve's _setup_logging() appends new handlers without
            # removing old ones, so each slot would otherwise add another
            # pair (file + console), causing N-fold log duplication.
            _clear_root_handlers()

            result = _run_single_file(
                slot_config,
                project_path,
                source_file,
                evaluator_path,
                slot_output_dir,
                checkpoint_path=None,  # each slot is a fresh OE run
                owns_backup=False,  # rotation manages backups externally
            )

            results[slot.file_path] = result

            # Save best result to output/best/
            if result.best_code:
                (best_dir / source_file.name).write_text(result.best_code, encoding="utf-8")

            # Save rotation state after each slot
            rotation_state = {
                "current_slot_index": i + 1,
                "schedule": [
                    {
                        "file_path": s.file_path,
                        "start_iter": s.start_iter,
                        "end_iter": s.end_iter,
                    }
                    for s in schedule
                ],
            }
            rotation_state_path.write_text(
                json.dumps(rotation_state, indent=2),
                encoding="utf-8",
            )
            logger.info("Saved rotation state: slot %d/%d complete", i + 1, len(schedule))
    finally:
        # Restore ALL original source files
        for f, backup in backups.items():
            try:
                f.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
            except OSError:
                logger.warning("Failed to restore backup for %s", f)
        logger.info("Restored %d source files from backups", len(backups))

    return results
