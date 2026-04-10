from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.request import urlopen, Request
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


def _ollama_base_url(api_base: str) -> str:
    """Derive the Ollama root URL from the configured api_base.

    Strips the ``/v1`` suffix so we can hit native endpoints like ``/api/tags``.
    """
    return api_base.rstrip("/").removesuffix("/v1")


def validate_ollama(config: CodeEvolveConfig) -> list[str]:
    """Check that Ollama is reachable and required models are available."""
    base = _ollama_base_url(config.ollama.api_base)
    errors = []
    try:
        resp = urlopen(f"{base}/api/tags", timeout=10)
        data = json.loads(resp.read())
        available = {m["name"] for m in data.get("models", [])}
        missing = []
        for model_name in [config.ollama.mutator_model, config.ollama.evaluator_model]:
            if model_name not in available:
                missing.append(model_name)
        if missing:
            for model_name in missing:
                errors.append(
                    f"Model '{model_name}' not found in Ollama. "
                    f"Pull it with: ollama pull {model_name}"
                )
            if available:
                errors.append(
                    f"Available models at {base}: "
                    + ", ".join(sorted(available))
                )
            else:
                errors.append(
                    f"No models found at {base}. "
                    "Is Ollama running with models pulled?"
                )
    except (URLError, OSError) as e:
        errors.append(f"Cannot connect to Ollama at {base}: {e}")
    return errors


def prime_ollama_models(config: CodeEvolveConfig) -> None:
    """Load Ollama models with the configured context size.

    Ollama defaults to 32K context which wastes VRAM on KV cache.
    Sending a request with num_ctx forces a reload at the right size,
    freeing VRAM for model layers so more runs on GPU.
    """
    base = _ollama_base_url(config.ollama.api_base)
    num_ctx = config.evolution.context_window

    for model_name in [config.ollama.mutator_model, config.ollama.evaluator_model]:
        payload = json.dumps({
            "model": model_name,
            "prompt": "",
            "options": {"num_ctx": num_ctx},
            "keep_alive": "10m",
        }).encode()
        try:
            req = Request(
                f"{base}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = urlopen(req, timeout=60)
            # Read the full streaming response to completion
            resp.read()
            logger.info("Loaded %s with num_ctx=%d", model_name, num_ctx)
        except (URLError, OSError) as e:
            logger.warning("Failed to prime %s: %s", model_name, e)


def build_openevolve_config_yaml(config: CodeEvolveConfig, output_dir: Path) -> Path:
    """Write an OpenEvolve-compatible config YAML and return its path."""
    oe_dict = config.to_openevolve_dict()
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
    initial_program: Path,
    evaluator_path: Path,
):
    """Run the evolutionary loop via OpenEvolve. Returns EvolutionResult."""
    from openevolve.api import run_evolution as oe_run_evolution

    _patch_extract_diffs()

    config = load_config(config_path)

    output_dir = project_path / ".codeevolve" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save a backup of the original source before evolution starts.
    # The evaluator overwrites the source file during each evaluation;
    # this backup ensures we can always restore even after a crash.
    backup_path = output_dir / f"{initial_program.name}.backup"
    backup_path.write_text(initial_program.read_text())
    logger.info("Saved source backup to %s", backup_path)

    oe_config_path = build_openevolve_config_yaml(config, output_dir)

    try:
        result = oe_run_evolution(
            initial_program=str(initial_program),
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
