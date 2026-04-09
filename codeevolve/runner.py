from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from openai import OpenAI

from codeevolve.config import CodeEvolveConfig, load_config


def validate_ollama(config: CodeEvolveConfig) -> list[str]:
    """Check that Ollama is reachable and required models are available."""
    errors = []
    try:
        client = OpenAI(base_url=config.ollama.api_base, api_key="ollama")
        models_response = client.models.list()
        available = {m.id for m in models_response.data}
        for model_name in [config.ollama.mutator_model, config.ollama.evaluator_model]:
            if model_name not in available:
                errors.append(
                    f"Model '{model_name}' not found in Ollama. "
                    f"Pull it with: ollama pull {model_name}"
                )
    except Exception as e:
        errors.append(f"Cannot connect to Ollama at {config.ollama.api_base}: {e}")
    return errors


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

    if llm_ran:
        lines.append(f"  |- LLM:      {llm_score:.2f}")
    else:
        lines.append(f"  |- LLM:      skipped (not top quartile)")

    lines.append(f"  '- Score:    {score:.2f} (best so far: {best_score:.2f})")
    return "\n".join(lines)


def run_evolution(
    config_path: Path,
    project_path: Path,
    initial_program: Path,
    evaluator_path: Path,
):
    """Run the evolutionary loop via OpenEvolve. Returns EvolutionResult."""
    from openevolve.api import run_evolution as oe_run_evolution

    config = load_config(config_path)

    output_dir = project_path / ".codeevolve" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    oe_config_path = build_openevolve_config_yaml(config, output_dir)

    result = oe_run_evolution(
        initial_program=str(initial_program),
        evaluator=str(evaluator_path),
        config=str(oe_config_path),
        iterations=config.evolution.max_iterations,
        output_dir=str(output_dir),
        cleanup=False,
    )

    best_dir = output_dir / "best"
    best_dir.mkdir(exist_ok=True)
    if result.best_code:
        (best_dir / initial_program.name).write_text(result.best_code)

    return result
