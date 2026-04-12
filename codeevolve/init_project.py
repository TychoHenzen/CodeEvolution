from __future__ import annotations

from pathlib import Path
from typing import Optional

import jinja2
import yaml

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_DEFAULTS_DIR = Path(__file__).parent / "defaults"


class _SingleQuotedStr(str):
    """Marker type so yaml.dump uses single-quoted style (backslash-safe)."""


class _ConfigDumper(yaml.Dumper):
    """Custom YAML dumper that single-quotes _SingleQuotedStr values."""


_ConfigDumper.add_representer(
    _SingleQuotedStr,
    lambda dumper, data: dumper.represent_scalar(
        "tag:yaml.org,2002:str", str(data), style="'"
    ),
)


def find_cargo_toml(project_path: Path) -> Path:
    """Find and validate Cargo.toml in the project directory."""
    cargo_toml = project_path / "Cargo.toml"
    if not cargo_toml.exists():
        raise FileNotFoundError(
            f"No Cargo.toml found in {project_path}. "
            "Run this command from a Rust project directory, or use --path."
        )
    return cargo_toml


def insert_evolve_markers(rs_file: Path) -> None:
    """Wrap file content in EVOLVE-BLOCK markers if not already present."""
    content = rs_file.read_text(encoding="utf-8")
    if "EVOLVE-BLOCK-START" in content:
        return
    wrapped = f"// EVOLVE-BLOCK-START\n{content}// EVOLVE-BLOCK-END\n"
    rs_file.write_text(wrapped, encoding="utf-8")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _single_quote_backslash_strings(data):
    """Wrap strings containing backslashes in _SingleQuotedStr for clean YAML output."""
    if isinstance(data, dict):
        return {k: _single_quote_backslash_strings(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_single_quote_backslash_strings(v) for v in data]
    if isinstance(data, str) and "\\" in data:
        return _SingleQuotedStr(data)
    return data


def sync_project_config(config_path: Path) -> list[str]:
    """Merge latest defaults into an existing project config.

    New keys from defaults are added; existing project values are preserved.
    Returns a list of top-level keys that were added.
    """
    with open(_DEFAULTS_DIR / "evolution.yaml") as f:
        defaults = yaml.safe_load(f)
    with open(config_path, encoding="utf-8") as f:
        project = yaml.safe_load(f) or {}

    added_keys = [k for k in defaults if k not in project]

    merged = _deep_merge(defaults, project)
    merged = _single_quote_backslash_strings(merged)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(merged, f, Dumper=_ConfigDumper, default_flow_style=False,
                  sort_keys=False, allow_unicode=True)

    return added_keys


def regenerate_evaluator(
    project_path: Path,
    config_path: Path,
    focus_file: Path,
) -> None:
    """Regenerate evaluator.py from the template without touching config or markers."""

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
        keep_trailing_newline=True,
    )
    template = env.get_template("evaluator.py.j2")
    codeevolve_package_path = str(Path(__file__).parent.parent.resolve())

    evaluator_code = template.render(
        project_name=project_path.name,
        codeevolve_package_path=codeevolve_package_path.replace("\\", "/"),
        config_path=str(config_path.resolve()).replace("\\", "/"),
        project_path=str(project_path.resolve()).replace("\\", "/"),
        focus_file=str(focus_file.resolve()).replace("\\", "/"),
    )
    (config_path.parent / "evaluator.py").write_text(evaluator_code)


def generate_codeevolve_dir(
    project_path: Path,
    rs_files: list[Path],
    custom_benchmark: Optional[str] = None,
    custom_benchmark_regex: Optional[str] = None,
    include_globs: Optional[list[str]] = None,
    exclude_globs: Optional[list[str]] = None,
) -> Path:
    """Generate the .codeevolve/ directory with config, evaluator, and README."""
    codeevolve_dir = project_path / ".codeevolve"
    codeevolve_dir.mkdir(exist_ok=True)

    # --- Generate evolution.yaml ---
    with open(_DEFAULTS_DIR / "evolution.yaml", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    if custom_benchmark:
        config_data["benchmarks"]["custom_command"] = custom_benchmark
    if custom_benchmark_regex:
        config_data["benchmarks"]["custom_command_score_regex"] = _SingleQuotedStr(
            custom_benchmark_regex
        )

    # Override globs if workspace detection provided them
    if include_globs is not None:
        config_data["include_globs"] = include_globs
    if exclude_globs is not None:
        config_data["exclude_globs"] = exclude_globs

    config_path = codeevolve_dir / "evolution.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, Dumper=_ConfigDumper, default_flow_style=False,
                  sort_keys=False, allow_unicode=True)

    # --- Generate evaluator.py ---
    regenerate_evaluator(project_path, config_path, focus_file=rs_files[0])

    # --- Generate README.md ---
    file_list = "\n".join(f"  - {f.relative_to(project_path)}" for f in rs_files)
    readme = f"""# CodeEvolution Setup

This directory was generated by `codeevolve init`. Here's what each file does:

## Files

- **evolution.yaml** — Configuration for the evolutionary optimizer. Controls which
  model to use, how many iterations to run, fitness weights, and more.
  Edit this to tune the evolution. All fields have sensible defaults.

- **evaluator.py** — The fitness function that scores each code candidate. This is
  called automatically by the optimizer. You generally don't need to edit this.

## Evolved Files

The following source files have been marked for evolution:
{file_list}

Each file has `// EVOLVE-BLOCK-START` and `// EVOLVE-BLOCK-END` markers around
the code that the optimizer is allowed to change. You can move these markers to
control which parts of your code get evolved.

## How to Run

1. Download the model GGUF if you haven't already
2. Set `server_path` and `model_path` in `evolution.yaml` to point to your
   llama-server binary and .gguf model file
3. Start the evolution: `codeevolve run`

The server is started and stopped automatically by `codeevolve run`.

The optimizer will try hundreds of variations of your code, keeping only the ones
that compile, pass tests, and score well on code quality metrics.

Results are saved to `.codeevolve/output/`. The best version is in `output/best/`.

## How It Works

Each candidate goes through 4 checks:
1. **Build + Tests** — Must compile and pass all tests (hard requirement)
2. **Clippy Analysis** — Fewer warnings = better score
3. **Performance** — Faster compile time and smaller binary = better score
4. **AI Review** — An AI model rates code quality (only for top candidates)

Press Ctrl+C at any time to stop. Your best result so far is always saved.
"""
    (codeevolve_dir / "README.md").write_text(readme)

    return codeevolve_dir
