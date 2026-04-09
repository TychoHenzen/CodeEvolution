# CodeEvolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool that wraps OpenEvolve to provide batteries-included evolutionary code optimization for Rust projects.

**Architecture:** Thin wrapper over OpenEvolve. `codeevolve init` scans a Rust project and generates config + evaluator files. `codeevolve run` calls OpenEvolve's `run_evolution()` API. The core value-add is a 4-layer gated Rust evaluation pipeline (build/test -> Clippy -> benchmarks -> LLM judgment).

**Tech Stack:** Python 3.13, Click (CLI), OpenEvolve (evolutionary engine), Jinja2 (templates), PyYAML (config), openai (Ollama client)

**Spec:** `docs/superpowers/specs/2026-04-09-codeevolution-design.md`

---

## File Map

```
codeevolve/
├── __init__.py              # Package version
├── cli.py                   # Click CLI: init + run commands
├── config.py                # Config dataclass + YAML loading
├── init_project.py          # Project scanning + file generation
├── runner.py                # OpenEvolve integration + progress display
├── evaluator/
│   ├── __init__.py          # Re-exports evaluate()
│   ├── pipeline.py          # 4-layer gated evaluation orchestrator
│   ├── cargo.py             # cargo build, test, clippy, timings, binary size
│   ├── benchmark.py         # Optional user benchmark command
│   └── llm_judge.py         # Ollama 1.5B quality judgment
├── templates/
│   └── evaluator.py.j2     # Jinja2 template for generated evaluator.py
└── defaults/
    └── evolution.yaml       # Default config tuned for Rust + Ollama

tests/
├── conftest.py              # Shared fixtures (temp Rust projects, mock Ollama)
├── test_config.py
├── test_cargo.py
├── test_benchmark.py
├── test_llm_judge.py
├── test_pipeline.py
├── test_init_project.py
├── test_runner.py
├── test_cli.py
└── fixtures/
    ├── sample_crate/        # Minimal Rust crate for testing
    │   ├── Cargo.toml
    │   └── src/
    │       └── lib.rs
    └── clippy_output.json   # Captured clippy JSON for parsing tests
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `codeevolve/__init__.py`
- Create: `codeevolve/evaluator/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/fixtures/sample_crate/Cargo.toml`
- Create: `tests/fixtures/sample_crate/src/lib.rs`

- [ ] **Step 1: Initialize git repo**

```bash
cd /mnt/d/Ollama/CodeEvolution
git init
```

- [ ] **Step 2: Create .gitignore**

```gitignore
__pycache__/
*.pyc
.venv/
*.egg-info/
dist/
build/
.codeevolve/output/
.idea/
target/
```

- [ ] **Step 3: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=75.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "codeevolve"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "click>=8.1",
    "openevolve>=0.2.27",
    "jinja2>=3.1",
    "pyyaml>=6.0",
    "openai>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-mock>=3.14",
]

[project.scripts]
codeevolve = "codeevolve.cli:main"
```

- [ ] **Step 4: Create package init files**

`codeevolve/__init__.py`:
```python
__version__ = "0.1.0"
```

`codeevolve/evaluator/__init__.py`:
```python
```

`tests/__init__.py`:
```python
```

- [ ] **Step 5: Create test fixtures — minimal Rust crate**

`tests/fixtures/sample_crate/Cargo.toml`:
```toml
[package]
name = "sample_crate"
version = "0.1.0"
edition = "2021"
```

`tests/fixtures/sample_crate/src/lib.rs`:
```rust
// EVOLVE-BLOCK-START
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
// EVOLVE-BLOCK-END

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
    }
}
```

- [ ] **Step 6: Create conftest.py with shared fixtures**

`tests/conftest.py`:
```python
import shutil
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_crate(tmp_path: Path) -> Path:
    """Copy the sample Rust crate to a temp directory so tests can modify it."""
    dest = tmp_path / "sample_crate"
    shutil.copytree(FIXTURES_DIR / "sample_crate", dest)
    return dest


@pytest.fixture
def clippy_output_json() -> str:
    """Captured clippy JSON output for parsing tests."""
    return (FIXTURES_DIR / "clippy_output.json").read_text()
```

- [ ] **Step 7: Create clippy fixture**

`tests/fixtures/clippy_output.json`:
```json
[
  {
    "reason": "compiler-message",
    "message": {
      "code": {"code": "clippy::needless_return", "explanation": null},
      "level": "warning",
      "message": "unneeded `return` statement",
      "spans": [{"file_name": "src/lib.rs", "line_start": 3, "line_end": 3}]
    }
  },
  {
    "reason": "compiler-message",
    "message": {
      "code": {"code": "clippy::cast_possible_truncation", "explanation": null},
      "level": "warning",
      "message": "casting `i64` to `i32` may truncate the value",
      "spans": [{"file_name": "src/lib.rs", "line_start": 7, "line_end": 7}]
    }
  },
  {
    "reason": "compiler-message",
    "message": {
      "code": {"code": "clippy::unwrap_used", "explanation": null},
      "level": "warning",
      "message": "used `unwrap()` on a `Result` value",
      "spans": [{"file_name": "src/lib.rs", "line_start": 12, "line_end": 12}]
    }
  }
]
```

- [ ] **Step 8: Install in dev mode and verify**

```bash
cd /mnt/d/Ollama/CodeEvolution
source .venv/bin/activate  # or .venv/Scripts/activate on Windows
pip install -e ".[dev]"
pytest --co  # collect tests, expect 0 collected (no test files yet)
```

- [ ] **Step 9: Commit**

```bash
git add .gitignore pyproject.toml codeevolve/ tests/
git commit -m "feat: project scaffolding with dependencies and test fixtures"
```

---

### Task 2: Config Module

**Files:**
- Create: `codeevolve/config.py`
- Create: `codeevolve/defaults/evolution.yaml`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for config loading**

`tests/test_config.py`:
```python
from pathlib import Path

import pytest

from codeevolve.config import CodeEvolveConfig, load_config


def test_load_default_config():
    """Loading with no path returns defaults."""
    config = load_config()
    assert config.ollama.api_base == "http://localhost:11434/v1"
    assert config.ollama.mutator_model == "qwen2.5-coder:7b-instruct-q4_K_M"
    assert config.ollama.evaluator_model == "qwen2.5-coder:1.5b-instruct-q4_K_M"
    assert config.evolution.max_iterations == 500
    assert config.fitness.static_analysis_weight == 0.35


def test_load_config_from_yaml(tmp_path: Path):
    """Loading from a YAML file overrides defaults."""
    yaml_content = """
ollama:
  api_base: "http://custom:1234/v1"
evolution:
  max_iterations: 100
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)

    config = load_config(config_path)
    assert config.ollama.api_base == "http://custom:1234/v1"
    assert config.evolution.max_iterations == 100
    # non-overridden fields keep defaults
    assert config.ollama.mutator_model == "qwen2.5-coder:7b-instruct-q4_K_M"


def test_load_config_missing_file():
    """Loading a nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.yaml"))


def test_config_clippy_weights_default():
    config = load_config()
    assert config.fitness.clippy_weights.correctness == 5
    assert config.fitness.clippy_weights.suspicious == 3
    assert config.fitness.clippy_weights.complexity == 2
    assert config.fitness.clippy_weights.perf == 2
    assert config.fitness.clippy_weights.style == 1


def test_config_to_openevolve_dict():
    """Config converts to an OpenEvolve-compatible dict."""
    config = load_config()
    oe_dict = config.to_openevolve_dict()
    assert oe_dict["max_iterations"] == 500
    assert oe_dict["diff_based_evolution"] is True
    assert oe_dict["llm"]["api_base"] == "http://localhost:11434/v1"
    assert oe_dict["llm"]["models"][0]["name"] == "qwen2.5-coder:7b-instruct-q4_K_M"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'codeevolve.config'`

- [ ] **Step 3: Create the default evolution.yaml**

`codeevolve/defaults/evolution.yaml`:
```yaml
# CodeEvolution default configuration
# All fields have sensible defaults for Rust + Ollama on RTX 4060 (8GB VRAM)

ollama:
  api_base: "http://localhost:11434/v1"
  mutator_model: "qwen2.5-coder:7b-instruct-q4_K_M"
  evaluator_model: "qwen2.5-coder:1.5b-instruct-q4_K_M"

evolution:
  max_iterations: 500
  population_size: 100
  num_islands: 3
  migration_interval: 20
  context_window: 4096
  diff_based_evolution: true

rust:
  cargo_path: "cargo"
  target_dir: null
  test_args: []
  clippy_args: []

fitness:
  static_analysis_weight: 0.35
  performance_weight: 0.35
  llm_judgment_weight: 0.30
  clippy_weights:
    correctness: 5
    suspicious: 3
    complexity: 2
    perf: 2
    style: 1

benchmarks:
  measure_compile_time: true
  measure_binary_size: true
  custom_command: null
  custom_command_score_regex: null

llm_judgment:
  enabled: true
  top_quartile_only: true
  num_runs: 3
  dimensions:
    - readability
    - rust_idiomaticity
    - maintainability
    - design
```

- [ ] **Step 4: Implement config.py**

`codeevolve/config.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

_DEFAULTS_DIR = Path(__file__).parent / "defaults"


@dataclass
class OllamaConfig:
    api_base: str = "http://localhost:11434/v1"
    mutator_model: str = "qwen2.5-coder:7b-instruct-q4_K_M"
    evaluator_model: str = "qwen2.5-coder:1.5b-instruct-q4_K_M"


@dataclass
class EvolutionConfig:
    max_iterations: int = 500
    population_size: int = 100
    num_islands: int = 3
    migration_interval: int = 20
    context_window: int = 4096
    diff_based_evolution: bool = True


@dataclass
class RustConfig:
    cargo_path: str = "cargo"
    target_dir: Optional[str] = None
    test_args: list[str] = field(default_factory=list)
    clippy_args: list[str] = field(default_factory=list)


@dataclass
class ClippyWeights:
    correctness: int = 5
    suspicious: int = 3
    complexity: int = 2
    perf: int = 2
    style: int = 1


@dataclass
class FitnessConfig:
    static_analysis_weight: float = 0.35
    performance_weight: float = 0.35
    llm_judgment_weight: float = 0.30
    clippy_weights: ClippyWeights = field(default_factory=ClippyWeights)


@dataclass
class BenchmarksConfig:
    measure_compile_time: bool = True
    measure_binary_size: bool = True
    custom_command: Optional[str] = None
    custom_command_score_regex: Optional[str] = None


@dataclass
class LlmJudgmentConfig:
    enabled: bool = True
    top_quartile_only: bool = True
    num_runs: int = 3
    dimensions: list[str] = field(
        default_factory=lambda: ["readability", "rust_idiomaticity", "maintainability", "design"]
    )


@dataclass
class CodeEvolveConfig:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    rust: RustConfig = field(default_factory=RustConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    benchmarks: BenchmarksConfig = field(default_factory=BenchmarksConfig)
    llm_judgment: LlmJudgmentConfig = field(default_factory=LlmJudgmentConfig)

    def to_openevolve_dict(self) -> dict:
        """Convert to a dict compatible with OpenEvolve's Config.from_dict()."""
        return {
            "max_iterations": self.evolution.max_iterations,
            "diff_based_evolution": self.evolution.diff_based_evolution,
            "file_suffix": ".rs",
            "llm": {
                "api_base": self.ollama.api_base,
                "api_key": "ollama",
                "models": [
                    {"name": self.ollama.mutator_model, "weight": 1.0},
                ],
                "temperature": 0.7,
                "max_tokens": self.evolution.context_window,
            },
            "database": {
                "population_size": self.evolution.population_size,
                "num_islands": self.evolution.num_islands,
                "migration_interval": self.evolution.migration_interval,
            },
            "evaluator": {
                "timeout": 120,
                "parallel_evaluations": 1,
            },
        }


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dict_to_config(data: dict) -> CodeEvolveConfig:
    """Build a CodeEvolveConfig from a flat dict (parsed YAML)."""
    ollama = OllamaConfig(**data.get("ollama", {}))
    evolution = EvolutionConfig(**data.get("evolution", {}))
    rust_data = data.get("rust", {})
    rust_data.pop(None, None)  # remove null target_dir key if present
    rust = RustConfig(**{k: v for k, v in rust_data.items() if v is not None or k != "target_dir"})
    fitness_data = data.get("fitness", {})
    clippy_data = fitness_data.pop("clippy_weights", {})
    clippy_weights = ClippyWeights(**clippy_data)
    fitness = FitnessConfig(**fitness_data, clippy_weights=clippy_weights)
    benchmarks_data = data.get("benchmarks", {})
    benchmarks = BenchmarksConfig(**{k: v for k, v in benchmarks_data.items()})
    llm_judgment = LlmJudgmentConfig(**data.get("llm_judgment", {}))
    return CodeEvolveConfig(
        ollama=ollama,
        evolution=evolution,
        rust=rust,
        fitness=fitness,
        benchmarks=benchmarks,
        llm_judgment=llm_judgment,
    )


def load_config(path: Optional[Path] = None) -> CodeEvolveConfig:
    """Load config from YAML, merging with defaults. No path = pure defaults."""
    defaults_path = _DEFAULTS_DIR / "evolution.yaml"
    with open(defaults_path) as f:
        base_data = yaml.safe_load(f)

    if path is None:
        return _dict_to_config(base_data)

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        override_data = yaml.safe_load(f) or {}

    merged = _deep_merge(base_data, override_data)
    return _dict_to_config(merged)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add codeevolve/config.py codeevolve/defaults/ tests/test_config.py
git commit -m "feat: config module with YAML loading and OpenEvolve conversion"
```

---

### Task 3: Cargo Integration (Layers 1 & 2)

**Files:**
- Create: `codeevolve/evaluator/cargo.py`
- Create: `tests/test_cargo.py`

- [ ] **Step 1: Write failing tests for cargo operations**

`tests/test_cargo.py`:
```python
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codeevolve.evaluator.cargo import (
    CargoResult,
    parse_clippy_json,
    run_cargo_build,
    run_cargo_clippy,
    run_cargo_test,
    categorize_lint,
    compute_clippy_score,
)
from codeevolve.config import ClippyWeights


# --- Clippy JSON parsing ---

def test_parse_clippy_json(clippy_output_json: str):
    warnings = parse_clippy_json(clippy_output_json)
    assert len(warnings) == 3
    assert warnings[0]["code"] == "clippy::needless_return"
    assert warnings[1]["code"] == "clippy::cast_possible_truncation"
    assert warnings[2]["code"] == "clippy::unwrap_used"


def test_parse_clippy_json_empty():
    warnings = parse_clippy_json("[]")
    assert warnings == []


# --- Lint categorization ---

def test_categorize_lint_style():
    assert categorize_lint("clippy::needless_return") == "style"


def test_categorize_lint_correctness():
    assert categorize_lint("clippy::wrong_self_convention") == "correctness"


def test_categorize_lint_suspicious():
    assert categorize_lint("clippy::cast_possible_truncation") == "suspicious"


def test_categorize_lint_unknown_defaults_to_style():
    assert categorize_lint("clippy::some_unknown_lint") == "style"


# --- Clippy score computation ---

def test_compute_clippy_score_no_warnings():
    score = compute_clippy_score({}, ClippyWeights())
    assert score == 0


def test_compute_clippy_score_weighted():
    counts = {"correctness": 1, "suspicious": 2, "style": 3}
    weights = ClippyWeights()
    # -(5*1 + 3*2 + 1*3) = -(5 + 6 + 3) = -14
    assert compute_clippy_score(counts, weights) == -14


# --- Cargo subprocess calls (using real cargo if available) ---

def test_run_cargo_build_success(sample_crate: Path):
    result = run_cargo_build(sample_crate)
    assert result.success is True
    assert result.elapsed_seconds > 0


def test_run_cargo_build_failure(tmp_path: Path):
    bad_crate = tmp_path / "bad"
    bad_crate.mkdir()
    (bad_crate / "Cargo.toml").write_text('[package]\nname = "bad"\nversion = "0.1.0"\nedition = "2021"')
    (bad_crate / "src").mkdir()
    (bad_crate / "src" / "lib.rs").write_text("fn broken( {}")  # syntax error
    result = run_cargo_build(bad_crate)
    assert result.success is False
    assert result.error_output != ""


def test_run_cargo_test_success(sample_crate: Path):
    run_cargo_build(sample_crate)  # must build first
    result = run_cargo_test(sample_crate)
    assert result.success is True
    assert result.tests_passed >= 1
    assert result.tests_failed == 0


def test_run_cargo_clippy_returns_warnings(sample_crate: Path):
    result = run_cargo_clippy(sample_crate)
    assert result.success is True
    assert isinstance(result.warnings, list)
    # sample_crate is clean, so few/no warnings expected
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cargo.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'codeevolve.evaluator.cargo'`

- [ ] **Step 3: Implement cargo.py**

`codeevolve/evaluator/cargo.py`:
```python
from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from codeevolve.config import ClippyWeights

# Mapping of known clippy lint prefixes to categories.
# Clippy organizes lints into groups: correctness, suspicious, complexity, perf, style, etc.
# See: https://rust-lang.github.io/rust-clippy/master/index.html
_LINT_CATEGORIES = {
    # correctness (deny-by-default, serious bugs)
    "approx_constant": "correctness",
    "wrong_self_convention": "correctness",
    "invalid_regex": "correctness",
    "erasing_op": "correctness",
    "if_let_mutex": "correctness",
    "derive_ord_xor_partial_ord": "correctness",
    "enum_clike_unportable_variant": "correctness",
    "unit_cmp": "correctness",
    "not_unsafe_ptr_arg_deref": "correctness",
    # suspicious
    "cast_possible_truncation": "suspicious",
    "cast_sign_loss": "suspicious",
    "cast_possible_wrap": "suspicious",
    "unwrap_used": "suspicious",
    "expect_used": "suspicious",
    "float_cmp": "suspicious",
    "mut_mut": "suspicious",
    # complexity
    "too_many_arguments": "complexity",
    "type_complexity": "complexity",
    "cognitive_complexity": "complexity",
    "option_option": "complexity",
    "collapsible_if": "complexity",
    "collapsible_else_if": "complexity",
    # perf
    "large_enum_variant": "perf",
    "box_collection": "perf",
    "redundant_clone": "perf",
    "unnecessary_to_owned": "perf",
    "manual_memcpy": "perf",
    # style (default for unknown lints)
    "needless_return": "style",
    "let_and_return": "style",
    "redundant_field_names": "style",
    "match_bool": "style",
    "single_match": "style",
}


def categorize_lint(lint_code: str) -> str:
    """Map a clippy lint code like 'clippy::needless_return' to a category."""
    name = lint_code.removeprefix("clippy::")
    return _LINT_CATEGORIES.get(name, "style")


@dataclass
class CargoResult:
    success: bool
    elapsed_seconds: float = 0.0
    error_output: str = ""
    # cargo test specific
    tests_passed: int = 0
    tests_failed: int = 0
    # cargo clippy specific
    warnings: list[dict] = field(default_factory=list)
    warning_counts: dict[str, int] = field(default_factory=dict)


def run_cargo_build(
    project_path: Path, cargo_path: str = "cargo", target_dir: Optional[str] = None
) -> CargoResult:
    """Run cargo build and return success/failure with timing."""
    cmd = [cargo_path, "build"]
    if target_dir:
        cmd.extend(["--target-dir", target_dir])

    start = time.monotonic()
    proc = subprocess.run(cmd, cwd=project_path, capture_output=True, text=True, timeout=120)
    elapsed = time.monotonic() - start

    return CargoResult(
        success=proc.returncode == 0,
        elapsed_seconds=elapsed,
        error_output=proc.stderr if proc.returncode != 0 else "",
    )


def run_cargo_test(
    project_path: Path, cargo_path: str = "cargo", extra_args: Optional[list[str]] = None
) -> CargoResult:
    """Run cargo test and parse pass/fail counts."""
    cmd = [cargo_path, "test"]
    if extra_args:
        cmd.extend(extra_args)

    start = time.monotonic()
    proc = subprocess.run(cmd, cwd=project_path, capture_output=True, text=True, timeout=120)
    elapsed = time.monotonic() - start

    passed = 0
    failed = 0
    for line in proc.stdout.splitlines():
        # Matches: "test result: ok. 3 passed; 0 failed; 0 ignored; ..."
        m = re.search(r"(\d+) passed.*?(\d+) failed", line)
        if m:
            passed += int(m.group(1))
            failed += int(m.group(2))

    return CargoResult(
        success=proc.returncode == 0,
        elapsed_seconds=elapsed,
        error_output=proc.stderr if proc.returncode != 0 else "",
        tests_passed=passed,
        tests_failed=failed,
    )


def parse_clippy_json(raw_json: str) -> list[dict]:
    """Parse clippy --message-format=json output into a list of warning dicts."""
    warnings = []
    # clippy outputs one JSON object per line, or a JSON array
    try:
        data = json.loads(raw_json)
        if isinstance(data, list):
            items = data
        else:
            items = [data]
    except json.JSONDecodeError:
        # line-delimited JSON
        items = []
        for line in raw_json.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    for item in items:
        if item.get("reason") != "compiler-message":
            continue
        msg = item.get("message", {})
        code_info = msg.get("code")
        if not code_info or not code_info.get("code"):
            continue
        code = code_info["code"]
        if not code.startswith("clippy::"):
            continue
        warnings.append({
            "code": code,
            "message": msg.get("message", ""),
            "level": msg.get("level", "warning"),
            "file": msg.get("spans", [{}])[0].get("file_name", ""),
            "line": msg.get("spans", [{}])[0].get("line_start", 0),
        })
    return warnings


def run_cargo_clippy(
    project_path: Path,
    cargo_path: str = "cargo",
    extra_args: Optional[list[str]] = None,
) -> CargoResult:
    """Run cargo clippy with JSON output and parse warnings."""
    cmd = [cargo_path, "clippy", "--message-format=json"]
    if extra_args:
        cmd.extend(extra_args)

    start = time.monotonic()
    proc = subprocess.run(cmd, cwd=project_path, capture_output=True, text=True, timeout=120)
    elapsed = time.monotonic() - start

    warnings = parse_clippy_json(proc.stdout)

    counts: dict[str, int] = {}
    for w in warnings:
        cat = categorize_lint(w["code"])
        counts[cat] = counts.get(cat, 0) + 1

    return CargoResult(
        success=proc.returncode == 0 or len(warnings) >= 0,  # clippy returns 0 even with warnings
        elapsed_seconds=elapsed,
        warnings=warnings,
        warning_counts=counts,
    )


def compute_clippy_score(counts: dict[str, int], weights: ClippyWeights) -> int:
    """Compute weighted clippy penalty score. More negative = worse."""
    return -(
        weights.correctness * counts.get("correctness", 0)
        + weights.suspicious * counts.get("suspicious", 0)
        + weights.complexity * counts.get("complexity", 0)
        + weights.perf * counts.get("perf", 0)
        + weights.style * counts.get("style", 0)
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cargo.py -v
```

Expected: all tests PASS (requires `cargo` on PATH for the subprocess tests)

- [ ] **Step 5: Commit**

```bash
git add codeevolve/evaluator/cargo.py tests/test_cargo.py
git commit -m "feat: cargo integration — build, test, clippy parsing and scoring"
```

---

### Task 4: Benchmark Module (Layer 3)

**Files:**
- Create: `codeevolve/evaluator/benchmark.py`
- Create: `tests/test_benchmark.py`

- [ ] **Step 1: Write failing tests for benchmark module**

`tests/test_benchmark.py`:
```python
import re
from pathlib import Path

import pytest

from codeevolve.evaluator.benchmark import (
    measure_binary_size,
    measure_compile_time,
    run_user_benchmark,
    BenchmarkResult,
)


def test_measure_compile_time(sample_crate: Path):
    seconds = measure_compile_time(sample_crate)
    assert seconds > 0


def test_measure_binary_size(sample_crate: Path):
    # Build first so a binary exists
    import subprocess
    subprocess.run(["cargo", "build"], cwd=sample_crate, capture_output=True)

    size_bytes = measure_binary_size(sample_crate)
    # Library crate produces a .rlib, should be > 0
    assert size_bytes > 0


def test_run_user_benchmark_success(tmp_path: Path):
    script = tmp_path / "bench.sh"
    script.write_text("#!/bin/bash\necho 'time: 42.5ms'")
    script.chmod(0o755)

    result = run_user_benchmark(
        command=str(script),
        cwd=tmp_path,
        score_regex=r"time: ([\d.]+)ms",
    )
    assert result.success is True
    assert result.score == 42.5


def test_run_user_benchmark_no_regex(tmp_path: Path):
    script = tmp_path / "bench.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    result = run_user_benchmark(command=str(script), cwd=tmp_path, score_regex=None)
    assert result.success is True
    assert result.score == 1.0  # exit 0 = pass = score 1.0


def test_run_user_benchmark_failure(tmp_path: Path):
    script = tmp_path / "bench.sh"
    script.write_text("#!/bin/bash\nexit 1")
    script.chmod(0o755)

    result = run_user_benchmark(command=str(script), cwd=tmp_path, score_regex=None)
    assert result.success is False
    assert result.score == 0.0


def test_run_user_benchmark_regex_no_match(tmp_path: Path):
    script = tmp_path / "bench.sh"
    script.write_text("#!/bin/bash\necho 'no numbers here'")
    script.chmod(0o755)

    result = run_user_benchmark(
        command=str(script),
        cwd=tmp_path,
        score_regex=r"time: ([\d.]+)ms",
    )
    assert result.success is True
    assert result.score == 0.0  # regex didn't match, score 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_benchmark.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'codeevolve.evaluator.benchmark'`

- [ ] **Step 3: Implement benchmark.py**

`codeevolve/evaluator/benchmark.py`:
```python
from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class BenchmarkResult:
    success: bool
    score: float
    output: str = ""


def measure_compile_time(project_path: Path, cargo_path: str = "cargo") -> float:
    """Clean-build the project and return wall-clock compile time in seconds."""
    # Clean first to get a full build time
    subprocess.run(
        [cargo_path, "clean"], cwd=project_path, capture_output=True, timeout=30
    )
    start = time.monotonic()
    subprocess.run(
        [cargo_path, "build"], cwd=project_path, capture_output=True, timeout=120
    )
    return time.monotonic() - start


def measure_binary_size(project_path: Path, target_dir: Optional[str] = None) -> int:
    """Return the total size in bytes of compiled artifacts in target/debug/."""
    target = Path(target_dir) if target_dir else project_path / "target"
    debug_dir = target / "debug"
    if not debug_dir.exists():
        return 0

    total = 0
    # Look for library (.rlib) and binary files (no extension or .exe)
    for f in debug_dir.iterdir():
        if f.is_file() and (
            f.suffix in (".rlib", ".exe", ".dll", ".so", ".dylib", "")
            and not f.name.startswith(".")
            and f.stat().st_size > 1000  # skip tiny metadata files
        ):
            total += f.stat().st_size
    return total


def run_user_benchmark(
    command: str,
    cwd: Path,
    score_regex: Optional[str] = None,
    timeout: int = 120,
) -> BenchmarkResult:
    """Run user-provided benchmark command, optionally extract a score via regex."""
    proc = subprocess.run(
        command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )

    if score_regex is None:
        return BenchmarkResult(
            success=proc.returncode == 0,
            score=1.0 if proc.returncode == 0 else 0.0,
            output=proc.stdout,
        )

    match = re.search(score_regex, proc.stdout)
    if match:
        score = float(match.group(1))
    else:
        score = 0.0

    return BenchmarkResult(
        success=proc.returncode == 0,
        score=score,
        output=proc.stdout,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_benchmark.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add codeevolve/evaluator/benchmark.py tests/test_benchmark.py
git commit -m "feat: benchmark module — compile time, binary size, user benchmarks"
```

---

### Task 5: LLM Judge (Layer 4)

**Files:**
- Create: `codeevolve/evaluator/llm_judge.py`
- Create: `tests/test_llm_judge.py`

- [ ] **Step 1: Write failing tests for LLM judge**

`tests/test_llm_judge.py`:
```python
import json
from unittest.mock import MagicMock, patch

import pytest

from codeevolve.evaluator.llm_judge import (
    LlmJudgment,
    build_judgment_prompt,
    parse_judgment_response,
    judge_code,
)


def test_build_judgment_prompt():
    code = "fn add(a: i32, b: i32) -> i32 { a + b }"
    dimensions = ["readability", "rust_idiomaticity"]
    prompt = build_judgment_prompt(code, dimensions)
    assert "readability" in prompt
    assert "rust_idiomaticity" in prompt
    assert "fn add" in prompt
    assert "1-5" in prompt  # Likert scale reference


def test_parse_judgment_response_valid():
    response = json.dumps({
        "readability": 4,
        "rust_idiomaticity": 5,
        "maintainability": 3,
        "design": 4,
    })
    scores = parse_judgment_response(response, ["readability", "rust_idiomaticity", "maintainability", "design"])
    assert scores == {"readability": 4, "rust_idiomaticity": 5, "maintainability": 3, "design": 4}


def test_parse_judgment_response_with_reasoning():
    response = """Here is my analysis...
    
```json
{"readability": 4, "rust_idiomaticity": 3}
```"""
    scores = parse_judgment_response(response, ["readability", "rust_idiomaticity"])
    assert scores == {"readability": 4, "rust_idiomaticity": 3}


def test_parse_judgment_response_invalid():
    scores = parse_judgment_response("not json at all", ["readability"])
    assert scores == {}


def test_parse_judgment_response_clamps_scores():
    response = json.dumps({"readability": 10, "design": -1})
    scores = parse_judgment_response(response, ["readability", "design"])
    assert scores["readability"] == 5  # clamped to max
    assert scores["design"] == 1  # clamped to min


@patch("codeevolve.evaluator.llm_judge._call_ollama")
def test_judge_code_aggregates_runs(mock_call):
    """judge_code runs N times and takes medians."""
    mock_call.side_effect = [
        json.dumps({"readability": 3, "design": 4}),
        json.dumps({"readability": 5, "design": 4}),
        json.dumps({"readability": 4, "design": 2}),
    ]
    result = judge_code(
        code="fn main() {}",
        api_base="http://localhost:11434/v1",
        model="test-model",
        dimensions=["readability", "design"],
        num_runs=3,
    )
    assert result.dimension_scores["readability"] == 4  # median of [3,5,4]
    assert result.dimension_scores["design"] == 4  # median of [4,4,2]
    assert result.combined_score == 4.0  # mean of [4, 4]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_llm_judge.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'codeevolve.evaluator.llm_judge'`

- [ ] **Step 3: Implement llm_judge.py**

`codeevolve/evaluator/llm_judge.py`:
```python
from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class LlmJudgment:
    dimension_scores: dict[str, float] = field(default_factory=dict)
    combined_score: float = 0.0


def build_judgment_prompt(code: str, dimensions: list[str]) -> str:
    """Build a prompt asking the LLM to judge code quality on given dimensions."""
    dim_list = "\n".join(f"- **{d}**: score 1-5" for d in dimensions)
    return f"""You are an expert Rust code reviewer. Evaluate the following code on each dimension using a 1-5 Likert scale (1=poor, 5=excellent).

Think step by step about the code quality, then provide your scores as a JSON object.

**Dimensions:**
{dim_list}

**Code:**
```rust
{code}
```

Respond with your reasoning first, then a JSON code block containing only the dimension scores. Example:
```json
{{{", ".join(f'"{d}": 3' for d in dimensions)}}}
```"""


def parse_judgment_response(response: str, dimensions: list[str]) -> dict[str, float]:
    """Extract dimension scores from LLM response text."""
    # Try to find a JSON block in the response
    json_match = re.search(r"```json\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
    text_to_parse = json_match.group(1) if json_match else response

    try:
        data = json.loads(text_to_parse)
    except json.JSONDecodeError:
        # Try to find any JSON object in the text
        obj_match = re.search(r"\{[^{}]+\}", response)
        if not obj_match:
            return {}
        try:
            data = json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            return {}

    scores = {}
    for dim in dimensions:
        if dim in data:
            val = data[dim]
            if isinstance(val, (int, float)):
                scores[dim] = max(1, min(5, int(val)))  # clamp to 1-5
    return scores


def _call_ollama(api_base: str, model: str, prompt: str) -> str:
    """Make a single chat completion call to Ollama."""
    client = OpenAI(base_url=api_base, api_key="ollama")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


def judge_code(
    code: str,
    api_base: str,
    model: str,
    dimensions: list[str],
    num_runs: int = 3,
) -> LlmJudgment:
    """Run LLM judgment multiple times and aggregate via median."""
    prompt = build_judgment_prompt(code, dimensions)

    all_scores: dict[str, list[float]] = {d: [] for d in dimensions}

    for _ in range(num_runs):
        response = _call_ollama(api_base, model, prompt)
        scores = parse_judgment_response(response, dimensions)
        for dim in dimensions:
            if dim in scores:
                all_scores[dim].append(scores[dim])

    # Compute medians
    dimension_medians = {}
    for dim in dimensions:
        vals = all_scores[dim]
        if vals:
            dimension_medians[dim] = statistics.median(vals)
        else:
            dimension_medians[dim] = 1.0  # worst score if LLM never returned this dim

    combined = statistics.mean(dimension_medians.values()) if dimension_medians else 1.0

    return LlmJudgment(dimension_scores=dimension_medians, combined_score=combined)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_llm_judge.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add codeevolve/evaluator/llm_judge.py tests/test_llm_judge.py
git commit -m "feat: LLM judge — Ollama-based code quality evaluation with median aggregation"
```

---

### Task 6: Evaluation Pipeline (Orchestrator)

**Files:**
- Create: `codeevolve/evaluator/pipeline.py`
- Update: `codeevolve/evaluator/__init__.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests for the pipeline**

`tests/test_pipeline.py`:
```python
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeevolve.config import load_config
from codeevolve.evaluator.pipeline import EvaluationPipeline, EvaluationResult


@pytest.fixture
def pipeline():
    config = load_config()
    return EvaluationPipeline(config)


def test_evaluation_result_fields():
    r = EvaluationResult(
        passed_gates=True,
        combined_score=0.75,
        static_score=-5,
        perf_score=0.8,
        llm_score=0.0,
    )
    assert r.combined_score == 0.75


@patch("codeevolve.evaluator.pipeline.run_cargo_build")
def test_pipeline_build_failure_returns_zero(mock_build, pipeline):
    mock_build.return_value = MagicMock(success=False, error_output="error", elapsed_seconds=1.0)

    result = pipeline.evaluate("/fake/path")
    assert result.passed_gates is False
    assert result.combined_score == 0.0


@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_build")
def test_pipeline_test_failure_returns_zero(mock_build, mock_test, pipeline):
    mock_build.return_value = MagicMock(success=True, elapsed_seconds=1.0)
    mock_test.return_value = MagicMock(success=False, error_output="test failed", tests_passed=0, tests_failed=1, elapsed_seconds=1.0)

    result = pipeline.evaluate("/fake/path")
    assert result.passed_gates is False
    assert result.combined_score == 0.0


@patch("codeevolve.evaluator.pipeline.judge_code")
@patch("codeevolve.evaluator.pipeline.measure_binary_size")
@patch("codeevolve.evaluator.pipeline.measure_compile_time")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_build")
def test_pipeline_full_pass(mock_build, mock_test, mock_clippy, mock_compile_time, mock_binary_size, mock_judge, pipeline):
    mock_build.return_value = MagicMock(success=True, elapsed_seconds=1.0)
    mock_test.return_value = MagicMock(success=True, tests_passed=5, tests_failed=0, elapsed_seconds=1.0)
    mock_clippy.return_value = MagicMock(success=True, warnings=[], warning_counts={}, elapsed_seconds=0.5)
    mock_compile_time.return_value = 2.5
    mock_binary_size.return_value = 1_000_000
    mock_judge.return_value = MagicMock(combined_score=4.0, dimension_scores={"readability": 4})

    # Force top-quartile by setting low history
    pipeline._score_history = [0.1, 0.2]

    result = pipeline.evaluate("/fake/path")
    assert result.passed_gates is True
    assert result.combined_score > 0


@patch("codeevolve.evaluator.pipeline.measure_binary_size")
@patch("codeevolve.evaluator.pipeline.measure_compile_time")
@patch("codeevolve.evaluator.pipeline.run_cargo_clippy")
@patch("codeevolve.evaluator.pipeline.run_cargo_test")
@patch("codeevolve.evaluator.pipeline.run_cargo_build")
def test_pipeline_skips_llm_if_not_top_quartile(mock_build, mock_test, mock_clippy, mock_compile_time, mock_binary_size, pipeline):
    mock_build.return_value = MagicMock(success=True, elapsed_seconds=1.0)
    mock_test.return_value = MagicMock(success=True, tests_passed=5, tests_failed=0, elapsed_seconds=1.0)
    mock_clippy.return_value = MagicMock(success=True, warnings=[{"code": "clippy::style"}] * 20, warning_counts={"style": 20}, elapsed_seconds=0.5)
    mock_compile_time.return_value = 10.0
    mock_binary_size.return_value = 5_000_000

    # Fill history with high scores so this one won't be top quartile
    pipeline._score_history = [0.9, 0.95, 0.85, 0.88, 0.92, 0.87, 0.91, 0.86]

    result = pipeline.evaluate("/fake/path")
    assert result.passed_gates is True
    assert result.llm_score == 0.0  # skipped
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'codeevolve.evaluator.pipeline'`

- [ ] **Step 3: Implement pipeline.py**

`codeevolve/evaluator/pipeline.py`:
```python
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from codeevolve.config import CodeEvolveConfig
from codeevolve.evaluator.cargo import (
    compute_clippy_score,
    run_cargo_build,
    run_cargo_clippy,
    run_cargo_test,
)
from codeevolve.evaluator.benchmark import (
    measure_binary_size,
    measure_compile_time,
    run_user_benchmark,
)
from codeevolve.evaluator.llm_judge import judge_code


@dataclass
class EvaluationResult:
    passed_gates: bool
    combined_score: float
    static_score: float = 0.0
    perf_score: float = 0.0
    llm_score: float = 0.0
    build_time: float = 0.0
    tests_passed: int = 0
    tests_failed: int = 0
    clippy_warnings: int = 0
    binary_size: int = 0
    compile_time: float = 0.0
    error: str = ""


class EvaluationPipeline:
    """4-layer gated evaluation pipeline for Rust code."""

    def __init__(self, config: CodeEvolveConfig):
        self.config = config
        self._score_history: list[float] = []
        # Track min/max for normalization
        self._static_min: Optional[float] = None
        self._static_max: Optional[float] = None
        self._perf_min: Optional[float] = None
        self._perf_max: Optional[float] = None

    def _update_range(self, value: float, current_min: Optional[float], current_max: Optional[float]):
        new_min = value if current_min is None else min(current_min, value)
        new_max = value if current_max is None else max(current_max, value)
        return new_min, new_max

    def _normalize(self, value: float, min_val: Optional[float], max_val: Optional[float]) -> float:
        if min_val is None or max_val is None or min_val == max_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)

    def _is_top_quartile(self, pre_llm_score: float) -> bool:
        if len(self._score_history) < 4:
            return True  # not enough data, always run LLM
        threshold = statistics.quantiles(self._score_history, n=4)[2]  # 75th percentile
        return pre_llm_score >= threshold

    def evaluate(self, program_path: str) -> EvaluationResult:
        """Run the full 4-layer evaluation pipeline on a candidate program."""
        project_path = Path(program_path).parent
        cargo = self.config.rust.cargo_path
        cfg = self.config

        # --- Layer 1: Hard gates ---
        build = run_cargo_build(project_path, cargo, cfg.rust.target_dir)
        if not build.success:
            return EvaluationResult(
                passed_gates=False, combined_score=0.0,
                build_time=build.elapsed_seconds, error=build.error_output,
            )

        test = run_cargo_test(project_path, cargo, cfg.rust.test_args or None)
        if not test.success:
            return EvaluationResult(
                passed_gates=False, combined_score=0.0,
                build_time=build.elapsed_seconds,
                tests_passed=test.tests_passed, tests_failed=test.tests_failed,
                error=test.error_output,
            )

        # --- Layer 2: Static analysis ---
        clippy = run_cargo_clippy(project_path, cargo, cfg.rust.clippy_args or None)
        raw_static = compute_clippy_score(clippy.warning_counts, cfg.fitness.clippy_weights)
        self._static_min, self._static_max = self._update_range(
            raw_static, self._static_min, self._static_max
        )
        norm_static = self._normalize(raw_static, self._static_min, self._static_max)

        # --- Layer 3: Performance ---
        compile_time = 0.0
        binary_size = 0
        bench_score = 0.0
        perf_components = []

        if cfg.benchmarks.measure_compile_time:
            compile_time = measure_compile_time(project_path, cargo)
            # Lower compile time is better, so invert
            perf_components.append(-compile_time)

        if cfg.benchmarks.measure_binary_size:
            binary_size = measure_binary_size(project_path, cfg.rust.target_dir)
            # Lower binary size is better, so invert
            perf_components.append(-binary_size)

        if cfg.benchmarks.custom_command:
            bench_result = run_user_benchmark(
                cfg.benchmarks.custom_command,
                project_path,
                cfg.benchmarks.custom_command_score_regex,
            )
            bench_score = bench_result.score
            perf_components.append(bench_score)

        raw_perf = sum(perf_components) if perf_components else 0.0
        self._perf_min, self._perf_max = self._update_range(
            raw_perf, self._perf_min, self._perf_max
        )
        norm_perf = self._normalize(raw_perf, self._perf_min, self._perf_max)

        # --- Pre-LLM combined score ---
        w_static = cfg.fitness.static_analysis_weight
        w_perf = cfg.fitness.performance_weight
        pre_llm = (w_static * norm_static + w_perf * norm_perf) / (w_static + w_perf)
        self._score_history.append(pre_llm)

        # --- Layer 4: LLM judgment (top quartile only) ---
        norm_llm = 0.0
        if cfg.llm_judgment.enabled and (
            not cfg.llm_judgment.top_quartile_only or self._is_top_quartile(pre_llm)
        ):
            code = Path(program_path).read_text()
            judgment = judge_code(
                code=code,
                api_base=cfg.ollama.api_base,
                model=cfg.ollama.evaluator_model,
                dimensions=cfg.llm_judgment.dimensions,
                num_runs=cfg.llm_judgment.num_runs,
            )
            norm_llm = (judgment.combined_score - 1.0) / 4.0  # normalize 1-5 to 0-1

        # --- Combined score ---
        combined = (
            cfg.fitness.static_analysis_weight * norm_static
            + cfg.fitness.performance_weight * norm_perf
            + cfg.fitness.llm_judgment_weight * norm_llm
        )

        return EvaluationResult(
            passed_gates=True,
            combined_score=combined,
            static_score=raw_static,
            perf_score=raw_perf,
            llm_score=norm_llm,
            build_time=build.elapsed_seconds,
            tests_passed=test.tests_passed,
            tests_failed=test.tests_failed,
            clippy_warnings=len(clippy.warnings),
            binary_size=binary_size,
            compile_time=compile_time,
        )
```

- [ ] **Step 4: Update evaluator __init__.py to re-export evaluate()**

`codeevolve/evaluator/__init__.py`:
```python
from codeevolve.evaluator.pipeline import EvaluationPipeline, EvaluationResult

__all__ = ["EvaluationPipeline", "EvaluationResult"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_pipeline.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add codeevolve/evaluator/ tests/test_pipeline.py
git commit -m "feat: 4-layer gated evaluation pipeline orchestrator"
```

---

### Task 7: Evaluator Template + Defaults

**Files:**
- Create: `codeevolve/templates/evaluator.py.j2`

- [ ] **Step 1: Create the Jinja2 evaluator template**

This template generates the `evaluator.py` file that OpenEvolve calls. It bridges OpenEvolve's `evaluate(program_path)` API to our pipeline.

`codeevolve/templates/evaluator.py.j2`:
```python
"""Auto-generated evaluator for OpenEvolve.

This file is called by OpenEvolve for each candidate program.
It runs the CodeEvolution 4-layer evaluation pipeline.
Do not edit unless you know what you're doing.

Generated by: codeevolve init
Project: {{ project_name }}
"""
import sys
from pathlib import Path

# Add the codeevolve package to the path so this evaluator can import it.
# This is necessary because OpenEvolve runs evaluators as standalone scripts.
sys.path.insert(0, "{{ codeevolve_package_path }}")

from codeevolve.config import load_config
from codeevolve.evaluator.pipeline import EvaluationPipeline

_CONFIG_PATH = Path("{{ config_path }}")
_config = load_config(_CONFIG_PATH)
_pipeline = EvaluationPipeline(_config)


def evaluate(program_path: str) -> dict:
    """OpenEvolve calls this for each candidate. Returns a metrics dict."""
    result = _pipeline.evaluate(program_path)
    return {
        "combined_score": result.combined_score,
        "passed_gates": float(result.passed_gates),
        "static_score": result.static_score,
        "perf_score": result.perf_score,
        "llm_score": result.llm_score,
        "clippy_warnings": float(result.clippy_warnings),
        "compile_time": result.compile_time,
        "binary_size": float(result.binary_size),
        "tests_passed": float(result.tests_passed),
    }
```

- [ ] **Step 2: Commit**

```bash
git add codeevolve/templates/
git commit -m "feat: Jinja2 evaluator template for OpenEvolve integration"
```

---

### Task 8: Init Command

**Files:**
- Create: `codeevolve/init_project.py`
- Create: `tests/test_init_project.py`

- [ ] **Step 1: Write failing tests for init_project**

`tests/test_init_project.py`:
```python
from pathlib import Path

import pytest

from codeevolve.init_project import (
    find_cargo_toml,
    scan_rs_files,
    insert_evolve_markers,
    generate_codeevolve_dir,
)


def test_find_cargo_toml(sample_crate: Path):
    result = find_cargo_toml(sample_crate)
    assert result == sample_crate / "Cargo.toml"


def test_find_cargo_toml_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Cargo.toml"):
        find_cargo_toml(tmp_path)


def test_find_cargo_toml_workspace_rejected(tmp_path: Path):
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text('[workspace]\nmembers = ["crate_a"]')
    with pytest.raises(ValueError, match="workspace"):
        find_cargo_toml(tmp_path)


def test_scan_rs_files(sample_crate: Path):
    files = scan_rs_files(sample_crate)
    assert len(files) == 1
    assert files[0].name == "lib.rs"


def test_scan_rs_files_nested(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.rs").write_text("fn main() {}")
    (src / "utils.rs").write_text("pub fn foo() {}")
    sub = src / "sub"
    sub.mkdir()
    (sub / "mod.rs").write_text("pub mod inner;")

    files = scan_rs_files(tmp_path)
    assert len(files) == 3


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
    assert rs_file.read_text() == original  # unchanged


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

    # Check evaluator.py is valid Python (can be parsed)
    evaluator_code = (codeevolve_dir / "evaluator.py").read_text()
    compile(evaluator_code, "evaluator.py", "exec")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_init_project.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'codeevolve.init_project'`

- [ ] **Step 3: Implement init_project.py**

`codeevolve/init_project.py`:
```python
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import jinja2
import yaml

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_DEFAULTS_DIR = Path(__file__).parent / "defaults"


def find_cargo_toml(project_path: Path) -> Path:
    """Find and validate Cargo.toml in the project directory."""
    cargo_toml = project_path / "Cargo.toml"
    if not cargo_toml.exists():
        raise FileNotFoundError(
            f"No Cargo.toml found in {project_path}. "
            "Run this command from a Rust project directory, or use --path."
        )
    content = cargo_toml.read_text()
    if "[workspace]" in content:
        raise ValueError(
            f"Found a workspace Cargo.toml at {cargo_toml}. "
            "CodeEvolution v1 supports single crates only. "
            "Point --path at a specific member crate instead."
        )
    return cargo_toml


def scan_rs_files(project_path: Path) -> list[Path]:
    """Find all .rs files under src/."""
    src_dir = project_path / "src"
    if not src_dir.exists():
        return []
    return sorted(src_dir.rglob("*.rs"))


def insert_evolve_markers(rs_file: Path) -> None:
    """Wrap file content in EVOLVE-BLOCK markers if not already present."""
    content = rs_file.read_text()
    if "EVOLVE-BLOCK-START" in content:
        return
    wrapped = f"// EVOLVE-BLOCK-START\n{content}// EVOLVE-BLOCK-END\n"
    rs_file.write_text(wrapped)


def generate_codeevolve_dir(
    project_path: Path,
    rs_files: list[Path],
    custom_benchmark: Optional[str] = None,
    custom_benchmark_regex: Optional[str] = None,
) -> Path:
    """Generate the .codeevolve/ directory with config, evaluator, and README."""
    codeevolve_dir = project_path / ".codeevolve"
    codeevolve_dir.mkdir(exist_ok=True)

    # --- Generate evolution.yaml ---
    with open(_DEFAULTS_DIR / "evolution.yaml") as f:
        config_data = yaml.safe_load(f)

    if custom_benchmark:
        config_data["benchmarks"]["custom_command"] = custom_benchmark
    if custom_benchmark_regex:
        config_data["benchmarks"]["custom_command_score_regex"] = custom_benchmark_regex

    config_path = codeevolve_dir / "evolution.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    # --- Generate evaluator.py ---
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
        keep_trailing_newline=True,
    )
    template = env.get_template("evaluator.py.j2")

    # Find the codeevolve package path (parent of this file's parent)
    codeevolve_package_path = str(Path(__file__).parent.parent.resolve())

    evaluator_code = template.render(
        project_name=project_path.name,
        codeevolve_package_path=codeevolve_package_path,
        config_path=str(config_path.resolve()),
    )
    (codeevolve_dir / "evaluator.py").write_text(evaluator_code)

    # --- Generate README.md ---
    file_list = "\n".join(f"  - {f.relative_to(project_path)}" for f in rs_files)
    readme = f"""# CodeEvolution Setup

This directory was generated by `codeevolve init`. Here's what each file does:

## Files

- **evolution.yaml** — Configuration for the evolutionary optimizer. Controls which
  Ollama models to use, how many iterations to run, fitness weights, and more.
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

1. Make sure Ollama is running: `ollama serve`
2. Pull the required models (first time only):
   - `ollama pull qwen2.5-coder:7b-instruct-q4_K_M`
   - `ollama pull qwen2.5-coder:1.5b-instruct-q4_K_M`
3. Start the evolution: `codeevolve run`

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_init_project.py -v
```

Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add codeevolve/init_project.py tests/test_init_project.py
git commit -m "feat: init_project — scans Rust crate and generates .codeevolve/ directory"
```

---

### Task 9: Runner (OpenEvolve Integration)

**Files:**
- Create: `codeevolve/runner.py`
- Create: `tests/test_runner.py`

- [ ] **Step 1: Write failing tests for runner**

`tests/test_runner.py`:
```python
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeevolve.config import load_config
from codeevolve.runner import (
    validate_ollama,
    build_openevolve_config_yaml,
    format_iteration_line,
    run_evolution,
)


@patch("codeevolve.runner.OpenAI")
def test_validate_ollama_success(mock_openai_cls):
    mock_client = MagicMock()
    mock_client.models.list.return_value = MagicMock(
        data=[MagicMock(id="qwen2.5-coder:7b-instruct-q4_K_M"), MagicMock(id="qwen2.5-coder:1.5b-instruct-q4_K_M")]
    )
    mock_openai_cls.return_value = mock_client

    config = load_config()
    errors = validate_ollama(config)
    assert errors == []


@patch("codeevolve.runner.OpenAI")
def test_validate_ollama_missing_model(mock_openai_cls):
    mock_client = MagicMock()
    mock_client.models.list.return_value = MagicMock(
        data=[MagicMock(id="qwen2.5-coder:7b-instruct-q4_K_M")]
    )
    mock_openai_cls.return_value = mock_client

    config = load_config()
    errors = validate_ollama(config)
    assert len(errors) == 1
    assert "qwen2.5-coder:1.5b-instruct-q4_K_M" in errors[0]


@patch("codeevolve.runner.OpenAI")
def test_validate_ollama_unreachable(mock_openai_cls):
    mock_openai_cls.side_effect = Exception("Connection refused")

    config = load_config()
    errors = validate_ollama(config)
    assert len(errors) == 1
    assert "Connection refused" in errors[0]


def test_build_openevolve_config_yaml(tmp_path: Path):
    config = load_config()
    yaml_path = build_openevolve_config_yaml(config, tmp_path)
    assert yaml_path.exists()
    content = yaml_path.read_text()
    assert "qwen2.5-coder:7b-instruct-q4_K_M" in content


def test_format_iteration_line_success():
    line = format_iteration_line(
        iteration=12,
        total=500,
        file_changed="src/lib.rs",
        diff_lines=47,
        build_ok=True,
        build_time=1.2,
        tests_ok=True,
        tests_passed=23,
        tests_failed=0,
        clippy_warnings=3,
        parent_clippy_warnings=7,
        binary_size=1_400_000,
        parent_binary_size=1_500_000,
        llm_ran=False,
        score=0.72,
        best_score=0.81,
    )
    assert "12/500" in line
    assert "src/lib.rs" in line
    assert "improved" in line.lower() or "3 warnings" in line


def test_format_iteration_line_build_failure():
    line = format_iteration_line(
        iteration=13,
        total=500,
        file_changed="src/utils.rs",
        diff_lines=12,
        build_ok=False,
        build_time=0.5,
        error="error[E0308]: mismatched types",
    )
    assert "FAILED" in line
    assert "0.00" in line
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_runner.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'codeevolve.runner'`

- [ ] **Step 3: Implement runner.py**

`codeevolve/runner.py`:
```python
from __future__ import annotations

import shutil
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

    # Clippy
    clippy_delta = ""
    if parent_clippy_warnings is not None:
        if clippy_warnings < parent_clippy_warnings:
            clippy_delta = f" (was {parent_clippy_warnings}) - improved"
        elif clippy_warnings > parent_clippy_warnings:
            clippy_delta = f" (was {parent_clippy_warnings}) - regressed"
    lines.append(f"  |- Clippy:   {clippy_warnings} warnings{clippy_delta}")

    # Binary size
    size_mb = binary_size / 1_000_000
    size_delta = ""
    if parent_binary_size is not None:
        parent_mb = parent_binary_size / 1_000_000
        if binary_size < parent_binary_size:
            size_delta = f" (was {parent_mb:.1f} MB) - improved"
        elif binary_size > parent_binary_size:
            size_delta = f" (was {parent_mb:.1f} MB) - regressed"
    lines.append(f"  |- Size:     {size_mb:.1f} MB{size_delta}")

    # LLM
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

    # Build OpenEvolve config
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

    # Save best result
    best_dir = output_dir / "best"
    best_dir.mkdir(exist_ok=True)
    if result.best_code:
        (best_dir / initial_program.name).write_text(result.best_code)

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_runner.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add codeevolve/runner.py tests/test_runner.py
git commit -m "feat: runner — Ollama validation, OpenEvolve config generation, progress display"
```

---

### Task 10: CLI

**Files:**
- Create: `codeevolve/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for CLI commands**

`tests/test_cli.py`:
```python
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from codeevolve.cli import main


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_cli_help(cli_runner):
    result = cli_runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "init" in result.output
    assert "run" in result.output


def test_init_help(cli_runner):
    result = cli_runner.invoke(main, ["init", "--help"])
    assert result.exit_code == 0
    assert "--path" in result.output


def test_run_help(cli_runner):
    result = cli_runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output


def test_init_no_cargo_toml(cli_runner, tmp_path):
    result = cli_runner.invoke(main, ["init", "--path", str(tmp_path)])
    assert result.exit_code != 0
    assert "Cargo.toml" in result.output


@patch("codeevolve.cli.generate_codeevolve_dir")
@patch("codeevolve.cli.insert_evolve_markers")
@patch("codeevolve.cli.scan_rs_files")
@patch("codeevolve.cli.find_cargo_toml")
def test_init_success(mock_find, mock_scan, mock_markers, mock_generate, cli_runner, sample_crate):
    mock_find.return_value = sample_crate / "Cargo.toml"
    mock_scan.return_value = [sample_crate / "src" / "lib.rs"]
    mock_generate.return_value = sample_crate / ".codeevolve"

    result = cli_runner.invoke(main, ["init", "--path", str(sample_crate)], input="1\n\n")
    assert result.exit_code == 0


@patch("codeevolve.cli.validate_ollama")
def test_run_ollama_not_reachable(mock_validate, cli_runner, tmp_path):
    mock_validate.return_value = ["Cannot connect to Ollama"]
    config_path = tmp_path / "evolution.yaml"
    config_path.write_text("ollama:\n  api_base: http://localhost:11434/v1\n")

    result = cli_runner.invoke(main, ["run", "--config", str(config_path)])
    assert result.exit_code != 0
    assert "Cannot connect" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cli.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'codeevolve.cli'`

- [ ] **Step 3: Implement cli.py**

`codeevolve/cli.py`:
```python
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

    # Step 1: Find and validate Cargo.toml
    try:
        find_cargo_toml(path)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Step 2: Scan for .rs files
    rs_files = scan_rs_files(path)
    if not rs_files:
        click.echo("Error: No .rs files found in src/", err=True)
        sys.exit(1)

    # Step 3: Interactive file selection
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

    # Step 4: Optional benchmark
    custom_benchmark = None
    custom_regex = None
    if click.confirm("\nDo you have a custom benchmark command?", default=False):
        custom_benchmark = click.prompt("Benchmark command (e.g., cargo bench)")
        custom_regex = click.prompt(
            "Regex to extract score from output (or Enter to use exit code)",
            default="",
            show_default=False,
        )
        custom_regex = custom_regex if custom_regex else None

    # Step 5: Insert markers
    click.echo("\nMarking files for evolution...")
    for f in selected:
        insert_evolve_markers(f)
        click.echo(f"  Marked: {f.relative_to(path)}")

    # Step 6: Generate .codeevolve/
    click.echo("\nGenerating configuration...")
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
    project_path = config_path.parent.parent  # .codeevolve/evolution.yaml -> project root

    # Load and validate
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        click.echo(f"Error: Config not found at {config_path}", err=True)
        click.echo("Run 'codeevolve init' first to set up your project.", err=True)
        sys.exit(1)

    click.echo(f"  Loading config from {config_path.relative_to(project_path.parent)}")

    # Validate Ollama
    errors = validate_ollama(config)
    if errors:
        for e in errors:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"  Connected to Ollama ({config.ollama.mutator_model}, {config.ollama.evaluator_model})")
    click.echo(f"  Starting evolution ({config.evolution.max_iterations} iterations, population {config.evolution.population_size})")
    click.echo()

    # Find initial program (first .rs file with EVOLVE-BLOCK markers)
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cli.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add codeevolve/cli.py tests/test_cli.py
git commit -m "feat: CLI with init and run commands"
```

---

### Task 11: Integration Smoke Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write an integration test for the full init flow**

`tests/test_integration.py`:
```python
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

    # Verify EVOLVE-BLOCK markers were inserted
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
```

- [ ] **Step 2: Run the integration test**

```bash
pytest tests/test_integration.py -v
```

Expected: PASS

- [ ] **Step 3: Run all tests together**

```bash
pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 4: Verify CLI entry point works**

```bash
codeevolve --help
codeevolve init --help
codeevolve run --help
```

Expected: help text prints for all three commands

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration smoke test for full init flow"
```

---

### Task 12: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md with actual project commands**

Replace the existing CLAUDE.md with updated content reflecting the implemented project:

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CodeEvolution is a CLI tool that wraps OpenEvolve to provide batteries-included evolutionary code optimization for Rust projects. Users run `codeevolve init` in a Rust project, then `codeevolve run` to evolve their code using LLMs via Ollama.

See `Reference.md` for the design rationale and `docs/superpowers/specs/2026-04-09-codeevolution-design.md` for the full spec.

## Tech Stack

- Python 3.13, Click (CLI), OpenEvolve (evolutionary engine), Jinja2 (templates), PyYAML, openai (Ollama client)
- Ollama for local LLM inference at `http://localhost:11434/v1`
- Target models: Qwen2.5-Coder-7B (mutator) + 1.5B (evaluator) via Ollama

## Commands

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a single test
pytest tests/test_cargo.py::test_parse_clippy_json -v

# Run the CLI
codeevolve --help
codeevolve init --path /path/to/rust/project
codeevolve run --config .codeevolve/evolution.yaml
```

## Architecture

The system is a thin wrapper over OpenEvolve with two CLI commands:

1. **`codeevolve init`** (`init_project.py`) — scans a Rust crate, inserts EVOLVE-BLOCK markers, generates `.codeevolve/` with config YAML + evaluator.
2. **`codeevolve run`** (`runner.py`) — validates Ollama, builds OpenEvolve config, calls `run_evolution()`, displays progress.

The core value-add is the **4-layer evaluation pipeline** (`evaluator/pipeline.py`):
- Layer 1: `cargo.py` — hard gates (cargo build + cargo test)
- Layer 2: `cargo.py` — Clippy static analysis with weighted lint scoring
- Layer 3: `benchmark.py` — compile time, binary size, optional user benchmark
- Layer 4: `llm_judge.py` — Ollama-based quality judgment (top-quartile only, 3-run median)

Config is a single dataclass hierarchy (`config.py`) loaded from YAML, with defaults in `codeevolve/defaults/evolution.yaml`.

## Key Constraints

- Rust-only in v1 (no multi-language)
- Ollama-only (no external API support)
- Single-crate only (workspace members must be targeted individually via --path)
- Both models must fit in 8GB VRAM simultaneously
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with implemented project details"
```
