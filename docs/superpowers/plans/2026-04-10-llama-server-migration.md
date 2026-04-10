# llama-server Migration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Ollama backend with llama-server (llama.cpp) for LLM inference, using a single partially-offloaded 14B model started internally as a subprocess.

**Architecture:** Remove all Ollama-specific code. Add a new `llama_server.py` module that manages the llama-server process lifecycle (start/health-poll/stop). Rename config from `OllamaConfig` to `LlamaServerConfig` with fields for server binary path, GGUF model path, GPU layers, port, etc. The OpenAI-compatible evaluator code (`llm_judge.py`, `llm_fixer.py`) needs only field-name changes since llama-server exposes the same `/v1/chat/completions` endpoint.

**Tech Stack:** Python 3.13, subprocess (for llama-server lifecycle), openai SDK (unchanged), llama.cpp llama-server

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `codeevolve/llama_server.py` | **Create** | Start/stop/health-check llama-server subprocess |
| `codeevolve/config.py` | Modify | Replace `OllamaConfig` with `LlamaServerConfig`, update `CodeEvolveConfig`, update `to_openevolve_dict()` |
| `codeevolve/defaults/evolution.yaml` | Modify | Replace `ollama:` section with `llama_server:` section, remove `context_window` from `evolution:` |
| `codeevolve/runner.py` | Modify | Remove `validate_ollama()`, `prime_ollama_models()`, `_ollama_base_url()`. Add `validate_server()` |
| `codeevolve/evaluator/llm_judge.py` | Modify | Rename `_call_ollama` to `_call_llm`, change api_key |
| `codeevolve/evaluator/llm_fixer.py` | Modify | Change api_key |
| `codeevolve/evaluator/pipeline.py` | Modify | Update `cfg.ollama.*` references to `cfg.llama_server.*` |
| `codeevolve/cli.py` | Modify | Replace Ollama lifecycle with LlamaServer context manager, update help text |
| `codeevolve/init_project.py` | Modify | Update README template |
| `tests/test_llama_server.py` | **Create** | Tests for server lifecycle |
| `tests/test_config.py` | Modify | Update assertions for new config shape |
| `tests/test_runner.py` | Modify | Replace Ollama validation tests with server validation test |
| `tests/test_llm_judge.py` | Modify | Update mock target name |
| `tests/test_pipeline.py` | Modify | No changes needed (config loaded from defaults, field access is internal) |
| `tests/test_cli.py` | Modify | Update mock target for validate |

---

### Task 1: Create `llama_server.py` — Server Lifecycle

**Files:**
- Create: `codeevolve/llama_server.py`
- Test: `tests/test_llama_server.py`

- [ ] **Step 1: Write failing tests for LlamaServer**

```python
# tests/test_llama_server.py
import subprocess
from unittest.mock import MagicMock, patch, call
from pathlib import Path

import pytest

from codeevolve.llama_server import LlamaServer
from codeevolve.config import LlamaServerConfig


@pytest.fixture
def server_config():
    return LlamaServerConfig(
        server_path="llama-server",
        model_path="/models/test.gguf",
        port=8080,
        gpu_layers=30,
        context_size=4096,
        threads=8,
    )


@patch("codeevolve.llama_server.urlopen")
@patch("codeevolve.llama_server.subprocess.Popen")
def test_start_launches_process(mock_popen, mock_urlopen, server_config):
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc
    # Health check succeeds immediately
    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"status":"ok"}'
    mock_urlopen.return_value = mock_resp

    server = LlamaServer(server_config)
    server.start()

    args = mock_popen.call_args[0][0]
    assert "llama-server" in args[0]
    assert "-m" in args
    assert "/models/test.gguf" in args
    assert "--port" in args
    assert "8080" in args
    assert "-ngl" in args
    assert "30" in args
    assert "-c" in args
    assert "4096" in args
    assert "-t" in args
    assert "8" in args
    assert "--flash-attn" in args

    server.stop()
    mock_proc.terminate.assert_called_once()


@patch("codeevolve.llama_server.urlopen")
@patch("codeevolve.llama_server.subprocess.Popen")
def test_context_manager(mock_popen, mock_urlopen, server_config):
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc
    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"status":"ok"}'
    mock_urlopen.return_value = mock_resp

    with LlamaServer(server_config) as server:
        assert server._process is not None

    mock_proc.terminate.assert_called_once()


@patch("codeevolve.llama_server.urlopen")
@patch("codeevolve.llama_server.subprocess.Popen")
def test_start_raises_on_early_exit(mock_popen, mock_urlopen, server_config):
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 1  # exited immediately
    mock_proc.stderr = MagicMock()
    mock_proc.stderr.read.return_value = b"error loading model"
    mock_popen.return_value = mock_proc

    server = LlamaServer(server_config)
    with pytest.raises(RuntimeError, match="llama-server exited"):
        server.start()


@patch("codeevolve.llama_server.urlopen")
@patch("codeevolve.llama_server.subprocess.Popen")
def test_no_flash_attn(mock_popen, mock_urlopen, server_config):
    server_config.flash_attn = False
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc
    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"status":"ok"}'
    mock_urlopen.return_value = mock_resp

    server = LlamaServer(server_config)
    server.start()

    args = mock_popen.call_args[0][0]
    assert "--flash-attn" not in args

    server.stop()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_llama_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'codeevolve.llama_server'`

- [ ] **Step 3: Implement `llama_server.py`**

```python
# codeevolve/llama_server.py
from __future__ import annotations

import logging
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen

logger = logging.getLogger(__name__)


class LlamaServer:
    """Manages a llama-server subprocess lifecycle."""

    def __init__(self, config):
        self._config = config
        self._process: subprocess.Popen | None = None

    def _build_args(self) -> list[str]:
        cfg = self._config
        args = [
            cfg.server_path,
            "-m", cfg.model_path,
            "--port", str(cfg.port),
            "-ngl", str(cfg.gpu_layers),
            "-c", str(cfg.context_size),
            "-t", str(cfg.threads),
            "--cache-type-k", cfg.cache_type_k,
            "--cache-type-v", cfg.cache_type_v,
        ]
        if cfg.flash_attn:
            args.append("--flash-attn")
        return args

    def start(self, timeout: float = 120) -> None:
        args = self._build_args()
        logger.info("Starting llama-server: %s", " ".join(args))

        self._process = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # Check it didn't exit immediately
        time.sleep(0.5)
        if self._process.poll() is not None:
            stderr = self._process.stderr.read().decode(errors="replace")
            raise RuntimeError(
                f"llama-server exited immediately (code {self._process.returncode}): {stderr}"
            )

        self._wait_until_ready(timeout)
        logger.info("llama-server ready on port %d", self._config.port)

    def _wait_until_ready(self, timeout: float) -> None:
        url = f"http://localhost:{self._config.port}/health"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                resp = urlopen(url, timeout=2)
                if resp.status == 200:
                    return
            except (URLError, OSError):
                pass
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode(errors="replace")
                raise RuntimeError(
                    f"llama-server exited during startup (code {self._process.returncode}): {stderr}"
                )
            time.sleep(1)
        raise TimeoutError(
            f"llama-server not ready after {timeout}s on port {self._config.port}"
        )

    def stop(self) -> None:
        if self._process is None:
            return
        logger.info("Stopping llama-server (pid %d)", self._process.pid)
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_llama_server.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codeevolve/llama_server.py tests/test_llama_server.py
git commit -m "feat: add llama_server.py for llama-server lifecycle management"
```

---

### Task 2: Update `config.py` — Replace OllamaConfig

**Files:**
- Modify: `codeevolve/config.py`
- Modify: `codeevolve/defaults/evolution.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for new config shape**

Replace `tests/test_config.py` content:

```python
from pathlib import Path

import pytest

from codeevolve.config import CodeEvolveConfig, load_config


def test_load_default_config():
    """Loading with no path returns defaults."""
    config = load_config()
    assert config.llama_server.server_path == "llama-server"
    assert config.llama_server.model_path == "qwen2.5-coder-14b-instruct-q4_k_m.gguf"
    assert config.llama_server.port == 8080
    assert config.llama_server.gpu_layers == 30
    assert config.llama_server.context_size == 4096
    assert config.llama_server.threads == 8
    assert config.llama_server.flash_attn is True
    assert config.evolution.max_iterations == 500
    assert config.fitness.static_analysis_weight == 0.35


def test_load_config_from_yaml(tmp_path: Path):
    """Loading from a YAML file overrides defaults."""
    yaml_content = """
llama_server:
  port: 9090
  gpu_layers: 20
evolution:
  max_iterations: 100
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)

    config = load_config(config_path)
    assert config.llama_server.port == 9090
    assert config.llama_server.gpu_layers == 20
    assert config.evolution.max_iterations == 100
    # non-overridden fields keep defaults
    assert config.llama_server.model_path == "qwen2.5-coder-14b-instruct-q4_k_m.gguf"


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
    assert oe_dict["diff_based_evolution"] is False
    assert oe_dict["llm"]["api_base"] == "http://localhost:8080/v1"
    assert "qwen2.5-coder-14b-instruct-q4_k_m" in oe_dict["llm"]["models"][0]["name"]


def test_config_api_base_property():
    """api_base is derived from port."""
    config = load_config()
    assert config.llama_server.api_base == "http://localhost:8080/v1"


def test_config_model_name_property():
    """model_name is derived from model_path stem."""
    config = load_config()
    assert config.llama_server.model_name == "qwen2.5-coder-14b-instruct-q4_k_m"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -v`
Expected: FAIL — `AttributeError: 'CodeEvolveConfig' object has no attribute 'llama_server'`

- [ ] **Step 3: Update `config.py`**

Replace the `OllamaConfig` dataclass and update all references:

```python
# Replace OllamaConfig (lines 13-17) with:
@dataclass
class LlamaServerConfig:
    server_path: str = "llama-server"
    model_path: str = "qwen2.5-coder-14b-instruct-q4_k_m.gguf"
    port: int = 8080
    gpu_layers: int = 30
    context_size: int = 4096
    threads: int = 8
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    flash_attn: bool = True

    @property
    def api_base(self) -> str:
        return f"http://localhost:{self.port}/v1"

    @property
    def model_name(self) -> str:
        return Path(self.model_path).stem
```

In `EvolutionConfig`, remove the `context_window` field (line 26).

In `CodeEvolveConfig` (line 76), change `ollama: OllamaConfig` to `llama_server: LlamaServerConfig`.

In `to_openevolve_dict()` (lines 129-138), update the `llm` block:
```python
"llm": {
    "api_base": self.llama_server.api_base,
    "api_key": "no-key",
    "models": [
        {"name": self.llama_server.model_name, "weight": 1.0},
    ],
    "temperature": 1.0,
    "max_tokens": 16384,
    "timeout": 300,
},
```

In `_dict_to_config()` (line 181), change:
```python
llama_server = LlamaServerConfig(**data.get("llama_server", {}))
```
And line 191-192:
```python
return CodeEvolveConfig(
    llama_server=llama_server,
    ...
)
```

- [ ] **Step 4: Update `defaults/evolution.yaml`**

Replace the full file:
```yaml
# CodeEvolution default configuration
# All fields have sensible defaults for Rust + llama.cpp on RTX 4060 (8GB VRAM)

llama_server:
  server_path: "llama-server"
  model_path: "qwen2.5-coder-14b-instruct-q4_k_m.gguf"
  port: 8080
  gpu_layers: 30
  context_size: 4096
  threads: 8
  cache_type_k: "q8_0"
  cache_type_v: "q8_0"
  flash_attn: true

evolution:
  max_iterations: 500
  population_size: 100
  num_islands: 3
  migration_interval: 20
  diff_based_evolution: false
  max_fix_attempts: 5

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
  top_quartile_only: false
  num_runs: 3
  dimensions:
    - readability
    - rust_idiomaticity
    - maintainability
    - design
    - elegance
    - clean code
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add codeevolve/config.py codeevolve/defaults/evolution.yaml tests/test_config.py
git commit -m "refactor: replace OllamaConfig with LlamaServerConfig"
```

---

### Task 3: Update `runner.py` — Remove Ollama Functions

**Files:**
- Modify: `codeevolve/runner.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write failing tests for new runner**

Replace the Ollama validation tests in `tests/test_runner.py`:

```python
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Optional

import pytest

from codeevolve.config import load_config
from codeevolve.runner import (
    validate_server,
    build_openevolve_config_yaml,
    format_iteration_line,
    _normalize_llm_diffs,
)


@patch("codeevolve.runner.urlopen")
def test_validate_server_success(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b'{"status":"ok"}'
    mock_urlopen.return_value = mock_resp
    config = load_config()
    errors = validate_server(config)
    assert errors == []


@patch("codeevolve.runner.urlopen")
def test_validate_server_unreachable(mock_urlopen):
    from urllib.error import URLError
    mock_urlopen.side_effect = URLError("Connection refused")
    config = load_config()
    errors = validate_server(config)
    assert len(errors) == 1
    assert "Cannot connect" in errors[0]


def test_build_openevolve_config_yaml(tmp_path: Path):
    config = load_config()
    yaml_path = build_openevolve_config_yaml(config, tmp_path)
    assert yaml_path.exists()
    content = yaml_path.read_text()
    assert "qwen2.5-coder-14b-instruct-q4_k_m" in content
```

Keep the existing `test_format_iteration_line_*` and `TestNormalizeLlmDiffs` tests unchanged.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_runner.py -v`
Expected: FAIL — `ImportError: cannot import name 'validate_server'`

- [ ] **Step 3: Update `runner.py`**

Remove `_ollama_base_url()`, `validate_ollama()`, and `prime_ollama_models()`. Replace with:

```python
def validate_server(config: CodeEvolveConfig) -> list[str]:
    """Check that llama-server is reachable on the configured port."""
    url = f"http://localhost:{config.llama_server.port}/health"
    errors = []
    try:
        resp = urlopen(url, timeout=10)
        if resp.status != 200:
            errors.append(
                f"llama-server health check failed (HTTP {resp.status}) "
                f"on port {config.llama_server.port}"
            )
    except (URLError, OSError) as e:
        errors.append(
            f"Cannot connect to llama-server on port {config.llama_server.port}: {e}"
        )
    return errors
```

No other changes needed in runner.py — `build_openevolve_config_yaml` and `run_evolution` use `config.to_openevolve_dict()` which was already updated in Task 2.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_runner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codeevolve/runner.py tests/test_runner.py
git commit -m "refactor: replace validate_ollama with validate_server for llama-server"
```

---

### Task 4: Update Evaluator Files

**Files:**
- Modify: `codeevolve/evaluator/llm_judge.py`
- Modify: `codeevolve/evaluator/llm_fixer.py`
- Modify: `codeevolve/evaluator/pipeline.py`
- Modify: `tests/test_llm_judge.py`

- [ ] **Step 1: Update `llm_judge.py`**

Rename `_call_ollama` to `_call_llm` and change the api_key:

```python
# Line 68-76: rename function and change api_key
def _call_llm(api_base: str, model: str, prompt: str) -> str:
    """Make a single chat completion call to llama-server."""
    client = OpenAI(base_url=api_base, api_key="no-key")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content or ""
```

Update the call site in `judge_code()` (line 90):
```python
response = _call_llm(api_base, model, prompt)
```

- [ ] **Step 2: Update `llm_fixer.py`**

Change api_key on line 87:
```python
client = OpenAI(base_url=api_base, api_key="no-key")
```

- [ ] **Step 3: Update `pipeline.py`**

Change all `cfg.ollama.*` references to `cfg.llama_server.*`:

Line 160 (`_try_llm_fix`):
```python
cfg.llama_server.api_base, cfg.llama_server.model_name,
```

Lines 449-450 (`_evaluate_candidate`):
```python
api_base=cfg.llama_server.api_base,
model=cfg.llama_server.model_name,
```

- [ ] **Step 4: Update `tests/test_llm_judge.py`**

Change the mock target on line 57:
```python
@patch("codeevolve.evaluator.llm_judge._call_llm")
def test_judge_code_aggregates_runs(mock_call):
```

- [ ] **Step 5: Run all evaluator tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_llm_judge.py tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add codeevolve/evaluator/llm_judge.py codeevolve/evaluator/llm_fixer.py codeevolve/evaluator/pipeline.py tests/test_llm_judge.py
git commit -m "refactor: update evaluator LLM calls from Ollama to llama-server"
```

---

### Task 5: Update `cli.py` — Server Lifecycle & Help Text

**Files:**
- Modify: `codeevolve/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Update `cli.py`**

Update imports (line 31):
```python
from codeevolve.runner import validate_server, run_evolution
from codeevolve.llama_server import LlamaServer
```

Update the `init` command's "Next steps" text (lines 114-119):
```python
    click.echo(f"\nSetup complete! Files created in {codeevolve_dir.relative_to(path)}/")
    click.echo("\nNext steps:")
    click.echo("  1. Download the model GGUF (if not already done):")
    click.echo("       huggingface-cli download Qwen/Qwen2.5-Coder-14B-Instruct-GGUF")
    click.echo("  2. Update .codeevolve/evolution.yaml with paths to llama-server and the .gguf file")
    click.echo("  3. Start evolving:  codeevolve run")
```

Replace the run command's server section (lines 176-184):
```python
    click.echo(f"  Starting llama-server ({config.llama_server.model_path})...")
    try:
        server = LlamaServer(config.llama_server)
        server.start()
    except (RuntimeError, TimeoutError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"  llama-server ready on port {config.llama_server.port}")
    click.echo(f"  Starting evolution ({config.evolution.max_iterations} iterations, population {config.evolution.population_size})")
    click.echo()
```

Wrap the evolution run in a try/finally to stop the server:
```python
    try:
        result = run_evolution(config_path, project_path, initial, evaluator_path)
        click.echo("\n-- Summary " + "-" * 45)
        click.echo(f"  Best score:      {result.best_score:.2f}")
        click.echo(f"  Best candidate:  .codeevolve/output/best/")
        click.echo(f"  Metrics CSV:     .codeevolve/output/metrics.csv")
        click.echo(f"  All candidates:  .codeevolve/output/")
    except KeyboardInterrupt:
        click.echo("\n\nStopped by user. Best result saved to .codeevolve/output/best/")
    finally:
        server.stop()
```

Remove the `validate_ollama` error block and `prime_ollama_models` call entirely.

- [ ] **Step 2: Update `tests/test_cli.py`**

Change the validate mock (line 52-58):
```python
@patch("codeevolve.cli.LlamaServer")
def test_run_server_start_fails(mock_server_cls, cli_runner, tmp_path):
    mock_server_cls.return_value.start.side_effect = RuntimeError("model not found")
    config_path = tmp_path / "evolution.yaml"
    config_path.write_text("llama_server:\n  port: 8080\n")
    result = cli_runner.invoke(main, ["run", "--config", str(config_path)])
    assert result.exit_code != 0
    assert "model not found" in result.output
```

- [ ] **Step 3: Run CLI tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add codeevolve/cli.py tests/test_cli.py
git commit -m "feat: start/stop llama-server from codeevolve run command"
```

---

### Task 6: Update `init_project.py` — README Template

**Files:**
- Modify: `codeevolve/init_project.py`

- [ ] **Step 1: Update the README template**

In `generate_codeevolve_dir()`, update the `readme` string (lines 117-161):

Replace the `evolution.yaml` description:
```
- **evolution.yaml** — Configuration for the evolutionary optimizer. Controls which
  model to use, how many iterations to run, fitness weights, and more.
  Edit this to tune the evolution. All fields have sensible defaults.
```

Replace the "How to Run" section:
```
## How to Run

1. Download the model GGUF if you haven't already
2. Set `server_path` and `model_path` in `evolution.yaml` to point to your
   llama-server binary and .gguf model file
3. Start the evolution: `codeevolve run`

The server is started and stopped automatically by `codeevolve run`.
```

- [ ] **Step 2: Run init tests to verify nothing broke**

Run: `.venv/Scripts/python.exe -m pytest tests/test_init_project.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add codeevolve/init_project.py
git commit -m "docs: update generated README for llama-server"
```

---

### Task 7: Run Full Test Suite

- [ ] **Step 1: Run all tests**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: Fix any remaining failures**

Grep for any remaining `ollama` references in source code (not tests):
```bash
grep -rn "ollama" codeevolve/ --include="*.py" --include="*.yaml"
```
Expected: No matches (except possibly comments that should be updated).

- [ ] **Step 3: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: clean up remaining Ollama references"
```

---

### Task 8: Update CLAUDE.md

- [ ] **Step 1: Update CLAUDE.md**

Update the following sections:
- "Target models" line → mention 14B model + llama-server
- "Tech Stack" → replace "openai (Ollama client)" with "openai (llama-server client)"
- Remove "Ollama for local LLM inference" line, replace with llama-server
- Update the `codeevolve run` command description

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for llama-server migration"
```
