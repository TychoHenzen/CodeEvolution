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
    import subprocess
    subprocess.run(["cargo", "build"], cwd=sample_crate, capture_output=True)
    size_bytes = measure_binary_size(sample_crate)
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
    assert result.score == 1.0


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
    assert result.score == 0.0
