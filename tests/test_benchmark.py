import re
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from codeevolve.evaluator.benchmark import (
    measure_binary_size,
    measure_compile_time,
    measure_loc,
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


def test_measure_loc(tmp_path: Path):
    rs_file = tmp_path / "test.rs"
    rs_file.write_text("fn main() {\n    // comment\n    println!(\"hi\");\n\n}\n")
    # non-empty, non-comment lines: "fn main() {", '    println!("hi");', "}"
    assert measure_loc(rs_file) == 3


def test_measure_loc_empty_file(tmp_path: Path):
    rs_file = tmp_path / "empty.rs"
    rs_file.write_text("")
    assert measure_loc(rs_file) == 0


def test_run_user_benchmark_success(tmp_path: Path):
    import sys
    script = tmp_path / "bench.py"
    script.write_text("print('time: 42.5ms')")
    result = run_user_benchmark(
        command=f"{sys.executable} {script}",
        cwd=tmp_path,
        score_regex=r"time: ([\d.]+)ms",
    )
    assert result.success is True
    assert result.score == 42.5


def test_run_user_benchmark_no_regex(tmp_path: Path):
    import sys
    script = tmp_path / "bench.py"
    script.write_text("pass")
    result = run_user_benchmark(command=f"{sys.executable} {script}", cwd=tmp_path, score_regex=None)
    assert result.success is True
    assert result.score == 1.0


def test_run_user_benchmark_failure(tmp_path: Path):
    import sys
    script = tmp_path / "bench.py"
    script.write_text("import sys; sys.exit(1)")
    result = run_user_benchmark(command=f"{sys.executable} {script}", cwd=tmp_path, score_regex=None)
    assert result.success is False
    assert result.score == 0.0


def test_run_user_benchmark_regex_no_match(tmp_path: Path):
    import sys
    script = tmp_path / "bench.py"
    script.write_text("print('no numbers here')")
    result = run_user_benchmark(
        command=f"{sys.executable} {script}",
        cwd=tmp_path,
        score_regex=r"time: ([\d.]+)ms",
    )
    assert result.success is True
    assert result.score == 0.0


def test_run_user_benchmark_timeout_with_partial_output(tmp_path: Path):
    """When a benchmark times out, partial stdout is still searched for a score."""
    partial = "convex_hull/10000 time:   [933.59 ms 941.18 ms 947.50 ms]\n"
    exc = subprocess.TimeoutExpired(cmd="cargo bench", timeout=120)
    exc.stdout = partial.encode()
    with patch("subprocess.run", side_effect=exc):
        result = run_user_benchmark(
            command="cargo bench",
            cwd=tmp_path,
            score_regex=r"time:\s+\[\S+ \S+\s+([\d.]+) ms",
        )
    assert result.success is False
    assert result.score == 941.18


def test_run_user_benchmark_timeout_no_output(tmp_path: Path):
    """When a benchmark times out with no output at all, score is 0."""
    exc = subprocess.TimeoutExpired(cmd="cargo bench", timeout=120)
    exc.stdout = None
    with patch("subprocess.run", side_effect=exc):
        result = run_user_benchmark(
            command="cargo bench",
            cwd=tmp_path,
            score_regex=r"time:\s+\[\S+ \S+\s+([\d.]+) ms",
        )
    assert result.success is False
    assert result.score == 0.0


def test_run_user_benchmark_timeout_no_regex(tmp_path: Path):
    """Timeout without regex returns success=False, score=0."""
    exc = subprocess.TimeoutExpired(cmd="cargo bench", timeout=120)
    exc.stdout = b""
    with patch("subprocess.run", side_effect=exc):
        result = run_user_benchmark(
            command="cargo bench",
            cwd=tmp_path,
            score_regex=None,
        )
    assert result.success is False
    assert result.score == 0.0
