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
    for f in debug_dir.iterdir():
        if f.is_file() and (
            f.suffix in (".rlib", ".exe", ".dll", ".so", ".dylib", "")
            and not f.name.startswith(".")
            and f.stat().st_size > 1000
        ):
            total += f.stat().st_size
    return total


def measure_loc(program_path: Path) -> int:
    """Count non-empty, non-comment lines in the evolved file."""
    total = 0
    for line in program_path.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("//"):
            total += 1
    return total


_TIME_UNIT_TO_MS = {"s": 1000.0, "ms": 1.0, "µs": 0.001, "us": 0.001, "ns": 0.000001}


def _extract_score(regex: str, text: str) -> float:
    """Extract a numeric score from text using a regex.

    If the regex has one capture group, the raw number is returned.
    If it has two groups and the second is a time unit (s/ms/µs/us/ns),
    the value is normalized to milliseconds. This handles Criterion
    output where different benchmarks report in different units.
    """
    match = re.search(regex, text)
    if not match:
        return 0.0
    score = float(match.group(1))
    if match.lastindex and match.lastindex >= 2:
        unit = match.group(2)
        score *= _TIME_UNIT_TO_MS.get(unit, 1.0)
    return score


def run_user_benchmark(
    command: str,
    cwd: Path,
    score_regex: Optional[str] = None,
    timeout: int = 120,
) -> BenchmarkResult:
    """Run user-provided benchmark command, optionally extract a score via regex.

    If the process times out, partial stdout is still searched for a score.
    This handles benchmarks where some cases complete but slower ones time out.

    If the regex has two capture groups (value + unit), the score is
    normalized to milliseconds using standard time unit conversions.
    """
    try:
        proc = subprocess.run(
            command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        stdout = proc.stdout
        returncode = proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        returncode = 124

    if score_regex is None:
        return BenchmarkResult(
            success=returncode == 0,
            score=1.0 if returncode == 0 else 0.0,
            output=stdout,
        )
    score = _extract_score(score_regex, stdout)
    return BenchmarkResult(
        success=returncode == 0,
        score=score,
        output=stdout,
    )
