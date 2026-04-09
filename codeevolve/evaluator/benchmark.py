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
