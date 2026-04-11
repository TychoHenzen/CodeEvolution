from __future__ import annotations

import logging
import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    success: bool
    score: float
    output: str = ""


def measure_compile_time(
    project_path: Path, cargo_path: str = "cargo", runs: int = 1,
) -> float:
    """Clean-build the project and return wall-clock seconds.

    A single clean build is sufficient for comparing candidates — the noise
    between runs (~10-20%) is small relative to algorithmic code changes,
    and each extra run costs a full cargo clean + build cycle (~30s).
    """
    times: list[float] = []
    for i in range(runs):
        subprocess.run(
            [cargo_path, "clean"], cwd=project_path, capture_output=True, timeout=30,
        )
        start = time.monotonic()
        subprocess.run(
            [cargo_path, "build"], cwd=project_path, capture_output=True, timeout=120,
        )
        elapsed = time.monotonic() - start
        times.append(elapsed)
    result = statistics.median(times) if len(times) > 1 else times[0]
    logger.info("compile_time: runs=%s, result=%.2fs", ["%.2f" % t for t in times], result)
    return result


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


def measure_release_binary_size(
    project_path: Path,
    binary_package: str,
    cargo_path: str = "cargo",
    target_dir: Optional[str] = None,
    upx_path: Optional[str] = None,
    upx_args: Optional[list[str]] = None,
) -> int:
    """Build a specific package in release mode and return binary size.

    If upx_path is provided, compress the binary and return the compressed size.
    Returns size in bytes. Returns 0 if build fails.
    """
    import platform

    # Build in release mode for the specific package
    cmd = [cargo_path, "build", "--release", "-p", binary_package]
    if target_dir:
        cmd.extend(["--target-dir", target_dir])
    try:
        proc = subprocess.run(
            cmd, cwd=project_path, capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            logger.warning(
                "Release build failed for package %s: %s",
                binary_package, proc.stderr[:500],
            )
            return 0
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("Release build error for package %s: %s", binary_package, e)
        return 0

    # Locate the binary
    target_base = Path(target_dir) if target_dir else project_path / "target"
    release_dir = target_base / "release"

    # Try both .exe (Windows) and no-extension (Unix)
    is_windows = platform.system() == "Windows"
    if is_windows:
        binary_path = release_dir / f"{binary_package}.exe"
    else:
        binary_path = release_dir / binary_package

    # Fallback: check both variants
    if not binary_path.exists():
        alt = release_dir / (f"{binary_package}.exe" if not is_windows else binary_package)
        if alt.exists():
            binary_path = alt
        else:
            logger.warning(
                "Release binary not found at %s (or %s)",
                binary_path, alt,
            )
            return 0

    # Optional UPX compression
    if upx_path:
        upx_cmd = [upx_path] + (upx_args or []) + [str(binary_path)]
        try:
            proc = subprocess.run(
                upx_cmd, capture_output=True, text=True, timeout=120,
            )
            if proc.returncode != 0:
                logger.warning("UPX compression failed: %s", proc.stderr[:300])
                # Still return the uncompressed size
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("UPX error: %s", e)

    size = binary_path.stat().st_size
    logger.info(
        "Release binary size for %s: %d bytes (%s)",
        binary_package, size,
        "UPX-compressed" if upx_path else "uncompressed",
    )
    return size


def find_release_binary_size(
    project_path: Path,
    binary_package: str,
    target_dir: Optional[str] = None,
    upx_path: Optional[str] = None,
    upx_args: Optional[list[str]] = None,
) -> int:
    """Measure the size of an already-built release binary, optionally after UPX.

    Assumes the release binary already exists (e.g. from cargo clippy --release).
    Returns size in bytes, or 0 if the binary is not found.
    """
    import platform

    target_base = Path(target_dir) if target_dir else project_path / "target"
    release_dir = target_base / "release"

    is_windows = platform.system() == "Windows"
    if is_windows:
        binary_path = release_dir / f"{binary_package}.exe"
    else:
        binary_path = release_dir / binary_package

    if not binary_path.exists():
        alt = release_dir / (f"{binary_package}.exe" if not is_windows else binary_package)
        if alt.exists():
            binary_path = alt
        else:
            logger.warning(
                "Release binary not found at %s (or %s)", binary_path, alt,
            )
            return 0

    if upx_path:
        upx_cmd = [upx_path] + (upx_args or []) + [str(binary_path)]
        try:
            proc = subprocess.run(
                upx_cmd, capture_output=True, text=True, timeout=120,
            )
            if proc.returncode != 0:
                logger.warning("UPX compression failed: %s", proc.stderr[:300])
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("UPX error: %s", e)

    size = binary_path.stat().st_size
    logger.info(
        "Release binary size for %s: %d bytes (%s)",
        binary_package, size,
        "UPX-compressed" if upx_path else "uncompressed",
    )
    return size


def measure_loc(program_path: Path) -> int:
    """Count non-empty, non-comment lines in the evolved file."""
    total = 0
    for line in program_path.read_text(encoding="utf-8").splitlines():
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
    try:
        match = re.search(regex, text)
    except re.PatternError as e:
        raise ValueError(
            f"Invalid custom_command_score_regex in evolution.yaml: {e}\n"
            f"  Pattern: {regex!r}\n"
            f"  Hint: If you edited evolution.yaml by hand, ensure the regex "
            f"is inside single quotes, e.g.:\n"
            f"    custom_command_score_regex: 'time:\\s+(\\d+) (ms|us|ns)'"
        ) from e
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
