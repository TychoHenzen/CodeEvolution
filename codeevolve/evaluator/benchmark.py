from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    success: bool
    score: float
    output: str = ""


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

    The regex may contain extra capture groups, including time units from
    Criterion output. We look for the first capture group that parses as a
    float, then normalize it if an adjacent or nearby capture group is a
    recognized time unit.
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

    groups = list(match.groups())
    numeric_groups: list[tuple[int, float]] = []
    for idx, group in enumerate(groups):
        if group is None:
            continue
        try:
            numeric_groups.append((idx, float(group)))
        except ValueError:
            continue

    if not numeric_groups:
        logger.warning(
            "custom_command_score_regex matched text but no numeric capture was found; "
            "pattern=%r text=%r",
            regex,
            text[:200],
        )
        return 0.0

    # Criterion reports time as fastest / average / slowest. When the regex
    # captures multiple numeric values, the average is the middle capture.
    value_index, score = numeric_groups[len(numeric_groups) // 2]

    unit: str | None = None
    for idx in (value_index + 1, value_index - 1):
        if 0 <= idx < len(groups):
            candidate = groups[idx]
            if candidate in _TIME_UNIT_TO_MS:
                unit = candidate
                break

    if unit is None:
        for group in groups:
            if group in _TIME_UNIT_TO_MS:
                unit = group
                break

    if unit is not None:
        score *= _TIME_UNIT_TO_MS[unit]

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
        # shell=True: command comes from the user's own evolution.yaml config,
        # same trust level as a Makefile or shell script.
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
