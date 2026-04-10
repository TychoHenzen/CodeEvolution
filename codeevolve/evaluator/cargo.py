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
_LINT_CATEGORIES = {
    # correctness
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
    # style
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
    tests_passed: int = 0
    tests_failed: int = 0
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
    try:
        data = json.loads(raw_json)
        if isinstance(data, list):
            items = data
        else:
            items = [data]
    except json.JSONDecodeError:
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
        if msg.get("level") != "warning":
            continue
        code_info = msg.get("code")
        if not code_info or not code_info.get("code"):
            continue
        code = code_info["code"]
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
        success=proc.returncode == 0 or len(warnings) >= 0,
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
