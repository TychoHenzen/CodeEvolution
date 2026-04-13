from __future__ import annotations

import json
import logging
import re
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class LlmJudgment:
    dimension_scores: dict[str, float] = field(default_factory=dict)
    combined_score: float = 0.0


def get_git_diff(file_path: Path) -> str:
    """Get the git diff of a file against HEAD."""
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD", "--", str(file_path)],
            capture_output=True, text=True, timeout=10,
            cwd=file_path.parent,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("git diff failed: %s", exc)
        return ""


def build_judgment_prompt(diff: str, dimensions: list[str]) -> str:
    """Build a prompt asking the LLM to judge whether a diff is an improvement."""
    dim_list = "\n".join(
        f"- **{d}**: a decimal number from -0.99 to +0.99 "
        f"(negative=regression, 0=neutral, positive=improvement)"
        for d in dimensions
    )
    return f"""You are an expert Rust code reviewer. You are given a diff of changes to a Rust source file. Evaluate whether this diff is an IMPROVEMENT or a REGRESSION on each dimension.

Score each dimension from -0.99 to +0.99:
- Negative scores mean the change made things WORSE on that dimension
- Zero means no meaningful change
- Positive scores mean the change IMPROVED that dimension
- Use the full range: 0.3 is a minor improvement, 0.7 is a strong improvement, -0.5 is a significant regression

IMPORTANT: Scores MUST be fractional decimals, not integers. For example 0.45 or -0.32, never just 0 or 1.

Think step by step about what the diff actually changes, then provide your scores as a JSON object.

**Dimensions:**
{dim_list}

**Diff:**
```diff
{diff}
```

Respond with a JSON code block containing only the dimension scores. Example:
```json
{{{", ".join(f'"{d}": 0.45' for d in dimensions)}}}
```"""


def parse_judgment_response(response: str, dimensions: list[str]) -> dict[str, float]:
    """Extract dimension scores from LLM response text.

    Scores are in [-0.99, +0.99] range.
    """
    json_match = re.search(r"```json\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
    text_to_parse = json_match.group(1) if json_match else response

    try:
        data = json.loads(text_to_parse)
    except json.JSONDecodeError:
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
                scores[dim] = max(-0.99, min(0.99, float(val)))
    return scores


def _call_llm(api_base: str, model: str, prompt: str) -> str:
    """Make a single chat completion call."""
    client = OpenAI(base_url=api_base, api_key="no-key")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


def _normalize_score(raw: float) -> float:
    """Map [-0.99, +0.99] to [0, 1].  0 -> 0.5, +0.99 -> ~1.0, -0.99 -> ~0.0."""
    return (raw + 1.0) / 2.0


def judge_code(
    file_path: Path,
    api_base: str,
    model: str,
    dimensions: list[str],
    num_runs: int = 1,
) -> LlmJudgment:
    """Run diff-based LLM judgment and aggregate via median.

    Gets the git diff of the file against HEAD, asks the LLM to score
    the diff as improvement/regression on each dimension ([-0.99, +0.99]),
    then normalizes to [0, 1] for the combined score.
    """
    diff = get_git_diff(file_path)
    if not diff:
        logger.info("LLM judge: no git diff, returning neutral score 0.5")
        return LlmJudgment(
            dimension_scores={d: 0.5 for d in dimensions},
            combined_score=0.5,
        )

    prompt = build_judgment_prompt(diff, dimensions)
    all_scores: dict[str, list[float]] = {d: [] for d in dimensions}
    for run_idx in range(num_runs):
        response = _call_llm(api_base, model, prompt)
        scores = parse_judgment_response(response, dimensions)
        logger.info("LLM judge run %d/%d (raw): %s", run_idx + 1, num_runs, scores)
        if not scores:
            logger.warning(
                "LLM judge run %d: failed to parse scores from response (len=%d)",
                run_idx + 1, len(response),
            )
        for dim in dimensions:
            if dim in scores:
                all_scores[dim].append(scores[dim])

    # Median of raw scores per dimension, then normalize to [0, 1]
    dimension_normalized: dict[str, float] = {}
    raw_medians: dict[str, float] = {}
    for dim in dimensions:
        vals = all_scores[dim]
        if vals:
            raw = statistics.median(vals)
            raw_medians[dim] = raw
            dimension_normalized[dim] = _normalize_score(raw)
        else:
            raw_medians[dim] = 0.0
            dimension_normalized[dim] = 0.5

    combined_raw = statistics.mean(raw_medians.values()) if raw_medians else 0.0
    combined = _normalize_score(combined_raw)
    logger.info(
        "LLM judge result: raw_medians=%s, combined_raw=%.3f, combined=%.3f",
        raw_medians, combined_raw, combined,
    )
    return LlmJudgment(dimension_scores=dimension_normalized, combined_score=combined)
