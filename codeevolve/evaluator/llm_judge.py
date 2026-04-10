from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import dataclass, field

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class LlmJudgment:
    dimension_scores: dict[str, float] = field(default_factory=dict)
    combined_score: float = 0.0


def build_judgment_prompt(code: str, dimensions: list[str]) -> str:
    """Build a prompt asking the LLM to judge code quality on given dimensions."""
    dim_list = "\n".join(f"- **{d}**: a decimal number from 0.0 to 1.0" for d in dimensions)
    return f"""You are an expert Rust code reviewer. Evaluate the following code on each dimension using a score from 0.0 to 1.0 (0.0=terrible, 0.5=average, 1.0=excellent).

IMPORTANT: Scores MUST be fractional decimals, not integers. For example 0.65 or 0.82, never just 0 or 1.

Think step by step about the code quality, then provide your scores as a JSON object.

**Dimensions:**
{dim_list}

**Code:**
```rust
{code}
```

Respond with your reasoning first, then a JSON code block containing only the dimension scores. Example:
```json
{{{", ".join(f'"{d}": 0.65' for d in dimensions)}}}
```"""


def parse_judgment_response(response: str, dimensions: list[str]) -> dict[str, float]:
    """Extract dimension scores from LLM response text."""
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
                scores[dim] = max(0.0, min(1.0, float(val)))
    return scores


def _call_llm(api_base: str, model: str, prompt: str) -> str:
    """Make a single chat completion call to llama-server."""
    client = OpenAI(base_url=api_base, api_key="no-key")
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
    for run_idx in range(num_runs):
        response = _call_llm(api_base, model, prompt)
        scores = parse_judgment_response(response, dimensions)
        logger.info("LLM judge run %d/%d: %s", run_idx + 1, num_runs, scores)
        if not scores:
            logger.warning(
                "LLM judge run %d: failed to parse scores from response (len=%d)",
                run_idx + 1, len(response),
            )
        for dim in dimensions:
            if dim in scores:
                all_scores[dim].append(scores[dim])
    dimension_medians = {}
    for dim in dimensions:
        vals = all_scores[dim]
        if vals:
            dimension_medians[dim] = statistics.median(vals)
        else:
            dimension_medians[dim] = 0.0
    combined = statistics.mean(dimension_medians.values()) if dimension_medians else 0.0
    logger.info("LLM judge result: medians=%s, combined=%.2f", dimension_medians, combined)
    return LlmJudgment(dimension_scores=dimension_medians, combined_score=combined)
