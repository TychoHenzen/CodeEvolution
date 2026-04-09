from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class LlmJudgment:
    dimension_scores: dict[str, float] = field(default_factory=dict)
    combined_score: float = 0.0


def build_judgment_prompt(code: str, dimensions: list[str]) -> str:
    """Build a prompt asking the LLM to judge code quality on given dimensions."""
    dim_list = "\n".join(f"- **{d}**: score 1-5" for d in dimensions)
    return f"""You are an expert Rust code reviewer. Evaluate the following code on each dimension using a 1-5 Likert scale (1=poor, 5=excellent).

Think step by step about the code quality, then provide your scores as a JSON object.

**Dimensions:**
{dim_list}

**Code:**
```rust
{code}
```

Respond with your reasoning first, then a JSON code block containing only the dimension scores. Example:
```json
{{{", ".join(f'"{d}": 3' for d in dimensions)}}}
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
                scores[dim] = max(1, min(5, int(val)))
    return scores


def _call_ollama(api_base: str, model: str, prompt: str) -> str:
    """Make a single chat completion call to Ollama."""
    client = OpenAI(base_url=api_base, api_key="ollama")
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
    for _ in range(num_runs):
        response = _call_ollama(api_base, model, prompt)
        scores = parse_judgment_response(response, dimensions)
        for dim in dimensions:
            if dim in scores:
                all_scores[dim].append(scores[dim])
    dimension_medians = {}
    for dim in dimensions:
        vals = all_scores[dim]
        if vals:
            dimension_medians[dim] = statistics.median(vals)
        else:
            dimension_medians[dim] = 1.0
    combined = statistics.mean(dimension_medians.values()) if dimension_medians else 1.0
    return LlmJudgment(dimension_scores=dimension_medians, combined_score=combined)
