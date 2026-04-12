"""LLM-based error recovery for failed builds/tests."""
from __future__ import annotations

import logging
import re
from typing import Sequence

from openai import OpenAI

logger = logging.getLogger(__name__)

# Temperature escalation per retry attempt (index = attempt number)
_TEMPERATURE_SCHEDULE = [0.3, 0.5, 0.7, 0.7, 0.7]


def build_fix_prompt(
    code: str,
    error_type: str,
    error_output: str,
    previous_attempts: Sequence[str] = (),
    test_context: str = "",
    frozen_context: str = "",
) -> str:
    """Build a prompt asking the LLM to fix a compilation or test error."""
    history_block = ""
    if previous_attempts:
        # Only show the last 2 attempts to keep prompt size reasonable
        recent = previous_attempts[-2:]
        history_block = "\n**Previous fix attempts that FAILED (do NOT repeat these):**\n"
        for i, prev in enumerate(recent, 1):
            history_block += f"\nAttempt {i} (FAILED):\n```rust\n{prev[:3000]}\n```\n"
        history_block += (
            "\nThe above fixes did NOT work. You MUST try a DIFFERENT approach.\n"
        )

    test_block = ""
    if test_context:
        capped = test_context[:4000]
        test_block = f"""
**Tests that must pass (DO NOT modify these — they are outside your control):**
```rust
{capped}
```
"""

    frozen_block = ""
    if frozen_context:
        capped = frozen_context[:3000]
        frozen_block = f"""
**Frozen code (outside the evolvable region — DO NOT change these definitions):**
```rust
{capped}
```
"""

    return f"""You are an expert Rust developer. The following code section failed to {error_type}.

**Error output:**
```
{error_output[:4000]}
```
{frozen_block}{test_block}{history_block}
**Code to fix (ONLY this code is under your control):**
```rust
{code}
```

Fix ONLY the code above so it compiles and passes tests. You MAY add `use` statements if needed to fix compilation errors. Do NOT add tests, modules, main functions, struct definitions, enum definitions, type aliases, derive macros, or trait impls that weren't there before. Do NOT change struct definitions, derive macros, or field visibility that are defined outside the evolvable region. Only fix the errors. Output ONLY the fixed code inside a single ```rust code block.
"""


def parse_code_response(response: str) -> str | None:
    """Extract Rust code from LLM response."""
    # Try to find ```rust ... ``` block
    match = re.search(r"```rust\s*\n(.*?)\n```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic code block
    match = re.search(r"```\s*\n(.*?)\n```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def build_regenerate_prompt(
    original_code: str,
    previous_error: str,
    test_context: str = "",
    frozen_context: str = "",
) -> str:
    """Build a prompt asking the LLM to generate a new improved version of code.

    Unlike build_fix_prompt, this starts from the *original working code* and
    asks for a fresh improvement, using the previous error as guidance on what
    to avoid.
    """
    test_block = ""
    if test_context:
        capped = test_context[:4000]
        test_block = f"""
**Tests that must pass (DO NOT modify these — they are outside your control):**
```rust
{capped}
```
"""

    frozen_block = ""
    if frozen_context:
        capped = frozen_context[:3000]
        frozen_block = f"""
**Frozen code (outside the evolvable region — DO NOT change these definitions):**
```rust
{capped}
```
"""

    error_block = ""
    if previous_error:
        capped = previous_error[:3000]
        error_block = f"""
**A previous attempt to improve this code caused the following error:**
```
{capped}
```
Avoid making changes that would cause similar errors.
"""

    return f"""You are an expert Rust developer. Improve the following Rust code section.

{frozen_block}{test_block}{error_block}
**Code to improve (ONLY this code is under your control):**
```rust
{original_code}
```

Generate an improved version of this code. The code MUST compile with cargo build, pass all tests, and produce zero clippy warnings. Do NOT add tests, modules, main functions, struct definitions, enum definitions, type aliases, derive macros, or trait impls that weren't there before. Output ONLY the improved code inside a single ```rust code block.
"""


def attempt_regenerate(
    original_code: str,
    previous_error: str,
    api_base: str,
    model: str,
    test_context: str = "",
    frozen_context: str = "",
) -> str | None:
    """Ask the LLM to generate a new improved version of code.

    Unlike attempt_fix, this starts from the *original working code* and asks
    for an entirely new attempt, using the previous failure as guidance.
    Returns the new code or None on failure.
    """
    prompt = build_regenerate_prompt(
        original_code, previous_error,
        test_context=test_context, frozen_context=frozen_context,
    )

    try:
        client = OpenAI(base_url=api_base, api_key="no-key")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=16384,
        )
        content = response.choices[0].message.content or ""
        regenerated = parse_code_response(content)

        if regenerated:
            logger.info(
                "LLM regeneration produced %d-char candidate", len(regenerated),
            )
            return regenerated
        else:
            logger.warning("LLM regeneration: failed to parse code (len=%d)", len(content))
            return None

    except Exception as e:
        logger.warning("LLM regeneration call failed: %s", e)
        return None


def attempt_fix(
    code: str,
    error_type: str,
    error_output: str,
    api_base: str,
    model: str,
    previous_attempts: Sequence[str] = (),
    attempt_number: int = 0,
    test_context: str = "",
    frozen_context: str = "",
) -> str | None:
    """Ask the LLM to fix broken code. Returns fixed code or None on failure.

    Parameters
    ----------
    previous_attempts:
        Code strings from earlier fix attempts that failed — included in
        the prompt so the LLM avoids repeating the same mistake.
    attempt_number:
        Zero-based retry index, used to escalate temperature so later
        attempts explore more diverse solutions.
    test_context:
        Source code of tests that must pass — shown to the LLM so it
        understands what the tests expect.
    frozen_context:
        Code outside the evolvable region (e.g. struct definitions) —
        shown so the LLM knows what it must not change.
    """
    prompt = build_fix_prompt(
        code, error_type, error_output, previous_attempts,
        test_context=test_context, frozen_context=frozen_context,
    )
    temperature = _TEMPERATURE_SCHEDULE[min(attempt_number, len(_TEMPERATURE_SCHEDULE) - 1)]

    try:
        client = OpenAI(base_url=api_base, api_key="no-key")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=16384,
        )
        content = response.choices[0].message.content or ""
        fixed_code = parse_code_response(content)

        if fixed_code:
            logger.info(
                "LLM fixer produced %d-char fix (attempt %d, temp=%.1f)",
                len(fixed_code), attempt_number + 1, temperature,
            )
            return fixed_code
        else:
            logger.warning("LLM fixer failed to parse code from response (len=%d)", len(content))
            return None

    except Exception as e:
        logger.warning("LLM fixer call failed: %s", e)
        return None
