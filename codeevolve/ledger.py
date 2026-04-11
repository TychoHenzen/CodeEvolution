"""Tech debt ledger parser.

Reads TECH_DEBT_LEDGER.md from a Rust project and returns a ranked list of
LedgerEntry objects sorted by combined_score descending.

Expected table format (produced by the tech-debt-score skill)::

    | File Path | Type | Structural | Semantic | Combined | Top Issue | Last Reviewed | Trend |
    |-----------|------|-----------|----------|----------|-----------|---------------|-------|
    | crates/foo/src/bar.rs | prod | 57.70 | 0 | 57.70 | magic_literals (320) | 2026-04-02 | — |

Key columns (0-indexed):
  0 — File Path
  1 — Type  ("prod" or "test")
  4 — Combined  (float)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Column indices in the markdown table
_COL_FILE_PATH = 0
_COL_TYPE = 1
_COL_COMBINED = 4

# Minimum number of pipe-separated cells required for a valid data row
_MIN_COLS = 5


@dataclass(frozen=True)
class LedgerEntry:
    """A single entry from the tech debt ledger."""

    file_path: str
    file_type: str  # "prod" or "test"
    combined_score: float


def parse_ledger(path: Path, prod_only: bool = True) -> list[LedgerEntry]:
    """Parse TECH_DEBT_LEDGER.md and return entries sorted by combined_score descending.

    Args:
        path: Path to the TECH_DEBT_LEDGER.md file.
        prod_only: If True (default), return only entries with file_type == "prod".
                   If False, return all entries regardless of type.

    Returns:
        List of LedgerEntry sorted by combined_score descending.
        Returns an empty list if the file is missing, empty, or malformed.
    """
    if not path.exists():
        logger.debug("Ledger file not found: %s", path)
        return []

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Could not read ledger file %s: %s", path, exc)
        return []

    entries: list[LedgerEntry] = []

    for line in text.splitlines():
        stripped = line.strip()

        # Data rows start and end with '|' and are not separator rows (---|---)
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        if stripped.replace("|", "").replace("-", "").replace(" ", "") == "":
            # Separator row like |---|---|
            continue

        cells = [cell.strip() for cell in stripped.split("|")]
        # split on '|' yields an empty string at index 0 and end; strip them
        cells = cells[1:-1]

        if len(cells) < _MIN_COLS:
            continue

        file_path = cells[_COL_FILE_PATH]
        file_type = cells[_COL_TYPE].lower()
        combined_raw = cells[_COL_COMBINED]

        # Skip the header row
        if file_path.lower() in ("file path", "filepath", "file"):
            continue

        try:
            combined_score = float(combined_raw)
        except (ValueError, TypeError):
            # Non-numeric cell — skip (likely a header or malformed row)
            continue

        if prod_only and file_type != "prod":
            continue

        entries.append(
            LedgerEntry(
                file_path=file_path,
                file_type=file_type,
                combined_score=combined_score,
            )
        )

    return sorted(entries, key=lambda e: e.combined_score, reverse=True)
