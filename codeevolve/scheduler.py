"""Rotation schedule builder.

Allocates evolution iterations proportionally to tech debt scores across files,
chunked to discrete ``chunk_size`` boundaries so the runner can checkpoint
cleanly between slots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from codeevolve.ledger import LedgerEntry


@dataclass(frozen=True)
class ScheduleSlot:
    """A contiguous block of iterations assigned to a single file."""

    file_path: str
    start_iter: int
    end_iter: int  # exclusive


def build_schedule(
    entries: list[LedgerEntry],
    total_iterations: int,
    chunk_size: int = 10,
) -> list[ScheduleSlot]:
    """Build a rotation schedule proportional to tech debt scores.

    Args:
        entries: LedgerEntry list, expected sorted by combined_score descending.
                 Files with zero or negative scores are treated as if they have
                 a negligible score but are still guaranteed at least one chunk.
        total_iterations: Total iteration budget.
        chunk_size: Granularity of allocation; every file gets a whole number of
                    chunks.  Must be >= 1.

    Returns:
        List of :class:`ScheduleSlot` with non-overlapping, contiguous ranges.
        Highest-scoring file first.  Empty when the budget is too small to fit
        even a single chunk.
    """
    if not entries:
        return []

    # Ensure highest-scoring files come first regardless of caller order.
    entries = sorted(entries, key=lambda e: e.combined_score, reverse=True)

    # How many chunks fit in the total budget?
    total_chunks = total_iterations // chunk_size
    if total_chunks == 0:
        return []

    n = len(entries)

    # --- Proportional allocation -------------------------------------------
    total_score = sum(e.combined_score for e in entries)

    if total_score <= 0:
        # All scores are zero or negative: distribute equally
        raw_chunks = [total_chunks / n] * n
    else:
        raw_chunks = [
            (e.combined_score / total_score) * total_chunks for e in entries
        ]

    # Floor each file to at least 1 chunk
    floored: list[int] = [max(1, math.floor(rc)) for rc in raw_chunks]

    # If the minimum-1-chunk constraint pushes us over budget, trim from the
    # lowest-scoring files first (they're at the tail end of the sorted list).
    used = sum(floored)
    if used > total_chunks:
        # Trim excess from the tail
        excess = used - total_chunks
        for i in range(n - 1, -1, -1):
            if excess == 0:
                break
            reducible = floored[i] - 1  # can reduce to 0 only if we must
            cut = min(excess, reducible)
            floored[i] -= cut
            excess -= cut
        # If there's still excess (every file was already at 1 chunk and we
        # have more files than chunks), drop trailing files entirely.
        if excess > 0:
            # Mark files beyond what we can fit as having 0 chunks; we'll
            # keep only the files that fit.
            # Walk from the tail and zero out until excess is cleared.
            for i in range(n - 1, -1, -1):
                if excess == 0:
                    break
                if floored[i] > 0:
                    excess -= floored[i]
                    floored[i] = 0

    # --- Leftover redistribution -------------------------------------------
    # After flooring, some chunks may be unused.  Give them to highest-scoring
    # files first (1 extra chunk each until the surplus is gone).
    used = sum(floored)
    leftover = total_chunks - used

    i = 0
    while leftover > 0 and i < n:
        if floored[i] > 0:  # only boost files that already have an allocation
            floored[i] += 1
            leftover -= 1
        i += 1

    # --- Build slots -------------------------------------------------------
    slots: list[ScheduleSlot] = []
    cursor = 0
    for entry, chunks in zip(entries, floored):
        if chunks == 0:
            continue
        iters = chunks * chunk_size
        slots.append(
            ScheduleSlot(
                file_path=entry.file_path,
                start_iter=cursor,
                end_iter=cursor + iters,
            )
        )
        cursor += iters

    return slots


def build_roundrobin_schedule(
    file_paths: list[str],
    total_iterations: int,
    chunk_size: int = 10,
) -> list[ScheduleSlot]:
    """Build an equal-weight round-robin schedule for files without debt scores.

    Each file receives an equal share of the iteration budget, rounded down to
    the nearest chunk boundary.  Any leftover chunks (from floor rounding) are
    distributed one-at-a-time to the first files in the list.

    Args:
        file_paths: Ordered list of relative file paths to schedule.
        total_iterations: Total iteration budget.
        chunk_size: Granularity of allocation; every file gets a whole number of
                    chunks.  Must be >= 1.

    Returns:
        List of :class:`ScheduleSlot` with non-overlapping, contiguous ranges,
        one slot per file (files that end up with 0 chunks are omitted).
        Empty when the budget is too small to fit even a single chunk or when
        *file_paths* is empty.
    """
    if not file_paths:
        return []

    # Reuse build_schedule() with equal scores — avoids duplicating allocation
    # logic.  Score value doesn't matter as long as they're all identical.
    fake_entries = [
        LedgerEntry(file_path=fp, file_type="prod", combined_score=1.0)
        for fp in file_paths
    ]
    return build_schedule(fake_entries, total_iterations=total_iterations, chunk_size=chunk_size)
