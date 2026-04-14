"""Rotation schedule builder.

Allocates evolution iterations proportionally to tech debt scores across files,
chunked to discrete ``chunk_size`` boundaries so the runner can checkpoint
cleanly between slots.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from collections.abc import Mapping

from codeevolve.ledger import LedgerEntry


@dataclass(frozen=True)
class ScheduleSlot:
    """A contiguous block of iterations assigned to a single file."""

    file_path: str
    start_iter: int
    end_iter: int  # exclusive


_LENGTH_BIAS_SCALE = 5.0


def _length_multiplier(line_count: int) -> float:
    """Return a gentle multiplier that grows with file length.

    The multiplier is intentionally shallow so size nudges the schedule rather
    than overwhelming the existing debt-score and dependency weighting.
    """
    return 1.0 + (math.log1p(max(1, line_count)) / _LENGTH_BIAS_SCALE)


def _apply_length_bias(
    entries: list[LedgerEntry],
    file_lengths: Mapping[str, int] | None,
) -> list[LedgerEntry]:
    """Return a copy of *entries* with line-count bias folded into scores."""
    if not file_lengths:
        return entries

    weighted: list[LedgerEntry] = []
    for entry in entries:
        line_count = file_lengths.get(entry.file_path)
        if line_count is None:
            weighted.append(entry)
            continue
        weighted_score = entry.combined_score * _length_multiplier(line_count)
        weighted.append(
            LedgerEntry(
                file_path=entry.file_path,
                file_type=entry.file_type,
                combined_score=weighted_score,
            )
        )
    return weighted


def build_schedule(
    entries: list[LedgerEntry],
    total_iterations: int,
    chunk_size: int = 10,
    file_lengths: Mapping[str, int] | None = None,
    shuffle: bool = False,
) -> list[ScheduleSlot]:
    """Build a rotation schedule proportional to tech debt scores.

    Args:
        entries: LedgerEntry list, expected sorted by combined_score descending.
                 Files with zero or negative scores are treated as if they have
                 a negligible score but are still guaranteed at least one chunk.
        total_iterations: Total iteration budget.
        chunk_size: Granularity of allocation; every file gets a whole number of
                    chunks.  Must be >= 1.
        file_lengths: Optional mapping of file_path -> line count. When present,
            longer files get a gentle score boost before chunk allocation.
        shuffle: When True, randomly permute slot ordering weighted by iteration
            count so restarts visit files in a different order each time.

    Returns:
        List of :class:`ScheduleSlot` with non-overlapping, contiguous ranges.
        When *shuffle* is False, highest-scoring file first.  Empty when the
        budget is too small to fit even a single chunk.
    """
    if not entries:
        return []

    entries = _apply_length_bias(entries, file_lengths)

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
    # First pass: collect (file_path, iters) pairs without cursor positions
    raw_slots: list[tuple[str, int]] = []
    for entry, chunks in zip(entries, floored):
        if chunks == 0:
            continue
        raw_slots.append((entry.file_path, chunks * chunk_size))

    # Optionally shuffle: weighted random permutation so restarts don't
    # always evolve files in the same order.  Weight = iteration count
    # (proportional to importance) so higher-priority files still tend to
    # appear earlier, but with randomisation.
    if shuffle and len(raw_slots) > 1:
        raw_slots = _weighted_shuffle(raw_slots)

    slots: list[ScheduleSlot] = []
    cursor = 0
    for file_path, iters in raw_slots:
        slots.append(
            ScheduleSlot(
                file_path=file_path,
                start_iter=cursor,
                end_iter=cursor + iters,
            )
        )
        cursor += iters

    return slots


def _weighted_shuffle(items: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Randomly permute items with probability proportional to iteration count.

    Files with more iterations (= higher importance) are more likely to appear
    earlier, but every ordering is possible.  This avoids the restart-bias
    problem where a deterministic sort always evolves the same files first.
    """
    result: list[tuple[str, int]] = []
    remaining = list(items)
    while remaining:
        weights = [iters for _, iters in remaining]
        # random.choices returns a list; pick one item
        [chosen] = random.choices(remaining, weights=weights, k=1)
        result.append(chosen)
        remaining.remove(chosen)
    return result


def build_roundrobin_schedule(
    file_paths: list[str],
    total_iterations: int,
    chunk_size: int = 10,
    file_lengths: Mapping[str, int] | None = None,
    shuffle: bool = False,
) -> list[ScheduleSlot]:
    """Build an equal-weight round-robin schedule for files without debt scores.

    Each file receives an equal share of the iteration budget, rounded down to
    the nearest chunk boundary.  Any leftover chunks (from floor rounding) are
    distributed one-at-a-time to the first files in the list.  When
    *file_lengths* is provided, longer files get a gentle bonus so the schedule
    spends more time on larger source files.

    Args:
        file_paths: Ordered list of relative file paths to schedule.
        total_iterations: Total iteration budget.
        chunk_size: Granularity of allocation; every file gets a whole number of
                    chunks.  Must be >= 1.
        file_lengths: Optional mapping of file_path -> line count. When present,
            longer files get a gentle score boost before chunk allocation.
        shuffle: When True, randomly permute slot ordering so restarts visit
            files in a different order each time.

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
    return build_schedule(
        fake_entries,
        total_iterations=total_iterations,
        chunk_size=chunk_size,
        file_lengths=file_lengths,
        shuffle=shuffle,
    )
