"""Tests for codeevolve.scheduler — build_schedule() and ScheduleSlot."""

from __future__ import annotations

import pytest

from codeevolve.ledger import LedgerEntry
from codeevolve.scheduler import ScheduleSlot, build_schedule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(path: str, score: float) -> LedgerEntry:
    return LedgerEntry(file_path=path, file_type="prod", combined_score=score)


# ---------------------------------------------------------------------------
# Edge cases: empty / too-small budget
# ---------------------------------------------------------------------------

def test_empty_entries_returns_empty_schedule():
    result = build_schedule([], total_iterations=500, chunk_size=10)
    assert result == []


def test_total_iterations_less_than_chunk_size_returns_empty():
    entries = [_entry("a.rs", 100.0)]
    result = build_schedule(entries, total_iterations=5, chunk_size=10)
    assert result == []


def test_total_iterations_equals_chunk_size_returns_one_slot():
    entries = [_entry("a.rs", 100.0)]
    result = build_schedule(entries, total_iterations=10, chunk_size=10)
    assert len(result) == 1
    assert result[0].start_iter == 0
    assert result[0].end_iter == 10


def test_total_iterations_zero_returns_empty():
    entries = [_entry("a.rs", 50.0)]
    result = build_schedule(entries, total_iterations=0, chunk_size=10)
    assert result == []


# ---------------------------------------------------------------------------
# Single entry
# ---------------------------------------------------------------------------

def test_single_entry_gets_all_iterations():
    entries = [_entry("only.rs", 99.0)]
    result = build_schedule(entries, total_iterations=500, chunk_size=10)
    assert len(result) == 1
    slot = result[0]
    assert slot.file_path == "only.rs"
    assert slot.start_iter == 0
    assert slot.end_iter == 500


def test_single_entry_truncates_to_chunk_boundary():
    """total_iterations=505 with chunk_size=10 → 50 chunks → 500 used."""
    entries = [_entry("only.rs", 99.0)]
    result = build_schedule(entries, total_iterations=505, chunk_size=10)
    assert len(result) == 1
    assert result[0].end_iter == 500  # truncated to 50 chunks


# ---------------------------------------------------------------------------
# Proportional allocation
# ---------------------------------------------------------------------------

def test_proportional_allocation_exact():
    """A=60%, B=30%, C=10% with 500 iters → 300, 150, 50."""
    entries = [
        _entry("a.rs", 60.0),
        _entry("b.rs", 30.0),
        _entry("c.rs", 10.0),
    ]
    result = build_schedule(entries, total_iterations=500, chunk_size=10)
    assert len(result) == 3
    assert result[0].file_path == "a.rs"
    assert result[0].start_iter == 0
    assert result[0].end_iter == 300

    assert result[1].file_path == "b.rs"
    assert result[1].start_iter == 300
    assert result[1].end_iter == 450

    assert result[2].file_path == "c.rs"
    assert result[2].start_iter == 450
    assert result[2].end_iter == 500


def test_proportional_allocation_all_iterations_accounted_for():
    """Last slot end_iter must equal total usable iterations."""
    entries = [
        _entry("a.rs", 60.0),
        _entry("b.rs", 30.0),
        _entry("c.rs", 10.0),
    ]
    total = 500
    chunk = 10
    result = build_schedule(entries, total_iterations=total, chunk_size=chunk)
    usable = (total // chunk) * chunk
    assert result[-1].end_iter == usable


def test_proportional_allocation_non_divisible_total():
    """507 iters, chunk=10 → usable=500; slots should cover exactly 500."""
    entries = [
        _entry("a.rs", 50.0),
        _entry("b.rs", 50.0),
    ]
    result = build_schedule(entries, total_iterations=507, chunk_size=10)
    assert result[-1].end_iter == 500


# ---------------------------------------------------------------------------
# Contiguous, non-overlapping ranges
# ---------------------------------------------------------------------------

def test_slots_are_contiguous():
    entries = [
        _entry("a.rs", 70.0),
        _entry("b.rs", 20.0),
        _entry("c.rs", 10.0),
    ]
    result = build_schedule(entries, total_iterations=300, chunk_size=10)
    for i in range(1, len(result)):
        assert result[i].start_iter == result[i - 1].end_iter, (
            f"Gap between slot {i-1} and slot {i}"
        )


def test_slots_start_at_zero():
    entries = [_entry("a.rs", 80.0), _entry("b.rs", 20.0)]
    result = build_schedule(entries, total_iterations=100, chunk_size=10)
    assert result[0].start_iter == 0


def test_slots_non_overlapping():
    entries = [_entry(f"f{i}.rs", float(10 - i)) for i in range(5)]
    result = build_schedule(entries, total_iterations=200, chunk_size=10)
    for i in range(1, len(result)):
        assert result[i].start_iter >= result[i - 1].end_iter


# ---------------------------------------------------------------------------
# Minimum 1 chunk per file
# ---------------------------------------------------------------------------

def test_minimum_one_chunk_per_file():
    """Even a tiny score should yield at least one chunk if budget allows."""
    entries = [
        _entry("big.rs", 990.0),
        _entry("tiny.rs", 1.0),
    ]
    result = build_schedule(entries, total_iterations=200, chunk_size=10)
    tiny_slot = next(s for s in result if s.file_path == "tiny.rs")
    assert tiny_slot.end_iter - tiny_slot.start_iter >= 10


# ---------------------------------------------------------------------------
# More files than chunks (tight budget)
# ---------------------------------------------------------------------------

def test_more_files_than_chunks_drops_lowest_scorers():
    """20 files but only 100 iters (10 chunks) with chunk_size=10.

    Only the top 10 files should appear in the schedule.
    """
    entries = [_entry(f"f{i:02d}.rs", float(20 - i)) for i in range(20)]
    result = build_schedule(entries, total_iterations=100, chunk_size=10)
    total_iters = sum(s.end_iter - s.start_iter for s in result)
    assert total_iters == 100  # all 10 chunks used
    # All slots should be from the higher-scoring files (first 10 in list)
    paths_in_schedule = {s.file_path for s in result}
    for slot in result:
        idx = int(slot.file_path[1:3])  # e.g. "f00" → 0
        assert idx < 10, f"Low-scoring file {slot.file_path} snuck into schedule"
    assert len(result) <= 10


def test_more_files_than_chunks_all_iterations_accounted():
    """Ensure no iterations are lost when files must be dropped."""
    entries = [_entry(f"f{i}.rs", float(10 - i)) for i in range(10)]
    # Only 3 chunks total
    result = build_schedule(entries, total_iterations=30, chunk_size=10)
    total_iters = sum(s.end_iter - s.start_iter for s in result)
    assert total_iters == 30


# ---------------------------------------------------------------------------
# Leftover redistribution
# ---------------------------------------------------------------------------

def test_leftover_goes_to_highest_scoring_files():
    """When rounding causes leftover chunks, top files get the extras.

    3 files, scores 3/3/4 → 10 total, 100 iters, chunk=10 → 10 chunks.
    Raw: 3→30%, 3→30%, 4→40% → 3, 3, 4 chunks → exact, no leftover.

    Use scores 1/1/1 with 10 chunks → 3.33 each → floor=3 each → 9 used,
    leftover=1 → extra chunk goes to file[0].
    """
    entries = [_entry(f"f{i}.rs", 1.0) for i in range(3)]
    result = build_schedule(entries, total_iterations=100, chunk_size=10)
    iters = [s.end_iter - s.start_iter for s in result]
    total = sum(iters)
    assert total == 100  # all 10 chunks used
    # The first (highest-scoring, but equal here) file should have gotten extra
    assert iters[0] >= iters[1]
    assert iters[0] >= iters[2]


def test_leftover_distribution_order():
    """Leftover chunks should be distributed top-down."""
    # 7 files, equal scores, 100 iters, chunk=10 → 10 chunks
    # Each gets floor(10/7)=1 chunk → 7 used, leftover=3
    # Top 3 files get an extra chunk each → [2,2,2,1,1,1,1]
    entries = [_entry(f"f{i}.rs", 1.0) for i in range(7)]
    result = build_schedule(entries, total_iterations=100, chunk_size=10)
    iters = [s.end_iter - s.start_iter for s in result]
    assert sum(iters) == 100
    assert iters[0] == 20
    assert iters[1] == 20
    assert iters[2] == 20
    assert iters[3] == 10
    assert iters[4] == 10
    assert iters[5] == 10
    assert iters[6] == 10


# ---------------------------------------------------------------------------
# Ordering: highest-scoring file first
# ---------------------------------------------------------------------------

def test_schedule_ordered_highest_score_first():
    """Entries fed in descending order → slots appear in same order."""
    entries = [
        _entry("high.rs", 80.0),
        _entry("mid.rs", 15.0),
        _entry("low.rs", 5.0),
    ]
    result = build_schedule(entries, total_iterations=200, chunk_size=10)
    assert result[0].file_path == "high.rs"
    assert result[-1].file_path == "low.rs"


# ---------------------------------------------------------------------------
# ScheduleSlot dataclass
# ---------------------------------------------------------------------------

def test_schedule_slot_fields():
    slot = ScheduleSlot(file_path="a.rs", start_iter=0, end_iter=100)
    assert slot.file_path == "a.rs"
    assert slot.start_iter == 0
    assert slot.end_iter == 100


def test_schedule_slot_is_frozen():
    slot = ScheduleSlot(file_path="a.rs", start_iter=0, end_iter=100)
    with pytest.raises(Exception):
        slot.start_iter = 50  # type: ignore[misc]


def test_schedule_slot_equality():
    a = ScheduleSlot("a.rs", 0, 100)
    b = ScheduleSlot("a.rs", 0, 100)
    assert a == b
