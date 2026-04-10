"""Tests for per-generation CSV metrics logging in the evaluator template."""
import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

from codeevolve.evaluator.pipeline import EvaluationResult


def _make_csv_logger(csv_path: Path):
    """Build the same _log_metrics_csv / _generation state the rendered evaluator has."""
    fields = [
        "generation", "combined_score", "passed_gates", "static_score",
        "perf_score", "llm_score", "clippy_warnings", "compile_time",
        "binary_size", "tests_passed", "loc",
    ]
    state = {"generation": 0}

    def log(metrics: dict):
        state["generation"] += 1
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow({"generation": state["generation"], **metrics})

    return log


def _metrics_dict(**overrides) -> dict:
    defaults = {
        "combined_score": 0.75,
        "passed_gates": 1.0,
        "static_score": -3.0,
        "perf_score": -100.0,
        "llm_score": 0.5,
        "clippy_warnings": 2.0,
        "compile_time": 2.0,
        "binary_size": 500_000.0,
        "tests_passed": 10.0,
        "loc": 42.0,
    }
    defaults.update(overrides)
    return defaults


def test_csv_created_with_headers(tmp_path):
    csv_path = tmp_path / "output" / "metrics.csv"
    log = _make_csv_logger(csv_path)

    log(_metrics_dict())

    assert csv_path.exists()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["generation"] == "1"
    assert "combined_score" in reader.fieldnames
    assert "passed_gates" in reader.fieldnames
    assert "loc" in reader.fieldnames


def test_csv_appends_multiple_generations(tmp_path):
    csv_path = tmp_path / "output" / "metrics.csv"
    log = _make_csv_logger(csv_path)

    log(_metrics_dict(combined_score=0.5, clippy_warnings=5.0))
    log(_metrics_dict(combined_score=0.7, clippy_warnings=3.0))
    log(_metrics_dict(combined_score=0.9, clippy_warnings=1.0))

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    assert rows[0]["generation"] == "1"
    assert rows[1]["generation"] == "2"
    assert rows[2]["generation"] == "3"
    assert float(rows[0]["combined_score"]) == 0.5
    assert float(rows[2]["combined_score"]) == 0.9


def test_csv_records_failed_gates(tmp_path):
    csv_path = tmp_path / "output" / "metrics.csv"
    log = _make_csv_logger(csv_path)

    log(_metrics_dict(passed_gates=0.0, combined_score=0.0))

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    assert float(rows[0]["passed_gates"]) == 0.0
    assert float(rows[0]["combined_score"]) == 0.0


def test_csv_all_fields_present(tmp_path):
    csv_path = tmp_path / "output" / "metrics.csv"
    log = _make_csv_logger(csv_path)

    log(_metrics_dict())

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    expected = {
        "generation", "combined_score", "passed_gates", "static_score",
        "perf_score", "llm_score", "clippy_warnings", "compile_time",
        "binary_size", "tests_passed", "loc",
    }
    assert set(reader.fieldnames) == expected
    assert all(rows[0][k] != "" for k in expected)


def test_csv_header_not_duplicated_on_append(tmp_path):
    csv_path = tmp_path / "output" / "metrics.csv"
    log = _make_csv_logger(csv_path)

    log(_metrics_dict())
    log(_metrics_dict())

    raw = csv_path.read_text()
    # "generation" only appears in the header row, not duplicated
    assert raw.count("generation") == 1
    # Two data rows plus the header = 3 non-empty lines
    lines = [l for l in raw.strip().splitlines() if l]
    assert len(lines) == 3
