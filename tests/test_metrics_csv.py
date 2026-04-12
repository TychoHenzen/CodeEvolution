"""Tests for per-generation CSV metrics logging in the evaluator template."""
import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

from codeevolve.evaluator.pipeline import EvaluationResult


def _make_csv_logger(csv_path: Path):
    """Build the same _log_metrics_csv / _generation state the rendered evaluator has."""
    fields = [
        "generation", "combined_score", "passed_gates",
        "perf_ratio", "llm_score", "clippy_warnings", "compile_time",
        "binary_size", "tests_passed", "tests_failed", "build_time", "loc",
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
        "perf_ratio": -100.0,
        "llm_score": 0.5,
        "clippy_warnings": 2.0,
        "compile_time": 2.0,
        "binary_size": 500_000.0,
        "tests_passed": 10.0,
        "tests_failed": 0.0,
        "build_time": 1.5,
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
        "generation", "combined_score", "passed_gates",
        "perf_ratio", "llm_score", "clippy_warnings", "compile_time",
        "binary_size", "tests_passed", "tests_failed", "build_time", "loc",
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


# ---------------------------------------------------------------------------
# Metrics normalization / splitting tests (evaluator.py.j2 boundary logic)
# ---------------------------------------------------------------------------


def _build_openevolve_metrics(result) -> dict:
    """Replicate the metrics dict that evaluator.py.j2 returns to OpenEvolve.

    This mirrors the template logic so we can test it without rendering Jinja2.
    """
    return {
        "combined_score": result.combined_score / 2.0,
        "perf_ratio": result.perf_ratio,
        "llm_score": result.llm_score / 2.0,
        "compile_time": result.compile_time,
        "binary_size": result.binary_size,
        "loc": result.loc,
    }


def _build_csv_metrics(result, openevolve_metrics) -> dict:
    """Replicate the csv_metrics dict from evaluator.py.j2."""
    return {
        **openevolve_metrics,
        "passed_gates": int(result.passed_gates),
        "clippy_warnings": result.clippy_warnings,
        "tests_passed": result.tests_passed,
        "tests_failed": result.tests_failed,
        "build_time": result.build_time,
    }


def test_openevolve_metrics_exclude_raw_counts():
    """Metrics returned to OpenEvolve must NOT contain raw counts."""
    result = EvaluationResult(
        passed_gates=True,
        combined_score=0.8,
        perf_score=0.5,
        perf_ratio=1.0,
        llm_score=0.6,
        build_time=3.5,
        tests_passed=42,
        tests_failed=2,
        clippy_warnings=7,
        binary_size=1.05,
        compile_time=0.95,
        loc=0.98,
    )
    metrics = _build_openevolve_metrics(result)
    # These raw-count keys must NOT be in the dict returned to OpenEvolve
    assert "tests_passed" not in metrics
    assert "tests_failed" not in metrics
    assert "clippy_warnings" not in metrics
    assert "passed_gates" not in metrics
    assert "build_time" not in metrics


def test_openevolve_metrics_halve_bounded_scores():
    """Scores naturally in [0,1] should be halved to centre around 0.5."""
    result = EvaluationResult(
        passed_gates=True,
        combined_score=0.8,
        perf_score=0.5,      # baseline norm_perf (internal, not sent to OpenEvolve)
        perf_ratio=1.0,      # baseline raw ratio (sent to OpenEvolve)
        llm_score=0.7,
        binary_size=1.0,
        compile_time=1.0,
        loc=1.0,
    )
    metrics = _build_openevolve_metrics(result)
    assert metrics["combined_score"] == 0.4    # 0.8 / 2.0
    assert metrics["llm_score"] == 0.35        # 0.7 / 2.0
    # perf_ratio uses raw ratio (baseline=1.0), consistent with compile_time/binary_size/loc
    assert metrics["perf_ratio"] == 1.0


def test_openevolve_metrics_pass_through_ratios():
    """Ratio-based metrics (perf_score, compile_time, binary_size, loc) pass through unchanged."""
    result = EvaluationResult(
        passed_gates=True,
        combined_score=0.5,
        perf_score=0.5,
        perf_ratio=1.15,     # 15% performance improvement over baseline
        llm_score=0.0,
        compile_time=1.2,    # 20% faster than baseline
        binary_size=0.85,    # 15% larger than baseline
        loc=1.1,             # 10% fewer LoC
    )
    metrics = _build_openevolve_metrics(result)
    assert metrics["perf_ratio"] == 1.15
    assert metrics["compile_time"] == 1.2
    assert metrics["binary_size"] == 0.85
    assert metrics["loc"] == 1.1


def test_csv_metrics_include_everything():
    """CSV metrics should contain all fields including raw counts."""
    result = EvaluationResult(
        passed_gates=True,
        combined_score=0.8,
        perf_score=0.5,
        perf_ratio=1.0,
        llm_score=0.6,
        build_time=2.5,
        tests_passed=10,
        tests_failed=1,
        clippy_warnings=3,
        binary_size=1.0,
        compile_time=1.0,
        loc=1.0,
    )
    oe_metrics = _build_openevolve_metrics(result)
    csv_metrics = _build_csv_metrics(result, oe_metrics)

    # All OpenEvolve metrics present
    for key in oe_metrics:
        assert key in csv_metrics

    # Additional diagnostic fields present
    assert csv_metrics["passed_gates"] == 1
    assert csv_metrics["clippy_warnings"] == 3
    assert csv_metrics["tests_passed"] == 10
    assert csv_metrics["tests_failed"] == 1
    assert csv_metrics["build_time"] == 2.5


def test_openevolve_metrics_all_values_comparable_scale():
    """All values returned to OpenEvolve should be in a comparable range,
    roughly centred around 0.5-1.0, never dominating like raw counts."""
    result = EvaluationResult(
        passed_gates=True,
        combined_score=0.8,
        perf_score=0.5,
        perf_ratio=1.0,
        llm_score=0.7,
        build_time=5.0,
        tests_passed=100,    # would dominate if included
        tests_failed=0,
        clippy_warnings=50,  # would dominate AND invert polarity if included
        binary_size=1.1,
        compile_time=0.9,
        loc=1.05,
    )
    metrics = _build_openevolve_metrics(result)
    # All values should be in a reasonable range (say 0 to 2)
    for key, val in metrics.items():
        assert 0.0 <= val <= 2.0, f"{key}={val} is out of comparable range"
