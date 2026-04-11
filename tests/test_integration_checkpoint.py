"""Integration tests for checkpoint/resume and rotation workflow.

These tests exercise the full wiring between ledger → scheduler → rotation →
checkpoint → resume.  Only the OpenEvolve controller (_run_single_file) is
mocked; the ledger parser, scheduler, and rotation loop run with real data.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeevolve.ledger import parse_ledger
from codeevolve.scheduler import build_schedule, build_roundrobin_schedule, ScheduleSlot
from codeevolve.runner import run_evolution_with_rotation, find_latest_checkpoint
from codeevolve.config import load_config, CodeEvolveConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rust_project(tmp_path: Path) -> Path:
    """Create a minimal fake Rust project with .codeevolve config."""
    # Create Cargo.toml
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "test"\nversion = "0.1.0"')

    # Create source files with EVOLVE-BLOCK markers
    src = tmp_path / "src"
    src.mkdir()
    for name in ["lib.rs", "utils.rs", "engine.rs"]:
        (src / name).write_text(
            f"// EVOLVE-BLOCK-START\nfn {name.replace('.rs', '')}() {{}}\n// EVOLVE-BLOCK-END\n"
        )

    # Create .codeevolve directory
    codeevolve = tmp_path / ".codeevolve"
    codeevolve.mkdir()

    # Create evaluator.py
    (codeevolve / "evaluator.py").write_text("def evaluate(path): return {'combined_score': 0.5}")

    # Create evolution.yaml
    (codeevolve / "evolution.yaml").write_text(
        "provider: local\n"
        "evolution:\n"
        "  max_iterations: 30\n"
        "  checkpoint_interval: 10\n"
        "  tech_debt_ledger: ''\n"
    )

    # Create TECH_DEBT_LEDGER.md
    ledger = tmp_path / "TECH_DEBT_LEDGER.md"
    ledger.write_text(
        "## Summary\n\n"
        "| File Path | Type | Structural | Semantic | Combined | Top Issue | Last Reviewed | Trend |\n"
        "|-----------|------|-----------|----------|----------|-----------|---------------|-------|\n"
        "| src/lib.rs | prod | 30.0 | 0 | 30.0 | magic_literals | 2026-04-02 | — |\n"
        "| src/utils.rs | prod | 20.0 | 0 | 20.0 | duplicate_blocks | 2026-04-02 | — |\n"
        "| src/engine.rs | prod | 10.0 | 0 | 10.0 | complexity | 2026-04-02 | — |\n"
    )

    return tmp_path


def _make_mock_result(score: float = 0.75, code: str = "pub fn improved() {}") -> MagicMock:
    """Build a mock EvolutionResult."""
    result = MagicMock()
    result.best_score = score
    result.best_code = code
    result.best_program = MagicMock()
    result.metrics = {"combined_score": score}
    result.output_dir = "/tmp/test"
    return result


# ---------------------------------------------------------------------------
# Test 1: End-to-end rotation with mock ledger
# ---------------------------------------------------------------------------

class TestEndToEndRotationWithLedger:
    """Full rotation flow: ledger → schedule → rotation loop → rotation_state.json."""

    def test_schedule_computed_from_ledger(self, rust_project: Path):
        """parse_ledger + build_schedule produce correct slot ordering."""
        ledger_path = rust_project / "TECH_DEBT_LEDGER.md"
        entries = parse_ledger(ledger_path, prod_only=True)

        # 3 prod entries expected, sorted by score descending
        assert len(entries) == 3
        assert entries[0].file_path == "src/lib.rs"
        assert entries[0].combined_score == 30.0
        assert entries[1].file_path == "src/utils.rs"
        assert entries[2].file_path == "src/engine.rs"

        # Build schedule with 30 iterations, chunk=10 → 3 chunks total
        schedule = build_schedule(entries, total_iterations=30, chunk_size=10)
        assert len(schedule) == 3
        # Highest-debt file gets the most iterations
        assert schedule[0].file_path == "src/lib.rs"

    def test_rotation_loop_calls_run_single_file_for_each_slot(self, rust_project: Path):
        """run_evolution_with_rotation calls _run_single_file once per slot."""
        ledger_path = rust_project / "TECH_DEBT_LEDGER.md"
        entries = parse_ledger(ledger_path, prod_only=True)
        schedule = build_schedule(entries, total_iterations=30, chunk_size=10)

        config_path = rust_project / ".codeevolve" / "evolution.yaml"
        evaluator_path = rust_project / ".codeevolve" / "evaluator.py"
        source_files = [rust_project / e.file_path for e in entries]

        mock_result = _make_mock_result()

        with patch("codeevolve.runner._run_single_file", return_value=mock_result) as mock_rsf:
            results = run_evolution_with_rotation(
                config_path, rust_project, schedule, source_files, evaluator_path,
            )

        assert mock_rsf.call_count == 3
        assert len(results) == 3
        assert "src/lib.rs" in results
        assert "src/utils.rs" in results
        assert "src/engine.rs" in results

    def test_rotation_state_json_written_after_all_slots(self, rust_project: Path):
        """rotation_state.json is written and reflects completed slot count."""
        ledger_path = rust_project / "TECH_DEBT_LEDGER.md"
        entries = parse_ledger(ledger_path, prod_only=True)
        schedule = build_schedule(entries, total_iterations=30, chunk_size=10)

        config_path = rust_project / ".codeevolve" / "evolution.yaml"
        evaluator_path = rust_project / ".codeevolve" / "evaluator.py"
        source_files = [rust_project / e.file_path for e in entries]

        mock_result = _make_mock_result()

        with patch("codeevolve.runner._run_single_file", return_value=mock_result):
            run_evolution_with_rotation(
                config_path, rust_project, schedule, source_files, evaluator_path,
            )

        state_path = rust_project / ".codeevolve" / "output" / "rotation_state.json"
        assert state_path.exists(), "rotation_state.json should be written"

        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["current_slot_index"] == 3
        assert len(state["schedule"]) == 3

    def test_slots_ordered_by_debt_score_descending(self, rust_project: Path):
        """Slots are ordered from highest to lowest combined_score."""
        ledger_path = rust_project / "TECH_DEBT_LEDGER.md"
        entries = parse_ledger(ledger_path, prod_only=True)
        schedule = build_schedule(entries, total_iterations=30, chunk_size=10)

        # Slot file_paths must follow debt order: lib > utils > engine
        slot_paths = [s.file_path for s in schedule]
        assert slot_paths[0] == "src/lib.rs"
        assert slot_paths[1] == "src/utils.rs"
        assert slot_paths[2] == "src/engine.rs"

    def test_rotation_calls_correct_source_file_per_slot(self, rust_project: Path):
        """Each slot passes the correct source file to _run_single_file."""
        ledger_path = rust_project / "TECH_DEBT_LEDGER.md"
        entries = parse_ledger(ledger_path, prod_only=True)
        schedule = build_schedule(entries, total_iterations=30, chunk_size=10)

        config_path = rust_project / ".codeevolve" / "evolution.yaml"
        evaluator_path = rust_project / ".codeevolve" / "evaluator.py"
        source_files = [rust_project / e.file_path for e in entries]

        mock_result = _make_mock_result()

        with patch("codeevolve.runner._run_single_file", return_value=mock_result) as mock_rsf:
            run_evolution_with_rotation(
                config_path, rust_project, schedule, source_files, evaluator_path,
            )

        # positional arg index 2 is initial_program (source_file)
        call_sources = [call[0][2] for call in mock_rsf.call_args_list]
        assert call_sources[0] == rust_project / "src/lib.rs"
        assert call_sources[1] == rust_project / "src/utils.rs"
        assert call_sources[2] == rust_project / "src/engine.rs"


# ---------------------------------------------------------------------------
# Test 2: Checkpoint/resume round-trip
# ---------------------------------------------------------------------------

class TestCheckpointResumeRoundTrip:
    """Interrupt after slot 1 then resume picks up from slot 1, not slot 0."""

    def test_interrupt_writes_slot_index_1(self, rust_project: Path):
        """When slot 2 raises, rotation_state.json records current_slot_index=1."""
        ledger_path = rust_project / "TECH_DEBT_LEDGER.md"
        entries = parse_ledger(ledger_path, prod_only=True)
        schedule = build_schedule(entries, total_iterations=30, chunk_size=10)

        config_path = rust_project / ".codeevolve" / "evolution.yaml"
        evaluator_path = rust_project / ".codeevolve" / "evaluator.py"
        source_files = [rust_project / e.file_path for e in entries]

        mock_ok = _make_mock_result()

        # Raise on the second call (slot index 1)
        with patch(
            "codeevolve.runner._run_single_file",
            side_effect=[mock_ok, RuntimeError("simulated interrupt")],
        ):
            with pytest.raises(RuntimeError, match="simulated interrupt"):
                run_evolution_with_rotation(
                    config_path, rust_project, schedule, source_files, evaluator_path,
                )

        state_path = rust_project / ".codeevolve" / "output" / "rotation_state.json"
        assert state_path.exists()
        state = json.loads(state_path.read_text(encoding="utf-8"))
        # Slot 0 completed → current_slot_index == 1 (next slot to run)
        assert state["current_slot_index"] == 1

    def test_resume_from_slot_1_skips_slot_0(self, rust_project: Path):
        """Resuming with checkpoint_path reads rotation_state.json and skips slot 0."""
        ledger_path = rust_project / "TECH_DEBT_LEDGER.md"
        entries = parse_ledger(ledger_path, prod_only=True)
        schedule = build_schedule(entries, total_iterations=30, chunk_size=10)

        config_path = rust_project / ".codeevolve" / "evolution.yaml"
        evaluator_path = rust_project / ".codeevolve" / "evaluator.py"
        source_files = [rust_project / e.file_path for e in entries]

        # Pre-populate rotation_state.json: slot 0 done, resume from slot 1
        output_dir = rust_project / ".codeevolve" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        rotation_state = {
            "current_slot_index": 1,
            "schedule": [
                {"file_path": s.file_path, "start_iter": s.start_iter, "end_iter": s.end_iter}
                for s in schedule
            ],
        }
        (output_dir / "rotation_state.json").write_text(
            json.dumps(rotation_state), encoding="utf-8",
        )

        mock_result = _make_mock_result()

        with patch("codeevolve.runner._run_single_file", return_value=mock_result) as mock_rsf:
            results = run_evolution_with_rotation(
                config_path, rust_project, schedule, source_files, evaluator_path,
                checkpoint_path="/fake/checkpoint",  # triggers resume logic
            )

        # Only slots 1 and 2 should run (slot 0 was skipped)
        assert mock_rsf.call_count == 2

        # The first executed slot should be src/utils.rs (slot index 1)
        first_executed_source = mock_rsf.call_args_list[0][0][2]
        assert first_executed_source == rust_project / "src/utils.rs"

    def test_resume_does_not_rerun_completed_slot(self, rust_project: Path):
        """Slot 0 file is not in results when resuming from slot 1."""
        ledger_path = rust_project / "TECH_DEBT_LEDGER.md"
        entries = parse_ledger(ledger_path, prod_only=True)
        schedule = build_schedule(entries, total_iterations=30, chunk_size=10)

        config_path = rust_project / ".codeevolve" / "evolution.yaml"
        evaluator_path = rust_project / ".codeevolve" / "evaluator.py"
        source_files = [rust_project / e.file_path for e in entries]

        output_dir = rust_project / ".codeevolve" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "rotation_state.json").write_text(
            json.dumps({"current_slot_index": 1, "schedule": []}),
            encoding="utf-8",
        )

        mock_result = _make_mock_result()

        with patch("codeevolve.runner._run_single_file", return_value=mock_result):
            results = run_evolution_with_rotation(
                config_path, rust_project, schedule, source_files, evaluator_path,
                checkpoint_path="/fake/checkpoint",
            )

        # src/lib.rs (slot 0) was skipped — not in results
        assert "src/lib.rs" not in results
        assert "src/utils.rs" in results
        assert "src/engine.rs" in results

    def test_no_checkpoint_path_ignores_rotation_state(self, rust_project: Path):
        """Without checkpoint_path, rotation_state.json is ignored and all slots run."""
        ledger_path = rust_project / "TECH_DEBT_LEDGER.md"
        entries = parse_ledger(ledger_path, prod_only=True)
        schedule = build_schedule(entries, total_iterations=30, chunk_size=10)

        config_path = rust_project / ".codeevolve" / "evolution.yaml"
        evaluator_path = rust_project / ".codeevolve" / "evaluator.py"
        source_files = [rust_project / e.file_path for e in entries]

        # Pre-populate rotation_state claiming all slots done
        output_dir = rust_project / ".codeevolve" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "rotation_state.json").write_text(
            json.dumps({"current_slot_index": 3, "schedule": []}),
            encoding="utf-8",
        )

        mock_result = _make_mock_result()

        with patch("codeevolve.runner._run_single_file", return_value=mock_result) as mock_rsf:
            results = run_evolution_with_rotation(
                config_path, rust_project, schedule, source_files, evaluator_path,
                checkpoint_path=None,  # no checkpoint → ignore rotation state
            )

        # All 3 slots should run
        assert mock_rsf.call_count == 3


# ---------------------------------------------------------------------------
# Test 3: --fresh flag clears checkpoints AND rotation state
# ---------------------------------------------------------------------------

class TestFreshFlagClearsState:
    """--fresh deletes both checkpoints/ dir and rotation_state.json."""

    def test_fresh_removes_checkpoints_dir(self, rust_project: Path):
        """--fresh deletes the checkpoints directory."""
        from click.testing import CliRunner
        from codeevolve.cli import main

        # Create the .codeevolve/output/checkpoints dir with a dummy checkpoint
        output_dir = rust_project / ".codeevolve" / "output"
        cp_dir = output_dir / "checkpoints" / "checkpoint_5"
        cp_dir.mkdir(parents=True)
        (cp_dir / "metadata.json").write_text("{}")

        config_path = rust_project / ".codeevolve" / "evolution.yaml"

        # Override config to use local provider so no proxy is started
        config_path.write_text(
            "provider: local\n"
            "evolution:\n"
            "  max_iterations: 10\n"
            "  checkpoint_interval: 10\n"
        )

        runner = CliRunner()
        mock_result = _make_mock_result()

        with patch("codeevolve.runner._run_single_file", return_value=mock_result), \
             patch("codeevolve.cli.LlamaServer") as mock_server_cls:
            mock_server = MagicMock()
            mock_server_cls.return_value = mock_server
            result = runner.invoke(
                main,
                ["run", "--config", str(config_path), "--fresh"],
            )

        # Checkpoints dir should be gone
        assert not (output_dir / "checkpoints").exists(), (
            f"checkpoints dir should be deleted. CLI output:\n{result.output}"
        )

    def test_fresh_removes_rotation_state_json(self, rust_project: Path):
        """--fresh deletes rotation_state.json before the run starts.

        We verify this by intercepting run_evolution_with_rotation and checking
        that rotation_state.json no longer exists at the point the run begins,
        even though the rotation loop will re-create it afterwards.
        """
        from click.testing import CliRunner
        from codeevolve.cli import main

        # Create rotation_state.json
        output_dir = rust_project / ".codeevolve" / "output"
        output_dir.mkdir(parents=True)
        rotation_state_path = output_dir / "rotation_state.json"
        rotation_state_path.write_text(json.dumps({"current_slot_index": 2, "schedule": []}))

        config_path = rust_project / ".codeevolve" / "evolution.yaml"
        config_path.write_text(
            "provider: local\n"
            "evolution:\n"
            "  max_iterations: 10\n"
            "  checkpoint_interval: 10\n"
        )

        # Track whether rotation_state.json exists when run_evolution_with_rotation
        # is first called (i.e. after --fresh cleaned it up).
        state_existed_at_run_time = []

        def _capture_rotation_state(*args, **kwargs):
            state_existed_at_run_time.append(rotation_state_path.exists())
            return {}

        runner = CliRunner()

        with patch("codeevolve.cli.run_evolution_with_rotation", side_effect=_capture_rotation_state), \
             patch("codeevolve.cli.LlamaServer") as mock_server_cls:
            mock_server_cls.return_value = MagicMock()
            result = runner.invoke(
                main,
                ["run", "--config", str(config_path), "--fresh"],
            )

        # The --fresh message should appear in output
        assert "Clearing existing checkpoints and rotation state" in result.output, (
            f"Expected --fresh message not found. CLI output:\n{result.output}"
        )
        # At the time run_evolution_with_rotation was called, rotation_state.json should
        # already have been deleted by --fresh
        assert state_existed_at_run_time, "run_evolution_with_rotation was never called"
        assert not state_existed_at_run_time[0], (
            "rotation_state.json still existed when rotation started — --fresh did not delete it"
        )

    def test_fresh_tolerates_missing_checkpoints_dir(self, rust_project: Path):
        """--fresh does not raise if checkpoints dir doesn't exist."""
        from click.testing import CliRunner
        from codeevolve.cli import main

        config_path = rust_project / ".codeevolve" / "evolution.yaml"
        config_path.write_text(
            "provider: local\n"
            "evolution:\n"
            "  max_iterations: 10\n"
            "  checkpoint_interval: 10\n"
        )

        runner = CliRunner()
        mock_result = _make_mock_result()

        with patch("codeevolve.runner._run_single_file", return_value=mock_result), \
             patch("codeevolve.cli.LlamaServer") as mock_server_cls:
            mock_server = MagicMock()
            mock_server_cls.return_value = mock_server
            result = runner.invoke(
                main,
                ["run", "--config", str(config_path), "--fresh"],
            )

        # Should not error even with no prior state
        assert result.exit_code == 0, f"CLI exited with code {result.exit_code}:\n{result.output}"


# ---------------------------------------------------------------------------
# Test 4: Fallback round-robin when no ledger configured
# ---------------------------------------------------------------------------

class TestRoundRobinFallback:
    """Multiple marked files + no tech_debt_ledger → round-robin schedule used."""

    def test_roundrobin_schedule_built_for_multiple_files(self, rust_project: Path):
        """build_roundrobin_schedule returns one slot per file."""
        file_paths = ["src/lib.rs", "src/utils.rs", "src/engine.rs"]
        schedule = build_roundrobin_schedule(file_paths, total_iterations=30, chunk_size=10)

        assert len(schedule) == 3
        slot_paths = {s.file_path for s in schedule}
        assert slot_paths == set(file_paths)

    def test_roundrobin_equal_iteration_distribution(self, rust_project: Path):
        """Round-robin gives roughly equal iterations to each file."""
        file_paths = ["src/lib.rs", "src/utils.rs", "src/engine.rs"]
        schedule = build_roundrobin_schedule(file_paths, total_iterations=30, chunk_size=10)

        # 30 iterations / 3 files = 10 each
        for slot in schedule:
            assert (slot.end_iter - slot.start_iter) == 10

    def test_roundrobin_rotation_loop_calls_all_files(self, rust_project: Path):
        """run_evolution_with_rotation is called for all files in round-robin mode."""
        config_path = rust_project / ".codeevolve" / "evolution.yaml"
        config_path.write_text(
            "provider: local\n"
            "evolution:\n"
            "  max_iterations: 30\n"
            "  checkpoint_interval: 10\n"
            "  tech_debt_ledger: ''\n"  # no ledger
        )
        evaluator_path = rust_project / ".codeevolve" / "evaluator.py"

        source_files = [
            rust_project / "src/lib.rs",
            rust_project / "src/utils.rs",
            rust_project / "src/engine.rs",
        ]

        file_paths = [str(f.relative_to(rust_project)) for f in source_files]
        schedule = build_roundrobin_schedule(file_paths, total_iterations=30, chunk_size=10)

        mock_result = _make_mock_result()

        with patch("codeevolve.runner._run_single_file", return_value=mock_result) as mock_rsf:
            results = run_evolution_with_rotation(
                config_path, rust_project, schedule, source_files, evaluator_path,
            )

        assert mock_rsf.call_count == 3
        assert len(results) == 3

    def test_roundrobin_schedule_contiguous_ranges(self, rust_project: Path):
        """Round-robin slots have contiguous, non-overlapping iteration ranges."""
        file_paths = ["src/lib.rs", "src/utils.rs", "src/engine.rs"]
        schedule = build_roundrobin_schedule(file_paths, total_iterations=30, chunk_size=10)

        # Verify contiguous ranges
        for i in range(1, len(schedule)):
            assert schedule[i].start_iter == schedule[i - 1].end_iter


# ---------------------------------------------------------------------------
# Test 5: Single file falls back to original (non-rotation) path
# ---------------------------------------------------------------------------

class TestSingleFileFallback:
    """Single marked file → run_evolution() called, not rotation."""

    def _make_single_file_project(self, tmp_path: Path) -> Path:
        """Create a project with exactly one marked file."""
        project = tmp_path / "single_file_project"
        project.mkdir()
        (project / "Cargo.toml").write_text('[package]\nname = "test"\nversion = "0.1.0"')

        src = project / "src"
        src.mkdir()
        (src / "lib.rs").write_text(
            "// EVOLVE-BLOCK-START\nfn lib() {}\n// EVOLVE-BLOCK-END\n"
        )

        codeevolve = project / ".codeevolve"
        codeevolve.mkdir()
        (codeevolve / "evaluator.py").write_text("def evaluate(path): return {'combined_score': 0.5}")
        (codeevolve / "evolution.yaml").write_text(
            "provider: local\n"
            "evolution:\n"
            "  max_iterations: 10\n"
            "  checkpoint_interval: 10\n"
            "  tech_debt_ledger: ''\n"
        )

        return project

    def test_single_marked_file_uses_run_evolution(self, tmp_path: Path):
        """When only 1 file is marked, run_evolution() is called via the CLI."""
        from click.testing import CliRunner
        from codeevolve.cli import main

        project = self._make_single_file_project(tmp_path)
        config_path = project / ".codeevolve" / "evolution.yaml"

        runner = CliRunner()
        mock_result = _make_mock_result()

        with patch("codeevolve.cli.run_evolution", return_value=mock_result) as mock_re, \
             patch("codeevolve.cli.run_evolution_with_rotation") as mock_rew, \
             patch("codeevolve.cli.LlamaServer") as mock_server_cls:
            mock_server_cls.return_value = MagicMock()
            result = runner.invoke(
                main,
                ["run", "--config", str(config_path)],
            )

        # run_evolution should be called, not rotation
        mock_re.assert_called_once()
        mock_rew.assert_not_called()

    def test_single_file_rotation_not_used(self, tmp_path: Path):
        """Rotation is skipped when there is only 1 marked file."""
        from click.testing import CliRunner
        from codeevolve.cli import main

        project = self._make_single_file_project(tmp_path)
        config_path = project / ".codeevolve" / "evolution.yaml"

        runner = CliRunner()
        mock_result = _make_mock_result()

        with patch("codeevolve.cli.run_evolution", return_value=mock_result), \
             patch("codeevolve.cli.run_evolution_with_rotation") as mock_rew, \
             patch("codeevolve.cli.LlamaServer") as mock_server_cls:
            mock_server_cls.return_value = MagicMock()
            result = runner.invoke(
                main,
                ["run", "--config", str(config_path)],
            )

        mock_rew.assert_not_called()

    def test_single_file_run_evolution_gets_correct_args(self, tmp_path: Path):
        """run_evolution receives the correct config_path and project_path."""
        from click.testing import CliRunner
        from codeevolve.cli import main

        project = self._make_single_file_project(tmp_path)
        config_path = project / ".codeevolve" / "evolution.yaml"

        runner = CliRunner()
        mock_result = _make_mock_result()

        with patch("codeevolve.cli.run_evolution", return_value=mock_result) as mock_re, \
             patch("codeevolve.cli.LlamaServer") as mock_server_cls:
            mock_server_cls.return_value = MagicMock()
            result = runner.invoke(
                main,
                ["run", "--config", str(config_path)],
            )

        # Verify run_evolution got the resolved config_path
        call_config_path = mock_re.call_args[0][0]
        assert call_config_path == config_path.resolve()
