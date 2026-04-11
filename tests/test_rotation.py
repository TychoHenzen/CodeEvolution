"""Tests for rotation loop — run_evolution_with_rotation()."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from codeevolve.config import load_config
from codeevolve.scheduler import ScheduleSlot
from codeevolve.runner import run_evolution_with_rotation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_result(score: float = 0.75, code: str = "pub fn improved() {}"):
    """Build a mock EvolutionResult."""
    result = MagicMock()
    result.best_score = score
    result.best_code = code
    result.best_program = MagicMock()
    result.metrics = {"combined_score": score}
    result.output_dir = "/tmp/test"
    return result


def _setup_source_files(project_path: Path, file_paths: list[str]) -> list[Path]:
    """Create source files with EVOLVE-BLOCK markers at the given relative paths."""
    source_files = []
    for rel_path in file_paths:
        full_path = project_path / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(
            "// EVOLVE-BLOCK-START\npub fn hello() {}\n// EVOLVE-BLOCK-END\n",
            encoding="utf-8",
        )
        source_files.append(full_path)
    return source_files


def _make_config_path(project_path: Path) -> Path:
    """Create a minimal config file for testing."""
    config_dir = project_path / ".codeevolve"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "evolution.yaml"
    config_path.write_text(
        "provider: local\nllama_server:\n  port: 8080\n",
        encoding="utf-8",
    )
    return config_path


def _make_evaluator(project_path: Path) -> Path:
    """Create a minimal evaluator.py for testing."""
    eval_path = project_path / ".codeevolve" / "evaluator.py"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text("def evaluate(path): return {}", encoding="utf-8")
    return eval_path


# ---------------------------------------------------------------------------
# Tests: _run_single_file is called for each slot
# ---------------------------------------------------------------------------

class TestRotationCallsSingleFile:
    """run_evolution_with_rotation() calls _run_single_file for each slot."""

    def test_calls_single_file_for_each_slot(self, tmp_path: Path):
        project_path = tmp_path / "project"
        project_path.mkdir()
        config_path = _make_config_path(project_path)
        evaluator_path = _make_evaluator(project_path)

        source_files = _setup_source_files(project_path, [
            "src/render.rs",
            "src/shape.rs",
        ])

        schedule = [
            ScheduleSlot(file_path="src/render.rs", start_iter=0, end_iter=50),
            ScheduleSlot(file_path="src/shape.rs", start_iter=50, end_iter=100),
        ]

        mock_result = _make_mock_result()

        with patch("codeevolve.runner._run_single_file", return_value=mock_result) as mock_rsf:
            results = run_evolution_with_rotation(
                config_path, project_path, schedule, source_files,
                evaluator_path,
            )

        assert mock_rsf.call_count == 2

    def test_passes_correct_iterations_per_slot(self, tmp_path: Path):
        project_path = tmp_path / "project"
        project_path.mkdir()
        config_path = _make_config_path(project_path)
        evaluator_path = _make_evaluator(project_path)

        source_files = _setup_source_files(project_path, [
            "src/render.rs",
            "src/shape.rs",
        ])

        schedule = [
            ScheduleSlot(file_path="src/render.rs", start_iter=0, end_iter=30),
            ScheduleSlot(file_path="src/shape.rs", start_iter=30, end_iter=100),
        ]

        mock_result = _make_mock_result()

        with patch("codeevolve.runner._run_single_file", return_value=mock_result) as mock_rsf:
            run_evolution_with_rotation(
                config_path, project_path, schedule, source_files,
                evaluator_path,
            )

        # Check that config passed to first call had max_iterations=30
        first_call_config = mock_rsf.call_args_list[0][0][0]
        assert first_call_config.evolution.max_iterations == 30

        # Check that config passed to second call had max_iterations=70
        second_call_config = mock_rsf.call_args_list[1][0][0]
        assert second_call_config.evolution.max_iterations == 70

    def test_passes_correct_source_file_per_slot(self, tmp_path: Path):
        project_path = tmp_path / "project"
        project_path.mkdir()
        config_path = _make_config_path(project_path)
        evaluator_path = _make_evaluator(project_path)

        source_files = _setup_source_files(project_path, [
            "src/render.rs",
            "src/shape.rs",
        ])

        schedule = [
            ScheduleSlot(file_path="src/render.rs", start_iter=0, end_iter=50),
            ScheduleSlot(file_path="src/shape.rs", start_iter=50, end_iter=100),
        ]

        mock_result = _make_mock_result()

        with patch("codeevolve.runner._run_single_file", return_value=mock_result) as mock_rsf:
            run_evolution_with_rotation(
                config_path, project_path, schedule, source_files,
                evaluator_path,
            )

        # Check the initial_program (4th positional arg) matches the expected file
        first_call_source = mock_rsf.call_args_list[0][0][3]
        assert first_call_source == source_files[0]

        second_call_source = mock_rsf.call_args_list[1][0][3]
        assert second_call_source == source_files[1]

    def test_each_slot_gets_own_output_dir(self, tmp_path: Path):
        project_path = tmp_path / "project"
        project_path.mkdir()
        config_path = _make_config_path(project_path)
        evaluator_path = _make_evaluator(project_path)

        source_files = _setup_source_files(project_path, [
            "src/render.rs",
            "src/shape.rs",
        ])

        schedule = [
            ScheduleSlot(file_path="src/render.rs", start_iter=0, end_iter=50),
            ScheduleSlot(file_path="src/shape.rs", start_iter=50, end_iter=100),
        ]

        mock_result = _make_mock_result()

        with patch("codeevolve.runner._run_single_file", return_value=mock_result) as mock_rsf:
            run_evolution_with_rotation(
                config_path, project_path, schedule, source_files,
                evaluator_path,
            )

        # output_dir is the 6th positional arg
        first_output_dir = mock_rsf.call_args_list[0][0][5]
        second_output_dir = mock_rsf.call_args_list[1][0][5]

        assert first_output_dir != second_output_dir
        assert "slot_0" in str(first_output_dir)
        assert "slot_1" in str(second_output_dir)


# ---------------------------------------------------------------------------
# Tests: Rotation state file saved after each slot
# ---------------------------------------------------------------------------

class TestRotationStateSaved:
    """rotation_state.json is saved after each slot completes."""

    def test_state_file_created_after_each_slot(self, tmp_path: Path):
        project_path = tmp_path / "project"
        project_path.mkdir()
        config_path = _make_config_path(project_path)
        evaluator_path = _make_evaluator(project_path)

        source_files = _setup_source_files(project_path, ["src/render.rs"])

        schedule = [
            ScheduleSlot(file_path="src/render.rs", start_iter=0, end_iter=50),
        ]

        mock_result = _make_mock_result()

        with patch("codeevolve.runner._run_single_file", return_value=mock_result):
            run_evolution_with_rotation(
                config_path, project_path, schedule, source_files,
                evaluator_path,
            )

        state_path = project_path / ".codeevolve" / "output" / "rotation_state.json"
        assert state_path.exists()

        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["current_slot_index"] == 1
        assert len(state["schedule"]) == 1

    def test_state_file_tracks_multiple_slots(self, tmp_path: Path):
        project_path = tmp_path / "project"
        project_path.mkdir()
        config_path = _make_config_path(project_path)
        evaluator_path = _make_evaluator(project_path)

        source_files = _setup_source_files(project_path, [
            "src/render.rs",
            "src/shape.rs",
            "src/path.rs",
        ])

        schedule = [
            ScheduleSlot(file_path="src/render.rs", start_iter=0, end_iter=50),
            ScheduleSlot(file_path="src/shape.rs", start_iter=50, end_iter=80),
            ScheduleSlot(file_path="src/path.rs", start_iter=80, end_iter=100),
        ]

        mock_result = _make_mock_result()

        with patch("codeevolve.runner._run_single_file", return_value=mock_result):
            run_evolution_with_rotation(
                config_path, project_path, schedule, source_files,
                evaluator_path,
            )

        state_path = project_path / ".codeevolve" / "output" / "rotation_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["current_slot_index"] == 3
        assert len(state["schedule"]) == 3

    def test_state_file_contains_schedule_details(self, tmp_path: Path):
        project_path = tmp_path / "project"
        project_path.mkdir()
        config_path = _make_config_path(project_path)
        evaluator_path = _make_evaluator(project_path)

        source_files = _setup_source_files(project_path, ["src/render.rs"])

        schedule = [
            ScheduleSlot(file_path="src/render.rs", start_iter=0, end_iter=50),
        ]

        mock_result = _make_mock_result()

        with patch("codeevolve.runner._run_single_file", return_value=mock_result):
            run_evolution_with_rotation(
                config_path, project_path, schedule, source_files,
                evaluator_path,
            )

        state_path = project_path / ".codeevolve" / "output" / "rotation_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))

        slot_data = state["schedule"][0]
        assert slot_data["file_path"] == "src/render.rs"
        assert slot_data["start_iter"] == 0
        assert slot_data["end_iter"] == 50


# ---------------------------------------------------------------------------
# Tests: Resume from rotation state skips completed slots
# ---------------------------------------------------------------------------

class TestRotationResume:
    """Resume from rotation_state.json skips completed slots."""

    def test_resume_skips_completed_slots(self, tmp_path: Path):
        project_path = tmp_path / "project"
        project_path.mkdir()
        config_path = _make_config_path(project_path)
        evaluator_path = _make_evaluator(project_path)

        source_files = _setup_source_files(project_path, [
            "src/render.rs",
            "src/shape.rs",
            "src/path.rs",
        ])

        schedule = [
            ScheduleSlot(file_path="src/render.rs", start_iter=0, end_iter=50),
            ScheduleSlot(file_path="src/shape.rs", start_iter=50, end_iter=80),
            ScheduleSlot(file_path="src/path.rs", start_iter=80, end_iter=100),
        ]

        # Pre-populate rotation state indicating first 2 slots are done
        output_dir = project_path / ".codeevolve" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        rotation_state = {
            "current_slot_index": 2,
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
                config_path, project_path, schedule, source_files,
                evaluator_path,
                checkpoint_path="/fake/checkpoint",  # triggers resume logic
            )

        # Only the 3rd slot (index 2) should have been executed
        assert mock_rsf.call_count == 1
        called_source = mock_rsf.call_args_list[0][0][3]
        assert called_source == source_files[2]  # src/path.rs

    def test_resume_without_checkpoint_path_starts_from_zero(self, tmp_path: Path):
        """Even if rotation_state.json exists, without checkpoint_path we start from 0."""
        project_path = tmp_path / "project"
        project_path.mkdir()
        config_path = _make_config_path(project_path)
        evaluator_path = _make_evaluator(project_path)

        source_files = _setup_source_files(project_path, [
            "src/render.rs",
            "src/shape.rs",
        ])

        schedule = [
            ScheduleSlot(file_path="src/render.rs", start_iter=0, end_iter=50),
            ScheduleSlot(file_path="src/shape.rs", start_iter=50, end_iter=100),
        ]

        # Pre-populate rotation state
        output_dir = project_path / ".codeevolve" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "rotation_state.json").write_text(
            json.dumps({"current_slot_index": 1, "schedule": []}),
            encoding="utf-8",
        )

        mock_result = _make_mock_result()

        with patch("codeevolve.runner._run_single_file", return_value=mock_result) as mock_rsf:
            run_evolution_with_rotation(
                config_path, project_path, schedule, source_files,
                evaluator_path,
                checkpoint_path=None,  # no checkpoint -> start from 0
            )

        # Both slots should execute since checkpoint_path is None
        assert mock_rsf.call_count == 2

    def test_resume_all_slots_done_runs_nothing(self, tmp_path: Path):
        """If all slots are already done, no _run_single_file calls happen."""
        project_path = tmp_path / "project"
        project_path.mkdir()
        config_path = _make_config_path(project_path)
        evaluator_path = _make_evaluator(project_path)

        source_files = _setup_source_files(project_path, ["src/render.rs"])

        schedule = [
            ScheduleSlot(file_path="src/render.rs", start_iter=0, end_iter=50),
        ]

        # Mark all slots as done
        output_dir = project_path / ".codeevolve" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "rotation_state.json").write_text(
            json.dumps({"current_slot_index": 1, "schedule": []}),
            encoding="utf-8",
        )

        mock_result = _make_mock_result()

        with patch("codeevolve.runner._run_single_file", return_value=mock_result) as mock_rsf:
            results = run_evolution_with_rotation(
                config_path, project_path, schedule, source_files,
                evaluator_path,
                checkpoint_path="/fake/checkpoint",
            )

        assert mock_rsf.call_count == 0
        assert results == {}


# ---------------------------------------------------------------------------
# Tests: Results dict has entries for each evolved file
# ---------------------------------------------------------------------------

class TestRotationResults:
    """Results dict contains entries for each evolved file."""

    def test_results_dict_has_entries_for_each_file(self, tmp_path: Path):
        project_path = tmp_path / "project"
        project_path.mkdir()
        config_path = _make_config_path(project_path)
        evaluator_path = _make_evaluator(project_path)

        source_files = _setup_source_files(project_path, [
            "src/render.rs",
            "src/shape.rs",
        ])

        schedule = [
            ScheduleSlot(file_path="src/render.rs", start_iter=0, end_iter=50),
            ScheduleSlot(file_path="src/shape.rs", start_iter=50, end_iter=100),
        ]

        mock_result_render = _make_mock_result(score=0.80, code="// render improved")
        mock_result_shape = _make_mock_result(score=0.65, code="// shape improved")

        with patch("codeevolve.runner._run_single_file", side_effect=[mock_result_render, mock_result_shape]):
            results = run_evolution_with_rotation(
                config_path, project_path, schedule, source_files,
                evaluator_path,
            )

        assert "src/render.rs" in results
        assert "src/shape.rs" in results
        assert results["src/render.rs"].best_score == 0.80
        assert results["src/shape.rs"].best_score == 0.65

    def test_skipped_file_not_in_results(self, tmp_path: Path):
        """Files not in all_source_files are skipped and not in results."""
        project_path = tmp_path / "project"
        project_path.mkdir()
        config_path = _make_config_path(project_path)
        evaluator_path = _make_evaluator(project_path)

        # Only create one source file but schedule references a second
        source_files = _setup_source_files(project_path, ["src/render.rs"])

        schedule = [
            ScheduleSlot(file_path="src/render.rs", start_iter=0, end_iter=50),
            ScheduleSlot(file_path="src/missing.rs", start_iter=50, end_iter=100),
        ]

        mock_result = _make_mock_result()

        with patch("codeevolve.runner._run_single_file", return_value=mock_result) as mock_rsf:
            results = run_evolution_with_rotation(
                config_path, project_path, schedule, source_files,
                evaluator_path,
            )

        # Only render.rs should be in results; missing.rs was skipped
        assert "src/render.rs" in results
        assert "src/missing.rs" not in results
        assert mock_rsf.call_count == 1


# ---------------------------------------------------------------------------
# Tests: Best files saved to output/best/
# ---------------------------------------------------------------------------

class TestRotationBestFiles:
    """Best files are saved to output/best/."""

    def test_best_code_saved_to_best_dir(self, tmp_path: Path):
        project_path = tmp_path / "project"
        project_path.mkdir()
        config_path = _make_config_path(project_path)
        evaluator_path = _make_evaluator(project_path)

        source_files = _setup_source_files(project_path, ["src/render.rs"])

        schedule = [
            ScheduleSlot(file_path="src/render.rs", start_iter=0, end_iter=50),
        ]

        mock_result = _make_mock_result(code="pub fn best_render() {}")

        with patch("codeevolve.runner._run_single_file", return_value=mock_result):
            run_evolution_with_rotation(
                config_path, project_path, schedule, source_files,
                evaluator_path,
            )

        best_file = project_path / ".codeevolve" / "output" / "best" / "render.rs"
        assert best_file.exists()
        assert "best_render" in best_file.read_text(encoding="utf-8")

    def test_best_dir_has_files_for_each_evolved_file(self, tmp_path: Path):
        project_path = tmp_path / "project"
        project_path.mkdir()
        config_path = _make_config_path(project_path)
        evaluator_path = _make_evaluator(project_path)

        source_files = _setup_source_files(project_path, [
            "src/render.rs",
            "src/shape.rs",
        ])

        schedule = [
            ScheduleSlot(file_path="src/render.rs", start_iter=0, end_iter=50),
            ScheduleSlot(file_path="src/shape.rs", start_iter=50, end_iter=100),
        ]

        mock_result_render = _make_mock_result(code="// render best")
        mock_result_shape = _make_mock_result(code="// shape best")

        with patch("codeevolve.runner._run_single_file", side_effect=[mock_result_render, mock_result_shape]):
            run_evolution_with_rotation(
                config_path, project_path, schedule, source_files,
                evaluator_path,
            )

        best_dir = project_path / ".codeevolve" / "output" / "best"
        assert (best_dir / "render.rs").exists()
        assert (best_dir / "shape.rs").exists()

    def test_empty_best_code_not_saved(self, tmp_path: Path):
        """If best_code is empty, the file should not be saved to best/."""
        project_path = tmp_path / "project"
        project_path.mkdir()
        config_path = _make_config_path(project_path)
        evaluator_path = _make_evaluator(project_path)

        source_files = _setup_source_files(project_path, ["src/render.rs"])

        schedule = [
            ScheduleSlot(file_path="src/render.rs", start_iter=0, end_iter=50),
        ]

        mock_result = _make_mock_result(code="")

        with patch("codeevolve.runner._run_single_file", return_value=mock_result):
            run_evolution_with_rotation(
                config_path, project_path, schedule, source_files,
                evaluator_path,
            )

        best_file = project_path / ".codeevolve" / "output" / "best" / "render.rs"
        assert not best_file.exists()


# ---------------------------------------------------------------------------
# Tests: Source file backup and restore
# ---------------------------------------------------------------------------

class TestRotationBackupRestore:
    """Source files are backed up at the start and restored at the end."""

    def test_source_files_restored_after_rotation(self, tmp_path: Path):
        project_path = tmp_path / "project"
        project_path.mkdir()
        config_path = _make_config_path(project_path)
        evaluator_path = _make_evaluator(project_path)

        source_files = _setup_source_files(project_path, [
            "src/render.rs",
            "src/shape.rs",
        ])

        # Record original content
        original_contents = {f: f.read_text(encoding="utf-8") for f in source_files}

        schedule = [
            ScheduleSlot(file_path="src/render.rs", start_iter=0, end_iter=50),
        ]

        mock_result = _make_mock_result()

        with patch("codeevolve.runner._run_single_file", return_value=mock_result):
            run_evolution_with_rotation(
                config_path, project_path, schedule, source_files,
                evaluator_path,
            )

        # Verify all files were restored to original content
        for f in source_files:
            assert f.read_text(encoding="utf-8") == original_contents[f]

    def test_source_files_restored_on_error(self, tmp_path: Path):
        """Even if _run_single_file raises, source files should be restored."""
        project_path = tmp_path / "project"
        project_path.mkdir()
        config_path = _make_config_path(project_path)
        evaluator_path = _make_evaluator(project_path)

        source_files = _setup_source_files(project_path, ["src/render.rs"])
        original_content = source_files[0].read_text(encoding="utf-8")

        schedule = [
            ScheduleSlot(file_path="src/render.rs", start_iter=0, end_iter=50),
        ]

        with patch("codeevolve.runner._run_single_file", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                run_evolution_with_rotation(
                    config_path, project_path, schedule, source_files,
                    evaluator_path,
                )

        assert source_files[0].read_text(encoding="utf-8") == original_content
