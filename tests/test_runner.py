from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeevolve.config import load_config
from codeevolve.runner import (
    build_openevolve_config_yaml,
    _normalize_llm_diffs,
    _run_multi_file,
    _run_single_file,
    find_latest_checkpoint,
)


def test_build_openevolve_config_yaml(tmp_path: Path):
    config = load_config()
    yaml_path = build_openevolve_config_yaml(config, tmp_path)
    assert yaml_path.exists()
    content = yaml_path.read_text()
    assert "api_base" in content


class TestNormalizeLlmDiffs:
    """Tests for _normalize_llm_diffs which rewrites markdown-style diffs."""

    def test_canonical_format_unchanged(self):
        """Already-correct format should pass through untouched."""
        text = (
            "<<<<<<< SEARCH\nfn foo() {}\n=======\n"
            "fn foo() { bar(); }\n>>>>>>> REPLACE"
        )
        assert _normalize_llm_diffs(text) == text

    def test_markdown_h4_with_rust_fences(self):
        """#### SEARCH / #### REPLACE with ```rust fences should be rewritten."""
        text = (
            "#### SEARCH\n"
            "```rust\n"
            "fn foo() {}\n"
            "```\n\n"
            "#### REPLACE\n"
            "```rust\n"
            "fn foo() { bar(); }\n"
            "```"
        )
        result = _normalize_llm_diffs(text)
        assert "<<<<<<< SEARCH" in result
        assert "=======" in result
        assert ">>>>>>> REPLACE" in result
        assert "fn foo() {}" in result
        assert "fn foo() { bar(); }" in result
        assert "```" not in result

    def test_markdown_h3_with_plain_fences(self):
        """### SEARCH / ### REPLACE with bare ``` fences."""
        text = (
            "### SEARCH\n"
            "```\n"
            "let x = 1;\n"
            "```\n\n"
            "### REPLACE\n"
            "```\n"
            "let x = 2;\n"
            "```"
        )
        result = _normalize_llm_diffs(text)
        assert "<<<<<<< SEARCH" in result
        assert "let x = 1;" in result
        assert "let x = 2;" in result

    def test_multiple_blocks(self):
        """Multiple markdown diff blocks should all be rewritten."""
        text = (
            "#### SEARCH\n```rust\nfn a() {}\n```\n\n"
            "#### REPLACE\n```rust\nfn a() { x(); }\n```\n\n"
            "Some explanation text\n\n"
            "#### SEARCH\n```rust\nfn b() {}\n```\n\n"
            "#### REPLACE\n```rust\nfn b() { y(); }\n```"
        )
        result = _normalize_llm_diffs(text)
        assert result.count("<<<<<<< SEARCH") == 2
        assert result.count(">>>>>>> REPLACE") == 2

    def test_no_diffs_returns_text_unchanged(self):
        """Plain text with no diff patterns should pass through."""
        text = "Just some commentary about the code."
        assert _normalize_llm_diffs(text) == text


class TestFindLatestCheckpoint:
    """Tests for find_latest_checkpoint()."""

    def test_no_checkpoints_dir_returns_none(self, tmp_path: Path):
        """output_dir with no checkpoints/ subdirectory returns None."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        assert find_latest_checkpoint(output_dir) is None

    def test_empty_checkpoints_dir_returns_none(self, tmp_path: Path):
        """An existing but empty checkpoints/ directory returns None."""
        checkpoints_dir = tmp_path / "output" / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        assert find_latest_checkpoint(tmp_path / "output") is None

    def test_single_valid_checkpoint_returned(self, tmp_path: Path):
        """A single checkpoint_N directory with metadata.json is returned."""
        cp = tmp_path / "output" / "checkpoints" / "checkpoint_5"
        cp.mkdir(parents=True)
        (cp / "metadata.json").write_text("{}", encoding="utf-8")
        result = find_latest_checkpoint(tmp_path / "output")
        assert result == str(cp)

    def test_multiple_checkpoints_returns_latest(self, tmp_path: Path):
        """When multiple valid checkpoints exist, the highest N is returned."""
        base = tmp_path / "output" / "checkpoints"
        for n in (1, 3, 7, 10):
            cp = base / f"checkpoint_{n}"
            cp.mkdir(parents=True)
            (cp / "metadata.json").write_text("{}", encoding="utf-8")
        result = find_latest_checkpoint(tmp_path / "output")
        assert result == str(base / "checkpoint_10")

    def test_checkpoint_without_metadata_skipped(self, tmp_path: Path):
        """A checkpoint directory lacking metadata.json is not returned."""
        cp = tmp_path / "output" / "checkpoints" / "checkpoint_3"
        cp.mkdir(parents=True)
        # No metadata.json written
        assert find_latest_checkpoint(tmp_path / "output") is None

    def test_mix_valid_and_invalid_returns_latest_valid(self, tmp_path: Path):
        """The latest valid checkpoint is returned even if a higher-N one is invalid."""
        base = tmp_path / "output" / "checkpoints"
        # checkpoint_10: invalid (no metadata.json)
        (base / "checkpoint_10").mkdir(parents=True)
        # checkpoint_7: valid
        cp7 = base / "checkpoint_7"
        cp7.mkdir(parents=True)
        (cp7 / "metadata.json").write_text("{}", encoding="utf-8")
        # checkpoint_2: valid (but lower N)
        cp2 = base / "checkpoint_2"
        cp2.mkdir(parents=True)
        (cp2 / "metadata.json").write_text("{}", encoding="utf-8")
        result = find_latest_checkpoint(tmp_path / "output")
        assert result == str(cp7)

    def test_non_checkpoint_dirs_ignored(self, tmp_path: Path):
        """Directories not matching checkpoint_N pattern are ignored."""
        base = tmp_path / "output" / "checkpoints"
        # These should be ignored
        for name in ("best", "logs", "checkpoint_abc", "checkpoint_"):
            d = base / name
            d.mkdir(parents=True)
            (d / "metadata.json").write_text("{}", encoding="utf-8")
        assert find_latest_checkpoint(tmp_path / "output") is None


def test_run_multi_file_uses_workspace_bundle(sample_workspace, tmp_path):
    """When a workspace is detected, _run_multi_file uses create_workspace_bundle."""
    config = load_config()
    config_path = sample_workspace / ".codeevolve" / "evolution.yaml"
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    source_files = sorted((sample_workspace / "crates").rglob("*.rs"))
    # Filter to only files with EVOLVE-BLOCK markers
    marked = [f for f in source_files if "EVOLVE-BLOCK-START" in f.read_text()]

    # Create a mock evaluator file so the cascade-evaluation check can read it
    config_path.parent.mkdir(parents=True, exist_ok=True)
    eval_file = config_path.parent / "evaluator.py"
    eval_file.write_text("def evaluate(path): return {}", encoding="utf-8")

    # Build a mock Program that controller.run() will return
    mock_program = MagicMock()
    mock_program.code = "pub fn improved() {}"
    mock_program.metrics = {"combined_score": 0.85}

    # Mock the OpenEvolve controller: its constructor returns an instance
    # whose async run() resolves to the mock program.
    import asyncio

    async def _fake_run(**kwargs):
        return mock_program

    mock_controller = MagicMock()
    mock_controller.run = _fake_run

    with patch("openevolve.controller.OpenEvolve", return_value=mock_controller) as mock_oe_cls, \
         patch("codeevolve.bundler.create_workspace_bundle") as mock_ws_bundle, \
         patch("codeevolve.summary.summarize_files") as mock_summarize:
        mock_ws_bundle.return_value = "// bundle content"
        mock_summarize.return_value = {}

        # This should detect workspace and use create_workspace_bundle
        _run_multi_file(
            config, sample_workspace, marked,
            eval_file, output_dir,
        )
        mock_ws_bundle.assert_called_once()
        mock_oe_cls.assert_called_once()


class TestFinalCheckpointSave:
    """Verify _save_checkpoint is called after controller.run() in both code paths."""

    def _make_mock_controller(self, mock_program):
        """Build a mock OpenEvolve controller with _save_checkpoint and database."""
        import asyncio

        async def _fake_run(**kwargs):
            return mock_program

        mock_db = MagicMock()
        mock_db.last_iteration = 42

        mock_controller = MagicMock()
        mock_controller.run = _fake_run
        mock_controller.database = mock_db
        return mock_controller

    def test_single_file_saves_final_checkpoint(self, sample_crate, tmp_path):
        """_run_single_file calls _save_checkpoint after controller.run returns."""
        config = load_config()
        codeevolve_dir = sample_crate / ".codeevolve"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        src_file = sample_crate / "src" / "lib.rs"

        codeevolve_dir.mkdir(parents=True, exist_ok=True)
        eval_file = codeevolve_dir / "evaluator.py"
        eval_file.write_text("def evaluate(path): return {}", encoding="utf-8")

        mock_program = MagicMock()
        mock_program.code = "pub fn hello() {}"
        mock_program.metrics = {"combined_score": 0.75}

        mock_controller = self._make_mock_controller(mock_program)

        with patch("openevolve.controller.OpenEvolve", return_value=mock_controller):
            _run_single_file(
                config, sample_crate, src_file,
                eval_file, output_dir,
            )

        mock_controller._save_checkpoint.assert_called_once_with(42)

    def test_multi_file_saves_final_checkpoint(self, sample_workspace, tmp_path):
        """_run_multi_file calls _save_checkpoint after controller.run returns."""
        config = load_config()
        codeevolve_dir = sample_workspace / ".codeevolve"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        source_files = sorted((sample_workspace / "crates").rglob("*.rs"))
        marked = [f for f in source_files if "EVOLVE-BLOCK-START" in f.read_text()]

        codeevolve_dir.mkdir(parents=True, exist_ok=True)
        eval_file = codeevolve_dir / "evaluator.py"
        eval_file.write_text("def evaluate(path): return {}", encoding="utf-8")

        mock_program = MagicMock()
        mock_program.code = "// bundle"
        mock_program.metrics = {"combined_score": 0.80}

        mock_controller = self._make_mock_controller(mock_program)

        with patch("openevolve.controller.OpenEvolve", return_value=mock_controller), \
             patch("codeevolve.bundler.create_workspace_bundle", return_value="// bundle"), \
             patch("codeevolve.summary.summarize_files", return_value={}):
            _run_multi_file(
                config, sample_workspace, marked,
                eval_file, output_dir,
            )

        mock_controller._save_checkpoint.assert_called_once_with(42)

    def test_final_checkpoint_not_called_when_attribute_missing(self, sample_workspace, tmp_path):
        """If controller lacks _save_checkpoint, no error is raised."""
        config = load_config()
        codeevolve_dir = sample_workspace / ".codeevolve"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        source_files = sorted((sample_workspace / "crates").rglob("*.rs"))
        marked = [f for f in source_files if "EVOLVE-BLOCK-START" in f.read_text()]

        codeevolve_dir.mkdir(parents=True, exist_ok=True)
        eval_file = codeevolve_dir / "evaluator.py"
        eval_file.write_text("def evaluate(path): return {}", encoding="utf-8")

        mock_program = MagicMock()
        mock_program.code = "// bundle"
        mock_program.metrics = {"combined_score": 0.80}

        async def _fake_run(**kwargs):
            return mock_program

        mock_controller = MagicMock(spec=[])
        mock_controller.run = _fake_run

        with patch("openevolve.controller.OpenEvolve", return_value=mock_controller), \
             patch("codeevolve.bundler.create_workspace_bundle", return_value="// bundle"), \
             patch("codeevolve.summary.summarize_files", return_value={}):
            _run_multi_file(
                config, sample_workspace, marked,
                eval_file, output_dir,
            )
