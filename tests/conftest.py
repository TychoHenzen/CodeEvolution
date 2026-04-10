import shutil
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_crate(tmp_path: Path) -> Path:
    """Copy the sample Rust crate to a temp directory so tests can modify it."""
    dest = tmp_path / "sample_crate"
    shutil.copytree(FIXTURES_DIR / "sample_crate", dest)
    return dest


@pytest.fixture
def sample_workspace(tmp_path: Path) -> Path:
    """Copy the sample workspace to a temp directory so tests can modify it."""
    dest = tmp_path / "sample_workspace"
    shutil.copytree(FIXTURES_DIR / "sample_workspace", dest)
    return dest


@pytest.fixture
def clippy_output_json() -> str:
    """Captured clippy JSON output for parsing tests."""
    return (FIXTURES_DIR / "clippy_output.json").read_text()
