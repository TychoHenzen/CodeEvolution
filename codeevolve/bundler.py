"""Multi-file bundle format and focus rotation.

Bridges OpenEvolve's single-file model with our multi-file workspace.
OpenEvolve sees ONE string as the "program."  We pack a focus file
(editable) plus context summaries (read-only) into that string, and
unpack the LLM's edits back out.

Bundle format::

    // === CONTEXT (read-only — do NOT modify) ===
    // file: crates/engine_core/src/lib.rs
    pub struct Engine { ... }
    pub fn Engine::new() -> Self
    // file: crates/engine_render/src/renderer.rs
    pub struct Renderer { ... }
    impl Renderer { ... }
    // === END CONTEXT ===

    // === FOCUS: crates/engine_input/src/keyboard.rs ===
    // (This is the file you should improve. Output your improved version below.)
    <editable content here>
    // === END FOCUS ===
"""

from __future__ import annotations

import re
from pathlib import Path

from codeevolve.evaluator.pipeline import parse_evolve_block

# ---------------------------------------------------------------------------
# Bundle markers
# ---------------------------------------------------------------------------
_CONTEXT_START = "// === CONTEXT (read-only \u2014 do NOT modify) ==="
_CONTEXT_END = "// === END CONTEXT ==="
_FOCUS_PREFIX = "// === FOCUS: "
_FOCUS_SUFFIX = " ==="
_FOCUS_HINT = "// (This is the file you should improve. Output your improved version below.)"
_FOCUS_END = "// === END FOCUS ==="

# Regex for extracting focus content (between FOCUS header+hint and END FOCUS)
_FOCUS_RE = re.compile(
    r"^// === FOCUS: .+ ===$\n"
    r"^// \(This is the file you should improve\. Output your improved version below\.\)$\n"
    r"(.*?)"
    r"^// === END FOCUS ===$",
    re.MULTILINE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# Bundle creation
# ---------------------------------------------------------------------------

def create_bundle(
    focus_file: Path,
    all_files: list[Path],
    summaries: dict[Path, str],
    project_path: Path,
) -> str:
    """Create a bundled program string for OpenEvolve.

    The bundle contains:
    1. A read-only CONTEXT section with summaries of all OTHER files (not the focus)
    2. The FOCUS section with the full EVOLVE-BLOCK content from the focus file

    The LLM should only edit the FOCUS section.

    Args:
        focus_file: Absolute path to the file being evolved this iteration.
        all_files: All .rs files in the workspace (absolute paths).
        summaries: Pre-computed summaries keyed by absolute path (from
            ``summary.summarize_files``).  The focus file's summary is
            excluded automatically.
        project_path: Project root for computing relative paths.

    Returns:
        A single string that OpenEvolve treats as the "initial program."
    """
    # --- Build CONTEXT section ---
    context_lines: list[str] = [_CONTEXT_START]

    resolved_focus = focus_file.resolve()
    for f in all_files:
        if f.resolve() == resolved_focus:
            continue
        summary = summaries.get(f, "")
        if summary:
            context_lines.append(summary)

    context_lines.append(_CONTEXT_END)

    # --- Build FOCUS section ---
    focus_content = _read_focus_content(focus_file)
    try:
        rel_path = focus_file.resolve().relative_to(project_path.resolve())
        focus_path_str = rel_path.as_posix()
    except ValueError:
        focus_path_str = focus_file.as_posix()

    focus_lines: list[str] = [
        f"{_FOCUS_PREFIX}{focus_path_str}{_FOCUS_SUFFIX}",
        _FOCUS_HINT,
        focus_content,
        _FOCUS_END,
    ]

    return "\n".join(context_lines) + "\n\n" + "\n".join(focus_lines)


def _read_focus_content(focus_file: Path) -> str:
    """Read the editable content from a focus file.

    If the file has EVOLVE-BLOCK markers, returns only the content between
    the markers.  Otherwise, returns the full file content.
    """
    raw = focus_file.read_text(encoding="utf-8", errors="replace")
    parsed = parse_evolve_block(raw)
    if parsed is not None:
        _prefix, content, _suffix = parsed
        return content
    return raw


# ---------------------------------------------------------------------------
# Focus extraction
# ---------------------------------------------------------------------------

def extract_focus(bundle_text: str) -> str:
    """Extract just the editable focus file content from a bundle.

    Returns the text between the FOCUS markers (after the hint line and
    before ``// === END FOCUS ===``).  Leading/trailing whitespace within
    the focus section is preserved exactly as-is, except one leading and
    one trailing newline are stripped if present (an artefact of the
    join used in ``create_bundle``).

    Returns an empty string if the FOCUS markers are not found.
    """
    m = _FOCUS_RE.search(bundle_text)
    if m is None:
        return ""
    content = m.group(1)
    # Strip exactly one leading newline (artefact of joining) — but only
    # if the content is non-empty and starts with one.
    if content.startswith("\n"):
        content = content[1:]
    # Strip exactly one trailing newline before END FOCUS marker.
    if content.endswith("\n"):
        content = content[:-1]
    return content


def extract_focus_path(bundle_text: str) -> str | None:
    """Extract the relative path of the focus file from a bundle.

    Returns None if the FOCUS marker is not found.
    """
    m = re.search(r"^// === FOCUS: (.+?) ===$", bundle_text, re.MULTILINE)
    if m is None:
        return None
    return m.group(1)


# ---------------------------------------------------------------------------
# Focus rotation
# ---------------------------------------------------------------------------

class FocusRotation:
    """Round-robin iterator over files for focus selection.

    Deterministic: given the same file list, the order is always the same.
    Files are sorted by path to ensure reproducibility regardless of
    discovery order.
    """

    def __init__(self, files: list[Path]) -> None:
        if not files:
            raise ValueError("FocusRotation requires at least one file")
        # Sort for determinism
        self._files: list[Path] = sorted(files, key=lambda p: p.as_posix())
        self._index: int = 0

    def next(self) -> Path:
        """Return the next file to focus on and advance the pointer."""
        f = self._files[self._index]
        self._index = (self._index + 1) % len(self._files)
        return f

    def current(self) -> Path:
        """Return the current focus file without advancing."""
        return self._files[self._index]

    def __len__(self) -> int:
        return len(self._files)

    def __repr__(self) -> str:
        return (
            f"FocusRotation(files={len(self._files)}, "
            f"current={self._files[self._index].name!r})"
        )


def create_focus_rotation(files: list[Path]) -> FocusRotation:
    """Create a round-robin focus rotation over files.

    Args:
        files: List of file paths to rotate through.

    Returns:
        A ``FocusRotation`` that cycles through the files deterministically.
    """
    return FocusRotation(files)
