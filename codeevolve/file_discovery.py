from __future__ import annotations

import fnmatch
from pathlib import Path


def discover_rs_files(
    project_path: Path,
    include_globs: list[str],
    exclude_globs: list[str],
) -> list[Path]:
    """Find .rs files matching include globs, excluding those matching exclude globs.

    Args:
        project_path: Root directory to resolve globs against.
        include_globs: Glob patterns (relative to project_path) for files to include.
        exclude_globs: Glob patterns (relative to project_path) for files to exclude.

    Returns:
        Sorted list of unique absolute .rs file paths.
    """
    project_path = project_path.resolve()

    # Collect all candidates from include patterns
    candidates: set[Path] = set()
    for pattern in include_globs:
        for match in project_path.glob(pattern):
            if match.is_file() and match.suffix == ".rs":
                candidates.add(match.resolve())

    if not candidates or not exclude_globs:
        return sorted(candidates)

    # Filter out files matching any exclude pattern
    # Match against the path relative to project_path (as a POSIX string)
    result: list[Path] = []
    for candidate in candidates:
        rel = candidate.relative_to(project_path).as_posix()
        excluded = any(fnmatch.fnmatch(rel, pat) for pat in exclude_globs)
        if not excluded:
            result.append(candidate)

    return sorted(result)
