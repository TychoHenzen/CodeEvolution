"""Rust source file summarizer.

Fast, regex-based extraction of public API signatures for use as
read-only context during multi-file evolution. No cargo doc, no
rust-analyzer, no AST parsing.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches the start of a #[cfg(test)] attribute line (with optional whitespace).
_CFG_TEST_RE = re.compile(r"^\s*#\[cfg\(test\)\]")

# Matches any attribute line (so we can skip lines like #[derive(...)] etc.)
_ATTR_RE = re.compile(r"^\s*#\[")

# Opening / closing braces on their own or at end of various lines
_OPEN_BRACE_RE = re.compile(r"\{")
_CLOSE_BRACE_RE = re.compile(r"\}")

# pub type alias
_PUB_TYPE_RE = re.compile(r"^\s*(pub(?:\([^)]*\))?\s+type\s+\w+[^;]*);")

# pub const / static
_PUB_CONST_RE = re.compile(r"^\s*(pub(?:\([^)]*\))?\s+(?:const|static)\s+\w+[^;]*);")


# ---------------------------------------------------------------------------
# Core summarizer
# ---------------------------------------------------------------------------

def summarize_rs_file(file_path: Path, project_path: Path | None = None) -> str:
    """Extract pub API signatures from a Rust source file.

    Returns a compact text summary with:
    - pub fn signatures (full signature line, no body)
    - pub struct/enum definitions (name + fields/variants, no impls)
    - pub trait definitions (name + method signatures)
    - impl block headers (e.g. "impl Foo for Bar")
    - pub type aliases and pub const/static declarations

    Format::

        // file: crates/engine_core/src/color.rs
        pub struct Color { r: f32, g: f32, b: f32, a: f32 }
        pub fn Color::new(r: f32, g: f32, b: f32) -> Self
        impl Display for Color

    Args:
        file_path: Absolute path to the .rs source file.
        project_path: If given, the file header uses a path relative to this root.
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # File header
    if project_path is not None:
        try:
            rel = file_path.resolve().relative_to(project_path.resolve())
            header_path = rel.as_posix()
        except ValueError:
            header_path = file_path.as_posix()
    else:
        header_path = file_path.as_posix()

    items: list[str] = [f"// file: {header_path}"]

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------
    # brace_depth: current nesting level (0 = module top-level)
    brace_depth = 0

    # Stack of minimum brace depths that indicate "inside a cfg(test) block".
    # An entry N means: skip items when brace_depth >= N.
    # The entry is pushed when we enter the first { after a #[cfg(test)] attr.
    cfg_test_min_depths: list[int] = []

    # Set to True when we see #[cfg(test)] so the next opening brace starts a skip zone.
    pending_cfg_test = False

    def in_cfg_test() -> bool:
        return bool(cfg_test_min_depths) and brace_depth >= cfg_test_min_depths[-1]

    # Continuation buffer: some signatures span multiple lines.
    # e.g. `pub fn foo(\n    arg: T,\n) -> Ret {`
    continuation: list[str] = []
    continuation_mode: str = ""  # "fn" | "struct" | "enum" | "trait" | "impl"

    def flush_continuation() -> str | None:
        """Join continuation lines and extract the signature."""
        joined = " ".join(s.strip() for s in continuation)
        mode = continuation_mode
        continuation.clear()

        if mode == "fn":
            # Strip everything from the opening `{` onward
            sig = re.split(r"\s*\{", joined, maxsplit=1)[0]
            sig = sig.rstrip("; ").strip()
            if sig:
                return sig
        elif mode in ("struct", "enum", "trait"):
            sig = re.split(r"\s*\{", joined, maxsplit=1)[0]
            sig = sig.rstrip("; ").strip()
            if sig:
                return sig
        elif mode == "impl":
            sig = re.split(r"\s*\{", joined, maxsplit=1)[0].strip()
            if sig:
                return sig
        return None

    i = 0
    n = len(lines)

    while i < n:
        raw = lines[i]
        stripped = raw.strip()

        opens = len(_OPEN_BRACE_RE.findall(stripped))
        closes = len(_CLOSE_BRACE_RE.findall(stripped))

        # ---- Detect cfg(test) attribute ----
        if _CFG_TEST_RE.match(raw):
            pending_cfg_test = True
            i += 1
            continue

        # ---- Handle pending cfg(test): the next non-attr line that opens a brace ----
        if pending_cfg_test:
            # Skip attribute lines (e.g. #[allow(...)])
            if _ATTR_RE.match(raw):
                i += 1
                continue
            # This line should open the block (e.g. `mod tests {`)
            if opens > 0:
                # Push the depth INSIDE the block: brace_depth + 1 for the first `{`
                cfg_test_min_depths.append(brace_depth + 1)
                pending_cfg_test = False
            else:
                # No brace — unusual, cancel the pending flag
                pending_cfg_test = False
            # Fall through to normal processing so brace_depth is updated

        # ---- Pop finished cfg_test blocks ----
        new_depth = brace_depth + opens - closes
        while cfg_test_min_depths and new_depth < cfg_test_min_depths[-1]:
            cfg_test_min_depths.pop()

        # ---- Skip lines inside cfg(test) ----
        if in_cfg_test():
            brace_depth = new_depth
            i += 1
            continue

        # ---- Continuation mode ----
        if continuation:
            continuation.append(raw)
            brace_depth = new_depth

            # End of continuation: we have a `{` or `;` or something terminated it
            full_line = " ".join(s.strip() for s in continuation)
            if "{" in full_line or ";" in full_line:
                sig = flush_continuation()
                if sig:
                    items.append(sig)
            i += 1
            continue

        # ---- Normal line processing ----

        # pub fn
        if re.match(r"^\s*pub(?:\([^)]*\))?\s+(?:async\s+)?fn\b", raw):
            if "{" in stripped or ";" in stripped:
                # Single-line: extract inline
                m = re.match(
                    r"^\s*(pub(?:\([^)]*\))?\s+(?:async\s+)?fn\s+[^{;]+)",
                    raw,
                )
                if m:
                    sig = m.group(1).rstrip().rstrip(";").strip()
                    items.append(sig)
            else:
                # Multi-line signature
                continuation = [raw]
                continuation_mode = "fn"
            brace_depth = new_depth
            i += 1
            continue

        # pub struct / enum / trait
        if re.match(r"^\s*pub(?:\([^)]*\))?\s+(?:struct|enum|trait)\b", raw):
            if "{" in stripped or ";" in stripped:
                kind_m = re.match(r".*\b(struct|enum|trait)\b", stripped)
                kind = kind_m.group(1) if kind_m else ""
                if kind == "trait":
                    # Only emit the header: `pub trait Foo`
                    sig_m = re.match(
                        r"^\s*(pub(?:\([^)]*\))?\s+trait\s+\w+(?:<[^>]*>)?(?:\s*:\s*[^{]+)?)",
                        raw,
                    )
                    if sig_m:
                        items.append(sig_m.group(1).strip())
                else:
                    # Keep the whole single-line definition
                    sig = stripped.rstrip(";").rstrip()
                    if sig:
                        items.append(sig)
            else:
                continuation = [raw]
                continuation_mode = "struct"
            brace_depth = new_depth
            i += 1
            continue

        # pub type
        m = _PUB_TYPE_RE.match(raw)
        if m:
            items.append(m.group(1).strip() + ";")
            brace_depth = new_depth
            i += 1
            continue

        # pub const / static
        m = _PUB_CONST_RE.match(raw)
        if m:
            items.append(m.group(1).strip() + ";")
            brace_depth = new_depth
            i += 1
            continue

        # impl block header (only at top-level or inside a non-test block)
        if re.match(r"^\s*impl\b", raw) and "fn " not in stripped:
            if "{" in stripped:
                sig_m = re.match(r"^\s*(impl(?:<[^>]*>)?\s+.+?)\s*\{", raw)
                if sig_m:
                    items.append(sig_m.group(1).strip())
            else:
                continuation = [raw]
                continuation_mode = "impl"
            brace_depth = new_depth
            i += 1
            continue

        # Default: just update brace depth
        brace_depth = new_depth
        i += 1

    return "\n".join(items)


# ---------------------------------------------------------------------------
# Multi-file summarizer
# ---------------------------------------------------------------------------

def summarize_files(files: list[Path], project_path: Path) -> dict[Path, str]:
    """Summarize multiple Rust source files.

    Args:
        files: List of absolute paths to .rs files.
        project_path: Project root for computing relative paths in headers.

    Returns:
        Mapping from each file path to its summary text.
    """
    result: dict[Path, str] = {}
    for f in files:
        result[f] = summarize_rs_file(f, project_path)
    return result


# ---------------------------------------------------------------------------
# Content hash
# ---------------------------------------------------------------------------

def content_hash(content: str) -> str:
    """SHA-256 hash of content string, for cache keying.

    Args:
        content: Text content to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
