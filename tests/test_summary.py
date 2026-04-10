"""Tests for codeevolve.summary — Rust source file summarizer."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeevolve.summary import summarize_rs_file, summarize_files, content_hash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, name: str, content: str) -> Path:
    """Write a .rs file under tmp_path and return its absolute Path."""
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p.resolve()


def _summary(tmp_path: Path, content: str, name: str = "lib.rs") -> str:
    """Write content to a temp .rs file and return its summary (no project_path)."""
    f = _write(tmp_path, name, content)
    return summarize_rs_file(f)


def _lines(summary: str) -> list[str]:
    """Return non-header lines of a summary (skip the '// file:' line)."""
    return [ln for ln in summary.splitlines() if not ln.startswith("// file:")]


# ---------------------------------------------------------------------------
# File header
# ---------------------------------------------------------------------------

class TestFileHeader:
    def test_header_present(self, tmp_path: Path):
        f = _write(tmp_path, "src/lib.rs", "")
        summary = summarize_rs_file(f)
        assert summary.startswith("// file:")

    def test_header_uses_relative_path(self, tmp_path: Path):
        f = _write(tmp_path, "src/color.rs", "")
        summary = summarize_rs_file(f, project_path=tmp_path)
        assert summary.startswith("// file: src/color.rs")

    def test_header_nested_relative_path(self, tmp_path: Path):
        f = _write(tmp_path, "crates/engine/src/color.rs", "")
        summary = summarize_rs_file(f, project_path=tmp_path)
        assert summary.startswith("// file: crates/engine/src/color.rs")


# ---------------------------------------------------------------------------
# pub fn
# ---------------------------------------------------------------------------

class TestPubFn:
    def test_standalone_pub_fn(self, tmp_path: Path):
        src = "pub fn add(a: i32, b: i32) -> i32 { a + b }\n"
        lines = _lines(_summary(tmp_path, src))
        assert any("pub fn add" in ln for ln in lines)

    def test_pub_fn_no_body(self, tmp_path: Path):
        lines = _lines(_summary(tmp_path, "pub fn add(a: i32, b: i32) -> i32 { a + b }\n"))
        # Should not contain the body
        assert not any("{" in ln and "a + b" in ln for ln in lines)

    def test_pub_fn_in_impl_block(self, tmp_path: Path):
        src = (
            "impl Foo {\n"
            "    pub fn new() -> Self { Self {} }\n"
            "}\n"
        )
        lines = _lines(_summary(tmp_path, src))
        assert any("pub fn new" in ln for ln in lines)

    def test_pub_async_fn(self, tmp_path: Path):
        src = "pub async fn fetch(url: &str) -> Result<String, Error> { todo!() }\n"
        lines = _lines(_summary(tmp_path, src))
        assert any("pub async fn fetch" in ln for ln in lines)

    def test_multiline_fn_signature(self, tmp_path: Path):
        src = (
            "pub fn complex(\n"
            "    x: f32,\n"
            "    y: f32,\n"
            ") -> f32 {\n"
            "    x + y\n"
            "}\n"
        )
        lines = _lines(_summary(tmp_path, src))
        # Should have collected the signature across multiple lines
        assert any("pub fn complex" in ln for ln in lines)

    def test_private_fn_skipped(self, tmp_path: Path):
        src = "fn private_fn(x: i32) -> i32 { x }\n"
        lines = _lines(_summary(tmp_path, src))
        assert not any("private_fn" in ln for ln in lines)

    def test_pub_crate_fn_included(self, tmp_path: Path):
        src = "pub(crate) fn internal(x: i32) {}\n"
        lines = _lines(_summary(tmp_path, src))
        assert any("pub(crate)" in ln and "internal" in ln for ln in lines)


# ---------------------------------------------------------------------------
# pub struct
# ---------------------------------------------------------------------------

class TestPubStruct:
    def test_struct_single_line(self, tmp_path: Path):
        src = "pub struct Point { x: f32, y: f32 }\n"
        lines = _lines(_summary(tmp_path, src))
        assert any("pub struct Point" in ln for ln in lines)

    def test_tuple_struct(self, tmp_path: Path):
        src = "pub struct Meters(f64);\n"
        lines = _lines(_summary(tmp_path, src))
        assert any("pub struct Meters" in ln for ln in lines)

    def test_struct_with_generics(self, tmp_path: Path):
        src = "pub struct Vec2<T> { x: T, y: T }\n"
        lines = _lines(_summary(tmp_path, src))
        assert any("pub struct Vec2" in ln for ln in lines)

    def test_private_struct_skipped(self, tmp_path: Path):
        src = "struct Hidden { value: i32 }\n"
        lines = _lines(_summary(tmp_path, src))
        assert not any("Hidden" in ln for ln in lines)


# ---------------------------------------------------------------------------
# pub enum
# ---------------------------------------------------------------------------

class TestPubEnum:
    def test_enum_single_line(self, tmp_path: Path):
        src = "pub enum Color { Red, Green, Blue }\n"
        lines = _lines(_summary(tmp_path, src))
        assert any("pub enum Color" in ln for ln in lines)

    def test_enum_with_data(self, tmp_path: Path):
        src = "pub enum Shape { Circle(f32), Rect(f32, f32) }\n"
        lines = _lines(_summary(tmp_path, src))
        assert any("pub enum Shape" in ln for ln in lines)

    def test_private_enum_skipped(self, tmp_path: Path):
        src = "enum Internal { A, B }\n"
        lines = _lines(_summary(tmp_path, src))
        assert not any("Internal" in ln for ln in lines)


# ---------------------------------------------------------------------------
# pub trait
# ---------------------------------------------------------------------------

class TestPubTrait:
    def test_trait_header_extracted(self, tmp_path: Path):
        src = (
            "pub trait Animal {\n"
            "    fn name(&self) -> &str;\n"
            "    fn sound(&self) -> &str;\n"
            "}\n"
        )
        lines = _lines(_summary(tmp_path, src))
        assert any("pub trait Animal" in ln for ln in lines)

    def test_trait_with_supertrait(self, tmp_path: Path):
        src = (
            "pub trait Drawable: Sized {\n"
            "    fn draw(&self);\n"
            "}\n"
        )
        lines = _lines(_summary(tmp_path, src))
        assert any("pub trait Drawable" in ln for ln in lines)

    def test_private_trait_skipped(self, tmp_path: Path):
        src = (
            "trait Secret {\n"
            "    fn hidden();\n"
            "}\n"
        )
        lines = _lines(_summary(tmp_path, src))
        assert not any("Secret" in ln for ln in lines)


# ---------------------------------------------------------------------------
# impl headers
# ---------------------------------------------------------------------------

class TestImplHeaders:
    def test_plain_impl(self, tmp_path: Path):
        src = "impl Foo {\n    pub fn bar(&self) {}\n}\n"
        lines = _lines(_summary(tmp_path, src))
        assert any(ln.startswith("impl Foo") for ln in lines)

    def test_trait_impl(self, tmp_path: Path):
        src = "impl Display for Color {\n    fn fmt(&self, f: &mut Formatter) -> fmt::Result { todo!() }\n}\n"
        lines = _lines(_summary(tmp_path, src))
        assert any("impl Display for Color" in ln for ln in lines)

    def test_generic_impl(self, tmp_path: Path):
        src = "impl<T: Clone> MyTrait for Vec<T> {\n    fn do_it(&self) {}\n}\n"
        lines = _lines(_summary(tmp_path, src))
        assert any("impl" in ln and "MyTrait" in ln for ln in lines)


# ---------------------------------------------------------------------------
# pub type aliases and pub const
# ---------------------------------------------------------------------------

class TestTypeAndConst:
    def test_pub_type_alias(self, tmp_path: Path):
        src = "pub type Result<T> = std::result::Result<T, MyError>;\n"
        lines = _lines(_summary(tmp_path, src))
        assert any("pub type Result" in ln for ln in lines)

    def test_pub_const(self, tmp_path: Path):
        src = "pub const MAX_SIZE: usize = 1024;\n"
        lines = _lines(_summary(tmp_path, src))
        assert any("pub const MAX_SIZE" in ln for ln in lines)

    def test_pub_static(self, tmp_path: Path):
        src = 'pub static VERSION: &str = "1.0.0";\n'
        lines = _lines(_summary(tmp_path, src))
        assert any("pub static VERSION" in ln for ln in lines)

    def test_private_const_skipped(self, tmp_path: Path):
        src = "const INTERNAL: u32 = 42;\n"
        lines = _lines(_summary(tmp_path, src))
        assert not any("INTERNAL" in ln for ln in lines)


# ---------------------------------------------------------------------------
# cfg(test) skipping
# ---------------------------------------------------------------------------

class TestCfgTestSkip:
    def test_fn_in_cfg_test_skipped(self, tmp_path: Path):
        src = (
            "pub fn real_fn() {}\n"
            "\n"
            "#[cfg(test)]\n"
            "mod tests {\n"
            "    pub fn test_helper() {}\n"
            "    #[test]\n"
            "    fn it_works() {}\n"
            "}\n"
        )
        lines = _lines(_summary(tmp_path, src))
        assert any("real_fn" in ln for ln in lines)
        assert not any("test_helper" in ln for ln in lines)
        assert not any("it_works" in ln for ln in lines)

    def test_struct_in_cfg_test_skipped(self, tmp_path: Path):
        src = (
            "pub struct Real {}\n"
            "#[cfg(test)]\n"
            "mod tests {\n"
            "    pub struct MockHelper {}\n"
            "}\n"
        )
        lines = _lines(_summary(tmp_path, src))
        assert any("Real" in ln for ln in lines)
        assert not any("MockHelper" in ln for ln in lines)

    def test_items_after_cfg_test_block_included(self, tmp_path: Path):
        src = (
            "pub fn before() {}\n"
            "#[cfg(test)]\n"
            "mod tests {\n"
            "    fn hidden() {}\n"
            "}\n"
            "pub fn after() {}\n"
        )
        lines = _lines(_summary(tmp_path, src))
        assert any("before" in ln for ln in lines)
        assert any("after" in ln for ln in lines)
        assert not any("hidden" in ln for ln in lines)


# ---------------------------------------------------------------------------
# Private items skipped
# ---------------------------------------------------------------------------

class TestPrivateItemsSkipped:
    def test_private_fn(self, tmp_path: Path):
        src = "fn helper(x: i32) -> i32 { x * 2 }\n"
        lines = _lines(_summary(tmp_path, src))
        assert not any("helper" in ln for ln in lines)

    def test_mixed_visibility(self, tmp_path: Path):
        src = (
            "pub fn public_api() {}\n"
            "fn private_impl() {}\n"
            "pub struct PublicStruct {}\n"
            "struct PrivateStruct {}\n"
        )
        lines = _lines(_summary(tmp_path, src))
        assert any("public_api" in ln for ln in lines)
        assert not any("private_impl" in ln for ln in lines)
        assert any("PublicStruct" in ln for ln in lines)
        assert not any("PrivateStruct" in ln for ln in lines)


# ---------------------------------------------------------------------------
# summarize_files
# ---------------------------------------------------------------------------

class TestSummarizeFiles:
    def test_returns_dict_keyed_by_path(self, tmp_path: Path):
        f1 = _write(tmp_path, "src/a.rs", "pub fn a() {}\n")
        f2 = _write(tmp_path, "src/b.rs", "pub fn b() {}\n")
        result = summarize_files([f1, f2], project_path=tmp_path)
        assert set(result.keys()) == {f1, f2}

    def test_each_summary_has_header(self, tmp_path: Path):
        f1 = _write(tmp_path, "src/a.rs", "pub fn a() {}\n")
        result = summarize_files([f1], project_path=tmp_path)
        assert result[f1].startswith("// file:")

    def test_empty_list(self, tmp_path: Path):
        result = summarize_files([], project_path=tmp_path)
        assert result == {}

    def test_relative_paths_in_summaries(self, tmp_path: Path):
        f = _write(tmp_path, "crates/foo/src/lib.rs", "pub fn foo() {}\n")
        result = summarize_files([f], project_path=tmp_path)
        assert "crates/foo/src/lib.rs" in result[f]


# ---------------------------------------------------------------------------
# content_hash
# ---------------------------------------------------------------------------

class TestContentHash:
    def test_deterministic(self):
        text = "pub fn foo() {}"
        assert content_hash(text) == content_hash(text)

    def test_different_content_different_hash(self):
        assert content_hash("pub fn foo() {}") != content_hash("pub fn bar() {}")

    def test_empty_string(self):
        h = content_hash("")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest

    def test_returns_str(self):
        assert isinstance(content_hash("hello"), str)

    def test_hash_length(self):
        assert len(content_hash("some content")) == 64
