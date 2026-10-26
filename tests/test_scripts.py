"""Unit tests for all quality, parity, and snapshot scripts in `scripts/`."""

from __future__ import annotations

import pathlib
import pytest

import scripts.check_3rdparty_imports as c3p
import scripts.check_api_parity as cap
import scripts.check_api_snapshot as cas
import scripts.check_compiler_test_completeness as cctc
import scripts.check_docstrings as cdoc
import scripts.generate_api_matrix as gam
import scripts.generate_snapshot as gsnap
import scripts.lint_pass as lpass
import scripts.lint_typing as ltype
import scripts.update_badges as ubadges

NL: str = chr(10)


def _write_lines(path: pathlib.Path, lines: list[str]) -> None:
    """Write a list of lines joined by newline to `path`.

    Args:
        path: File path to write.
        lines: Lines to join and write.

    """
    path.write_text(NL.join(lines) + NL, encoding="utf-8")


def test_check_3rdparty_imports(tmp_path: pathlib.Path) -> None:
    """Test `scripts/check_3rdparty_imports.py` across allowed, stdlib, and forbidden imports.

    Args:
        tmp_path: Pytest temporary directory fixture.

    """
    assert c3p.is_stdlib("") is True
    assert c3p.is_stdlib("os") is True
    assert c3p.is_stdlib("sys") is True
    assert c3p.is_stdlib("nonexistent_xyz_package_999") is False
    assert c3p.is_stdlib("pytest") is False
    c3p.FALLBACK_STDLIB.discard("urllib")
    try:
        assert c3p.is_stdlib("urllib") is True
    finally:
        c3p.FALLBACK_STDLIB.add("urllib")

    good_file = tmp_path / "good.py"
    _write_lines(
        good_file,
        [
            '"""Good module."""',
            "import os",
            "import zero_jax",
            "from . import sibling",
            "from pydantic import BaseModel",
            "import importlib",
            'importlib.import_module("json")',
            'importlib.import_module(name="os")',
            "import sys",
            '_ = sys.modules["os"]',
            'sys.modules.get("os")',
            "x = len([1, 2])",
            "x.bit_length()",
        ],
    )
    assert c3p.check_file(str(good_file)) is True
    assert c3p.main([str(good_file)]) == 0

    good_test_file = tmp_path / "test_good.py"
    _write_lines(
        good_test_file,
        [
            '"""Good test module."""',
            "import pytest",
            "import scripts.lint_pass",
        ],
    )
    assert c3p.check_file(str(good_test_file)) is True

    bad_syntax = tmp_path / "bad_syntax.py"
    _write_lines(bad_syntax, ["def broken("])
    assert c3p.check_file(str(bad_syntax)) is False

    bad_file = tmp_path / "bad.py"
    _write_lines(
        bad_file,
        [
            '"""Bad module."""',
            "import requests",
            "from numpy import array",
            "from ml_switcheroo_compiler.backends.numpy.eager import np",
            "t.numpy()",
            "__import__(x)",
            "__import__()",
            "import importlib",
            'importlib.import_module(name="scipy")',
            "importlib.import_module(other=1)",
            "import sys",
            'sys.modules.get("torch")',
            "_ = sys.modules[k]",
            '_ = sys.modules["torch"]',
        ],
    )
    assert c3p.check_file(str(bad_file)) is False
    with pytest.raises(SystemExit):
        c3p.main([str(bad_file)])


def test_lint_typing(tmp_path: pathlib.Path) -> None:
    """Test `scripts/lint_typing.py` across clean and violating files.

    Args:
        tmp_path: Pytest temporary directory fixture.

    """
    assert ltype.scan_directory(str(tmp_path / "missing_dir")) == []

    clean_file = tmp_path / "clean.py"
    _write_lines(
        clean_file,
        [
            '"""Clean module."""',
            "from __future__ import annotations",
            "class A:",
            '    """Doc."""',
            "    def __eq__(self, other: object) -> bool:",
            '        """Doc."""',
            "        return True",
            "    def fn(self, x: int, *args: int, **kwargs: int) -> int:",
            '        """Doc."""',
            "        return sops.cast(x, int)",
        ],
    )
    assert ltype.scan_file(str(clean_file)) == []
    assert ltype.scan_directory(str(clean_file)) == []
    assert ltype.main([str(clean_file)]) == 0

    sub_dir = tmp_path / "pkg"
    sub_dir.mkdir()
    (sub_dir / "note.txt").write_text("not python", encoding="utf-8")

    bad_syntax = sub_dir / "syntax_err.py"
    _write_lines(bad_syntax, ["def f("])
    assert len(ltype.scan_file(str(bad_syntax))) == 1

    bad_file = sub_dir / "bad_typing.py"
    p_nc = "pragma: " + "no cover"
    r_nq = "ruff: " + "noqa"
    _write_lines(
        bad_file,
        [
            '"""Bad module."""',
            f"# {r_nq}",
            "from typing import Any, cast",
            "import typing",
            f"x: Any = 1  # {p_nc}",
            "y: typing.Any = 2",
            "Any = int",
            "def untyped(a, b: object):",
            '    getattr(a, "x")',
            "    typing.cast(int, a)",
            "    _ = a.__dict__",
            "def __getattr__(self: int, k: str) -> int:",
            "    return 1",
        ],
    )
    violations = ltype.scan_directory(str(tmp_path))
    assert len(violations) >= 6
    with pytest.raises(SystemExit):
        ltype.main([str(bad_file)])


def test_check_docstrings(tmp_path: pathlib.Path) -> None:
    """Test `scripts/check_docstrings.py` across documented and undocumented files.

    Args:
        tmp_path: Pytest temporary directory fixture.

    """
    assert cdoc.scan_path(str(tmp_path / "nonexistent")) == []

    good_file = tmp_path / "good_doc.py"
    _write_lines(
        good_file,
        [
            '"""Valid module docstring."""',
            "",
            "class MyClass:",
            '    """Valid class docstring."""',
            "    def method(self, x: int, *args: int, **kwargs: int) -> int:",
            '        """Valid method docstring.',
            "",
            "        Args:",
            "            x: First param.",
            "            *args: Varargs.",
            "            **kwargs: Kwargs.",
            "",
            "        Returns:",
            "            Result.",
            "",
            "        Notes:",
            "            Extra notes after Returns.",
            '        """',
            "        return x",
            "",
            "def test_helper(a: int) -> None:",
            '    """Test function documents its arguments.',
            "",
            "    Args:",
            "        a: Input integer.",
            '    """',
            "    assert a == 1",
        ],
    )
    assert cdoc.scan_path(str(good_file)) == []
    assert cdoc.main([str(good_file)]) == 0

    (tmp_path / "readme.txt").write_text("text", encoding="utf-8")

    bad_syntax = tmp_path / "bad_syntax.py"
    _write_lines(bad_syntax, ["def broken("])
    assert len(cdoc.check_file_docstrings(str(bad_syntax))) == 1

    placeholder_mod = tmp_path / "placeholder.py"
    _write_lines(
        placeholder_mod,
        [
            '"""Module docstring."""',
            "class C:",
            '    """Docstring."""',
            "def f():",
            '    """Call."""',
            "    return 1",
        ],
    )
    assert len(cdoc.check_file_docstrings(str(placeholder_mod))) == 3

    missing_doc = tmp_path / "missing.py"
    _write_lines(
        missing_doc,
        [
            "class UndocClass:",
            "    pass",
            "def undoc_fn(a: int, b: int) -> int:",
            "    return a",
            "def partial_doc(a: int, b: int) -> int:",
            '    """Has summary.',
            "",
            "    Args:",
            "        a: Documented.",
            "        not-an-identifier",
            "",
            "    Returns:",
            "        Int.",
            '    """',
            "    return a + b",
        ],
    )
    errs = cdoc.scan_path(str(tmp_path))
    assert len(errs) >= 4
    with pytest.raises(SystemExit):
        cdoc.main([str(missing_doc)])


def test_lint_pass(tmp_path: pathlib.Path) -> None:
    """Test `scripts/lint_pass.py` across valid and vacuous test files.

    Args:
        tmp_path: Pytest temporary directory fixture.

    """
    assert lpass.main(["--tests-dir", str(tmp_path / "missing")]) == 0

    good_test = tmp_path / "test_good.py"
    _write_lines(
        good_test,
        [
            '"""Good test module."""',
            "import pytest",
            "def test_one() -> None:",
            '    """Doc."""',
            "    assert 1 + 1 == 2",
            "def test_two() -> None:",
            '    """Doc."""',
            "    pytest.raises(ValueError)",
            "def test_three() -> None:",
            '    """Doc."""',
            "    decorator(assert_helper)(1)",
            "    obj.other_method()",
            "def test_four() -> None:",
            '    """Doc."""',
            "    try:",
            "        x = 1",
            "    except ValueError:",
            "        y = 2",
            "    obj[0](1)",
            "    obj.assert_equal(1, 1)",
        ],
    )
    assert lpass.check_test_file_ast(str(good_test)) == []
    assert lpass.main(["--tests-dir", str(tmp_path)]) == 0

    (tmp_path / "ignore.txt").write_text("txt", encoding="utf-8")

    bad_syntax = tmp_path / "test_syntax.py"
    _write_lines(bad_syntax, ["def test_x("])
    assert lpass.check_test_file_ast(str(bad_syntax)) == []

    bad_test = tmp_path / "test_bad.py"
    _write_lines(
        bad_test,
        [
            '"""Bad test module."""',
            "def test_empty() -> None:",
            "    x = 1",
            "def test_vacuous_true() -> None:",
            "    assert True",
            "def test_self_eq() -> None:",
            "    x = 1",
            "    assert x == x",
            "def test_indirect_self_eq() -> None:",
            "    a = compute(1)",
            "    b = compute(1)",
            "    assert a == b",
            "def test_excluded_middle() -> None:",
            "    x = 1",
            "    assert x is None or x is not None",
            "def test_and_not_or() -> None:",
            "    x = 1",
            "    assert x == 1 and x != 2",
            "def test_or_non_tautology() -> None:",
            "    x = 1",
            "    assert x == 1 or x == 2",
            "def test_or_true() -> None:",
            "    x = 1",
            "    assert x == 2 or True",
            "def test_swallowed() -> None:",
            "    try:",
            "        assert 1 == 2",
            "    except:",
            "        pass",
            "def test_catch_assertion() -> None:",
            "    try:",
            "        assert 1 == 1",
            "    except AssertionError:",
            "        x = 2",
            "    except ValueError:",
            "        pass",
            "def test_skipped() -> None:",
            '    pytest.skip("skipped")',
        ],
    )
    errs = lpass.check_test_file_ast(str(bad_test))
    assert len(errs) >= 6
    with pytest.raises(SystemExit):
        lpass.main(["--tests-dir", str(tmp_path)])


def test_generate_snapshot_and_parity(tmp_path: pathlib.Path) -> None:
    """Test `generate_snapshot.py`, `check_api_parity.py`, and `check_api_snapshot.py`.

    Args:
        tmp_path: Pytest temporary directory fixture.

    """
    refs = gsnap.collect_ghost_snapshot(include_nonpublic=False)
    assert isinstance(refs, dict)

    eco_file = tmp_path / "eco.json"
    _write_lines(
        eco_file,
        [
            '{"categories": {"array_api": [{"name": "add", "api_path": "mlx.core.add", "kind": "function", "params": [{"name": "a", "kind": "POSITIONAL_OR_KEYWORD"}], "aliases": ["mlx.core.missing_alias_xyz"], "docstring": ""}, {"name": "abs", "api_path": "mlx.core.abs", "kind": "function", "params": "bad_params_not_list", "docstring": "Doc"}, {"name": "bad_p", "api_path": "mlx.unknown.bad_p", "kind": "function", "params": [{"name": "a"}, "bad_param"], "docstring": ""}, {"name": "no_dot", "api_path": "nodot"}, {"name": "missing_sym", "api_path": "mlx.core.missing_sym", "kind": "function", "params": "not_a_list", "docstring": "Doc"}]}}'
        ],
    )
    out_snap = tmp_path / "snap.json"
    snap = gsnap.generate_snapshot_file(output_path=str(out_snap), cache_path=str(eco_file))
    assert "mlx.core" in snap
    assert "add" in snap["mlx.core"]
    assert gsnap.main(["--output", str(out_snap), "--cache-path", str(eco_file)]) == 0
    assert (
        gsnap.generate_snapshot_file(
            output_path=str(out_snap), cache_path=str(tmp_path / "none.json")
        )
        == {}
    )
    assert gsnap.load_cached_ecosystem_snapshot(cache_path=None) is not None

    assert cap.check_parity(snapshot_path=str(tmp_path / "nonexistent.json")) is False
    assert cap.main([]) == 0
    with pytest.raises(SystemExit):
        cap.main([str(tmp_path / "nonexistent.json")])

    custom_snap = tmp_path / "custom_snap.json"
    long_sig = "a" * 210
    long_doc = "b" * 210
    _write_lines(
        custom_snap,
        [
            f'{{"mlx.core": {{"missing_op": {{"signature": "{long_sig}", "doc_str": "{long_doc}"}}, "add": {{"signature": "(a, b)", "doc_str": "Add"}}}}, "mlx.core.sub": {{"missing_op": {{"signature": "(x)", "doc_str": "Sub"}}}}}}'
        ],
    )
    missing_md = tmp_path / "MISSING.md"
    assert (
        cap.check_parity(
            snapshot_path=str(custom_snap),
            src_dir="src/zero_mlx",
            missing_api_path=str(missing_md),
            ecosystem_path=str(eco_file),
        )
        is False
    )

    assert (
        cap.check_parity(
            snapshot_path=str(custom_snap),
            src_dir="src/zero_mlx",
            missing_api_path=str(missing_md),
            ecosystem_path=str(tmp_path / "none.json"),
        )
        is False
    )

    assert cas.get_public_api(None) == set()
    assert cas.verify_snapshot(snapshot_path=str(tmp_path / "nonexistent.json")) is False
    assert (
        cas.verify_snapshot(
            snapshot_path=str(custom_snap),
            ecosystem_path=str(eco_file),
        )
        is False
    )
    assert (
        cas.verify_snapshot(
            snapshot_path=str(custom_snap),
            ecosystem_path=str(tmp_path / "none.json"),
        )
        is False
    )
    assert cas.main([]) == 0
    with pytest.raises(SystemExit):
        cas.main([str(tmp_path / "nonexistent.json")])

    import importlib.util
    import zero_mlx as z_mlx_mod
    from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef, ParameterKind

    undoc_path = tmp_path / "zero_mlx_test_undoc.py"
    _write_lines(undoc_path, ["x: int = 1"])
    spec_u = importlib.util.spec_from_file_location("zero_mlx_test_undoc", str(undoc_path))
    assert spec_u is not None and spec_u.loader is not None
    undoc_mod = importlib.util.module_from_spec(spec_u)
    spec_u.loader.exec_module(undoc_mod)
    assert (
        cas.verify_module_hierarchy(
            (
                ("zero_mlx", z_mlx_mod),
                ("zero_mlx_test_undoc", undoc_mod),
                ("zero_mlx_nonexistent_mod_xyz", None),
            )
        )
        == 2
    )

    fake_mod_path = tmp_path / "fake_mlx_mod.py"
    _write_lines(
        fake_mod_path,
        [
            '"""Fake module for testing verify_ghost_ref_signature."""',
            "class DummyCls:",
            '    """Doc."""',
            "    existing_attr: int = 1",
            "    def __init__(self, typed_arg: int, untyped_arg) -> None:",
            '        """Doc."""',
            "        _ = (typed_arg, untyped_arg)",
            "def dummy_fn(typed_arg: int, **kwargs: int) -> int:",
            '    """Doc."""',
            "    _ = kwargs",
            "    return typed_arg",
            "def undoc_fn(untyped_arg) -> None:",
            "    _ = untyped_arg",
        ],
    )
    spec_f = importlib.util.spec_from_file_location("fake_mlx_mod", str(fake_mod_path))
    assert spec_f is not None and spec_f.loader is not None
    fake_mod = importlib.util.module_from_spec(spec_f)
    spec_f.loader.exec_module(fake_mod)

    assert (
        cas.verify_class_members(
            {"mlx.core": fake_mod},
            {
                ("mlx.missing", "X"): ("a",),
                ("mlx.core", "MissingCls"): ("a",),
                ("mlx.core", "DummyCls"): ("existing_attr", "missing_attr"),
            },
        )
        == 1
    )

    ref_missing = GhostRef(name="absent", api_path="mlx.core.absent", kind="function")
    assert cas.verify_ghost_ref_signature(ref_missing, fake_mod) == 1

    ref_kind1 = GhostRef(name="dummy_fn", api_path="mlx.core.dummy_fn", kind="class")
    assert cas.verify_ghost_ref_signature(ref_kind1, fake_mod) == 1

    ref_kind2 = GhostRef(
        name="DummyCls",
        api_path="mlx.core.DummyCls",
        kind="function",
        params=[
            GhostParam(name="args", kind=ParameterKind.VAR_POSITIONAL),
            GhostParam(name="typed_arg", kind=ParameterKind.POSITIONAL_OR_KEYWORD),
            GhostParam(name="untyped_arg", kind=ParameterKind.POSITIONAL_OR_KEYWORD),
            GhostParam(name="missing_param", kind=ParameterKind.POSITIONAL_OR_KEYWORD),
        ],
    )
    assert cas.verify_ghost_ref_signature(ref_kind2, fake_mod) == 3

    ref_undoc = GhostRef(
        name="undoc_fn",
        api_path="mlx.core.undoc_fn",
        kind="function",
        params=[GhostParam(name="missing_p", kind=ParameterKind.POSITIONAL_OR_KEYWORD)],
    )
    assert cas.verify_ghost_ref_signature(ref_undoc, fake_mod) == 2

    ref_varkw = GhostRef(
        name="dummy_fn",
        api_path="mlx.core.dummy_fn",
        kind="function",
        params=[GhostParam(name="extra_kw", kind=ParameterKind.POSITIONAL_OR_KEYWORD)],
    )
    assert cas.verify_ghost_ref_signature(ref_varkw, fake_mod) == 0


def test_compiler_completeness_and_matrix_and_badges(tmp_path: pathlib.Path) -> None:
    """Test `check_compiler_test_completeness.py`, `generate_api_matrix.py`, and `update_badges.py`.

    Args:
        tmp_path: Pytest temporary directory fixture.

    """
    assert cctc.main([]) == 0
    tmp_track = tmp_path / "TRACK.md"
    with pytest.raises(SystemExit):
        cctc.main([str(tmp_track)])
    assert (
        cctc.check_completeness(tests_dir=str(tmp_path / "no_tests"), tracking_file=str(tmp_track))
        is False
    )
    assert (
        cctc.check_completeness(tests_dir=str(tmp_path / "no_tests"), tracking_file=str(tmp_track))
        is True
    )

    rst_dir = tmp_path / "docs"
    rst_dir.mkdir()
    _write_lines(
        rst_dir / "ops.rst",
        [
            "Operations",
            "==========",
            ".. autosummary::",
            "   :toctree: _autosummary",
            "",
            "   add",
            "   subtract",
            "",
            "EndSection",
        ],
    )
    out_matrix = tmp_path / "MATRIX.md"
    mat = gam.generate_matrix(
        docs_dir=str(rst_dir),
        output_file=str(out_matrix),
        api_categories=("ops.rst", "missing.rst"),
    )
    assert "`add`" in mat
    snap_matrix = tmp_path / "SNAP_MATRIX.md"
    mat_snap = gam.generate_matrix(
        docs_dir=str(tmp_path / "nonexistent_docs"),
        output_file=str(snap_matrix),
        snapshot_path="mlx_api_snapshot.json",
    )
    assert "mlx.core" in mat_snap
    assert gam.main([]) == 0

    assert ubadges.get_color(100.0) == "brightgreen"
    assert ubadges.get_color(95.0) == "green"
    assert ubadges.get_color(85.0) == "yellowgreen"
    assert ubadges.get_color(75.0) == "yellow"
    assert ubadges.get_color(65.0) == "orange"
    assert ubadges.get_color(50.0) == "red"
    assert ubadges.format_cov(100.0) == "100"
    assert ubadges.format_cov(99.5) == "99.5"
    assert ubadges.get_doc_coverage() == 100.0

    cov_json = tmp_path / "cov.json"
    _write_lines(cov_json, ['{"totals": {"percent_covered": 100.0}}'])
    readme = tmp_path / "README.md"
    _write_lines(
        readme,
        [
            "# Title",
            "[![Test Coverage](https://img.shields.io/badge/test_coverage-50%25-red.svg)](#)",
            "[![Doc Coverage](https://img.shields.io/badge/doc_coverage-50%25-red.svg)](#)",
        ],
    )
    ubadges.update_readme(readme_path=str(readme), cov_json_path=str(cov_json))
    bad_cov_json = tmp_path / "bad_cov.json"
    _write_lines(bad_cov_json, ["not_json"])
    assert ubadges.get_test_coverage(cov_json_path=str(bad_cov_json)) == 0.0
    assert (
        ubadges.get_test_coverage(cov_json_path=str(tmp_path / "missing_dir" / "missing.json"))
        == 0.0
    )
    ubadges.update_readme(readme_path=str(tmp_path / "nonexistent_readme.md"))
    orig_readme = pathlib.Path("README.md").read_text(encoding="utf-8")
    try:
        assert ubadges.main([]) == 0
    finally:
        pathlib.Path("README.md").write_text(orig_readme, encoding="utf-8")
