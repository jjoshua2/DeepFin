from __future__ import annotations

import pytest

from scripts import check_free_threading_compat as check


def test_initially_enabled_gil_fails_before_dependency_imports(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    imported: list[str] = []
    monkeypatch.setattr("sys.argv", ["check_free_threading_compat"])
    monkeypatch.setattr(check.sys, "_is_gil_enabled", lambda: True, raising=False)
    monkeypatch.setattr(check.importlib, "import_module", imported.append)

    with pytest.raises(SystemExit, match="GIL is enabled at startup"):
        check.main()

    assert imported == []
    assert "PASS:" not in capsys.readouterr().out


def test_interpreter_without_gil_status_fails_before_dependency_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    monkeypatch.setattr("sys.argv", ["check_free_threading_compat"])
    monkeypatch.delattr(check.sys, "_is_gil_enabled", raising=False)
    monkeypatch.setattr(check.importlib, "import_module", imported.append)

    with pytest.raises(SystemExit, match="cannot report free-threading status"):
        check.main()
    assert imported == []


@pytest.mark.parametrize("enabling_module", ["zarr", "chess_anti_engine.nnue._nnue_ext"])
def test_import_that_enables_gil_is_named_and_cannot_pass(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
    enabling_module: str,
) -> None:
    imported: list[str] = []
    gil_enabled = False

    def import_module(name: str) -> None:
        nonlocal gil_enabled
        imported.append(name)
        if name == enabling_module:
            gil_enabled = True

    monkeypatch.setattr("sys.argv", ["check_free_threading_compat"])
    monkeypatch.setattr(check.sys, "_is_gil_enabled", lambda: gil_enabled, raising=False)
    monkeypatch.setattr(check.importlib, "import_module", import_module)

    with pytest.raises(SystemExit) as exc:
        check.main()
    assert exc.value.code == 1
    assert enabling_module in imported
    output = capsys.readouterr().out
    assert f"{enabling_module}: import enabled the GIL" in output
    assert "PASS:" not in output
    if enabling_module == "zarr":
        assert "already enabled; module not assessed" in output


@pytest.mark.parametrize("include_training", [False, True])
def test_success_requires_nnue_and_optionally_training_imports(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
    include_training: bool,
) -> None:
    imported: list[str] = []
    monkeypatch.setattr("sys.argv", ["check_free_threading_compat", *(["--include-training"] if include_training else [])])
    monkeypatch.setattr(check.sys, "_is_gil_enabled", lambda: False, raising=False)
    monkeypatch.setattr(check.importlib, "import_module", imported.append)

    check.main()

    assert "chess_anti_engine.nnue._nnue_ext" in imported
    assert ("ray" in imported) == include_training
    assert "PASS: all requested imports left the GIL disabled" in capsys.readouterr().out


def test_failed_import_cannot_pass_even_if_gil_stays_disabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    def import_module(name: str) -> None:
        if name == "chess_anti_engine.nnue._nnue_ext":
            raise ImportError("native extension unavailable")

    monkeypatch.setattr("sys.argv", ["check_free_threading_compat"])
    monkeypatch.setattr(check.sys, "_is_gil_enabled", lambda: False, raising=False)
    monkeypatch.setattr(check.importlib, "import_module", import_module)
    with pytest.raises(SystemExit) as exc:
        check.main()
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "native extension unavailable" in output
    assert "PASS:" not in output
