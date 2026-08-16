"""Engine discovery must work from a WORKTREE, or the e2e suite cannot fail.

`test_e2e_smoke.py`, `test_selfplay_resume.py` and `test_sparse_multipv_labels.py`
all skip their whole module when `find_stockfish()` returns None. `main` got away
with a single ABSOLUTE candidate because that path resolves from anywhere on this
one machine; making it checkout-relative (correct, and required for a public repo)
resolved to nothing in every worktree and fresh clone, because
`e2e_server/publish/` is UNTRACKED runtime output -- `git ls-files e2e_server`
returns 0 files. MEASURED before the fix: 18 silent skips in a worktree; after it,
18 tests run.

So these tests are not about path strings. They are about the discovery still
having somewhere to look when the checkout has no engine of its own.
"""
from __future__ import annotations

import stat
from pathlib import Path

import pytest

from tests import stockfish_binary
from tests.stockfish_binary import ENV_VAR, find_stockfish, stockfish_candidates


def _fake_engine(tmp_path: Path, name: str = "stockfish") -> Path:
    p = tmp_path / name
    p.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkout with NO published engine and nothing on PATH.

    This is the state of every worktree and every fresh clone, and it is the
    state in which the pre-fix discovery returned None. Applied via
    `usefixtures` per test (not module-wide) because the last test in this
    file deliberately wants the REAL tree.
    """
    empty = tmp_path / "checkout"
    empty.mkdir()
    monkeypatch.setattr(stockfish_binary, "_REPO_ROOT", empty)
    monkeypatch.setattr(stockfish_binary, "_main_checkout", lambda: None)
    monkeypatch.setattr(stockfish_binary.shutil, "which", lambda _name: None)
    monkeypatch.delenv(ENV_VAR, raising=False)


@pytest.mark.usefixtures("isolated")
def test_a_checkout_with_no_engine_anywhere_still_reports_none() -> None:
    """The negative control: without it every assertion below is unfalsifiable."""
    assert find_stockfish() is None


@pytest.mark.usefixtures("isolated")
def test_the_env_override_is_honoured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`CAE_STOCKFISH` is the seam a fresh clone or CI runner uses."""
    engine = _fake_engine(tmp_path)
    monkeypatch.setenv(ENV_VAR, str(engine))
    assert find_stockfish() == str(engine)


@pytest.mark.usefixtures("isolated")
def test_the_env_override_is_read_at_call_time_not_import_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A module-level constant frozen at import cannot be set by a runner."""
    assert find_stockfish() is None
    engine = _fake_engine(tmp_path, "sf_late")
    monkeypatch.setenv(ENV_VAR, str(engine))
    assert find_stockfish() == str(engine)


@pytest.mark.usefixtures("isolated")
def test_the_path_install_is_a_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`shutil.which` is what actually saves a fresh clone on this machine."""
    engine = _fake_engine(tmp_path, "sf_on_path")
    monkeypatch.setattr(stockfish_binary.shutil, "which", lambda _n: str(engine))
    assert find_stockfish() == str(engine)


@pytest.mark.usefixtures("isolated")
def test_the_main_checkout_is_a_candidate_from_a_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The published engine is untracked, so only the main checkout has one."""
    main = tmp_path / "main_checkout"
    published = main / "e2e_server" / "publish"
    published.mkdir(parents=True)
    engine = _fake_engine(published)
    monkeypatch.setattr(stockfish_binary, "_main_checkout", lambda: main)
    assert find_stockfish() == str(engine)


@pytest.mark.usefixtures("isolated")
def test_the_override_outranks_the_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit answer must not be shadowed by a discovered one."""
    main = tmp_path / "main_checkout"
    published = main / "e2e_server" / "publish"
    published.mkdir(parents=True)
    _fake_engine(published)
    monkeypatch.setattr(stockfish_binary, "_main_checkout", lambda: main)
    chosen = _fake_engine(tmp_path, "sf_explicit")
    monkeypatch.setenv(ENV_VAR, str(chosen))
    assert find_stockfish() == str(chosen)


@pytest.mark.usefixtures("isolated")
def test_candidates_are_deduped_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stockfish_binary.shutil, "which", lambda _n: "/usr/bin/stockfish")
    cands = stockfish_candidates()
    assert len(cands) == len(set(cands)), cands
    assert cands.index("/usr/bin/stockfish") < cands.index("/usr/games/stockfish")


@pytest.mark.usefixtures("isolated")
def test_a_non_executable_file_is_not_an_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dud = tmp_path / "not_executable"
    dud.write_text("", encoding="utf-8")
    monkeypatch.setenv(ENV_VAR, str(dud))
    assert find_stockfish() is None


def test_discovery_actually_resolves_in_this_checkout() -> None:
    """The end-to-end claim, on the REAL tree this is running in.

    ⚑ Deliberately NOT skipif-guarded on the thing it is testing -- that is the
    shape of the defect. It is xfail-on-absence instead, so a machine with no
    engine at all reports "no engine here", not a green tick.
    """
    resolved = find_stockfish()
    if resolved is None:
        pytest.xfail(
            "no Stockfish discoverable from this checkout; the e2e suite will "
            f"skip. Set {ENV_VAR} or publish one. Candidates tried: "
            f"{stockfish_candidates()}"
        )
    assert Path(resolved).is_file()
