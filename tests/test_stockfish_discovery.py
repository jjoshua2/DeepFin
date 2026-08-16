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

from chess_anti_engine.utils import engine_discovery
from tests.stockfish_binary import ENV_VAR, find_stockfish, stockfish_candidates

REPO_ROOT = Path(__file__).resolve().parents[1]


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
    monkeypatch.setattr(engine_discovery, "REPO_ROOT", empty)
    monkeypatch.setattr(engine_discovery, "main_checkout", lambda _root=None: None)
    monkeypatch.setattr(engine_discovery.shutil, "which", lambda _name: None)
    # ⚑ AND the distro literals. Mocking `shutil.which` does NOT remove them —
    # `stockfish_candidates()` appends them unconditionally — so on a host with
    # `/usr/bin/stockfish` installed this fixture was not isolated at all and the
    # negative control, the late-env test and the non-executable test would every
    # one of them discover the real system engine. That is a green suite here and
    # a red one on a distro-installed box, i.e. exactly backwards.
    monkeypatch.setattr(engine_discovery, "DISTRO_CANDIDATES", ())
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
    monkeypatch.setattr(engine_discovery.shutil, "which", lambda _n: str(engine))
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
    monkeypatch.setattr(engine_discovery, "main_checkout", lambda _root=None: main)
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
    monkeypatch.setattr(engine_discovery, "main_checkout", lambda _root=None: main)
    chosen = _fake_engine(tmp_path, "sf_explicit")
    monkeypatch.setenv(ENV_VAR, str(chosen))
    assert find_stockfish() == str(chosen)


@pytest.mark.usefixtures("isolated")
def test_candidates_are_deduped_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    # Restores the real literals: this test is ABOUT their ordering, so it is the
    # one case that must not have them stripped by `isolated`.
    monkeypatch.setattr(engine_discovery, "DISTRO_CANDIDATES",
                        ("/usr/bin/stockfish", "/usr/games/stockfish"))
    monkeypatch.setattr(engine_discovery.shutil, "which", lambda _n: "/usr/bin/stockfish")
    cands = stockfish_candidates()
    assert len(cands) == len(set(cands)), cands
    assert cands.index("/usr/bin/stockfish") < cands.index("/usr/games/stockfish")


@pytest.mark.usefixtures("isolated")
def test_a_distro_installed_engine_does_not_defeat_the_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ The fixture's own control, and the Codex finding pinned.

    On a host with a distro Stockfish the pre-review `isolated` fixture left the
    literals in place, so `find_stockfish()` returned the SYSTEM engine and the
    negative control above passed only because this particular box has none
    installed. Simulated here by putting a real executable at a literal path, so
    the assertion holds on both kinds of host.
    """
    distro = _fake_engine(tmp_path, "usr_bin_stockfish")
    monkeypatch.setattr(engine_discovery, "DISTRO_CANDIDATES", (str(distro),))
    assert find_stockfish() == str(distro), (
        "fixture broken: the literal candidates are not being consulted at all"
    )
    monkeypatch.setattr(engine_discovery, "DISTRO_CANDIDATES", ())
    assert find_stockfish() is None, (
        "the `isolated` fixture cannot remove the distro literals, so every "
        "negative assertion in this file is unfalsifiable on a host that has one"
    )


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


@pytest.mark.usefixtures("isolated")
def test_the_script_default_falls_back_to_the_main_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ Codex finding: `scripts/blindspot_*` had the worktree bug too.

    Their `--stockfish` default was `Path(__file__).parents[1]/e2e_server/...`,
    which is the SAME checkout-relative candidate that returned None in a
    worktree — and unlike the test helper they got no multi-root discovery, so
    running any of them from the mandated worktree pointed at a nonexistent
    executable. `main`'s absolute default reached the published engine; the
    scrub took that away without replacing it.
    """
    main = tmp_path / "main_checkout"
    published = main / "e2e_server" / "publish"
    published.mkdir(parents=True)
    engine = _fake_engine(published)
    monkeypatch.setattr(engine_discovery, "main_checkout", lambda _root=None: main)
    assert engine_discovery.default_stockfish() == str(engine)


@pytest.mark.usefixtures("isolated")
def test_the_script_default_is_never_none() -> None:
    """A `--stockfish` default of None turns "not found" into a TypeError deep
    inside an engine constructor. With nothing discoverable it must still name
    the checkout path, so the failure reads "no such file: <path>".
    """
    assert find_stockfish() is None            # nothing to discover, by fixture
    fallback = engine_discovery.default_stockfish()
    assert fallback.endswith(str(engine_discovery.PUBLISHED)), fallback


@pytest.mark.parametrize("script", [
    "blindspot_deepsf_calibrate.py", "blindspot_netside_vet.py",
    "blindspot_value_gap.py",
])
def test_no_blindspot_script_hardcodes_its_own_engine_path(script: str) -> None:
    """One definition, consumed — not three copies that drift.

    Asserted on the SOURCE because the constant is evaluated at import and a
    value check cannot tell a shared lookup from a lucky literal.
    """
    src = (REPO_ROOT / "scripts" / script).read_text(encoding="utf-8")
    assert "default_stockfish()" in src, f"{script} does not use the shared lookup"
    assert '"e2e_server"' not in src, (
        f"{script} still builds its own engine path; the shared discovery is "
        "the only definition"
    )
