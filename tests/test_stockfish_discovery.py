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

import ast
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


#: ⚑⚑ EVERY former call site, not the three someone happened to check.
#:
#: The first version of this list held only the three `blindspot_*` scripts
#: Codex named, and the PR body claimed on that basis that "the lookup now lives
#: in ONE place". It did not: the SAME commit had converted
#: `bench_production_sf_workers.py` and `diagnose_gumbel_roots.py` from an
#: absolute path (which reached the published engine) to
#: `_REPO / "e2e_server/publish/stockfish"` (which resolves to nothing in a
#: worktree), and `bench_vs_sf.py` had carried a private candidate list since
#: before the PR. A partial consolidation is worse than none, because it is the
#: partial one that gets described as finished.
_ENGINE_CONSUMERS = (
    "blindspot_deepsf_calibrate.py", "blindspot_netside_vet.py",
    "blindspot_value_gap.py", "bench_production_sf_workers.py",
    "diagnose_gumbel_roots.py", "bench_vs_sf.py",
)


def _code_string_literals(tree: ast.AST) -> list[str]:
    """Every string literal the module EXECUTES with — docstrings excluded.

    ⚑ Comments and docstrings are where the defect gets DESCRIBED, so a plain
    substring grep over the source cannot tell "this script builds the published
    engine path" from "this script explains why it must not". Parsing separates
    them; grepping cannot.
    """
    docstrings = {
        node.body[0].value
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    return [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node not in docstrings
    ]


def _stockfish_defaults(tree: ast.AST) -> list[ast.expr]:
    """The `default=` expression of every `--stockfish*` argparse argument.

    Module-level names are resolved one hop, because three of these scripts
    write `_DEFAULT_SF = default_stockfish()` at import and then pass the name.
    """
    bindings: dict[str, ast.expr] = {}
    module = tree if isinstance(tree, ast.Module) else None
    for stmt in (module.body if module is not None else []):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name):
                bindings[target.id] = stmt.value

    defaults: list[ast.expr] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        flags = [a.value for a in node.args if isinstance(a, ast.Constant)]
        if not any(isinstance(f, str) and f.startswith("--stockfish") for f in flags):
            continue
        for kw in node.keywords:
            if kw.arg != "default":
                continue
            value = kw.value
            if isinstance(value, ast.Name) and value.id in bindings:
                value = bindings[value.id]
            defaults.append(value)
    return defaults


@pytest.mark.parametrize("script", _ENGINE_CONSUMERS)
def test_no_script_hardcodes_its_own_engine_path(script: str) -> None:
    """One definition, consumed — not six copies that drift.

    ⚑ Checked on the PARSED DEFAULT EXPRESSION, not with a substring grep. The
    previous revision asserted `'"e2e_server"' not in src`, which is a
    double-quoted, path-separator-free spelling: it could not see
    `default=str(_REPO / "e2e_server/publish/stockfish")` — the exact form this
    PR introduced in two of the files below — so the guard was green while the
    defect it names was in the tree. A grep for a string is not a check on the
    value that reaches argparse.
    """
    tree = ast.parse((REPO_ROOT / "scripts" / script).read_text(encoding="utf-8"))
    offending = [s for s in _code_string_literals(tree) if "e2e_server" in s]
    assert not offending, (
        f"{script} still names the published-engine path itself ({offending}); "
        "the shared discovery in chess_anti_engine.utils.engine_discovery is the "
        "only definition, and it is the only one that looks in the main checkout"
    )
    defaults = _stockfish_defaults(tree)
    assert defaults, f"{script} has no --stockfish* argument any more"
    for default in defaults:
        # `bench_vs_sf.py` alone defaults to None and resolves at CALL time,
        # because it must raise rather than name a path that does not exist.
        # That it consumes the shared list is proved by value, below.
        if isinstance(default, ast.Constant) and default.value is None:
            assert script == "bench_vs_sf.py", f"{script} defaults --stockfish to None"
            continue
        assert isinstance(default, ast.Call), ast.dump(default)
        assert isinstance(default.func, ast.Name), ast.dump(default)
        assert default.func.id == "default_stockfish", (
            f"{script}'s --stockfish default is {ast.unparse(default)!r}, not a "
            "call to the shared default_stockfish()"
        )


def test_bench_vs_sf_uses_the_shared_candidate_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ The one consumer with its own resolver, checked BY VALUE.

    It keeps a resolver because it must RAISE rather than return a path that
    does not exist, and because `$STOCKFISH_PATH` is its documented seam. What
    it must not keep is its own candidate LIST — so patch the shared one and
    watch the answer follow.
    """
    import scripts.bench_vs_sf as bench

    engine = _fake_engine(tmp_path, "sf_from_shared_list")
    monkeypatch.setattr(bench, "stockfish_candidates", lambda: [str(engine)])
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    assert bench._resolve_stockfish_path(None) == str(engine)

    # ...and with the shared list empty it raises, rather than silently
    # falling back to a private literal.
    monkeypatch.setattr(bench, "stockfish_candidates", list)
    with pytest.raises(FileNotFoundError):
        bench._resolve_stockfish_path(None)


# ---------------------------------------------------------------------------
# #441 review N3 — discovery must SAY which engine it picked, and the deep-SF
# tools must RECORD it.
#
# `find_stockfish()` returning `/usr/games/stockfish` and returning
# `<checkout>/e2e_server/publish/stockfish` are the same type and the same
# shape. Before this, three of the four `blindspot_*` tools printed neither and
# stored neither, so a substituted binary would have relabelled the audit set
# and the artifact would have looked identical — the same shape as the recorded
# burn where an SF cache key omitted engine identity.
#
#: The tools whose OUTPUT is a deep-SF label or a keep/kill verdict. For these,
#: the engine is part of the result, not part of the invocation.
_LABEL_TOOLS = (
    "blindspot_deepsf_calibrate.py", "blindspot_deepsf_gate.py",
    "blindspot_netside_vet.py", "blindspot_value_gap.py",
)


@pytest.mark.usefixtures("isolated")
def test_resolve_reports_WHICH_source_answered(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """The source label, per branch, by construction rather than by reading."""
    engine = _fake_engine(tmp_path, "sf")

    monkeypatch.setenv(ENV_VAR, str(engine))
    assert engine_discovery.resolve_stockfish() == (str(engine), engine_discovery.SOURCE_ENV)
    monkeypatch.delenv(ENV_VAR)

    # Nothing anywhere: "missing", DISTINCT from "found somewhere unexpected".
    assert engine_discovery.resolve_stockfish() == (None, engine_discovery.SOURCE_MISSING)

    # A distro engine — the substitution the announcement exists to call out.
    monkeypatch.setattr(engine_discovery, "DISTRO_CANDIDATES", (str(engine),))
    assert engine_discovery.resolve_stockfish() == (str(engine), engine_discovery.SOURCE_DISTRO)
    monkeypatch.setattr(engine_discovery, "DISTRO_CANDIDATES", ())

    # ...and on PATH.
    monkeypatch.setattr(engine_discovery.shutil, "which", lambda _n: str(engine))
    assert engine_discovery.resolve_stockfish() == (str(engine), engine_discovery.SOURCE_PATH)


@pytest.mark.usefixtures("isolated")
def test_a_SUBSTITUTED_engine_is_called_out_and_a_published_one_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑ The announcement must DISCRIMINATE, or it is decoration.

    Printing the path unconditionally would pass a "does it print" test while
    telling the reader nothing about the thing that changed: the pre-#441
    literal had exactly one possible value, and the derived default does not.
    """
    engine = _fake_engine(tmp_path, "sf")

    monkeypatch.setattr(engine_discovery, "DISTRO_CANDIDATES", (str(engine),))
    engine_discovery.announce_engine("t", str(engine))
    substituted = capsys.readouterr().err
    assert "source=distro" in substituted, substituted
    assert "NOT the one this checkout publishes" in substituted, substituted

    # The checkout's own engine: reported, but NOT flagged.
    monkeypatch.setattr(engine_discovery, "DISTRO_CANDIDATES", ())
    published = tmp_path / "checkout_with_engine"
    (published / engine_discovery.PUBLISHED).parent.mkdir(parents=True)
    real = _fake_engine((published / engine_discovery.PUBLISHED).parent, "stockfish")
    monkeypatch.setattr(engine_discovery, "REPO_ROOT", published)
    engine_discovery.announce_engine("t", str(real))
    ok = capsys.readouterr().err
    assert "source=checkout" in ok, ok
    assert "NOT the one this checkout publishes" not in ok, (
        "the checkout's OWN engine was flagged as a substitution — an alarm "
        "that fires every run is one people stop reading"
    )


def test_the_recorded_identity_is_the_CONTENT_not_just_the_path(tmp_path: Path) -> None:
    """⚑ A path is not an identity: the same path holds a different engine
    after a rebuild, which is exactly how a stale cache key produced wrong
    labels here before."""
    a = _fake_engine(tmp_path, "sf_a")
    same_path = tmp_path / "rebuilt"
    same_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    before = engine_discovery.engine_identity(str(same_path))
    same_path.write_text("#!/bin/sh\necho different\nexit 0\n", encoding="utf-8")
    after = engine_discovery.engine_identity(str(same_path))

    assert before["path"] == after["path"]
    assert before["sha256"] != after["sha256"], (
        "the record does not change when the BINARY changes at the same path — "
        "it is a path record, not an engine identity"
    )
    assert engine_discovery.engine_identity(str(a))["sha256"] is not None

    # Never raises on an unreadable engine: the tool must still produce a result.
    missing = engine_discovery.engine_identity(str(tmp_path / "nope"))
    assert missing["sha256"] is None
    assert missing["path"] is not None


def test_the_sidecar_record_lands_next_to_the_artifact(tmp_path: Path) -> None:
    import json

    out = tmp_path / "nested" / "audit.jsonl"
    side = engine_discovery.write_engine_record(out, {"path": "/x", "sha256": "abc"})
    assert side == tmp_path / "nested" / "audit.jsonl.engine.json"
    assert json.loads(side.read_text(encoding="utf-8"))["sha256"] == "abc"
    # ⚑ A SIDECAR, so it cannot corrupt a JSONL that downstream scripts parse
    # positionally. The artifact itself is untouched.
    assert not out.exists()


@pytest.mark.parametrize("script", _LABEL_TOOLS)
def test_every_deep_sf_label_tool_announces_its_engine(script: str) -> None:
    """⚑ Checked on the CALL, not on a substring of the source.

    Before #441's review, `blindspot_deepsf_calibrate.py` printed
    `stockfish=...` and the other three — a GATE, a net-side VET and a
    value-gap probe, all of which decide what survives — printed nothing and
    stored nothing. A per-file `print` would drift back apart, which is why
    they all go through the one helper.
    """
    tree = ast.parse((REPO_ROOT / "scripts" / script).read_text(encoding="utf-8"))
    called = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "announce_engine" in called, (
        f"{script} produces deep-SF labels or verdicts without ever saying "
        "which binary produced them"
    )
    assert "write_engine_record" in called, (
        f"{script} prints the engine but its ARTIFACT still carries no engine "
        "identity — the print goes into a redirected log nobody keeps"
    )
