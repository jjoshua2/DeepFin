"""The shallow-SF label cache must be keyed by WHICH ENGINE wrote it.

MEASURED 2026-08-16, before this guard existed: two `audit_targets.py` runs
differing ONLY in `--stockfish` (dev-20260420-ed651aab vs dev-20260810-5062aee5)
produced BYTE-IDENTICAL `cand.sf_soft` on all four positions, and the second run
never launched Stockfish at all -- the cache matched on (nodes, multipv) alone,
so arm NEW silently read arm OLD's labels. The prereg 2x2 for the teacher
upgrade would therefore have cost four 4000-position labeling runs and returned
a statistic that is 0 by construction.

That is this codebase's signature defect in the ruler itself: a value (the
`--stockfish` binary) accepted and then silently ignored. These tests pin the
identity key, the refusal on a mixed cache, and the `id name` read.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from chess_anti_engine.eval.audit import AuditPosition
from scripts.audit_targets import (
    UNRECORDED_SF_ID,
    _shallow_sf_records,
    engine_identity,
    refuse_if_not_a_shallow_sf_cache,
    resolve_sf_cache_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

NODES, MULTIPV = 500_000, 40


def _pos(key: str) -> AuditPosition:
    return AuditPosition(
        key=key, fen="8/8/8/8/8/8/8/K6k w - - 0 1", phase=0, source=0,
        move_cp={"a1b1": 0.0}, best_cp=0.0, deep_wdl=(0.0, 1.0, 0.0),
        sf_nodes=NODES, sf_depth=10,
    )


def _cache_row(key: str, sf_id: str | None) -> dict:
    row = {
        "key": key, "nodes_requested": NODES, "multipv": MULTIPV,
        "cp": 12, "mate": None, "wdl": [0.1, 0.8, 0.1],
        "pvs": [{"move": "a1b1", "cp": 12, "mate": None, "wdl": [0.1, 0.8, 0.1]}],
    }
    if sf_id is not None:
        row["sf_id"] = sf_id
    return row


def _write_cache(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _fake_engine(tmp_path: Path, name: str) -> str:
    """A minimal UCI responder: enough to answer `uci` with an `id name`."""
    p = tmp_path / f"engine_{abs(hash(name)) % 10**8}.py"
    p.write_text(
        "import sys\n"
        "for line in sys.stdin:\n"
        "    if line.strip() == 'uci':\n"
        f"        sys.stdout.write('id name {name}\\n')\n"
        "        sys.stdout.write('uciok\\n')\n"
        "        sys.stdout.flush()\n"
        "    elif line.strip() == 'quit':\n"
        "        break\n",
        encoding="utf-8",
    )
    sh = tmp_path / f"engine_{abs(hash(name)) % 10**8}.sh"
    sh.write_text(f"#!/bin/sh\nexec {sys.executable} {p}\n", encoding="utf-8")
    sh.chmod(sh.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(sh)


def _no_labeling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any attempt to actually label explode, so 'reused?' is unambiguous."""
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("StockfishUCI was constructed: the cache was NOT reused")
    monkeypatch.setattr("scripts.audit_targets.StockfishUCI", boom)


def test_a_different_engine_does_not_reuse_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug, pinned. Rows written by engine OLD must not serve engine NEW."""
    cache = _write_cache(tmp_path / "c.jsonl", [_cache_row("k0", "SF OLD")])
    monkeypatch.setattr("scripts.audit_targets.engine_identity", lambda p, **k: "SF NEW")
    labeled: list[str] = []
    monkeypatch.setattr(
        "scripts.audit_targets.StockfishUCI",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("relabel attempted")),
    )
    with pytest.raises(RuntimeError, match="relabel attempted"):
        _shallow_sf_records(
            [_pos("k0")], cache_path=cache, stockfish="/fake/new",
            nodes=NODES, multipv=MULTIPV, workers=1, nice=15,
        )
    assert labeled == []


def test_the_same_engine_still_reuses_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must not simply disable caching."""
    cache = _write_cache(tmp_path / "c.jsonl", [_cache_row("k0", "SF OLD")])
    monkeypatch.setattr("scripts.audit_targets.engine_identity", lambda p, **k: "SF OLD")
    _no_labeling(monkeypatch)
    out = _shallow_sf_records(
        [_pos("k0")], cache_path=cache, stockfish="/fake/old",
        nodes=NODES, multipv=MULTIPV, workers=1, nice=15,
    )
    assert set(out) == {"k0"}


def test_legacy_rows_without_sf_id_are_not_credited_to_a_named_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production cache's 10,000 rows carry no sf_id at all.

    Serving them to a named engine is precisely how the 2x2 read zero, so
    unrecorded provenance must be treated as foreign, not as a match.
    """
    cache = _write_cache(tmp_path / "c.jsonl", [_cache_row("k0", None)])
    monkeypatch.setattr("scripts.audit_targets.engine_identity", lambda p, **k: "SF NEW")
    monkeypatch.setattr(
        "scripts.audit_targets.StockfishUCI",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("relabel attempted")),
    )
    with pytest.raises(RuntimeError, match="relabel attempted"):
        _shallow_sf_records(
            [_pos("k0")], cache_path=cache, stockfish="/fake/new",
            nodes=NODES, multipv=MULTIPV, workers=1, nice=15,
        )


def test_a_cacheonly_run_refuses_a_mixed_provenance_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No --stockfish: reading two engines' rows as one ruler must be refused."""
    cache = _write_cache(tmp_path / "c.jsonl", [
        _cache_row("k0", "SF OLD"), _cache_row("k1", "SF NEW"),
    ])
    _no_labeling(monkeypatch)
    with pytest.raises(SystemExit) as e:
        _shallow_sf_records(
            [_pos("k0"), _pos("k1")], cache_path=cache, stockfish=None,
            nodes=NODES, multipv=MULTIPV, workers=1, nice=15,
        )
    assert "different engines" in str(e.value)


def test_a_cacheonly_run_on_a_single_engine_cache_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _write_cache(tmp_path / "c.jsonl", [
        _cache_row("k0", None), _cache_row("k1", None),
    ])
    _no_labeling(monkeypatch)
    out = _shallow_sf_records(
        [_pos("k0"), _pos("k1")], cache_path=cache, stockfish=None,
        nodes=NODES, multipv=MULTIPV, workers=1, nice=15,
    )
    assert set(out) == {"k0", "k1"}


def test_unrecorded_id_is_a_distinct_bucket_not_a_wildcard() -> None:
    assert UNRECORDED_SF_ID not in ("", None)


def test_engine_identity_reads_the_engines_own_id_name(tmp_path: Path) -> None:
    """Identity comes from the ENGINE, never the path.

    Production's `stockfish_path` is a two-hop symlink whose intermediate name
    is misleading, so a path-derived key would record the wrong provenance and
    still look plausible -- the file here is deliberately named nothing like
    what it reports.
    """
    exe = _fake_engine(tmp_path, "Stockfish dev-20260810-5062aee5")
    assert "misleading" not in exe
    assert engine_identity(exe) == "Stockfish dev-20260810-5062aee5"


def test_engine_identity_refuses_a_binary_that_reports_no_name(tmp_path: Path) -> None:
    sh = tmp_path / "silent.sh"
    sh.write_text("#!/bin/sh\nread x\necho uciok\n", encoding="utf-8")
    sh.chmod(sh.stat().st_mode | stat.S_IEXEC)
    with pytest.raises(SystemExit, match="id name"):
        engine_identity(str(sh), timeout_s=10.0)


def test_written_rows_carry_the_engine_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A row written today must be reusable-by-identity tomorrow."""
    cache = tmp_path / "c.jsonl"
    monkeypatch.setattr("scripts.audit_targets.engine_identity", lambda p, **k: "SF NEW")

    class _FakePv:
        move_uci, cp, mate, wdl = "a1b1", 5, None, (0.1, 0.8, 0.1)

    class _FakeRes:
        cp, mate, wdl = 5, None, (0.1, 0.8, 0.1)
        pvs = (_FakePv(),)

    class _FakeEngine:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def search(self, _fen: str, **_kwargs: object) -> _FakeRes:
            return _FakeRes()

        def close(self) -> None:
            pass

    monkeypatch.setattr("scripts.audit_targets.StockfishUCI", _FakeEngine)
    _shallow_sf_records(
        [_pos("k0")], cache_path=cache, stockfish="/fake/new",
        nodes=NODES, multipv=MULTIPV, workers=1, nice=15,
    )
    rows = [json.loads(x) for x in cache.read_text(encoding="utf-8").splitlines() if x]
    assert [r["sf_id"] for r in rows] == ["SF NEW"], rows
    assert os.path.exists(cache)


# --------------------------------------------------------------------------
# The REPEAT control. Engine identity fixes the OLD-vs-NEW arm and does
# NOTHING for a repeat: both runs are the same binary on purpose, so they
# share an `sf_id`, run 2 matches every cached row, and the engine is never
# launched. The prereg makes "run OLD twice and read the paired flip count"
# a MANDATORY first step and pre-commits d_obs=0 as a STOP -- so without a
# cache bypass that step is structurally guaranteed to halt the experiment,
# for a reason that has nothing to do with the pipeline's noise.
# --------------------------------------------------------------------------


class _DisagreeingEngine:
    """Returns a DIFFERENT cp on every call, across every instance.

    The point of the counter being a class attribute: if run 2 relabels at
    all, its rows cannot match run 1's, whatever the ordering. So "the two
    runs agree" can only mean the cache was served -- there is no
    determinism story available to explain it away.
    """

    calls = 0

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def search(self, _fen: str, **_kwargs: object) -> object:
        type(self).calls += 1
        cp = 100 * type(self).calls

        class _Pv:
            move_uci, mate, wdl = "a1b1", None, (0.1, 0.8, 0.1)

        class _Res:
            mate, wdl = None, (0.1, 0.8, 0.1)

        pv = _Pv()
        pv.cp = cp                      # pyright: ignore[reportAttributeAccessIssue]
        res = _Res()
        res.cp = cp                     # pyright: ignore[reportAttributeAccessIssue]
        res.pvs = (pv,)                 # pyright: ignore[reportAttributeAccessIssue]
        return res

    def close(self) -> None:
        pass


def _label_once(cache: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr("scripts.audit_targets.engine_identity", lambda p, **k: "SF SAME")
    monkeypatch.setattr("scripts.audit_targets.StockfishUCI", _DisagreeingEngine)
    return _shallow_sf_records(
        [_pos("k0")], cache_path=cache, stockfish="/fake/same",
        nodes=NODES, multipv=MULTIPV, workers=1, nice=15,
    )


def test_a_repeat_on_the_SHARED_cache_never_relabels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑⚑ #440 B5, pinned as BEHAVIOUR rather than as a warning in prose.

    The engine disagrees with itself on every call. If the repeat measured
    anything, the two runs would differ. They do not, and the call count
    proves why: the engine ran once.
    """
    _DisagreeingEngine.calls = 0
    cache = tmp_path / "shared.jsonl"
    first = _label_once(cache, monkeypatch)
    second = _label_once(cache, monkeypatch)
    assert _DisagreeingEngine.calls == 1, "run 2 relabelled; the premise changed"
    assert first["k0"]["cp"] == second["k0"]["cp"] == 100
    # d_obs would be 0 here -- and it is a statement about the cache, not
    # about the pipeline's run-to-run variance.


def test_a_repeat_on_a_FRESH_cache_does_relabel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bypass, and the reason it is the only fix: a distinct path relabels."""
    _DisagreeingEngine.calls = 0
    first = _label_once(tmp_path / "run1.jsonl", monkeypatch)
    second = _label_once(tmp_path / "run2.jsonl", monkeypatch)
    assert _DisagreeingEngine.calls == 2
    assert first["k0"]["cp"] == 100
    assert second["k0"]["cp"] == 200
    assert (tmp_path / "run2.jsonl").exists()


def test_resolve_sf_cache_path_defaults_beside_the_audit_set() -> None:
    assert resolve_sf_cache_path(Path("data/audit_set_v1.jsonl"), None) == Path(
        "data/audit_set_v1.jsonl.shallow_sf.jsonl"
    )


def test_resolve_sf_cache_path_honours_the_override() -> None:
    got = resolve_sf_cache_path(Path("data/audit_set_v1.jsonl"), Path("/tmp/rep2.jsonl"))
    assert got == Path("/tmp/rep2.jsonl")


def test_the_cli_resolves_the_override_before_anything_expensive_loads(
    tmp_path: Path,
) -> None:
    """⚑ EXECUTED, not read off the source.

    A resolver nothing calls is this codebase's signature defect. Drive the
    real `main()` with a checkpoint that cannot load: the run must still have
    printed which cache it would use, and it must be the overridden one. That
    also pins the ORDER -- an operator who mistyped the path finds out now
    rather than an hour into labelling.
    """
    override = tmp_path / "repeat2.shallow_sf.jsonl"
    r = subprocess.run(
        [sys.executable, "scripts/audit_targets.py",
         "--checkpoint", str(tmp_path / "nope.pt"),
         "--audit-set", str(tmp_path / "set.jsonl"),
         "--sf-cache", str(override)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": ".", "CUDA_VISIBLE_DEVICES": ""},
        timeout=600,
    )
    assert r.returncode != 0, "the bogus checkpoint should still fail the run"
    assert f"[sf-soft] cache {override}" in r.stdout, r.stdout[-3000:]
    assert "(--sf-cache override)" in r.stdout


def test_the_labeling_pass_announces_the_path_it_actually_uses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The one announcement that cannot lie: it prints its own parameter."""
    _DisagreeingEngine.calls = 0
    cache = tmp_path / "announced.jsonl"
    _label_once(cache, monkeypatch)
    assert f"[sf-soft] cache in use {cache}" in capsys.readouterr().out


def _one_row_audit_set(tmp_path: Path) -> Path:
    """The smallest thing `load_audit_set` accepts — one scored position."""
    p = tmp_path / "set.jsonl"
    p.write_text(json.dumps({
        "key": "k0",
        "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
        "phase": 0, "source": 0,
        "multipv": [{"move": "a1b1", "cp": 0}],
        "wdl": [0, 1000, 0],
        "nodes": 1000000, "depth": 40,
    }) + "\n", encoding="utf-8")
    return p


def test_main_hands_the_RESOLVED_path_to_the_labelling_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑⚑ THE WIRING CHECK, EXECUTED THROUGH `main()` — this is the one that counts.

    It replaces a source-level "the resolver is called exactly once" assertion
    that I wrote, recorded as adequate, and that was WRONG: the independent
    review of PR #446 defeated it with one token, using the constant this same
    change introduced —

        cache_path=args.audit_set.with_suffix(
            args.audit_set.suffix + SHALLOW_SF_CACHE_SUFFIX)

    `resolve_sf_cache_path(` still appears exactly once and the literal
    `shallow_sf.jsonl` is in the constant rather than in `main`'s source, so
    both string guards pass while the run announces the override and labels
    into the DEFAULT. 16 tests green with the mutant in.

    ⚑ And the reason I gave for settling — "reaching the labelling call needs a
    real checkpoint and an hour of Stockfish" — was FALSE, which is the part
    worth remembering. `net_source_from_args` only RECORDS `--checkpoint`; the
    model loads long after `_shallow_sf_records`. So `main()` reaches the
    labelling call in-process with a bogus checkpoint and a one-row audit set,
    in ~15 s, with no Stockfish, no torch load and no GPU. A limit assumed
    rather than measured turned a closable gap into a documented one.
    [[predict_the_exact_count_before_running]]
    """
    from scripts import audit_targets

    seen: dict[str, object] = {}

    def _capture(*_a: object, **kw: object) -> None:
        seen.update(kw)
        raise SystemExit("captured")

    monkeypatch.setattr(audit_targets, "_shallow_sf_records", _capture)
    override = tmp_path / "repeat2.shallow_sf.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "audit_targets.py",
        "--checkpoint", str(tmp_path / "nope.pt"),
        "--audit-set", str(_one_row_audit_set(tmp_path)),
        "--sf-cache", str(override),
        # Repo-relative: this file's own location, never an absolute home path.
        "--config", str(REPO_ROOT / "configs/pbt2_small.yaml"),
        "--allow-stale-config",
        "--device", "cpu",
    ])
    with pytest.raises(SystemExit):
        audit_targets.main()
    assert seen.get("cache_path") == override, seen


def test_main_resolves_the_cache_path_exactly_once() -> None:
    """⚑ A STRUCTURAL check, and its limit is stated rather than implied.

    The mutant this exists for SURVIVED every executing test in this file:
    `main` resolves and prints the override, then hands
    `resolve_sf_cache_path(args.audit_set, None)` to the labelling pass — the
    announcement is truthful and the run uses the default. Nothing here can
    observe that by execution, because reaching the labelling call needs a real
    checkpoint and an hour of Stockfish; the repo's other `main()` wiring
    checks (tests/test_audit_search_profiles.py) hit the same wall.

    So the invariant is enforced one level up: the resolution happens ONCE, and
    `main` never derives a cache path by hand. A second derivation is the only
    way the printed and the used path can diverge.
    """
    import inspect

    from scripts import audit_targets

    src = inspect.getsource(audit_targets.main)
    assert src.count("resolve_sf_cache_path(") == 1, (
        "main resolves the shallow-SF cache path more than once; the printed "
        "path and the labelled path can now disagree"
    )
    assert "shallow_sf.jsonl" not in src.replace(
        '"<audit-set>.shallow_sf.jsonl. ⚑ REQUIRED for a repeat "', ""
    ), "main derives a cache path by hand instead of calling the resolver"


# --------------------------------------------------------------------------
# The APPEND-collision guard. `_shallow_sf_records` opens the cache in mode
# "a", so a mis-pointed --sf-cache does not fail -- it silently grows another
# file. PR #446's review showed an identity check on the audit set named on
# THIS command line is not enough, and gave three routes past it.
# --------------------------------------------------------------------------


def _audit_row(key: str) -> dict:
    """An audit-set record: `multipv` is a LIST of PV dicts."""
    return {
        "key": key, "fen": "8/8/8/8/8/8/8/K6k w - - 0 1", "phase": 0, "source": 0,
        "multipv": [{"move": "a1b1", "cp": 0}], "wdl": [0, 1000, 0],
        "nodes": 1000000, "depth": 40,
    }


def test_the_resolver_refuses_an_override_that_aliases_the_audit_set(
    tmp_path: Path,
) -> None:
    aset = tmp_path / "set.jsonl"
    aset.write_text(json.dumps(_audit_row("k0")) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="audit set itself"):
        resolve_sf_cache_path(aset, aset)
    # ...including through a directory-traversal spelling of the same file.
    with pytest.raises(SystemExit, match="audit set itself"):
        resolve_sf_cache_path(aset, tmp_path / "sub" / ".." / "set.jsonl")


def test_the_resolver_lets_a_genuinely_new_path_through(tmp_path: Path) -> None:
    """The guard must not simply refuse overrides."""
    aset = tmp_path / "set.jsonl"
    got = resolve_sf_cache_path(aset, tmp_path / "repeat2.jsonl")
    assert got == tmp_path / "repeat2.jsonl"
    assert resolve_sf_cache_path(aset, None) == tmp_path / "set.jsonl.shallow_sf.jsonl"


def test_a_FRESH_cache_that_aliases_the_dump_is_refused(tmp_path: Path) -> None:
    """⚑⚑ THE COLLISION THE OTHER TWO GUARDS STRUCTURALLY CANNOT SEE.

    `--sf-cache X --dump-per-position X` where X does not exist yet passes BOTH
    existing guards: `refuse_if_not_a_shallow_sf_cache` inspects CONTENT and
    returns early on a missing file, and the audit-set alias check compares
    against a different path. The run then appends an hour of shallow-SF rows
    to X, and `write_audit_cache(..., force=True)` truncates X at the end and
    replaces it with the per-position dump — the run destroys its own most
    expensive output, with no error and nothing in the log that looks wrong.

    ⚑ This is the same lesson as the announced-vs-used mutant one level over:
    a guard that reads the FILE cannot see a hazard that exists only between
    two ARGUMENTS. Found by Codex review of PR #446 (P2).
    """
    aset = tmp_path / "set.jsonl"
    fresh = tmp_path / "runs" / "arm_a.jsonl"
    assert not fresh.exists(), "the whole point is that neither file exists yet"
    with pytest.raises(SystemExit, match="SAME path"):
        resolve_sf_cache_path(aset, fresh, fresh)


def test_the_dump_alias_check_resolves_symlinks_and_dot_spellings(
    tmp_path: Path,
) -> None:
    """A `./` spelling or a symlink must not route around the collision check."""
    aset = tmp_path / "set.jsonl"
    (tmp_path / "runs").mkdir()
    dump = tmp_path / "runs" / "arm_a.jsonl"
    dotted = tmp_path / "runs" / "." / "arm_a.jsonl"
    with pytest.raises(SystemExit, match="SAME path"):
        resolve_sf_cache_path(aset, dotted, dump)
    dump.write_text("", encoding="utf-8")
    link = tmp_path / "runs" / "link.jsonl"
    link.symlink_to(dump)
    with pytest.raises(SystemExit, match="SAME path"):
        resolve_sf_cache_path(aset, link, dump)


def test_distinct_cache_and_dump_paths_still_pass(tmp_path: Path) -> None:
    """The complement: the repeat control's own shape must not be refused."""
    aset = tmp_path / "set.jsonl"
    got = resolve_sf_cache_path(
        aset, tmp_path / "repeat_A1.shallow_sf.jsonl", tmp_path / "repeat_A1.jsonl",
    )
    assert got == tmp_path / "repeat_A1.shallow_sf.jsonl"
    # And the default cache never collides with a dump, since it is derived
    # from the audit set rather than named by the operator.
    assert resolve_sf_cache_path(aset, None, tmp_path / "repeat_A1.jsonl") == (
        tmp_path / "set.jsonl.shallow_sf.jsonl"
    )


def test_main_passes_the_dump_path_to_the_resolver() -> None:
    """⚑ The guard is worthless if `main` calls the resolver with two arguments.

    Source-level and labelled as such — but the executing wiring test one
    screen up (`test_main_hands_the_RESOLVED_path_to_the_labelling_pass`) does
    not exercise `--dump-per-position`, and adding a third argument to a call
    site is exactly the edit that gets made in one place and forgotten in the
    other.
    """
    import inspect

    import scripts.audit_targets as at

    src = inspect.getsource(at.main)
    calls = [
        line for line in src.splitlines() if "resolve_sf_cache_path(" in line
    ]
    assert len(calls) == 1, f"expected one resolver call in main, got {calls}"
    call_block = src.split("resolve_sf_cache_path(", 1)[1].split(")", 1)[0]
    assert "args.dump_per_position" in call_block, call_block


def test_a_HARDLINK_to_the_audit_set_is_refused_at_the_append(
    tmp_path: Path,
) -> None:
    """⚑⚑ The route the resolver structurally CANNOT close.

    `os.link` gives the frozen set a second name; `Path.resolve()` compares
    paths and cannot see inodes, so the alias passes the resolver. The content
    check at the append-open catches it, which is the reason that check lives
    beside the `open(..., "a")` and not only in the resolver.
    """
    aset = tmp_path / "set.jsonl"
    aset.write_text(json.dumps(_audit_row("k0")) + "\n", encoding="utf-8")
    hard = tmp_path / "hard.jsonl"
    os.link(aset, hard)
    assert resolve_sf_cache_path(aset, hard) == hard      # resolver passes it
    with pytest.raises(SystemExit, match="NOT a shallow-SF cache"):
        refuse_if_not_a_shallow_sf_cache(hard)
    # and the frozen set is untouched
    assert aset.read_text(encoding="utf-8").count("\n") == 1


def test_a_DIFFERENT_frozen_audit_set_is_refused(tmp_path: Path) -> None:
    """`--sf-cache data/audit_set_v2.jsonl` — a set this run never names."""
    other = tmp_path / "v2.jsonl"
    other.write_text(json.dumps(_audit_row("k0")) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="NOT a shallow-SF cache"):
        refuse_if_not_a_shallow_sf_cache(other)


def test_a_per_position_DUMP_is_refused(tmp_path: Path) -> None:
    """Arguably the likelier typo: the dump path, two flags away on the line."""
    dump = tmp_path / "arm.jsonl"
    dump.write_text(
        json.dumps({"key": "k0", "phase": 0, "cand": {"raw": {"top1": 12.0}}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="NOT a shallow-SF cache"):
        refuse_if_not_a_shallow_sf_cache(dump)


def test_a_real_shallow_sf_cache_and_a_fresh_path_both_pass(tmp_path: Path) -> None:
    """⚑ The guard must not refuse the two cases the flag exists for."""
    refuse_if_not_a_shallow_sf_cache(tmp_path / "does_not_exist.jsonl")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    refuse_if_not_a_shallow_sf_cache(empty)
    good = _write_cache(tmp_path / "good.jsonl", [_cache_row("k0", "SF OLD")])
    refuse_if_not_a_shallow_sf_cache(good)


def test_the_labelling_pass_runs_the_guard_not_just_the_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ Wired, not merely defined: the check must fire from inside the pass.

    A guard function nothing calls is this codebase's signature defect, and the
    resolver cannot stand in for it — the hardlink above proves the two see
    different things.
    """
    aset = tmp_path / "set.jsonl"
    aset.write_text(json.dumps(_audit_row("k0")) + "\n", encoding="utf-8")
    _no_labeling(monkeypatch)
    monkeypatch.setattr("scripts.audit_targets.engine_identity", lambda p, **k: "SF X")
    with pytest.raises(SystemExit, match="NOT a shallow-SF cache"):
        _shallow_sf_records(
            [_pos("k0")], cache_path=aset, stockfish="/fake/x",
            nodes=NODES, multipv=MULTIPV, workers=1, nice=15,
        )
