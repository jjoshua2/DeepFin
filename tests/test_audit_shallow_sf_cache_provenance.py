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
    resolve_sf_cache_path,
)

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
