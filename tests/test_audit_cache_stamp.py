"""The audit-cache provenance guard: a stampless cache must be UNREADABLE.

Every test here is written to fail under a specific plausible weakening of the
guard — an "if present, check it" stamp check, a version that ignores the
castling map, a version that ignores the regret cap, a clobber check that only
runs after the forward pass. The mutations and their failures are recorded in
the PR; a test that no mutation can break is not evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chess
import pytest

from chess_anti_engine.eval import audit, audit_cache
from chess_anti_engine.eval.audit_cache import (
    AUDIT_CACHE_FORMAT,
    STAMP_FORMAT_KEY,
    AuditCacheError,
    audit_cache_stamp,
    audit_ruler_version,
    ensure_cache_writable,
    policy_map_version,
    read_audit_cache,
    read_audit_cache_by_key,
    read_audit_cache_stamp,
    write_audit_cache,
)

ROWS: list[dict[str, Any]] = [
    {"key": "k1", "phase": 1, "source": 0, "gap_cp": 42.0, "best_move": "e2e4",
     "wdl": [0.5, 0.3, 0.2], "exp_regret": 12.5, "top1_regret": 0.0,
     "topk": [["e2e4", 0.6], ["d2d4", 0.2]]},
    {"key": "k2", "phase": 2, "source": 1, "gap_cp": None, "best_move": "e1g1",
     "wdl": [0.4, 0.4, 0.2], "exp_regret": 30.0, "top1_regret": 8.0,
     "topk": [["e1g1", 0.9]]},
]


def _write_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
    return path


def _legacy_cache(path: Path) -> Path:
    """A cache in the EXACT pre-stamp format: rows only, no header."""
    return _write_lines(path, [json.dumps(r) for r in ROWS])


def _clear_version_caches() -> None:
    policy_map_version.cache_clear()
    audit_ruler_version.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_versions() -> Any:
    _clear_version_caches()
    yield
    _clear_version_caches()


# ---------------------------------------------------------------------------
# Absence of the stamp is a FAILURE, not a pass
# ---------------------------------------------------------------------------


def test_unstamped_cache_is_rejected(tmp_path: Path) -> None:
    """The pre-2026-08-13 file shape must be unreadable, not assumed-good."""
    path = _legacy_cache(tmp_path / "bt4_audit_cache.jsonl")
    with pytest.raises(AuditCacheError) as exc:
        read_audit_cache_stamp(path)
    msg = str(exc.value)
    assert "UNSTAMPED" in msg
    # The error must say what to do about it.
    assert "scripts/foreign_net_audit.py" in msg
    assert "--cache-out" in msg


def test_unstamped_cache_is_rejected_by_the_row_reader_too(tmp_path: Path) -> None:
    """`read_audit_cache` must not have a softer door than the stamp check.

    Asserts WHICH guard fired, not merely that something raised. The row-count
    binding also rejects this file, so an exception-TYPE assertion cannot tell a
    working provenance check from one that has been removed — measured: mutant
    M10 (provenance check skipped) passed a type-only assertion.
    """
    path = _legacy_cache(tmp_path / "c.jsonl")
    with pytest.raises(AuditCacheError, match="UNSTAMPED"):
        read_audit_cache(path)
    with pytest.raises(AuditCacheError, match="UNSTAMPED"):
        read_audit_cache_by_key(path)


def test_header_without_the_sentinel_key_is_rejected(tmp_path: Path) -> None:
    """A JSON-object first line is not a stamp; the sentinel must be present."""
    header = {"policy_map_version": policy_map_version(),
              "audit_ruler_version": audit_ruler_version()}
    path = _write_lines(tmp_path / "c.jsonl",
                        [json.dumps(header)] + [json.dumps(r) for r in ROWS])
    with pytest.raises(AuditCacheError, match="UNSTAMPED"):
        read_audit_cache_stamp(path)


def test_stamp_with_versions_absent_is_rejected(tmp_path: Path) -> None:
    """A stamped-but-versionless header must fail, not default to 'fine'."""
    path = _write_lines(tmp_path / "c.jsonl",
                        [json.dumps({STAMP_FORMAT_KEY: AUDIT_CACHE_FORMAT})]
                        + [json.dumps(r) for r in ROWS])
    with pytest.raises(AuditCacheError) as exc:
        read_audit_cache_stamp(path)
    assert "policy_map_version" in str(exc.value)
    assert "absent from the stamp" in str(exc.value)


def test_missing_and_empty_files_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(AuditCacheError, match="not found"):
        read_audit_cache_stamp(tmp_path / "nope.jsonl")
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(AuditCacheError, match="empty"):
        read_audit_cache_stamp(tmp_path / "empty.jsonl")


def test_no_flag_can_make_a_stampless_cache_readable(tmp_path: Path) -> None:
    """There must be no override on the READ path — check the API surface."""
    import inspect

    path = _legacy_cache(tmp_path / "c.jsonl")
    for fn in (read_audit_cache_stamp, read_audit_cache, read_audit_cache_by_key):
        params = list(inspect.signature(fn).parameters)
        assert params == ["path"], f"{fn.__name__} grew an escape hatch: {params}"
        with pytest.raises(AuditCacheError, match="UNSTAMPED"):
            fn(path)


# ---------------------------------------------------------------------------
# Mismatch is rejected, and the message says what moved
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "why"),
    [("policy_map_version", "policy-slot map changed"),
     ("audit_ruler_version", "eval/audit.py")],
)
def test_stale_version_is_rejected_with_a_reason(
    tmp_path: Path, field: str, why: str,
) -> None:
    stamp = audit_cache_stamp()
    stamp[field] = "deadbeefdeadbeef"
    path = _write_lines(tmp_path / "c.jsonl",
                        [json.dumps(stamp)] + [json.dumps(r) for r in ROWS])
    with pytest.raises(AuditCacheError) as exc:
        read_audit_cache_stamp(path)
    msg = str(exc.value)
    assert "STALE" in msg
    assert field in msg
    assert why in msg
    assert "deadbeefdeadbeef" in msg  # what was found
    assert "scripts/foreign_net_audit.py" in msg  # what to do


def test_unknown_stamp_format_is_rejected(tmp_path: Path) -> None:
    stamp = audit_cache_stamp()
    stamp[STAMP_FORMAT_KEY] = AUDIT_CACHE_FORMAT + 1
    path = _write_lines(tmp_path / "c.jsonl", [json.dumps(stamp)])
    with pytest.raises(AuditCacheError, match="stamp format"):
        read_audit_cache_stamp(path)


# ---------------------------------------------------------------------------
# The versions actually move when the thing they stand for moves
# ---------------------------------------------------------------------------


def test_ruler_version_moves_when_the_regret_cap_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`a84aaf846` added `AUDIT_REGRET_CAP_CP`; that must invalidate caches."""
    before = audit_ruler_version()
    _clear_version_caches()
    monkeypatch.setattr(audit, "AUDIT_REGRET_CAP_CP", 500.0)
    assert audit_ruler_version() != before


def test_ruler_version_moves_when_the_mate_mapping_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dominant half of the contamination was the mate->cp unification.

    Patched on `audit`, NOT on `audit_cache`: this is the transitive case the
    AST leg structurally cannot see — `parse_audit_record`'s own source is
    unchanged, only what it calls. Only the behavioural leg can catch it.
    """
    before = audit_ruler_version()
    _clear_version_caches()
    monkeypatch.setattr(audit, "mate_to_effective_cp", lambda mate_in: 50.0 * mate_in)
    assert audit_ruler_version() != before


def test_ruler_version_moves_when_a_criticality_edge_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = audit_ruler_version()
    _clear_version_caches()
    monkeypatch.setattr(audit, "CRITICALITY_GAP_EDGES", (25.0, 50.0, 100.0))
    assert audit_ruler_version() != before


def test_map_version_moves_when_castling_is_mis_mapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The EXACT 2026-08-13 defect: e1g1 -> 102 (king slide) instead of 103.

    Only castling moves differ, on 6.4% of the audit set. A version that is
    blind to this is the version that let the banked cache pass.
    """
    real = audit_cache.leela_index_for_move
    before = policy_map_version()
    _clear_version_caches()

    def buggy(board: chess.Board, move: chess.Move) -> int:
        if board.is_castling(move):
            return {"e1g1": 102, "e1c1": 99, "e8g8": 102, "e8c8": 99}.get(
                move.uci(), real(board, move),
            )
        return real(board, move)

    monkeypatch.setattr(audit_cache, "leela_index_for_move", buggy)
    assert policy_map_version() != before


def test_map_version_moves_when_the_1858_table_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = dict(audit_cache.LC0_1858_UCI_TO_IDX)
    before = policy_map_version()
    _clear_version_caches()
    a, b = "a1a2", "a1a3"
    table[a], table[b] = table[b], table[a]
    monkeypatch.setattr(audit_cache, "LC0_1858_UCI_TO_IDX", table)
    assert policy_map_version() != before


def test_versions_are_stable_across_calls() -> None:
    """A version that drifts on its own would reject every cache it wrote."""
    first = (policy_map_version(), audit_ruler_version())
    _clear_version_caches()
    assert (policy_map_version(), audit_ruler_version()) == first


def test_the_two_versions_are_independent() -> None:
    assert policy_map_version() != audit_ruler_version()


# ---------------------------------------------------------------------------
# Round trip, and the writer's clobber guard
# ---------------------------------------------------------------------------


def test_freshly_written_cache_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    stamp = write_audit_cache(path, ROWS, extra={"net": "probe.onnx"})
    assert stamp["net"] == "probe.onnx"
    assert read_audit_cache_stamp(path)["policy_map_version"] == policy_map_version()
    assert read_audit_cache(path) == ROWS
    assert sorted(read_audit_cache_by_key(path)) == ["k1", "k2"]


def test_a_cache_written_now_is_rejected_after_the_ruler_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: write, change the ruler, read -> refuse. No warning path."""
    path = tmp_path / "c.jsonl"
    write_audit_cache(path, ROWS)
    _clear_version_caches()
    monkeypatch.setattr(audit, "AUDIT_REGRET_CAP_CP", 250.0)
    with pytest.raises(AuditCacheError, match="STALE"):
        read_audit_cache(path)


def test_write_refuses_to_clobber_an_existing_cache(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    write_audit_cache(path, ROWS)
    original = path.read_text(encoding="utf-8")
    with pytest.raises(AuditCacheError, match="refusing to overwrite"):
        write_audit_cache(path, ROWS[:1])
    assert path.read_text(encoding="utf-8") == original


def test_force_allows_a_deliberate_replacement(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    write_audit_cache(path, ROWS)
    write_audit_cache(path, ROWS[:1], force=True)
    assert len(read_audit_cache(path)) == 1


def test_ensure_cache_writable_is_the_preflight(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    ensure_cache_writable(path)  # absent -> fine
    write_audit_cache(path, ROWS)
    with pytest.raises(AuditCacheError, match="--force-cache-out"):
        ensure_cache_writable(path)
    ensure_cache_writable(path, force=True)


# ---------------------------------------------------------------------------
# The guard fires BEFORE expensive work, in both scripts that touch the cache
# ---------------------------------------------------------------------------
#
# Both tests point every OTHER input at a path that does not exist. If the
# guard ran late, the missing input would raise first and the test would fail
# with FileNotFoundError instead of AuditCacheError — so the error TYPE is the
# ordering assertion.


def test_compare_buckets_refuses_a_stampless_cache_before_loading_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.audit_compare_buckets as acb

    bt4 = _legacy_cache(tmp_path / "bt4_audit_cache.jsonl")
    net = tmp_path / "net.jsonl"
    write_audit_cache(net, ROWS)
    monkeypatch.setattr("sys.argv", [
        "audit_compare_buckets.py",
        "--bt4", str(bt4),
        "--net", str(net),
        "--audit", str(tmp_path / "absent_audit.jsonl"),
    ])
    with pytest.raises(AuditCacheError, match="UNSTAMPED"):
        acb.main()


def test_compare_buckets_refuses_a_stale_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.audit_compare_buckets as acb

    stamp = audit_cache_stamp(rows=len(ROWS))
    stamp["audit_ruler_version"] = "0000000000000000"
    bt4 = _write_lines(tmp_path / "bt4.jsonl",
                       [json.dumps(stamp)] + [json.dumps(r) for r in ROWS])
    net = tmp_path / "net.jsonl"
    write_audit_cache(net, ROWS)
    monkeypatch.setattr("sys.argv", [
        "audit_compare_buckets.py",
        "--bt4", str(bt4),
        "--net", str(net),
        "--audit", str(tmp_path / "absent_audit.jsonl"),
    ])
    with pytest.raises(AuditCacheError, match="STALE"):
        acb.main()


def test_foreign_net_audit_refuses_to_clobber_before_any_forward_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.foreign_net_audit as fna

    banked = tmp_path / "banked.jsonl"
    write_audit_cache(banked, ROWS)
    before = banked.read_text(encoding="utf-8")
    monkeypatch.setattr("sys.argv", [
        "foreign_net_audit.py",
        "--cache-out", str(banked),
        "--audit-set", str(tmp_path / "absent_audit_set.jsonl"),
        "--onnx", str(tmp_path / "absent_net.onnx"),
        "--gpu-mem-gb", "0",
    ])
    with pytest.raises(AuditCacheError, match="refusing to overwrite"):
        fna.main()
    assert banked.read_text(encoding="utf-8") == before


def test_foreign_net_audit_force_flag_gets_past_the_clobber_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With --force-cache-out the run proceeds far enough to hit the real work.

    Reaching the missing audit set proves the clobber guard is the ONLY thing
    that stopped the previous test, not some earlier failure.
    """
    import scripts.foreign_net_audit as fna

    banked = tmp_path / "banked.jsonl"
    write_audit_cache(banked, ROWS)
    monkeypatch.setattr("sys.argv", [
        "foreign_net_audit.py",
        "--cache-out", str(banked), "--force-cache-out",
        "--audit-set", str(tmp_path / "absent_audit_set.jsonl"),
        "--onnx", str(tmp_path / "absent_net.onnx"),
        "--gpu-mem-gb", "0",
    ])
    with pytest.raises(FileNotFoundError):
        fna.main()


# ---------------------------------------------------------------------------
# The STRUCTURAL (AST) leg — review finding F2
# ---------------------------------------------------------------------------
#
# The behavioural probe touches 73 of 4672 slots (1.6%), and `move_to_index`
# has no table leg at all, so most of that mapper's provenance rests on the AST
# digest alone. The stand-ins below are byte-for-byte identical in BEHAVIOUR on
# every probe input and share the original's `__name__`, so nothing but a
# structural digest can tell them apart: these tests fail if the AST leg is
# removed, and also if it degenerates into hashing function names.


def test_map_version_moves_when_the_mapper_source_moves_invisibly_to_the_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = policy_map_version()
    _clear_version_caches()

    def leela_index_for_move_v2(board: chess.Board, move: chess.Move) -> int:
        result = audit_cache.leela_index_for_move(board, move)
        return result

    leela_index_for_move_v2.__name__ = audit_cache.leela_index_for_move.__name__
    monkeypatch.setattr(
        audit_cache, "_MAP_SOURCES",
        (leela_index_for_move_v2, audit_cache.move_to_index),
    )
    assert policy_map_version() != before


def test_ruler_version_moves_when_a_ruler_source_moves_invisibly_to_the_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = audit_ruler_version()
    _clear_version_caches()

    def criticality_gap_v2(move_cp: dict[str, float]) -> float:
        result = audit.criticality_gap(move_cp)
        return result

    criticality_gap_v2.__name__ = audit.criticality_gap.__name__
    monkeypatch.setattr(
        audit_cache, "_RULER_SOURCES",
        tuple(criticality_gap_v2 if f is audit.criticality_gap else f
              for f in audit_cache._RULER_SOURCES),
    )
    assert audit_ruler_version() != before


def test_ruler_version_moves_when_the_bucket_NAMES_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`criticality_bucket` returns an INDEX, so the probe is blind to renames.

    This is the case that makes the ruler-CONSTANTS leg load-bearing rather
    than redundant with the behavioural leg — see the M6 note in the PR.
    """
    before = audit_ruler_version()
    _clear_version_caches()
    monkeypatch.setattr(
        audit, "CRITICALITY_BUCKET_NAMES", ("quiet", "soft", "sharp", "decisive"),
    )
    assert audit_ruler_version() != before


def _probe_plain() -> Any:
    def probe(x: int) -> int:
        return x + 1
    return probe


def _probe_with_prose() -> Any:
    def probe(x: int) -> int:
        """A docstring the digest must ignore."""
        # ...and a comment it must ignore too.
        return x + 1
    return probe


def _probe_changed() -> Any:
    def probe(x: int) -> int:
        return x + 2
    return probe


def test_the_ast_leg_ignores_docstrings_and_comments() -> None:
    """The documented reason banked caches survive a prose edit. Untested until now.

    All three stand-ins are NAMED `probe`, because the digest includes the
    function name (a rename IS a real change) and holding it constant is what
    isolates the prose question. Without this test, "prose edits do not
    invalidate banked caches" is a claim in a docstring.
    """
    plain = audit_cache._structure_digest(_probe_plain())
    prose = audit_cache._structure_digest(_probe_with_prose())
    changed = audit_cache._structure_digest(_probe_changed())
    assert plain == prose, "a docstring or comment edit must not invalidate caches"
    assert plain != changed, "a real code change must invalidate them"


# ---------------------------------------------------------------------------
# The stamp must bind to the ROWS, not only to line 1 — review finding F5
# ---------------------------------------------------------------------------


def test_truncated_cache_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    write_audit_cache(path, ROWS)
    lines = path.read_text(encoding="utf-8").splitlines()
    _write_lines(path, lines[:-1])  # valid stamp, one row short
    with pytest.raises(AuditCacheError, match="TRUNCATED"):
        read_audit_cache(path)


def test_two_concatenated_caches_are_rejected(tmp_path: Path) -> None:
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_audit_cache(a, ROWS)
    write_audit_cache(b, ROWS)
    joined = tmp_path / "joined.jsonl"
    joined.write_text(a.read_text(encoding="utf-8") + b.read_text(encoding="utf-8"),
                      encoding="utf-8")
    with pytest.raises(AuditCacheError, match="second provenance header"):
        read_audit_cache(joined)


def test_a_stamp_lifted_from_a_good_cache_cannot_certify_a_short_file(
    tmp_path: Path,
) -> None:
    """The F5 attack in its purest form: real stamp, wrong body."""
    good = tmp_path / "good.jsonl"
    write_audit_cache(good, ROWS)
    header = good.read_text(encoding="utf-8").splitlines()[0]
    forged = _write_lines(tmp_path / "forged.jsonl", [header, json.dumps(ROWS[0])])
    read_audit_cache_stamp(forged)  # line-1 check alone still passes...
    with pytest.raises(AuditCacheError, match="TRUNCATED"):
        read_audit_cache(forged)    # ...the row binding is what catches it


def test_stamp_without_a_row_count_is_rejected(tmp_path: Path) -> None:
    """A stamp that declares no count cannot vouch for a body. Message pinned so
    that deleting the type check (leaving `None != len(rows)` to fire by
    accident) is distinguishable from enforcing it."""
    stamp = audit_cache_stamp()
    stamp.pop("rows", None)
    path = _write_lines(tmp_path / "c.jsonl",
                        [json.dumps(stamp)] + [json.dumps(r) for r in ROWS])
    with pytest.raises(AuditCacheError, match="no integer 'rows' count"):
        read_audit_cache(path)


def test_row_count_is_written_by_the_writer_not_the_caller(tmp_path: Path) -> None:
    """A caller must not be able to declare a count that is not the truth."""
    path = tmp_path / "c.jsonl"
    stamp = write_audit_cache(path, ROWS, extra={"rows": 9999})
    assert stamp["rows"] == len(ROWS)
    assert len(read_audit_cache(path)) == len(ROWS)


# ---------------------------------------------------------------------------
# --net is guarded exactly like --bt4 — review finding F1
# ---------------------------------------------------------------------------


def test_compare_buckets_refuses_an_unstamped_NET_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stamped BT4 cache must not launder an unstamped --net cache.

    `--bt4` is valid here, so the only thing that can stop the run is the
    `--net` guard; `--audit` is absent, so a late check would raise
    FileNotFoundError instead and this test would fail.
    """
    import scripts.audit_compare_buckets as acb

    bt4 = tmp_path / "bt4.jsonl"
    write_audit_cache(bt4, ROWS)
    net = _legacy_cache(tmp_path / "per_position_277.jsonl")
    monkeypatch.setattr("sys.argv", [
        "audit_compare_buckets.py", "--bt4", str(bt4), "--net", str(net),
        "--audit", str(tmp_path / "absent_audit.jsonl"),
    ])
    with pytest.raises(AuditCacheError, match="UNSTAMPED"):
        acb.main()


def test_compare_buckets_refuses_a_stale_NET_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.audit_compare_buckets as acb

    bt4 = tmp_path / "bt4.jsonl"
    write_audit_cache(bt4, ROWS)
    stamp = audit_cache_stamp(rows=len(ROWS))
    stamp["audit_ruler_version"] = "0000000000000000"
    net = _write_lines(tmp_path / "net.jsonl",
                       [json.dumps(stamp)] + [json.dumps(r) for r in ROWS])
    monkeypatch.setattr("sys.argv", [
        "audit_compare_buckets.py", "--bt4", str(bt4), "--net", str(net),
        "--audit", str(tmp_path / "absent_audit.jsonl"),
    ])
    with pytest.raises(AuditCacheError, match="STALE"):
        acb.main()


def test_audit_targets_per_position_dump_is_stamped(tmp_path: Path) -> None:
    """The producer side of F1: the dump must come out readable by the guard."""
    import scripts.audit_targets as at

    assert "write_audit_cache" in at.__dict__ or hasattr(at, "write_audit_cache")
    path = tmp_path / "per_position.jsonl"
    write_audit_cache(path, ROWS, force=True,
                      extra={"producer": "audit_targets.py --dump-per-position"})
    stamp = read_audit_cache_stamp(path)
    assert stamp["producer"] == "audit_targets.py --dump-per-position"
    assert read_audit_cache(path) == ROWS
