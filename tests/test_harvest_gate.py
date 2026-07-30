"""Unit tests for the harvest → deep-SF gate → stage step.

SF is injected/mocked — no real Stockfish, no GPU. Covers: offset advance
idempotency, dedup (pool/holdout/emitted), vet-lost boundary, max-vet cap,
malformed-line skip, and dry-run write suppression.
"""
from __future__ import annotations

import json
from pathlib import Path

import chess
import pytest

from scripts.harvest_gate_step import (
    GateState,
    dump_gate_state,
    format_summary,
    load_gate_state,
    parse_seed_line,
    placement_key,
    position_key,
    read_new_lines,
    run_gate,
    stage_line,
)

# Usable mid-game positions with ≥2 legal moves (pass _fen_reject_reason).
_FEN_A = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
_FEN_B = "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2"
_FEN_C = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
_FEN_D = "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
_FEN_E = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5"

_STAMP = "2026-07-13T00:00:00Z"


def _key(fen: str) -> str:
    return position_key(fen)


def _line(fen: str, *, nq: float = 0.6, sq: float = -0.7, game: str = "g1", ply: int = 10) -> str:
    return f"{fen}  # nq={nq:.2f} sq={sq:.2f} sev=1 game={game} ply={ply}"


def _hist_line(
    start: str,
    moves: str,
    *,
    nq: float = 0.6,
    sq: float = -0.7,
    game: str = "g1",
    ply: int = 10,
) -> str:
    # Real harvest grammar: start_fen | uci moves  # provenance
    return (
        f"{start} | {moves}  # nq={nq:.2f} sq={sq:.2f} "
        f"sev=1 game={game} ply={ply}"
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(text)


# ── placement key parity with mine_blindspot_seeds ───────────────────────────


def test_placement_key_matches_mine_and_ignores_clocks() -> None:
    a = _FEN_A
    b = a.replace(" 3 3", " 40 41")
    assert placement_key(a) == placement_key(b)
    assert placement_key(a) == position_key(a)
    # epd form: placement + turn + castling + ep
    assert placement_key(a) == " ".join(chess.Board(a).fen().split()[:4])


# ── offset advance ───────────────────────────────────────────────────────────


def test_offset_advance_second_run_reads_zero(tmp_path: Path) -> None:
    harvest = tmp_path / "severe.p1.txt"
    _write(harvest, _line(_FEN_A) + "\n" + _line(_FEN_B) + "\n")

    offsets: dict[str, int] = {}
    lines1, off1, n1 = read_new_lines([str(harvest)], offsets)
    assert n1 == 2
    assert len(lines1) == 2
    assert off1[str(harvest)] == harvest.stat().st_size

    lines2, off2, n2 = read_new_lines([str(harvest)], off1)
    assert n2 == 0
    assert lines2 == []
    assert off2[str(harvest)] == off1[str(harvest)]

    # Append one more line — only the new line is returned.
    _append(harvest, _line(_FEN_C) + "\n")
    lines3, off3, n3 = read_new_lines([str(harvest)], off2)
    assert n3 == 1
    assert len(lines3) == 1
    assert _FEN_C in lines3[0]
    assert off3[str(harvest)] == harvest.stat().st_size


def test_offset_handles_partial_trailing_line(tmp_path: Path) -> None:
    harvest = tmp_path / "severe.p1.txt"
    full = _line(_FEN_A) + "\n"
    partial = _line(_FEN_B)[:20]  # incomplete
    _write(harvest, full + partial)

    lines, off, n = read_new_lines([str(harvest)], {})
    assert n == 1
    assert len(lines) == 1
    assert off[str(harvest)] == len(full.encode("utf-8"))

    # Finish the partial line + newline → second run picks it up.
    with open(harvest, "a", encoding="utf-8") as fh:
        fh.write(_line(_FEN_B)[20:] + "\n")
    lines2, _off2, n2 = read_new_lines([str(harvest)], off)
    assert n2 == 1
    assert _FEN_B in lines2[0]


# ── parse / malformed ────────────────────────────────────────────────────────


def test_malformed_line_skipped_not_crash() -> None:
    assert parse_seed_line("") is None
    assert parse_seed_line("   ") is None
    assert parse_seed_line("# comment only") is None
    assert parse_seed_line("not a fen at all") is None
    assert parse_seed_line("8/8/8/8/8/8/8/8 w - - 0 1") is None  # no kings
    # Forced single-legal-move positions are rejected by the loader predicate.
    forced = "r5k1/p1P3pp/3QN1q1/3P1pB1/8/5P2/P6P/2n1q1K1 w - - 0 31"
    assert parse_seed_line(forced) is None
    # Good line works.
    got = parse_seed_line(_line(_FEN_A))
    assert got is not None
    key, body = got
    assert key == _key(_FEN_A)
    assert body == _FEN_A


def test_history_line_keys_on_terminal() -> None:
    # startpos + e2e4 e7e5 → terminal is the position after those two plies.
    start = chess.Board().fen()
    raw = _hist_line(start, "e2e4 e7e5")
    got = parse_seed_line(raw)
    assert got is not None
    key, body = got
    b = chess.Board()
    b.push_uci("e2e4")
    b.push_uci("e7e5")
    assert key == position_key(b.fen())
    assert "|" in body


# ── dedup ────────────────────────────────────────────────────────────────────


def test_dedup_drops_pool_holdout_and_emitted(tmp_path: Path) -> None:
    out = tmp_path / "staged.txt"
    # Pool/holdout/emitted share keys with A/B/C; only D is new.
    exclude = {_key(_FEN_A), _key(_FEN_B)}
    state = GateState(emitted={_key(_FEN_C)})
    scores = {
        _key(_FEN_D): -0.95,
        _key(_FEN_E): -0.90,
    }

    def sf_score(fen: str) -> float | None:
        return scores.get(position_key(fen))

    new_lines = [
        _line(_FEN_A),  # pool
        _line(_FEN_B),  # holdout
        _line(_FEN_C),  # already emitted
        _line(_FEN_D),  # keep
        _line(_FEN_D),  # dup of D within batch
        _line(_FEN_E),  # keep
    ]
    counts, staged, new_state = run_gate(
        new_lines=new_lines,
        state=state,
        exclude_keys=exclude,
        sf_score=sf_score,
        vet_lost_below=-0.80,
        max_vet_per_run=30,
        dry_run=False,
        stamp=_STAMP,
        out_path=str(out),
        new_offsets={},
    )
    assert counts.unique_new == 5  # A,B,C,D,E (second D not unique)
    assert counts.after_dedup == 2  # D, E
    assert counts.vetted_kept == 2
    assert counts.vetted_rejected == 0
    assert len(staged) == 2
    assert _key(_FEN_D) in new_state.emitted
    assert _key(_FEN_E) in new_state.emitted
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert _FEN_D in text
    assert _FEN_E in text
    assert "staged " + _STAMP in text


def test_previously_rejected_not_revet(tmp_path: Path) -> None:
    out = tmp_path / "staged.txt"
    state = GateState(rejected={_key(_FEN_A)})
    calls: list[str] = []

    def sf_score(fen: str) -> float | None:
        calls.append(fen)
        return -0.99

    counts, staged, _st = run_gate(
        new_lines=[_line(_FEN_A)],
        state=state,
        exclude_keys=set(),
        sf_score=sf_score,
        vet_lost_below=-0.80,
        max_vet_per_run=30,
        dry_run=False,
        stamp=_STAMP,
        out_path=str(out),
    )
    assert counts.after_dedup == 0
    assert counts.vetted_kept == 0
    assert staged == []
    assert calls == []  # never re-vet


# ── vet-lost-below boundary ──────────────────────────────────────────────────


def test_vet_lost_below_boundary(tmp_path: Path) -> None:
    out = tmp_path / "staged.txt"
    # A at exactly -0.80 → keep; B at -0.799 → reject; C at -0.95 → keep.
    threshold = -0.80
    scores = {
        _key(_FEN_A): -0.80,
        _key(_FEN_B): -0.799,
        _key(_FEN_C): -0.95,
    }

    def sf_score(fen: str) -> float | None:
        return scores[position_key(fen)]

    counts, staged, new_state = run_gate(
        new_lines=[_line(_FEN_A), _line(_FEN_B), _line(_FEN_C)],
        state=GateState(),
        exclude_keys=set(),
        sf_score=sf_score,
        vet_lost_below=threshold,
        max_vet_per_run=30,
        dry_run=False,
        stamp=_STAMP,
        out_path=str(out),
    )
    assert counts.vetted_kept == 2
    assert counts.vetted_rejected == 1
    assert _key(_FEN_A) in new_state.emitted
    assert _key(_FEN_C) in new_state.emitted
    assert _key(_FEN_B) in new_state.rejected
    assert len(staged) == 2


# ── max-vet-per-run cap ──────────────────────────────────────────────────────


def test_max_vet_per_run_cap_reports_capped(tmp_path: Path) -> None:
    out = tmp_path / "staged.txt"
    fens = [_FEN_A, _FEN_B, _FEN_C, _FEN_D, _FEN_E]
    calls: list[str] = []

    def sf_score(fen: str) -> float | None:
        calls.append(position_key(fen))
        return -0.99

    counts, staged, new_state = run_gate(
        new_lines=[_line(f) for f in fens],
        state=GateState(),
        exclude_keys=set(),
        sf_score=sf_score,
        vet_lost_below=-0.80,
        max_vet_per_run=2,
        dry_run=False,
        stamp=_STAMP,
        out_path=str(out),
    )
    assert counts.after_dedup == 5
    assert counts.vetted_kept == 2
    assert counts.capped == 3
    assert len(calls) == 2  # hard cap honored
    assert len(staged) == 2
    # Newest-first: the two NEWEST (E, D) are vetted; the older three stay
    # pending (not silently lost).
    assert len(new_state.pending) == 3
    pending_keys = {k for k, _ in new_state.pending}
    assert pending_keys == {_key(_FEN_A), _key(_FEN_B), _key(_FEN_C)}

    # Second run with no new lines drains pending under the same cap.
    calls.clear()
    counts2, staged2, new_state2 = run_gate(
        new_lines=[],
        state=new_state,
        exclude_keys=set(),
        sf_score=sf_score,
        vet_lost_below=-0.80,
        max_vet_per_run=2,
        dry_run=False,
        stamp=_STAMP,
        out_path=str(out),
    )
    assert counts2.new_lines == 0
    assert counts2.vetted_kept == 2
    assert counts2.capped == 1
    assert len(calls) == 2
    assert len(staged2) == 2
    assert len(new_state2.pending) == 1


# ── dry-run ──────────────────────────────────────────────────────────────────


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "staged.txt"
    state_path = tmp_path / "state.json"
    state = GateState()

    def sf_score(fen: str) -> float | None:
        raise AssertionError(f"SF must not be called in dry-run, got {fen}")

    counts, staged, returned = run_gate(
        new_lines=[_line(_FEN_A), _line(_FEN_B)],
        state=state,
        exclude_keys=set(),
        sf_score=sf_score,
        vet_lost_below=-0.80,
        max_vet_per_run=30,
        dry_run=True,
        stamp=_STAMP,
        out_path=str(out),
        new_offsets={str(tmp_path / "h.txt"): 99},
    )
    assert counts.new_lines == 2
    assert counts.unique_new == 2
    assert counts.after_dedup == 2
    assert counts.vetted_kept == 0
    assert counts.vetted_rejected == 0
    assert counts.capped == 0
    assert staged == []
    assert not out.exists()
    assert not state_path.exists()
    # Caller state untouched.
    assert returned is state
    assert state.emitted == set()
    assert state.offsets == {}

    summary = format_summary(counts, dry_run=True)
    assert summary.startswith("harvest_gate:")
    assert "DRY:0" in summary


# ── end-to-end idempotent second pass ────────────────────────────────────────


def test_full_step_idempotent_with_state(tmp_path: Path) -> None:
    harvest = tmp_path / "severe.p1.txt"
    out = tmp_path / "staged.txt"
    _write(harvest, _line(_FEN_A) + "\n" + _line(_FEN_B) + "\n")

    def sf_score(_fen: str) -> float | None:
        return -0.99

    lines1, off1, _ = read_new_lines([str(harvest)], {})
    counts1, _s1, st1 = run_gate(
        new_lines=lines1,
        state=GateState(),
        exclude_keys=set(),
        sf_score=sf_score,
        vet_lost_below=-0.80,
        max_vet_per_run=30,
        dry_run=False,
        stamp=_STAMP,
        out_path=str(out),
        new_offsets=off1,
    )
    assert counts1.vetted_kept == 2
    assert counts1.staged_total == 2
    assert st1.offsets[str(harvest)] == harvest.stat().st_size

    # Persist + reload state like main() does.
    raw = dump_gate_state(st1)
    st_reloaded = load_gate_state(raw)
    lines2, off2, n2 = read_new_lines([str(harvest)], st_reloaded.offsets)
    assert n2 == 0
    assert lines2 == []

    calls: list[str] = []

    def sf_score_guard(fen: str) -> float | None:
        calls.append(fen)
        return -0.99

    counts2, staged2, st2 = run_gate(
        new_lines=lines2,
        state=st_reloaded,
        exclude_keys=set(),
        sf_score=sf_score_guard,
        vet_lost_below=-0.80,
        max_vet_per_run=30,
        dry_run=False,
        stamp=_STAMP,
        out_path=str(out),
        new_offsets=off2,
    )
    assert counts2.new_lines == 0
    assert counts2.unique_new == 0
    assert counts2.after_dedup == 0
    assert counts2.vetted_kept == 0
    assert counts2.vetted_rejected == 0
    assert counts2.capped == 0
    assert staged2 == []
    assert calls == []
    assert counts2.staged_total == 2  # unchanged
    assert st2.offsets == st_reloaded.offsets


def test_state_roundtrip() -> None:
    st = GateState(
        offsets={"/a": 10, "/b": 20},
        emitted={"e1"},
        rejected={"r1"},
        pending=[("p1", "line1")],
    )
    raw = dump_gate_state(st)
    # JSON-serializable
    blob = json.dumps(raw)
    back = load_gate_state(json.loads(blob))
    assert back.offsets == {"/a": 10, "/b": 20}
    assert back.emitted == {"e1"}
    assert back.rejected == {"r1"}
    assert back.pending == [("p1", "line1")]
    assert load_gate_state(None).offsets == {}
    assert load_gate_state({}).emitted == set()


def test_stage_line_uses_stamp() -> None:
    ln = stage_line(_FEN_A, stamp="STAMP1", score=-0.91)
    assert ln.startswith(_FEN_A)
    assert "# staged STAMP1" in ln
    assert "deep_sq=-0.910" in ln


def test_summary_format() -> None:
    from scripts.harvest_gate_step import GateCounts

    c = GateCounts(
        new_lines=4, unique_new=3, after_dedup=2,
        vetted_kept=1, vetted_rejected=1, sf_failed=0, capped=0, staged_total=5,
    )
    s = format_summary(c)
    assert s == (
        "harvest_gate: new_lines=4 unique_new=3 after_dedup=2 "
        "vetted_kept=1 vetted_rejected=1 sf_failed=0 capped=0 staged_total=5"
    )


# ── newest-first ordering: fresh captures beat the stale pending backlog ──────


def test_newest_first_vets_new_before_pending() -> None:
    # One old pending entry + one fresh new line, budget for exactly one vet.
    # Newest-first must spend it on the NEW capture, leaving the old one pending.
    out = "/dev/null"

    def sf_score(_fen: str) -> float | None:
        return -0.99  # everything "lost" — so the CHOICE of what to vet is what matters

    counts, _staged, st = run_gate(
        new_lines=[_line(_FEN_A)],
        state=GateState(pending=[(_key(_FEN_B), _line(_FEN_B))]),
        exclude_keys=set(),
        sf_score=sf_score,
        vet_lost_below=-0.80,
        max_vet_per_run=1,
        dry_run=False,
        stamp=_STAMP,
        out_path=out,
    )
    assert counts.vetted_kept == 1
    assert _key(_FEN_A) in st.emitted          # the NEW capture was vetted
    assert _key(_FEN_A) not in {k for k, _ in st.pending}
    assert _key(_FEN_B) not in st.emitted      # the old pending was NOT reached
    assert _key(_FEN_B) in {k for k, _ in st.pending}


def test_pending_cap_expires_stale_tail() -> None:
    # Four stale pending + a tiny vet budget: the queue must be bounded to the
    # newest `pending_cap`, expiring the oldest tail rather than hoarding it.
    stale = [(_key(f), _line(f)) for f in (_FEN_B, _FEN_C, _FEN_D, _FEN_E)]

    def sf_score(_fen: str) -> float | None:
        return 0.99  # not-lost → rejected, so nothing is emitted; all flow to pending logic

    _counts, _staged, st = run_gate(
        new_lines=[],
        state=GateState(pending=stale),
        exclude_keys=set(),
        sf_score=sf_score,
        vet_lost_below=-0.80,
        max_vet_per_run=1,
        dry_run=False,
        stamp=_STAMP,
        out_path="/dev/null",
        pending_cap=2,
    )
    # The cap must expire the OLDEST tail and keep the NEWEST — a regression that
    # kept the oldest `pending_cap` (the stale-backlog starvation this fix cures)
    # would also satisfy `len <= 2`, so pin the surviving keys explicitly.
    # Newest-first vets E (→ rejected); un-vetted {B(oldest),C,D} trim to newest 2.
    pending_keys = {k for k, _ in st.pending}
    assert len(st.pending) == 2
    assert pending_keys == {_key(_FEN_C), _key(_FEN_D)}
    assert _key(_FEN_B) not in pending_keys      # oldest tail expired
    assert _key(_FEN_E) in st.rejected           # newest was vetted


# ── sf failure → re-queue (NOT reject): a transient SF outage must not ────────
# permanently discard a genuine candidate; it re-vets next run.


def test_sf_none_requeues_not_rejects(tmp_path: Path) -> None:
    out = tmp_path / "staged.txt"

    def sf_score(_fen: str) -> float | None:
        return None

    counts, staged, st = run_gate(
        new_lines=[_line(_FEN_A)],
        state=GateState(),
        exclude_keys=set(),
        sf_score=sf_score,
        vet_lost_below=-0.80,
        max_vet_per_run=30,
        dry_run=False,
        stamp=_STAMP,
        out_path=str(out),
    )
    assert counts.vetted_kept == 0
    assert counts.vetted_rejected == 0
    assert counts.sf_failed == 1
    assert staged == []
    # NOT permanently rejected — re-queued as pending for a later run.
    assert _key(_FEN_A) not in st.rejected
    assert any(k == _key(_FEN_A) for k, _ in st.pending)
    assert not out.exists() or out.read_text(encoding="utf-8").strip() == ""


def test_max_vet_zero_caps_all(tmp_path: Path) -> None:
    out = tmp_path / "staged.txt"
    calls: list[str] = []

    def sf_score(fen: str) -> float | None:
        calls.append(fen)
        return -0.99

    counts, staged, st = run_gate(
        new_lines=[_line(_FEN_A), _line(_FEN_B)],
        state=GateState(),
        exclude_keys=set(),
        sf_score=sf_score,
        vet_lost_below=-0.80,
        max_vet_per_run=0,
        dry_run=False,
        stamp=_STAMP,
        out_path=str(out),
    )
    assert counts.capped == 2
    assert counts.vetted_kept == 0
    assert calls == []
    assert staged == []
    assert len(st.pending) == 2


@pytest.mark.parametrize("bad", ["", "garbage", "8/8/8/8/8/8/8/4K3 w - - 0 1"])
def test_run_gate_tolerates_malformed_among_good(tmp_path: Path, bad: str) -> None:
    out = tmp_path / "staged.txt"

    def sf_score(_fen: str) -> float | None:
        return -0.99

    counts, staged, _st = run_gate(
        new_lines=[bad, _line(_FEN_A), bad],
        state=GateState(),
        exclude_keys=set(),
        sf_score=sf_score,
        vet_lost_below=-0.80,
        max_vet_per_run=30,
        dry_run=False,
        stamp=_STAMP,
        out_path=str(out),
    )
    assert counts.new_lines == 3
    assert counts.unique_new == 1
    assert counts.vetted_kept == 1
    assert len(staged) == 1


# ── source ordering: curriculum-first ────────────────────────────────────────


def _sp_line(fen: str, *, sp: int | None, game: str = "g1") -> str:
    """Harvest line with (or deliberately without) the sp= provenance tag."""
    tag = "" if sp is None else f" sp={sp}"
    return (f"{fen}  # nq=0.60 sq=-0.70 sev=1 game={game} ply=10{tag}")


def test_source_rank_reads_the_tag() -> None:
    from scripts.harvest_gate_step import (
        SRC_CURRICULUM,
        SRC_SELFPLAY,
        SRC_UNKNOWN,
        source_rank,
    )

    assert source_rank(_sp_line(_FEN_A, sp=0)) == SRC_CURRICULUM
    assert source_rank(_sp_line(_FEN_A, sp=1)) == SRC_SELFPLAY
    # Absent tag must NOT be guessed as either source.
    assert source_rank(_sp_line(_FEN_A, sp=None)) == SRC_UNKNOWN
    # `sp=` must not be matched inside another token (e.g. a game id ending sp=1).
    assert source_rank(f"{_FEN_A}  # game=xsp=1 ply=3") == SRC_UNKNOWN
    assert SRC_CURRICULUM < SRC_UNKNOWN < SRC_SELFPLAY


def test_curriculum_vetted_before_selfplay_under_budget() -> None:
    """The whole point: a 1-vet budget must be spent on the curriculum capture.

    Ordering is what buys the 1.9x yield, so this pins the CHOICE, not the
    outcome — sf_score is constant so only the pick can differ.
    """
    calls: list[str] = []

    def sf_score(fen: str) -> float | None:
        calls.append(position_key(fen))
        return -0.99

    # Selfplay line is NEWEST, so plain newest-first would vet it first.
    counts, _staged, st = run_gate(
        new_lines=[_sp_line(_FEN_A, sp=0), _sp_line(_FEN_B, sp=1)],
        state=GateState(),
        exclude_keys=set(),
        sf_score=sf_score,
        vet_lost_below=-0.80,
        max_vet_per_run=1,
        dry_run=False,
        stamp=_STAMP,
        out_path="/dev/null",
    )
    assert counts.vetted_kept == 1
    assert calls == [_key(_FEN_A)]              # curriculum, despite being older
    assert _key(_FEN_A) in st.emitted
    assert _key(_FEN_B) in {k for k, _ in st.pending}   # deferred, NOT dropped


def test_untagged_backlog_order_is_unchanged() -> None:
    """NEGATIVE CONTROL. With no sp= tags anywhere, this change must be a
    perfect no-op: every line ranks UNKNOWN and the stable sort must preserve
    newest-first exactly. If the sort ever reorders equal ranks, the existing
    backlog silently reshuffles on a pure format change."""
    calls: list[str] = []

    def sf_score(fen: str) -> float | None:
        calls.append(position_key(fen))
        return -0.99

    run_gate(
        new_lines=[_sp_line(f, sp=None) for f in (_FEN_A, _FEN_B, _FEN_C)],
        state=GateState(),
        exclude_keys=set(),
        sf_score=sf_score,
        vet_lost_below=-0.80,
        max_vet_per_run=3,
        dry_run=False,
        stamp=_STAMP,
        out_path="/dev/null",
    )
    # Newest-first over the untagged run, byte-for-byte the pre-change order.
    assert calls == [_key(_FEN_C), _key(_FEN_B), _key(_FEN_A)]


def test_tagged_curriculum_outranks_untagged_pending() -> None:
    """A fresh curriculum capture must outrank the untagged backlog, but an
    untagged line must still outrank a tagged SELFPLAY one — untagged sits in
    the middle rather than being lumped in with selfplay."""
    calls: list[str] = []

    def sf_score(fen: str) -> float | None:
        calls.append(position_key(fen))
        return -0.99

    # Order the INPUT so newest-first and rank order disagree: curriculum is
    # written OLDEST and selfplay NEWEST. Written the other way round, plain
    # newest-first already yields the expected sequence and the test passes
    # with the sort removed — i.e. it would assert nothing.
    run_gate(
        new_lines=[
            _sp_line(_FEN_C, sp=0),      # curriculum, OLDEST  -> must be first
            _sp_line(_FEN_B, sp=None),   # untagged            -> middle
            _sp_line(_FEN_A, sp=1),      # selfplay, NEWEST    -> must be last
        ],
        state=GateState(),
        exclude_keys=set(),
        sf_score=sf_score,
        vet_lost_below=-0.80,
        max_vet_per_run=3,
        dry_run=False,
        stamp=_STAMP,
        out_path="/dev/null",
    )
    assert calls == [_key(_FEN_C), _key(_FEN_B), _key(_FEN_A)]
