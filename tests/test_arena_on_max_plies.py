"""--on-max-plies: what a strict-Syzygy arena does with a game that reaches the cap.

The play loops are the production code; the move chooser and the tablebase are
stubbed so that no game can end before the cap and none is ever TB-resolved.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import numpy as np
import pytest
import torch

import chess_anti_engine.selfplay.match as match_mod
import chess_anti_engine.tablebase as tb_mod
import scripts.arena_standard as arena
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.uci.model_loader import ARCH_SCHEMA_VERSION
from chess_anti_engine.utils.game_log import read_game_log

# Black to move after 1. e4: the stub plays the first legal move, which cannot
# mate or repeat inside a handful of plies.
_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
_CAP = 6
_FOOLS_MATE = ("f2f3", "e7e5", "g2g4", "d8h4")


class _FakeTablebase:
    def close(self) -> None:
        pass


class _FakeProbe:
    """What run_arena reads off the probe for its record; the stubbed search never probes."""

    n_wdl = n_dtz = probes = hits = 0


@pytest.fixture
def never_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_pick(_model: object, sub_boards: list[chess.Board], **_kw: object) -> list[int]:
        return [0] * len(sub_boards)

    def fake_apply(boards: list[chess.Board], idxs: list[int], _actions: list[int],
                   *, strict: bool) -> None:
        assert strict
        for i in idxs:
            b = boards[i]
            ply = len(b.move_stack)
            # From the standard start: Fool's mate (decided on ply 4); anything else: the first legal move.
            if b.root().fen() == chess.STARTING_FEN and ply < len(_FOOLS_MATE):
                b.push_uci(_FOOLS_MATE[ply])
            else:
                b.push(next(iter(b.legal_moves)))

    monkeypatch.setattr(match_mod, "pick_moves_for_boards", fake_pick)
    monkeypatch.setattr(match_mod, "apply_actions_to_boards", fake_apply)
    monkeypatch.setattr(tb_mod, "rule50_match_result", lambda *_a, **_k: None)


def _search() -> arena.SideSearch:
    return arena.SideSearch(shape="test", source="test", gumbel={}, vloss_weight=0,
                            target_batch=0)


def _play(rolling: bool, on_max_plies: str, games: list[dict[str, Any]],
          unresolved: list[dict[str, Any]], *, openings: list[chess.Board] | None = None,
          max_plies: int = _CAP) -> list[float]:
    def sink(**g: Any) -> None:
        games.append(g)

    def usink(**g: Any) -> None:
        unresolved.append(g)

    openings = openings or [chess.Board(_FEN), chess.Board(_FEN)]
    common: dict[str, Any] = {
        "device": "cpu", "rng": np.random.default_rng(7), "sims_candidate": 1, "sims_reference": 1,
        "max_plies": max_plies, "temperature": 0.1, "gumbel_add_noise": False,
        "search_candidate": _search(), "search_reference": _search(),
        "syzygy_tablebase": _FakeTablebase(), "tb_probe": object(),
        "pgn_sink": sink, "on_max_plies": on_max_plies, "unresolved_sink": usink,
    }
    if rolling:
        return arena.play_paired_games_matched_sims_rolling(None, None, openings, pool_size=4, **common)
    return arena.play_paired_games_matched_sims(None, None, openings, **common)


@pytest.mark.usefixtures("never_resolved")
@pytest.mark.parametrize("rolling", [True, False])
def test_abort_banks_the_unresolved_game_before_raising(rolling: bool) -> None:
    games: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    with pytest.raises(RuntimeError, match=r"refusing a fabricated draw \(--on-max-plies abort\)"):
        _play(rolling, "abort", games, unresolved)
    # All four games reach the cap together: EVERY one is banked, not just the first.
    assert len(unresolved) == 4, "every offending game must be banked, not lost with the process"
    assert sorted((g["pair_id"], g["half"]) for g in unresolved) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    for g in unresolved:
        assert len(g["moves"]) == _CAP
        assert g["start_fen"] == _FEN
        assert g["loop"] == ("rolling" if rolling else "chunked")
    assert not any(x["termination"] == "max_plies" for x in games), "abort never scores it"


@pytest.mark.usefixtures("never_resolved")
@pytest.mark.parametrize("rolling", [True, False])
def test_draw_scores_every_capped_game_half_and_writes_it_as_a_draw(rolling: bool) -> None:
    games: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    scores = _play(rolling, "draw", games, unresolved)
    assert scores == [1.0, 1.0]  # pair score = sum of two halves at 0.5
    assert len(games) == 4
    assert {g["termination"] for g in games} == {"max_plies"}
    assert {g["result"] for g in games} == {"1/2-1/2"}
    assert unresolved == []


# ---------------------------------------------------------------------------
# run_arena: fingerprint, PGN tag, result row count, sidecar
# ---------------------------------------------------------------------------

_CFG = ModelConfig(kind="transformer", embed_dim=32, num_layers=1, num_heads=2,
                   use_smolgen=False, policy_encoding="lc0_1858")


def _save(path: Path, seed: int) -> Path:
    torch.manual_seed(seed)
    torch.save({"model": build_model(_CFG).state_dict(),
                "arch": {"_schema_version": ARCH_SCHEMA_VERSION, **dataclasses.asdict(_CFG)}},
               str(path))
    return path


@pytest.fixture
def strict_tb_stubbed(monkeypatch: pytest.MonkeyPatch, never_resolved: None) -> None:
    del never_resolved
    monkeypatch.setattr(tb_mod, "open_strict_match_tablebase", lambda *_a, **_k: _FakeTablebase())
    monkeypatch.setattr(tb_mod, "SyzygyProbe", lambda *_a, **_k: _FakeProbe())


def _run(tmp_path: Path, tag: str, on_max_plies: str, *, syzygy: bool = True) -> dict:
    fen = tmp_path / "openings.fen"
    fen.write_text(_FEN + "\n")
    side = arena.resolve_search_shape("training")
    return arena.run_arena(
        candidate=str(_save(tmp_path / "a.pt", 0)), reference=str(_save(tmp_path / "b.pt", 1)),
        games=2, openings_path=None, openings_fen=fen, opening_plies=16, mode="matched_sims",
        sims_candidate=2, sims_reference=2, ms_per_move=0, max_plies=_CAP,
        temperature=0.0, gumbel_add_noise=False, device="cpu", seed=3,
        out_path=tmp_path / f"{tag}.jsonl", game_log_path=tmp_path / f"{tag}.games.jsonl",
        pgn_out=tmp_path / f"{tag}.pgn", max_concurrent_games=4, compile_models=False,
        search_candidate=side, search_reference=side, syzygy_path=str(tmp_path / "tb") if syzygy else None,
        on_max_plies=on_max_plies,
    )


@pytest.mark.usefixtures("strict_tb_stubbed")
def test_draw_is_fingerprinted_tagged_and_counted(tmp_path: Path) -> None:
    rec = _run(tmp_path, "draw", "draw")
    assert rec["on_max_plies"] == "draw"
    assert rec["max_plies_unresolved"] == 2
    header = read_game_log(tmp_path / "draw.games.jsonl").header
    assert header["settings"]["on_max_plies"] == "draw"
    with (tmp_path / "draw.pgn").open() as fh:
        g = chess.pgn.read_game(fh)
    assert g is not None
    assert g.headers["OnMaxPlies"] == "draw"
    assert g.headers["Termination"] == "max_plies"
    assert not arena.unresolved_games_path(tmp_path / "draw.games.jsonl").exists()


@pytest.mark.usefixtures("strict_tb_stubbed")
def test_abort_keeps_the_default_fingerprint_and_writes_the_sidecar(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="refusing a fabricated draw"):
        _run(tmp_path, "abort", "abort")
    log = tmp_path / "abort.games.jsonl"
    # The default stays out of the settings: every pre-existing fingerprint is unchanged.
    assert "on_max_plies" not in read_game_log(log).header["settings"]
    rows = [json.loads(x) for x in arena.unresolved_games_path(log).read_text().splitlines()]
    assert len(rows) == 2          # both games of the one pair reach the cap
    r = rows[0]
    assert r["result"] == "*"
    assert r["plies"] == _CAP == len(r["moves_uci"]) == len(r["moves_san"])
    assert r["start_fen"] == _FEN
    board = chess.Board(_FEN)
    for u in r["moves_uci"]:
        board.push_uci(u)
    assert board.fen() == r["final_fen"]
    assert r["fingerprint"] == read_game_log(log).header["fingerprint"]


@pytest.mark.usefixtures("never_resolved")
@pytest.mark.parametrize("rolling", [True, False])
def test_a_game_decided_on_the_last_ply_is_written_before_the_abort(rolling: bool) -> None:
    """Review of #1462: in the chunked loop a game mated on the final ply was written only by the
    end-of-chunk sweep, which ran after the raise, so a resume re-drew a FINISHED game."""
    games: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    with pytest.raises(RuntimeError, match=r"2 unresolved game\(s\)"):
        _play(rolling, "abort", games, unresolved, openings=[chess.Board(), chess.Board(_FEN)], max_plies=4)
    mated = sorted((g["pair_id"], g["half"], g["result"]) for g in games)
    assert mated == [(0, 0, "0-1"), (0, 1, "0-1")]
    assert sorted((g["pair_id"], g["half"]) for g in unresolved) == [(1, 0), (1, 1)]


def test_the_max_plies_count_skips_a_row_it_cannot_key(tmp_path: Path) -> None:
    """Review of #1462: the count runs after every game is played and must never raise."""
    log = tmp_path / "x.games.jsonl"
    rows = [{"kind": "header", "fingerprint": "f", "settings": {}},
            {"kind": "game", "pair_id": 0, "half": 0, "termination": "max_plies"},
            {"kind": "game", "pair_id": 0, "half": 1, "termination": "rules"},
            {"kind": "game", "half": 0, "termination": "max_plies"}]          # no pair_id
    log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert arena.count_max_plies_games(log) == (1, 1)
    assert arena.count_max_plies_games(tmp_path / "missing.games.jsonl") == (None, 0)
    side = arena.unresolved_games_path(log)
    side.write_text(json.dumps({"kind": "unresolved_at_max_plies"}) + "\n" + '{"kind": "unres')  # torn tail
    assert arena.count_banked_aborts(side) == 1


@pytest.mark.usefixtures("strict_tb_stubbed")
def test_a_torn_sidecar_tail_is_repaired_before_the_next_abort_appends(tmp_path: Path) -> None:
    side = arena.unresolved_games_path(tmp_path / "torn.games.jsonl")
    side.write_text(json.dumps({"kind": "unresolved_at_max_plies", "pair_id": 9}) + "\n" + '{"kind": "unres')
    with pytest.raises(RuntimeError, match="refusing a fabricated draw"):
        _run(tmp_path, "torn", "abort")
    rows = [json.loads(x) for x in side.read_text().splitlines()]   # every line parses: nothing fused
    assert [r.get("pair_id") for r in rows] == [9, 0, 0]


@pytest.mark.usefixtures("never_resolved")
def test_draw_without_syzygy_is_refused_and_a_non_strict_row_says_what_it_applied(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="applies only with --syzygy"):
        _run(tmp_path, "nodraw", "draw", syzygy=False)
    rec = _run(tmp_path, "plain", "abort", syzygy=False)
    assert rec["on_max_plies"] == "draw_non_strict"
    assert rec["max_plies_unresolved"] == 2
    assert rec["max_plies_aborts_banked"] == 0
