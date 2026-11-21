"""``--<seat>-value-from`` / ``--<seat>-value-recal``: a seat whose search value is not its own net's.

Take-effect proofs, each against the arena's real loader, evaluator and C search
on CPU with tiny checkpoints:

* the identity hybrid (value from the seat's own checkpoint, T=1, b=0) plays
  the plain seat's games move for move, and its evaluator outputs are equal;
* a recalibration changes the evaluator's WDL by exactly ``/T + [0, b, 0]``;
* a value net with W and L swapped flips the evaluator's WDL, and with a
  material oracle the swapped value declines a free queen the plain one takes;
* the result row, the PGN tags, the game-log provenance and the fingerprint
  carry the value-from checkpoint's sha256 and [T, b]; a resume onto a
  changed value is refused.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import chess
import chess.pgn
import numpy as np
import pytest
import torch
from torch import nn

import scripts.arena_standard as arena
from chess_anti_engine.eval.hybrid_value import SHARED_ATTRS, HybridValueNet, build_seat_model
from chess_anti_engine.eval.seat_value import SeatValue, parse_recal
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.uci.model_loader import ARCH_SCHEMA_VERSION, load_model_from_checkpoint
from chess_anti_engine.utils.game_log import read_game_log

_CFG = ModelConfig(kind="transformer", embed_dim=32, num_layers=1, num_heads=2,
                   use_smolgen=False, policy_encoding="lc0_1858")
_FENS = (
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path: Path, seed: int, *, swap_wl: bool = False) -> Path:
    torch.manual_seed(seed)
    sd = build_model(_CFG).state_dict()
    if swap_wl:
        for key in ("value_wdl.net.2.weight", "value_wdl.net.2.bias"):
            sd[key] = sd[key][[2, 1, 0]].clone()
    torch.save({"model": sd, "arch": {"_schema_version": ARCH_SCHEMA_VERSION, **dataclasses.asdict(_CFG)}},
               str(path))
    return path


@pytest.fixture
def nets(tmp_path: Path) -> dict[str, Path]:
    return {"a": _save(tmp_path / "a.pt", 0), "b": _save(tmp_path / "b.pt", 1),
            "a_swap": _save(tmp_path / "a_swap.pt", 0, swap_wl=True)}


def _arena(tmp_path: Path, nets: dict[str, Path], tag: str, *, value_candidate: SeatValue | None = None,
           value_reference: SeatValue | None = None, resume: bool = False,
           game_log: Path | None = None, **kw: Any) -> dict:
    fen = tmp_path / "openings.fen"
    fen.write_text("\n".join(_FENS) + "\n")
    side = arena.resolve_search_shape("training")
    kw.setdefault("compile_models", False)
    return arena.run_arena(
        candidate=str(nets["a"]), reference=str(nets["b"]), games=4,
        openings_path=None, openings_fen=fen, opening_plies=16, mode="matched_sims",
        sims_candidate=4, sims_reference=4, ms_per_move=0, max_plies=24,
        temperature=0.0, gumbel_add_noise=False, device="cpu", seed=3,
        out_path=tmp_path / f"{tag}.jsonl", game_log_path=game_log or tmp_path / f"{tag}.games.jsonl",
        pgn_out=tmp_path / f"{tag}.pgn", max_concurrent_games=4,
        search_candidate=side, search_reference=side, resume=resume,
        value_candidate=value_candidate, value_reference=value_reference, **kw,
    )


def _played(tmp_path: Path, tag: str) -> dict[tuple[int, int], str]:
    log = read_game_log(tmp_path / f"{tag}.games.jsonl")
    return {(int(g["pair_id"]), int(g["half"])): g["played_uci_sha256"] for g in log.games}


def _pgn_games(path: Path) -> list[chess.pgn.Game]:
    out = []
    with path.open() as fh:
        while (g := chess.pgn.read_game(fh)) is not None:
            out.append(g)
    return out


# ---------------------------------------------------------------------------
# parsing and the module
# ---------------------------------------------------------------------------

def test_parse_recal_accepts_t_b_and_refuses_the_rest() -> None:
    assert parse_recal("0.5445,-0.6383") == (0.5445, -0.6383)
    assert parse_recal(" 1 , 0 ") == (1.0, 0.0)
    for bad in ("0.5", "0,1,2", "0,1", "-1,0", "nan,0", "1,inf", "a,b"):
        with pytest.raises(ValueError, match="value recalibration"):
            parse_recal(bad)


def test_a_seat_without_value_options_is_the_plain_net(nets: dict[str, Path]) -> None:
    plain = load_model_from_checkpoint(nets["a"])
    assert build_seat_model(plain, SeatValue(), device="cpu", load=load_model_from_checkpoint) is plain


def test_the_identity_hybrid_evaluates_exactly_like_the_plain_net(nets: dict[str, Path]) -> None:
    from chess_anti_engine.inference import LocalModelEvaluator

    plain = load_model_from_checkpoint(nets["a"])
    hybrid = build_seat_model(load_model_from_checkpoint(nets["a"]),
                              SeatValue(value_from=str(nets["a"]), recal=(1.0, 0.0)),
                              device="cpu", load=load_model_from_checkpoint)
    assert isinstance(hybrid, HybridValueNet)
    assert hybrid.value_net is not None
    x = np.random.default_rng(0).random((6, *_input_shape(plain)), dtype=np.float32)
    pol_p, wdl_p = LocalModelEvaluator(plain, device="cpu").evaluate_encoded(x)
    pol_h, wdl_h = LocalModelEvaluator(hybrid, device="cpu").evaluate_encoded(x)
    np.testing.assert_array_equal(pol_h, pol_p)
    np.testing.assert_array_equal(wdl_h, wdl_p)


def _input_shape(model: nn.Module) -> tuple[int, int, int]:
    from chess_anti_engine.encoding.model_inputs import encode_position_for_model

    return tuple(encode_position_for_model(model, chess.Board()).shape)


def test_the_recalibration_reaches_the_evaluator_exactly(nets: dict[str, Path]) -> None:
    from chess_anti_engine.inference import LocalModelEvaluator

    plain = load_model_from_checkpoint(nets["a"])
    t, b = 0.5445, -0.6383
    hybrid = build_seat_model(load_model_from_checkpoint(nets["a"]), SeatValue(recal=(t, b)),
                              device="cpu", load=load_model_from_checkpoint)
    assert isinstance(hybrid, HybridValueNet)
    assert hybrid.value_net is None   # recal alone: one forward
    x = np.random.default_rng(1).random((5, *_input_shape(plain)), dtype=np.float32)
    pol_p, wdl_p = LocalModelEvaluator(plain, device="cpu").evaluate_encoded(x)
    pol_h, wdl_h = LocalModelEvaluator(hybrid, device="cpu").evaluate_encoded(x)
    np.testing.assert_array_equal(pol_h, pol_p)
    np.testing.assert_allclose(wdl_h, wdl_p / t + np.array([0.0, b, 0.0], dtype=np.float32), rtol=1e-6, atol=1e-6)


def test_value_from_a_wl_swapped_checkpoint_flips_the_evaluators_wdl(nets: dict[str, Path]) -> None:
    from chess_anti_engine.inference import LocalModelEvaluator

    plain = load_model_from_checkpoint(nets["a"])
    hybrid = build_seat_model(load_model_from_checkpoint(nets["a"]), SeatValue(value_from=str(nets["a_swap"])),
                              device="cpu", load=load_model_from_checkpoint)
    x = np.random.default_rng(2).random((5, *_input_shape(plain)), dtype=np.float32)
    pol_p, wdl_p = LocalModelEvaluator(plain, device="cpu").evaluate_encoded(x)
    pol_h, wdl_h = LocalModelEvaluator(hybrid, device="cpu").evaluate_encoded(x)
    np.testing.assert_array_equal(pol_h, pol_p)          # the policy is still the seat's own
    np.testing.assert_allclose(wdl_h, wdl_p[:, [2, 1, 0]], rtol=1e-5, atol=1e-6)


class _MaterialOracle(nn.Module):
    """WDL logits from material: side to move's planes 0-5 minus the opponent's 6-11."""

    def __init__(self, like: nn.Module, *, swap_wl: bool) -> None:
        super().__init__()
        for attr in SHARED_ATTRS:
            if hasattr(like, attr):
                object.__setattr__(self, attr, getattr(like, attr))
        self.register_buffer("w", torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0]))
        self.swap_wl = swap_wl

    def forward(self, x: torch.Tensor, **_kw: Any) -> dict[str, torch.Tensor]:
        counts = x[:, :12].float().sum(dim=(2, 3))
        w = cast("torch.Tensor", self.w)
        diff = (counts[:, :6] * w).sum(-1) - (counts[:, 6:12] * w).sum(-1)
        wdl = torch.stack([diff, torch.zeros_like(diff), -diff], dim=-1)
        return {"wdl": wdl[:, [2, 1, 0]] if self.swap_wl else wdl}


def test_a_wl_swapped_value_declines_the_free_queen_the_plain_value_takes(nets: dict[str, Path]) -> None:
    """The search reads the hybrid's value: swap W and L and it plays the losing move."""
    from chess_anti_engine.moves import index_to_move
    from chess_anti_engine.selfplay.match import pick_moves_for_boards

    board = chess.Board("7k/8/8/q7/8/8/8/R6K w - - 0 1")   # Rxa5 wins the undefended queen
    side = arena.resolve_search_shape("training")
    picked = {}
    for swap in (False, True):
        policy = load_model_from_checkpoint(nets["a"])
        model = HybridValueNet(policy, _MaterialOracle(policy, swap_wl=swap))
        (idx,) = pick_moves_for_boards(
            model, [board.copy()], device="cpu", rng=np.random.default_rng(0),
            mcts_type="gumbel", mcts_simulations=64, temperature=0.0, gumbel_add_noise=False,
            gumbel_overrides=dict(side.gumbel), gumbel_vloss_weight=side.vloss_weight,
            gumbel_target_batch=side.target_batch,
        )
        picked[swap] = index_to_move(int(idx), board).uci()
    assert picked[False] == "a1a5"
    assert picked[True] != "a1a5"


# ---------------------------------------------------------------------------
# the arena, end to end
# ---------------------------------------------------------------------------

def test_the_identity_hybrid_plays_the_plain_seats_games(tmp_path: Path, nets: dict[str, Path]) -> None:
    plain = _arena(tmp_path, nets, "plain")
    hyb = _arena(tmp_path, nets, "hyb", value_candidate=SeatValue(value_from=str(nets["a"]), recal=(1.0, 0.0)))
    assert _played(tmp_path, "hyb") == _played(tmp_path, "plain")
    assert hyb["pentanomial"] == plain["pentanomial"]
    # ...and the value options are consumed: a different value net plays different games.
    _arena(tmp_path, nets, "hyb_b", value_candidate=SeatValue(value_from=str(nets["a_swap"])))
    assert _played(tmp_path, "hyb_b") != _played(tmp_path, "plain")


def test_a_hybrid_row_pgn_and_log_record_both_checkpoints_and_the_recal(
    tmp_path: Path, nets: dict[str, Path],
) -> None:
    recal = (0.5445, -0.6383)
    plain = _arena(tmp_path, nets, "plain")
    rec = _arena(tmp_path, nets, "hyb", value_candidate=SeatValue(value_from=str(nets["b"]), recal=recal),
                 value_reference=SeatValue(recal=(2.0, 0.25)))
    assert "candidate_value" not in plain
    assert "reference_value" not in plain
    assert rec["candidate_sha256"] == _sha(nets["a"])
    assert rec["candidate_value"]["value_from"]["sha256"] == _sha(nets["b"])
    assert rec["candidate_value"]["recal"] == list(recal)
    assert rec["reference_value"] == {"value_from": None, "recal": [2.0, 0.25]}
    assert rec["game_log_fingerprint"] != plain["game_log_fingerprint"]
    header = read_game_log(tmp_path / "hyb.games.jsonl").info["provenance"]
    assert header["candidate"]["value"]["value_from"]["sha256"] == _sha(nets["b"])
    assert "value" not in read_game_log(tmp_path / "plain.games.jsonl").info["provenance"]["candidate"]
    games = _pgn_games(tmp_path / "hyb.pgn")
    assert games
    for g in games:
        h = g.headers
        assert h["CandidateSHA256"] == _sha(nets["a"])
        assert h["CandidateValueFromSHA256"] == _sha(nets["b"])
        assert h["CandidateValueRecal"] == f"T={recal[0]!r},b={recal[1]!r}"
        assert h["ReferenceValueRecal"] == "T=2.0,b=0.25"
        assert "ReferenceValueFrom" not in h
    names = {g.headers["White"] for g in games} | {g.headers["Black"] for g in games}
    plain_names = {g.headers["White"] for g in _pgn_games(tmp_path / "plain.pgn")}
    assert not names & plain_names, "a hybrid seat must not share a PGN engine name with the plain seats"


def test_a_resume_onto_a_changed_value_is_refused(tmp_path: Path, nets: dict[str, Path]) -> None:
    log = tmp_path / "h.games.jsonl"
    seat = SeatValue(value_from=str(nets["b"]), recal=(0.5, -0.5))
    _arena(tmp_path, nets, "h", value_candidate=seat, game_log=log)
    again = _arena(tmp_path, nets, "h", value_candidate=seat, game_log=log, resume=True)
    assert again["resumed_pairs"] == 2
    with pytest.raises(SystemExit):     # [T, b] is in the fingerprint
        _arena(tmp_path, nets, "h", value_candidate=SeatValue(value_from=str(nets["b"]), recal=(0.5, -0.4)),
               game_log=log, resume=True)
    with pytest.raises(SystemExit):     # dropping the hybrid is a different run
        _arena(tmp_path, nets, "h", game_log=log, resume=True)
    _save(nets["b"], 7)                 # same path, other bytes: the provenance check refuses
    with pytest.raises(SystemExit, match="different bytes"):
        _arena(tmp_path, nets, "h", value_candidate=seat, game_log=log, resume=True)


@pytest.mark.parametrize(("kw", "match"), [
    ({"compile_models": True}, "--compile on is refused"),
    ({"volatility_candidate": {"q_scale": 0.1}}, "--volatility"),
])
def test_unsupported_combinations_are_refused(tmp_path: Path, nets: dict[str, Path], kw: dict, match: str) -> None:
    with pytest.raises(SystemExit, match=match):
        _arena(tmp_path, nets, "x", value_candidate=SeatValue(recal=(1.0, 0.0)), **kw)


def test_the_cli_parses_both_seats_and_refuses_a_bad_recal(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(arena, "run_arena", lambda **kw: seen.update(kw))
    base = ["arena_standard.py", "--candidate", "c.pt", "--reference", "r.pt", "--search-shape", "training",
            "--device", "cpu", "--games", "4"]
    monkeypatch.setattr("sys.argv", [*base, "--candidate-value-from", "v.pt", "--candidate-value-recal",
                                     "0.5,-0.25", "--reference-value-recal", "2,0"])
    arena.main()
    assert seen["value_candidate"] == SeatValue(value_from="v.pt", recal=(0.5, -0.25))
    assert seen["value_reference"] == SeatValue(recal=(2.0, 0.0))
    assert seen["compile_models"] is False      # auto never compiles a hybrid seat
    monkeypatch.setattr("sys.argv", [*base, "--candidate-value-recal", "0,1"])
    with pytest.raises(SystemExit, match="T must be finite and > 0"):
        arena.main()


def test_seat_value_records_are_json_and_stable() -> None:
    seat = SeatValue(value_from="v.pt", recal=(0.5, -0.25))
    assert json.loads(json.dumps(seat.settings_record())) == {"value_from": "v.pt", "recal": [0.5, -0.25]}
    assert not SeatValue().active


def test_a_candidate_only_hybrid_keys_its_own_game_log(tmp_path: Path, nets: dict[str, Path]) -> None:
    """The fingerprint binds the candidate seat's value settings on their own."""
    plain = _arena(tmp_path, nets, "p")
    hyb = _arena(tmp_path, nets, "h", value_candidate=SeatValue(recal=(1.0, 0.0)))
    other = _arena(tmp_path, nets, "o", value_candidate=SeatValue(recal=(1.0, 0.5)))
    assert len({plain["game_log_fingerprint"], hyb["game_log_fingerprint"], other["game_log_fingerprint"]}) == 3


def test_provenance_differences_name_a_changed_value_block() -> None:
    def prov(value: dict[str, Any] | None) -> dict[str, Any]:
        seat: dict[str, Any] = {"path": "c.pt", "sha256": "c" * 64, "files": []}
        if value is not None:
            seat["value"] = value
        return {"candidate": seat, "reference": {"path": "r.pt", "sha256": "r" * 64, "files": []},
                "openings_sha256": None}

    block = {"value_from": {"path": "v.pt", "sha256": "v" * 64, "files": []}, "recal": [0.5, -0.5]}
    assert arena.provenance_differences(prov(block), prov(block)) == []
    assert arena.provenance_differences(prov(block), prov({**block, "recal": [0.5, -0.4]}))
    assert arena.provenance_differences(
        prov(block), prov({**block, "value_from": {"path": "v.pt", "sha256": "w" * 64, "files": []}}))
    assert arena.provenance_differences(prov(block), prov(None))
    assert arena.provenance_differences(prov(None), prov(None)) == []


def test_an_explicit_pgn_name_on_a_hybrid_seat_still_gets_the_suffix(tmp_path: Path, nets: dict[str, Path]) -> None:
    """Review of #1457: an explicit --pgn-candidate-name must not let a hybrid share a plain arm's name."""
    _arena(tmp_path, nets, "named", value_candidate=SeatValue(recal=(0.5, -0.25)),
           pgn_candidate_name="armX", pgn_reference_name="base")
    names = {g.headers["White"] for g in _pgn_games(tmp_path / "named.pgn")}
    assert names == {"armX+recal(0.5,-0.25)", "base"}
