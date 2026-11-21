"""simple_rl sibling-ordering value term (``--w-sibling``) and its pair materialiser.

Covered:
- the parent-POV sign of a child's value, on a hand-built position;
- the loss weight is min(v_A, v_B) / 64, and pairs are drawn in proportion to it;
- hold-out pairs are never trained;
- w_sibling 0 attaches nothing and trains bit-identically;
- the term's gradient reaches the WDL head and the trunk, but not the policy head;
- on a tiny real collect, every materialised child input is byte-equal to an
  independent rebuild, and the uci and hash-bridge rebuilds agree;
- the CLI and settings refuse what the term cannot honour.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.simple_rl import collect as collect_mod
from chess_anti_engine.simple_rl import sibling as sib
from chess_anti_engine.simple_rl.collect import CollectSettings, collect_generation
from chess_anti_engine.simple_rl.learn import TrainSettings, build_trainer, train_generation
from tests.simple_rl_helpers import QuietStockfish, save_tiny_checkpoint, tiny_model_config

ENC = {"input_history_encoding": "lc0_root_legacy_meta", "input_extra_features": "v2_threats"}
PIECE_VALUES = torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0])


class MaterialNet(torch.nn.Module):
    """WDL logits (m, 0, -m) for the side to move, m = its material lead.
    Planes 0-5 are the side to move's pieces, 6-11 the opponent's."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        ours = (x[:, 0:6].sum((2, 3)) * PIECE_VALUES).sum(1)
        theirs = (x[:, 6:12].sum((2, 3)) * PIECE_VALUES).sum(1)
        m = (ours - theirs) * self.scale
        return {"wdl": torch.stack([m, torch.zeros_like(m), -m], dim=1)}


def _child(board: chess.Board, uci: str) -> torch.Tensor:
    from chess_anti_engine.encoding import rep_fix

    rep_fix.apply(True, boards_discarded=True)  # the test's boards are transient
    b = board.copy()
    b.push_uci(uci)
    return torch.from_numpy(sib.encode(b, ENC).astype(np.float32))[None]


def test_child_value_is_in_the_parents_pov_on_a_hand_built_position() -> None:
    # White to move can take a loose queen (d1xd8) or shuffle the king.
    parent = chess.Board("3qk3/8/8/8/8/8/8/3QK3 w - - 0 1")
    net = MaterialNet()
    xa, xb = _child(parent, "d1d8"), _child(parent, "e1f2")
    v_take = float(sib.child_values(net(xa)["wdl"]).detach()[0])
    v_quiet = float(sib.child_values(net(xb)["wdl"]).detach()[0])
    # After d1xd8+ Black (to move) is a queen down: the child's W < L, so the
    # PARENT's value of that child is positive and above the quiet move's (0).
    assert v_take > 0.5
    assert v_quiet == pytest.approx(0.0, abs=1e-6)
    # Search agrees (dq > 0): the residual dv - dq is small with the right sign
    # convention and large with the wrong one.
    dq = torch.tensor([v_take - v_quiet])
    _, resid = sib.sibling_loss(net, xa, xb, dq, torch.tensor([64.0]))
    assert abs(float(resid)) < 1e-6


def test_weight_is_min_visits_over_64() -> None:
    net = MaterialNet()
    parent = chess.Board("3qk3/8/8/8/8/8/8/3QK3 w - - 0 1")
    xa, xb = _child(parent, "d1d8"), _child(parent, "e1f2")
    xa, xb = xa.repeat(3, 1, 1, 1), xb.repeat(3, 1, 1, 1)
    dq = torch.zeros(3)
    mv = torch.tensor([2.0, 40.0, 64.0])  # min visits of pairs (2, 40), (40, 64), (64, 90)
    weighted, resid = sib.sibling_loss(net, xa, xb, dq, mv)
    huber = torch.nn.functional.huber_loss(resid, torch.zeros_like(resid), delta=0.2, reduction="none")
    np.testing.assert_allclose(weighted.detach().numpy(), (mv / 64.0 * huber).detach().numpy(), rtol=1e-6)
    assert float(weighted[0]) == pytest.approx(float(huber[0]) * 2 / 64)


def test_pairs_are_drawn_in_proportion_to_min_visits() -> None:
    visits = np.full(1858, -1.0, dtype=np.float32)
    visits[[10, 20, 30]] = [40.0, 40.0, 2.0]
    picks = [sib.choose_pairs(np.asarray([10, 20, 30]), visits, 1, sib.pair_rng(0, g, 0))[0] for g in range(3000)]
    share = sum(p == (10, 20) for p in picks) / len(picks)
    assert share == pytest.approx(40 / 44, abs=0.02)  # weights 40 / 2 / 2
    assert all(visits[a] >= visits[b] for a, b in picks)  # A = more visited
    again = [sib.choose_pairs(np.asarray([30, 20, 10]), visits, 1, sib.pair_rng(0, g, 0))[0] for g in range(50)]
    assert again == picks[:50]  # independent of the candidates' order
    assert len(sib.choose_pairs(np.asarray([10, 20, 30]), visits, 5, sib.pair_rng(0, 1, 0))) == 3


def test_gradient_reaches_the_wdl_head_and_trunk_not_the_policy_head() -> None:
    torch.manual_seed(0)
    trainer = build_trainer(tiny_model_config(), TrainSettings(device="cpu"), log_dir=None)
    model = trainer.model
    model.train()
    parent = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    xa = torch.cat([_child(parent, u) for u in ("f1b5", "d2d4", "b1c3")])
    xb = torch.cat([_child(parent, u) for u in ("a2a3", "h2h3", "g2g3")])
    weighted, _ = sib.sibling_loss(model, xa, xb, torch.tensor([0.3, -0.2, 0.5]), torch.tensor([8.0, 4.0, 2.0]))
    weighted.mean().backward()
    grads = {n: (p.grad.abs().sum().item() if p.grad is not None else 0.0) for n, p in model.named_parameters()}
    wdl = [n for n in grads if "wdl" in n or "value" in n]
    trunk = [n for n in grads if n.startswith(("embed", "input", "layers", "blocks", "encoder"))]
    policy = [n for n in grads if "policy" in n]
    assert wdl, sorted(grads)[:40]
    assert trunk, sorted(grads)[:40]
    assert policy, sorted(grads)[:40]
    assert sum(grads[n] for n in wdl) > 0
    assert sum(grads[n] for n in trunk) > 0
    assert sum(grads[n] for n in policy) == 0


def test_model_has_no_train_mode_dependent_layers() -> None:
    """The children go through the net in train mode; that equals eval mode here."""
    torch.manual_seed(0)
    trainer = build_trainer(tiny_model_config(), TrainSettings(device="cpu"), log_dir=None)
    model = trainer.model
    bad = [type(m).__name__ for m in model.modules()
           if isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout)) and getattr(m, "p", 1) != 0]
    assert not bad
    x = _child(chess.Board(), "e2e4").repeat(4, 1, 1, 1)
    model.train()
    with torch.no_grad():
        a = model(x)["wdl"]
    model.eval()
    with torch.no_grad():
        b = model(x)["wdl"]
    torch.testing.assert_close(a, b)


# ---------------------------------------------------------------------------
# A tiny real collect
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    tmp = tmp_path_factory.mktemp("sibling")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(collect_mod, "_open_stockfish", lambda _s: QuietStockfish([0.0, 1.0, 0.0]))
        ckpt = save_tiny_checkpoint(tmp / "model.pt", seed=5)
        openings = tmp / "train.jsonl"
        lines = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | e2e4 e7e5",
                 "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | d2d4 d7d5 c2c4",
                 "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | g1f3",
                 "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | c2c4 g8f6"]
        openings.write_text("".join(json.dumps({"id": f"o{i}", "book": 1, "line": ln}) + "\n"
                                    for i, ln in enumerate(lines)))
        settings = CollectSettings(games=8, concurrency=8, sims=32, topk=4, max_plies=40, seed=9, book_mix=(1.0,))
        run_dir = tmp / "run"
        collect = run_dir / "gen_0001" / "collect"
        collect_generation(gen_dir=collect, generation=1, checkpoint=ckpt, train_jsonl=openings, settings=settings)
    side = tmp / "siblings"
    sib.materialise(collect, side / "gen_0001", k=2, min_visits=2, seed=3, holdout_frac=0.25, log=lambda _m: None)
    return {"tmp": tmp, "run": run_dir, "collect": collect, "ckpt": ckpt, "side": side}


def test_self_check_is_byte_equal_on_every_game(run: dict[str, Path]) -> None:
    res = sib.self_check(run["collect"], games=1000, seed=0)
    assert res["ok"], res["mismatches"][:3]
    assert res["roots"] == res["roots_byte_equal"] > 50
    assert res["children"] == res["children_byte_equal"] > 50


def _independent_children(collect: Path, game_id: int, ply: int, actions: tuple[int, int]) -> list[np.ndarray]:
    """Rebuild a row's children from games.jsonl (uci) with python-chess only,
    encoded under the collect's declared encoding and repetition regime."""
    from chess_anti_engine.encoding import rep_fix
    from chess_anti_engine.moves.encode import index_to_move_for_encoding

    arch = json.loads((collect / "manifest.json").read_text())["config"]["arch"]
    enc = {"input_history_encoding": arch["input_history_encoding"],
           "input_extra_features": arch["input_extra_features"]}
    rep_fix.apply(bool(arch["history_rep_fix"]), boards_discarded=True)

    games = {json.loads(r)["game_id"]: json.loads(r) for r in (collect / "games.jsonl").read_text().splitlines()}
    g = games[game_id]
    line = next(json.loads(r)["line"] for r in (collect.parents[2] / "train.jsonl").read_text().splitlines()
                if json.loads(r)["id"] == g["opening_id"])
    fen, _, book = line.partition(" | ")
    b = chess.Board(fen.strip())
    for u in book.split():
        b.push_uci(u)
    for u in g["uci"][: ply - int(g["start_ply"])]:
        b.push_uci(u)
    out = []
    for a in actions:
        b.push(index_to_move_for_encoding(int(a), b))
        out.append(sib.encode(b, enc))
        b.pop()
    return out


def test_materialised_children_are_byte_equal_to_an_independent_rebuild(run: dict[str, Path]) -> None:
    man = json.loads((run["side"] / "gen_0001" / sib.SIDECAR_MANIFEST).read_text())
    assert man["complete"]
    assert man["totals"]["pairs"] > 20
    assert man["history_rep_fix"]["collect_declared"] == man["history_rep_fix"]["process_realised"]
    checked = 0
    for name in man["shards"]:
        d = run["side"] / "gen_0001" / name.replace(".zarr", "")
        z = {k: np.load(d / f"{k}.npy") for k in sib.PAIR_FIELDS}
        assert (z["visits_a"] >= 2).all()
        assert (z["visits_b"] >= 2).all()
        assert (z["visits_a"] >= z["visits_b"]).all()
        np.testing.assert_array_equal(z["dq"], z["q_a"] - z["q_b"])
        np.testing.assert_array_equal(z["min_visits"], np.minimum(z["visits_a"], z["visits_b"]))
        for j in range(len(z["dq"])):
            ca, cb = _independent_children(run["collect"], int(z["game_id"][j]), int(z["ply_index"][j]),
                                           (int(z["action_a"][j]), int(z["action_b"][j])))
            assert ca.tobytes() == z["x_a"][j].tobytes()
            assert cb.tobytes() == z["x_b"][j].tobytes()
            checked += 1
    assert checked == man["totals"]["pairs"]


def test_hash_bridge_rebuild_equals_the_uci_rebuild(run: dict[str, Path]) -> None:
    """Strip the per-game move list (a pre-#1444 collect): the bridged rebuild must
    produce byte-identical sidecar files."""
    old = run["tmp"] / "old_collect" / "gen_0001" / "collect"
    shutil.copytree(run["collect"], old)
    games = [json.loads(r) for r in (old / "games.jsonl").read_text().splitlines()]
    (old / "games.jsonl").write_text("".join(json.dumps({k: v for k, v in g.items() if k != "uci"}) + "\n"
                                             for g in games))
    out = run["tmp"] / "old_side"
    man = sib.materialise(old, out, k=2, min_visits=2, seed=3, holdout_frac=0.25, log=lambda _m: None)
    assert man["move_source"].startswith("played_move_index")
    ref = json.loads((run["side"] / "gen_0001" / sib.SIDECAR_MANIFEST).read_text())
    for name, rec in ref["shards"].items():
        assert man["shards"][name]["files"] == rec["files"], name


def test_an_unrebuildable_collect_is_refused_before_writing(run: dict[str, Path]) -> None:
    bad = run["tmp"] / "bad_collect" / "gen_0001" / "collect"
    shutil.copytree(run["collect"], bad)
    games = [json.loads(r) for r in (bad / "games.jsonl").read_text().splitlines()]
    g = games[0]
    g["uci"] = list(reversed(g["uci"]))  # a move list that does not replay the game
    (bad / "games.jsonl").write_text("".join(json.dumps(x) + "\n" for x in games))
    out = run["tmp"] / "bad_side"
    with pytest.raises(sib.CollectNotRebuildable):
        sib.materialise(bad, out, k=2, min_visits=2, seed=3, holdout_frac=0.25, log=lambda _m: None)
    assert not out.exists()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _settings(_run: dict[str, Path], **kw: Any) -> TrainSettings:
    base: dict[str, Any] = {"device": "cpu", "batch_size": 32, "holdout_frac": 0.25, "seed": 7, "lr": 1e-3,
                            "mirror_prob": 0.0, "probe_rows": 16}
    return TrainSettings(**{**base, **kw})


def _fresh_run(run: dict[str, Path], name: str) -> Path:
    d = run["tmp"] / name
    (d / "gen_0001").mkdir(parents=True)
    shutil.copytree(run["collect"], d / "gen_0001" / "collect", copy_function=shutil.copy2)
    return d


def test_holdout_pairs_are_never_trained_and_the_term_takes_effect(run: dict[str, Path],
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    used: list[np.ndarray] = []
    orig = sib.SiblingTerm._batch_pairs

    def spy(self: sib.SiblingTerm) -> np.ndarray:
        ids = orig(self)
        used.append(ids)
        assert not self.pairs.holdout[ids].any()
        return ids

    monkeypatch.setattr(sib.SiblingTerm, "_batch_pairs", spy)
    d = _fresh_run(run, "train_on")
    s = train_generation(run_dir=d, generation=1, init_checkpoint=run["ckpt"], prev_trainer_checkpoint=None,
                         settings=_settings(run, w_sibling=1.0, sibling_rows=8, sibling_dir=str(run["side"])))
    sm = s["sibling"]
    ids = np.concatenate(used)
    assert sm["pairs"] == ids.size > 0
    assert sm["steps"] == s["updates"]
    assert len(sm["per_step"]) == s["updates"]
    assert sm["probe_before"]["pairs"] > 0
    assert sm["probe_after"]["mean_abs_resid"] != sm["probe_before"]["mean_abs_resid"]
    manifest = json.loads((d / "gen_0001" / "train" / "manifest.json").read_text())
    assert manifest["config"]["settings"]["w_sibling"] == 1.0
    assert "sibling_gen_0001" in manifest["inputs"]
    assert manifest["summary"]["sibling"]["sidecars"]["generations"]["1"]["config"]["k"] == 2


def test_a_training_row_with_a_holdout_pair_is_refused(run: dict[str, Path]) -> None:
    from chess_anti_engine.simple_rl.collect import generation_shards, shard_row_meta
    from chess_anti_engine.simple_rl.learn import is_holdout_game

    meta = [shard_row_meta(s) for s in generation_shards(run["collect"])]
    gids = np.concatenate([m["game_id"] for m in meta])
    held_rows = np.flatnonzero([is_holdout_game(int(g), 0.25) for g in gids])
    assert held_rows.size
    row_ids = np.stack([np.ones(held_rows.size, np.int64), held_rows.astype(np.int64)], axis=1)
    with pytest.raises(ValueError, match="hold-out"):
        sib.load_sidecars(run["run"], [1], row_ids, run["side"], holdout_frac=0.25, newest=1)
    with pytest.raises(ValueError, match="hold-out split"):
        sib.load_sidecars(run["run"], [1], row_ids[:0], run["side"], holdout_frac=0.05, newest=1)


def test_w_sibling_zero_is_bit_identical_and_attaches_nothing(run: dict[str, Path]) -> None:
    a, b = _fresh_run(run, "off_a"), _fresh_run(run, "off_b")
    sa = train_generation(run_dir=a, generation=1, init_checkpoint=run["ckpt"], prev_trainer_checkpoint=None,
                          settings=_settings(run))
    assert sa["sibling"] is None
    sb = train_generation(run_dir=b, generation=1, init_checkpoint=run["ckpt"], prev_trainer_checkpoint=None,
                          settings=_settings(run, w_sibling=0.0))
    ma = torch.load(a / "gen_0001" / "train" / "model.pt", weights_only=False)["model"]
    mb = torch.load(b / "gen_0001" / "train" / "model.pt", weights_only=False)["model"]
    for k in ma:
        assert torch.equal(ma[k], mb[k]), k
    assert sa["losses"] == sb["losses"]
    manifest = json.loads((a / "gen_0001" / "train" / "manifest.json").read_text())
    assert not any(k.startswith("sibling_") for k in manifest["inputs"])


def test_settings_refuse_what_the_term_cannot_honour(run: dict[str, Path]) -> None:
    side = str(run["side"])
    for kw, match in (({"w_sibling": 1.0}, "sibling_dir"),
                      ({"w_sibling": -1.0}, ">= 0"),
                      ({"w_sibling": float("nan")}, "finite"),
                      ({"sibling_dir": side}, "ignored"),
                      ({"sibling_rows": 64}, "ignored"),
                      ({"w_sibling": 1.0, "sibling_dir": side, "sibling_rows": 8, "accum_steps": 2}, "accum_steps"),
                      ({"w_sibling": 1.0, "sibling_dir": side, "sibling_rows": 8, "sample_with_replacement": True},
                       "exact-once"),
                      ({"w_sibling": 1.0, "sibling_dir": side, "sibling_rows": 0}, "sibling_rows"),
                      ({"w_sibling": 1.0, "sibling_dir": side, "sibling_rows": 33}, "sibling_rows")):
        with pytest.raises(ValueError, match=match):
            _settings(run, **kw).validate()


def test_a_partial_or_foreign_sidecar_is_refused(run: dict[str, Path]) -> None:
    from chess_anti_engine.simple_rl.collect import generation_shards

    if len(generation_shards(run["collect"])) < 1:
        pytest.skip("no shards")
    part = run["tmp"] / "partial" / "gen_0001"
    sib.materialise(run["collect"], part, k=2, min_visits=2, seed=3, holdout_frac=0.25, shards=[0],
                    log=lambda _m: None)
    man = json.loads((part / sib.SIDECAR_MANIFEST).read_text())
    rows = np.zeros((0, 2), np.int64)
    if not man["complete"]:
        with pytest.raises(ValueError, match="partial"):
            sib.load_sidecars(run["run"], [1], rows, part.parent, holdout_frac=0.25, newest=1)
    tampered = run["tmp"] / "tampered"
    shutil.copytree(run["side"], tampered)
    first = next(p for p in (tampered / "gen_0001").iterdir() if p.is_dir())
    arr = np.load(first / "dq.npy")
    np.save(first / "dq.npy", arr + 1.0)
    with pytest.raises(ValueError, match="sha256"):
        sib.load_sidecars(run["run"], [1], rows, tampered, holdout_frac=0.25, newest=1)


def test_cli_flags_reach_the_settings_and_are_pinned_with_defaults(tmp_path: Path) -> None:
    from scripts import simple_rl_loop

    assert simple_rl_loop.ADDED_SINCE_PINNING["w_sibling"] == 0.0
    assert simple_rl_loop.ADDED_SINCE_PINNING["sibling_rows"] == 128
    assert simple_rl_loop.ADDED_SINCE_PINNING["sibling_dir"] is None
    common = ["run", "--run-dir", str(tmp_path / "r"), "--openings-dir", str(tmp_path / "o"), "--generations", "1"]
    off = simple_rl_loop.train_settings(simple_rl_loop.parser().parse_args(common), tmp_path / "r")
    assert (off.w_sibling, off.sibling_rows, off.sibling_dir) == (0.0, 128, None)
    on = simple_rl_loop.train_settings(simple_rl_loop.parser().parse_args(
        [*common, "--w-sibling", "0.5", "--sibling-rows", "64", "--sibling-dir", str(tmp_path / "s")]), tmp_path / "r")
    assert (on.w_sibling, on.sibling_rows, on.sibling_dir) == (0.5, 64, str(tmp_path / "s"))
    with pytest.raises(ValueError, match="sibling_dir"):
        simple_rl_loop.train_settings(simple_rl_loop.parser().parse_args([*common, "--w-sibling", "0.5"]),
                                      tmp_path / "r")


def test_materialiser_cli_self_check_exit_codes(run: dict[str, Path]) -> None:
    from scripts import simple_rl_sibling_pairs

    assert simple_rl_sibling_pairs.main(["--collect", str(run["collect"]), "--self-check", "3"]) == 0
    bad = run["tmp"] / "bad_cli" / "gen_0001" / "collect"
    shutil.copytree(run["collect"], bad)
    games = [json.loads(r) for r in (bad / "games.jsonl").read_text().splitlines()]
    for g in games:
        g["uci"] = g["uci"][1:] + g["uci"][:1]
    (bad / "games.jsonl").write_text("".join(json.dumps(x) + "\n" for x in games))
    assert simple_rl_sibling_pairs.main(["--collect", str(bad), "--self-check", "3"]) == 1
    assert simple_rl_sibling_pairs.main(["--collect", str(bad), "--out", str(run["tmp"] / "bad_cli_out")]) == 2


def test_snapshot_resume_restores_the_sibling_counters(run: dict[str, Path]) -> None:
    straight = _fresh_run(run, "snap_straight")
    s = _settings(run, w_sibling=1.0, sibling_rows=8, sibling_dir=str(run["side"]))
    whole = train_generation(run_dir=straight, generation=1, init_checkpoint=run["ckpt"], prev_trainer_checkpoint=None,
                             settings=s)
    if whole["updates"] < 2:
        pytest.skip("phase too short to snapshot")
    snap = _fresh_run(run, "snap_resumed")
    s2 = replace(s, snapshot_updates=(1,))
    done = train_generation(run_dir=snap, generation=1, init_checkpoint=run["ckpt"], prev_trainer_checkpoint=None,
                            settings=s2)
    assert done["sibling"]["pairs"] == whole["sibling"]["pairs"]
    assert done["sibling"]["per_step"] == whole["sibling"]["per_step"]


def test_a_bridge_with_two_matching_sequences_is_refused(run: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """The gap rule fills only when EXACTLY one move sequence reaches the next row's hash."""
    old = run["tmp"] / "ambiguous" / "gen_0001" / "collect"
    shutil.copytree(run["collect"], old)
    games = [json.loads(r) for r in (old / "games.jsonl").read_text().splitlines()]
    (old / "games.jsonl").write_text("".join(json.dumps({k: v for k, v in g.items() if k != "uci"}) + "\n"
                                             for g in games))
    real = sib._bridges

    def twice(board: chess.Board, depth: int, want: int, enc: dict[str, str]) -> list[list[chess.Move]]:
        found = real(board, depth, want, enc)
        return found + found  # every gap now has two matching sequences

    monkeypatch.setattr(sib, "_bridges", twice)
    with pytest.raises(sib.CollectNotRebuildable, match="exactly one"):
        sib.materialise(old, run["tmp"] / "ambiguous_side", k=2, min_visits=2, seed=3, holdout_frac=0.25,
                        log=lambda _m: None)
    assert not (run["tmp"] / "ambiguous_side").exists()
