"""In-flight selfplay resume: byte-exact re-encode, and a gate that can fail.

The load-bearing claim is that a suspended game can be rebuilt from its move
list alone. That is only true because the replay reconstructs a REAL move
stack — the history planes are silently wrong for a board built from a bare
FEN or a stack-less copy, which has produced two retracted findings in this
repo. So the first test does not check that resume "works": it captures every
in-flight ply's encoded inputs from the live C play path, resumes, and demands
``np.array_equal`` on each one.

The second half is the negative control. A resume path that only ever succeeds
proves nothing, so the persisted state is corrupted three ways (a dropped move,
a truncated record column, shredded bytes) and each must be REJECTED with a
counted reason rather than silently producing a shorter or misaligned game.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, cast

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding.lc0 import LC0_HISTORY_LEGACY
from chess_anti_engine.selfplay import play_batch
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.manager import CompletedGameBatch
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.selfplay.resume import (
    RESUME_FILE_SUFFIX,
    RESUME_FORMAT_VERSION,
    resume_inflight_games,
    should_resume_game,
    suspend_inflight_games,
)
from chess_anti_engine.selfplay.state import SelfplayState
from chess_anti_engine.stockfish import StockfishUCI
from chess_anti_engine.utils.config_yaml import (
    flatten_run_config_defaults,
    load_yaml_file,
)
from tests.stockfish_binary import find_stockfish

SF_PATH = find_stockfish()
FINGERPRINT = "test-fingerprint"
TRIAL_ID = "trial-a"

_PROD = flatten_run_config_defaults(load_yaml_file("configs/pbt2_small.yaml"))


def _prod_game_kwargs() -> dict[str, Any]:
    """Encoding-relevant GameConfig fields, taken from the PRODUCTION yaml.

    Plane layout differs by input_history_encoding, so a byte-exactness claim
    made under a different encoding would not be a claim about production.
    Reading the live config (rather than hardcoding) means the test follows a
    production switch instead of quietly testing a dead encoding.
    """
    assert _PROD["input_history_encoding"] == "lc0_root_legacy_meta"
    assert _PROD["input_extra_features"] == "v2_threats"
    return {
        "input_history_encoding": str(_PROD["input_history_encoding"]),
        "input_extra_features": str(_PROD["input_extra_features"]),
        "record_lc0_root_input": bool(_PROD["record_lc0_root_input"]),
        "history_rep_fix": bool(_PROD["history_rep_fix"]),
        "record_relations": bool(_PROD.get("record_relations") or False),
    }


def _game_config(**overrides: Any) -> GameConfig:
    kwargs: dict[str, Any] = {
        "max_plies": 200,
        "selfplay_fraction": 0.0,  # curriculum: SF labels on every recorded ply
        "sf_policy_temp": 0.012,
        "sf_policy_label_smooth": 0.01,
        "sf_wdl_use_cp_logistic": True,
        "sf_wdl_cp_slope": 0.010,
        "sf_wdl_cp_draw_width": 60.0,
        **_prod_game_kwargs(),
    }
    kwargs.update(overrides)
    return GameConfig(**kwargs)


_SEARCH = SearchConfig(
    simulations=4, fast_simulations=2, mcts_type="gumbel", playout_cap_fraction=1.0,
)
_OPENING = OpeningConfig(random_start_plies=0)
_DIFF_FOCUS = DiffFocusConfig(min_keep=1.0)


def _tiny_model() -> Any:
    from chess_anti_engine.model import ModelConfig, build_model

    torch.manual_seed(0)
    return build_model(
        ModelConfig(embed_dim=32, num_layers=1, num_heads=2, use_smolgen=False),
    ).eval()


def _play_session(
    *,
    sf: StockfishUCI,
    game: GameConfig,
    games: int,
    max_steps: int,
    on_state_ready: Any = None,
    on_suspend: Any = None,
    on_game_complete: Any = None,
    seed: int = 0,
) -> SelfplayState:
    """Run a continuous session for ``max_steps`` loop steps; return its state."""
    captured: list[SelfplayState] = []

    def _ready(state: SelfplayState) -> None:
        captured.append(state)
        if on_state_ready is not None:
            on_state_ready(state)

    steps = {"n": 0}

    def _stop() -> bool:
        steps["n"] += 1
        return steps["n"] > max_steps

    play_batch(
        _tiny_model(), device="cpu", rng=np.random.default_rng(seed),
        stockfish=sf,
        games=games, target_games=0,
        stop_fn=_stop,
        on_state_ready=_ready,
        on_suspend=on_suspend,
        on_game_complete=on_game_complete,
        opponent=OpponentConfig(),
        temp=TemperatureConfig(temperature=1.0),
        search=_SEARCH,
        opening=_OPENING,
        diff_focus=_DIFF_FOCUS,
        game=game,
    )
    assert captured
    return captured[0]


def _inflight_records(state: SelfplayState) -> dict[tuple[int, ...], list[Any]]:
    """In-flight games keyed by their move list (stable across suspend/resume)."""
    out: dict[tuple[int, ...], list[Any]] = {}
    for i in range(state.batch_size):
        if state.finalized_arr[i] or state.done_arr[i]:
            continue
        if not state.samples_per_game[i]:
            continue
        out[tuple(int(m) for m in state.move_idx_history[i])] = list(
            state.samples_per_game[i]
        )
    return out


def _fresh_state(game: GameConfig, *, batch_size: int, seed: int = 5) -> SelfplayState:
    """A state shaped like the one play_batch builds, without playing.

    ``rep_fix.apply`` is process-global and normally called by play_batch;
    resume re-encodes through the same globals, so a direct state must set it
    too or the repetition planes would depend on test ordering.
    """
    rep_fix.apply(bool(game.history_rep_fix))
    return SelfplayState.create(
        model=None, device="cpu", rng=np.random.default_rng(seed),
        stockfish=cast("Any", _NullStockfish()),
        evaluator=cast("Any", _NullEvaluator()),
        batch_size=batch_size, continuous=True, target=batch_size,
        opponent=OpponentConfig(), temp=TemperatureConfig(), search=_SEARCH,
        opening=_OPENING, diff_focus=_DIFF_FOCUS, game=game,
    )


class _NullEvaluator:
    """Evaluator stand-in: resume never runs the net, only the encoders."""

    def evaluate_encoded(self, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = int(xs.shape[0])
        return np.zeros((n, 1858), np.float32), np.zeros((n, 3), np.float32)


class _NullStockfish:
    nodes = 1


def _bytes_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Exact-representation equality, including -0.0 vs +0.0 and NaN payloads."""
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and a.tobytes() == np.ascontiguousarray(b).tobytes()
    )


def _assert_record_identical(before: Any, after: Any, where: str) -> None:
    assert _bytes_equal(np.ascontiguousarray(before.x), after.x), f"{where}: x"
    for name in ("x_lc0_root", "relations", "legal_mask", "policy_probs",
                 "net_wdl_est", "search_wdl_est", "sf_policy_target", "sf_wdl",
                 "sf_wdl_original", "sf_multipv_raw", "sf_label_meta",
                 "sf_legal_mask"):
        want = getattr(before, name)
        got = getattr(after, name)
        if want is None:
            assert got is None, f"{where}: {name} appeared from nowhere"
            continue
        assert got is not None, f"{where}: {name} was lost"
        assert _bytes_equal(np.ascontiguousarray(want), got), f"{where}: {name}"
    for name in ("pov_color", "ply_index", "move_offset", "pos_hash", "has_policy",
                 "priority", "priority_policy_kl", "priority_q_delta",
                 "sample_weight", "keep_prob", "sf_move_index",
                 "sf_played_move_index", "sf_played_rank", "sf_played_regret",
                 "gumbel_policy_diag", "is_sf_refute_opp"):
        assert getattr(before, name) == getattr(after, name), f"{where}: {name}"


def _suspend_all(state: SelfplayState, out_dir: Path) -> int:
    report = suspend_inflight_games(
        state, out_dir=out_dir, compat_fingerprint=FINGERPRINT,
        model_sha="a" * 64, model_step=7, trial_id=TRIAL_ID,
    )
    return report.persisted


def _state_files(out_dir: Path) -> list[Path]:
    return sorted(out_dir.glob(f"*{RESUME_FILE_SUFFIX}"))


def _rewrite(path: Path, mutate: Any) -> None:
    """Load an npz state file, let ``mutate`` edit the dict, write it back."""
    with np.load(path, allow_pickle=False) as npz:
        arrays: dict[str, Any] = {k: np.array(npz[k]) for k in npz.files}
    mutate(arrays)
    with path.open("wb") as fh:
        np.savez(fh, **arrays)


def _meta_of(arrays: dict[str, Any]) -> dict[str, Any]:
    return json.loads(bytes(arrays["meta_json"]).decode("utf-8"))


def _set_meta(arrays: dict[str, Any], meta: dict[str, Any]) -> None:
    arrays["meta_json"] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8,
    )


# ── the byte-exactness proof ────────────────────────────────────────────────


@pytest.mark.skipif(SF_PATH is None, reason="Stockfish not found")
@pytest.mark.parametrize(
    "encoding_overrides",
    [
        pytest.param({}, id="production-encoding"),
        # The alternate-input columns are None under the production encoding
        # (lc0-root history already IS the root input), so a second arm is the
        # only way x_lc0_root and relations get exercised at all.
        pytest.param(
            {
                "input_history_encoding": LC0_HISTORY_LEGACY,
                "record_lc0_root_input": True,
                "record_relations": True,
            },
            id="legacy-encoding-with-alt-inputs",
        ),
    ],
)
def test_resume_reencodes_every_inflight_ply_byte_exactly(
    tmp_path: Path, encoding_overrides: dict[str, Any],
) -> None:
    """Replay + re-encode must reproduce the live C path's inputs bit for bit."""
    assert SF_PATH is not None
    game = _game_config(**encoding_overrides)
    out_dir = tmp_path / "resume"
    sf = StockfishUCI(SF_PATH, nodes=80, multipv=4)
    try:
        state = _play_session(sf=sf, game=game, games=3, max_steps=10)
        # The claim under test is about the PRODUCTION C path
        # (batch_process_ply). Without the extension the records would come
        # from the Python fallback, which encodes through the same helper the
        # resume uses — the comparison would degrade toward self-comparison
        # while still passing.
        assert state.has_c_ply, "byte-exactness must be proven against the C path"
        before = _inflight_records(state)
        assert before, "no in-flight games to resume"
        assert sum(len(recs) for recs in before.values()) >= 3
        assert any(
            rec.sf_policy_target is not None
            for recs in before.values() for rec in recs
        ), "no SF labels captured — the expensive fields would go untested"

        assert _suspend_all(state, out_dir) == len(before)

        target = _fresh_state(game, batch_size=state.batch_size)
        report = resume_inflight_games(
            target, in_dir=out_dir, compat_fingerprint=FINGERPRINT,
            trial_id=TRIAL_ID,
        )
    finally:
        sf.close()

    assert report.discarded == 0
    assert report.resumed == len(before)
    after = _inflight_records(target)
    assert set(after) == set(before)
    for moves, want in before.items():
        got = after[moves]
        assert len(got) == len(want)
        for k, (rec_before, rec_after) in enumerate(zip(want, got, strict=True)):
            _assert_record_identical(rec_before, rec_after, f"game {moves[:4]} ply {k}")
    # ...and the planes are not trivially equal because they are all zero.
    any_x = next(iter(after.values()))[0].x
    assert np.count_nonzero(any_x) > 0
    assert any_x.shape[0] == (175 if game.input_extra_features == "v2_threats" else 146)


@pytest.mark.skipif(SF_PATH is None, reason="Stockfish not found")
def test_resume_restores_slot_state_and_counts_completed_games(
    tmp_path: Path,
) -> None:
    """End-to-end through the play_batch hooks: suspended games come back, keep
    playing, and land in outcome_stats as `resumed_inflight_games` when they
    finish — the number that proves resumption reached a shard."""
    assert SF_PATH is not None
    out_dir = tmp_path / "resume"
    sf = StockfishUCI(SF_PATH, nodes=80, multipv=4)
    completed: list[CompletedGameBatch] = []
    try:
        first = _play_session(
            sf=sf, game=_game_config(max_plies=200), games=2, max_steps=8,
            on_suspend=lambda st: _suspend_all(st, out_dir),
        )
        inflight = _inflight_records(first)
        assert inflight
        played = max(len(m) for m in inflight)
        at_resume: list[dict[str, Any]] = []

        def _resume(state: SelfplayState) -> None:
            report = resume_inflight_games(
                state, in_dir=out_dir, compat_fingerprint=FINGERPRINT,
                trial_id=TRIAL_ID,
            )
            # Snapshot NOW: a resumed game that finishes gets its slot
            # recycled, which clears the per-slot flag by design.
            at_resume.append({
                "resumed": int(report.resumed),
                "flags": list(state.resumed_from_disk),
                "plies": [
                    len(state.move_idx_history[i]) for i in range(state.batch_size)
                ],
                "records": [
                    len(state.samples_per_game[i]) for i in range(state.batch_size)
                ],
                "not_done": [
                    not state.done_arr[i] and not state.finalized_arr[i]
                    for i in range(state.batch_size)
                ],
            })

        # A ply budget just past what the resumed games have already played, so
        # they finish quickly instead of running a full game in a unit test.
        _play_session(
            sf=sf, game=_game_config(max_plies=played + 2), games=2, max_steps=60,
            on_state_ready=_resume, on_game_complete=completed.append, seed=1,
        )
    finally:
        sf.close()

    snap = at_resume[0]
    assert snap["resumed"] == len(inflight)
    assert any(snap["flags"]), "no slot was marked as resumed"
    for i, was_resumed in enumerate(snap["flags"]):
        if not was_resumed:
            continue
        # A resumed slot arrives mid-game: moves already played, records
        # already carried, and not marked done/finalized.
        assert snap["plies"][i] > 0
        assert snap["records"][i] > 0
        assert snap["not_done"][i]
    resumed_completions = sum(
        int(cg.outcome_stats.get("resumed_inflight_games", 0)) for cg in completed
    )
    assert resumed_completions >= 1, (
        "a resumed game finished but outcome_stats never said so"
    )
    assert resumed_completions <= len(completed)
    # Files are consumed, not replayed forever.
    assert not _state_files(out_dir)


# ── negative controls: the gate must be able to fail ────────────────────────


def _one_suspended_game(tmp_path: Path) -> tuple[Path, GameConfig, SelfplayState]:
    """A single hand-built in-flight game persisted to disk.

    Built without Stockfish so the corruption tests stay fast and deterministic:
    what they exercise is the LOADER's validation, which never sees an engine.
    """
    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.moves import POLICY_SIZE, index_to_move
    from chess_anti_engine.selfplay.state import _NetRecord

    game = _game_config()
    state = _fresh_state(game, batch_size=2)
    board = chess.Board()
    cb = CBoard.from_board(board)
    moves: list[int] = []
    state.boards[0] = board.copy()
    assert state.starting_boards is not None
    state.starting_boards[0] = board.copy()
    state.samples_per_game[0] = []
    state.move_idx_history[0] = []
    for ply in range(6):
        legal = np.asarray(cb.legal_move_indices(), dtype=np.int64)
        probs = np.zeros(POLICY_SIZE, dtype=np.float32)
        probs[legal] = 1.0 / float(legal.size)
        if ply % 2 == 0:  # only "net" plies carry records
            rec = _NetRecord(
                x=np.zeros((1,), np.float32),  # replaced by the re-encode
                policy_probs=probs,
                net_wdl_est=np.array([0.2, 0.5, 0.3], np.float32),
                search_wdl_est=np.array([0.25, 0.5, 0.25], np.float32),
                pov_color=chess.WHITE if cb.turn else chess.BLACK,
                ply_index=int(cb.ply),
                has_policy=True,
                priority=1.0,
                sample_weight=1.0,
                keep_prob=1.0,
                legal_mask=_mask_of(cb),
                move_offset=len(state.move_idx_history[0]),
                pos_hash=int(cb.zobrist_hash),
            )
            state.samples_per_game[0].append(rec)
        action = int(legal[0])
        board.push(index_to_move(action, board))
        cb.push_index(action)
        state.move_idx_history[0].append(action)
        moves.append(action)
    state.cboards[0] = cb
    state.starting_ply_arr[0] = 0
    state.done_arr[1] = 1  # keep slot 1 out of the way

    out_dir = tmp_path / "resume"
    assert _suspend_all(state, out_dir) == 1
    return _state_files(out_dir)[0], game, state


def _mask_of(cb: Any) -> np.ndarray:
    from chess_anti_engine.moves import POLICY_SIZE

    mask = np.zeros(POLICY_SIZE, dtype=np.uint8)
    mask[np.asarray(cb.legal_move_indices(), dtype=np.int64)] = 1
    return mask


def _resume_one(path: Path, game: GameConfig) -> Any:
    target = _fresh_state(game, batch_size=2)
    report = resume_inflight_games(
        target, in_dir=path.parent, compat_fingerprint=FINGERPRINT,
        trial_id=TRIAL_ID,
    )
    return target, report


def test_negative_control_dropped_move_is_rejected(tmp_path: Path) -> None:
    """Drop one move from the persisted list. Every later record then belongs
    to a different position — but NOT a different colour (the side to move at a
    given offset is fixed by parity), and the shifted move indices can still be
    legal. This control is why the record carries a position hash: without it
    the game resumed silently with mislabeled planes.
    """
    path, game, _src = _one_suspended_game(tmp_path)

    def _drop(arrays: dict[str, np.ndarray]) -> None:
        meta = _meta_of(arrays)
        meta["moves"] = list(meta["moves"])[1:]
        _set_meta(arrays, meta)

    _rewrite(path, _drop)
    target, report = _resume_one(path, game)

    assert report.resumed == 0
    assert report.discarded == 1
    assert report.reasons.get("position_mismatch") == 1
    assert not any(target.samples_per_game[i] for i in range(target.batch_size))
    assert not any(target.resumed_from_disk)


def test_negative_control_dropped_trailing_move_is_rejected(tmp_path: Path) -> None:
    """Drop the move AFTER the last record. Every record still lands on its own
    position, so the per-record check cannot see it — the suspended position's
    hash is what catches a game that would resume one ply behind reality."""
    path, game, _src = _one_suspended_game(tmp_path)

    def _drop_last(arrays: dict[str, np.ndarray]) -> None:
        meta = _meta_of(arrays)
        meta["moves"] = list(meta["moves"])[:-1]
        _set_meta(arrays, meta)

    _rewrite(path, _drop_last)
    _target, report = _resume_one(path, game)

    assert report.resumed == 0
    assert report.reasons.get("final_position_mismatch") == 1


def test_negative_control_truncated_record_column_is_rejected(tmp_path: Path) -> None:
    """Drop the last row of ONE per-record column. The header still claims
    n_records, so a loader that trusted array lengths would silently build a
    game one ply short with every later field shifted."""
    path, game, _src = _one_suspended_game(tmp_path)

    def _truncate(arrays: dict[str, np.ndarray]) -> None:
        arrays["ply_index"] = arrays["ply_index"][:-1]

    _rewrite(path, _truncate)
    _target, report = _resume_one(path, game)

    assert report.resumed == 0
    assert report.reasons.get("record_count_mismatch") == 1


def test_negative_control_shredded_bytes_are_rejected(tmp_path: Path) -> None:
    """A half-written file (worker killed mid-suspend) must be unreadable, not
    partially decoded."""
    path, game, _src = _one_suspended_game(tmp_path)
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])

    _target, report = _resume_one(path, game)

    assert report.resumed == 0
    assert report.discarded == 1


def test_config_mismatch_and_version_mismatch_are_discarded(tmp_path: Path) -> None:
    """The single decision point rejects a game recorded under different
    record-shaping config, or a different on-disk format."""
    path, game, _src = _one_suspended_game(tmp_path)
    target = _fresh_state(game, batch_size=2)
    report = resume_inflight_games(
        target, in_dir=path.parent, compat_fingerprint="a-different-fingerprint",
        trial_id=TRIAL_ID,
    )
    assert report.resumed == 0
    assert report.reasons.get("config_mismatch") == 1

    assert should_resume_game(
        {"format_version": RESUME_FORMAT_VERSION + 1,
         "compat_fingerprint": FINGERPRINT},
        compat_fingerprint=FINGERPRINT,
    ) == "version_mismatch"
    assert should_resume_game(
        {"format_version": RESUME_FORMAT_VERSION, "compat_fingerprint": "other"},
        compat_fingerprint=FINGERPRINT,
    ) == "config_mismatch"
    assert should_resume_game(
        {"format_version": RESUME_FORMAT_VERSION, "compat_fingerprint": FINGERPRINT,
         "trial_id": "other"},
        compat_fingerprint=FINGERPRINT, trial_id=TRIAL_ID,
    ) == "trial_mismatch"
    ok = {
        "format_version": RESUME_FORMAT_VERSION,
        "compat_fingerprint": FINGERPRINT,
        "trial_id": TRIAL_ID,
        "root_fen": chess.STARTING_FEN,
        "moves": [], "opening_moves": [],
        "final_pos_hash": 12345,
        "saved_at": time.time(),
    }
    assert should_resume_game(
        ok, compat_fingerprint=FINGERPRINT, trial_id=TRIAL_ID,
    ) is None
    assert should_resume_game(
        {**ok, "final_pos_hash": 0}, compat_fingerprint=FINGERPRINT,
        trial_id=TRIAL_ID,
    ) == "malformed_meta"


def test_negative_control_zeroed_tail_hash_is_rejected(tmp_path: Path) -> None:
    """The tail guard must not be switchable off by the file it guards. Blanking
    `final_pos_hash` is rejected outright, so the dropped-trailing-move control
    above cannot be defeated by also clearing the field."""
    path, game, _src = _one_suspended_game(tmp_path)

    def _blank(arrays: dict[str, Any]) -> None:
        meta = _meta_of(arrays)
        meta["final_pos_hash"] = 0
        meta["moves"] = list(meta["moves"])[:-1]
        _set_meta(arrays, meta)

    _rewrite(path, _blank)
    _target, report = _resume_one(path, game)

    assert report.resumed == 0
    assert report.reasons.get("malformed_meta") == 1


def test_negative_control_tampered_ply_index_is_rejected(tmp_path: Path) -> None:
    """`ply_index` is not derivable from the moves, and finalize keys its
    sf_p0 one-ply shift off it — a wrong value shifts a teacher instead of
    failing. On the C play path it must equal the replayed CBoard ply."""
    path, game, _src = _one_suspended_game(tmp_path)

    def _shift(arrays: dict[str, Any]) -> None:
        arrays["ply_index"] = arrays["ply_index"] + 100

    _rewrite(path, _shift)
    _target, report = _resume_one(path, game)

    assert report.resumed == 0
    assert report.reasons.get("ply_index_mismatch") == 1


def test_a_game_from_another_trial_is_discarded(tmp_path: Path) -> None:
    """A trial reassignment also tears the session down. A game played under
    another trial's hyperparameters must not be finished and uploaded here."""
    path, game, _src = _one_suspended_game(tmp_path)
    target = _fresh_state(game, batch_size=2)
    report = resume_inflight_games(
        target, in_dir=path.parent, compat_fingerprint=FINGERPRINT,
        trial_id="some-other-trial",
    )
    assert report.resumed == 0
    assert report.reasons.get("trial_mismatch") == 1


def test_stale_games_are_discarded_and_swept(tmp_path: Path) -> None:
    """A game the fleet never came back for is from another era of the run —
    and nothing else on disk would ever delete it."""
    path, game, _src = _one_suspended_game(tmp_path)

    def _age(arrays: dict[str, Any]) -> None:
        meta = _meta_of(arrays)
        meta["saved_at"] = time.time() - 100_000  # ~27.8h, past the 6h default
        _set_meta(arrays, meta)

    _rewrite(path, _age)
    target = _fresh_state(game, batch_size=2)
    report = resume_inflight_games(
        target, in_dir=path.parent, compat_fingerprint=FINGERPRINT,
        trial_id=TRIAL_ID,
    )
    # Expiry runs through the SINGLE decision point, so it is counted with a
    # reason rather than silently swept.
    assert report.resumed == 0
    assert report.reasons.get("stale") == 1
    assert not _state_files(path.parent)

    # The sweep removes the debris a killed suspend/resume leaves behind.
    orphan_tmp = path.parent / f"x{RESUME_FILE_SUFFIX}.tmp"
    orphan_claimed = path.parent / f"y{RESUME_FILE_SUFFIX}.claimed"
    orphan_tmp.write_bytes(b"junk")
    orphan_claimed.write_bytes(b"junk")
    old = time.time() - 86_400
    os.utime(orphan_tmp, (old, old))
    os.utime(orphan_claimed, (old, old))
    _fresh = _fresh_state(game, batch_size=2)
    resume_inflight_games(
        _fresh, in_dir=path.parent, compat_fingerprint=FINGERPRINT,
        trial_id=TRIAL_ID, max_age_s=3600.0,
    )
    assert not orphan_tmp.exists()
    assert not orphan_claimed.exists()


def test_resume_never_overwrites_a_seeded_slot(tmp_path: Path) -> None:
    """A fenlist slot has already CONSUMED a doled blind-spot seed at create
    time and the seed line cannot be pushed back, so resuming over it would
    destroy the seed silently — the dole's historical failure mode."""
    path, game, _src = _one_suspended_game(tmp_path)
    target = _fresh_state(game, batch_size=2)
    # Both slots look seeded => nothing may be overwritten.
    target.opening_source_arr[0] = "fenlist"
    target.opening_source_arr[1] = "fenlist_dole"
    report = resume_inflight_games(
        target, in_dir=path.parent, compat_fingerprint=FINGERPRINT,
        trial_id=TRIAL_ID,
    )
    assert (report.resumed, report.discarded) == (0, 0)
    assert not any(target.resumed_from_disk)
    # The state file is left for the next session rather than claimed and lost.
    assert _state_files(path.parent)

    # One free slot => exactly that slot is used, and the seeded one is intact.
    target2 = _fresh_state(game, batch_size=2)
    target2.opening_source_arr[0] = "fenlist"
    target2.opening_source_arr[1] = "book"
    report2 = resume_inflight_games(
        target2, in_dir=path.parent, compat_fingerprint=FINGERPRINT,
        trial_id=TRIAL_ID,
    )
    assert report2.resumed == 1
    assert target2.resumed_from_disk == [False, True]
    assert target2.opening_source_arr[0] == "fenlist"


def test_missing_state_directory_is_a_silent_no_op(tmp_path: Path) -> None:
    game = _game_config()
    target = _fresh_state(game, batch_size=2)
    report = resume_inflight_games(
        target, in_dir=tmp_path / "does-not-exist", compat_fingerprint=FINGERPRINT,
        trial_id=TRIAL_ID,
    )
    assert (report.resumed, report.discarded) == (0, 0)
    assert not any(target.resumed_from_disk)


def test_a_state_file_is_claimed_once(tmp_path: Path) -> None:
    """Two sessions (or two selfplay threads) must not both resume one game."""
    path, game, _src = _one_suspended_game(tmp_path)
    first_target, first = _resume_one(path, game)
    second_target, second = _resume_one(path, game)

    assert first.resumed == 1
    assert (second.resumed, second.discarded) == (0, 0)
    assert any(first_target.resumed_from_disk)
    assert not any(second_target.resumed_from_disk)


def test_empty_slots_are_not_persisted(tmp_path: Path) -> None:
    """A slot that has played nothing is indistinguishable from a fresh game;
    persisting it would churn files for no gain."""
    game = _game_config()
    state = _fresh_state(game, batch_size=3)
    out_dir = tmp_path / "resume"
    report = suspend_inflight_games(
        state, out_dir=out_dir, compat_fingerprint=FINGERPRINT,
        model_sha="", model_step=0,
    )
    assert report.persisted == 0
    assert report.reasons.get("empty_slot") == 3
    assert not _state_files(out_dir)


@pytest.mark.parametrize("untracked", ["move_offset", "pos_hash"])
def test_record_without_replay_bookkeeping_is_not_persisted(
    tmp_path: Path, untracked: str,
) -> None:
    """move_offset says WHERE a record's position is and pos_hash proves the
    replay found it. A record missing either takes its whole game out of the
    resume path — the alternative is a guessed position."""
    from chess_anti_engine.moves import POLICY_SIZE
    from chess_anti_engine.selfplay.state import _NetRecord

    tracked: dict[str, int] = {"move_offset": 0, "pos_hash": 12345}
    del tracked[untracked]  # left at its "not tracked" default
    game = _game_config()
    state = _fresh_state(game, batch_size=1)
    state.move_idx_history[0] = [0]
    state.samples_per_game[0] = [
        _NetRecord(
            x=np.zeros((1,), np.float32),
            policy_probs=np.zeros(POLICY_SIZE, np.float32),
            net_wdl_est=np.zeros(3, np.float32),
            search_wdl_est=np.zeros(3, np.float32),
            pov_color=chess.WHITE, ply_index=0, has_policy=True,
            priority=1.0, sample_weight=1.0, keep_prob=1.0,
            **tracked,
        ),
    ]
    out_dir = tmp_path / "resume"
    report = suspend_inflight_games(
        state, out_dir=out_dir, compat_fingerprint=FINGERPRINT,
        model_sha="", model_step=0,
    )
    assert report.persisted == 0
    assert report.reasons.get("serialize_failed") == 1


# ── worker wiring ───────────────────────────────────────────────────────────


def test_every_restart_key_is_classified_for_resume() -> None:
    """Completeness guard: a new restart key must be declared either as
    record-shaping (games recorded under the old value are discarded) or as
    exempt. Unclassified would silently mean "exempt", which is how a game ends
    up carrying two different record schemas into one shard."""
    from chess_anti_engine.worker import WorkerSession

    restart = set(WorkerSession._RECO_RESTART_KEYS)
    compat = set(WorkerSession._RESUME_COMPAT_KEYS)
    exempt = set(WorkerSession._RESUME_COMPAT_EXEMPT_KEYS)
    assert not compat & exempt, sorted(compat & exempt)
    assert not restart - (compat | exempt), sorted(restart - (compat | exempt))
    assert not (compat | exempt) - restart, sorted((compat | exempt) - restart)


def test_resume_flag_is_watched_and_defaults_off() -> None:
    """The key must exist in code BEFORE it can appear in a live yaml (the
    validator rejects the whole reload on an unknown key), and it must reach a
    worker rather than sit in the trial config."""
    from chess_anti_engine.model import ModelConfig
    from chess_anti_engine.tune.distributed_runtime import build_recommended_worker
    from chess_anti_engine.tune.trial_config import TrialConfig
    from chess_anti_engine.utils.config_yaml import SELFPLAY_CONFIG_KEYS
    from chess_anti_engine.worker import WorkerSession

    key = "selfplay_resume_inflight_games"
    assert key in SELFPLAY_CONFIG_KEYS
    assert key in WorkerSession._RECO_WATCH_KEYS
    assert getattr(TrialConfig(), key) is False
    flat = flatten_run_config_defaults({"selfplay": {key: True}})
    assert flat[key] is True
    assert getattr(TrialConfig.from_dict(flat), key) is True
    # ...and it is actually PUBLISHED: the manifest block is the only channel
    # to a worker, so a key missing here would leave the worker on its default
    # while the yaml claimed otherwise.
    published = build_recommended_worker(
        config=flat, model_cfg=ModelConfig(), sf_nodes=1000, mcts_simulations=32,
    )
    assert published[key] is True
    assert build_recommended_worker(
        config={}, model_cfg=ModelConfig(), sf_nodes=1000, mcts_simulations=32,
    )[key] is False


def test_compat_fingerprint_moves_only_with_record_shaping_keys() -> None:
    from chess_anti_engine.worker import WorkerSession

    session = object.__new__(WorkerSession)
    base: dict[str, Any] = {
        k: i for i, k in enumerate(WorkerSession._RECO_RESTART_KEYS)
    }
    fp = session._resume_compat_fingerprint(base)
    for key in WorkerSession._RESUME_COMPAT_EXEMPT_KEYS:
        moved = dict(base)
        moved[key] = "changed"
        assert session._resume_compat_fingerprint(moved) == fp, key
    for key in WorkerSession._RESUME_COMPAT_KEYS:
        moved = dict(base)
        moved[key] = "changed"
        assert session._resume_compat_fingerprint(moved) != fp, key
