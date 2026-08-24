"""Gated lc0-root repetition-plane fix (selfplay.history_rep_fix).

Default off must keep the C encoding byte-identical to before; on must make the
per-history-slot repetition planes bit-match python-chess, including deep
positions where an irreversible move cleared the C hash_stack.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding import _lc0_ext, encode_position, rep_fix
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.moves import move_to_index

PROD_MODES = ("lc0_root", "lc0_root_legacy_meta")

# A 180-ply game whose ply-174 position is a 3-fold repetition older than the
# C hash_stack window after a later pawn move — the reconstruction encoder
# misses it (plane 90), python-chess catches it.
_DEEP_REP_GAME = ["h2h3", "a7a5", "f2f4", "a5a4", "g2g3", "g7g5", "b2b4", "e7e6", "b4b5", "g5g4", "h3g4", "e8e7", "d2d3", "a8a6", "c1b2", "c7c5", "c2c4", "h7h5", "f1h3", "d8e8", "e1f2", "a6a8", "b2e5", "d7d5", "d1a4", "h5g4", "e5b8", "e8d7", "h3g4", "h8h6", "b1d2", "e7f6", "h1h3", "g8e7", "a1e1", "d7c7", "a4a7", "h6h8", "b5b6", "a8b8", "e1d1", "c7d7", "e2e3", "e7g8", "d1c1", "h8h3", "a7b8", "d7e7", "d2b3", "h3h4", "c1d1", "h4h3", "g4h5", "d5d4", "f2f3", "h3h5", "d1a1", "h5h4", "f3f2", "h4h2", "f2e1", "f8h6", "b8a7", "e6e5", "b3c5", "h2g2", "c5d7", "c8d7", "g1h3", "f6g6", "a1b1", "e7d8", "g3g4", "g8e7", "f4e5", "d8f8", "g4g5", "e7d5", "b1c1", "d5c3", "c1c2", "f8e8", "e5e6", "g2g5", "c2f2", "g5b5", "f2f6", "g6h7", "e6d7", "b5g5", "a7a3", "h6g7", "h3g1", "g5d5", "d7e8q", "d5d7", "e8e4", "c3e4", "f6f5", "h7g8", "g1e2", "g7f6", "f5f4", "e4g3", "a3f8", "g8h7", "f4f2", "g3f1", "f8d6", "d7d8", "d6f6", "f1h2", "f6g7", "h7g7", "f2f4", "d8a8", "e2g3", "f7f5", "a2a4", "a8a7", "f4f1", "g7f6", "f1g1", "f5f4", "g1f1", "f6g7", "g3h1", "a7a4", "e3f4", "a4a7", "f4f5", "h2f3", "f1f3", "a7a4", "f3h3", "a4b4", "h3h4", "b4c4", "d3c4", "g7f8", "h4h3", "f8f7", "h3a3", "f7g8", "a3a1", "g8g7", "h1f2", "g7f8", "c4c5", "f8e7", "f2d1", "e7d8", "e1f1", "d8e7", "d1b2", "e7f8", "b2c4", "f8e8", "c4e5", "e8d8", "a1a2", "d8c8", "a2e2", "c8d8", "e2e4", "d8c8", "e5g6", "c8d8", "g6e7", "d8d7", "e7d5", "d7c8", "d5e7", "c8d7", "e7g8", "d7d8", "c5c6", "d8c8", "e4e2", "c8b8"]


def _force_flag_off() -> None:
    """Put the module sentinel AND every C global back on the default (off).

    ⚑ ``rep_fix.apply(False, ...)`` on its own cannot do this, and that is not a
    theory: it returns early whenever the value it already holds matches, while
    several tests below poke ``_lc0_ext.set_history_rep_fix`` DIRECTLY. The
    extension global is write-only from Python — there is no getter — so after
    such a poke ``rep_fix.current()`` reports ``False`` while the encoder is
    measurably fix-ON, and ``apply(False)`` short-circuits instead of repairing
    it. MEASURED on the deep-repetition game: after
    ``apply(False)`` → ``set_history_rep_fix(True)`` → ``apply(False)`` the C
    encoding still matches python-chess exactly (fix-ON) with the sentinel
    reading ``False``. ``test_a_direct_setter_poke_is_repaired_by_this_fixture``
    pins the whole sequence.

    Clearing the sentinel first makes the call actually write to every loaded
    module, so this file cannot hand the next test FILE an encoder regime that
    ``rep_fix.current()`` — and therefore the suite-wide restore fixture in
    ``tests/conftest.py``, which can only see that sentinel — denies.
    """
    rep_fix._current = None
    rep_fix.apply(False, boards_discarded=True)


@pytest.fixture(autouse=True)
def _reset_flag():
    _force_flag_off()
    yield
    _force_flag_off()


def _build(moves: Sequence[str]) -> tuple[CBoard, chess.Board]:
    b = chess.Board()
    cb = CBoard.from_board(b)
    for u in moves:
        m = chess.Move.from_uci(u)
        cb.push_index(move_to_index(m, b))
        b.push(m)
    return cb, b


def test_default_off_unchanged_and_diverges_like_before():
    """Default off: the deep-repetition game still diverges from python-chess on
    the history-slot repetition plane (documents the bug the fix targets)."""
    _lc0_ext.set_history_rep_fix(False)
    cb, b = _build(_DEEP_REP_GAME)
    c = encode_cboard(cb, input_history_encoding="lc0_root_legacy_meta", input_extra_features="v1")
    p = encode_position(b, input_history_encoding="lc0_root_legacy_meta", input_extra_features="v1")
    diff = [pl for pl in range(c.shape[0]) if not np.array_equal(c[pl], p[pl])]
    assert diff == [90], f"expected the known plane-90 divergence, got {diff}"


def test_fix_on_matches_python_on_deep_repetition():
    _lc0_ext.set_history_rep_fix(True)
    cb, b = _build(_DEEP_REP_GAME)
    for mode in PROD_MODES:
        c = encode_cboard(cb, input_history_encoding=mode, input_extra_features="v1")
        p = encode_position(b, input_history_encoding=mode, input_extra_features="v1")
        assert np.array_equal(c, p), f"fix-on diverged for {mode!r}"


def test_fix_on_matches_python_over_random_games():
    import random

    # Before any boards exist: per-slot recording follows the ordering
    # contract (apply before construction/push), like the production paths.
    rep_fix.apply(True, boards_discarded=True)
    rng = random.Random(2024)
    for g in range(30):
        b = chess.Board()
        cb = CBoard.from_board(b)
        for ply in range(rng.randint(40, 160)):
            moves = list(b.legal_moves)
            if not moves:
                break
            m = rng.choice(moves)
            cb.push_index(move_to_index(m, b))
            b.push(m)
            if ply % 5:
                continue
            for mode in PROD_MODES:
                c = encode_cboard(cb, input_history_encoding=mode, input_extra_features="v1")
                p = encode_position(b, input_history_encoding=mode, input_extra_features="v1")
                assert np.array_equal(c, p), f"g{g} ply{ply} {mode}: fix-on diverged"


def test_exp_config_passes_yaml_selfplay_allowlist():
    """configs/exp_repetition_fix.yaml must flatten — history_rep_fix has to be
    in the selfplay: key allowlist or the advertised gated config can't run."""
    from pathlib import Path

    from chess_anti_engine.utils.config_yaml import (
        flatten_run_config_defaults,
        load_yaml_file,
    )

    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "exp_repetition_fix.yaml"
    flat = flatten_run_config_defaults(load_yaml_file(str(cfg_path)))
    assert flat["history_rep_fix"] is True


def test_model_config_persists_history_rep_fix():
    """The flag is model identity: it must survive the manifest round trip and
    be applied to the process-global encoder state when the model is built."""
    from chess_anti_engine.model import (
        ModelConfig,
        build_model,
        model_config_from_manifest_dict,
        model_config_to_manifest_dict,
    )

    cfg = ModelConfig(kind="tiny", history_rep_fix=True)
    md = model_config_to_manifest_dict(cfg)
    assert md["history_rep_fix"] is True
    assert model_config_from_manifest_dict(md).history_rep_fix is True
    # Absent key (pre-fix manifest) means off.
    assert model_config_from_manifest_dict({}).history_rep_fix is False

    model = build_model(cfg)
    assert model.history_rep_fix is True
    # build_model must have applied the flag to the C encoders, so a
    # rep-fix-trained checkpoint evaluates on the planes it was trained on.
    cb, b = _build(_DEEP_REP_GAME)
    c = encode_cboard(cb, input_history_encoding="lc0_root_legacy_meta", input_extra_features="v1")
    p = encode_position(b, input_history_encoding="lc0_root_legacy_meta", input_extra_features="v1")
    assert np.array_equal(c, p), "build_model did not apply history_rep_fix"


def test_shard_meta_and_upload_buffer_persist_history_rep_fix():
    """Shard identity: the flag is recorded in ShardMeta and mixed-flag game
    batches must not silently merge into one upload buffer."""
    from chess_anti_engine.replay.shard import ShardMeta
    from chess_anti_engine.selfplay.state import CompletedGameBatch
    from chess_anti_engine.worker_buffer import (
        _buffer_add_completed_game,
        _BufferedUpload,
    )

    assert ShardMeta(history_rep_fix=True).history_rep_fix is True
    # Pre-field shard dicts load with the flag unset (provably off).
    legacy_meta: dict[str, Any] = {"positions": 1}
    assert ShardMeta(**legacy_meta).history_rep_fix is None

    def batch(flag: bool) -> CompletedGameBatch:
        return CompletedGameBatch(
            samples=cast("list[Any]", [object()]),  # only len() is used
            input_history_encoding="lc0_root_legacy_meta",
            history_rep_fix=flag,
            positions=1,
        )

    buf = _BufferedUpload()
    _buffer_add_completed_game(
        buf=buf, game_batch=batch(True), now_s=0.0, model_sha="sha", model_step=1,
    )
    assert buf.history_rep_fix is True
    with pytest.raises(ValueError, match="metadata mismatch"):
        _buffer_add_completed_game(
            buf=buf, game_batch=batch(False), now_s=0.0, model_sha="sha", model_step=1,
        )


def test_server_accumulator_rejects_mixed_history_rep_fix():
    from chess_anti_engine.server.app import _BufferedUploadAccumulator

    acc = _BufferedUploadAccumulator(
        trial_id=None, model_sha256="sha", created_at_unix=0.0, last_update_unix=0.0,
    )
    meta = {"input_history_encoding": "lc0_root_legacy_meta", "history_rep_fix": True}
    acc.add_upload(samples=[], meta=meta, now_unix=0.0)
    assert acc.history_rep_fix is True
    with pytest.raises(ValueError, match="mixed history_rep_fix"):
        acc.add_upload(samples=[], meta={**meta, "history_rep_fix": False}, now_unix=0.0)


def _replay_sample(flag: bool):
    from chess_anti_engine.replay.buffer import ReplaySample

    return ReplaySample(
        x=np.zeros((146, 8, 8), dtype=np.float32),
        policy_target=np.full(4672, 1.0 / 4672, dtype=np.float32),
        wdl_target=1,
        input_history_encoding="lc0_root_legacy_meta",
        history_rep_fix=flag,
    )


def test_replay_arrays_carry_history_rep_fix(tmp_path):
    """Replay identity: the flag rides the chunk arrays, refuses to mix within a
    single sample batch, TOLERATES a cross-shard window mix (benign), and
    survives the shard save/load round trip."""
    from chess_anti_engine.replay.disk_buffer import _concat_sparse_batches
    from chess_anti_engine.replay.shard import (
        HISTORY_REP_FIX_ARRAY_KEY,
        ShardMeta,
        arrays_to_samples,
        load_shard_arrays,
        samples_to_arrays,
        save_local_shard_arrays,
    )

    on = samples_to_arrays([_replay_sample(True), _replay_sample(True)])
    off = samples_to_arrays([_replay_sample(False)])
    assert str(np.asarray(on[HISTORY_REP_FIX_ARRAY_KEY]).item()) == "true"
    assert str(np.asarray(off[HISTORY_REP_FIX_ARRAY_KEY]).item()) == "false"
    with pytest.raises(ValueError, match="mixed ReplaySample history_rep_fix"):
        samples_to_arrays([_replay_sample(True), _replay_sample(False)])

    # The cross-shard buffer merge TOLERATES a rep-fix mix (a window straddling
    # a rep-fix rollout would otherwise crash; the planes differ on ~0.2% of
    # positions and the change is net-neutral). It resolves toward "true".
    merged = _concat_sparse_batches([on, off])
    assert str(np.asarray(merged[HISTORY_REP_FIX_ARRAY_KEY]).item()) == "true"

    # Disk round trip: persisted as the shard attr, rematerialized on load,
    # restored onto samples.
    p = save_local_shard_arrays(
        tmp_path / "s.zarr", arrs=on,
        meta=ShardMeta(positions=2, history_rep_fix=True),
    )
    arrs, meta = load_shard_arrays(p)
    assert meta["history_rep_fix"] is True
    assert all(s.history_rep_fix for s in arrays_to_samples(arrs))
    # Legacy shard without the attr loads as off.
    p2 = save_local_shard_arrays(
        tmp_path / "s2.zarr", arrs=off, meta=ShardMeta(positions=1),
    )
    arrs2, _meta2 = load_shard_arrays(p2)
    assert not any(s.history_rep_fix for s in arrays_to_samples(arrs2))


def test_pick_moves_applies_model_rep_fix_flag(monkeypatch):
    """Same-process arenas alternate models, so each model's flag must be
    applied before its own moves are encoded — not just at load time."""
    from chess_anti_engine.model import ModelConfig, build_model
    from chess_anti_engine.selfplay import match as match_mod

    applied: list[bool] = []
    monkeypatch.setattr(
        match_mod.rep_fix, "apply",
        lambda v, **_kw: applied.append(bool(v)),
    )

    rng = np.random.default_rng(0)
    for flag in (True, False):
        model = build_model(ModelConfig(kind="tiny", history_rep_fix=flag)).eval()
        applied.clear()  # build_model also applies; isolate the search-time call
        match_mod.pick_moves_for_boards(
            model, [chess.Board()], device="cpu", rng=rng,
            mcts_type="gumbel", mcts_simulations=2, temperature=1.0,
            c_puct=2.5, gumbel_add_noise=False,
        )
        assert applied == [flag]


def test_from_board_with_history_sets_per_slot_flags():
    """from_board (opening with move history) must also produce fix-on parity,
    exercising the from_board per-slot population path rather than push."""
    rep_fix.apply(True, boards_discarded=True)
    # Build a board with a real repetition, then reconstruct via from_board so
    # the history comes from python's _stack, not from C pushes.
    b = chess.Board()
    for u in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]:
        b.push(chess.Move.from_uci(u))
    cb = CBoard.from_board(b)
    for mode in PROD_MODES:
        c = encode_cboard(cb, input_history_encoding=mode, input_extra_features="v1")
        p = encode_position(b, input_history_encoding=mode, input_extra_features="v1")
        assert np.array_equal(c, p), f"from_board fix-on diverged for {mode!r}"


# ---------------------------------------------------------------------------
# Ordering-contract guard (audit E3)
# ---------------------------------------------------------------------------

# The shuffle line from scratchpad/code_audit_20260803/enc_repfix_midgame_flip.py:
# it repeats positions inside the 8-slot history window after an irreversible
# move, so only the per-slot flags recorded at push time can see the repeats.
_MIDGAME_FLIP_LINE = [
    "e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "b1c3", "f8c5",
    "f3g5", "e8g8", "g5f3", "g8h8", "f3g5", "h8g8", "g5f3", "g8h8",
    "f3g5", "h8g8", "g5f3", "g8h8",
]


def test_a_direct_setter_poke_is_repaired_by_this_fixture() -> None:
    """The desync ``apply``'s idempotence cannot repair, and the repair.

    Read the ENCODER, not the flag: the two disagree in exactly the window this
    guards. Without the sentinel clear in ``_force_flag_off`` this file exits
    with the C encoders in fix-ON while every later test file — and the autouse
    restore fixture in ``tests/conftest.py``, which has only ``current()`` to go
    on — believes the flag is off.
    """
    def encoder_is_fix_on() -> bool:
        cb, b = _build(_DEEP_REP_GAME)
        c = encode_cboard(
            cb, input_history_encoding="lc0_root_legacy_meta", input_extra_features="v1",
        )
        p = encode_position(
            b, input_history_encoding="lc0_root_legacy_meta", input_extra_features="v1",
        )
        return bool(np.array_equal(c, p))

    _force_flag_off()
    assert not encoder_is_fix_on()

    _lc0_ext.set_history_rep_fix(True)  # what the tests above do
    assert encoder_is_fix_on()
    assert rep_fix.current() is False, "the sentinel already disagrees with the encoder"

    # What this fixture used to do at teardown, and why it was not enough.
    rep_fix.apply(False, boards_discarded=True)
    assert encoder_is_fix_on(), (
        "premise changed: apply() no longer short-circuits on an unchanged value, "
        "so the sentinel clear in _force_flag_off may now be redundant"
    )

    _force_flag_off()
    assert not encoder_is_fix_on()
    assert rep_fix.current() is False


def test_apply_is_idempotent_and_reports_current() -> None:
    rep_fix.apply(True, boards_discarded=True)
    assert rep_fix.current() is True
    rep_fix.apply(True)  # same value: not a flip, no keyword needed
    assert rep_fix.current() is True


def test_midgame_flip_with_live_board_raises() -> None:
    """The exact shape that produced a third, wrong repetition pattern.

    Audit E3 measured slots ``[1,1,1,1,1,1,1,0]`` under either clean regime and
    ``[1,0,1,0,1,0,1,0]`` when the flag alternated per ply on one live CBoard —
    half the repetition slots silently blanked. The guard must refuse the flip
    rather than encode that.
    """
    def play_flipping_the_flag_per_ply() -> None:
        b = chess.Board()
        cb = CBoard.from_board(b)
        for i, u in enumerate(_MIDGAME_FLIP_LINE):
            rep_fix.apply(i % 2 == 0)  # the per-move-cycle arena shape
            m = chess.Move.from_uci(u)
            cb.push_index(move_to_index(m, b))
            b.push(m)

    rep_fix.apply(True, boards_discarded=True)
    with pytest.raises(rep_fix.RepFixFlipError, match="boards_discarded"):
        play_flipping_the_flag_per_ply()
    # The flag is still the value every push so far was made under.
    assert rep_fix.current() is True


def test_certified_flip_produces_a_clean_regime() -> None:
    """A flip that really does discard its boards still yields a clean regime.

    Guards the guard: ``boards_discarded=True`` must not be a no-op keyword, and
    rebuilding the board after the flip must reproduce the fix-off encoding
    exactly (the arena/selfplay pattern).
    """
    def build() -> np.ndarray:
        b = chess.Board()
        cb = CBoard.from_board(b)
        for u in _MIDGAME_FLIP_LINE:
            m = chess.Move.from_uci(u)
            cb.push_index(move_to_index(m, b))
            b.push(m)
        return encode_cboard(
            cb, input_history_encoding="lc0_root_legacy_meta",
            input_extra_features="v2_threats",
        )

    rep_fix.apply(False, boards_discarded=True)
    off = build()
    rep_fix.apply(True, boards_discarded=True)
    on = build()
    rep_fix.apply(False, boards_discarded=True)
    off_again = build()
    assert np.array_equal(off, off_again)
    # Both clean regimes agree on THIS line (audit E3): the finding is that the
    # mid-game flip is a third answer, not that the two regimes differ here.
    assert np.array_equal(off, on)


def test_match_path_passes_no_cboards_so_its_flip_exemption_holds(monkeypatch):
    """Pin the PRECONDITION that makes ``boards_discarded=True`` true in match.py.

    Review F4. ``pick_moves_for_boards`` flips the process-global flag per move
    cycle, which is safe only because every C search entry point rebuilds its
    CBoards from the python board at the start of the call — i.e. because match
    passes no ``cboards=``. Nothing observed that. If a future long-lived-CBoard
    optimisation starts handing the search a board that outlives the flip, the
    keyword stays green and the repetition planes go silently wrong (the exact
    scenario the E3 guard exists for), so pin it here instead: this test goes red
    the moment match.py hands the search a pre-built board.

    Both search families are checked, because the exemption is claimed for both.
    """
    import chess

    from chess_anti_engine.selfplay import match as match_mod

    calls: list[dict] = []

    def _record(*args, **kwargs):
        calls.append(dict(kwargs))
        n = len(args[1])
        return ([np.zeros(1)] * n, [0] * n, [0.0] * n, [np.zeros(1)] * n)

    monkeypatch.setattr(match_mod, "_run_gumbel_root_many_c", _record)
    monkeypatch.setattr(match_mod, "_run_mcts_many_c", _record)
    monkeypatch.setattr(match_mod, "run_gumbel_root_many", _record)
    monkeypatch.setattr(match_mod, "run_mcts_many", _record)

    model = cast("Any", type("M", (), {
        "input_history_encoding": "lc0_root_legacy_meta",
        "input_extra_features": "v2_threats",
        "policy_encoding": "lc0_1858",
        "use_dynamic_relations": False,
        "history_rep_fix": True,
    })())
    boards = [chess.Board()]
    for mcts_type in ("gumbel", "puct"):
        calls.clear()
        match_mod.pick_moves_for_boards(
            model, boards, device="cpu", rng=np.random.default_rng(0),
            mcts_type=mcts_type, mcts_simulations=2, temperature=1.0,
            c_puct=2.5, gumbel_add_noise=False,
        )
        assert calls, f"no search invoked for {mcts_type}"
        assert "cboards" not in calls[0], (
            f"{mcts_type}: match.py now hands the search a pre-built CBoard, so "
            "the boards_discarded=True on its rep_fix.apply is no longer true — "
            "a board pushed under the other model's flag can now survive the "
            "flip and encode repetition planes matching neither regime (E3)"
        )
