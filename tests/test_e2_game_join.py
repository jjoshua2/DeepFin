from dataclasses import replace
from types import SimpleNamespace as NS
import json
import chess
import numpy as np
import pytest
from chess_anti_engine.moves.encode import move_to_index
from chess_anti_engine.selfplay.provenance import (
    GameStartProvenance,
    GameProvenance,
    NetRecordProvenance,
)
from chess_anti_engine.train.e2_game_join import BufferedGameJoin, replay_observation


def fixture(ordinal=0, gid=123, retained=True, terminal=True):
    b = chess.Board()
    actions = []
    for u in ["f2f3", "e7e5", "g2g4", "d8h4"] if terminal else ["e2e4"]:
        m = chess.Move.from_uci(u)
        actions.append(move_to_index(m, b))
        b.push(m)
    start = GameStartProvenance(
        ordinal,
        0,
        chess.STARTING_FEN,
        (),
        chess.STARTING_FEN,
        0,
        "standard",
        True,
        True,
    )
    result = b.result(claim_draw=False)
    p = GameProvenance(
        start,
        gid,
        tuple(actions),
        b.fen(en_passant="legal"),
        (NetRecordProvenance(0, 0, True, True, False),),
        result,
        result,
        result,
        "natural" if terminal else "sf_default_draw",
        None,
        not terminal,
        b.ply(),
        int(retained),
    )
    s = NS(x=np.zeros((175, 8, 8), np.float32), game_id=gid, ply_index=0, wdl_target=2)
    batch = NS(
        samples=[s],
        positions=1,
        input_history_encoding="lc0_root_legacy_meta",
        history_rep_fix=True,
    )
    return p, batch


def test_join_natural_snapshot_and_exclusion(tmp_path):
    out = []
    join = make_join(
        tmp_path, run_id="r", attempts=2, on_joined=lambda a, s: out.append((a, s))
    )
    p, b = fixture()
    join.on_game_complete(b)
    b.samples[0].wdl_target = 0
    join.on_game_provenance(p)
    p, _ = fixture(1, 456, retained=False, terminal=False)
    join.on_game_provenance(p)
    assert join.finish()["status"] == "ALL_ATTEMPTS_JOINED_NOT_BANK_COMMITTED"
    assert out[0][0].natural.result == "0-1"
    assert out[0][1][0].wdl_target == 2
    assert out[1][0].exclusion == "censored_or_adjudicated"
    assert out[1][1] == ()
    assert (
        json.loads((tmp_path / "r/0.observation.json").read_text())[
            "full175_replay_qualified"
        ]
        is False
    )


@pytest.mark.parametrize("kind", ["zero", "tb", "claimable"])
def test_ineligible_does_not_abort(tmp_path, kind):
    out = []
    j = make_join(
        tmp_path, run_id="r", attempts=1, on_joined=lambda a, s: out.append(a)
    )
    p, b = fixture(retained=kind != "zero")
    if kind == "tb":
        p = replace(p, tablebase_rescore_result="0-1")
    if kind == "claimable":
        p, b = fixture(terminal=False)
        p = replace(p, resolution="natural", cap_reached=False)
    if p.retained_samples:
        j.on_game_complete(b)
    j.on_game_provenance(p)
    j.finish()
    assert out[0].exclusion in {
        "zero_retained_rows",
        "tablebase_rescored",
        "not_automatic_terminal",
    }


@pytest.mark.parametrize(
    "kind", ["gid", "missing", "duplicate", "final", "action", "row"]
)
def test_bad_join_poisoned(tmp_path, kind):
    j = make_join(tmp_path, run_id="r", attempts=1, on_joined=lambda a, s: None)
    p, b = fixture()
    if kind == "gid":
        p = replace(p, game_id=999)
    if kind == "final":
        p = replace(p, final_fen=chess.STARTING_FEN)
    if kind == "action":
        p = replace(p, played_move_indices=(99999,))
    if kind == "row":
        p = replace(p, records=())
    if kind != "missing":
        j.on_game_complete(b)
    if kind == "duplicate":
        j.on_game_provenance(p)
    with pytest.raises(ValueError, match=r".+"):
        j.on_game_provenance(p)
    with pytest.raises(ValueError, match="previously failed"):
        j.finish()


def test_owner_callback_failure_no_bank_commit(tmp_path):
    def fail(*_):
        raise OSError("disk")

    j = make_join(tmp_path, run_id="r", attempts=1, on_joined=fail)
    p, b = fixture()
    j.on_game_complete(b)
    with pytest.raises(OSError, match="disk"):
        j.on_game_provenance(p)
    with pytest.raises(ValueError, match=r".+"):
        j.finish()
    assert not list(tmp_path.rglob("*BANK*"))
    assert json.loads((tmp_path / "r/0.observation.json").read_text())[
        "status"
    ].endswith("NOT_BANK_COMMITTED")


def test_incomplete_and_pending_fail(tmp_path):
    j = make_join(tmp_path, run_id="r", attempts=2, on_joined=lambda a, s: None)
    p, b = fixture()
    j.on_game_complete(b)
    j.on_game_provenance(p)
    with pytest.raises(ValueError, match=r".+"):
        j.finish()


def test_mutable_history_rejected():
    p, _ = fixture()
    with pytest.raises(ValueError, match=r".+"):
        replay_observation(replace(p, played_move_indices=list(p.played_move_indices)))


def test_owner_state_refuses_old_sink_or_continuous(tmp_path):
    j = make_join(tmp_path, run_id="r", attempts=1, on_joined=lambda a, s: None)
    with pytest.raises(ValueError, match="finite"):
        j.on_state_ready(NS(continuous=True, natural_outcome_sink=None))
    with pytest.raises(ValueError, match="previously failed"):
        j.finish()


def test_directory_sync_failure_has_no_observation(tmp_path, monkeypatch):
    import os

    j = make_join(tmp_path, run_id="r", attempts=1, on_joined=lambda a, s: None)
    p, _ = fixture(retained=False)
    original = os.fsync
    calls = 0

    def fail_second(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory sync")
        original(fd)

    monkeypatch.setattr(os, "fsync", fail_second)
    with pytest.raises(OSError, match="directory sync"):
        j.on_game_provenance(p)
    assert not list((tmp_path / "r").glob("*.observation.json"))
    with pytest.raises(ValueError, match="previously failed"):
        j.finish()


def make_join(directory, **kwargs):
    j = BufferedGameJoin(directory, **kwargs)
    j.on_state_ready(
        NS(
            continuous=False,
            natural_outcome_sink=None,
            game=NS(
                input_history_encoding="lc0_root_legacy_meta",
                history_rep_fix=True,
                syzygy_path="",
            ),
        )
    )
    return j


@pytest.mark.parametrize("fault", ["mode", "offset", "pov", "refute", "tb_nohit"])
def test_source_contract_gates(tmp_path, fault):
    out = []
    j = make_join(
        tmp_path, run_id="r", attempts=1, on_joined=lambda a, s: out.append(a)
    )
    p, b = fixture()
    if fault == "mode":
        b.input_history_encoding = "legacy"
        with pytest.raises(ValueError, match="encoding"):
            j.on_game_complete(b)
        return
    if fault == "offset":
        p = replace(p, records=(replace(p.records[0], move_offset=1),))
    if fault == "pov":
        p = replace(p, records=(replace(p.records[0], pov_color=False),))
    if fault == "refute":
        p = replace(p, records=(replace(p.records[0], is_sf_refute_opp=True),))
    if fault == "tb_nohit":
        j.on_state_ready(
            NS(
                continuous=False,
                natural_outcome_sink=None,
                game=NS(
                    input_history_encoding="lc0_root_legacy_meta",
                    history_rep_fix=True,
                    syzygy_path="/synthetic/tb",
                ),
            )
        )
    j.on_game_complete(b)
    if fault in ("offset", "pov"):
        with pytest.raises(ValueError, match="mismatch"):
            j.on_game_provenance(p)
        return
    j.on_game_provenance(p)
    if fault == "refute":
        assert out[0].exclusion == "sf_refute_actor"
    if fault == "tb_nohit":
        data = json.loads(next((tmp_path / "r/0").glob("*.candidate.json")).read_text())
        assert data["tb_rescore_enabled"] is True
        assert not data["stored_bank_labels_admitted"]


def test_real_producer_callbacks_capture_all_capped_attempts(tmp_path):
    from tests.test_game_provenance import _collect
    from chess_anti_engine.selfplay.config import GameConfig

    out = []
    j = BufferedGameJoin(
        tmp_path, run_id="producer", attempts=3, on_joined=lambda a, s: out.append(a)
    )
    _collect(
        game=GameConfig(
            max_plies=2,
            selfplay_fraction=1.0,
            input_history_encoding="lc0_root_legacy_meta",
            history_rep_fix=True,
        ),
        on_state_ready=j.on_state_ready,
        on_game_complete=j.on_game_complete,
        on_game_provenance=j.on_game_provenance,
    )
    assert j.finish()["attempts"] == 3
    assert all(a.exclusion == "censored_or_adjudicated" for a in out)
