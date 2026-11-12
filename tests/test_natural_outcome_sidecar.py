import json
from types import SimpleNamespace as NS
import chess
import numpy as np
import pytest
from chess_anti_engine.natural_outcome import capture_natural_outcome
from chess_anti_engine.natural_outcome_sidecar import NaturalOutcomeSidecar


def board():
    b = chess.Board()
    for u in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        b.push_uci(u)
    return b


def receipt():
    return capture_natural_outcome(
        board(),
        resolved_result="0-1",
        adjudicated=False,
        tb_adjudicated=False,
        resigned=False,
        censored=False,
    )


def sample():
    return NS(
        x=np.zeros((175, 8, 8), dtype=np.float32),
        game_id=123,
        ply_index=0,
        wdl_target=2,
    )


def test_durable_crosswalk_and_exclusive_write(tmp_path):
    r = receipt()
    sink = NaturalOutcomeSidecar(tmp_path)
    s = sample()
    sink(r, [s], post_rescore_result="0-1", tb_rescore_enabled=True)
    raw = (tmp_path / (r.trace_sha256 + ".candidate.json")).read_bytes()
    v = json.loads(raw)
    assert v["natural"]["result"] == "0-1"
    assert v["rows"][0]["game_id"] == 123
    assert v["tb_rescore_enabled"] is True
    assert v["status"] == "CAPTURE_CANDIDATE_NOT_BANK_COMMITTED"
    assert v["stored_bank_labels_admitted"] is False
    assert v["input_planes_reencoded"] is False
    with pytest.raises(FileExistsError):
        sink(r, [s], post_rescore_result="0-1", tb_rescore_enabled=True)
    assert (tmp_path / (r.trace_sha256 + ".candidate.json")).read_bytes() == raw
    assert not list(tmp_path.glob(".pending-*"))


@pytest.mark.parametrize("fault", ["result", "pov", "label", "input", "duplicate"])
def test_reject_bad_crosswalk_without_publication(tmp_path, fault):
    s = sample()
    samples = [s]
    result = "0-1"
    if fault == "result":
        result = "1-0"
    if fault == "pov":
        s.x[108] = 1
    if fault == "label":
        s.wdl_target = 0
    if fault == "input":
        s.x[0, 0, 0] = float("nan")
    if fault == "duplicate":
        samples = [s, s]
    with pytest.raises(ValueError, match=r".+"):
        NaturalOutcomeSidecar(tmp_path)(
            receipt(), samples, post_rescore_result=result, tb_rescore_enabled=False
        )
    assert not list(tmp_path.iterdir())


def test_fsync_failure_propagates(tmp_path, monkeypatch):
    import os

    def fail(_fd):
        raise OSError("disk failed")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError, match="disk failed"):
        NaturalOutcomeSidecar(tmp_path)(
            receipt(), [sample()], post_rescore_result="0-1", tb_rescore_enabled=False
        )
    assert not list(tmp_path.iterdir())


def test_production_finalize_captures_before_rescore_and_emits_after_sink(monkeypatch):
    from chess_anti_engine.selfplay import finalize as f

    b = board()
    events = []
    all_samples = []
    stats = NS(
        total_game_plies=0,
        checkmate_games=0,
        stalemate_games=0,
        sf_d6_sum=0.0,
        sf_d6_n=0,
    )
    cb = NS(ply=4, is_checkmate=lambda: True, is_stalemate=lambda: False)

    def sink(r, _rows, **kwargs):
        events.append("sink")
        assert r.result == "0-1"
        assert kwargs["post_rescore_result"] == "1-0"
        assert not all_samples
        raise ValueError("fail closed")

    state = NS(
        cboards=[cb],
        replay_board=lambda i: b,
        stats=stats,
        samples_per_game=[[]],
        natural_outcome_sink=sink,
        game=NS(syzygy_path="tb"),
    )
    monkeypatch.setattr(
        f, "_resolve_terminal_result", lambda *a: ("0-1", False, False, None)
    )
    monkeypatch.setattr(f, "_log_terminal_anomalies", lambda *a, **kw: None)
    monkeypatch.setattr(f, "flush_async_sf_labels_for_records", lambda *a: None)

    def rescore(*_a):
        events.append("rescore")
        b.pop()
        return "1-0", {}

    monkeypatch.setattr(f, "_rescore_with_syzygy", rescore)
    monkeypatch.setattr(f, "_update_aggregate_stats", lambda *a, **kw: NS())
    monkeypatch.setattr(f, "_compute_volatility_and_sf_delta", lambda *a: ({}, {}))
    monkeypatch.setattr(f, "_build_replay_samples", lambda *a, **kw: [sample()])
    with pytest.raises(ValueError, match="fail closed"):
        f.finalize_game(state, 0, all_samples, None)
    assert events == ["rescore", "sink"]
    assert not all_samples


def test_categorical_uses_blended_scalar_not_histogram_mixture():
    from chess_anti_engine.train.targets import categorical_target_value, hlgauss_target

    search = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    q = categorical_target_value(
        1.0, None, blend_frac=0.0, search_wdl=search, search_blend_frac=0.8
    )
    assert q == pytest.approx(0.8 * (-0.3) + 0.2)
    actual = hlgauss_target(q, num_bins=32, sigma=0.04)
    wrong = 0.8 * hlgauss_target(-0.3, num_bins=32, sigma=0.04) + 0.2 * hlgauss_target(
        1.0, num_bins=32, sigma=0.04
    )
    assert not np.allclose(actual, wrong, atol=1e-3)


def test_directory_fsync_failure_leaves_only_unqualified_candidate(
    tmp_path, monkeypatch
):
    import os

    real_fsync = os.fsync
    calls = 0

    def fail_second(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory durability failed")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_second)
    r = receipt()
    with pytest.raises(OSError, match="directory durability failed"):
        NaturalOutcomeSidecar(tmp_path)(
            r, [sample()], post_rescore_result="0-1", tb_rescore_enabled=True
        )
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name.endswith(".candidate.json")
    value = json.loads(files[0].read_text())
    assert value["status"] == "CAPTURE_CANDIDATE_NOT_BANK_COMMITTED"
    assert value["stored_bank_labels_admitted"] is False
    assert not list(tmp_path.glob("*.committed*"))
