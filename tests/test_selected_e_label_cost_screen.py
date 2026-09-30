"""Synthetic CPU contract tests; no ONNX session or registered E bank opens."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding.encode import encode_position
from scripts import selected_e_label_cost_screen as screen


class Meta:
    def __init__(self, name: str, width: int, dtype: str) -> None:
        self.name = name
        self.shape = [None, width]
        self.type = dtype


class BT4Session:
    def __init__(self, *, corrupt: bool = False, nonuniform: bool = False,
                 corrupt_extra: bool = False) -> None:
        self.corrupt = corrupt
        self.nonuniform = nonuniform
        self.corrupt_extra = corrupt_extra
        self.feeds: list[np.ndarray] = []

    def get_outputs(self) -> list[Meta]:
        return [Meta("policy", 1858, "tensor(float)"),
                Meta("wdl", 3, "tensor(float)")]

    def run(self, names: list[str], data: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert names == ["policy", "wdl"]
        feed = data["input"]
        assert feed.shape[1:] == (112, 8, 8)
        self.feeds.append(feed.copy())
        policy = np.zeros((len(feed), 1858), dtype=np.float32)
        if self.nonuniform:
            policy[:] = ((np.arange(1858) % 17) - 8).astype(np.float32) / 8
        elif self.corrupt:
            policy[:, 0] = 1
        wdl = np.tile(np.array([.55, .35, .10] if self.nonuniform else
                               [.25, .5, .25], dtype=np.float32), (len(feed), 1))
        if self.corrupt_extra and len(self.feeds) > 1:
            wdl[:] = np.array([.6, .3, .1], dtype=np.float32)
        return [policy, wdl]


class CeresSession:
    def __init__(self, *, corrupt: bool = False, nonuniform: bool = False) -> None:
        self.corrupt = corrupt
        self.nonuniform = nonuniform
        self.feeds: list[np.ndarray] = []

    def run(self, names: list[str], data: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert names == ["policy", "value", "value2"]
        feed = data["squares_byte"]
        assert feed.shape == (32, 64, 137)
        assert feed.dtype == np.uint8
        self.feeds.append(feed.copy())
        policy = np.zeros((32, 1858), dtype=np.float16)
        if self.nonuniform:
            policy[:] = (((np.arange(1858) % 23) - 11) / 8).astype(np.float16)
        primary = np.zeros((32, 3), dtype=np.float16)
        if self.nonuniform:
            primary[:] = np.array([1.25, -.5, -.75], dtype=np.float16)
        elif self.corrupt:
            primary[:, 0] = 1
        secondary = np.zeros((32, 3), dtype=np.float16)
        if self.nonuniform:
            secondary[:] = np.array([-.25, .75, -.5], dtype=np.float16)
        return [policy, primary, secondary]


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float64) / temperature
    mass = np.exp(scaled - scaled.max())
    return mass / mass.sum()


def _reference_targets(x: np.ndarray, legal: np.ndarray, *, nonuniform: bool
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray]:
    """Independent NumPy calibration, with producer legal maps only."""
    planes = screen.bt4_feed.stored_feed(x[None])
    board = screen.bt4_policy.board_from_stored_x(
        x, planes[0], input_history_encoding=screen.HISTORY)
    bt4_logits = (np.asarray(((np.arange(1858) % 17) - 8) / 8, dtype=np.float32)
                  if nonuniform else np.zeros(1858, dtype=np.float32))
    _, _, dense = screen.bt4_policy.compact_legal_policy(board, bt4_logits)
    base = dense.astype(np.float64)
    base /= base.sum()
    positive = (legal != 0) & (base > 0)
    scaled = np.log(base[positive]) / .5
    scaled = np.exp(scaled - scaled.max())
    stored_t05 = np.zeros(1858, dtype=np.float16)
    stored_t05[positive] = (scaled / scaled.sum()).astype(np.float16)
    b = stored_t05.astype(np.float64)
    b /= b.sum()
    selected_bt4 = b.astype(np.float16)

    feed = screen.tpg.stored_x_to_ceres_tpg_bytes(
        x[None], input_history_encoding=screen.HISTORY, history_rep_fix=True)
    gather = screen.mapping.leela_gather_indices(*screen.tpg.ceres_tpg_gather_context(feed))
    c_logits = ((((np.arange(1858) % 23) - 11) / 8).astype(np.float16)
                if nonuniform else np.zeros(1858, dtype=np.float16))
    legal_indices = np.flatnonzero(legal)
    c = np.zeros(1858, dtype=np.float64)
    c[legal_indices] = _softmax(c_logits[gather[0, legal_indices]], .5)
    selected_ceres = c.astype(np.float16)

    native = np.asarray([.55, .35, .10] if nonuniform else
                        [.25, .5, .25], dtype=np.float32).astype(np.float64)
    native /= native.sum()
    primary = np.asarray([1.25, -.5, -.75] if nonuniform else
                         [0, 0, 0], dtype=np.float16)
    secondary = np.asarray([-.25, .75, -.5] if nonuniform else
                           [0, 0, 0], dtype=np.float16)
    calibrated = .6 * _softmax(primary, .55) + .4 * _softmax(secondary, 1.5)
    dual_policy = (.5 * b + .5 * c).astype(np.float16)
    dual_wdl = (.5 * native + .5 * calibrated).astype(np.float16)
    return (selected_bt4, native.astype(np.float16), selected_ceres,
            calibrated.astype(np.float16), dual_policy, dual_wdl)


def rows(n: int = 6, *, nonuniform: bool = False) -> list[screen.Row]:
    result = []
    for offset in range(n):
        board = chess.Board()
        board.halfmove_clock = offset
        x = encode_position(board, input_history_encoding=screen.HISTORY,
                            input_extra_features="v2_threats").astype(np.float16)
        legal = screen._compact_legal_mask(board)
        route_key = (b"factorial58-E-teacher-row-v0\0" + bytes(32)
                     + (0).to_bytes(8, "big") + offset.to_bytes(8, "big")
                     + bytes.fromhex("11" * 32))
        teacher = "BT4" if sha256(route_key).digest()[0] & 1 == 0 else "Ceres"
        b_policy, b_wdl, c_policy, c_wdl, dual_policy, dual_wdl = _reference_targets(
            x, legal, nonuniform=nonuniform)
        policy, wdl = ((b_policy, b_wdl) if teacher == "BT4" else (c_policy, c_wdl))
        result.append(screen.Row(
            cohort_manifest_sha256="00" * 32, shard_ordinal=0,
            shard_rows=n, row_offset=offset,
            input_sha256=sha256(x.tobytes()).hexdigest(),
            legal_sha256=sha256(legal.tobytes()).hexdigest(),
            selected_teacher=teacher,
            policy_target_sha256=sha256(policy.tobytes()).hexdigest(),
            search_wdl_sha256=sha256(wdl.tobytes()).hexdigest(),
            dual_policy_target_sha256=sha256(dual_policy.tobytes()).hexdigest(),
            dual_search_wdl_sha256=sha256(dual_wdl.tobytes()).hexdigest(),
            x=x, legal=legal))
    return result


def bindings(*, corrupt_bt4: bool = False, corrupt_ceres: bool = False,
             nonuniform: bool = False, corrupt_bt4_extra: bool = False):
    b = BT4Session(corrupt=corrupt_bt4, nonuniform=nonuniform,
                   corrupt_extra=corrupt_bt4_extra)
    c = CeresSession(corrupt=corrupt_ceres, nonuniform=nonuniform)
    return (screen.BT4Session(b, "input", np.dtype("float32"), "policy", "wdl"),
            screen.CeresSession(c), b, c)


def test_matched_targets_rosters_and_repeat_last_padding() -> None:
    sample = rows()
    seed = "11" * 32
    bt4_ids, ceres_ids = screen.admitted_roster(sample, seed)
    s_bt4, s_ceres, b_s, c_s = bindings()
    d_bt4, d_ceres, b_d, c_d = bindings()
    receipt = screen.matched_cpu_kernel(
        sample, seed, s_bt4, s_ceres, d_bt4, d_ceres)
    assert receipt["qualification"] == "cpu-kernel-only-no-launch"
    assert receipt["selected_counts"] == {"BT4": len(bt4_ids),
                                           "Ceres": len(ceres_ids)}
    assert receipt["dual_policy_target"].shape == (len(sample), 1858)
    assert receipt["dual_search_wdl"].shape == (len(sample), 3)
    assert receipt["dual_policy_sha256"] == sha256(
        receipt["dual_policy_target"].tobytes()).hexdigest()
    assert len(b_s.feeds) == len(c_s.feeds) == 1
    np.testing.assert_array_equal(b_s.feeds[0], b_d.feeds[0])
    np.testing.assert_array_equal(c_s.feeds[0], c_d.feeds[0])
    np.testing.assert_array_equal(c_s.feeds[0][len(ceres_ids):],
        np.repeat(c_s.feeds[0][len(ceres_ids) - 1:len(ceres_ids)],
                  32 - len(ceres_ids), axis=0))
    assert all(call["physical_rows"] == 32 for call in receipt["selected_calls"]
               if call["teacher"] == "Ceres")


def test_multi_batch_selected_prefix_stays_separate_from_dual_extra() -> None:
    sample = rows(80)
    seed = "11" * 32
    bt4_ids, ceres_ids = screen.admitted_roster(sample, seed)
    assert len(bt4_ids) > 32
    assert len(ceres_ids) > 32
    s_bt4, s_ceres, b_s, c_s = bindings()
    d_bt4, d_ceres, b_d, c_d = bindings()
    receipt = screen.matched_cpu_kernel(
        sample, seed, s_bt4, s_ceres, d_bt4, d_ceres)
    assert receipt["rows"] == 80
    assert len(b_s.feeds) == len(c_s.feeds) == 2
    for chosen, dual in ((b_s, b_d), (c_s, c_d)):
        for index, feed in enumerate(chosen.feeds):
            np.testing.assert_array_equal(feed, dual.feeds[index])
        assert len(dual.feeds) > len(chosen.feeds)


@pytest.mark.parametrize("teacher", ["BT4", "Ceres"])
def test_changed_dual_native_head_fails_even_if_target_would_normalize(teacher: str) -> None:
    sample = rows()
    s_bt4, s_ceres, _, _ = bindings()
    d_bt4, d_ceres, _, _ = bindings(
        corrupt_bt4=teacher == "BT4", corrupt_ceres=teacher == "Ceres")
    with pytest.raises(ValueError, match="mismatch"):
        screen.matched_cpu_kernel(sample, "11" * 32,
                                  s_bt4, s_ceres, d_bt4, d_ceres)


def test_nonuniform_selected_and_original_e_blend_bytes() -> None:
    sample = rows(nonuniform=True)
    # Fixed independent NumPy-reference pins. The D pair also matched the
    # frozen bootstrap_sffree_targets.mixed_targets in a separate CPU process.
    assert sample[0].selected_teacher == "Ceres"
    assert sample[0].policy_target_sha256 == "d36b746d18eb5f9f6ea0615b90682646fd9dea00cbab0aa5409ed1e1be9ddefc"
    assert sample[0].search_wdl_sha256 == "a99f9d79228e301ddbd583b9fe0f8894ba793b3cf813518fa4cc54800665ced2"
    assert sample[1].selected_teacher == "BT4"
    assert sample[1].policy_target_sha256 == "b3521957a15da92efd554b2dee8aa9664e3985392fdf0a6164f8b966caefbfa9"
    assert sample[1].search_wdl_sha256 == "d997f5c9ed4c0b1759077bbaaa675e64548c6f20f6548175db411c149485d77a"
    assert sample[0].dual_policy_target_sha256 == "d453294787de04711f556c0dc374bc759a938d5ebd5ab2627c36db6a9d17e8b6"
    assert sample[0].dual_search_wdl_sha256 == "e8db75e8303f757b00cda8212a118d55753af60a74a50cdfe2e963a1764edebe"
    s_bt4, s_ceres, _, _ = bindings(nonuniform=True)
    d_bt4, d_ceres, _, _ = bindings(nonuniform=True)
    receipt = screen.matched_cpu_kernel(
        sample, "11" * 32, s_bt4, s_ceres, d_bt4, d_ceres)
    for index, row in enumerate(sample):
        assert sha256(receipt["selected_policy_target"][index].tobytes()).hexdigest() == row.policy_target_sha256
        assert sha256(receipt["selected_search_wdl"][index].tobytes()).hexdigest() == row.search_wdl_sha256
        assert sha256(receipt["dual_policy_target"][index].tobytes()).hexdigest() == row.dual_policy_target_sha256
        assert sha256(receipt["dual_search_wdl"][index].tobytes()).hexdigest() == row.dual_search_wdl_sha256
    assert receipt["selected_policy_sha256"] != receipt["dual_policy_sha256"]
    assert receipt["selected_wdl_sha256"] != receipt["dual_wdl_sha256"]


def test_changed_dual_extra_native_head_fails_blend_pin() -> None:
    sample = rows(nonuniform=True)
    s_bt4, s_ceres, _, _ = bindings(nonuniform=True)
    d_bt4, d_ceres, _, _ = bindings(nonuniform=True, corrupt_bt4_extra=True)
    with pytest.raises(ValueError, match="original-E dual blend byte mismatch"):
        screen.matched_cpu_kernel(sample, "11" * 32,
                                  s_bt4, s_ceres, d_bt4, d_ceres)


def test_bad_row_pin_or_legal_roster_refuses_before_sessions() -> None:
    sample = rows()
    legal = sample[0].legal.copy()
    legal[0] ^= 1
    bad = replace(sample[0], legal_sha256=sha256(legal.tobytes()).hexdigest(),
                  legal=legal)
    s_bt4, s_ceres, b, c = bindings()
    with pytest.raises(ValueError, match="legal mask"):
        screen.matched_cpu_kernel([bad, *sample[1:]], "11" * 32,
                                  s_bt4, s_ceres, s_bt4, s_ceres)
    assert not b.feeds
    assert not c.feeds
    with pytest.raises(ValueError, match="pin differs"):
        replace(sample[0], input_sha256="ff" * 32)


def test_route_or_independent_target_pin_mismatch_fails() -> None:
    sample = rows()
    wrong = "Ceres" if sample[0].selected_teacher == "BT4" else "BT4"
    s_bt4, s_ceres, b, c = bindings()
    with pytest.raises(ValueError, match="roster differs"):
        screen.matched_cpu_kernel([replace(sample[0], selected_teacher=wrong),
                                   *sample[1:]], "11" * 32,
                                  s_bt4, s_ceres, s_bt4, s_ceres)
    assert not b.feeds
    assert not c.feeds
    d_bt4, d_ceres, _, _ = bindings()
    with pytest.raises(ValueError, match="independently pinned"):
        screen.matched_cpu_kernel([replace(sample[0], policy_target_sha256="ff" * 32),
                                   *sample[1:]], "11" * 32,
                                  s_bt4, s_ceres, d_bt4, d_ceres)


def test_duplicate_input_refuses_before_sessions() -> None:
    sample = rows()
    s_bt4, s_ceres, b, c = bindings()
    with pytest.raises(ValueError, match="duplicate"):
        screen.matched_cpu_kernel([sample[0], sample[0], *sample[1:]],
                                  "11" * 32, s_bt4, s_ceres, s_bt4, s_ceres)
    assert not b.feeds
    assert not c.feeds


def test_cli_is_no_launch(capsys: pytest.CaptureFixture[str]) -> None:
    screen.main([])
    assert '"status": "NO-LAUNCH"' in capsys.readouterr().out
    with pytest.raises(SystemExit) as rejected:
        screen.main(["--execute"])
    assert rejected.value.code == 2
