"""Synthetic CPU contract tests; no ONNX session or registered E bank opens."""
from __future__ import annotations

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
    def __init__(self, *, corrupt: bool = False) -> None:
        self.corrupt = corrupt
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
        policy[:, 0] = 1 if self.corrupt else 0
        wdl = np.tile(np.array([.25, .5, .25], dtype=np.float32), (len(feed), 1))
        return [policy, wdl]


class CeresSession:
    def __init__(self, *, corrupt: bool = False) -> None:
        self.corrupt = corrupt
        self.feeds: list[np.ndarray] = []

    def run(self, names: list[str], data: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert names == ["policy", "value", "value2"]
        feed = data["squares_byte"]
        assert feed.shape == (32, 64, 137)
        assert feed.dtype == np.uint8
        self.feeds.append(feed.copy())
        policy = np.zeros((32, 1858), dtype=np.float16)
        primary = np.zeros((32, 3), dtype=np.float16)
        primary[:, 0] = 1 if self.corrupt else 0
        secondary = np.zeros((32, 3), dtype=np.float16)
        return [policy, primary, secondary]


def rows(n: int = 6) -> list[screen.Row]:
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
        policy = np.zeros(1858, dtype=np.float16)
        policy[legal != 0] = np.float16(1 / legal.sum())
        wdl = np.array([.25, .5, .25] if teacher == "BT4" else
                       [1 / 3, 1 / 3, 1 / 3], dtype=np.float16)
        result.append(screen.Row(
            cohort_manifest_sha256="00" * 32, shard_ordinal=0,
            shard_rows=n, row_offset=offset,
            input_sha256=sha256(x.tobytes()).hexdigest(),
            legal_sha256=sha256(legal.tobytes()).hexdigest(),
            selected_teacher=teacher,
            policy_target_sha256=sha256(policy.tobytes()).hexdigest(),
            search_wdl_sha256=sha256(wdl.tobytes()).hexdigest(),
            x=x, legal=legal))
    return result


def bindings(*, corrupt_bt4: bool = False, corrupt_ceres: bool = False):
    b = BT4Session(corrupt=corrupt_bt4)
    c = CeresSession(corrupt=corrupt_ceres)
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


def test_bad_row_pin_or_legal_roster_refuses_before_sessions() -> None:
    sample = rows()
    legal = sample[0].legal.copy()
    legal[0] ^= 1
    bad = screen.Row(sample[0].cohort_manifest_sha256, 0, len(sample), 0,
                     sample[0].input_sha256, sha256(legal.tobytes()).hexdigest(),
                     sample[0].selected_teacher, sample[0].policy_target_sha256,
                     sample[0].search_wdl_sha256,
                     sample[0].x, legal)
    s_bt4, s_ceres, b, c = bindings()
    with pytest.raises(ValueError, match="legal mask"):
        screen.matched_cpu_kernel([bad, *sample[1:]], "11" * 32,
                                  s_bt4, s_ceres, s_bt4, s_ceres)
    assert not b.feeds
    assert not c.feeds
    with pytest.raises(ValueError, match="pin differs"):
        screen.Row(sample[0].cohort_manifest_sha256, 0, len(sample), 0,
                   "ff" * 32, sample[0].legal_sha256, sample[0].selected_teacher,
                   sample[0].policy_target_sha256, sample[0].search_wdl_sha256,
                   sample[0].x, sample[0].legal)


def test_route_or_independent_target_pin_mismatch_fails() -> None:
    from dataclasses import replace
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
