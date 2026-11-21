"""simple_rl retarget: the no-rescale target rebuilt offline on the SAME games.

At move temperature 0 the played move does not depend on the stored target,
so a collection searched with ``--no-target-q-rescale`` plays the same games
as one with the rescale on. That pair is the ground truth: retargeting the
rescaled collection must reproduce the searched no-rescale collection's
``policy_target`` row for row. The retargeted collect is a new sealed run dir
(gen_0000 + the collect) with lineage to its source, a distinct digest, every
other array byte-identical, and an arm can train on it with --collect-from.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from chess_anti_engine.simple_rl.collect import load_generation_rows
from chess_anti_engine.simple_rl.common import sha256_tree
from chess_anti_engine.simple_rl.retarget import rebuild_rows, retarget_collect
from scripts import simple_rl_loop
from tests.test_simple_rl_resume import _setup
from tests.test_simple_rl_snapshots_arms import _argv, _stage

# float16 storage of both targets: one f16 ulp near 1 is 4.9e-4, so two
# independently rounded copies may differ by up to that (measured: see the test).
TOL = 1e-3
TEMP0 = ("--temp-start", "0", "--temp-floor", "0")


def _collect(tmp_path: Path, name: str, openings: Path, model_config: Path, *extra: str) -> Path:
    run = tmp_path / name
    assert simple_rl_loop.main(_stage(_argv(run, openings, model_config, *TEMP0, *extra), "collect")) == 0
    return run


@pytest.fixture(scope="module")
def pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path, Path, Path]:
    tmp_path = tmp_path_factory.mktemp("rt")
    mp = pytest.MonkeyPatch()
    try:
        openings, model_config = _setup(tmp_path, mp)
        rescaled = _collect(tmp_path, "rescaled", openings, model_config)
        searched = _collect(tmp_path, "searched_norescale", openings, model_config, "--no-target-q-rescale")
        out = tmp_path / "retargeted"
        retarget_collect(source_run_dir=rescaled, generation=1, out_run_dir=out)
    finally:
        mp.undo()
    return openings, model_config, rescaled, searched, out


def _games(run: Path) -> list[str]:
    return [json.dumps(json.loads(ln).get("moves")) for ln in
            (run / "gen_0001" / "collect" / "games.jsonl").read_text().splitlines()]


def test_the_retarget_reproduces_a_searched_no_rescale_target_on_the_same_games(
        pair: tuple[Path, Path, Path, Path, Path]) -> None:
    _, _, rescaled, searched, out = pair
    assert _games(rescaled) == _games(searched)  # temperature 0: same games, so rows pair one to one
    a, ma = load_generation_rows(rescaled / "gen_0001" / "collect")
    s, ms = load_generation_rows(searched / "gen_0001" / "collect")
    r, mr = load_generation_rows(out / "gen_0001" / "collect")
    assert np.array_equal(ma["x_hash"], ms["x_hash"])
    assert np.array_equal(mr["x_hash"], ma["x_hash"])
    pol = a["has_policy"].astype(bool)
    diff = np.abs(r["policy_target"].astype(np.float64) - s["policy_target"].astype(np.float64))[pol].max(axis=1)
    print(f"retarget vs searched no-rescale: max {diff.max():.3g}, p99 {np.quantile(diff, 0.99):.3g}, "
          f"rows {int(pol.sum())}")
    assert diff.max() <= TOL
    # Non-vacuous: the two targets really differ on these games.
    moved = np.abs(a["policy_target"].astype(np.float64) - s["policy_target"].astype(np.float64))[pol].max(axis=1)
    assert float(moved.max()) > 0.05


def test_the_retargeted_collect_is_a_new_sealed_artifact_with_lineage(
        pair: tuple[Path, Path, Path, Path, Path]) -> None:
    _, _, rescaled, _, out = pair
    src_c, out_c = rescaled / "gen_0001" / "collect", out / "gen_0001" / "collect"
    sm = json.loads((src_c / "manifest.json").read_text())
    om = json.loads((out_c / "manifest.json").read_text())
    rec = om["retarget"]
    assert rec["source_run_dir"] == str(rescaled.resolve())
    assert rec["source_manifest_sha256"] == hashlib.sha256((src_c / "manifest.json").read_bytes()).hexdigest()
    assert rec["target_q_rescale"] is False
    assert rec["rows"]["rebuilt"] > 20
    assert rec["rows"]["rebuilt"] + rec["rows"]["kept_solved_root"] + rec["rows"]["kept_tablebase_override"] \
        == rec["rows"]["policy_rows"]
    assert om["outputs"]["shards"]["sha256"] != sm["outputs"]["shards"]["sha256"]  # a distinct digest
    assert om["outputs"]["shards"]["sha256"] == sha256_tree(out_c / "shards")
    assert om["outputs"]["games"]["sha256"] == sm["outputs"]["games"]["sha256"]
    # The settings state the rebuilt target; everything else is the source's (the games' play config).
    assert om["config"]["settings"]["target_q_rescale"] is False
    assert sm["config"]["settings"]["target_q_rescale"] is True
    assert {k: v for k, v in om["config"]["settings"].items() if k != "target_q_rescale"} == \
        {k: v for k, v in sm["config"]["settings"].items() if k != "target_q_rescale"}
    assert om["config"]["dataclasses"] == sm["config"]["dataclasses"]
    assert rec["games_played_with_target_q_rescale"] is True
    assert om["inputs"]["checkpoint"]["sha256"] == sm["inputs"]["checkpoint"]["sha256"]
    assert (out / "gen_0000" / "manifest.json").is_file()
    # Every array except the target is byte-identical to the source's.
    a, _ = load_generation_rows(src_c)
    r, _ = load_generation_rows(out_c)
    assert set(a) == set(r)
    for k in a:
        if k != "policy_target":
            assert np.array_equal(a[k], r[k]), k
            assert a[k].dtype == r[k].dtype, k
    assert r["policy_target"].dtype == a["policy_target"].dtype


def test_an_arm_trains_on_the_retargeted_collect_and_the_source_never_moves(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pair: tuple[Path, Path, Path, Path, Path]) -> None:
    openings, model_config, rescaled, _, out = pair
    monkeypatch.setattr("chess_anti_engine.simple_rl.collect._open_stockfish",
                        lambda _s: __import__("tests.simple_rl_helpers", fromlist=["QuietStockfish"])
                        .QuietStockfish([0.0, 1.0, 0.0]))
    before = sha256_tree(rescaled)
    # The rescale-ON flags (the source's) are refused: they would pin a target the arm does not train on.
    wrong = _argv(tmp_path / "arm_wrong", openings, model_config, *TEMP0, "--collect-from", str(out),
                  init=out / "gen_0000" / "model.pt")
    with pytest.raises(SystemExit, match="target_q_rescale"):
        simple_rl_loop.main(wrong)
    arm = _argv(tmp_path / "arm", openings, model_config, *TEMP0, "--no-target-q-rescale", "--collect-from",
                str(out), init=out / "gen_0000" / "model.pt")
    assert simple_rl_loop.main(arm) == 0
    pinned = json.loads((tmp_path / "arm" / "run_config.json").read_text())["settings"]
    assert pinned["no_target_q_rescale"] is True
    ref = json.loads((tmp_path / "arm" / "gen_0001" / "collect" / "manifest.json").read_text())
    assert ref["collect_reference"]["source_run_dir"] == str(out.resolve())
    assert (tmp_path / "arm" / "gen_0001" / "train" / "manifest.json").is_file()
    assert sha256_tree(rescaled) == before


def test_retarget_refusals(pair: tuple[Path, Path, Path, Path, Path], tmp_path: Path) -> None:
    _, _, rescaled, searched, out = pair
    with pytest.raises(ValueError, match="exists"):
        retarget_collect(source_run_dir=rescaled, generation=1, out_run_dir=out)
    with pytest.raises(ValueError, match="already has target_q_rescale"):
        retarget_collect(source_run_dir=searched, generation=1, out_run_dir=tmp_path / "x")
    assert not (tmp_path / "x").exists()
    with pytest.raises(ValueError, match="is itself a retarget"):
        retarget_collect(source_run_dir=out, generation=1, out_run_dir=tmp_path / "y")


def test_a_kill_before_publishing_leaves_nothing_a_rerun_refuses(
        pair: tuple[Path, Path, Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, rescaled, _, _ = pair
    out = tmp_path / "killed"
    real_rename = Path.rename

    def killed(_self: Path, _target: Path) -> Path:
        raise KeyboardInterrupt("killed between seal and publish")

    monkeypatch.setattr(Path, "rename", killed)
    with pytest.raises(KeyboardInterrupt):
        retarget_collect(source_run_dir=rescaled, generation=1, out_run_dir=out)
    monkeypatch.setattr(Path, "rename", real_rename)
    assert not out.exists()
    assert not (tmp_path / ".killed.partial").exists()
    retarget_collect(source_run_dir=rescaled, generation=1, out_run_dir=out)  # the rerun is accepted
    assert (out / "gen_0001" / "collect" / "manifest.json").is_file()


def _arrays(n: int = 3) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    width = 8
    target = np.zeros((n, width), dtype=np.float16)
    target[:, :2] = 0.5
    arrays = {"policy_target": target, "has_policy": np.ones(n, dtype=np.uint8),
              "has_root_log_prior": np.zeros(n, dtype=np.uint8)}
    return arrays, {"tb_policy_override": np.zeros(n, dtype=bool)}


def test_a_policy_row_without_root_stats_is_refused_unless_solved_or_tablebase() -> None:
    arrays, meta = _arrays()
    with pytest.raises(ValueError, match="3 policy rows have no root statistics"):
        rebuild_rows(arrays, meta, cfg=None)
    arrays["policy_target"][0] = 0
    arrays["policy_target"][0, 3] = 1.0  # a solved root: one-hot, kept
    meta["tb_policy_override"][1] = True  # a tablebase override, kept
    arrays["has_policy"][2] = 0  # no policy target: untouched
    out, counts = rebuild_rows(arrays, meta, cfg=None)
    assert np.array_equal(out, arrays["policy_target"])
    assert counts["kept_solved_root"] == 1
    assert counts["kept_tablebase_override"] == 1
    assert counts["no_policy_untouched"] == 1
    assert counts["rebuilt"] == 0
