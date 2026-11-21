"""simple_rl train snapshots and training arms on one referenced collection.

Snapshots: mid-phase weights + metrics that do not change the training (final
weights bit-identical with and without them on CPU), resume a killed phase from
the newest snapshot without rewriting or skipping one, and are refused before
pinning when they reach past the phase's planned updates.

Arms: a run whose collect is a sealed reference to another run's collect
trains on byte-identical batches in the same order as a sibling arm with the
same --seed, re-verifies the source at every use, and can never move it.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from chess_anti_engine.simple_rl import learn as learn_mod
from chess_anti_engine.simple_rl.common import sha256_tree
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint
from scripts import simple_rl_loop
from tests.test_simple_rl_resume import Killed, _setup


def _argv(run_dir: Path, openings: Path, model_config: Path, *extra: str, init: Path | None = None) -> list[str]:
    gen0 = ["--init-checkpoint", str(init)] if init else ["--random-init", "--model-config", str(model_config)]
    return ["run", "--run-dir", str(run_dir), "--openings-dir", str(openings), *gen0,
            "--generations", "1", "--games", "4", "--concurrency", "4", "--sims", "8", "--max-plies", "30",
            "--rows-per-shard", "25", "--book-mix", "1.0", "--batch-size", "8", "--lr", "1e-3",
            "--holdout-frac", "0.5", "--presentations", "1", "--no-eval", "--seed", "77", *extra]


def _stage(argv: list[str], command: str, *extra: str) -> list[str]:
    out = [command, *argv[1:]]
    i = out.index("--generations")
    del out[i:i + 2]
    return [*out, "--generation", "1", *extra]


def _train(run_dir: Path) -> Path:
    return run_dir / "gen_0001" / "train"


def _weights(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)["model"]


def _same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.fixture
def straight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path, int]:
    openings, model_config = _setup(tmp_path, monkeypatch)
    run = tmp_path / "straight"
    assert simple_rl_loop.main(_argv(run, openings, model_config)) == 0
    updates = json.loads((_train(run) / "metrics.json").read_text())["summary"]["updates"]
    assert updates >= 4  # room for two interior snapshots
    return openings, model_config, run, int(updates)


def test_snapshots_leave_training_bit_identical_and_are_sealed(
        tmp_path: Path, straight: tuple[Path, Path, Path, int]) -> None:
    openings, model_config, run, updates = straight
    snaps = [1, updates // 2, updates]
    snap_run = tmp_path / "snap"
    assert simple_rl_loop.main(_argv(snap_run, openings, model_config,
                                     "--snapshot-updates", ",".join(map(str, snaps)))) == 0
    final = _weights(_train(snap_run) / "model.pt")
    assert _same(final, _weights(_train(run) / "model.pt"))
    assert (_train(snap_run) / "presented.npz").read_bytes() == (_train(run) / "presented.npz").read_bytes()
    # The snapshot at the last update IS the final model: a snapshot is the live state, not a copy of the start.
    assert _same(_weights(_train(snap_run) / "snapshots" / f"step_{updates}" / "model.pt"), final)
    assert not _same(_weights(_train(snap_run) / "snapshots" / "step_1" / "model.pt"), final)

    manifest = json.loads((_train(snap_run) / "manifest.json").read_text())
    for k in snaps:
        d = _train(snap_run) / "snapshots" / f"step_{k}"
        assert sorted(p.name for p in d.iterdir()) == ["metrics.json", "model.pt", "snapshot.json"]  # resume files gone
        for name in ("model", "metrics"):
            ref = manifest["outputs"][f"snapshot_{k}_{name}"]
            assert ref["sha256"] == hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest()
        metrics = json.loads((d / "metrics.json").read_text())
        assert metrics["update"] == k
        assert set(metrics["probe_vs_previous"]) >= {"mean_abs_dq", "policy_kl_prev_to_new", "top1_agree_prev"}
        assert set(metrics["heldout"]) >= {"policy_ce", "policy_entropy"}
        assert metrics["realised_lr_param_groups"]
        # Loadable by the arena's / evaluate's loader as a candidate path.
        load_model_from_checkpoint(d / "model.pt", device="cpu")
    last = json.loads((_train(snap_run) / "snapshots" / f"step_{updates}" / "metrics.json").read_text())
    summary = manifest["summary"]
    assert last["probe_vs_previous"] == summary["probe_vs_previous"]
    assert last["heldout"] == summary["heldout_after"]
    assert [s["update"] for s in summary["snapshots"]] == snaps
    assert json.loads((snap_run / "run_config.json").read_text())["settings"]["snapshot_updates"] == ",".join(
        map(str, snaps))


def test_kill_after_a_snapshot_resumes_without_rewriting_or_skipping(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, straight: tuple[Path, Path, Path, int]) -> None:
    openings, model_config, run, updates = straight
    k1, k2 = 1, updates - 1
    argv = _argv(tmp_path / "killed", openings, model_config, "--snapshot-updates", f"{k1},{k2}")
    real = learn_mod.write_snapshot

    def die_after_first(**kw: Any) -> Any:
        rec = real(**kw)
        if kw["update"] == k1:
            raise Killed
        return rec

    monkeypatch.setattr(learn_mod, "write_snapshot", die_after_first)
    with pytest.raises(Killed):
        simple_rl_loop.main(argv)
    monkeypatch.setattr(learn_mod, "write_snapshot", real)
    train = _train(tmp_path / "killed")
    first = train / "snapshots" / f"step_{k1}"
    assert not (train / "manifest.json").exists()
    stamp = {n: (first / n).stat() for n in ("model.pt", "metrics.json", "snapshot.json")}
    assert (first / "resume_trainer.pt").is_file()

    assert simple_rl_loop.main(argv) == 0
    for n, st in stamp.items():  # not rewritten
        now = (first / n).stat()
        assert (now.st_ino, now.st_mtime_ns) == (st.st_ino, st.st_mtime_ns)
    assert (train / "snapshots" / f"step_{k2}" / "snapshot.json").is_file()  # not skipped
    summary = json.loads((train / "metrics.json").read_text())["summary"]
    assert summary["resumed_from_snapshot"] == k1
    assert [s["update"] for s in summary["snapshots"]] == [k1, k2]
    # The resumed phase is the uninterrupted one.
    assert _same(_weights(train / "model.pt"), _weights(_train(run) / "model.pt"))
    assert (train / "presented.npz").read_bytes() == (_train(run) / "presented.npz").read_bytes()
    assert not any(train.rglob("resume_*.pt"))


def test_stale_snapshots_of_another_plan_are_not_resumed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, straight: tuple[Path, Path, Path, int]) -> None:
    """A killed attempt's snapshot whose resume key no longer matches (here a
    different --lr, adopted with --allow-config-change) is discarded, not resumed."""
    openings, model_config, _run, updates = straight
    argv = _argv(tmp_path / "k", openings, model_config, "--snapshot-updates", f"1,{updates}")
    real = learn_mod.write_snapshot

    def die(**kw: Any) -> Any:
        real(**kw)
        raise Killed

    monkeypatch.setattr(learn_mod, "write_snapshot", die)
    with pytest.raises(Killed):
        simple_rl_loop.main(argv)
    monkeypatch.setattr(learn_mod, "write_snapshot", real)
    changed = list(argv)
    changed[changed.index("--lr") + 1] = "5e-4"
    assert simple_rl_loop.main([*changed, "--allow-config-change"]) == 0
    summary = json.loads((_train(tmp_path / "k") / "metrics.json").read_text())["summary"]
    assert summary["resumed_from_snapshot"] is None


def test_snapshot_plan_is_refused_before_pinning(straight: tuple[Path, Path, Path, int]) -> None:
    openings, model_config, run, updates = straight
    config = (run / "run_config.json").read_bytes()
    redo = _stage(_argv(run, openings, model_config), "train", "--redo", "--allow-config-change")
    with pytest.raises(SystemExit, match="reach past the phase's"):
        simple_rl_loop.main([*redo, "--snapshot-updates", f"1,{updates + 1}"])
    assert (run / "run_config.json").read_bytes() == config
    assert (_train(run) / "manifest.json").is_file()  # nothing unsealed
    with pytest.raises(ValueError, match="strictly increasing"):
        simple_rl_loop.main([*redo, "--snapshot-updates", "2,2"])
    with pytest.raises(ValueError, match="exact-once"):
        simple_rl_loop.main([*redo, "--snapshot-updates", "1", "--sample-with-replacement", "--updates", "3"])
    assert (run / "run_config.json").read_bytes() == config


def test_standalone_collect_validates_the_train_flags_it_pins(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    openings, model_config = _setup(tmp_path, monkeypatch)
    run = tmp_path / "run"
    argv = _stage(_argv(run, openings, model_config), "collect")
    with pytest.raises(ValueError, match="presentations"):
        simple_rl_loop.main([*argv, "--presentations", "3"])
    assert not run.exists()


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    openings, model_config = _setup(tmp_path, monkeypatch)
    src = tmp_path / "source"
    assert simple_rl_loop.main(_stage(_argv(src, openings, model_config), "collect")) == 0
    return openings, model_config, src


def _arm(tmp_path: Path, name: str, source: tuple[Path, Path, Path], *extra: str) -> list[str]:
    openings, model_config, src = source
    return _argv(tmp_path / name, openings, model_config, "--collect-from", str(src), *extra,
                 init=src / "gen_0000" / "model.pt")


def test_arms_on_one_collect_see_identical_batches(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: tuple[Path, Path, Path]) -> None:
    src = source[2]
    src_digest = sha256_tree(src / "gen_0001")
    served: dict[str, list[str]] = {}
    arm = {"name": ""}
    real = learn_mod.ScheduledSampler.sample_batch_arrays

    def recording(self: Any, batch_size: int) -> dict[str, np.ndarray]:
        batch = real(self, batch_size)
        h = hashlib.sha256()
        for k in sorted(batch):
            h.update(k.encode())
            h.update(np.ascontiguousarray(batch[k]).tobytes())
        served.setdefault(arm["name"], []).append(h.hexdigest())
        return batch

    monkeypatch.setattr(learn_mod.ScheduledSampler, "sample_batch_arrays", recording)
    for name, w in (("a", "1.0"), ("b", "0.0")):
        arm["name"] = name
        assert simple_rl_loop.main(_arm(tmp_path, name, source, "--w-wdl", w)) == 0
    a, b = tmp_path / "a", tmp_path / "b"
    assert served["a"] == served["b"]
    assert len(served["a"]) > 1
    assert (_train(a) / "presented.npz").read_bytes() == (_train(b) / "presented.npz").read_bytes()
    assert (a / "probe.npz").read_bytes() == (b / "probe.npz").read_bytes()
    ma = json.loads((_train(a) / "manifest.json").read_text())["summary"]
    mb = json.loads((_train(b) / "manifest.json").read_text())["summary"]
    assert (ma["realised_w_wdl"], mb["realised_w_wdl"]) == (1.0, 0.0)  # each arm's own pinned setting
    assert ma["holdout_rows"] == mb["holdout_rows"] > 0
    assert not _same(_weights(_train(a) / "model.pt"), _weights(_train(b) / "model.pt"))
    ref = json.loads((a / "gen_0001" / "collect" / "manifest.json").read_text())
    assert ref["collect_reference"]["source_collect_dir"] == str((src / "gen_0001" / "collect").resolve())
    assert any(k.startswith("source_shard_") for k in ref["inputs"])
    assert sorted(p.name for p in (a / "gen_0001" / "collect").iterdir()) == ["manifest.json"]

    # The arm's redo machinery acts on the arm only: the source never moves.
    redo = _stage(_arm(tmp_path, "a", source, "--w-wdl", "1.0"), "collect", "--redo", "--redo-downstream")
    assert simple_rl_loop.main(redo) == 0
    assert not (_train(a) / "manifest.json").exists()
    assert sha256_tree(src / "gen_0001") == src_digest


def test_a_changed_source_refuses_the_arm(tmp_path: Path, source: tuple[Path, Path, Path]) -> None:
    src = source[2]
    argv = _arm(tmp_path, "a", source)
    assert simple_rl_loop.main(_stage(argv, "collect")) == 0
    games = src / "gen_0001" / "collect" / "games.jsonl"
    games.write_text(games.read_text() + "\n")
    with pytest.raises(SystemExit, match="source collect this arm references changed"):
        simple_rl_loop.main(argv)  # train on the reference
    assert not (_train(tmp_path / "a") / "manifest.json").exists()
    # ...and re-referencing re-checks the source against ITS OWN seal.
    with pytest.raises(SystemExit, match="no longer matches the sha256 its manifest sealed"):
        simple_rl_loop.main(_stage(argv, "collect", "--redo"))


def test_arm_refusals(tmp_path: Path, source: tuple[Path, Path, Path]) -> None:
    openings, model_config, src = source
    # Unsealed source generation.
    with pytest.raises(SystemExit, match="is not sealed"):
        simple_rl_loop.main([*_stage(_arm(tmp_path, "a", source), "collect")[:-1], "2"])
    assert not (tmp_path / "a").exists()  # refused before pinning
    # Collect flags that would be pinned but have no effect on the rows.
    with pytest.raises(SystemExit, match="collect settings differ"):
        simple_rl_loop.main(_arm(tmp_path, "b", source, "--sims", "16"))
    assert not (tmp_path / "b").exists()
    # Another --seed: it also draws the train plan, so arms must share the source's.
    seeded = _arm(tmp_path, "s", source)
    seeded[seeded.index("--seed") + 1] = "78"
    with pytest.raises(SystemExit, match=r"collect settings differ.*'seed'"):
        simple_rl_loop.main(seeded)
    assert not (tmp_path / "s").exists()  # refused before pinning
    # Another generation-0 model.
    payload = torch.load(src / "gen_0000" / "model.pt", map_location="cpu", weights_only=False)
    key = next(k for k, v in payload["model"].items() if v.is_floating_point())
    payload["model"][key] = payload["model"][key] + 1e-3
    torch.save(payload, tmp_path / "other_gen0.pt")
    other = _argv(tmp_path / "c", openings, model_config, "--collect-from", str(src), init=tmp_path / "other_gen0.pt")
    with pytest.raises(SystemExit, match="generation-0 model is not the source's"):
        simple_rl_loop.main(other)
    # A reference to a reference.
    assert simple_rl_loop.main(_stage(_arm(tmp_path, "d", source), "collect")) == 0
    chained = _argv(tmp_path / "e", openings, model_config, "--collect-from", str(tmp_path / "d"),
                    init=src / "gen_0000" / "model.pt")
    with pytest.raises(SystemExit, match="itself a reference"):
        simple_rl_loop.main(chained)


def test_capped_taint_applies_through_the_reference(tmp_path: Path, source: tuple[Path, Path, Path]) -> None:
    src = source[2]
    path = src / "gen_0001" / "collect" / "manifest.json"
    m = json.loads(path.read_text())
    # As if collected on CUDA at an evaluator cap below boards x topk (old-format record).
    m["config"]["settings"].update(device="cuda", eval_max_batch=8)
    m["stats"]["eval_max_batch_realised"] = 8
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="evaluator cap"):
        simple_rl_loop.main(_arm(tmp_path, "a", source))
    assert simple_rl_loop.main(_arm(tmp_path, "b", source, "--allow-capped-collect-rows")) == 0
    summary = json.loads((_train(tmp_path / "b") / "metrics.json").read_text())["summary"]
    assert summary["trained_on_capped_rows"] is True


def test_arm_rows_are_the_sources(tmp_path: Path, source: tuple[Path, Path, Path]) -> None:
    from chess_anti_engine.simple_rl.collect import load_generation_rows

    src = source[2]
    assert simple_rl_loop.main(_stage(_arm(tmp_path, "a", source), "collect")) == 0
    ra, ma = load_generation_rows(tmp_path / "a" / "gen_0001" / "collect")
    rs, ms = load_generation_rows(src / "gen_0001" / "collect")
    assert ra.keys() == rs.keys()
    assert all(np.array_equal(ra[k], rs[k]) for k in ra)
    assert np.array_equal(ma["game_id"], ms["game_id"])
    shutil.rmtree(src / "gen_0001")  # the source unsealed/moved: the arm cannot train on it
    with pytest.raises(SystemExit):
        simple_rl_loop.main(_arm(tmp_path, "a", source))


# ---------------------------------------------------------------------------
# init-optimizer family check (pre-pin)
# ---------------------------------------------------------------------------

def _groups(opt: Any) -> list[dict[str, Any]]:
    return opt.state_dict()["param_groups"]


def test_optimizer_family_recognises_every_trainer_optimizer() -> None:
    from chess_anti_engine.simple_rl.learn import UNRECOGNISED_OPTIMIZER, _optimizer_family

    p = [torch.nn.Parameter(torch.zeros(3, 3))]
    assert _optimizer_family(_groups(torch.optim.NAdam(p, decoupled_weight_decay=True))) == "nadamw"
    assert _optimizer_family(_groups(torch.optim.AdamW(p))) == "adamw"
    from pytorch_optimizer import SOAP

    assert _optimizer_family(_groups(SOAP(p))) == "soap"
    # Aurora / Muon branch groups carry the OTHER flag as False in either optimizer.
    aurora = [{"use_aurora": True, "params": [0]}, {"use_aurora": False, "use_muon": False, "params": [1]}]
    muon = [{"use_muon": True, "params": [0]}, {"use_muon": False, "use_aurora": False, "params": [1]}]
    assert _optimizer_family(aurora) == "aurora"
    assert _optimizer_family(muon) == "muon"
    assert _optimizer_family(_groups(torch.optim.SGD(p, lr=0.1))) == UNRECOGNISED_OPTIMIZER
    assert _optimizer_family([]) is None


def test_adamw_state_is_refused_for_a_nadamw_run(tmp_path: Path) -> None:
    from chess_anti_engine.simple_rl.learn import TrainSettings, check_init_optimizer_source

    p = [torch.nn.Parameter(torch.zeros(3, 3))]
    opt = torch.optim.AdamW(p)
    p[0].grad = torch.ones(3, 3)
    opt.step()
    payload = {"opt": opt.state_dict(), "scheduler": {}}
    with pytest.raises(ValueError, match="written by adamw, not the requested --optimizer nadamw"):
        check_init_optimizer_source(payload, tmp_path / "x.pt", TrainSettings(optimizer="nadamw"))
    check_init_optimizer_source(payload, tmp_path / "x.pt", TrainSettings(optimizer="adamw"))
    sgd = torch.optim.SGD(p, lr=0.1, momentum=0.9)
    sgd.step()
    with pytest.raises(ValueError, match="no Trainer optimizer writes"):
        check_init_optimizer_source({"opt": sgd.state_dict(), "scheduler": {}}, tmp_path / "x.pt",
                                    TrainSettings(optimizer="adamw"))


def test_train_generation_itself_reverifies_a_reference(tmp_path: Path, source: tuple[Path, Path, Path]) -> None:
    """Not only the driver: the train stage re-checks a referenced source before loading its rows."""
    from chess_anti_engine.simple_rl.learn import TrainSettings, train_generation

    src = source[2]
    arm = tmp_path / "a"
    assert simple_rl_loop.main(_stage(_arm(tmp_path, "a", source), "collect")) == 0
    starts = src / "gen_0001" / "collect" / "starts.txt"
    starts.write_text(starts.read_text() + "x\n")
    with pytest.raises(SystemExit, match="source collect this arm references changed"):
        train_generation(run_dir=arm, generation=1, init_checkpoint=arm / "gen_0000" / "model.pt",
                         prev_trainer_checkpoint=None, settings=TrainSettings(batch_size=8, seed=77))


def test_an_off_policy_generation_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same gen-0 model, but the source's generation-2 collect was played by the SOURCE's gen-1 model, not
    the arm's (trained at another lr): referencing it would train the arm on another policy's rows."""
    openings, model_config = _setup(tmp_path, monkeypatch)
    src = tmp_path / "src"
    two = _argv(src, openings, model_config)
    two[two.index("--generations") + 1] = "2"
    assert simple_rl_loop.main(two) == 0
    arm = _argv(tmp_path / "arm", openings, model_config, "--collect-from", str(src),
                init=src / "gen_0000" / "model.pt")
    arm[arm.index("--generations") + 1] = "2"
    arm[arm.index("--lr") + 1] = "5e-4"
    with pytest.raises(SystemExit, match="off-policy for this arm"):
        simple_rl_loop.main(arm)
    assert (tmp_path / "arm" / "gen_0001" / "train" / "manifest.json").is_file()
    assert not (tmp_path / "arm" / "gen_0002" / "collect" / "manifest.json").exists()


def test_train_generation_refuses_a_bad_plan_before_writing_anything(source: tuple[Path, Path, Path]) -> None:
    from chess_anti_engine.simple_rl.learn import TrainSettings, planned_phase_updates, train_generation

    src = source[2]
    settings = TrainSettings(batch_size=8, holdout_frac=0.5, seed=77)
    planned = planned_phase_updates(src, 1, settings)
    assert planned is not None
    before = sorted(p.relative_to(src) for p in src.rglob("*"))
    bad = TrainSettings(batch_size=8, holdout_frac=0.5, seed=77, snapshot_updates=(1, planned + 1))
    with pytest.raises(ValueError, match="reach past the phase's"):
        train_generation(run_dir=src, generation=1, init_checkpoint=src / "gen_0000" / "model.pt",
                         prev_trainer_checkpoint=None, settings=bad)
    assert sorted(p.relative_to(src) for p in src.rglob("*")) == before  # no train dir, no probe.npz
    # The metadata-only count is the batch plan's: the same settings train exactly `planned` updates.
    ok = TrainSettings(batch_size=8, holdout_frac=0.5, seed=77, snapshot_updates=(planned,))
    summary = train_generation(run_dir=src, generation=1, init_checkpoint=src / "gen_0000" / "model.pt",
                               prev_trainer_checkpoint=None, settings=ok)
    assert summary["updates"] == planned


def test_a_cosine_phase_resumed_from_a_snapshot_is_the_uninterrupted_phase(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The resumed Trainer's scheduler (the per-group-floor cosine) continues where the
    killed one stopped: final weights and the realised lr equal the straight run's."""
    openings, model_config = _setup(tmp_path, monkeypatch)
    cos = ("--lr", "1e-3", "--lr-floor", "1e-4", "--lr-schedule", "cosine-phase")
    run = tmp_path / "straight_cos"
    assert simple_rl_loop.main(_argv(run, openings, model_config, *cos)) == 0
    updates = int(json.loads((_train(run) / "metrics.json").read_text())["summary"]["updates"])
    k1 = max(1, updates // 2)
    argv = _argv(tmp_path / "killed_cos", openings, model_config, *cos, "--snapshot-updates", str(k1))
    real = learn_mod.write_snapshot

    def die(**kw: Any) -> Any:
        real(**kw)
        raise Killed

    monkeypatch.setattr(learn_mod, "write_snapshot", die)
    with pytest.raises(Killed):
        simple_rl_loop.main(argv)
    monkeypatch.setattr(learn_mod, "write_snapshot", real)
    assert simple_rl_loop.main(argv) == 0
    train = _train(tmp_path / "killed_cos")
    summary = json.loads((train / "metrics.json").read_text())["summary"]
    assert summary["resumed_from_snapshot"] == k1
    assert _same(_weights(train / "model.pt"), _weights(_train(run) / "model.pt"))
    want = json.loads((_train(run) / "metrics.json").read_text())["summary"]["realised_optimizer"]
    assert summary["realised_optimizer"]["param_group_lr"] == want["param_group_lr"]
