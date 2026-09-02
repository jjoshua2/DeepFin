"""The three lc0-control DRIVERS, at their ``main()`` entry points.

⚑⚑ WHY THIS FILE EXISTS. The library modules under it were well tested and the
WIRING that decides whether any of them takes effect was not: a reviewer
applied TWO mutants at once — deleting the realized-guard call from
``lc0_control_train`` outright, and breaking ``_LossCapture.worst`` to
``readouts[0]`` so a leak starting late is never seen — and all 36 tests
still passed. This codebase's signature defect is a value that is accepted and
then silently ignored, and an entry point with no test is where that lives.

Everything here runs on CPU at a toy width. The architecture is deliberately
NOT production's: these tests are about control flow, and
``tests/test_lc0_control_config.py`` plus ``tests/test_lc0_control_arch.py``
own the "is it the right net" question.
"""
from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from chess_anti_engine.eval import lc0_control_arch, lc0_control_replay
from chess_anti_engine.moves import COMPACT_POLICY_SIZE
from chess_anti_engine.train.target_builder import DEFAULT_CATEGORICAL_BINS
from chess_anti_engine.replay.sample import ReplaySample
from chess_anti_engine.replay.shard import (
    ShardMeta,
    samples_to_arrays,
    save_local_shard_arrays,
)
from chess_anti_engine.train.value_blend_guard import (
    CategoricalRebuildReadout,
    ValueBlendMisconfigured,
    value_blend_readout,
)
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file
from scripts import lc0_control_eval, lc0_control_heldout, lc0_control_train

REPO = Path(__file__).resolve().parent.parent
CONTROL = REPO / "configs" / "lc0_positive_control.yaml"
PLANES = 175


def _tiny_config(tmp_path: Path, **overrides: Any) -> Path:
    """The control config at toy width, with the value blend left alone.

    Built by LOADING the shipped config and shrinking it, not by writing a new
    one: every trainer knob under test — the optimizer, the loss weights, the
    blend — stays whatever production's copy says, so a rename in the real file
    breaks this test instead of drifting away from it.
    """
    raw = yaml.safe_load(CONTROL.read_text(encoding="utf-8"))
    raw["model"].update(embed_dim=32, num_layers=2, num_heads=2, ffn_mult=1.5)
    raw["model"].pop("ffn_mult_by_layer", None)
    raw["train"]["batch_size"] = 4
    for section, values in overrides.items():
        raw.setdefault(section, {}).update(values)
    path = tmp_path / "tiny_control.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _live_stub(path: Path, model: dict[str, Any], control_config: Path) -> Path:
    """A stand-in for the LIVE production yaml: this ``model:``, LIVE's trainer.

    ⚑ A stub carrying only ``model:`` is not a plausible live config, and since
    launch guard 0c reads the SAME ``$CHESS_LIVE_PRODUCTION_CONFIG`` it would
    refuse the run for reasons the architecture test is not about — every
    trainer kwarg would fall back to a library default. So the stub takes the
    control's own ``train:``/``selfplay:`` with the two declared value-blend
    deviations put BACK to live's values, which is what makes it "live".
    """
    from chess_anti_engine.eval.lc0_control_trainer import LIVE_TRAINER_PIN

    raw = yaml.safe_load(control_config.read_text(encoding="utf-8"))
    live_kwargs = LIVE_TRAINER_PIN["kwargs"]
    raw["train"]["sf_wdl_frac"] = float(live_kwargs["sf_wdl_frac"])
    raw["train"]["search_wdl_frac"] = float(live_kwargs["search_wdl_frac"])
    raw["model"] = dict(model)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _lc0_like_sample(
    seed: int, *, with_sf_wdl: bool, with_search_wdl: bool = True,
    with_categorical_target: bool = False,
) -> ReplaySample:
    """One converted-lc0-looking row: a search WDL, and no SF label.

    ⚑ ``with_search_wdl=False`` is review F1's corpus: `compute_loss` falls the
    SEARCH component back to the raw one-hot exactly as it falls the SF
    component back, and `--shards` is a CLI argument like any other.
    """
    rng = np.random.default_rng(seed)
    policy = rng.random(COMPACT_POLICY_SIZE).astype(np.float32)
    policy /= policy.sum()
    legal = np.zeros(COMPACT_POLICY_SIZE, dtype=np.uint8)
    legal[rng.choice(COMPACT_POLICY_SIZE, size=20, replace=False)] = 1
    search = rng.random(3).astype(np.float32)
    search /= search.sum()
    sample = ReplaySample(
        x=rng.random((PLANES, 8, 8)).astype(np.float32),
        policy_target=policy,
        wdl_target=int(seed % 3),
        legal_mask=legal,
        search_wdl=search if with_search_wdl else None,
        moves_left=0.1,
        has_policy=True,
        is_selfplay=True,
        is_network_turn=True,
        game_id=seed,
        ply_index=0,
    )
    if with_sf_wdl:
        sf = rng.random(3).astype(np.float32)
        sample.sf_wdl = sf / sf.sum()
    if with_categorical_target:
  # ⚑ The column the CONVERTER does not write. Its absence is the only reason
  # production's armed `rebuild_categorical_target` is inert on this arm, so
  # the corpus that would arm it has to be constructible here.
        categorical = np.zeros((DEFAULT_CATEGORICAL_BINS,), dtype=np.float32)
        categorical[DEFAULT_CATEGORICAL_BINS - 1] = 1.0
        sample.categorical_target = categorical
    return sample


def _write_shards(
    shard_dir: Path, seeds: list[int], *, with_sf_wdl: bool = False,
    with_search_wdl: bool = True, with_categorical_target: bool = False,
) -> Path:
    shard_dir.mkdir(parents=True, exist_ok=True)
    samples = [
        _lc0_like_sample(
            s, with_sf_wdl=with_sf_wdl, with_search_wdl=with_search_wdl,
            with_categorical_target=with_categorical_target,
        )
        for s in seeds
    ]
    save_local_shard_arrays(
        shard_dir / "shard_000000.zarr",
        arrs=samples_to_arrays(samples),
        meta=ShardMeta(
            positions=len(samples),
            input_history_encoding="lc0_root_legacy_meta",
            history_rep_fix=True,
            policy_encoding="lc0_1858",
            policy_size=COMPACT_POLICY_SIZE,
        ),
    )
    return shard_dir


# ── lc0_control_train: the realized guard, and whether it is REACHED ──────────


def test_the_history_identity_is_read_back_off_the_written_summary(
    tmp_path: Path,
) -> None:
    """⚑ Fable (round 2 delta): ``corpus.history_identity`` off ``summary.json``
    as ``main`` writes it, not only via ``history_identity_record``.  One
    directory stamped history-aware (schema 3, ``zero_history: false``) and one
    unstamped (the bare-FEN reading): refused without the flag, recorded as
    mixed with it.
    """
    import zarr

    zero = _write_shards(tmp_path / "zero", list(range(8)))
    hist = _write_shards(tmp_path / "hist", list(range(8, 16)))
    group = zarr.open_group(str(hist / "shard_000000.zarr"), mode="a")
    group.attrs.update({"derive_corpus_row_schema": 3, "zero_history": False})
    argv = [
        "--config", str(_tiny_config(tmp_path)), "--shards", str(zero), str(hist),
        "--out-dir", str(tmp_path / "run"), "--steps", "1", "--batch-size", "4",
        "--device", "cpu", "--no-compile", "--allow-arch-drift",
        "--allow-invalid-control",
    ]
    with pytest.raises(SystemExit, match="input-history identities"):
        lc0_control_train.main(argv)
    assert not (tmp_path / "run" / "summary.json").exists()

    rc = lc0_control_train.main([*argv, "--allow-mixed-history"])
    assert rc == 0
    summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert summary["corpus"]["history_identity"] == {
        "row_schemas": ["1", "3"], "zero_history": [False, True],
        "mixed_within": [], "in_progress": [], "mixed": True,
        "allow_mixed_history": True,
    }


def test_a_partial_corpus_stamp_is_read_back_off_the_written_summary(
    tmp_path: Path,
) -> None:
    """``corpus.partial_corpus`` off ``summary.json`` as ``main`` writes it:
    a shard stamped ``corpus_complete: false`` is refused, and recorded with
    ``--allow-partial-corpus``."""
    import zarr

    shards = _write_shards(tmp_path / "rows", list(range(16)))
    group = zarr.open_group(str(shards / "shard_000000.zarr"), mode="a")
    group.attrs.update({
        "corpus_complete": False, "corpus_run_finished_claim": False,
        "corpus_shards_adopted": 3, "corpus_rows_claimed": 300, "corpus_rows_derived": 16,
    })
    argv = [
        "--config", str(_tiny_config(tmp_path)), "--shards", str(shards),
        "--out-dir", str(tmp_path / "run"), "--steps", "1", "--batch-size", "4",
        "--device", "cpu", "--no-compile", "--allow-arch-drift",
        "--allow-invalid-control",
    ]
    with pytest.raises(SystemExit, match="PARTIAL corpus"):
        lc0_control_train.main(argv)
    assert not (tmp_path / "run" / "summary.json").exists()

    rc = lc0_control_train.main([*argv, "--allow-partial-corpus"])
    assert rc == 0
    summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert summary["corpus"]["partial_corpus"] == {
        "incomplete_shards": {"rows/shard_000000.zarr": {
            "run_finished_claim": False, "shards_adopted": 3,
            "rows_claimed": 300, "rows_derived": 16,
        }},
        "partial": True, "allow_partial_corpus": True,
    }


def test_the_control_run_completes_and_writes_a_summary(tmp_path: Path) -> None:
    """Baseline. Without this the failure tests below cannot be told from a
    driver that always exits non-zero."""
    shards = _write_shards(tmp_path / "rows", list(range(16)))
    out = tmp_path / "run"
    rc = lc0_control_train.main([
        "--config", str(_tiny_config(tmp_path)), "--shards", str(shards),
        "--out-dir", str(out), "--steps", "1", "--batch-size", "4",
        "--device", "cpu", "--no-compile", "--allow-arch-drift",
        "--allow-invalid-control",
    ])
    assert rc == 0
    assert (out / "checkpoint.pt").is_file()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    realized = summary["realized"]
    assert realized["leaked_to_outcome"] == 0.0
  # ⚑ 0.00, not the 0.30 this asserted until 2026-08-16. The LIVE blend is
  # `sf 0.69 + search 0.31 = 1.00`, so production puts NO weight on the raw
  # game outcome and neither does the re-pointed arm.
    assert realized["outcome_borne_frac (game_frac + leak)"] == pytest.approx(0.0)
    assert realized["search_wdl_frac (realized)"] == pytest.approx(1.0)
  # The categorical rebuild carries production's armed 0.69/0.31 and is inert
  # here because the rows have no `categorical_target` column — measured, not
  # assumed, and recorded in the artifact so a later corpus change is visible.
    categorical = summary["realized_categorical"]
    assert categorical["categorical_target column present"] == 0.0
    assert categorical["rebuild applies to this batch"] == 0.0
    assert categorical["categorical outcome_borne_frac"] == pytest.approx(0.0)
    assert summary["valid_control"] is False, "--allow-arch-drift must be recorded"
    assert any("warmup_steps" in p for p in summary["validity_problems"]), (
        "a 1-step run cannot have left LR warmup and the artifact must say so"
    )


def test_the_realized_guard_fails_the_run_and_writes_no_checkpoint(
    tmp_path: Path,
) -> None:
    """⚑⚑ THE HEADLINE CHECK, PROVEN REACHABLE.

    It previously could not fail from any input: launch guard 1 refuses every
    ``sf_wdl_frac > 0`` config, nothing downstream can raise the frac, and
    ``--allow-leak`` — the only way past guard 1 — skipped this assert too. The
    flag now opens only the LAUNCH gates, so this is the state that fires it:
    production's blend, pointed at rows that carry no SF label.
    """
    shards = _write_shards(tmp_path / "rows", list(range(16)))
  # ⚑ THE LIVE production blend (`sf 0.69 + search 0.31 = 1.00`, `game_frac`
  # 0.00), not `main`'s stale 0.50/0.20. On SF-labelled rows this is a clean
  # pass; on lc0 rows the 0.69 falls to the raw outcome. Using the stale pair
  # here would ALSO have tripped the outcome-borne bar, for the unrelated
  # reason that its intended `game_frac` is 0.30.
    config = _tiny_config(
        tmp_path, train={"sf_wdl_frac": 0.69, "search_wdl_frac": 0.31},
    )
    out = tmp_path / "leak_run"
    with pytest.raises(SystemExit) as excinfo:
        lc0_control_train.main([
            "--config", str(config), "--shards", str(shards),
            "--out-dir", str(out), "--steps", "1", "--batch-size", "4",
            "--device", "cpu", "--no-compile", "--allow-arch-drift",
            "--allow-invalid-control", "--allow-leak",
        ])
    assert "RAW GAME OUTCOME" in str(excinfo.value)
    assert not (out / "checkpoint.pt").exists(), (
        "a run that failed the realized guard must leave no checkpoint behind"
    )


def test_the_categorical_guard_fails_the_run_on_a_corpus_that_gains_a_target(
    tmp_path: Path,
) -> None:
    """⚑⚑ THE ARMED MECHANISM, DRIVEN THROUGH THE ENTRY POINT.

    The arm carries production's `rebuild_categorical_target: true` with
    `categorical_blend_frac: 0.69`, and that is a NO-OP today only because
    `scripts/lc0_data_to_rows.py` writes no `categorical_target` column. The
    inertness is a property of the CONVERTER, so this constructs the corpus the
    converter does not currently produce and requires the run to refuse it: with
    no `sf_wdl`, 0.69 of the categorical target becomes the raw game outcome.

    ⚑ Through `main()`, not by calling the assert. The finding this covers is a
    WIRING one — a guard whose call site is missing is a guard that reads as
    protection — and the batch it judges is `compute_loss`'s SECOND POSITIONAL
    argument, which a wrapper reading `kwargs` would silently see as empty.
    """
    shards = _write_shards(
        tmp_path / "cat_rows", list(range(16)), with_categorical_target=True,
    )
    out = tmp_path / "cat_run"
    with pytest.raises(SystemExit) as excinfo:
        lc0_control_train.main([
            "--config", str(_tiny_config(tmp_path)), "--shards", str(shards),
            "--out-dir", str(out), "--steps", "1", "--batch-size", "4",
            "--device", "cpu", "--no-compile", "--allow-arch-drift",
            "--allow-invalid-control",
        ])
    message = str(excinfo.value)
    assert "CATEGORICAL target rebuild" in message
    assert "0.6900 of its mass" in message
    assert not (out / "checkpoint.pt").exists()


def test_the_trainer_publishes_the_effective_label_mass_and_warns_on_a_leak(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """⚑⚑ REVIEW F3 — THE PR'S ONLY PRODUCTION-PATH BEHAVIOUR CHANGE, TESTED.

    ``TrainMetrics.sf_wdl_effective_frac`` / ``search_wdl_effective_frac`` and
    ``Trainer._warn_if_value_blend_leaks_to_outcome`` shipped with ZERO tests:
    deleting both the ``_RATIO_METRIC_FIELDS`` entries and the call left 81
    tests green, and a column publishing a constant 0.0 would make the
    TB-derived leak read as a FULL leak on production forever.

    Both directions, because they kill different mutants:

    * SF-labelled rows -> the column reads 1.0 and the trainer stays SILENT.
      Deleting the ratio-map entry leaves the field at its 0.0 default, which
      reads as "nothing was labelled" and fires the warning spuriously.
    * unlabelled rows -> the trainer WARNS. Deleting the call in
      ``train_steps`` silences it while every other column stays right.

    Driven through the real trainer, not by calling the method: the finding is
    about wiring, and a direct call cannot see a missing call site.
    """
    labelled = _write_shards(tmp_path / "sf_rows", list(range(16)), with_sf_wdl=True)
  # ⚑ THE LIVE production blend (`sf 0.69 + search 0.31 = 1.00`, `game_frac`
  # 0.00), not `main`'s stale 0.50/0.20. On SF-labelled rows this is a clean
  # pass; on lc0 rows the 0.69 falls to the raw outcome. Using the stale pair
  # here would ALSO have tripped the outcome-borne bar, for the unrelated
  # reason that its intended `game_frac` is 0.30.
    config = _tiny_config(
        tmp_path, train={"sf_wdl_frac": 0.69, "search_wdl_frac": 0.31},
    )
    out = tmp_path / "labelled_run"
    with caplog.at_level("WARNING", logger="chess_anti_engine.train.trainer"):
        rc = lc0_control_train.main([
            "--config", str(config), "--shards", str(labelled),
            "--out-dir", str(out), "--steps", "1", "--batch-size", "4",
            "--device", "cpu", "--no-compile", "--allow-arch-drift",
            "--allow-invalid-control", "--allow-leak",
        ])
    assert rc == 0
    metrics = json.loads((out / "summary.json").read_text(encoding="utf-8"))["metrics"]
    assert metrics["sf_wdl_effective_frac"] == pytest.approx(1.0), (
        "the column must reach TrainMetrics; at its 0.0 default the TB-derived "
        "leak reads as a full leak on production"
    )
    assert metrics["search_wdl_effective_frac"] == pytest.approx(1.0)
    assert not [r for r in caplog.records if "RAW GAME OUTCOME" in r.getMessage()], (
        "every row carries both labels — nothing leaked"
    )

    caplog.clear()
    unlabelled = _write_shards(tmp_path / "lc0_rows", list(range(16)))
    with caplog.at_level("WARNING", logger="chess_anti_engine.train.trainer"), \
            pytest.raises(SystemExit):
        lc0_control_train.main([
            "--config", str(config), "--shards", str(unlabelled),
            "--out-dir", str(tmp_path / "leak_warn"), "--steps", "1",
            "--batch-size", "4", "--device", "cpu", "--no-compile",
            "--allow-arch-drift", "--allow-invalid-control", "--allow-leak",
        ])
    warned = [r.getMessage() for r in caplog.records if "RAW GAME OUTCOME" in r.getMessage()]
    assert warned, "the production-path warning did not fire on a real leak"
    assert "0.6900 of the WDL target" in warned[0]
    assert "sf_wdl_frac leaked 0.6900" in warned[0]


def _cpu_trainer(config_path: Path) -> Any:
    """The real ``Trainer``, at toy width, on the CPU — no steps run."""
    from chess_anti_engine.model import build_model, model_config_from_flat_config
    from chess_anti_engine.train.trainer import Trainer, trainer_kwargs_from_config

    cfg = flatten_run_config_defaults(load_yaml_file(str(config_path)))
    kwargs = trainer_kwargs_from_config(cfg)
    kwargs["device"] = "cpu"
    kwargs["use_compile"] = False
    model_cfg = model_config_from_flat_config(cfg)
    return Trainer(build_model(model_cfg), model_config=model_cfg, **kwargs)


def _train_metrics(**overrides: Any) -> Any:
    from chess_anti_engine.train.trainer import TrainMetrics

    required = {
        field.name: 0.0 for field in dataclasses.fields(TrainMetrics)
        if field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
    }
    return TrainMetrics(**required, **overrides)


def test_the_leak_warning_reports_the_normalized_weight_not_the_raw_frac(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """⚑⚑ Codex 3794881264 — PRODUCTION CODE, and the reviewer's own example.

    ``compute_loss`` puts both fracs through ``normalize_value_blend_fracs``, so
    when they sum above 1 the APPLIED weights are smaller than the trainer
    attributes — and ``_warn_if_value_blend_leaks_to_outcome`` multiplied the
    label shortfall by the RAW attributes. With ``sf_wdl_frac 0.8`` +
    ``search_wdl_frac 0.8`` and 50% effective SF coverage the objective realizes a
    **0.25** leak and the warning reported **0.40** — 1.6x, enough to cross the
    0.01 incident bar on a leak the trained objective never had.

    The NUMBER is asserted, in both directions: 0.2500 present and 0.4000 absent.
    Asserting only "the helper was called" would pass on a version that
    normalized and then reported the raw product anyway.
    """
    trainer = _cpu_trainer(_tiny_config(tmp_path))
  # Exactly how production sets them: `_sync_trainer_weights` does
  # `setattr(trainer, wk, ...)` every iteration, which is also why the check
  # cannot live at construction time.
    trainer.sf_wdl_frac = 0.8
    trainer.search_wdl_frac = 0.8
    metrics = _train_metrics(
        train_steps_done=1, sf_wdl_effective_frac=0.5,
        search_wdl_effective_frac=1.0,
    )
    with caplog.at_level("WARNING", logger="chess_anti_engine.train.trainer"):
        trainer._warn_if_value_blend_leaks_to_outcome(metrics)
    warned = [r.getMessage() for r in caplog.records
              if "RAW GAME OUTCOME" in r.getMessage()]
    assert len(warned) == 1, "the leak is 0.25, well above the 0.01 bar"
    assert "0.2500 of the WDL target" in warned[0], warned[0]
    assert "sf_wdl_frac leaked 0.2500" in warned[0], warned[0]
    assert "0.4000" not in warned[0], (
        "0.40 is the RAW product sf_wdl_frac * (1 - coverage); the objective "
        "applies 0.25"
    )

  # And the number is the LOSS's arithmetic, not a second copy of it.
    from chess_anti_engine.train.losses import normalize_value_blend_fracs

    sf_frac, search_frac, _game = normalize_value_blend_fracs(0.8, 0.8)
    assert sf_frac * (1.0 - 0.5) + search_frac * (1.0 - 1.0) == pytest.approx(0.25)


def test_the_normalized_leak_warning_fires_through_the_real_training_step(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The same fix on the WIRED path, at a leak size the two versions disagree
    about: an over-subscribed 0.8/0.8 blend on rows with no ``sf_wdl`` realizes
    0.50 (the renormalised SF share) and the raw arithmetic reports 0.80.

    Deterministic regardless of batch composition, because SF coverage is 0 and
    search coverage is 1 for EVERY row in this corpus — no reliance on which
    rows the buffer happened to draw.
    """
    shards = _write_shards(tmp_path / "lc0_rows", list(range(16)))
    config = _tiny_config(
        tmp_path, train={"sf_wdl_frac": 0.8, "search_wdl_frac": 0.8},
    )
    with caplog.at_level("WARNING", logger="chess_anti_engine.train.trainer"), \
            pytest.raises(SystemExit):
        lc0_control_train.main([
            "--config", str(config), "--shards", str(shards),
            "--out-dir", str(tmp_path / "oversubscribed"), "--steps", "1",
            "--batch-size", "4", "--device", "cpu", "--no-compile",
            "--allow-arch-drift", "--allow-invalid-control", "--allow-leak",
        ])
    warned = [r.getMessage() for r in caplog.records
              if "RAW GAME OUTCOME" in r.getMessage()]
    assert warned, "the production-path warning did not fire on a real leak"
    assert "0.5000 of the WDL target" in warned[0], warned[0]
    assert "0.8000" not in warned[0], (
        "0.80 is the raw sum of the two attributes; compute_loss renormalises "
        "them to 0.5/0.5 before applying them"
    )


def test_launch_refuses_a_production_blend_without_allow_leak(tmp_path: Path) -> None:
    shards = _write_shards(tmp_path / "rows", list(range(8)))
  # ⚑ THE LIVE production blend (`sf 0.69 + search 0.31 = 1.00`, `game_frac`
  # 0.00), not `main`'s stale 0.50/0.20. On SF-labelled rows this is a clean
  # pass; on lc0 rows the 0.69 falls to the raw outcome. Using the stale pair
  # here would ALSO have tripped the outcome-borne bar, for the unrelated
  # reason that its intended `game_frac` is 0.30.
    config = _tiny_config(
        tmp_path, train={"sf_wdl_frac": 0.69, "search_wdl_frac": 0.31},
    )
    with pytest.raises(SystemExit, match="REFUSING TO LAUNCH"):
        lc0_control_train.main([
            "--config", str(config), "--shards", str(shards),
            "--out-dir", str(tmp_path / "x"), "--steps", "1", "--batch-size", "4",
            "--device", "cpu", "--no-compile", "--allow-arch-drift",
            "--allow-invalid-control",
        ])


def test_an_all_outcome_value_target_is_refused(tmp_path: Path) -> None:
    """⚑ Review F6 / Codex #2. ``sf=0`` and ``search=0`` leaks NOTHING and
    trains 100% of the value target on the raw one-hot outcome — the old gate
    passed it, printing ``leaked_to_outcome 0.00``."""
    lc0_rows = _write_shards(tmp_path / "lc0", list(range(8)))
    config = _tiny_config(tmp_path, train={"sf_wdl_frac": 0.0, "search_wdl_frac": 0.0})
    with pytest.raises(SystemExit) as excinfo:
        lc0_control_train.main([
            "--config", str(config), "--shards", str(lc0_rows),
            "--out-dir", str(tmp_path / "y"), "--steps", "1", "--batch-size", "4",
            "--device", "cpu", "--no-compile", "--allow-arch-drift",
            "--allow-invalid-control",
        ])
    assert "collapses onto the raw game outcome" in str(excinfo.value)


def test_the_realized_guard_also_catches_the_all_outcome_target(tmp_path: Path) -> None:
    """⚑ The ``outcome_borne_frac`` bar, REACHED THROUGH THE DRIVER.

    Launch guard 2 refuses ``sf=0 / search=0`` first, so the realized bar is a
    second line of defence and needs the launch gates open to be exercised at
    all. That is exactly what ``--allow-leak`` is for now — and this run leaks
    NOTHING (``leaked_to_outcome`` is 0.00, because ``sf_wdl_frac`` is 0) while
    training 100% of the value target on the raw outcome. The old guard passed
    it.
    """
    shards = _write_shards(tmp_path / "rows", list(range(16)))
    config = _tiny_config(tmp_path, train={"sf_wdl_frac": 0.0, "search_wdl_frac": 0.0})
    out = tmp_path / "all_outcome"
    with pytest.raises(SystemExit) as excinfo:
        lc0_control_train.main([
            "--config", str(config), "--shards", str(shards),
            "--out-dir", str(out), "--steps", "1", "--batch-size", "4",
            "--device", "cpu", "--no-compile", "--allow-arch-drift",
            "--allow-invalid-control", "--allow-leak",
        ])
    message = str(excinfo.value)
    assert "1.0000 of its mass on the RAW GAME OUTCOME" in message
    assert "leaked=0.0000" in message, (
        "the leak is genuinely zero here — that is why the leak bar could not "
        "see this state"
    )
    assert not (out / "checkpoint.pt").exists()


def test_a_corpus_with_no_search_label_is_refused_at_launch(tmp_path: Path) -> None:
    """⚑⚑ REVIEW F1 AT THE DRIVER — the LARGER term, on the launch path.

    The shipped control is `sf_wdl_frac: 0.0` / `search_wdl_frac: 0.70`, so on
    rows that carry no `has_search_wdl` **the whole value target is the raw
    game outcome** — and every guard passed, because the corpus preflight
    measured only the SF label. Today's converted corpus reads
    `has_search_wdl 204800/204800`, so nothing produced so far is wrong; a
    mixed `--shards` list or a v5 conversion lands here.
    """
    rows = _write_shards(tmp_path / "no_search", list(range(8)), with_search_wdl=False)
    with pytest.raises(SystemExit) as excinfo:
        lc0_control_train.main([
            "--config", str(_tiny_config(tmp_path)), "--shards", str(rows),
            "--out-dir", str(tmp_path / "ns"), "--steps", "1", "--batch-size", "4",
            "--device", "cpu", "--no-compile", "--allow-arch-drift",
            "--allow-invalid-control",
        ])
    message = str(excinfo.value)
    assert "NO search value label" in message
    assert "search_wdl label coverage" not in message, "that line is a print, not the refusal"


def test_a_partially_labelled_corpus_is_refused(tmp_path: Path) -> None:
    """⚑ Codex #2 at its own level: ``any()`` called a mixed corpus labelled.

    One SF-labelled row anywhere in ``--shards`` used to set ``have_sf=True``,
    which made the converter's gate return ``[]`` and waved the whole corpus
    through. Coverage is 2/10 here, and neither regime's reasoning applies.
    """
    lc0_rows = _write_shards(tmp_path / "lc0", list(range(8)))
    sf_rows = _write_shards(tmp_path / "sf", [100, 101], with_sf_wdl=True)
    with pytest.raises(SystemExit, match="PARTIALLY sf_wdl-labelled"):
        lc0_control_train.main([
            "--config", str(_tiny_config(tmp_path)),
            "--shards", str(lc0_rows), str(sf_rows),
            "--out-dir", str(tmp_path / "z"), "--steps", "1", "--batch-size", "4",
            "--device", "cpu", "--no-compile", "--allow-arch-drift",
            "--allow-invalid-control",
        ])


def test_the_realized_guard_is_judged_on_the_worst_step_not_the_first() -> None:
    """⚑ The reviewer's second mutant: ``worst`` -> ``readouts[0]``.

    A leak that begins partway through a run — a live edit, a controller push —
    is invisible to the first step and diluted to nothing by a mean over 800.
    """
    capture = lc0_control_train._LossCapture(
        rebuild_categorical=False, categorical_params=None,
    )
    capture.readouts = [
        value_blend_readout(sf_wdl_frac=0.0, search_wdl_frac=0.7,
                            sf_effective_rows=0.0, search_effective_rows=512.0,
                            batch_rows=512.0),
        value_blend_readout(sf_wdl_frac=0.5, search_wdl_frac=0.2,
                            sf_effective_rows=0.0, search_effective_rows=512.0,
                            batch_rows=512.0),
    ]
    assert capture.worst.leaked_to_outcome == pytest.approx(0.5)


def test_the_categorical_guard_is_judged_on_the_worst_step_not_the_first() -> None:
    """⚑ The MIRROR of the test above, and it was missing until review F1.

    ``worst`` had a purpose-built test; its sibling ``worst_categorical`` had
    none, so the mutant ``categorical_readouts[0]`` survived the whole suite.
    That is the "fix one axis, leave its sibling" shape — here inside the very
    commit that named it.

    The scenario is reachable with no code change: ``stage_shards`` symlinks N
    converted directories into one pool, so a MIXED-VINTAGE corpus is the normal
    case. Early batches draw from today's rows (no ``categorical_target`` column
    at all, so the rebuild is inert), and a later batch draws from a shard that
    carries the column with no ``sf_wdl`` — at which point the whole rebuilt
    categorical target is the raw one-hot game outcome. Judged on the first
    readout that reads 0.0 and the guard passes; judged on the worst it is 1.0
    and the run is refused with no checkpoint written.
    """
    capture = lc0_control_train._LossCapture(
        rebuild_categorical=True, categorical_params=None,
    )
    inert = CategoricalRebuildReadout(
        rebuild_enabled=True, blend_frac=0.69, search_blend_frac=0.0,
        target_present=False, sf_labelled_frac=0.0, search_labelled_frac=0.0,
        batch_rows=512.0,
    )
    leaking = CategoricalRebuildReadout(
        rebuild_enabled=True, blend_frac=0.69, search_blend_frac=0.0,
        target_present=True, sf_labelled_frac=0.0, search_labelled_frac=0.0,
        batch_rows=512.0,
    )
  # The premise of the test, asserted rather than assumed: the first readout
  # really is the benign one, so picking [0] really would report a clean run.
    assert inert.outcome_borne_frac == pytest.approx(0.0)
    capture.categorical_readouts = [inert, leaking]
    assert capture.worst_categorical is leaking
    assert capture.worst_categorical.outcome_borne_frac == pytest.approx(1.0)


def test_the_capture_records_the_kwargs_compute_loss_was_called_with() -> None:
    """A configured value is not an applied value — this is the applied one."""
    capture = lc0_control_train._LossCapture(
        rebuild_categorical=False, categorical_params=None,
    )
    capture.observe(
        {},
        {"sf_wdl_frac": 0.5, "search_wdl_frac": 0.2},
        {
            "sf_wdl_effective_rows": torch.tensor(0.0),
            "search_wdl_effective_rows": torch.tensor(8.0),
            "batch_rows": torch.tensor(8.0),
        },
    )
    assert capture.calls == 1
    assert capture.worst.sf_wdl_frac == pytest.approx(0.5)
    assert capture.worst.leaked_to_outcome == pytest.approx(0.5)


def test_the_architecture_guard_refuses_a_drifted_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard 0 must be the reason the run stops, before any GPU time is spent.

    ⚑ THE DRIFTED CONTROL IS NOW CONSTRUCTED, NOT ASSUMED. Until 2026-08-16
    this passed the SHIPPED config in and expected a refusal, because the
    shipped config was drifted — it predated the bt4heads promotion and could
    not carry keys that were not yet in `main`'s schema. That premise expired
    when PR #439 merged and the config was re-pointed, and a test that reads
    "the guard refuses" while actually asserting "the config is broken" stops
    testing the guard the moment the config is fixed.

    So: the shipped config PASSES (guard 0 is not a constant refusal), a copy
    with one `model:` key moved is REFUSED and names that key, and
    `allow_drift` downgrades the refusal to a banner.
    """
    monkeypatch.delenv("CHESS_LIVE_PRODUCTION_CONFIG", raising=False)
    assert not lc0_control_train.preflight_architecture(
        CONTROL, allow_drift=False,
    ).startswith("DRIFTED"), "the shipped control config must pass guard 0"

    raw = yaml.safe_load(CONTROL.read_text(encoding="utf-8"))
    raw["model"]["num_layers"] = int(raw["model"]["num_layers"]) + 1
    drifted = tmp_path / "drifted_control.yaml"
    drifted.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        lc0_control_train.preflight_architecture(drifted, allow_drift=False)
    assert "architecture is NOT production's" in str(excinfo.value)
    assert "num_layers" in str(excinfo.value)
    assert lc0_control_train.preflight_architecture(
        drifted, allow_drift=True,
    ).startswith("DRIFTED")


def test_the_trainer_guard_refuses_a_drifted_control_at_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑⚑ GUARD 0c AT THE ENTRY POINT — WITHOUT THIS ITS CALL SITE IS OPTIONAL.

    `tests/test_lc0_control_config.py` proves the library assert can refuse a
    drifted recipe; this proves the DRIVER calls it. Measured: with only the
    library tests, replacing `preflight_trainer(...)` in `main()` with a
    constant left the whole lc0-control suite green — this repo's signature
    defect, and precisely why `tests/test_lc0_control_drivers.py` exists.

    Also pins the `--allow-leak` split: that flag exists to restore
    production's value blend so the realized guard can be seen firing, which
    IS a trainer deviation, so it must downgrade this guard while
    `--allow-arch-drift` must not.
    """
    monkeypatch.delenv("CHESS_LIVE_PRODUCTION_CONFIG", raising=False)
    shards = _write_shards(tmp_path / "rows", list(range(8)))
    config = _tiny_config(tmp_path, train={"w_moves_left": 0.99})
    with pytest.raises(SystemExit) as excinfo:
        lc0_control_train.main([
            "--config", str(config), "--shards", str(shards),
            "--out-dir", str(tmp_path / "drift_run"), "--steps", "1",
            "--batch-size", "4", "--device", "cpu", "--no-compile",
            "--allow-arch-drift", "--allow-invalid-control",
        ])
    assert "TRAINER is NOT production's" in str(excinfo.value)
    assert "w_moves_left" in str(excinfo.value)
    assert not (tmp_path / "drift_run" / "checkpoint.pt").exists()

  # ...and `--allow-leak` downgrades it, or the leak demonstration that review
  # F2 made reachable becomes unreachable again from the other side.
    assert lc0_control_train.preflight_trainer(
        flatten_run_config_defaults(load_yaml_file(str(config))), allow_leak=True,
    ).startswith("DRIFTED")


def test_the_driver_is_what_reads_the_live_config_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ REVIEW F6, at the level where the env var is now honoured.

    ``assert_control_matches_live_architecture`` no longer reads the
    environment — the DRIVER resolves it and passes the path in. That moved a
    behaviour, so it needs a test at the new location, or "the check prefers
    the live file" becomes a claim about deleted code.

    Both directions: a live file the control matches PASSES and names the live
    file, and one it does not REFUSES.
    """
    config = _tiny_config(tmp_path)
    control_model = yaml.safe_load(config.read_text(encoding="utf-8"))["model"]

    live = tmp_path / "live_pbt2_small.yaml"
    live.write_text(yaml.safe_dump({"model": dict(control_model)}), encoding="utf-8")
    monkeypatch.setenv("CHESS_LIVE_PRODUCTION_CONFIG", str(live))
    provenance = lc0_control_train.preflight_architecture(config, allow_drift=False)
    assert str(live) in provenance, "the guard did not read the file the env named"

    moved = dict(control_model)
    moved["num_layers"] = int(moved["num_layers"]) + 1
    live.write_text(yaml.safe_dump({"model": moved}), encoding="utf-8")
    with pytest.raises(SystemExit, match="num_layers"):
        lc0_control_train.preflight_architecture(config, allow_drift=False)


def test_the_built_model_is_checked_against_the_pinned_parameter_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑⚑ REVIEW F4 AT THE DRIVER — guard 0b, which had no caller at all.

    The config-level check compares ``model:`` keys; this one compares the net
    those keys BUILT against the pinned count. Nothing passed ``model=`` before
    2026-08-16, so ``LIVE_ARCH_PIN['trainable_params_unique_storage']`` — the
    arch fix's own headline number — gated nothing on any path. (The count is
    also re-measured for real, against the live architecture, by
    ``test_lc0_control_arch.py::test_this_tree_builds_the_pinned_live_architecture``;
    this one is about the DRIVER passing ``model=`` at all.)

    The live file here matches the control exactly, so the section check
    passes and the ONLY thing that can refuse the run is the parameter count.
    """
    config = _tiny_config(tmp_path)
    control_model = yaml.safe_load(config.read_text(encoding="utf-8"))["model"]
    live = _live_stub(tmp_path / "live_pbt2_small.yaml", control_model, config)
    monkeypatch.setenv("CHESS_LIVE_PRODUCTION_CONFIG", str(live))
  # The pin's `model:` section IS the live file's here, so the pin's count
  # applies — and it is deliberately not the count this tree builds.
    monkeypatch.setitem(
        lc0_control_arch.LIVE_ARCH_PIN, "model", dict(control_model),
    )
    monkeypatch.setitem(
        lc0_control_arch.LIVE_ARCH_PIN, "trainable_params_unique_storage", 12_345,
    )
    shards = _write_shards(tmp_path / "rows", list(range(8)))
    with pytest.raises(SystemExit, match="trainable params"):
        lc0_control_train.main([
            "--config", str(config), "--shards", str(shards),
            "--out-dir", str(tmp_path / "pin_run"), "--steps", "1",
            "--batch-size", "4", "--device", "cpu", "--no-compile",
        ])


# ── lc0_control_train: the REPLAY BUFFER (launch guard 0d) ────────────────────


def _recording_buffer(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Capture the kwargs the driver passes AND the buffer that came out.

    ⚑ Read back off the CONSTRUCTED object, never off the config — the same
    discipline ``tune/trainable_init.py`` states for these exact keys: a config
    echo proves the yaml parsed, not that the buffer consumed it.
    """
    seen: dict[str, Any] = {}
    original = lc0_control_train.DiskReplayBuffer

    def recording(*args: Any, **kwargs: Any) -> Any:
        seen["passed"] = dict(kwargs)
        buf = original(*args, **kwargs)
        seen["realized"] = {
            "shuffle_cap": buf._shuffle_cap,
            "refresh_interval": buf._refresh_interval,
            "refresh_shards": buf._refresh_shards,
            "shard_recency_exponent": buf._shard_recency_exponent,
            "diff_focus_pol_scale": buf.diff_focus_pol_scale,
            "diff_focus_q_weight": buf.diff_focus_q_weight,
            "input_planes": buf._input_planes,
        }
        return buf

    monkeypatch.setattr(lc0_control_train, "DiskReplayBuffer", recording)
    return seen


def test_the_driver_builds_productions_buffer_not_the_constructor_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑⚑ CODEX 3791327310, at the level that decides it.

    The driver passed three buffer kwargs and took ``disk_buffer.py``'s
    CONSTRUCTOR DEFAULTS for the rest, so it sampled from a 20,000-row hot pool
    against production's 100,000 — and both existing pins passed, because
    ``trainer_kwargs_from_config`` reads none of these keys. Seven axes off, not
    the three the review named and not the two the driver's own comment claimed.

    ⚑ Asserted against ``LIVE_REPLAY_PIN``, not against literals: a test that
    hard-codes 100000 keeps passing after production moves, which is the failure
    mode the pin exists for.
    """
    seen = _recording_buffer(monkeypatch)
    shards = _write_shards(tmp_path / "rows", list(range(16)))
    out = tmp_path / "run"
    rc = lc0_control_train.main([
        "--config", str(_tiny_config(tmp_path)), "--shards", str(shards),
        "--out-dir", str(out), "--steps", "2", "--batch-size", "4",
        "--device", "cpu", "--no-compile", "--allow-arch-drift",
        "--allow-invalid-control",
    ])
    assert rc == 0
    live = lc0_control_replay.LIVE_REPLAY_PIN["kwargs"]
    realized = seen["realized"]
    for key in ("shuffle_cap", "refresh_interval", "refresh_shards",
                "diff_focus_pol_scale", "diff_focus_q_weight", "input_planes"):
        assert realized[key] == live[key], f"{key} is not production's"
  # And each of those is genuinely NOT the constructor default, or the assertion
  # above would pass on the defective code too.
    defaults = lc0_control_replay._buffer_defaults()
    for key in ("shuffle_cap", "refresh_interval", "refresh_shards",
                "diff_focus_pol_scale", "diff_focus_q_weight"):
        assert realized[key] != defaults[key], (
            f"{key} equals the DiskReplayBuffer default, so this test cannot "
            "distinguish the fix from the defect"
        )
  # The two DECLARED deviations are applied, and recorded as realized.
    assert realized["shard_recency_exponent"] == 0.0
    assert seen["passed"]["deterministic_refresh"] is True
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["realized_replay_after_guard"]["shuffle_cap"] == live["shuffle_cap"]
    assert summary["realized_replay_after_guard"]["shard_recency_exponent"] == 0.0
    assert "replay" in summary["replay_judged_against"].lower() or summary[
        "replay_judged_against"
    ], "the artifact must name what guard 0d judged against"


def test_the_replay_guard_refuses_a_drifted_control_at_launch(
    tmp_path: Path,
) -> None:
    """A hot pool 5x smaller than production's is a plateau of the RIG, and in
    the held-out slope it is indistinguishable from the plateau H_stack is
    about. So it is a launch REFUSAL, not a banner."""
    config = _tiny_config(tmp_path, tune={"shuffle_buffer_size": 20_000})
    shards = _write_shards(tmp_path / "rows", list(range(8)))
    with pytest.raises(SystemExit) as excinfo:
        lc0_control_train.main([
            "--config", str(config), "--shards", str(shards),
            "--out-dir", str(tmp_path / "drift"), "--steps", "1",
            "--batch-size", "4", "--device", "cpu", "--no-compile",
            "--allow-arch-drift", "--allow-invalid-control",
        ])
    message = str(excinfo.value)
    assert "REPLAY BUFFER is NOT production" in message
    assert "shuffle_cap: control=20000" in message


def test_allow_leak_downgrades_the_replay_guard_to_a_banner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Same split as guard 0c: the flag exists to let a deliberately-wrong run
    REACH the realized per-step guard, and a plumbing smoke that had to match
    production's 100,000-row shuffle pool is not a plumbing smoke."""
    config = _tiny_config(tmp_path, tune={"shuffle_buffer_size": 20_000})
    shards = _write_shards(tmp_path / "rows", list(range(8)))
    rc = lc0_control_train.main([
        "--config", str(config), "--shards", str(shards),
        "--out-dir", str(tmp_path / "leaky"), "--steps", "1",
        "--batch-size", "4", "--device", "cpu", "--no-compile",
        "--allow-arch-drift", "--allow-invalid-control", "--allow-leak",
    ])
    assert rc == 0
    assert "IGNORING launch guard" in capsys.readouterr().out


def test_every_disk_replay_buffer_kwarg_is_classified() -> None:
    """⚑⚑ THE ANTI-DRIFT HALF. A hand-listed set of buffer kwargs has exactly
    the defect it closes: the next knob added to ``DiskReplayBuffer`` is
    silently absent from the pin and every gate still reports green."""
    lc0_control_replay.assert_buffer_kwargs_are_classified()


def test_an_unclassified_buffer_kwarg_is_a_loud_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check above is vacuous until it is shown to fire. Dropping a mapping
    entry is the same event as ``DiskReplayBuffer`` gaining a parameter."""
    mapping = dict(lc0_control_replay.CONFIG_KWARGS)
    mapping.pop("shuffle_cap")
    monkeypatch.setattr(lc0_control_replay, "CONFIG_KWARGS", mapping)
    with pytest.raises(lc0_control_replay.ControlReplayDrift, match="shuffle_cap"):
        lc0_control_replay.assert_buffer_kwargs_are_classified()
  # ⚑ AND THROUGH THE FUNCTION EVERY CALLER USES. Testing the assert only where
  # it is called directly leaves "somebody deletes the call from
  # `replay_kwargs_signature`" invisible — a guard that is correct and no longer
  # reached, which is this repo's signature defect.
    with pytest.raises(lc0_control_replay.ControlReplayDrift, match="shuffle_cap"):
        lc0_control_replay.replay_kwargs_signature(
            flatten_run_config_defaults(load_yaml_file(str(CONTROL))),
        )


def test_a_driver_override_with_no_recorded_reason_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``LC0_REPLAY_DEVIATIONS`` is the mapping the guard reads AND the mapping
    the driver's overrides must be keys of. Without this, "the driver quietly
    overrides a fourth kwarg" is invisible to both."""
    monkeypatch.setattr(
        lc0_control_replay, "LC0_REPLAY_DEVIATIONS",
        {"shard_recency_exponent": "..."},
    )
    with pytest.raises(
        lc0_control_replay.ControlReplayDrift, match="deterministic_refresh",
    ):
        lc0_control_replay.apply_control_deviations(
            lc0_control_replay.LIVE_REPLAY_PIN["kwargs"],
        )


# ── lc0_control_train: the prereg's LAST-vs-MID-BUDGET pair ───────────────────


def test_the_run_writes_a_mid_budget_checkpoint_from_one_trajectory(
    tmp_path: Path,
) -> None:
    """⚑⚑ CODEX 3791327305. The prereg's PRIMARY yardstick is two slopes "both
    measured LAST vs MID-BUDGET"; the driver had one ``train_steps`` call and
    one ``trainer.save``, so it could not produce the deciding statistic at all.
    """
    shards = _write_shards(tmp_path / "rows", list(range(16)))
    out = tmp_path / "run"
    rc = lc0_control_train.main([
        "--config", str(_tiny_config(tmp_path)), "--shards", str(shards),
        "--out-dir", str(out), "--steps", "4", "--batch-size", "4",
        "--device", "cpu", "--no-compile", "--allow-arch-drift",
        "--allow-invalid-control",
    ])
    assert rc == 0
    assert (out / "checkpoint_mid.pt").is_file()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["mid_checkpoint"]["step"] == 2
    assert summary["mid_checkpoint"]["of_steps"] == 4
  # ⚑ NOT THE SAME WEIGHTS. A mid checkpoint saved before any step, or after
  # every step, would satisfy every assertion above and give `compare` a pair
  # that is discordant on zero rows.
    mid = torch.load(out / "checkpoint_mid.pt", map_location="cpu", weights_only=False)
    last = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert any(
        not torch.equal(mid["model"][k].float(), last["model"][k].float())
        for k in mid["model"] if isinstance(mid["model"][k], torch.Tensor)
    ), "mid and last are byte-identical, so the pair carries no trajectory"


def test_the_budget_runs_in_production_sized_windows_and_mid_lands_on_a_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑⚑ THE CADENCE CONFOUND — the arm's LR trajectory was not production's.

    With `lr_release_cycle_steps: 0`, `train_steps` derives the release cycle from
    ITS OWN `steps` argument, so ONE call of N steps held LR flat for 80% of the
    experiment and annealed once: MID at 50% of budget sat at full base LR and
    LAST sat at 0.1x after a full anneal, putting an anneal into the deciding
    MID->LAST slope that production — which calls `train_steps` at ~88 steps per
    iteration, i.e. a sawtooth — never places between two of its own checkpoints.

    Asserted on the CALLS, because that is where the cycle length comes from: the
    driver must issue an EVEN number of equal windows, and the mid checkpoint must
    land on a window boundary so MID and LAST sit at the same phase of the release
    cycle (both at the bottom).
    """
    shards = _write_shards(tmp_path / "rows", list(range(16)))
    calls: list[int] = []
    real = lc0_control_train.Trainer.train_steps

    def spy(self: Any, buf: Any, *, batch_size: int, steps: int) -> Any:
        calls.append(int(steps))
        return real(self, buf, batch_size=batch_size, steps=steps)

    monkeypatch.setattr(lc0_control_train.Trainer, "train_steps", spy)
    out = tmp_path / "cadence"
    assert lc0_control_train.main([
        "--config", str(_tiny_config(tmp_path)), "--shards", str(shards),
        "--out-dir", str(out), "--steps", "8", "--batch-size", "4",
        "--device", "cpu", "--no-compile", "--allow-arch-drift",
        "--allow-invalid-control",
        "--train-window-steps", "2",
    ]) == 0
    assert calls == [2, 2, 2, 2], (
        "the budget must run as four 2-step windows, not one 8-step call"
    )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["train_window_steps"] == 2
    assert summary["train_windows"] == 4
    assert summary["mid_checkpoint"]["step"] == 4, "a window boundary"
    assert summary["mid_checkpoint"]["step"] % 2 == 0
    assert not [p for p in summary["validity_problems"] if "LR cadence" in p], (
        summary["validity_problems"]
    )


def test_an_odd_window_count_trains_the_remainder_and_is_a_valid_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑⚑ REVIEW F1/F3 AND CODEX 3796113403, in one run.

    The previous revision required ``steps % (2 * window) == 0``. That relocated
    N1's outcome instead of fixing it — at the default window 88 only multiples of
    **176** were valid, i.e. 175 of every 176 budgets INCLUDING the arm's own
    20,000 — and it also ran ``steps // window`` FULL windows, silently discarding
    the remainder while ``summary["steps"]`` went on reporting the request.

    So this asserts the two OUTCOMES, not the arithmetic: an odd window count is a
    CLEAN cadence, and every requested step is actually trained (the final window
    is SHORT, which still ends at the LR release-cycle bottom — see
    ``test_the_mid_budget_split_would_have_re_ramped_the_lr`` for the measurement
    of that scheduler property).
    """
    shards = _write_shards(tmp_path / "rows", list(range(16)))
    calls: list[int] = []
    real = lc0_control_train.Trainer.train_steps

    def spy(self: Any, buf: Any, *, batch_size: int, steps: int) -> Any:
        calls.append(int(steps))
        return real(self, buf, batch_size=batch_size, steps=steps)

    monkeypatch.setattr(lc0_control_train.Trainer, "train_steps", spy)
    out = tmp_path / "odd_windows"
    assert lc0_control_train.main([
        "--config", str(_tiny_config(tmp_path)), "--shards", str(shards),
        "--out-dir", str(out), "--steps", "10", "--batch-size", "4",
        "--device", "cpu", "--no-compile", "--allow-arch-drift",
        "--allow-invalid-control", "--train-window-steps", "4",
    ]) == 0
    assert calls == [4, 4, 2], (
        "three windows, the last one SHORT: the remainder must be trained, not "
        f"discarded (got {calls})"
    )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["train_windows"] == 3
    assert summary["steps"] == summary["steps_realized"] == 10, (
        "the artifact's budget must be the budget that ran"
    )
    assert not [p for p in summary["validity_problems"] if "LR cadence" in p], (
        "an odd window count is not a cadence problem — it was one only under the "
        f"divisibility rule this fix removed: {summary['validity_problems']}"
    )


def test_the_default_window_is_productions_realized_cadence() -> None:
    """88 is a REALIZED production number (views-targeting sets steps/iteration),
    so it is a default here and not a yaml key — and a default nobody pins is a
    number that drifts."""
    assert lc0_control_train.PRODUCTION_TRAIN_WINDOW_STEPS == 88
  # The plan function is what the driver acts on, so pin ITS arithmetic.
    assert lc0_control_train.train_window_plan(steps=1760, window=88) == (88, 20, None)
  # ⚑ A CEILING window count and NO divisibility requirement: 20001 steps is 228
  # windows, the last of which is short, and it is a CLEAN plan.
    assert lc0_control_train.train_window_plan(steps=20001, window=88) == (
        88, 228, None,
    )
  # The one remaining problem: a budget with no INTERIOR boundary to put MID on.
    window, count, problem = lc0_control_train.train_window_plan(steps=100, window=88)
    assert (window, count) == (88, 2)
    assert problem is not None
    assert "less than two" in problem
    window, count, problem = lc0_control_train.train_window_plan(steps=4, window=88)
    assert (window, count) == (4, 1)
    assert problem is not None
    assert "less than two" in problem
  # ⚑ 2 x window is the smallest CLEAN budget, and it puts MID at exactly 0.5.
    assert lc0_control_train.train_window_plan(steps=176, window=88) == (88, 2, None)
    assert lc0_control_train.mid_step_on_window_boundary(
        steps=176, window=88, frac=0.5,
    ) == 88


def test_the_mid_budget_split_would_have_re_ramped_the_lr() -> None:
    """⚑⚑ WHY THE CHECKPOINT IS TAKEN FROM INSIDE ONE ``train_steps`` CALL.

    The obvious implementation — ``train_steps(N/2)`` twice, saving between —
    is wrong, and this is the measurement rather than the argument. With
    ``lr_schedule: sqrt_release`` and ``lr_release_cycle_steps: 0`` (what this
    arm and production both run) ``train_steps`` derives the release cycle from
    ITS OWN ``steps`` argument and feeds the scheduler ``local_step``, which
    restarts at 0 every call. Two half-calls therefore run the release ramp
    TWICE, so LAST and MID would sit on different LR trajectories and the slope
    between them would be part schedule artifact.
    """
    from chess_anti_engine.train.trainer import _SqrtReleaseLRScheduler

    class _Opt:
        def __init__(self) -> None:
            self.param_groups = [{"lr": 1.0}]

    scheduler = _SqrtReleaseLRScheduler(
        _Opt(), cycle_steps=0, release_start_frac=0.8, min_scale=0.1,
    )
    total = 100
    one_call = [
        scheduler._scale_for_window_step(i, cycle_steps=total) for i in range(total)
    ]
    split = [
        scheduler._scale_for_window_step(i, cycle_steps=total // 2)
        for _half in range(2) for i in range(total // 2)
    ]
    assert one_call != split, (
        "if these agreed, splitting the call would be a valid implementation"
    )
  # The mechanism, named: the split ramp has already released and RECOVERED by
  # the midpoint, which the single ramp has not.
    assert one_call[total // 2] == 1.0
    assert split[total // 2 - 1] < 1.0
    assert split[total // 2] == 1.0


def test_a_run_with_no_mid_checkpoint_is_not_a_valid_control(
    tmp_path: Path,
) -> None:
    """Half the deciding statistic, recorded as such. Otherwise `compare` would
    happily pair this run's LAST against some OTHER run's LAST."""
    shards = _write_shards(tmp_path / "rows", list(range(16)))
    out = tmp_path / "run"
    rc = lc0_control_train.main([
        "--config", str(_tiny_config(tmp_path)), "--shards", str(shards),
        "--out-dir", str(out), "--steps", "2", "--batch-size", "4",
        "--device", "cpu", "--no-compile", "--allow-arch-drift",
        "--allow-invalid-control",
        "--mid-checkpoint-frac", "0",
    ])
    assert rc == 0
    assert not (out / "checkpoint_mid.pt").exists()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["mid_checkpoint"] is None
    assert summary["valid_control"] is False
    assert any("mid-budget" in p for p in summary["validity_problems"])


def test_a_failed_realized_guard_leaves_no_mid_checkpoint_either(
    tmp_path: Path,
) -> None:
    """⚑ The mid checkpoint is written DURING the run, i.e. before the realized
    guard can have run. "No checkpoint written" has to stay literally true, or a
    refused run leaves a scorable artifact behind whose metadata says nothing
    about the refusal."""
    shards = _write_shards(tmp_path / "rows", list(range(16)))
    config = _tiny_config(
        tmp_path, train={"sf_wdl_frac": 0.69, "search_wdl_frac": 0.31},
    )
    out = tmp_path / "leak_run"
    with pytest.raises(SystemExit):
        lc0_control_train.main([
            "--config", str(config), "--shards", str(shards),
            "--out-dir", str(out), "--steps", "4", "--batch-size", "4",
            "--device", "cpu", "--no-compile", "--allow-arch-drift",
            "--allow-invalid-control", "--allow-leak",
        ])
    assert not (out / "checkpoint.pt").exists()
    assert not (out / "checkpoint_mid.pt").exists()


def _run(tmp_path: Path, out: Path, shards: Path, *extra: str) -> int:
    """A clean 4-step control run, for the tests that vary ONE argument."""
    return lc0_control_train.main([
        "--config", str(_tiny_config(tmp_path)), "--shards", str(shards),
        "--out-dir", str(out), "--steps", "4", "--batch-size", "4",
        "--device", "cpu", "--no-compile", "--allow-arch-drift",
        "--allow-invalid-control", *extra,
    ])


def test_a_mid_checkpoint_at_a_custom_fraction_is_not_the_preregistered_one(
    tmp_path: Path,
) -> None:
    """⚑ Codex 3794881260. ANY positive fraction is clamped to an interior step,
    so ``--mid-checkpoint-frac 0.99`` wrote a checkpoint, satisfied the
    "no mid-budget checkpoint" entry, and left the run reading VALID — while the
    LAST-vs-99%-budget contrast it supports measures the final 1% of training and
    would be presented as the prereg's LAST vs MID-BUDGET slope.

    ⚑ THE OTHER GATE IS OFF ON PURPOSE. A mid checkpoint IS written here (step 3
    of 4), so the pre-existing "no mid-budget checkpoint" problem cannot fire and
    cannot make this test pass with the new clause reverted — asserted, not
    assumed, by requiring that entry's ABSENCE.
    """
    shards = _write_shards(tmp_path / "rows", list(range(16)))
    out = tmp_path / "late_mid"
    assert _run(tmp_path, out, shards, "--mid-checkpoint-frac", "0.99") == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["mid_checkpoint"]["step"] == 3
    problems = summary["validity_problems"]
    assert not [p for p in problems if "no mid-budget checkpoint" in p], (
        "the backstop must be silent here, or this test cannot see its own fix"
    )
    assert [p for p in problems
            if "at step 3 of 4 = 0.7500 of the budget, more than 0.01 from the "
            "preregistered 0.5" in p], problems

    baseline = tmp_path / "prereg_mid"
    assert _run(tmp_path, baseline, shards) == 0
    prereg = json.loads((baseline / "summary.json").read_text(encoding="utf-8"))
    assert prereg["mid_checkpoint"]["step"] == 2
    assert not [p for p in prereg["validity_problems"]
                if "mid-checkpoint-frac" in p], (
        "the preregistered 0.5 must NOT be recorded as a deviation"
    )


def test_the_mid_budget_guard_judges_the_REALIZED_step_not_the_knob(
    tmp_path: Path,
) -> None:
    """⚑ The reviewer's violating input: ``--steps 3 --mid-checkpoint-frac 0.5``.

    ``mid_step = int(frac * steps)`` clamped to the interior, so the knob being
    the preregistered 0.5 does NOT mean the checkpoint landed at half the budget —
    here it lands at step 1 of 3, **33.3%**, and the first version of this guard
    recorded nothing because it compared the KNOB. That is this repo's own rule
    ("a configured value is not an applied value") broken inside the guard added
    to enforce a preregistered value.

    ⚑ The backstop is disarmed by construction: a mid checkpoint IS written, so
    the "no mid-budget checkpoint" entry cannot be what fires — asserted below.
    """
    shards = _write_shards(tmp_path / "rows", list(range(16)))
    out = tmp_path / "odd_budget"
    assert lc0_control_train.main([
        "--config", str(_tiny_config(tmp_path)), "--shards", str(shards),
        "--out-dir", str(out), "--steps", "3", "--batch-size", "4",
        "--device", "cpu", "--no-compile", "--allow-arch-drift",
        "--allow-invalid-control",
        "--mid-checkpoint-frac", "0.5",
    ]) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["mid_checkpoint"]["step"] == 1, "1 of 3 = 33.3% of the budget"
    problems = summary["validity_problems"]
    assert not [p for p in problems if "no mid-budget checkpoint" in p], (
        "the backstop must be silent here, or this test cannot see its own fix"
    )
    assert [p for p in problems
            if "at step 1 of 3 = 0.3333 of the budget" in p], problems


def _mid_frac_is_recorded(*, saved_at_step: int, steps: int) -> bool:
    """The driver's own predicate, evaluated without running the driver.

    ⚑ The reviewer's failing budgets are 1001 and 20001 steps, which cannot be
    RUN in a unit test — so the arithmetic is checked here and the WIRING is
    checked by the driver-level tests either side of this one. Both halves are
    needed: an arithmetic-only test cannot see a predicate that stopped being
    called, and a driver-only test cannot reach 20001 steps.
    """
  # ⚑ THE DRIVER'S OWN PREDICATE, not a restatement of it: a test that reproduces
  # `abs(realized - 0.5) > tol` in the test file cannot see the comparison
  # operator change in the driver, which is the exact defect (exact `!=`) this
  # table exists to catch.
    return lc0_control_train.mid_fraction_deviates(
        saved_at_step=saved_at_step, steps=steps,
    )


@pytest.mark.parametrize(
    ("steps", "recorded"),
    [
  # ⚑⚑ THE REVIEWER'S ARITHMETIC TABLE, verbatim. Every ODD budget was recorded
  # INVALID by the previous exact `!=`, unwaivably, so a day of GPU at 20001 steps
  # became permanently unquotable over a 0.0025% discrepancy.
        (1000, False),
        (1001, False),
        (20000, False),
        (20001, False),
        (87, False),
  # ...and the deliberate misplacements still fire.
        (3, True),
        (5, True),
    ],
)
def test_the_mid_fraction_tolerance_admits_truncation_and_still_catches_misplacement(
    steps: int, recorded: bool,
) -> None:
    mid = min(max(int(0.5 * steps), 1), max(steps - 1, 0))
    assert _mid_frac_is_recorded(saved_at_step=mid, steps=steps) is recorded, (
        f"steps={steps} mid={mid} realized={mid / steps:.6f}"
    )


def test_the_mid_fraction_floor_is_named_by_the_refusal_it_causes(
    tmp_path: Path,
) -> None:
    """⚑⚑ THE HONEST VERSION OF A CLAIM THIS TEST USED TO MAKE FALSELY.

    Before MID was snapped to a window boundary, the tolerance was tighter than
    integer truncation only below 50 steps — every such run was already
    disqualified by `warmup_steps`, so the entry could never be the SOLE reason a
    run was invalid, and this test asserted exactly that. Snapping changes it: a
    boundary can miss 0.5 by up to half a window, so the tolerance is unreachable
    below `min_budget_for_mid_tolerance(window)` — 4400 steps at the default 88,
    which is ABOVE production's 1000-step warmup. Budgets in between ARE refused
    by this entry alone.

    That is a deliberate trade (the alternative is widening the prereg's mid point
    for budgets nobody needs — the arm runs 17,600-20,000), and the requirement it
    creates is that the refusal be ACTIONABLE: it must name the floor rather than
    reporting an arithmetic mismatch the operator cannot act on. Asserted here,
    both directions.
    """
    warmup = int(flatten_run_config_defaults(
        load_yaml_file(str(_tiny_config(tmp_path))),
    )["warmup_steps"])
    window = lc0_control_train.PRODUCTION_TRAIN_WINDOW_STEPS
    floor = lc0_control_train.min_budget_for_mid_tolerance(window)
    assert floor == 4400, "half a window over the 0.01 tolerance, at window 88"
    assert floor > warmup, (
        "if the floor were below warmup this entry could not be a sole reason and "
        "the message below would be unnecessary"
    )
  # A budget in the gap: refused, and the message names the floor.
    mid = lc0_control_train.mid_step_on_window_boundary(
        steps=2000, window=window, frac=0.5,
    )
    problems = lc0_control_train.control_validity_problems(
        allow_leak=False, allow_arch_drift=False, has_purity_receipt=True,
        steps=2000, warmup_steps=warmup, mid_saved_at_step=mid,
        mid_checkpoint_frac=0.5, window_steps=window, cadence_problem=None,
        device="cuda", configured_device="cuda", batch_size=512,
        configured_batch_size=512, live_config_unread=False, live_game_frac=0.0,
    )
    assert len(problems) == 1, problems
    assert f"unreachable below {floor} steps" in problems[0], problems
  # And at the floor the entry cannot fire at all.
    for steps in (floor, floor + 1, floor + window, 17600, 20000, 20001):
        boundary = lc0_control_train.mid_step_on_window_boundary(
            steps=steps, window=window, frac=0.5,
        )
        assert not _mid_frac_is_recorded(saved_at_step=boundary, steps=steps), (
            f"steps={steps} mid={boundary} realized={boundary / steps:.6f}"
        )


def test_the_arms_realistic_budget_is_a_valid_control_at_the_default_window(
) -> None:
    """⚑⚑ REVIEW F1 — THE OUTCOME GUARANTEE, AT THE DEFAULT WINDOW.

    The guarantee is not "the arithmetic is right", it is **a realistic budget the
    operator would actually type is accepted**. The test this replaces asserted
    the arithmetic and then dodged the guarantee: it passed
    `--train-window-steps 101`, so it never exercised the DEFAULT window, where
    every budget that was not a multiple of 176 was invalid — including the arm's
    own. Its name claimed a property it structurally could not check.

    So: the DEFAULT window (read off the constant, never a literal), the prereg's
    own budgets, and `control_validity_problems` — the ONE list `summary.json` and
    the launch refusal both come from — asserted EMPTY.
    """
    window = lc0_control_train.PRODUCTION_TRAIN_WINDOW_STEPS
    for steps in (17600, 20000, 20001, 4400, 100000):
        plan_window, n_windows, cadence = lc0_control_train.train_window_plan(
            steps=steps, window=window,
        )
        assert (plan_window, cadence) == (window, None), (steps, cadence)
        assert n_windows * window >= steps, "no step may be dropped from the plan"
        mid = lc0_control_train.mid_step_on_window_boundary(
            steps=steps, window=plan_window, frac=0.5,
        )
        assert mid % window == 0, (steps, mid)
        assert 0 < mid < steps, (steps, mid)
        problems = lc0_control_train.control_validity_problems(
            allow_leak=False, allow_arch_drift=False, has_purity_receipt=True,
            steps=steps, warmup_steps=1000, mid_saved_at_step=mid,
            mid_checkpoint_frac=0.5, window_steps=plan_window,
            cadence_problem=cadence, device="cuda", configured_device="cuda",
            batch_size=512, configured_batch_size=512, live_config_unread=False,
            live_game_frac=0.0,
        )
        assert problems == [], (steps, problems)


def test_the_mid_point_is_snapped_to_a_window_boundary_at_any_fraction() -> None:
    """⚑⚑ REVIEW F2 — the reviewer's own violating input, and the property that
    matters rather than the proxy that bounded it.

    Two guards bounded PROXIES of "MID and LAST sit at comparable LR": the
    realized fraction (±0.01) and the budget's divisibility. At 17,600 steps ±0.01
    is ±2 whole 88-step windows, so `--mid-checkpoint-frac 0.5028` put MID at LR
    scale 1.0 against LAST at 0.1 with both guards SILENT and
    `valid_control: true`. Snapping removes the state: whatever fraction is asked
    for, MID lands at the end of a window, i.e. at the release-cycle bottom, which
    is where LAST is.
    """
    window = lc0_control_train.PRODUCTION_TRAIN_WINDOW_STEPS
    for frac in (0.5, 0.5028, 0.4972, 0.51, 0.49):
        mid = lc0_control_train.mid_step_on_window_boundary(
            steps=17600, window=window, frac=frac,
        )
        assert mid % window == 0, (frac, mid)
  # The reviewer's input specifically: it used to land 1 step past a boundary.
    assert lc0_control_train.mid_step_on_window_boundary(
        steps=17600, window=window, frac=0.5028,
    ) == 8888
    assert int(0.5028 * 17600) % window != 0, (
        "if the UNSNAPPED value were already on a boundary this test would pass "
        "with the snapping reverted"
    )
  # ⚑ And the entry that reports it, in case any future path bypasses the snap.
    problems = lc0_control_train.control_validity_problems(
        allow_leak=False, allow_arch_drift=False, has_purity_receipt=True,
        steps=17600, warmup_steps=1000, mid_saved_at_step=8849,
        mid_checkpoint_frac=0.5028, window_steps=window, cadence_problem=None,
        device="cuda", configured_device="cuda", batch_size=512,
        configured_batch_size=512, live_config_unread=False, live_game_frac=0.0,
    )
    assert [p for p in problems if "NOT a multiple of the 88-step train window" in p], (
        problems
    )
  # ...and it does NOT fire on the snapped step, or it would refuse every run.
    assert not [
        p for p in lc0_control_train.control_validity_problems(
            allow_leak=False, allow_arch_drift=False, has_purity_receipt=True,
            steps=17600, warmup_steps=1000, mid_saved_at_step=8888,
            mid_checkpoint_frac=0.5028, window_steps=window, cadence_problem=None,
            device="cuda", configured_device="cuda", batch_size=512,
            configured_batch_size=512, live_config_unread=False,
            live_game_frac=0.0,
        ) if "train window" in p
    ]


def test_the_run_is_refused_at_launch_before_a_single_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑⚑ REVIEW F1 — A DAY OF GPU MUST NOT BUY AN UNQUOTABLE ARTIFACT.

    Every `validity_problems` entry is knowable from the arguments, and TWICE a
    full run ended `valid_control: false` — which `compare` refuses with NO
    waiver — for something its own command line already contained. The refusal is
    therefore at LAUNCH, and the proof is that `train_steps` is never called and
    no checkpoint is written; the escape is an explicit flag, whose effect is
    checked in the same test so a refusal that always fires cannot pass it.
    """
    shards = _write_shards(tmp_path / "rows", list(range(16)))
    calls: list[int] = []
    real = lc0_control_train.Trainer.train_steps

    def spy(self: Any, buf: Any, *, batch_size: int, steps: int) -> Any:
        calls.append(int(steps))
        return real(self, buf, batch_size=batch_size, steps=steps)

    monkeypatch.setattr(lc0_control_train.Trainer, "train_steps", spy)
    refused = tmp_path / "refused"
    with pytest.raises(SystemExit) as excinfo:
        lc0_control_train.main([
            "--config", str(_tiny_config(tmp_path)), "--shards", str(shards),
            "--out-dir", str(refused), "--steps", "4", "--batch-size", "4",
            "--device", "cpu", "--no-compile", "--allow-arch-drift",
        ])
    message = str(excinfo.value)
    assert "REFUSING TO LAUNCH" in message
    assert "--allow-arch-drift" in message
    assert "NO WAIVER" in message, "the refusal must say why finishing is useless"
    assert calls == [], "the refusal must precede the first optimizer step"
    assert not (refused / "checkpoint.pt").exists()
    assert not (refused / "checkpoint_mid.pt").exists()
    assert not (refused / "summary.json").exists()
  # And the flag is what a plumbing smoke passes — same arguments, rc 0.
    allowed = tmp_path / "allowed"
    assert lc0_control_train.main([
        "--config", str(_tiny_config(tmp_path)), "--shards", str(shards),
        "--out-dir", str(allowed), "--steps", "4", "--batch-size", "4",
        "--device", "cpu", "--no-compile", "--allow-arch-drift",
        "--allow-invalid-control",
    ]) == 0
    assert calls, "the flag must let the run reach training"
    summary = json.loads((allowed / "summary.json").read_text(encoding="utf-8"))
    assert summary["valid_control"] is False, "the flag does not make it valid"


def test_the_launch_refusal_and_the_artifact_list_the_same_problems(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑ ONE implementation, called twice — the launch check and the artifact must
    not be able to disagree. The only input that differs between the two calls is
    the mid step (predicted, then realized), so the banner printed at launch is
    compared with the list banked at the end of the SAME run."""
    shards = _write_shards(tmp_path / "rows", list(range(16)))
    out = tmp_path / "same_list"
    assert _run(tmp_path, out, shards) == 0
    printed = capsys.readouterr().out
    banner = printed.split("--allow-invalid-control: THIS RUN IS NOT A VALID CONTROL")[1]
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["validity_problems"], "this smoke must have problems to compare"
  # ⚑ The JOINED list, in order — not a per-entry `in` check. A per-entry check
  # passes when the launch list carries entries the artifact does not, which is
  # exactly the divergence a second copy of the list would produce.
    assert "\n  ".join(summary["validity_problems"]) in banner, (
        f"the launch banner and the artifact must be ONE list.\nlaunch: {banner}\n"
        f"artifact: {summary['validity_problems']}"
    )


def test_the_outcome_bar_is_derived_from_the_live_recipe_not_a_constant() -> None:
    """⚑⚑ THREAD 3795733617. `PRODUCTION_GAME_FRAC` is a hand-written 0.0, and the
    two keys it is derived from (`sf_wdl_frac`, `search_wdl_frac`) are exactly the
    two `LC0_TRAINER_DEVIATIONS` makes guard 0c IGNORE. So a production blend of
    0.50/0.20 leaves 0.30 on the raw game outcome, guard 0c stays silent by
    construction, and the arm would be stamped valid while no longer training
    production's objective.
    """
    from chess_anti_engine.eval.lc0_control_trainer import (
        LIVE_TRAINER_PIN,
        live_production_game_frac,
    )
    from chess_anti_engine.train.value_blend_guard import PRODUCTION_GAME_FRAC

  # The pin's own blend, through the loss's own normaliser.
    live, provenance = live_production_game_frac()
    assert live == pytest.approx(PRODUCTION_GAME_FRAC), (
        f"the constant is stale against {provenance}"
    )
  # A moved production blend: derived, not constant.
    moved = json.loads(json.dumps({
        "recorded": LIVE_TRAINER_PIN["recorded"],
        "kwargs": {**LIVE_TRAINER_PIN["kwargs"],
                   "sf_wdl_frac": 0.5, "search_wdl_frac": 0.2},
    }))
    shifted, _ = live_production_game_frac(pin=moved)
    assert shifted == pytest.approx(0.3)
  # ...and the run that keeps holding 0.0 is refused for it, by name.
    problems = lc0_control_train.control_validity_problems(
        allow_leak=False, allow_arch_drift=False, has_purity_receipt=True,
        steps=20000, warmup_steps=1000, mid_saved_at_step=10032,
        mid_checkpoint_frac=0.5, window_steps=88, cadence_problem=None,
        device="cuda", configured_device="cuda", batch_size=512,
        configured_batch_size=512, live_config_unread=False,
        live_game_frac=shifted,
    )
    assert [p for p in problems if "leaves game_frac=0.3000" in p], problems


def test_a_non_production_batch_size_is_recorded_as_a_validity_problem(
    tmp_path: Path,
) -> None:
    """⚑ Codex 3794881253. ``--batch-size`` is assigned after every preflight and
    ``batch_size`` is not a trainer kwarg, so guard 0c is structurally blind to
    it: a run at batch 2 against the configured 4 changed the examples per step —
    the gradient-noise regime — and still stamped ``valid_control: true``.

    Both directions, because only the pair kills the mutant: the deviating run
    records it AND the matching run does not.
    """
    shards = _write_shards(tmp_path / "rows", list(range(16)))
    deviating = tmp_path / "small_batch"
    assert lc0_control_train.main([
        "--config", str(_tiny_config(tmp_path)), "--shards", str(shards),
        "--out-dir", str(deviating), "--steps", "4", "--batch-size", "2",
        "--device", "cpu", "--no-compile", "--allow-arch-drift",
        "--allow-invalid-control",
    ]) == 0
    summary = json.loads((deviating / "summary.json").read_text(encoding="utf-8"))
    assert summary["batch_size"] == 2
    assert summary["configured_batch_size"] == 4
    assert [p for p in summary["validity_problems"]
            if "--batch-size 2 is not the configured 4" in p], (
        summary["validity_problems"]
    )

    matching = tmp_path / "config_batch"
    assert _run(tmp_path, matching, shards) == 0
    clean = json.loads((matching / "summary.json").read_text(encoding="utf-8"))
    assert clean["batch_size"] == clean["configured_batch_size"] == 4
    assert not [p for p in clean["validity_problems"] if "--batch-size" in p]


def test_a_failed_rerun_cannot_destroy_a_previous_runs_mid_checkpoint(
    tmp_path: Path,
) -> None:
    """⚑⚑ N2 — THE ONLY ITEM IN THIS REVIEW THAT DESTROYED DATA.

    `--out-dir` was created with `mkdir(exist_ok=True)`, so a rerun shared the
    directory with a completed run. When the rerun then failed its realized
    value-blend guard, its cleanup deleted `checkpoint_mid.pt` — scoped only by
    "did THIS run save a mid checkpoint", which was TRUE because run 2 had just
    written over run 1's file — so a COMPLETED run's half of the deciding
    statistic was irreversibly gone while its `summary.json` went on banking that
    file's path, step and sha256 and its `checkpoint.pt` still verified as
    `role: last`. In production that file is a day of GPU.

    The assertion the reviewer asked for is the third one: **run 1's MID
    checkpoint still exists, byte-for-byte, after the failed rerun.**
    """
    shards = _write_shards(tmp_path / "rows", list(range(16)))
    out = tmp_path / "shared"
    assert _run(tmp_path, out, shards) == 0
    mid = out / "checkpoint_mid.pt"
    assert mid.is_file()
    before = lc0_control_eval.sha256_file(mid)
    summary_before = (out / "summary.json").read_text(encoding="utf-8")

  # The rerun that used to destroy it: same --out-dir, a corpus that leaks, and
  # --allow-leak so it reaches the realized guard and fails there.
    leaky = _write_shards(tmp_path / "leak_rows", list(range(16)))
    config = _tiny_config(
        tmp_path, train={"sf_wdl_frac": 0.69, "search_wdl_frac": 0.31},
    )
    with pytest.raises(SystemExit) as excinfo:
        lc0_control_train.main([
            "--config", str(config), "--shards", str(leaky),
            "--out-dir", str(out), "--steps", "4", "--batch-size", "4",
            "--device", "cpu", "--no-compile", "--allow-arch-drift",
            "--allow-invalid-control", "--allow-leak",
        ])
  # ⚑ THE DATA ASSERTION FIRST, deliberately. If the refusal is removed the rerun
  # still raises (its realized guard fires), so a test that checked the MESSAGE
  # first would fail on the message and never reach the question that matters —
  # whether run 1's checkpoint is still on disk.
    assert mid.is_file(), "the previous run's MID checkpoint was DESTROYED"
    assert lc0_control_eval.sha256_file(mid) == before
    assert (out / "summary.json").read_text(encoding="utf-8") == summary_before
    assert "already holds" in str(excinfo.value), str(excinfo.value)
    assert "checkpoint_mid.pt" in str(excinfo.value)

  # ...and the escape RENAMES rather than deleting, so nothing is lost either way.
    assert lc0_control_train.main([
        "--config", str(_tiny_config(tmp_path)), "--shards", str(shards),
        "--out-dir", str(out), "--steps", "4", "--batch-size", "4",
        "--device", "cpu", "--no-compile", "--allow-arch-drift",
        "--allow-invalid-control",
        "--move-existing-aside",
    ]) == 0
    superseded = sorted(tmp_path.glob("shared.superseded_*"))
    assert len(superseded) == 1, [p.name for p in tmp_path.iterdir()]
    assert lc0_control_eval.sha256_file(
        superseded[0] / "checkpoint_mid.pt",
    ) == before, "the moved-aside run must be recoverable, not deleted"


def test_a_shard_directory_named_twice_is_refused_before_anything_is_staged(
    tmp_path: Path,
) -> None:
    """⚑ Codex 3794881271. Each occurrence was staged under its own
    ``shard_NNNNNN.zarr`` symlink, so the buffer drew that hour twice as often —
    and nothing downstream could see it: the coverage preflight SUMS over the
    list (its ratios stay right) and ``purity_receipt_problems`` compares SETS.

    ⚑ RESOLVED paths, so a second spelling of the same directory is caught too,
    and BEFORE staging — the assertion on the staging directory is what fails if
    the check is moved below it.
    """
    shards = _write_shards(tmp_path / "rows", list(range(16)))
    out = tmp_path / "dup"
    with pytest.raises(SystemExit) as excinfo:
        lc0_control_train.main([
            "--config", str(_tiny_config(tmp_path)), "--shards",
            str(shards), f"{shards}/.",
            "--out-dir", str(out), "--steps", "1", "--batch-size", "4",
            "--device", "cpu", "--no-compile", "--allow-arch-drift",
            "--allow-invalid-control",
        ])
    message = str(excinfo.value)
    assert "named more than once" in message
    assert str(shards.resolve()) in message
    assert not (out / "staged_shards").exists(), (
        "the refusal must precede staging, or the oversampled farm already exists"
    )

  # ...and the same corpus named ONCE runs, so this is about the duplicate and
  # not about the directory.
    assert _run(tmp_path, tmp_path / "single", shards) == 0


# ── lc0_control_heldout ───────────────────────────────────────────────────────


def _freeze(
    tmp_path: Path, shards: Path, *, sample: int, sources: list[Path] | None = None,
    allow_source_selection: bool = True,
) -> tuple[int, Path]:
    """⚑ ``--allow-source-selection`` by default, and that is a statement.

    The prereg's held-out population is the LAST SIX hourly tars, and `freeze`
    now refuses any other number of source directories unless the deviation is
    declared. Every fixture here builds ONE synthetic hour, so every fixture is
    declaring it; the tests that are ABOUT the six-source gate pass
    ``allow_source_selection=False`` or their own ``sources`` list.
    """
    out = tmp_path / "frozen.json"
    argv = ["freeze", "--out", str(out), "--sample", str(sample), "--seed", "0",
            "--shards", *[str(s) for s in (sources or [shards])]]
    if allow_source_selection:
        argv.append("--allow-source-selection")
    return lc0_control_heldout.main(argv), out


def test_freeze_refuses_a_pool_smaller_than_the_requested_sample(
    tmp_path: Path,
) -> None:
    """⚑ Codex #1. It used to write the short set, print a sha256 and exit 0 —
    and every downstream threshold is derived at n=100,000."""
    shards = _write_shards(tmp_path / "held", list(range(10)))
    rc, out = _freeze(tmp_path, shards, sample=100)
    assert rc == 1
    assert not out.exists(), "an underpowered frozen set must not exist on disk"


def test_freeze_writes_the_artifact_when_the_pool_is_large_enough(
    tmp_path: Path,
) -> None:
    shards = _write_shards(tmp_path / "held", list(range(10)))
    rc, out = _freeze(tmp_path, shards, sample=10)
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["row_ids"]) == len(payload["input_ids"]) == 10


def test_freeze_refuses_a_source_selection_that_is_not_the_preregistered_six(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑ Codex 3794881273. The prereg's held-out population is "the LAST 6 hourly
    tars by wall-clock"; ``freeze`` gated the row COUNT at 100,000 and accepted
    any number of source directories, so a set built from one hour — different
    temporal correlation, one net generation of a correlated stream — wrote a
    successful artifact and flowed through purity, scoring and comparison with no
    marker at all.

    ⚑ The row-count gate is SATISFIED here (10 rows requested, 10 supplied), so
    the pre-existing refusal cannot be what fails this run: the two gates are
    tested one at a time.
    """
    shards = _write_shards(tmp_path / "held", list(range(10)))
    rc, out = _freeze(tmp_path, shards, sample=10, allow_source_selection=False)
    assert rc == 1
    assert not out.exists(), (
        "an artifact from a non-preregistered population must not exist "
        "unstamped on disk"
    )
    err = capsys.readouterr().err
    assert "1 distinct source directory" in err
    assert "preregistered 6 hourly tars" in err


def test_freeze_counts_distinct_resolved_sources_not_occurrences(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑ The reviewer's violating input: ``freeze --shards h h h h h h``.

    ONE hour named six times reached the population gate as "6 sources" and
    PASSED it — the exact input `_stratified_sample`'s docstring is written about
    ("a single hour is one net generation's worth of a correlated stream") — and
    was stopped only by the UNRELATED duplicate-row-id refusal. This wave added
    `duplicate_resolved_dirs` for the same hazard on `--shards` in the same commit
    and did not reuse it here; it is now the shared implementation.

    ⚑ A repeat is refused with NO flag: `--allow-source-selection` declares a
    different NUMBER of hours, which is a legitimate smoke choice, while naming
    one hour twice is not a population anybody chose. Asserted below, so the
    refusal cannot be mistaken for the cardinality gate.
    """
    hour = _write_shards(tmp_path / "h", list(range(10)))
    out = tmp_path / "frozen.json"
    argv = ["freeze", "--out", str(out), "--sample", "10", "--seed", "0",
            "--shards", *[str(hour)] * 6]
    assert lc0_control_heldout.main(argv) == 1
    err = capsys.readouterr().err
    assert "named more than once" in err
    assert str(hour.resolve()) in err
    assert "duplicate row ids" not in err, (
        "the population gate must fire, not the unrelated row-id refusal"
    )
    assert not out.exists()
  # ...and the declared-cardinality flag does NOT clear it.
    assert lc0_control_heldout.main([*argv, "--allow-source-selection"]) == 1
    assert "named more than once" in capsys.readouterr().err


def test_a_frozen_set_banks_resolved_source_paths_not_basenames(
    tmp_path: Path,
) -> None:
    """Six genuinely distinct hours all called ``h`` banked ``["h"] * 6``, from
    which nobody can audit the prereg's actual claim — the LAST 6 hourly tars BY
    WALL-CLOCK. The distinctness gate needs the same field."""
    sources = [
        _write_shards(tmp_path / f"hour{i}" / "h", list(range(i * 10, i * 10 + 4)))
        for i in range(6)
    ]
    rc, out = _freeze(
        tmp_path, sources[0], sample=24, sources=sources,
        allow_source_selection=False,
    )
    assert rc == 0, "six distinct directories that happen to share a basename"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["sources"] == ["h"] * 6, "the basenames really do collide"
    assert len(set(payload["source_paths"])) == 6
    assert payload["preregistered_source_selection"] is True


def test_a_legacy_artifacts_basenames_are_not_resolved_against_the_readers_cwd(
    tmp_path: Path,
) -> None:
    """⚑ N4. The fallback for a pre-`source_paths` artifact handed BASENAMES to
    `duplicate_resolved_dirs`, which calls `Path(entry).resolve()` — so the
    duplicate message named `<scorer's cwd>/h`, a path with nothing to do with the
    frozen set, exactly where an operator has least context. Basenames are now
    compared as opaque strings and the message says they are unresolvable."""
    from chess_anti_engine.eval.lc0_control_rows import source_selection_problems

    legacy = {"sources": ["h", "h", "i"]}
    problems = source_selection_problems(legacy)
    assert any("cannot be resolved to directories" in p for p in problems), problems
    joined = " ".join(problems)
    assert str(Path.cwd()) not in joined, (
        "a basename must not be resolved against the reader's cwd"
    )
    assert "h appear(s) more than once" in joined
  # ...and a modern artifact still gets resolved paths in its message.
    modern = {"source_paths": [str(tmp_path), str(tmp_path)]}
    modern_problems = source_selection_problems(modern)
    assert any(str(tmp_path.resolve()) in p for p in modern_problems)
    assert not any("cannot be resolved" in p for p in modern_problems)


def test_freeze_accepts_the_preregistered_six_sources_unflagged(
    tmp_path: Path,
) -> None:
    """The other direction, or the gate above is a constant: SIX hours pass with
    no flag and the artifact says so."""
    sources = [
        _write_shards(tmp_path / f"hour{i}", list(range(i * 10, i * 10 + 4)))
        for i in range(6)
    ]
    rc, out = _freeze(
        tmp_path, sources[0], sample=24, sources=sources,
        allow_source_selection=False,
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["sources"]) == 6
    assert payload["preregistered_source_selection"] is True
    assert payload["source_selection_problems"] == []


def test_a_flagged_freeze_stamps_the_population_into_the_artifact(
    tmp_path: Path,
) -> None:
    """The declared deviation is carried, not merely permitted — `score` reads it
    back and `compare` refuses on it."""
    shards = _write_shards(tmp_path / "held", list(range(10)))
    rc, out = _freeze(tmp_path, shards, sample=10)
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["preregistered_source_selection"] is False
    assert payload["source_selection_problems"], "the stamp must name the reason"


def test_the_frozen_set_banks_a_game_cluster_key_per_row(tmp_path: Path) -> None:
    """⚑⚑ FREE NOW, IMPOSSIBLE AFTER THE FREEZE — which is the whole argument.

    The converter emits many PLIES per game, so the per-row hit indicators the
    yardstick averages are correlated within a game and the row-level McNemar CI
    (and `MATERIAL_BAR_PP`, derived at n=100,000 under independence) is optimistic
    by the design effect. The clustered ESTIMATOR is deferred — it needs the real
    corpus's plies-per-game distribution — but the KEY only exists before the
    freeze, so it is banked now.

    ⚑ Scoped by SOURCE DIRECTORY, because `game_id` is `enumerate()`'s index over
    ONE conversion invocation: two hours both number from 0, and a bare id would
    merge unrelated games into one cluster and UNDERSTATE the correlation — the
    flattering direction. Asserted here with two directories reusing ids 0-3.
    """
    def hour(name: str, seeds: list[int]) -> Path:
        """Distinct ROW CONTENT, deliberately COLLIDING ``game_id``s.

        The content must differ or the row-id gate refuses the set for an
        unrelated reason; the game ids must collide or this test's premise (that
        a bare `game_id` merges two hours' games) does not hold.
        """
        where = tmp_path / name
        where.mkdir(parents=True, exist_ok=True)
        samples = []
        for index, seed in enumerate(seeds):
            sample = _lc0_like_sample(seed, with_sf_wdl=False)
            sample.game_id = index
            samples.append(sample)
        save_local_shard_arrays(
            where / "shard_000000.zarr", arrs=samples_to_arrays(samples),
            meta=ShardMeta(positions=len(samples)),
        )
        return where

    hour_a = hour("hour_a", [0, 1, 2, 3])
    hour_b = hour("hour_b", [100, 101, 102, 103])
    rc, out = _freeze(tmp_path, hour_a, sample=8, sources=[hour_a, hour_b])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["cluster_keys"]) == len(payload["row_ids"]) == 8
    assert payload["cluster_keys_complete"] is True
    assert payload["cluster_key_kind"] == "source_dir#game_id"
  # ⚑ The two hours reuse game_ids 0..3, so a bare `game_id` would report FOUR
  # clusters for eight rows from eight different games. Source scoping gives 8.
    assert payload["distinct_clusters"] == 8, payload["cluster_keys"]
    assert len({k.split("#")[-1] for k in payload["cluster_keys"]}) == 4, (
        "the premise: the raw game ids really do collide across the two hours"
    )
    assert all(str(hour_a.resolve()) in k or str(hour_b.resolve()) in k
               for k in payload["cluster_keys"])


def test_subtract_trims_the_cluster_keys_with_the_rows(tmp_path: Path) -> None:
    """A row-aligned column that is not trimmed with the rows would give the
    subtracted set another set's correlation structure."""
    from chess_anti_engine.eval.lc0_control_rows import frozen_minus_exposed

    shards = _write_shards(tmp_path / "held", list(range(6)))
    rc, out = _freeze(tmp_path, shards, sample=6)
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    drop = payload["input_ids"][:2]
    trimmed = frozen_minus_exposed(payload, drop)
    assert trimmed["frozen_rows"] == 4
    assert len(trimmed["cluster_keys"]) == 4
    kept = [
        cluster
        for input_id, cluster in zip(
            payload["input_ids"], payload["cluster_keys"], strict=True,
        )
        if input_id not in set(drop)
    ]
    assert trimmed["cluster_keys"] == kept
    assert trimmed["distinct_clusters"] == len(set(kept))


def test_purity_exits_one_on_an_empty_train_directory(tmp_path: Path) -> None:
    """⚑ Codex #6, at the driver: it printed PURE and exited 0."""
    shards = _write_shards(tmp_path / "held", list(range(10)))
    _rc, frozen = _freeze(tmp_path, shards, sample=10)
    empty = tmp_path / "no_train"
    empty.mkdir()
    assert lc0_control_heldout.main([
        "purity", "--frozen", str(frozen), "--train-shards", str(empty),
    ]) == 1


def test_purity_exits_one_on_exposure_the_record_id_cannot_see(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑ Review F5: same inputs, different lc0 targets — 0 by id, N by input."""
    held = tmp_path / "held"
    train = tmp_path / "train"
    for target_seed, where in ((1, held), (2, train)):
        where.mkdir(parents=True, exist_ok=True)
        samples = []
        for seed in range(6):
            sample = _lc0_like_sample(seed, with_sf_wdl=False)
            rng = np.random.default_rng(9_000 + target_seed)
            policy = rng.random(COMPACT_POLICY_SIZE).astype(np.float32)
            sample.policy_target = policy / policy.sum()
            samples.append(sample)
        save_local_shard_arrays(
            where / "shard_000000.zarr", arrs=samples_to_arrays(samples),
            meta=ShardMeta(positions=len(samples)),
        )
    _rc, frozen = _freeze(tmp_path, held, sample=6)
    assert lc0_control_heldout.main([
        "purity", "--frozen", str(frozen), "--train-shards", str(train),
    ]) == 1
    out = capsys.readouterr().out
    assert "intersecting ids     0" in out, "the record-id count must still read 0"
    assert "EXPOSED inputs       6" in out


def test_the_purity_receipt_ties_the_trained_corpus_to_the_check(
    tmp_path: Path,
) -> None:
    """⚑⚑ REVIEW F5 — NOTHING TIED THE TRAINED ROWS TO THE PURITY CHECK.

    ``summary.json`` recorded no shard list and no frozen sha256, and
    ``purity --train-shards`` was a free-floating argument. The receipt makes
    the link machine-checkable, so all three directions are tested here: the
    covered corpus LAUNCHES and banks the sha, an UNCOVERED directory is
    refused, and a receipt recording ``pure: false`` is refused.
    """
    held = _write_shards(tmp_path / "held", list(range(6)))
    train = _write_shards(tmp_path / "train", list(range(100, 108)))
    other = _write_shards(tmp_path / "other", list(range(200, 206)))
    _rc, frozen = _freeze(tmp_path, held, sample=6)
    receipt = tmp_path / "purity.json"
    assert lc0_control_heldout.main([
        "purity", "--frozen", str(frozen), "--train-shards", str(train),
        "--receipt", str(receipt),
    ]) == 0
    banked = json.loads(receipt.read_text(encoding="utf-8"))
    assert banked["pure"] is True
    assert banked["train_shards"] == [str(train.resolve())]

    def _run(shards: list[Path], out: str) -> int:
        return lc0_control_train.main([
            "--config", str(_tiny_config(tmp_path)),
            "--shards", *[str(s) for s in shards],
            "--out-dir", str(tmp_path / out), "--steps", "1", "--batch-size", "4",
            "--device", "cpu", "--no-compile", "--allow-arch-drift",
            "--allow-invalid-control",
            "--purity-receipt", str(receipt),
        ])

    assert _run([train], "covered") == 0
    summary = json.loads(
        (tmp_path / "covered" / "summary.json").read_text(encoding="utf-8"),
    )
    assert summary["corpus"]["shard_dirs"] == [str(train.resolve())]
    assert summary["corpus"]["label_coverage"]["search_wdl"] == {
        "labelled_rows": 8, "rows": 8,
    }
    assert summary["purity_receipt"]["frozen_sha256"] == banked["frozen_sha256"]

    with pytest.raises(SystemExit, match="NOT covered by the purity receipt"):
        _run([train, other], "uncovered")

    banked["pure"] = False
    receipt.write_text(json.dumps(banked), encoding="utf-8")
    with pytest.raises(SystemExit, match="records pure=False"):
        _run([train], "impure")


def test_the_receipt_is_tied_to_the_frozen_set_by_CONTENT_not_only_by_NAME(
    tmp_path: Path,
) -> None:
    """⚑⚑ PR #438 review, finding 2 — two holes in one gate.

    ``frozen_sha256`` was written by `purity`, copied into ``summary.json`` and
    PRINTED at launch, and never once compared to the artifact it names. A
    receipt field that nothing reads is this repo's signature defect, and here
    it hid a real state: re-freezing the held-out set (new sample, new seed, a
    repaired split) renames nothing, so the directory-set check below stays
    green while the receipt now clears a set that no longer exists.

    The second hole is the comparison itself: ``used`` was resolved and
    ``covered`` was not, so the two sets were compared across normalisation.
    It agrees today only because this repo's writer happens to resolve before
    banking — which makes it an accident, not a check.
    """
    held = _write_shards(tmp_path / "held", list(range(6)))
    train = _write_shards(tmp_path / "train", list(range(100, 108)))
    _rc, frozen = _freeze(tmp_path, held, sample=6)
    receipt = tmp_path / "purity.json"
    assert lc0_control_heldout.main([
        "purity", "--frozen", str(frozen), "--train-shards", str(train),
        "--receipt", str(receipt),
    ]) == 0

  # Baseline: the honest receipt clears its own corpus with no problems, so
  # every assertion below is about the change and not about a broken fixture.
    _receipt, problems = lc0_control_train.purity_receipt_problems(receipt, [train])
    assert problems == [], problems

  # (a) SPELLING. A path that resolves to the cleared directory IS the cleared
  # directory. Under the unresolved comparison this reported "NOT covered".
    banked = json.loads(receipt.read_text(encoding="utf-8"))
    detour = train.parent / "zzz" / ".." / train.name
    assert Path(detour).resolve() == train.resolve()
    assert str(detour) != str(train.resolve())
    banked["train_shards"] = [str(detour)]
    receipt.write_text(json.dumps(banked), encoding="utf-8")
    _receipt, problems = lc0_control_train.purity_receipt_problems(receipt, [train])
    assert problems == [], problems

  # (b) CONTENT. Same path, same name, different bytes — the only tie that can
  # see a re-freeze.
    banked["train_shards"] = [str(train.resolve())]
    receipt.write_text(json.dumps(banked), encoding="utf-8")
    frozen.write_bytes(frozen.read_bytes() + b"\n")
    _receipt, problems = lc0_control_train.purity_receipt_problems(receipt, [train])
    assert any("has CHANGED since the" in p for p in problems), problems

  # (c) ABSENCE is a problem, not a skip: a gate that verifies only when the
  # artifact happens to be there is a gate that cannot fail.
    frozen.unlink()
    _receipt, problems = lc0_control_train.purity_receipt_problems(receipt, [train])
    assert any("is GONE" in p for p in problems), problems

  # (d) And a receipt that banks no hash at all cannot be waved through either.
    banked.pop("frozen_sha256")
    receipt.write_text(json.dumps(banked), encoding="utf-8")
    _receipt, problems = lc0_control_train.purity_receipt_problems(receipt, [train])
    assert any("names no frozen artifact" in p for p in problems), problems


def test_a_run_without_a_purity_receipt_is_not_a_valid_control(
    tmp_path: Path,
) -> None:
    """The receipt is optional so plumbing runs stay possible — but a run that
    skipped it must SAY so in the artifact, by name, rather than inherit a
    `valid_control: true` it did not earn."""
    shards = _write_shards(tmp_path / "rows", list(range(8)))
    out = tmp_path / "no_receipt"
    assert lc0_control_train.main([
        "--config", str(_tiny_config(tmp_path)), "--shards", str(shards),
        "--out-dir", str(out), "--steps", "1", "--batch-size", "4",
        "--device", "cpu", "--no-compile", "--allow-arch-drift",
        "--allow-invalid-control",
    ]) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["valid_control"] is False
    assert any("purity" in p for p in summary["validity_problems"])
    assert summary["purity_receipt"] is None


def test_a_run_judged_only_against_the_pins_is_not_a_valid_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ REVIEW F2. The artifact already SAID the live file was not read, in the
    provenance strings — and recorded no validity problem for it, so a full arm
    run with the env var unset could write `valid_control: true` while BOTH
    premise axes were judged against committed snapshots.

    That is strictly weaker evidence, for the same reason the receipt is: a pin
    cannot detect that the live file moved after it was recorded. The freshness
    tests that WOULD catch it skip when no live file is named, and making CI read
    an absolute host path was rejected — so the run artifact is the only place
    this can live.
    """
    monkeypatch.delenv(lc0_control_arch.LIVE_CONFIG_ENV, raising=False)
    shards = _write_shards(tmp_path / "rows", list(range(8)))
    out = tmp_path / "pin_only"
    assert lc0_control_train.main([
        "--config", str(_tiny_config(tmp_path)), "--shards", str(shards),
        "--out-dir", str(out), "--steps", "1", "--batch-size", "4",
        "--device", "cpu", "--no-compile", "--allow-arch-drift",
        "--allow-invalid-control",
    ]) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
  # The premise, asserted rather than assumed: the provenance really does carry
  # the sentinel, so this test is about the MISSING VERDICT and not about a
  # string that never appears.
    assert lc0_control_arch.LIVE_FILE_UNREAD in summary["trainer_judged_against"]
    assert any("COMMITTED PIN" in p for p in summary["validity_problems"]), (
        "a run judged only against the pins must say so in validity_problems"
    )


def test_the_realized_device_and_compile_survive_into_the_artifact(
    tmp_path: Path,
) -> None:
    """⚑ REVIEW F6. Guard 0c certifies `kwargs` against the live recipe and the
    driver THEN overwrites `device` and `use_compile` from the CLI. Both are in
    the pin ("cuda" / True), so without this the artifact gives a reader no way
    to tell which recipe actually ran."""
    shards = _write_shards(tmp_path / "rows", list(range(8)))
    out = tmp_path / "realized"
    assert lc0_control_train.main([
        "--config", str(_tiny_config(tmp_path)), "--shards", str(shards),
        "--out-dir", str(out), "--steps", "1", "--batch-size", "4",
        "--device", "cpu", "--no-compile", "--allow-arch-drift",
        "--allow-invalid-control",
    ]) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["realized_after_guard"] == {
        "device": "cpu", "configured_device": "cuda", "use_compile": False,
    }
  # ⚑ And `device` is no longer record-only: the CONFIGURED value is banked beside
  # the realized one, and the deviation is a validity problem in its own right.
  # The comment that used to defer this reasoned that an entry "would disqualify
  # every smoke" -- measurably false, this smoke already carries several.
    assert [p for p in summary["validity_problems"]
            if "--device cpu is not the configured cuda" in p], (
        summary["validity_problems"]
    )


def test_the_training_seed_is_banked_so_a_trajectory_can_be_reproduced(
    tmp_path: Path,
) -> None:
    """⚑ `--seed` seeds `torch.manual_seed` before `build_model` AND the replay
    buffer's RNG, and it was in neither the summary nor the checkpoint. `run_id`
    is content-derived, so it IDENTIFIES a trajectory and cannot REPRODUCE one —
    and the replay deviation `deterministic_refresh: true` is justified as "a pure
    function of the seed", which is unauditable if the seed is not recorded."""
    shards = _write_shards(tmp_path / "rows", list(range(16)))
    out = tmp_path / "seeded"
    assert lc0_control_train.main([
        "--config", str(_tiny_config(tmp_path)), "--shards", str(shards),
        "--out-dir", str(out), "--steps", "4", "--batch-size", "4",
        "--device", "cpu", "--no-compile", "--allow-arch-drift",
        "--allow-invalid-control", "--seed", "7",
    ]) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["seed"] == 7
  # ...and it is the seed that actually ran: seed 7 and seed 8 are different
  # trajectories, which is exactly what `run_id` reports.
    other = tmp_path / "seeded8"
    assert lc0_control_train.main([
        "--config", str(_tiny_config(tmp_path)), "--shards", str(shards),
        "--out-dir", str(other), "--steps", "4", "--batch-size", "4",
        "--device", "cpu", "--no-compile", "--allow-arch-drift",
        "--allow-invalid-control", "--seed", "8",
    ]) == 0
    other_summary = json.loads((other / "summary.json").read_text(encoding="utf-8"))
    assert other_summary["seed"] == 8
    assert other_summary["run_id"] != summary["run_id"]


def test_purity_exits_zero_on_a_genuinely_disjoint_split(tmp_path: Path) -> None:
    held = _write_shards(tmp_path / "held", list(range(6)))
    train = _write_shards(tmp_path / "train", list(range(100, 106)))
    _rc, frozen = _freeze(tmp_path, held, sample=6)
    assert lc0_control_heldout.main([
        "purity", "--frozen", str(frozen), "--train-shards", str(train),
    ]) == 0


# ── the repair path: dump the exposure, subtract it, re-verify ────────────────


def _exposed_split(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A held-out set of 6 rows, 2 of whose INPUTS are already in train.

    The two shared rows carry DIFFERENT policy targets on each side, so the
    record-id count stays 0 and only the input-level gate sees them — the
    2026-08-16 failure in miniature.
    """
    held = tmp_path / "held"
    train = tmp_path / "train"
    for target_seed, where, seeds in (
        (1, held, [0, 1, 2, 3, 4, 5]), (2, train, [0, 1, 90, 91]),
    ):
        where.mkdir(parents=True, exist_ok=True)
        samples = []
        for seed in seeds:
            sample = _lc0_like_sample(seed, with_sf_wdl=False)
            rng = np.random.default_rng(9_000 + target_seed + seed)
            policy = rng.random(COMPACT_POLICY_SIZE).astype(np.float32)
            sample.policy_target = policy / policy.sum()
            samples.append(sample)
        save_local_shard_arrays(
            where / "shard_000000.zarr", arrs=samples_to_arrays(samples),
            meta=ShardMeta(positions=len(samples)),
        )
    _rc, frozen = _freeze(tmp_path, held, sample=6)
    return held, train, frozen


def test_purity_dumps_the_exposed_ids_and_subtract_makes_the_set_pure(
    tmp_path: Path,
) -> None:
    """⚑⚑ THE REPAIR, END TO END, AT THE ENTRY POINTS.

    The 2026-08-16 gate banked ``exposed_inputs: 5065`` and five example
    hashes; rebuilding the split then needed a second 2h40m scan to recover an
    operand the first scan had already computed. This asserts the three steps
    are actually connected: FAIL writes the ids, ``subtract`` consumes them,
    and the resulting set PASSES against the same corpus.
    """
    _held, train, frozen = _exposed_split(tmp_path)
    dump = tmp_path / "exposed.json"
    assert lc0_control_heldout.main([
        "purity", "--frozen", str(frozen), "--train-shards", str(train),
        "--exposed-out", str(dump),
    ]) == 1
    banked = json.loads(dump.read_text(encoding="utf-8"))
    assert banked["exposed_inputs"] == 2
    assert len(banked["exposed_input_ids"]) == 2
    assert banked["intersecting_ids"] == 0, "record ids must still read clean"
    assert [row["row_id"] for row in banked["rows"]], "the dump must name ROWS"
    assert {row["input_id"] for row in banked["rows"]} == set(
        banked["exposed_input_ids"],
    )

    clean = tmp_path / "frozen_clean.json"
    assert lc0_control_heldout.main([
        "subtract", "--frozen", str(frozen), "--exposed", str(dump),
        "--out", str(clean),
    ]) == 0
    trimmed = json.loads(clean.read_text(encoding="utf-8"))
    assert trimmed["frozen_rows"] == 4
    assert trimmed["removed_rows"] == 2
    assert trimmed["derived_from_sha256"] == banked["frozen_sha256"]
  # ⚑ Clean BY CONSTRUCTION is exactly why it is re-verified: a subtraction
  # that removed the wrong rows would still produce a plausible artifact.
    receipt = tmp_path / "clean_receipt.json"
    assert lc0_control_heldout.main([
        "purity", "--frozen", str(clean), "--train-shards", str(train),
        "--receipt", str(receipt),
    ]) == 0
    assert json.loads(receipt.read_text(encoding="utf-8"))["pure"] is True


def test_the_exposed_dump_is_written_on_a_clean_split_too(tmp_path: Path) -> None:
    """"No file" must not be ambiguous between clean and flag-forgotten."""
    held = _write_shards(tmp_path / "held", list(range(6)))
    train = _write_shards(tmp_path / "train", list(range(100, 106)))
    _rc, frozen = _freeze(tmp_path, held, sample=6)
    dump = tmp_path / "exposed.json"
    assert lc0_control_heldout.main([
        "purity", "--frozen", str(frozen), "--train-shards", str(train),
        "--exposed-out", str(dump),
    ]) == 0
    banked = json.loads(dump.read_text(encoding="utf-8"))
    assert banked["exposed_input_ids"] == []
    assert banked["rows"] == []


def test_subtract_refuses_an_exposed_dump_from_another_frozen_set(
    tmp_path: Path,
) -> None:
    """⚑ Two free-floating CLI paths with nothing tying them together is the
    exact shape the purity receipt exists to stop; the same applies here."""
    _held, train, frozen = _exposed_split(tmp_path)
    dump = tmp_path / "exposed.json"
    assert lc0_control_heldout.main([
        "purity", "--frozen", str(frozen), "--train-shards", str(train),
        "--exposed-out", str(dump),
    ]) == 1
    banked = json.loads(dump.read_text(encoding="utf-8"))
    banked["frozen_sha256"] = "0" * 64
    dump.write_text(json.dumps(banked), encoding="utf-8")
    assert lc0_control_heldout.main([
        "subtract", "--frozen", str(frozen), "--exposed", str(dump),
        "--out", str(tmp_path / "clean.json"),
    ]) == 1
    assert not (tmp_path / "clean.json").exists()
  # ⚑ And a dump with NO sha at all is refused too, rather than satisfying the
  # check by absence.
    banked.pop("frozen_sha256")
    dump.write_text(json.dumps(banked), encoding="utf-8")
    assert lc0_control_heldout.main([
        "subtract", "--frozen", str(frozen), "--exposed", str(dump),
        "--out", str(tmp_path / "clean.json"),
    ]) == 1


def test_subtract_refuses_to_overwrite_the_set_a_result_is_recorded_against(
    tmp_path: Path,
) -> None:
    """`frozen_full.json` is the artifact the FAILED run is recorded against."""
    _held, train, frozen = _exposed_split(tmp_path)
    dump = tmp_path / "exposed.json"
    lc0_control_heldout.main([
        "purity", "--frozen", str(frozen), "--train-shards", str(train),
        "--exposed-out", str(dump),
    ])
    before = frozen.read_bytes()
    assert lc0_control_heldout.main([
        "subtract", "--frozen", str(frozen), "--exposed", str(dump),
        "--out", str(frozen),
    ]) == 1
    assert frozen.read_bytes() == before


def test_subtract_refuses_a_dump_whose_inputs_are_not_in_this_set(
    tmp_path: Path,
) -> None:
    """The distinct-input count is EXACTLY determined by the subtraction.

    Predicting it and asserting the actual is the standing rule here; a dump
    naming an input this set does not carry would otherwise trim fewer rows
    than it claims and report a clean set built from the wrong population.
    """
    _held, train, frozen = _exposed_split(tmp_path)
    dump = tmp_path / "exposed.json"
    lc0_control_heldout.main([
        "purity", "--frozen", str(frozen), "--train-shards", str(train),
        "--exposed-out", str(dump),
    ])
    banked = json.loads(dump.read_text(encoding="utf-8"))
    banked["exposed_input_ids"].append("f" * 32)
    dump.write_text(json.dumps(banked), encoding="utf-8")
    assert lc0_control_heldout.main([
        "subtract", "--frozen", str(frozen), "--exposed", str(dump),
        "--out", str(tmp_path / "clean.json"),
    ]) == 1


def test_the_purity_cache_is_reused_and_is_keyed_to_the_corpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑ A cache that always hits is a gate that cannot fail.

    Both directions at the entry point: an unchanged corpus is CACHED, and a
    corpus that grew is rescanned and reports the larger row count.
    """
    held = _write_shards(tmp_path / "held", list(range(6)))
    train = _write_shards(tmp_path / "train", list(range(100, 106)))
    _rc, frozen = _freeze(tmp_path, held, sample=6)
    cache = tmp_path / "cache"
    args = [
        "purity", "--frozen", str(frozen), "--train-shards", str(train),
        "--cache-dir", str(cache),
    ]
    assert lc0_control_heldout.main(args) == 0
    assert "train index          rebuilt by scanning" in capsys.readouterr().out
    assert lc0_control_heldout.main(args) == 0
    assert "train index          CACHED" in capsys.readouterr().out

    grown = _write_shards(tmp_path / "train2", list(range(200, 210)))
    assert lc0_control_heldout.main([
        "purity", "--frozen", str(frozen),
        "--train-shards", str(train), str(grown), "--cache-dir", str(cache),
    ]) == 0
    out = capsys.readouterr().out
    assert "train index          rebuilt by scanning" in out
    assert "train rows scanned   16 " in out


def test_chance_prints_the_jensen_pair(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    shards = _write_shards(tmp_path / "held", list(range(6)))
    _rc, frozen = _freeze(tmp_path, shards, sample=6)
    assert lc0_control_heldout.main([
        "chance", "--frozen", str(frozen), "--shards", str(shards),
    ]) == 0
    out = capsys.readouterr().out
  # 20 legal moves in every synthetic row, so the two statistics coincide here
  # and the test is about the driver printing both, not about Jensen.
    assert "E[1/n_legal]         0.050000" in out
    assert "1/E[n_legal]         0.050000" in out


# ── lc0_control_eval ──────────────────────────────────────────────────────────


def _scores(
    path: Path, hits: np.ndarray, *, checkpoint: str | None = None,
    role: str = "mid", population: str = "heldout",
) -> Path:
    np.savez_compressed(
        path,
        row_ids=np.array([f"id{i:04d}" for i in range(hits.size)], dtype="U32"),
        hit=hits.astype(np.uint8),
  # `valid_control: True` for the same reason as in `_paired_scores`: these
  # fixtures exercise the ARITHMETIC and the resolution bar, so the validity
  # refusal must not stand in front of them. Its own gate is pinned separately.
  # ⚑ Same reasoning for `run_id`/`checkpoint_role`: `compare` refuses a pair
  # with no trajectory identity, correctly and with its own test, so these carry
  # ONE run id and the prereg's {mid, last} roles.
        meta=np.array([json.dumps({
            "checkpoint": checkpoint, "rows": int(hits.size),
            "valid_control": True, "run_id": "trajectory-0",
            "checkpoint_role": role,
  # ⚑ Empty LIST, not absent: `compare` refuses an artifact with no held-out
  # population record (absent is not clean), which has its own test.
            "heldout_source_selection_problems": [],
  # ⚑ And the POPULATION role, for the fourth time over the same reasoning: an
  # artifact that does not say which of the prereg's two deltas it is gets its own
  # refusal, with its own test, and must not stand in front of these fixtures.
            "population": population,
        })], dtype=object),
        allow_pickle=True,
    )
    return path


def test_compare_refuses_a_slope_it_cannot_resolve(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑ The Δ_train gap: 5,000 paired rows gave a 1.0721 pp halfwidth against a
    bar derived at n=100,000, and the rig printed it without comment."""
    rng = np.random.default_rng(0)
    a = (rng.random(5_000) < 0.30).astype(np.uint8)
    b = (rng.random(5_000) < 0.33).astype(np.uint8)
    rc = lc0_control_eval.main([
        "compare", "--a", str(_scores(tmp_path / "a.npz", a)),
        "--b", str(_scores(tmp_path / "b.npz", b, role="last")),
    ])
    assert rc == 1
    assert "exceeds the pre-committed bar" in capsys.readouterr().err


def test_compare_reports_when_the_halfwidth_clears_the_bar(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The same refusal must not fire at the n the prereg was written for."""
    rng = np.random.default_rng(1)
    a = (rng.random(100_000) < 0.30).astype(np.uint8)
    b = a.copy()
    flip = rng.choice(100_000, size=300, replace=False)
    b[flip] = 1 - b[flip]
    rc = lc0_control_eval.main([
        "compare", "--a", str(_scores(tmp_path / "a.npz", a)),
        "--b", str(_scores(tmp_path / "b.npz", b, role="last")),
    ])
    assert rc == 0
    assert "halfwidth" in capsys.readouterr().out


def test_compare_mcnemar_arithmetic_is_pinned(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The estimator the 0.392 pp bar is derived from had no regression test.

    Hand-built cells: b = 10 (A only), c = 30 (B only), 1,000 concordant hits
    and 8,960 concordant misses, n = 10,000.
    """
    n, b_cells, c_cells, both = 10_000, 10, 30, 1_000
    hit_a = np.zeros(n, dtype=np.uint8)
    hit_b = np.zeros(n, dtype=np.uint8)
    hit_a[:both] = hit_b[:both] = 1
    hit_a[both:both + b_cells] = 1
    hit_b[both + b_cells:both + b_cells + c_cells] = 1
    lc0_control_eval.main([
        "compare", "--a", str(_scores(tmp_path / "a.npz", hit_a)),
        "--b", str(_scores(tmp_path / "b.npz", hit_b, role="last")),
  # ⚑ BOTH flags, and that is the fix rather than the test being lenient: a
  # `--max-halfwidth-pp` LOOSER than the material bar no longer switches off the
  # n floor (it is a relaxation, not a re-derivation), so this n=10,000 fixture —
  # which exists to pin the ESTIMATOR, not to make a resolution claim — has to
  # waive the resolution gate explicitly. The waived banner is printed and the
  # numbers still follow it.
        "--max-halfwidth-pp", "10.0", "--allow-underpowered",
    ])
    out = capsys.readouterr().out
    delta = (c_cells - b_cells) / n
    var = (b_cells + c_cells - (c_cells - b_cells) ** 2 / n) / n**2
    half = 1.959963985 * float(np.sqrt(var))
    assert f"delta (B - A)        {delta * 100:+.4f} pp" in out
    assert f"(halfwidth {half * 100:.4f} pp)" in out
  # Cross-checked against an independent implementation:
  # `scipy.stats.binomtest(10, 40, 0.5).pvalue` == 0.0022214337732293643.
    assert lc0_control_eval._exact_mcnemar_p(10, 30) == pytest.approx(
        0.0022214337732293643, rel=1e-9,
    )


def test_compare_refuses_unpaired_score_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    a = _scores(tmp_path / "a.npz", np.ones(10, dtype=np.uint8))
    b = tmp_path / "b.npz"
    np.savez_compressed(
        b, row_ids=np.array([f"other{i}" for i in range(10)], dtype="U32"),
        hit=np.ones(10, dtype=np.uint8),
  # `valid_control` so the validity refusal — which runs first, correctly, and
  # would otherwise mask this one — does not stand in front of the row-pairing
  # message this test is about.
        meta=np.array([json.dumps({
            "checkpoint": None, "valid_control": True,
            "run_id": "trajectory-0", "checkpoint_role": "last",
            "heldout_source_selection_problems": [], "population": "heldout",
        })], dtype=object),
        allow_pickle=True,
    )
    assert lc0_control_eval.main(["compare", "--a", str(a), "--b", str(b)]) == 1
    assert "cannot be paired" in capsys.readouterr().err


def test_the_paired_halfwidth_reproduces_the_preregs_resolution_table() -> None:
    """⚑ REVIEW F2a — the shipped defaults cited a file that did not exist.

    ``docs/lc0_positive_control_prereg.md`` is committed now, and its resolution
    table is DERIVED here rather than remembered, so a drift between the doc
    and the code is a test failure instead of a discrepancy nobody re-checks.
    """
    assert lc0_control_eval.paired_halfwidth_pp(
        n=100_000, discordance=0.10,
    ) == pytest.approx(0.196, abs=5e-4)
    assert lc0_control_eval.paired_halfwidth_pp(
        n=100_000, discordance=0.20,
    ) == pytest.approx(0.277, abs=5e-4)
    assert lc0_control_eval.MATERIAL_BAR_PP == 0.392
  # ...and it is the number the CLI actually applies, not a parallel constant.
    derived = 2 * lc0_control_eval.paired_halfwidth_pp(n=100_000, discordance=0.10)
    assert pytest.approx(derived, abs=5e-4) == lc0_control_eval.MATERIAL_BAR_PP


def test_the_shuffled_control_floor_is_the_collision_rate_not_one_over_n_legal() -> None:
    """⚑⚑ REVIEW F2b — GUARD 1 WAS COMPARED AGAINST THE WRONG QUANTITY.

    The control scores each prediction against ANOTHER row's target, so its
    expectation is ``SUM_m p_pred(m)*p_tgt(m)``. Verified against a brute-force
    average over MANY permutations of the same arrays, which is the definition
    the derivation claims and cannot be satisfied by restating the formula.
    """
    rng = np.random.default_rng(0)
  # A deliberately non-uniform target marginal, like real lc0 argmaxes.
    target = rng.choice(20, size=4_000, p=np.array([0.3, 0.2, 0.1] + [0.4 / 17] * 17))
    predicted = rng.choice(20, size=4_000, p=np.array([0.25, 0.25, 0.1] + [0.4 / 17] * 17))

    expected = lc0_control_eval.shuffled_control_expectation(predicted, target)
    empirical = float(np.mean([
        (predicted == target[np.random.default_rng(s).permutation(target.size)]).mean()
        for s in range(200)
    ]))
    assert expected == pytest.approx(empirical, abs=2e-3)

  # ...and it is nowhere near the uniform-mover floor the prereg named. With 20
  # equally-legal moves that floor is 0.05; the collision rate is well above it
  # here and ~19x BELOW it on real lc0 rows. Either way it is a different
  # quantity, and a gate cannot be judged against the wrong one.
    assert expected != pytest.approx(1 / 20, abs=1e-3)


def test_the_negative_control_gate_can_pass_and_can_fail() -> None:
    """⚑ BOTH DIRECTIONS REACHABLE. A gate that cannot fail is not a gate, and
    one that cannot pass is a constant."""
    at_floor, problem = lc0_control_eval.negative_control_problem(
        observed=0.0033, expected=0.003283, rows=100_000, max_z=5.0,
    )
    assert problem is None, "a control sitting at its own floor must PASS"
    assert abs(at_floor) < 5.0

    z, problem = lc0_control_eval.negative_control_problem(
        observed=0.35, expected=0.003283, rows=100_000, max_z=5.0,
    )
    assert problem is not None
    assert z > 100.0
    assert "manufacturing agreement" in problem
  # ⚑ Landing BELOW the floor is not a failure: there is no mechanism behind
  # it. Only the "still agrees" direction has one.
    _z, low = lc0_control_eval.negative_control_problem(
        observed=0.0, expected=0.003283, rows=100_000, max_z=5.0,
    )
    assert low is None
  # A control that scored nothing checked nothing.
    _z, empty = lc0_control_eval.negative_control_problem(
        observed=0.0, expected=0.0, rows=0, max_z=5.0,
    )
    assert empty is not None


def test_score_exits_one_when_the_negative_control_does_not_collapse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The gate, THROUGH ``cmd_score``. The arithmetic being right is not the
    same claim as the driver acting on it — that gap is what F2b was about.

    ``_score_rows`` is stubbed because a leaking rig cannot be built out of a
    real model on demand; everything downstream of it is the real path.
    """
    rows = 60
    shards = _write_shards(tmp_path / "held", list(range(rows)))
    _rc, frozen = _freeze(tmp_path, shards, sample=rows)

    def _rigged(*_args: Any, **_kwargs: Any) -> tuple[Any, int, Any, Any]:
        moves = np.arange(rows, dtype=np.int64) % 3
        return np.ones(rows, dtype=np.uint8), 0, moves, moves

    monkeypatch.setattr(lc0_control_eval, "_score_rows", _rigged)
    rc = lc0_control_eval.main([
        "score", "--config", str(_tiny_config(tmp_path)), "--frozen", str(frozen),
        "--shards", str(shards), "--out", str(tmp_path / "s.npz"),
        "--device", "cpu", "--shuffle-targets",
    ])
    assert rc == 1
    assert "manufacturing agreement" in capsys.readouterr().err
  # The artifact is still written: the floor and the z belong WITH the score,
  # or nobody can re-check the verdict later.
    meta = json.loads(str(np.load(tmp_path / "s.npz", allow_pickle=True)["meta"][0]))
    assert meta["shuffled_collision_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert meta["negative_control_z"] > 5.0
  # ⚑ The BAR, banked next to the z. `compare` refuses an artifact that failed
  # its own gate, and it cannot know which bar this run was gated at unless the
  # run wrote it down.
    assert meta["negative_control_max_z"] == pytest.approx(5.0)
  # ⚑⚑ THE FIELD THE WHOLE GATE KEYS ON, ASSERTED END-TO-END. Every provenance
  # test below hand-banks its own meta dict, so the READER was tested in
  # complete isolation from the WRITER — and an independent review of this PR
  # measured what that costs: renaming this key on the writer side ONLY
  # (`shuffled_target_seed`) left all 61 tests green while every real
  # `score --shuffle-targets` artifact read back None, `target_provenance`
  # labelled it "real lc0 targets", `score_provenance_problems` returned empty,
  # and the negative control printed the slope at exit 0. That is finding
  # 3791327309 reopened, with a green suite. One line closes it, and it has to
  # live HERE, on an artifact the real `cmd_score` wrote.
    assert meta["shuffled_targets_seed"] == 0


# ── compare: the NEGATIVE-CONTROL metadata it already loads ───────────────────


def _paired_scores(
    tmp_path: Path, *, meta_a: dict[str, Any], meta_b: dict[str, Any],
    n: int = 100_000, heldout_problems: list[str] | tuple[()] | None = (),
) -> tuple[Path, Path]:
    """The triage's reproduction of Codex finding 3791327309, verbatim.

    ``c=120`` / ``b=80`` discordant pairs at n=100,000, which is the pair that
    printed ``delta +0.0400 pp, CI [+0.0123, +0.0677]`` and EXITED 0 while --b
    carried ``shuffled_targets_seed: 0``.
    """
    rng = np.random.default_rng(7)
    a = (rng.random(n) < 0.30).astype(np.uint8)
    b = a.copy()
    b[rng.choice(np.flatnonzero(a == 0), size=120, replace=False)] = 1
    b[rng.choice(np.flatnonzero(a == 1), size=80, replace=False)] = 0
  # ⚑ `valid_control: True` is the DEFAULT here so each test below isolates the
  # one refusal it is about. `compare` refuses an artifact with no validity
  # record at all, which is correct and is pinned by its own test — but if that
  # refusal fired in every fixture, the shuffled-provenance tests would pass for
  # the wrong reason and a regression in the provenance gate would be invisible
  # behind an unrelated green. Callers that mean to test the validity gate
  # override it explicitly.
  # ⚑ And ONE trajectory with the prereg's {mid, last} roles, for the same
  # reason: `compare` refuses a pair with no `run_id`/`checkpoint_role`, which is
  # correct and has its own test, but if that refusal fired in every fixture the
  # tests below would pass for the wrong reason.
  # ⚑ And a RECORDED (empty) held-out population, for the third time over the same
  # reasoning: `heldout_problems=None` omits the key, which is its own refusal.
    common: dict[str, Any] = {
        "valid_control": True, "run_id": "trajectory-0", "population": "heldout",
    }
    if heldout_problems is not None:
        common["heldout_source_selection_problems"] = list(heldout_problems)
    meta_a = {**common, "checkpoint_role": "mid", **meta_a}
    meta_b = {**common, "checkpoint_role": "last", **meta_b}

    def write(path: Path, hits: np.ndarray, meta: dict[str, Any]) -> Path:
        np.savez_compressed(
            path,
            row_ids=np.array([f"id{i:06d}" for i in range(hits.size)], dtype="U32"),
            hit=hits, meta=np.array([json.dumps(meta)], dtype=object),
            allow_pickle=True,
        )
        return path

    return (
        write(tmp_path / "a.npz", a, meta_a),
        write(tmp_path / "b.npz", b, meta_b),
    )


def test_compare_refuses_a_negative_control_read_as_the_primary_slope(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑⚑ CODEX 3791327309. ``cmd_compare`` loaded the metadata and used exactly
    one field of it (``checkpoint``, for a print), so prereg guard 1 — the
    PERMUTED-TARGET negative control — differenced against a real-target score
    read as the primary learning slope and cleared every pre-committed gate.

    Three separate assertions, because they fail separately: the exit code (the
    only thing a caller sees), the message naming the field AND both values (the
    only thing that tells an operator what to fix), and the ABSENCE of the slope
    from stdout — a refusal that still prints ``delta`` leaves a quotable number
    on the screen, which is the whole failure.
    """
    a, b = _paired_scores(
        tmp_path,
        meta_a={"checkpoint": "checkpoint_mid.pt", "shuffled_targets_seed": None},
        meta_b={"checkpoint": "checkpoint.pt", "shuffled_targets_seed": 0},
    )
    rc = lc0_control_eval.main(["compare", "--a", str(a), "--b", str(b)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "shuffled_targets_seed" in captured.err
    assert "None" in captured.err, "the --a value must be named"
    assert "0" in captured.err, "the --b value must be named"
    assert "delta" not in captured.out, (
        "the slope must not be printed for a pair that is refused"
    )


def test_compare_refuses_two_negative_controls(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Equal provenance is not sufficient provenance: differencing two collision
    rates satisfies every other gate in the file and says nothing about the arm."""
    a, b = _paired_scores(
        tmp_path,
        meta_a={"checkpoint": "mid", "shuffled_targets_seed": 3},
        meta_b={"checkpoint": "last", "shuffled_targets_seed": 3},
    )
    assert lc0_control_eval.main(["compare", "--a", str(a), "--b", str(b)]) == 1
    assert "BOTH score files are negative controls" in capsys.readouterr().err


def test_the_shuffled_contrast_flag_waives_the_refusal_and_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The one comparison that DOES want the shuffled contrast still works, and
    the run is stamped so nobody reads the output as a primary slope."""
    a, b = _paired_scores(
        tmp_path,
        meta_a={"checkpoint": "mid", "shuffled_targets_seed": None},
        meta_b={"checkpoint": "last", "shuffled_targets_seed": 0},
    )
    rc = lc0_control_eval.main([
        "compare", "--a", str(a), "--b", str(b), "--allow-shuffled-contrast",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "--allow-shuffled-contrast" in out
    assert "delta" in out, "the waived path must still report the comparison"


def test_the_waiver_does_not_cover_a_control_that_failed_its_own_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑ "I meant to compare the shuffled control" is a statement about INTENT;
    a control that beat its own collision floor is a statement about the RIG.
    A rig that manufactures agreement supports no verdict in either direction,
    including a deliberately-contrasted one, so this refusal is unwaivable."""
    a, b = _paired_scores(
        tmp_path,
        meta_a={"checkpoint": "mid", "shuffled_targets_seed": None},
        meta_b={
            "checkpoint": "last", "shuffled_targets_seed": 0,
            "negative_control_z": 41.7, "negative_control_max_z": 5.0,
        },
    )
    assert lc0_control_eval.main([
        "compare", "--a", str(a), "--b", str(b), "--allow-shuffled-contrast",
    ]) == 1
    assert "FAILED its own negative-control gate" in capsys.readouterr().err


def test_a_z_below_the_bar_that_run_was_gated_at_is_not_a_failure() -> None:
    """⚑ The BAR comes off the artifact, not off ``NEGATIVE_CONTROL_Z``. A run
    given ``--negative-control-z 50`` passed at z=41.7, and re-judging its banked
    z against this module's constant would answer a question nobody asked."""
    meta = {"shuffled_targets_seed": 0, "negative_control_z": 41.7,
            "negative_control_max_z": 50.0}
    failed, z, bar = lc0_control_eval._failed_negative_control(meta)
    assert (failed, z, bar) == (False, 41.7, 50.0)
    assert lc0_control_eval._failed_negative_control(
        {"shuffled_targets_seed": 0, "negative_control_z": 41.7},
    )[0] is True, "with no banked bar it falls back to the module constant"


def test_an_artifact_predating_the_provenance_field_still_compares(
    tmp_path: Path,
) -> None:
    """Backward compatibility, and in the SAFE direction: a file written before
    ``shuffled_targets_seed`` existed reads as real targets, so the slope gates
    still apply to it rather than it being refused outright."""
    a, b = _paired_scores(tmp_path, meta_a={"checkpoint": "a"}, meta_b={"checkpoint": "b"})
    assert lc0_control_eval.main(["compare", "--a", str(a), "--b", str(b)]) == 0


def test_compare_refuses_a_checkpoint_the_driver_disqualified(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑⚑ THE FIELD THIS PR CREATED, ONE FILE SHORT OF THE GATE THAT NEEDS IT.

    `lc0_control_train.py` stamps `valid_control: false` for a run launched with
    `--allow-arch-drift` ("this is NOT production's architecture"),
    `--allow-leak`, no purity receipt, no mid checkpoint, or `--steps` under
    `warmup_steps`. `cmd_score` banked twelve meta keys and that was not one of
    them, so a checkpoint the driver itself disqualified scored clean and
    reported as the primary slope at exit 0. Found by independent review of
    #438: the PR establishes the pattern — bank the provenance, refuse before
    the arithmetic — and stopped at the field it had just introduced.

    NOT waivable: `--allow-unrecorded-validity` covers "no record", not "the
    driver said no".
    """
    a, b = _paired_scores(
        tmp_path,
        meta_a={"checkpoint": "a"},
        meta_b={"checkpoint": "b", "valid_control": False,
                "validity_problems": ["--allow-arch-drift: NOT production's net"]},
    )
    assert lc0_control_eval.main([
        "compare", "--a", str(a), "--b", str(b),
        "--allow-unrecorded-validity", "--allow-shuffled-contrast",
    ]) == 1
    captured = capsys.readouterr()
    assert "THE DRIVER DISQUALIFIED" in captured.err
    assert "--allow-arch-drift" in captured.err, "the recorded reason must survive"
    assert "delta" not in captured.out, (
        "the slope must not be printed for a disqualified checkpoint"
    )


def test_compare_refuses_an_artifact_with_no_validity_record_but_it_is_waivable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Absent is not clean — and the waiver has its OWN name.

    A score file with no `valid_control` cannot tell a clean run from one
    launched with `--allow-leak`, so the default is a refusal. It is waivable,
    because artifacts predating the field exist and reading one is a legitimate
    (declared) choice; it is NOT waivable by `--allow-shuffled-contrast`,
    because a waiver that clears more than its name says is how a gate becomes
    a decoration.
    """
    a, b = _paired_scores(
        tmp_path, meta_a={"checkpoint": "a", "valid_control": None},
        meta_b={"checkpoint": "b"},
    )
    assert lc0_control_eval.main(["compare", "--a", str(a), "--b", str(b)]) == 1
    assert "carries NO validity record" in capsys.readouterr().err
    # The unrelated waiver must NOT clear it.
    assert lc0_control_eval.main([
        "compare", "--a", str(a), "--b", str(b), "--allow-shuffled-contrast",
    ]) == 1
    capsys.readouterr()
    # Its own flag does, and says so.
    assert lc0_control_eval.main([
        "compare", "--a", str(a), "--b", str(b), "--allow-unrecorded-validity",
    ]) == 0
    assert "--allow-unrecorded-validity:" in capsys.readouterr().out


def test_compare_refuses_a_run_gated_more_leniently_than_the_module_floor(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑⚑ THE "UNWAIVABLE" REFUSAL WAS WAIVABLE FROM THE SCORE SIDE.

    Reading the bar off the artifact is right — re-judging a banked z against
    whatever this module's constant is at read time answers a question nobody
    asked. But `--negative-control-z` had no ceiling and `cmd_score` banked
    whatever it was handed, so `score --shuffle-targets --negative-control-z
    1e9` wrote an artifact whose control sat 41.7 sigma above the floor and was
    recorded as PASSING, and `compare` then printed the pre-fix slope at exit 0.
    Measured by the independent reviewer of #438, end-to-end.

    The bar is a CLAIM, not a measurement, so a run gated more leniently than
    `NEGATIVE_CONTROL_Z` is its own refusal.
    """
    a, b = _paired_scores(
        tmp_path, meta_a={"checkpoint": "a"},
        meta_b={"checkpoint": "b", "negative_control_z": 41.7,
                "negative_control_max_z": 1e9},
    )
    assert lc0_control_eval.main([
        "compare", "--a", str(a), "--b", str(b), "--allow-shuffled-contrast",
    ]) == 1
    captured = capsys.readouterr()
    assert "MORE LENIENTLY" in captured.err
    assert "delta" not in captured.out


def test_the_passing_readout_shows_the_negative_control_gate_it_judged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """This file's own standard, applied to itself.

    "⚑ PRINTED, not merely checked ... a gate that stopped running looks
    identical to a gate that ran and passed" — `compare` printed the
    target-provenance label and then judged the negative control in silence, so
    a reader of a PASSING readout could not see that the second gate ran, let
    alone at what bar.
    """
    a, b = _paired_scores(
        tmp_path,
        meta_a={"checkpoint": "a", "negative_control_z": 1.25,
                "negative_control_max_z": 5.0},
        meta_b={"checkpoint": "b"},
    )
    assert lc0_control_eval.main(["compare", "--a", str(a), "--b", str(b)]) == 0
    out = capsys.readouterr().out
    assert "neg-control: z=+1.25 vs bar 5" in out
    assert "neg-control: none (real targets)" in out


def test_compare_refuses_a_zero_discordance_comparison(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑ REVIEW F8. ``var = (b + c - (c-b)^2/n)/n^2`` is 0 at ``b == c == 0``,
    so the halfwidth bar — the only thing enforcing the prereg's n — passed at
    ANY n, including a file compared against itself."""
    hits = (np.arange(50) % 3 == 0).astype(np.uint8)
    a = _scores(tmp_path / "a.npz", hits)
    b = _scores(tmp_path / "b.npz", hits.copy(), role="last")
    assert lc0_control_eval.main(["compare", "--a", str(a), "--b", str(b)]) == 1
    err = capsys.readouterr().err
    assert "discordant on ZERO" in err
    assert "below the prereg's resolution point" in err


def test_the_shuffled_target_control_permutes_targets_not_hits() -> None:
    """⚑ Prereg guard 1. Permuting the HIT VECTOR preserves the rate exactly and
    gives a negative control that cannot fail; permuting the TARGETS does not.
    """
    predicted = np.arange(1_000, dtype=np.int64) % 20
    target = predicted.copy()
    permuted = target[np.random.default_rng(0).permutation(target.size)]
    assert (predicted == target).mean() == 1.0
    assert (predicted == permuted).mean() < 0.2


# ── compare: WHICH checkpoint, and from WHICH trajectory ──────────────────────


def test_the_summary_binds_its_verdict_to_the_checkpoints_it_wrote(
    tmp_path: Path,
) -> None:
    """⚑⚑ Codex 3794881256. ``valid_control`` is a verdict about a RUN and the
    artifact named no FILE, so ``--summary <valid run>/summary.json --checkpoint
    <other run>/checkpoint.pt`` banked ``valid_control: true`` for a checkpoint
    that summary had never seen — including one from a disqualified run.

    Writer AND reader, on artifacts the real driver wrote: per AMENDMENT 5's
    first finding, a field checked only against a hand-built dict is a field
    whose writer is untested.

    ⚑ The second run uses a different ``--seed``, deliberately. The run id is a
    digest OVER THE CHECKPOINT BYTES, so two runs with the same seed, config and
    corpus are bit-identical and share it — which is correct (they are the same
    trajectory) and would make this test about nothing.
    """
    shards = _write_shards(tmp_path / "rows", list(range(16)))
    out = tmp_path / "run"
    assert _run(tmp_path, out, shards) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    banked = {record["role"]: record for record in summary["checkpoints"]}
    assert set(banked) == {"mid", "last"}
    assert summary["run_id"]
    import hashlib

    assert banked["last"]["sha256"] == hashlib.sha256(
        (out / "checkpoint.pt").read_bytes(),
    ).hexdigest(), "the banked digest must be of the FILE, not of a description"

    for role, path in (("mid", out / "checkpoint_mid.pt"),
                       ("last", out / "checkpoint.pt")):
        read = lc0_control_eval.read_training_validity(out / "summary.json", path)
        assert read["checkpoint_role"] == role
        assert read["run_id"] == summary["run_id"]
        assert "verified" in str(read["checkpoint_identity"])

    other = tmp_path / "run2"
    assert lc0_control_train.main([
        "--config", str(_tiny_config(tmp_path)), "--shards", str(shards),
        "--out-dir", str(other), "--steps", "4", "--batch-size", "4",
        "--device", "cpu", "--no-compile", "--allow-arch-drift",
        "--allow-invalid-control", "--seed", "1",
    ]) == 0
    foreign = json.loads((other / "summary.json").read_text(encoding="utf-8"))
    assert foreign["run_id"] != summary["run_id"], (
        "two trajectories must not share a run id, or the compare gate is a "
        "constant"
    )
    with pytest.raises(SystemExit) as excinfo:
        lc0_control_eval.read_training_validity(
            out / "summary.json", other / "checkpoint.pt",
        )
    assert "does not describe" in str(excinfo.value)


def test_staging_refuses_a_non_symlink_intruder_rather_than_deleting_it(
    tmp_path: Path,
) -> None:
    """⚑ N5. `stage_shards` cleared only symlinks, so a real `.zarr` copied into
    `staged_shards` (or a partial write) survived into the next run's pool —
    `iter_shard_paths` reads it, and the trained corpus would not be the one
    `summary.json`'s `corpus.shard_dirs` names. Refused, not deleted: nothing on
    this path removes data it did not create."""
    shards = _write_shards(tmp_path / "rows", list(range(8)))
    staging = tmp_path / "staged" / "staged_shards"
    staging.mkdir(parents=True)
    intruder = staging / "shard_000999.zarr"
    intruder.mkdir()
    (intruder / "sentinel").write_text("not a symlink", encoding="utf-8")
    with pytest.raises(ValueError, match="not symlinks"):
        lc0_control_train.stage_shards([shards], staging)
    assert (intruder / "sentinel").is_file(), "the intruder must survive"


def test_the_score_artifact_carries_the_cluster_keys_and_compare_prints_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The BANKING half wired end to end, and the ESTIMATOR explicitly deferred.

    `compare`'s CI is row-level and the rows are not independent, so the readout
    must SAY that rather than leave it implicit — the pre-committed 0.392 pp bar
    is derived under independence. Estimating the design effect needs the real
    corpus; carrying the key does not.
    """
    rows = 12
    shards = _write_shards(tmp_path / "held", list(range(rows)))
    rc, frozen = _freeze(tmp_path, shards, sample=rows)
    assert rc == 0
    out = tmp_path / "run"
    assert _run(tmp_path, out, shards) == 0

    def _rigged(*_args: Any, **_kwargs: Any) -> tuple[Any, int, Any, Any]:
        moves = np.arange(rows, dtype=np.int64) % 3
        return np.ones(rows, dtype=np.uint8), 0, moves, moves

    monkeypatch.setattr(lc0_control_eval, "_score_rows", _rigged)
    paths = []
    for role in ("checkpoint_mid.pt", "checkpoint.pt"):
        path = tmp_path / f"{role}.npz"
        assert lc0_control_eval.main([
            "score", "--config", str(_tiny_config(tmp_path)),
            "--frozen", str(frozen), "--shards", str(shards),
            "--checkpoint", str(out / role), "--summary", str(out / "summary.json"),
            "--out", str(path), "--device", "cpu",
        ]) == 0
        paths.append(path)
    banked = np.load(paths[0], allow_pickle=True)
    payload = json.loads(Path(frozen).read_text(encoding="utf-8"))
    assert list(banked["cluster_keys"]) == payload["cluster_keys"], (
        "the score must carry the frozen set's keys, row-aligned"
    )
    meta = json.loads(str(banked["meta"][0]))
    assert meta["distinct_clusters"] == payload["distinct_clusters"]
    assert meta["cluster_keys_complete"] is True

  # ⚑ `compare` on these two is (correctly) REFUSED — this smoke banks
  # `valid_control: false`, which no flag waives — so the PRINT half is driven on
  # fixtures. Asserting it here would have needed a waiver that does not exist,
  # which is the gate working, not the test being awkward.
    capsys.readouterr()
    complete = {"distinct_clusters": 5_000, "cluster_keys_complete": True}
    a, b = _paired_scores(
        tmp_path, meta_a={"checkpoint": "a", **complete},
        meta_b={"checkpoint": "b", **complete},
    )
    assert lc0_control_eval.main(["compare", "--a", str(a), "--b", str(b)]) == 0
    out_text = capsys.readouterr().out
    assert "game clusters        5000 over 100000 rows (20.0 rows/cluster)" in out_text
    assert "OPTIMISTIC by the design effect" in out_text

  # ⚑⚑ REVIEW F5 — AND A PARTIAL KEY SET IS ITS OWN STATE. `cluster_keys_complete`
  # was banked and read by NOTHING: the wave-5 mutant that forced it False changed
  # no output and no exit code. It now REFUSES, under its own flag, and a partial
  # set never prints a rows/cluster ratio — `distinct_clusters` counts only
  # non-empty keys, so the ratio would overstate the cluster size in a known
  # direction.
    partial_dir = tmp_path / "partial"
    partial_dir.mkdir()
    a3, b3 = _paired_scores(
        partial_dir,
        meta_a={"checkpoint": "a", "distinct_clusters": 5_000,
                "cluster_keys_complete": False},
        meta_b={"checkpoint": "b", **complete},
    )
    assert lc0_control_eval.main(["compare", "--a", str(a3), "--b", str(b3)]) == 1
    refused = capsys.readouterr()
    assert "INCOMPLETE cluster-key set" in refused.err
    assert "--a carries" in refused.err, "the artifact that is partial is named"
    assert "delta (B - A)" not in refused.out, (
        "a refused pair must not put a slope on stdout"
    )
    assert lc0_control_eval.main([
        "compare", "--a", str(a3), "--b", str(b3), "--allow-partial-clusters",
    ]) == 0
    waived = capsys.readouterr().out
    assert "game clusters        PARTIAL" in waived
    assert "5000 over 100000 rows" not in waived, (
        "no cluster ratio may be printed from a key set with holes"
    )
    assert "delta (B - A)" in waived

  # ...and an artifact with no key says THAT, rather than printing nothing.
    unrecorded_dir = tmp_path / "unrecorded"
    unrecorded_dir.mkdir()
    a2, b2 = _paired_scores(
        unrecorded_dir, meta_a={"checkpoint": "a"}, meta_b={"checkpoint": "b"},
    )
    assert lc0_control_eval.main(["compare", "--a", str(a2), "--b", str(b2)]) == 0
    assert "game clusters        UNRECORDED" in capsys.readouterr().out


def test_two_checkpoints_with_identical_bytes_cannot_be_assigned_a_role(
    tmp_path: Path,
) -> None:
    """⚑ The sha→role lookup was FIRST-MATCH on a content key, so a byte-identical
    LAST would be labelled ``mid`` and the pair refused as ``{mid}`` — a refusal
    naming the wrong thing. Two identical checkpoints mean nothing changed between
    the mid step and the end, which is worth saying out loud."""
    ckpt = tmp_path / "checkpoint.pt"
    torch.save({"model": {}}, ckpt)
    digest = lc0_control_eval.sha256_file(ckpt)
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "valid_control": True, "run_id": "trajectory-0",
        "checkpoints": [
            {"role": "mid", "path": str(ckpt), "sha256": digest},
            {"role": "last", "path": str(ckpt), "sha256": digest},
        ],
    }), encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        lc0_control_eval.read_training_validity(summary, ckpt)
    message = str(excinfo.value)
    assert "SAME sha256" in message
    assert "mid, last" in message


def test_score_banks_the_trajectory_identity_and_the_heldout_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The WRITER side of both new gates, on an artifact the real ``cmd_score``
    wrote — ``_score_rows`` is stubbed because a real 100k score needs a GPU,
    everything downstream of it is the real path.

    Renaming either key on the writer alone is exactly the mutant AMENDMENT 5's
    first finding describes: the compare-side tests below would stay green while
    every real artifact read back ``None``.
    """
    rows = 12
    shards = _write_shards(tmp_path / "held", list(range(rows)))
    rc, frozen = _freeze(tmp_path, shards, sample=rows)
    assert rc == 0
    out = tmp_path / "run"
    assert _run(tmp_path, out, shards) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))

    def _rigged(*_args: Any, **_kwargs: Any) -> tuple[Any, int, Any, Any]:
        moves = np.arange(rows, dtype=np.int64) % 3
        return np.ones(rows, dtype=np.uint8), 0, moves, moves

    monkeypatch.setattr(lc0_control_eval, "_score_rows", _rigged)
    assert lc0_control_eval.main([
        "score", "--config", str(_tiny_config(tmp_path)), "--frozen", str(frozen),
        "--shards", str(shards), "--checkpoint", str(out / "checkpoint_mid.pt"),
        "--summary", str(out / "summary.json"),
        "--out", str(tmp_path / "s.npz"), "--device", "cpu",
    ]) == 0
    meta = json.loads(str(np.load(tmp_path / "s.npz", allow_pickle=True)["meta"][0]))
    assert meta["run_id"] == summary["run_id"]
    assert meta["checkpoint_role"] == "mid"
    assert meta["checkpoint_sha256"] == next(
        r["sha256"] for r in summary["checkpoints"] if r["role"] == "mid"
    )
  # ⚑ And the HELD-OUT population, recomputed from the frozen artifact's own
  # `sources` rather than trusted from its stamp: this frozen set is one hour,
  # not the preregistered six.
    assert meta["heldout_source_selection_problems"], meta
    assert "not the preregistered 6 hourly tars" in \
        meta["heldout_source_selection_problems"][0]


def test_a_row_with_no_game_id_gets_no_cluster_key() -> None:
    """⚑⚑ REVIEW F4. `game_id` is an OPTIONAL shard field with a `has_game_id`
    mask, so an unset row carries the int64 FILL VALUE 0 — and the first version
    read the column without the mask, emitting `"<source>#0"`: a real-looking key
    that MERGED every unkeyed row into game 0's cluster while
    `cluster_keys_complete` reported True. The docstring promised this case was
    handled; it covered an absent COLUMN and never an unset ROW, and the banked
    artifact — unrecoverable after the freeze — could not tell the two apart.
    """
    from chess_anti_engine.eval.lc0_control_rows import game_cluster_keys

  # Rows 0 and 2 carry a real id; rows 1 and 3 are unset and read back as 0.
    arrs = {
        "game_id": np.array([7, 0, 7, 0], dtype=np.int64),
        "has_game_id": np.array([1, 0, 1, 0], dtype=np.uint8),
    }
    keys = game_cluster_keys(arrs, source="hour0", rows=4)
    assert keys == ["hour0#7", "", "hour0#7", ""], keys
  # ⚑ THE MUTANT'S OWN OUTPUT, named: without the mask the unset rows would read
  # as game 0 of this source, which is a key a real row could also have.
    assert "hour0#0" not in keys, (
        "an unset row must not be given the fill value's key"
    )
  # An absent MASK still means every row is keyed (older shards), and an absent
  # COLUMN means none is.
    assert game_cluster_keys(
        {"game_id": np.array([1, 2], dtype=np.int64)}, source="h", rows=2,
    ) == ["h#1", "h#2"]
    assert game_cluster_keys({}, source="h", rows=2) == ["", ""]


def test_the_frozen_sample_is_proportional_to_each_sources_rows() -> None:
    """⚑ THREAD 3795733611. `remaining // sources_left` gave every source an
    approximately EQUAL quota, so on six hours of unequal size the SMALL hours
    were overrepresented in the frozen population and the result depended on
    source ORDER whenever a late source could not fill its equal share — while
    the docstring said "proportionally across sources".

    Asserted on a deliberately lopsided pool, where the equal-quota rule and the
    proportional one give different answers (that difference is what makes this
    test able to fail).
    """
    from chess_anti_engine.eval.lc0_control_rows import _stratified_sample

    sizes = [1_000, 100, 100, 100, 100, 100]
    per_source = [
        [(f"s{i}r{j}", f"in{i}r{j}", f"h{i}") for j in range(size)]
        for i, size in enumerate(sizes)
    ]
    sample = 300
    picked = _stratified_sample(per_source, sample=sample, seed=0)
    assert len(picked) == sample, "largest-remainder must be exact"
    counts = [sum(1 for row in picked if row[2] == f"h{i}") for i in range(len(sizes))]
    total = sum(sizes)
    for index, (size, got) in enumerate(zip(sizes, counts, strict=True)):
        want = sample * size / total
        assert abs(got - want) <= 1, (index, got, want, counts)
  # The equal-quota rule this replaces would have given the big hour 50 of 300.
    assert counts[0] > 100, (
        f"the 1000-row hour must dominate a proportional draw, got {counts}"
    )
  # ⚑ ORDER-INDEPENDENT: reversing the sources reverses the counts and nothing
  # else, which the `remaining // sources_left` rule did not satisfy.
    reversed_counts = [
        sum(1 for row in _stratified_sample(
            list(reversed(per_source)), sample=sample, seed=0,
        ) if row[2] == f"h{i}")
        for i in range(len(sizes))
    ]
    assert reversed_counts == counts, (counts, reversed_counts)


def test_the_population_role_is_banked_and_compare_requires_the_one_it_was_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑⚑ THREAD 3795733605. The prereg reads TWO deltas — held-out and train —
    and `score` classified EVERY artifact by the HELD-OUT six-source rule while
    banking no statement of which population it scored. So a train-side sample
    that happened to span six directories banked an empty
    `heldout_source_selection_problems` and could be presented as
    `Delta_heldout`, and an honest train sample spanning more directories needed a
    MISLEADING held-out waiver to compare at all.

    Both halves here: the role is WRITTEN by the real scorer (and the six-source
    rule is not applied to a train-side artifact), and `compare` REFUSES a
    population it was not asked for — unwaivably.
    """
    rows = 12
    shards = _write_shards(tmp_path / "held", list(range(rows)))
    rc, frozen = _freeze(tmp_path, shards, sample=rows)
    assert rc == 0
    out = tmp_path / "run"
    assert _run(tmp_path, out, shards) == 0

    def _rigged(*_args: Any, **_kwargs: Any) -> tuple[Any, int, Any, Any]:
        moves = np.arange(rows, dtype=np.int64) % 3
        return np.ones(rows, dtype=np.uint8), 0, moves, moves

    monkeypatch.setattr(lc0_control_eval, "_score_rows", _rigged)

    def score(name: str, *extra: str) -> dict[str, Any]:
        path = tmp_path / f"{name}.npz"
        assert lc0_control_eval.main([
            "score", "--config", str(_tiny_config(tmp_path)),
            "--frozen", str(frozen), "--shards", str(shards),
            "--checkpoint", str(out / "checkpoint_mid.pt"),
            "--summary", str(out / "summary.json"),
            "--out", str(path), "--device", "cpu", *extra,
        ]) == 0
        return json.loads(str(np.load(path, allow_pickle=True)["meta"][0]))

  # The default is the held-out population, and the six-source rule applies.
    heldout = score("heldout")
    assert heldout["population"] == "heldout"
    assert heldout["heldout_source_selection_problems"], heldout
  # ⚑ THE SAME ONE-HOUR FROZEN SET, scored as the TRAIN-side sample: the held-out
  # rule is a held-out rule, so it does NOT fire — which is what made the honest
  # train artifact need a misleading waiver before.
    train = score("train", "--population", "train")
    assert train["population"] == "train"
    assert train["heldout_source_selection_problems"] == [], train

  # ...and `compare` requires the role it was asked for.
    capsys.readouterr()
    a, b = _paired_scores(
        tmp_path, meta_a={"checkpoint": "a", "population": "train"},
        meta_b={"checkpoint": "b"},
    )
    assert lc0_control_eval.main(["compare", "--a", str(a), "--b", str(b)]) == 1
    err = capsys.readouterr().err
    assert "scored the TRAIN population but this comparison was asked for HELDOUT" \
        in err
    assert "Waivable by" not in err, "a population mismatch is unwaivable"
  # Both sides train, asked for train: accepted, and the readout says which.
    both_train = tmp_path / "both_train"
    both_train.mkdir()
    a2, b2 = _paired_scores(
        both_train,
        meta_a={"checkpoint": "a", "population": "train"},
        meta_b={"checkpoint": "b", "population": "train"},
        heldout_problems=None,
    )
    assert lc0_control_eval.main([
        "compare", "--a", str(a2), "--b", str(b2), "--population", "train",
    ]) == 0, "a train comparison must not need a held-out record at all"
    printed = capsys.readouterr().out
    assert "population           TRAIN" in printed
    assert "pop: train" in printed


def test_a_non_finite_halfwidth_override_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑ CODEX 3796113406. `--max-halfwidth-pp nan` was accepted by `float()`,
    switched the n floor off (`prereg_bar_in_force` false) and then compared
    `half_pp > nan` — FALSE for every finite halfwidth. Both gates off, exit 0,
    slope printed: a clamp is not a validator, and NaN propagates through every
    comparison as "no".
    """
    rng = np.random.default_rng(0)
    a = (rng.random(5_000) < 0.30).astype(np.uint8)
    b = (rng.random(5_000) < 0.33).astype(np.uint8)
    a_path = _scores(tmp_path / "a.npz", a)
    b_path = _scores(tmp_path / "b.npz", b, role="last")
    for bad in ("nan", "inf", "-inf", "0", "-1"):
  # ⚑ `--flag=value` form: argparse reads a bare `-1` as another OPTION.
        assert lc0_control_eval.main([
            "compare", "--a", str(a_path), "--b", str(b_path),
            f"--max-halfwidth-pp={bad}",
        ]) == 1, bad
        captured = capsys.readouterr()
        assert "is not a finite positive number" in captured.err, bad
        assert "delta (B - A)" not in captured.out, bad
  # ⚑ AND THE n FLOOR IS NO LONGER SWITCHED OFF BY THE PRESENCE OF AN OVERRIDE.
  # A LOOSER bar is a relaxation, not a re-derivation, so the floor stays in
  # force; only a bar at least as strict as the material one waives it.
    assert lc0_control_eval.main([
        "compare", "--a", str(a_path), "--b", str(b_path),
        "--max-halfwidth-pp", "10.0",
    ]) == 1
    assert "below the prereg's resolution point" in capsys.readouterr().err


def test_the_exact_mcnemar_tail_is_computable_at_the_preregs_own_n() -> None:
    """⚑ CODEX 3796113409. The enumerating version summed `math.comb(total, k)` —
    arbitrary-precision integers with tens of thousands of digits — once per k up
    to `min(b, c)`, so the readout the prereg says DECIDES the arm took tens of
    seconds at 20,000 discordant pairs and minutes at 100,000.

    The log-space path is checked two ways: it agrees with an EXACT rational
    reference across the switchover, and it finishes.
    """
    from fractions import Fraction

    def exact(b: int, c: int) -> float:
        total = b + c
        tail = Fraction(sum(math.comb(total, k) for k in range(min(b, c) + 1)),
                        2 ** total)
        return float(min(Fraction(1), 2 * tail))

  # The small-total path is unchanged and still bit-identical to scipy's answer.
    assert lc0_control_eval._exact_mcnemar_p(10, 30) == 0.0022214337732293643
    for b, c in ((300, 400), (500, 600), (700, 800), (1000, 1200), (1500, 1600)):
        assert lc0_control_eval._exact_mcnemar_p(b, c) == pytest.approx(
            exact(b, c), rel=1e-9,
        ), (b, c)
  # And the prereg-scale case, timed: the old implementation could not finish
  # this inside a test run.
    start = time.perf_counter()
    p_big = lc0_control_eval._exact_mcnemar_p(49_000, 51_000)
    elapsed = time.perf_counter() - start
    assert 0.0 < p_big < 1e-6, p_big
    assert elapsed < 2.0, f"the exact tail took {elapsed:.1f}s at 100,000 pairs"


def test_a_batch_size_deviation_is_terminal_for_compare_not_a_soft_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑⚑ WHAT "RECORDED RATHER THAN REFUSED" ACTUALLY BUYS, PINNED END TO END.

    `valid_control = not validity_problems` and `compare` refuses
    `valid_control: false` with NO waiver, so a `--batch-size` deviation is
    TERMINAL for the comparison — the driver's comments called it a soft record,
    which reads as "still comparable". The independent review executed this;
    it is now a test, so the policy cannot drift back into prose:

      * the run FINISHES and writes both checkpoints (what a plumbing smoke needs)
      * and no combination of waivers lets its score be paired.
    """
    rows = 12
    shards = _write_shards(tmp_path / "held", list(range(rows)))
    rc, frozen = _freeze(tmp_path, shards, sample=rows)
    assert rc == 0
    out = tmp_path / "small_batch"
    assert lc0_control_train.main([
        "--config", str(_tiny_config(tmp_path)), "--shards", str(shards),
        "--out-dir", str(out), "--steps", "4", "--batch-size", "2",
        "--device", "cpu", "--no-compile", "--allow-arch-drift",
        "--allow-invalid-control",
    ]) == 0
    assert (out / "checkpoint.pt").is_file(), "the run must still finish"
    assert (out / "checkpoint_mid.pt").is_file()

    def _rigged(*_args: Any, **_kwargs: Any) -> tuple[Any, int, Any, Any]:
        moves = np.arange(rows, dtype=np.int64) % 3
        return np.ones(rows, dtype=np.uint8), 0, moves, moves

    monkeypatch.setattr(lc0_control_eval, "_score_rows", _rigged)
    scores = []
    for role in ("checkpoint_mid.pt", "checkpoint.pt"):
        path = tmp_path / f"{role}.npz"
        assert lc0_control_eval.main([
            "score", "--config", str(_tiny_config(tmp_path)),
            "--frozen", str(frozen), "--shards", str(shards),
            "--checkpoint", str(out / role), "--summary", str(out / "summary.json"),
            "--out", str(path), "--device", "cpu",
        ]) == 0
        scores.append(path)
    meta = json.loads(str(np.load(scores[0], allow_pickle=True)["meta"][0]))
    assert meta["valid_control"] is False
    assert [p for p in meta["validity_problems"] if "--batch-size 2" in p]
    capsys.readouterr()
    assert lc0_control_eval.main([
        "compare", "--a", str(scores[0]), "--b", str(scores[1]),
        "--allow-shuffled-contrast", "--allow-unrecorded-validity",
        "--allow-unverified-trajectory", "--allow-non-prereg-heldout",
        "--allow-unrecorded-heldout", "--allow-underpowered",
    ]) == 1
    captured = capsys.readouterr()
    assert "THE DRIVER DISQUALIFIED" in captured.err
    assert "--batch-size 2" in captured.err, "the recorded reason must survive"
    assert "delta" not in captured.out


def test_compare_refuses_two_checkpoints_from_different_trajectories(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑⚑ The second half of Codex 3794881256: two INDIVIDUALLY VALID checkpoints
    are not the prereg's pair. Without a run id, ``compare`` would difference two
    runs' LAST checkpoints — which differ by their whole initialisation and data
    order — and report it as LAST vs MID-BUDGET.

    NOT waivable, and asserted with every waiver passed at once: a mismatch is a
    MEASUREMENT, not an absence, and ``--allow-unverified-trajectory`` covers
    only the absence.
    """
    a, b = _paired_scores(
        tmp_path,
        meta_a={"checkpoint": "run1/checkpoint_mid.pt", "run_id": "trajectory-1"},
        meta_b={"checkpoint": "run2/checkpoint.pt", "run_id": "trajectory-2"},
    )
    assert lc0_control_eval.main([
        "compare", "--a", str(a), "--b", str(b),
        "--allow-unverified-trajectory", "--allow-unrecorded-validity",
        "--allow-shuffled-contrast", "--allow-non-prereg-heldout",
    ]) == 1
    captured = capsys.readouterr()
    assert "DIFFERENT TRAJECTORIES" in captured.err
    assert "delta" not in captured.out, (
        "the slope must not be printed for a pair that is refused"
    )


def test_compare_refuses_a_pair_that_is_not_mid_and_last(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """One trajectory is not enough: the deciding statistic is the slope between
    the MID-BUDGET and the FINAL checkpoint, so two LASTs — or a random-init
    floor against a LAST — is a different reading wearing its name."""
    a, b = _paired_scores(
        tmp_path,
        meta_a={"checkpoint": "checkpoint.pt", "checkpoint_role": "last"},
        meta_b={"checkpoint": "checkpoint.pt", "checkpoint_role": "last"},
    )
    assert lc0_control_eval.main(["compare", "--a", str(a), "--b", str(b)]) == 1
    captured = capsys.readouterr()
    assert "not the prereg's MID/LAST pair" in captured.err
    assert "delta" not in captured.out


def test_compare_refuses_an_unidentified_pair_but_that_waiver_is_its_own(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Absent is not clean here either — and no OTHER flag clears it.

    ⚑ The one refusal in play is the identity one: the fixtures carry
    ``valid_control: True`` and real targets, so a green run under
    ``--allow-unverified-trajectory`` cannot be some other gate passing.
    """
    a, b = _paired_scores(
        tmp_path, meta_a={"checkpoint": "a", "run_id": None},
        meta_b={"checkpoint": "b"},
    )
    assert lc0_control_eval.main(["compare", "--a", str(a), "--b", str(b)]) == 1
    assert "NO trajectory identity" in capsys.readouterr().err
    for unrelated in ("--allow-shuffled-contrast", "--allow-unrecorded-validity",
                      "--allow-non-prereg-heldout"):
        assert lc0_control_eval.main([
            "compare", "--a", str(a), "--b", str(b), unrelated,
        ]) == 1, f"{unrelated} must not clear a refusal it does not name"
        capsys.readouterr()
    assert lc0_control_eval.main([
        "compare", "--a", str(a), "--b", str(b), "--allow-unverified-trajectory",
    ]) == 0
    out = capsys.readouterr().out
    assert "--allow-unverified-trajectory:" in out
    assert "delta" in out, "the waived path must still report the comparison"


@pytest.mark.parametrize(
    ("meta_a", "meta_b", "expected"),
    [
  # ⚑ THE REVIEWER'S OWN TWO VIOLATING INPUTS, verbatim from the independent
  # review of the third wave (rc 0 with the full slope printed on both).
        (
            {"checkpoint": "a", "run_id": None, "checkpoint_role": "last"},
            {"checkpoint": "b", "run_id": "trajectory-0", "checkpoint_role": "last"},
            "not the prereg's MID/LAST pair",
        ),
        (
            {"checkpoint": "a", "run_id": "trajectory-1", "checkpoint_role": None},
            {"checkpoint": "b", "run_id": "trajectory-2", "checkpoint_role": "last"},
            "DIFFERENT TRAJECTORIES",
        ),
    ],
)
def test_the_identity_waiver_leaves_the_other_two_refusals_in_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
    meta_a: dict[str, Any], meta_b: dict[str, Any], expected: str,
) -> None:
    """⚑⚑ THE WAIVER CLEARED TWO REFUSALS ITS OWN HELP CALLS UNWAIVABLE.

    The three identity checks were an `if / elif / elif` chain, so when EITHER
    artifact lacked identity the waivable branch matched and the two unwaivable
    branches NEVER EXECUTED. `--allow-unverified-trajectory` therefore cleared a
    LAST-vs-LAST pair and a two-trajectory pair — exactly the two failures the
    identity finding exists to stop — at rc 0 with the slope printed.

    ⚑ WHY THE WAVE'S OWN TESTS COULD NOT SEE IT, which is why these fixtures are
    shaped as they are: the mismatch test set BOTH `run_id`s (so `unidentified`
    was empty and the chain reached the mismatch branch), and the absence test's
    fixture pair already WAS `{mid, last}` (so the shadowed role branch had
    nothing to say). Each row here has ONE field absent and the OTHER field
    violating — the state the chain could not reach.
    """
    a, b = _paired_scores(tmp_path, meta_a=meta_a, meta_b=meta_b)
    assert lc0_control_eval.main([
        "compare", "--a", str(a), "--b", str(b),
        "--allow-unverified-trajectory",
    ]) == 1
    captured = capsys.readouterr()
    assert expected in captured.err
    assert "delta" not in captured.out, (
        "the slope must not be printed for a pair that is refused"
    )
  # ...and the absence itself IS waived, so the refusal above is the unwaivable
  # one speaking and not the waiver failing to apply.
    assert "--allow-unverified-trajectory:" in captured.out


def test_compare_refuses_an_artifact_with_no_heldout_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑ ABSENT WAS READING AS PREREGISTERED. `compare` read the field as
    `meta.get(...) or ()`, so an artifact written before it existed produced no
    problem, no banner and rc 0 — the same absent⇒clean direction `cmd_score`
    recomputes to avoid one level down, and the opposite of how the same function
    treats an absent `run_id`.

    Its own flag: "this set is declaredly not the prereg's six" and "nothing says
    which population this scored" are different claims.
    """
    a, b = _paired_scores(
        tmp_path, meta_a={"checkpoint": "a"}, meta_b={"checkpoint": "b"},
        heldout_problems=None,
    )
    assert lc0_control_eval.main(["compare", "--a", str(a), "--b", str(b)]) == 1
    assert "NO held-out population record" in capsys.readouterr().err
    for unrelated in ("--allow-non-prereg-heldout", "--allow-unverified-trajectory",
                      "--allow-unrecorded-validity"):
        assert lc0_control_eval.main([
            "compare", "--a", str(a), "--b", str(b), unrelated,
        ]) == 1, f"{unrelated} must not clear a refusal it does not name"
        capsys.readouterr()
    assert lc0_control_eval.main([
        "compare", "--a", str(a), "--b", str(b), "--allow-unrecorded-heldout",
    ]) == 0
    out = capsys.readouterr().out
    assert "--allow-unrecorded-heldout:" in out
    assert "delta" in out


def test_the_passing_readout_names_the_role_and_the_trajectory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """This file's own standard again: the mid/last binding is a GATE now, and a
    reader of a passing readout could not see that it ran."""
    a, b = _paired_scores(
        tmp_path, meta_a={"checkpoint": "a"}, meta_b={"checkpoint": "b"},
    )
    assert lc0_control_eval.main(["compare", "--a", str(a), "--b", str(b)]) == 0
    out = capsys.readouterr().out
    assert "role: mid" in out
    assert "role: last" in out
    assert "run: trajecto" in out, "the run id must be on the line, truncated"


def test_compare_refuses_a_non_preregistered_heldout_population(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑ Codex 3794881273's reader side. The stamp is not decoration: a score
    over a held-out set that is not the preregistered six hours is refused, and
    the waiver is its own flag so declaring the population does not also clear an
    unrecorded validity or a shuffled contrast."""
    a, b = _paired_scores(
        tmp_path, meta_a={"checkpoint": "a"},
        meta_b={"checkpoint": "b", "heldout_source_selection_problems": [
            "the frozen set was built from 1 source directory (hour0), not the "
            "preregistered 6 hourly tars.",
        ]},
    )
    assert lc0_control_eval.main(["compare", "--a", str(a), "--b", str(b)]) == 1
    err = capsys.readouterr().err
    assert "NON-PREREGISTERED held-out population" in err
    assert "1 source directory" in err, "the recorded reason must survive"
    assert lc0_control_eval.main([
        "compare", "--a", str(a), "--b", str(b), "--allow-shuffled-contrast",
    ]) == 1, "an unrelated waiver must not clear it"
    capsys.readouterr()
    assert lc0_control_eval.main([
        "compare", "--a", str(a), "--b", str(b), "--allow-non-prereg-heldout",
    ]) == 0
    assert "--allow-non-prereg-heldout:" in capsys.readouterr().out


# ── the eval scorer's checkpoint load ─────────────────────────────────────────


def test_scoring_refuses_a_checkpoint_that_does_not_fit_the_config(
    tmp_path: Path,
) -> None:
    """⚑ Codex #3. ``Trainer.load`` tolerates a partial load down to 50% of the
    keys, so a config/arch drift scores a HYBRID of trained and fresh tensors
    under the checkpoint's name. Top-1 is this arm's only yardstick."""
    from chess_anti_engine.model import build_model, model_config_from_flat_config
    from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

    (tmp_path / "wide_dir").mkdir()
    (tmp_path / "narrow_dir").mkdir()
    wide = _tiny_config(tmp_path / "wide_dir")
    raw = yaml.safe_load(wide.read_text(encoding="utf-8"))
    raw["model"]["embed_dim"] = 64
    wide_path = tmp_path / "wide.yaml"
    wide_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    flat = flatten_run_config_defaults(load_yaml_file(str(wide_path)))
    model = build_model(model_config_from_flat_config(flat))
    ckpt = tmp_path / "wide.pt"
    torch.save({"model": model.state_dict()}, ckpt)

    with pytest.raises(RuntimeError, match="require_complete"):
        lc0_control_eval._load_trainer(
            _tiny_config(tmp_path / "narrow_dir"), ckpt, "cpu", seed=0,
        )


def test_the_random_init_floor_is_reproducible_from_its_seed(
    tmp_path: Path,
) -> None:
    """⚑ Review F4 / Codex #4. Unseeded, four draws of the identical command
    spanned 2.18 pp — 5.6x the 0.392 pp material bar — straddling chance."""
    config = _tiny_config(tmp_path)

    def fingerprint(seed: int) -> float:
  # ⚑ Over EVERY parameter. The first tensor alone is a LayerNorm-style
  # constant init and is identical under any seed, so a probe that reads only
  # it passes whether or not the seed is applied — a check that cannot fail.
        trainer = lc0_control_eval._load_trainer(config, None, "cpu", seed=seed)
        return float(
            sum(float(p.detach().double().sum()) for p in trainer.model.parameters()),
        )

    assert fingerprint(7) == fingerprint(7)
    assert fingerprint(7) != fingerprint(8)


def test_the_guard_message_names_the_outcome_share_not_only_the_leak() -> None:
    """Both bars, in one place, because they fail on disjoint configurations."""
    from chess_anti_engine.train.value_blend_guard import (
        assert_no_silent_outcome_fallback,
    )

    all_outcome = value_blend_readout(
        sf_wdl_frac=0.0, search_wdl_frac=0.0,
        sf_effective_rows=8.0, search_effective_rows=8.0, batch_rows=8.0,
    )
    assert all_outcome.leaked_to_outcome == 0.0
    with pytest.raises(ValueBlendMisconfigured, match="RAW GAME OUTCOME"):
        assert_no_silent_outcome_fallback(all_outcome)
