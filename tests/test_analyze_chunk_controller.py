from __future__ import annotations

import errno
import hashlib
import importlib
import importlib.machinery
import inspect
import json
import os
import py_compile
import shutil
import stat
import subprocess
import sys
import sysconfig
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.eval.audit import (
    AuditPosition,
    legal_full_indices,
    phase_bucket,
    position_key,
)
from chess_anti_engine.moves import move_to_index
from scripts import analyze_chunk_controller as controller_module
from scripts import backtest_chunk_trajectory as trajectory_module
from scripts import chunk_trajectory_publication as publication_module
from scripts.approved_syzygy import (
    APPROVED_SYZYGY_CHECKSUM_CATALOG_ENTRIES_SHA256,
    APPROVED_SYZYGY_CHECKSUM_CATALOG_RAW_SHA256,
    APPROVED_SYZYGY_COMPONENTS,
    ApprovedSyzygyComponent,
    checksum_catalog_entries_sha256,
    filename_size_sha256,
)
from scripts.analyze_chunk_controller import (
    Transition,
    _complexity_continue,
    _evaluation_fold_ids,
    _fold_stage_counts,
    _rollout_selected_indices,
    _source_group,
    _stage_counts,
    _require_bootstrap_resolution,
    _require_safe_output_path,
    _score,
    _update_stability,
    analyze,
    cluster_bootstrap_delta,
    evaluate_horizon,
    evaluate_reachable_rollout,
    grouped_folds,
    held_horizon_predictions,
    load_transitions,
)
from scripts.backtest_chunk_trajectory import (
    _acquire_output_lock,
    _acquire_output_locks,
    _checkpoint_params_candidates,
    _open_authenticated_input,
    _load_authenticated_model_inputs,
    _params_candidate_inventory,
    _publish_output,
    _require_new_output_pair,
    _require_safe_preregistration_path,
    _require_safe_output_paths,
    _require_search_take_effect,
    _tablebase_inventory,
    _validate_registry_search_values,
)
from scripts.check_c_extensions_fresh import extension_spec


_TEST_GIT_FILES: dict[str, bytes] = {}


def test_approved_syzygy_layout_pins_remain_stable() -> None:
    assert (
        ApprovedSyzygyComponent(
            "syzygy_3-4-5", 510, 145, 655, 73_818_025_392,
            "796607668b96e5128e5493cb6fcbb8c0b1155ffd1f686a6e7e12b4a4fea61b78",
        ),
        ApprovedSyzygyComponent(
            "syzygy_6", 365, 365, 730, 160_225_616_032,
            "32b67df63f6005216c29778f3176d242a1f531aa1f225e4916bbc5a1ccfa6618",
        ),
    ) == APPROVED_SYZYGY_COMPONENTS
    assert APPROVED_SYZYGY_CHECKSUM_CATALOG_RAW_SHA256 == (
        "e5039f7d0a63bb8607cc2342357353f162f40ae601853f116e763809003683ab"
    )
    assert APPROVED_SYZYGY_CHECKSUM_CATALOG_ENTRIES_SHA256 == (
        "4e2a2577cc0bdae8025f71dc64b69bf0caec422ee220f2a2995f0ca122ad0d62"
    )


@pytest.fixture(autouse=True)
def _authenticate_test_preregistration(monkeypatch: pytest.MonkeyPatch) -> None:
    _TEST_GIT_FILES.clear()
    approved_components, catalog_rows = _synthetic_approved_syzygy_contract()
    catalog_entries_sha256 = checksum_catalog_entries_sha256(catalog_rows)
    catalog_wdl_count = sum(name.endswith(".rtbw") for name, _digest in catalog_rows)
    catalog_dtz_count = sum(name.endswith(".rtbz") for name, _digest in catalog_rows)
    for module in (controller_module, trajectory_module):
        monkeypatch.setattr(module, "_APPROVED_SYZYGY_COMPONENTS", approved_components)
        monkeypatch.setattr(
            module, "APPROVED_SYZYGY_CHECKSUM_CATALOG_RAW_SHA256", "a" * 64,
        )
        monkeypatch.setattr(
            module, "APPROVED_SYZYGY_CHECKSUM_CATALOG_SIZE", 1,
        )
        monkeypatch.setattr(
            module, "APPROVED_SYZYGY_CHECKSUM_CATALOG_ENTRY_COUNT",
            len(catalog_rows),
        )
        monkeypatch.setattr(
            module, "APPROVED_SYZYGY_CHECKSUM_CATALOG_WDL_COUNT",
            catalog_wdl_count,
        )
        monkeypatch.setattr(
            module, "APPROVED_SYZYGY_CHECKSUM_CATALOG_DTZ_COUNT",
            catalog_dtz_count,
        )
        monkeypatch.setattr(
            module, "APPROVED_SYZYGY_CHECKSUM_CATALOG_ENTRIES_SHA256",
            catalog_entries_sha256,
        )
    monkeypatch.setattr(
        controller_module,
        "_git_file_at_commit",
        lambda _commit, relative_path: _TEST_GIT_FILES.get(relative_path),
    )
    monkeypatch.setattr(
        controller_module,
        "_git_python_tree_at_commit",
        lambda _commit: {
            path: (
                "100644", "blob",
                hashlib.sha1(
                    f"blob {len(content)}\0".encode("ascii") + content
                ).hexdigest(),
            )
            for path, content in _TEST_GIT_FILES.items()
            if path.endswith(".py")
        },
    )
    monkeypatch.setattr(
        controller_module,
        "_preregistration_is_only_source_change",
        lambda _source, _producer, _relative_path: True,
    )


def _state(value: float = 0.0, *, gap: float = 0.1, stable: float = 0.0) -> dict[str, float]:
    return {
        "visit_gap": gap,
        "visit_entropy": 0.5,
        "q_gap": 0.0,
        "q_gap_missing": 0.0,
        "bestmove_flip": 0.0,
        "stable_chunks": stable,
        "q_drift": value,
        "q_drift_missing": 0.0,
        "visit_churn": 0.1,
        "visit_churn_missing": 0.0,
        "root_q": value,
        "phase": 1.0,
        "piece_count": 20.0,
        "legal_move_count": 30.0,
        "nodes": 50.0,
    }


def _transition(
    key: str,
    game: int,
    horizon: int,
    gain: float,
    *,
    current: bool,
    value: float = 0.0,
) -> Transition:
    return Transition(
        key=key,
        group_id=f"selfplay:{game}",
        horizon=horizon,
        hard_horizon=256,
        cost=50,
        gain=gain,
        regret_before=max(gain, 0.0) + 1.0,
        regret_after=max(gain, 0.0) + 1.0 - gain,
        complexity_continue=current,
        state=_state(value),
    )


def test_evaluation_preserves_signed_corrections_and_regressions() -> None:
    rows = [
        _transition("a", 1, 100, 1.0, current=True),
        _transition("b", 2, 100, -2.0, current=True),
        _transition("c", 3, 100, 0.5, current=False),
        _transition("d", 4, 100, 0.0, current=False),
    ]
    result = evaluate_horizon(
        rows,
        np.asarray([4.0, 3.0, 2.0, 1.0]),
        np.asarray([4.0, 1.0, 3.0, 2.0]),
    )

    assert result["corrections"] == 2
    assert result["regressions"] == 1
    assert result["policies"]["complexity_predicate"]["signed_gain"] == pytest.approx(-1.0)
    assert result["policies"]["random"]["signed_gain"] == pytest.approx(-0.25)
    assert result["policies"]["oracle"]["signed_gain"] == pytest.approx(1.5)
    assert {policy["spend"] for policy in result["policies"].values()} == {100}


def test_complexity_predicate_selects_boolean_continues_at_natural_spend() -> None:
    continuing = _transition("hard", 1, 100, 2.0, current=True)
    stopping = _transition("easy", 2, 100, -3.0, current=False)
    continuing_state = dict(continuing.state)
    continuing_state["stable_chunks"] = 20.0
    stopping_state = dict(stopping.state)
    stopping_state["stable_chunks"] = 2.0
    rows = [replace(continuing, state=continuing_state), replace(stopping, state=stopping_state)]

    result = evaluate_horizon(rows, np.zeros(2), np.zeros(2))

    assert result["policies"]["complexity_predicate"]["signed_gain"] == pytest.approx(2.0)


def test_reachable_rollout_never_allows_a_stopped_position_to_reenter() -> None:
    rows = [
        _transition(key, game, horizon, 100.0 if key == "d" and horizon > 100 else 0.0,
                    current=True)
        for game, key in enumerate(("a", "b", "c", "d"))
        for horizon in (100, 150, 200)
    ]
    scores = np.asarray([
        (-100.0 if row.key == "d" and row.horizon == 100 else 100.0)
        if row.key == "d" else float(-ord(row.key[0]))
        for row in rows
    ])
    counts = _stage_counts(4, 3, 0.5)
    assert counts == [3, 2, 1]
    selected = _rollout_selected_indices(rows, scores, counts)
    assert not any(rows[int(index)].key == "d" for index in selected)

    result = evaluate_reachable_rollout(
        rows, scores, scores, allocation_fraction=0.5,
    )
    assert result["selection_semantics"] == "nested_prefix_no_reentry"
    assert result["policies"]["M1"]["signed_gain"] == 0.0
    assert result["relaxed_oracle_signed_gain"] == 200.0


def test_reachable_rollout_uses_exact_nested_oracle_not_relaxed_bound() -> None:
    rows = [
        _transition("early", 1, 100, 100.0, current=True),
        _transition("early", 1, 150, 0.0, current=True),
        _transition("late", 2, 100, 0.0, current=True),
        _transition("late", 2, 150, 100.0, current=True),
    ]

    result = evaluate_reachable_rollout(
        rows, np.zeros(4), np.zeros(4), allocation_fraction=0.5,
    )

    assert result["stage_continue_counts"] == [1, 1]
    assert result["oracle_semantics"] == "exact_nested_stop_depth_assignment"
    assert result["reachable_oracle_signed_gain"] == 100.0
    assert result["relaxed_oracle_signed_gain"] == 200.0
    assert result["relaxation_gap"] == 100.0
    assert result["policies"]["oracle"]["signed_gain"] == 100.0


def test_reachable_gate_requires_oracle_headroom_at_every_rung() -> None:
    from scripts.analyze_chunk_controller import _minimum_reachable_rung_gain_delta

    rows = [
        _transition("early", 1, 100, 100.0, current=True),
        _transition("early", 1, 150, -100.0, current=True),
        _transition("late", 2, 100, 0.0, current=True),
        _transition("late", 2, 150, 100.0, current=True),
    ]
    scores = np.zeros(len(rows), dtype=np.float64)

    result = evaluate_reachable_rollout(
        rows, scores, scores, allocation_fraction=0.5,
    )

    assert result["oracle_over_random_headroom_mean"] == 25.0
    assert result["reachable_stage_diagnostics"][0][
        "oracle_over_random_headroom_mean"
    ] == -25.0
    assert _minimum_reachable_rung_gain_delta(
        rows, scores, scores, 0.5, min_oracle_headroom=1.0,
    ) is None


def test_fold_local_rollout_is_invariant_to_cross_fit_score_offsets() -> None:
    rows = [
        _transition(key, game, horizon, gain, current=True)
        for key, game, gain in (
            ("a", 1, 0.0),
            ("b", 2, 0.0),
            ("c", 3, 10.0),
            ("d", 4, -10.0),
        )
        for horizon in (100, 150)
    ]
    fold_ids = np.asarray([
        0 if row.key in {"a", "c"} else 1 for row in rows
    ])
    scores = np.asarray([
        {"a": 10.0, "b": 0.0, "c": 9.0, "d": -1.0}[row.key]
        for row in rows
    ])
    shifted = scores + np.where(fold_ids == 1, 1000.0, 0.0)

    pooled = evaluate_reachable_rollout(
        rows, scores, scores, allocation_fraction=0.5,
    )
    shifted_pooled = evaluate_reachable_rollout(
        rows, shifted, shifted, allocation_fraction=0.5,
    )
    assert pooled["policies"]["M1"]["signed_gain"] != (
        shifted_pooled["policies"]["M1"]["signed_gain"]
    )

    local = evaluate_reachable_rollout(
        rows, scores, scores, allocation_fraction=0.5, fold_ids=fold_ids,
    )
    shifted_local = evaluate_reachable_rollout(
        rows, shifted, shifted, allocation_fraction=0.5, fold_ids=fold_ids,
    )
    assert local["selection_semantics"] == "fold_local_nested_prefix_no_reentry"
    assert local["policies"] == shifted_local["policies"]
    assert local["reachable_stage_diagnostics"] == (
        shifted_local["reachable_stage_diagnostics"]
    )


def test_fold_local_quotas_are_nested_and_preserve_exact_global_spend() -> None:
    global_counts, fold_counts = _fold_stage_counts([5, 3, 2], 4, 0.35)

    assert global_counts == _stage_counts(10, 4, 0.35)
    assert [sum(counts[stage] for counts in fold_counts) for stage in range(4)] == (
        global_counts
    )
    assert all(counts == sorted(counts, reverse=True) for counts in fold_counts)


def test_held_horizon_cv_excludes_horizon_and_source_game() -> None:
    rows = [
        _transition(str(game), game, horizon, game / 10, current=game % 2 == 0)
        for game in range(8)
        for horizon in (100, 150, 200)
    ]
    predictions, diagnostics = held_horizon_predictions(rows, "M0", n_folds=4)

    assert np.isfinite(predictions).all()
    assert diagnostics
    for fold in diagnostics:
        assert fold["horizon"] not in fold["train_horizons"]
        assert set(fold["test_groups"]).isdisjoint(fold["train_groups"])
    fold_ids = _evaluation_fold_ids(rows, 4)
    assert all(
        len({int(fold_ids[index]) for index, row in enumerate(rows) if row.key == key}) == 1
        for key in {row.key for row in rows}
    )


def test_grouped_folds_never_split_a_game() -> None:
    groups = ["s:10", "s:10", "s:11", "s:12", "s:12", "s:12", "s:13", "s:14"]
    folds = grouped_folds(groups, 3)
    memberships: dict[str, int] = {}
    for fold_number, indices in enumerate(folds):
        for index in indices:
            group = groups[int(index)]
            memberships.setdefault(group, fold_number)
            assert memberships[group] == fold_number


def test_source_scoped_groups_do_not_collide() -> None:
    source_a = "\0".join(("/first", "7"))
    source_b = "\0".join(("/second", "7"))
    groups = [source_a, source_a, source_b, source_b]
    folds = grouped_folds(groups, 2)
    membership = {
        groups[int(index)]: fold_number
        for fold_number, fold in enumerate(folds)
        for index in fold
    }
    assert membership[source_a] != membership[source_b]


def test_source_game_group_unifies_one_game_split_across_shards() -> None:
    first = {"source_dir": "/snapshot", "shard": "a.zarr", "game_id": 7}
    second = {"source_dir": "/snapshot", "shard": "b.zarr", "game_id": 7}

    assert _source_group(first) == _source_group(second) == "/snapshot\0" + "7"


def test_trajectory_producer_uses_production_evaluator_stack_and_readback() -> None:
    from scripts import backtest_chunk_trajectory as producer

    source = inspect.getsource(producer._main)
    module_source = inspect.getsource(producer)
    assert "LocalModelEvaluator" not in source
    assert "DirectGPUEvaluator" in source
    assert "ThreadSafeGPUDispatcher" in source
    assert "BatchCoalescingDispatcher" in source
    assert "SyzygyProbe" in source
    assert "mcts_extension" in source
    assert "_make_evaluator_factory" in source
    assert "compile_mode=compile_mode" in source
    assert "realized_search_values" in source
    assert "_require_search_take_effect" in source
    assert "if int(args.walkers) == 1:" in source
    assert "args.walkers != _PRODUCTION_WALKERS" in source
    assert "worker.set_minibatch_size" in source
    assert "allow_terminal_shortcuts=True" in source
    assert '"root_child_q"' in source
    assert '"pv_actions"' in source
    assert '"checkpoint_params"' in source
    assert "load_audit_set(" not in source
    assert source.index("_load_audit_set_snapshot(") < source.index(
        "initial_input_artifacts ="
    )
    assert source.index("initial_input_artifacts =") < source.index("MatchedAuditRows(")
    assert source.index("output_locks = _acquire_output_locks") < source.index(
        "_load_authenticated_model_inputs("
    )
    assert "loader=load_model_from_checkpoint_artifacts" in source
    assert source.index("_load_authenticated_model_inputs(") < source.index(
        "worker = SearchWorker("
    )
    assert "tmp_path.unlink()" not in source
    assert '"raw_observations_preserved": True' in source
    assert module_source.index(
        'if __name__ == "__main__" and "--recover-publication" in sys.argv[1:]:'
    ) < module_source.index(
        "from scripts.native_import_guard import PREIMPORT_NATIVE_ARTIFACTS"
    )
    assert module_source.index(
        "from scripts.native_import_guard import PREIMPORT_NATIVE_ARTIFACTS"
    ) < module_source.index("from chess_anti_engine.eval.audit import")


def _restore_mtime(path: Path, original: os.stat_result) -> None:
    os.utime(
        path,
        ns=(int(original.st_atime_ns), int(original.st_mtime_ns)),
    )


def test_authenticated_checkpoint_load_rejects_fixed_size_alter_load_restore(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "trainer.pt"
    original = b"checkpoint-original"
    altered = b"checkpoint-altered!"
    assert len(original) == len(altered)
    checkpoint.write_bytes(original)
    original_stat = checkpoint.stat()

    def racing_loader(stream: Any, **_kwargs: Any) -> object:
        checkpoint.write_bytes(altered)
        _restore_mtime(checkpoint, original_stat)
        stream.seek(0)
        assert stream.read() == altered
        checkpoint.write_bytes(original)
        _restore_mtime(checkpoint, original_stat)
        return object()

    with pytest.raises(SystemExit, match="identity changed"):
        _load_authenticated_model_inputs(
            checkpoint,
            _params_candidate_inventory(checkpoint),
            loader=racing_loader,
            device="cpu",
            require_complete=True,
        )

    assert checkpoint.read_bytes() == original
    assert checkpoint.stat().st_mtime_ns == original_stat.st_mtime_ns


def test_authenticated_input_rejects_final_component_symlink_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = tmp_path / "trainer.pt"
    requested.write_bytes(b"authenticated")
    substitute = tmp_path / "substitute.pt"
    substitute.write_bytes(b"substituted!!")
    real_open = os.open

    def swap_before_open(path: Any, flags: int, *args: Any) -> int:
        requested.unlink()
        requested.symlink_to(substitute)
        return real_open(path, flags, *args)

    monkeypatch.setattr(os, "open", swap_before_open)
    with pytest.raises(SystemExit, match="model input"):
        _open_authenticated_input(requested)


def test_authenticated_params_load_rejects_fixed_size_alter_load_restore(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "trainer.pt"
    checkpoint.write_bytes(b"checkpoint")
    params = tmp_path / "params.json"
    original = b'{"model":"tiny"}'
    altered = b'{"model":"evil"}'
    assert len(original) == len(altered)
    params.write_bytes(original)
    original_stat = params.stat()

    def racing_loader(
        _stream: Any, *, params_json: bytes | None, **_kwargs: Any,
    ) -> object:
        params.write_bytes(altered)
        _restore_mtime(params, original_stat)
        # The loader receives the already-authenticated immutable bytes, not a
        # pathname it can race independently.
        assert params_json == original
        params.write_bytes(original)
        _restore_mtime(params, original_stat)
        return object()

    with pytest.raises(SystemExit, match="identity changed"):
        _load_authenticated_model_inputs(
            checkpoint,
            _params_candidate_inventory(checkpoint),
            loader=racing_loader,
            device="cpu",
            require_complete=True,
        )

    assert params.read_bytes() == original
    assert params.stat().st_mtime_ns == original_stat.st_mtime_ns


def test_authenticated_model_load_rejects_late_nearer_params_candidate(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "trial" / "nested" / "checkpoint_1"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "trainer.pt"
    checkpoint.write_bytes(b"checkpoint")
    deeper_params = tmp_path / "trial" / "params.json"
    deeper_params.write_text('{"model":"tiny"}')
    initial_inventory = _params_candidate_inventory(checkpoint)
    assert initial_inventory["selected_path"] == str(deeper_params)

    nearer_params = checkpoint_dir / "params.json"
    nearer_params.write_text('{"model":"different"}')
    loader_called = False

    def loader(_stream: Any, **_kwargs: Any) -> object:
        nonlocal loader_called
        loader_called = True
        return object()

    with pytest.raises(SystemExit, match="candidate inventory changed before model load"):
        _load_authenticated_model_inputs(
            checkpoint,
            initial_inventory,
            loader=loader,
            device="cpu",
            require_complete=True,
        )
    assert loader_called is False


def test_outputs_cannot_create_an_initially_absent_params_candidate(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "trial" / "checkpoint_1"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "trainer.pt"
    checkpoint.write_bytes(b"checkpoint")
    candidates = list(_checkpoint_params_candidates(checkpoint))
    nearest = candidates[0]
    assert not nearest.exists()

    with pytest.raises(SystemExit, match="aliases a consumed input artifact"):
        _require_safe_output_paths(
            nearest,
            Path(str(nearest) + ".meta.json"),
            protected_files=candidates,
            protected_directories=[],
        )
    with pytest.raises(SystemExit, match="aliases a consumed input artifact"):
        _require_safe_preregistration_path(
            nearest,
            protected_files=candidates,
            protected_directories=[],
        )
    assert not nearest.exists()


def test_params_inventory_records_negative_and_symlink_candidates(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "trial" / "nested" / "checkpoint_1"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "trainer.pt"
    checkpoint.write_bytes(b"checkpoint")
    (checkpoint_dir.parent / "params.json").mkdir()
    target = tmp_path / "architecture.json"
    target.write_text('{"model":"tiny"}')
    (checkpoint_dir.parent.parent / "params.json").symlink_to(target)

    inventory = _params_candidate_inventory(checkpoint)
    assert [row["state"] for row in inventory["candidates"][:3]] == [
        "absent", "nonregular", "symlink",
    ]
    assert inventory["selected_index"] == 2
    assert inventory["candidates"][2]["resolves_to_regular_file"] is True
    with pytest.raises(SystemExit, match="regular non-symlink"):
        _load_authenticated_model_inputs(
            checkpoint,
            inventory,
            loader=lambda *_args, **_kwargs: object(),
            device="cpu",
            require_complete=True,
        )


def test_params_inventory_detects_create_remove_restore_mtime(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "trial" / "checkpoint_1"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "trainer.pt"
    checkpoint.write_bytes(b"checkpoint")
    before = _params_candidate_inventory(checkpoint)
    parent_stat = checkpoint_dir.stat()
    candidate = checkpoint_dir / "params.json"

    candidate.write_text('{"model":"temporary"}')
    candidate.unlink()
    _restore_mtime(checkpoint_dir, parent_stat)

    after = _params_candidate_inventory(checkpoint)
    assert before != after
    assert (
        before["candidates"][0]["parent_path_components"][-1]["ctime_ns"]
        != after["candidates"][0]["parent_path_components"][-1]["ctime_ns"]
    )

def test_trajectory_preimport_guard_covers_every_loaded_project_extension() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    program = "\n".join((
        "import json, sys",
        "from pathlib import Path",
        "import scripts.backtest_chunk_trajectory",
        "from scripts.native_import_guard import EARLY_NATIVE_MODULES",
        "loaded = sorted(name for name, module in sys.modules.items() "
        "if name.startswith('chess_anti_engine.') "
        "and isinstance(getattr(module, '__file__', None), str) "
        "and Path(module.__file__).suffix in ('.so', '.pyd'))",
        "print(json.dumps({'guarded': sorted(EARLY_NATIVE_MODULES), 'loaded': loaded}))",
    ))

    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    inventory = json.loads(completed.stdout)

    assert inventory["loaded"] == inventory["guarded"]
    assert "chess_anti_engine.encoding._features_ext" in inventory["loaded"]


def _compile_marker_extension(
    output: Path, *, init_name: str, marker: Path,
) -> None:
    source = output.parent / f"{init_name}_shadow.c"
    source.write_text(
        "#include <Python.h>\n"
        "#include <stdio.h>\n"
        "static struct PyModuleDef module = {\n"
        "  PyModuleDef_HEAD_INIT, \"shadow\", NULL, -1, NULL\n"
        "};\n"
        f"PyMODINIT_FUNC PyInit_{init_name}(void) {{\n"
        f"  FILE *f = fopen({json.dumps(str(marker))}, \"w\");\n"
        "  if (f != NULL) { fputs(\"executed\", f); fclose(f); }\n"
        "  return PyModule_Create(&module);\n"
        "}\n"
    )
    include = sysconfig.get_paths()["include"]
    subprocess.run(
        [
            "gcc", "-shared", "-fPIC", f"-I{include}",
            str(source), "-o", str(output),
        ],
        check=True,
    )


def test_source_guard_rejects_extension_shadow_of_tracked_python(
    tmp_path: Path,
) -> None:
    checkout = _clone_clean_guard_checkout(tmp_path, native=False)
    marker = tmp_path / "shadow-executed"
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    shadow = checkout / "scripts" / f"reachable_oracle{suffix}"
    _compile_marker_extension(
        shadow, init_name="reachable_oracle", marker=marker,
    )

    # Demonstrate that ordinary import precedence selects and executes the
    # untracked extension instead of the tracked Python source.
    subprocess.run(
        [sys.executable, "-c", "import scripts.reachable_oracle"],
        cwd=checkout,
        env={**os.environ, "PYTHONPATH": str(checkout)},
        check=True,
    )
    assert marker.read_text() == "executed"
    marker.unlink()

    completed = subprocess.run(
        [
            sys.executable, "scripts/analyze_chunk_controller.py", "--help",
        ],
        cwd=checkout,
        env={**os.environ, "PYTHONPATH": str(checkout)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "project native extension is not explicitly authorized" in completed.stderr
    assert not marker.exists()


def test_source_guard_rejects_valid_extension_copied_to_new_project_fullname(
    tmp_path: Path,
) -> None:
    checkout = _clone_clean_guard_checkout(tmp_path, native=True)
    source_module = importlib.import_module(
        "chess_anti_engine.encoding._features_ext"
    )
    source = Path(str(source_module.__file__)).resolve()
    copied = checkout / "chess_anti_engine" / "eval" / source.name
    shutil.copy2(source, copied)
    fullname = "chess_anti_engine.eval._features_ext"

    ordinary = subprocess.run(
        [sys.executable, "-c", f"import {fullname}"],
        cwd=checkout,
        env={**os.environ, "PYTHONPATH": str(checkout)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert ordinary.returncode == 0, ordinary.stderr

    probe = "\n".join((
        "import runpy",
        "runpy.run_path('scripts/analyze_chunk_controller.py', run_name='guard_probe')",
        f"import {fullname}",
    ))
    guarded = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=checkout,
        env={**os.environ, "PYTHONPATH": str(checkout)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert guarded.returncode != 0
    assert "project native extension is not explicitly authorized" in guarded.stderr


def test_producer_preimport_rejects_symlinked_canonical_native_output(
    tmp_path: Path,
) -> None:
    checkout = _clone_clean_guard_checkout(tmp_path, native=False)
    source_module = importlib.import_module(
        "chess_anti_engine.encoding._features_ext"
    )
    source = Path(str(source_module.__file__)).resolve()
    symlink = checkout / "chess_anti_engine" / "encoding" / source.name
    symlink.symlink_to(source)

    completed = subprocess.run(
        [sys.executable, "scripts/backtest_chunk_trajectory.py", "--help"],
        cwd=checkout,
        env={**os.environ, "PYTHONPATH": str(checkout)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "decision-grade native extension is missing" in completed.stderr


@pytest.mark.parametrize("interference", ["finder", "fileless_module"])
def test_analyzer_provenance_fails_closed_on_import_guard_interference(
    tmp_path: Path, interference: str,
) -> None:
    checkout = _clone_clean_guard_checkout(tmp_path, native=False)
    setup = (
        "sys.meta_path.insert(0, object())"
        if interference == "finder"
        else (
            "sys.modules['scripts.fileless_probe'] = "
            "types.ModuleType('scripts.fileless_probe')"
        )
    )
    probe = "\n".join((
        "import json, runpy, sys, types",
        "ns = runpy.run_path('scripts/analyze_chunk_controller.py', run_name='guard_probe')",
        setup,
        "sources = ns['_analyzer_source_artifacts']()",
        "sha, dirty = ns['_git_state']()",
        "proof = ns['_analyzer_provenance'](sources, sha, dirty)",
        "status = proof['python_preimport']['source_only_import']",
        "print(json.dumps({'decision_grade': proof['decision_grade'], "
        "'active': status['active'], 'first': status['first_finder'], "
        "'loaded_passed': status['loaded_project_modules']['passed'], "
        "'source_recorded': 'scripts.fileless_probe' in sources}))",
    ))
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=checkout,
        env={**os.environ, "PYTHONPATH": str(checkout)},
        text=True,
        capture_output=True,
        check=True,
    )
    observed = json.loads(completed.stdout)
    assert observed["decision_grade"] is False
    if interference == "finder":
        assert observed["active"] is False
        assert observed["first"] is False
        assert observed["loaded_passed"] is True
    else:
        assert observed["active"] is True
        assert observed["first"] is True
        assert observed["loaded_passed"] is False
        assert observed["source_recorded"] is True


def _restored_source_execution_status(
    module: Any, tmp_path: Path,
) -> dict[str, Any]:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "dependency.py"
    original = b"VALUE = 1\n"
    altered = b"VALUE = 2\n"
    target.write_bytes(original)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "dependency.py"], check=True)
    subprocess.run(
        [
            "git", "-C", str(repo), "-c", "user.name=DeepFin Test",
            "-c", "user.email=test@deepfin.invalid", "commit", "-qm", "fixture",
        ],
        check=True,
    )
    git_sha = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
    ).strip()
    blob_oid = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD:dependency.py"], text=True,
    ).strip()
    initial = module._preimport_python_file_artifact(
        target, expected_oid=blob_oid, object_format="sha1",
    )
    snapshot = {
        "git_sha": git_sha,
        "git_object_format": "sha1",
        "repo_root": str(repo),
        "tracked_python_surface_sha256": "a" * 64,
        "files": {"dependency.py": initial},
    }
    original_stat = target.stat()
    namespace: dict[str, Any] = {}
    try:
        target.write_bytes(altered)
        exec(compile(target.read_bytes(), str(target), "exec"), namespace)
    finally:
        target.write_bytes(original)
        os.utime(
            target,
            ns=(int(original_stat.st_atime_ns), int(original_stat.st_mtime_ns)),
        )
    assert namespace["VALUE"] == 2
    assert subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True,
    ) == ""
    return module._preimport_python_surface_status(snapshot)


def test_producer_preimport_snapshot_detects_alter_execute_restore(
    tmp_path: Path,
) -> None:
    from scripts import backtest_chunk_trajectory as producer

    status = _restored_source_execution_status(producer, tmp_path)

    assert status["passed"] is False
    assert status["changed"] == ["dependency.py"]


def test_analyzer_preimport_snapshot_detects_alter_execute_restore(
    tmp_path: Path,
) -> None:
    status = _restored_source_execution_status(controller_module, tmp_path)

    assert status["passed"] is False
    assert status["changed"] == ["dependency.py"]


def test_python_preimport_snapshots_precede_project_and_third_party_imports() -> None:
    from scripts import backtest_chunk_trajectory as producer

    producer_source = inspect.getsource(producer)
    analyzer_source = inspect.getsource(controller_module)
    marker = "_PREIMPORT_PYTHON_SOURCES = _preimport_python_source_snapshot()"
    guard_marker = "_SOURCE_ONLY_IMPORT_GUARD = _install_authenticated_source_only_import("
    assert producer_source.index(marker) < producer_source.index(
        "from scripts import chunk_trajectory_publication"
    )
    assert producer_source.index(marker) < producer_source.index("import chess\n")
    assert producer_source.index(guard_marker) < producer_source.index(
        "from scripts import chunk_trajectory_publication"
    )
    assert analyzer_source.index(marker) < analyzer_source.index("import chess\n")
    assert analyzer_source.index(guard_marker) < analyzer_source.index("import chess\n")


def _clone_clean_guard_checkout(tmp_path: Path, *, native: bool) -> Path:
    source_root = Path(__file__).resolve().parents[1]
    checkout = tmp_path / "guard-checkout"
    subprocess.run(
        ["git", "clone", "-q", "--shared", str(source_root), str(checkout)],
        check=True,
    )
    if native:
        for module_name in controller_module._NATIVE_MODULES:
            module = importlib.import_module(module_name)
            source = Path(str(module.__file__)).resolve()
            relative = Path(*module_name.split(".")[:-1]) / source.name
            destination = checkout / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    assert subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain"], text=True,
    ) == ""
    return checkout


def _install_same_size_timestamp_pyc(
    checkout: Path, relative_path: str, marker: Path,
) -> None:
    source = checkout / relative_path
    good = source.read_bytes()
    original_stat = source.stat()
    payload = (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('bad bytecode executed')\n"
    ).encode()
    assert len(payload) < len(good)
    bad = payload + b"#" * (len(good) - len(payload))
    source.write_bytes(bad)
    os.utime(
        source,
        ns=(int(original_stat.st_atime_ns), int(original_stat.st_mtime_ns)),
    )
    py_compile.compile(str(source), doraise=True)
    source.write_bytes(good)
    os.utime(
        source,
        ns=(int(original_stat.st_atime_ns), int(original_stat.st_mtime_ns)),
    )
    assert source.stat().st_size == original_stat.st_size
    assert source.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain"], text=True,
    ) == ""


@pytest.mark.parametrize(
    ("entrypoint", "dependency", "dependency_module", "analyzer"),
    [
        (
            "scripts/analyze_chunk_controller.py",
            "scripts/reachable_oracle.py",
            "scripts.reachable_oracle",
            True,
        ),
        (
            "scripts/backtest_chunk_trajectory.py",
            "scripts/chunk_trajectory_publication.py",
            "scripts.chunk_trajectory_publication",
            False,
        ),
    ],
)
def test_entrypoint_ignores_valid_timestamp_pyc_and_keeps_valid_provenance(
    tmp_path: Path,
    entrypoint: str,
    dependency: str,
    dependency_module: str,
    analyzer: bool,
) -> None:
    checkout = _clone_clean_guard_checkout(tmp_path, native=True)
    marker = tmp_path / "bad-pyc-executed"
    _install_same_size_timestamp_pyc(checkout, dependency, marker)

    # Establish that the cache is valid and would execute under the ordinary loader.
    subprocess.run(
        [sys.executable, "-c", f"import {dependency_module}"],
        cwd=checkout,
        check=True,
    )
    assert marker.read_text() == "bad bytecode executed"
    marker.unlink()

    if analyzer:
        probe = "\n".join((
            "import json, runpy",
            f"ns = runpy.run_path({entrypoint!r}, run_name='guard_probe')",
            "sources = ns['_analyzer_source_artifacts']()",
            "sha, dirty = ns['_git_state']()",
            "status = ns['_preimport_python_surface_status'](ns['_PREIMPORT_PYTHON_SOURCES'])",
            "proof = ns['_analyzer_provenance'](sources, sha, dirty, preimport_start_status=status)",
            f"module = {dependency_module!r}",
            "loader = proof['python_preimport']['source_only_import']",
            "print(json.dumps({'decision_grade': proof['decision_grade'], "
            "'preimport': proof['python_preimport']['passed'], "
            "'surface': status['passed'], "
            "'source_only': sources[module]['source_only_import_verified'], "
            "'execution': loader['verified_modules'][module]['execution'], "
            "'bytecode_reads': loader['bytecode_cache_reads'], "
            "'failures': loader['failures']}))",
        ))
    else:
        probe = "\n".join((
            "import json, runpy",
            f"ns = runpy.run_path({entrypoint!r}, run_name='guard_probe')",
            "status = ns['_preimport_python_surface_status'](ns['_PREIMPORT_PYTHON_SOURCES'])",
            f"module = {dependency_module!r}",
            "loader = ns['_SOURCE_ONLY_IMPORT_GUARD'].status()",
            "print(json.dumps({'preimport': ns['_PREIMPORT_PYTHON_SOURCES']['passed'], "
            "'surface': status['passed'], "
            "'source_only': ns['_SOURCE_ONLY_IMPORT_GUARD'].module_verified("
            "module, 'scripts/chunk_trajectory_publication.py'), "
            "'execution': loader['verified_modules'][module]['execution'], "
            "'bytecode_reads': loader['bytecode_cache_reads'], "
            "'failures': loader['failures']}))",
        ))
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=checkout,
        env={**os.environ, "PYTHONPATH": str(checkout)},
        text=True,
        capture_output=True,
        check=True,
    )
    observed = json.loads(completed.stdout)
    assert not marker.exists()
    assert observed == {
        **({"decision_grade": True} if analyzer else {}),
        "preimport": True,
        "surface": True,
        "source_only": True,
        "execution": "compiled_authenticated_source_bytes",
        "bytecode_reads": False,
        "failures": [],
    }


def test_recovery_source_guard_needs_no_native_extensions(tmp_path: Path) -> None:
    checkout = _clone_clean_guard_checkout(tmp_path, native=False)
    output = tmp_path / "recovered.jsonl"
    meta = Path(str(output) + ".meta.json")
    pending_output = output.with_name(f".{output.name}.tmp-pending")
    pending_meta = meta.with_name(f".{meta.name}.tmp-pending")
    pending_output.write_text("completed bank\n")
    file_stat = pending_output.stat()
    manifest = {
        "schema": "deepfin.chunk_trajectory.v6",
        "complete": True,
        "output": {
            "path": str(output.resolve()),
            "size": file_stat.st_size,
            "mtime_ns": file_stat.st_mtime_ns,
            "ctime_ns": file_stat.st_ctime_ns,
            "device": file_stat.st_dev,
            "inode": file_stat.st_ino,
            "sha256": hashlib.sha256(pending_output.read_bytes()).hexdigest(),
        },
    }
    pending_meta.write_text(json.dumps(manifest))

    completed = subprocess.run(
        [
            sys.executable, "scripts/backtest_chunk_trajectory.py",
            "--recover-publication", "--out", str(output),
        ],
        cwd=checkout,
        env={**os.environ, "PYTHONPATH": str(checkout)},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "recovered evidence pair" in completed.stdout
    assert output.read_text() == "completed bank\n"
    assert json.loads(meta.read_text()) == manifest


def test_search_take_effect_rejects_an_inert_active_parameter() -> None:
    realized: dict[str, float | int | bool | str] = {
        "concurrency_mode": "walker_puct",
        "concurrency_workers": 2,
        "c_puct": 1.75,
    }
    _require_search_take_effect(
        expected_mode="walker_puct",
        expected_workers=2,
        active_parameters={"c_puct": 1.75},
        realized=realized,
    )
    with pytest.raises(RuntimeError, match=r"c_puct requested=99\.0 realized=1\.75"):
        _require_search_take_effect(
            expected_mode="walker_puct",
            expected_workers=2,
            active_parameters={"c_puct": 99.0},
            realized=realized,
        )


def test_stability_streak_resets_while_gumbel_survivor_trails_visits() -> None:
    last, stable = _update_stability(
        -1, 0, emitted_action=11, visit_gap=0.2, action_count=3,
    )
    assert (last, stable) == (11, 0)
    last, stable = _update_stability(
        last, stable, emitted_action=11, visit_gap=-0.1, action_count=3,
    )
    assert (last, stable) == (11, 0)
    last, stable = _update_stability(
        last, stable, emitted_action=11, visit_gap=0.3, action_count=3,
    )
    assert (last, stable) == (11, 1)


def test_stable_single_legal_move_is_decided_without_root_visits() -> None:
    assert not _complexity_continue(
        stable_chunks=2, visit_gap=0.0, action_count=1,
    )
    assert _complexity_continue(
        stable_chunks=1, visit_gap=0.0, action_count=1,
    )


def test_expected_score_is_stable_for_forced_mate_scale_cp() -> None:
    assert _score(-100_000.0) >= 0.0
    assert _score(100_000.0) <= 1.0
    assert _score(-100_000.0) < _score(0.0) < _score(100_000.0)


def test_budget_interactions_improve_held_horizon_prediction() -> None:
    rows: list[Transition] = []
    for game in range(20):
        value = game / 19.0
        for horizon in (100, 150, 200):
            remaining_fraction = 50.0 / horizon
            gain = (value - 0.5) * (remaining_fraction - 0.3)
            row = _transition(
                str(game), game, horizon, gain,
                current=game % 2 == 0, value=value,
            )
            state = dict(row.state)
            state["nodes"] = float(horizon - 50)
            rows.append(replace(row, state=state))

    m0, _ = held_horizon_predictions(rows, "M0", n_folds=5)
    m1, _ = held_horizon_predictions(rows, "M1", n_folds=5)
    truth = np.asarray([row.gain for row in rows])

    assert np.mean((m1 - truth) ** 2) < np.mean((m0 - truth) ** 2) * 0.5


def test_cluster_bootstrap_is_deterministic() -> None:
    rows = [
        _transition(str(game), game, horizon, (game - 2) / horizon,
                    current=game % 2 == 0)
        for game in range(6)
        for horizon in (100, 150)
    ]
    first = cluster_bootstrap_delta(
        rows,
        allocation_fraction=0.5, samples=20, seed=7, n_folds=3,
    )
    second = cluster_bootstrap_delta(
        rows,
        allocation_fraction=0.5, samples=20, seed=7, n_folds=3,
    )
    assert first == second
    assert first["selection_semantics"] == "fold_local_nested_prefix_no_reentry"
    point_folds = _evaluation_fold_ids(rows, 3)
    point = evaluate_reachable_rollout(
        rows, np.zeros(len(rows)), np.zeros(len(rows)),
        allocation_fraction=0.5, fold_ids=point_folds,
    )
    assert first["selection_semantics"] == point["selection_semantics"]
    assert first["oracle_semantics"] == point["oracle_semantics"]
    assert first["resampling_semantics"] == (
        "global_source_game_clusters_with_recomputed_evaluation_folds"
    )


def test_global_cluster_resample_can_omit_the_old_singleton_fold_game() -> None:
    rows = [
        _transition(str(game), game, horizon, 0.0, current=True)
        for game in range(9)
        for horizon in (100, 150, 200)
    ]
    point_folds = _evaluation_fold_ids(rows, 5)
    groups_by_fold = {
        fold: {
            row.group_id
            for row, row_fold in zip(rows, point_folds, strict=True)
            if int(row_fold) == fold
        }
        for fold in sorted(set(point_folds.tolist()))
    }
    singleton = next(iter(next(
        groups for groups in groups_by_fold.values() if len(groups) == 1
    )))

    sampled = controller_module._resample_source_game_clusters(
        rows, np.random.default_rng(3),
    )

    assert singleton not in {row.group_id for row in sampled}


def test_duplicate_bootstrap_cluster_cannot_cross_recomputed_folds() -> None:
    rows = [
        _transition(str(game), game, horizon, 0.0, current=True)
        for game in range(4)
        for horizon in (100, 150, 200)
    ]
    sampled = controller_module._resample_source_game_clusters(
        rows, np.random.default_rng(0),
    )
    fold_ids = _evaluation_fold_ids(sampled, 3)
    first_horizon = min(row.horizon for row in sampled)
    occurrences = {
        group: sum(
            row.group_id == group and row.horizon == first_horizon
            for row in sampled
        )
        for group in {row.group_id for row in sampled}
    }
    duplicate = next(group for group, count in occurrences.items() if count > 1)

    assert len({
        int(fold)
        for row, fold in zip(sampled, fold_ids, strict=True)
        if row.group_id == duplicate
    }) == 1
    assert len({
        row.key for row in sampled
        if row.group_id == duplicate and row.horizon == first_horizon
    }) == occurrences[duplicate]


def test_bootstrap_valid_fraction_counts_invalid_recomputed_fold_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _transition(str(game), game, horizon, 0.0, current=True)
        for game in range(6)
        for horizon in (100, 150)
    ]
    actual_fold_ids = controller_module._evaluation_fold_ids
    calls = 0

    def fail_first_fold_assignment(
        sampled: list[Transition], n_folds: int,
    ) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("degenerate global cluster draw")
        return actual_fold_ids(sampled, n_folds)

    monkeypatch.setattr(
        controller_module, "_evaluation_fold_ids", fail_first_fold_assignment,
    )
    monkeypatch.setattr(
        controller_module,
        "_refit_fold_predictions",
        lambda sampled, _fold_ids, *, model, n_folds: np.zeros(len(sampled)),
    )
    monkeypatch.setattr(
        controller_module,
        "_minimum_reachable_rung_gain_delta",
        lambda *_args, **_kwargs: 1.0,
    )

    result = cluster_bootstrap_delta(
        rows,
        allocation_fraction=0.5,
        samples=2,
        seed=7,
        n_folds=3,
    )

    assert result["requested_samples"] == 2
    assert result["valid_samples"] == 1
    assert result["invalid_samples"] == 1
    assert result["ineligible_samples"] == 0
    assert result["lower_tail_failure_samples"] == 1
    assert result["lower_tail_failure_fraction"] == 0.5
    assert result["valid_fraction"] == 0.5
    assert result["lower_95"] is None


def test_bootstrap_five_percent_ineligible_mass_has_no_finite_lower_95(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _transition(str(game), game, horizon, 1.0, current=True)
        for game in range(4)
        for horizon in (100, 150)
    ]
    fold_ids = _evaluation_fold_ids(rows, 2)
    calls = 0

    def five_percent_ineligible(*_args: object, **_kwargs: object) -> float | None:
        nonlocal calls
        calls += 1
        return None if calls <= 100 else 1.0

    monkeypatch.setattr(
        controller_module, "_resample_source_game_clusters", lambda *_args: rows,
    )
    monkeypatch.setattr(
        controller_module, "_evaluation_fold_ids", lambda *_args: fold_ids,
    )
    monkeypatch.setattr(
        controller_module,
        "_refit_fold_predictions",
        lambda *_args, **_kwargs: np.zeros(len(rows)),
    )
    monkeypatch.setattr(
        controller_module,
        "_minimum_reachable_rung_gain_delta",
        five_percent_ineligible,
    )

    bootstrap = cluster_bootstrap_delta(
        rows,
        allocation_fraction=0.5,
        samples=2000,
        seed=0,
        n_folds=2,
    )

    assert bootstrap["valid_samples"] == 1900
    assert bootstrap["invalid_samples"] == 0
    assert bootstrap["ineligible_samples"] == 100
    assert bootstrap["valid_fraction"] == 0.95
    assert bootstrap["lower_tail_failure_fraction"] == 0.05
    assert bootstrap["lower_95"] is None
    assert bootstrap["upper_95"] == 1.0

    m0 = np.zeros(len(rows), dtype=np.float64)
    m1 = np.asarray([row.gain for row in rows], dtype=np.float64)
    monkeypatch.setattr(
        controller_module,
        "held_horizon_predictions",
        lambda _rows, model, *, n_folds=5: (m0 if model == "M0" else m1, []),
    )
    monkeypatch.setattr(
        controller_module, "cluster_bootstrap_delta", lambda *_args, **_kwargs: bootstrap,
    )

    result = analyze(
        rows,
        n_folds=2,
        bootstrap_samples=2000,
        seed=0,
        allocation_fraction=0.5,
        min_capture_gain=0.0,
        min_oracle_headroom=0.0,
        min_bootstrap_valid_fraction=0.95,
    )

    assert result["bootstrap_resolution_passed"] is True
    assert result["statistical_gate_passed"] is False


def test_bootstrap_zero_invalid_draws_keep_the_ordinary_percentile_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _transition(str(game), game, horizon, 1.0, current=True)
        for game in range(4)
        for horizon in (100, 150)
    ]
    fold_ids = _evaluation_fold_ids(rows, 2)
    values = iter(float(index) for index in range(40))
    monkeypatch.setattr(
        controller_module, "_resample_source_game_clusters", lambda *_args: rows,
    )
    monkeypatch.setattr(
        controller_module, "_evaluation_fold_ids", lambda *_args: fold_ids,
    )
    monkeypatch.setattr(
        controller_module,
        "_refit_fold_predictions",
        lambda *_args, **_kwargs: np.zeros(len(rows)),
    )
    monkeypatch.setattr(
        controller_module,
        "_minimum_reachable_rung_gain_delta",
        lambda *_args, **_kwargs: next(values),
    )

    result = cluster_bootstrap_delta(
        rows,
        allocation_fraction=0.5,
        samples=40,
        seed=0,
        n_folds=2,
    )

    expected = np.arange(40, dtype=np.float64)
    assert result["invalid_samples"] == 0
    assert result["ineligible_samples"] == 0
    assert result["lower_tail_failure_fraction"] == 0.0
    assert result["valid_fraction"] == 1.0
    assert result["lower_95"] == pytest.approx(float(np.quantile(expected, 0.025)))
    assert result["upper_95"] == pytest.approx(float(np.quantile(expected, 0.975)))
    assert result["interval_semantics"] == (
        "unconditional_requested_replicates_with_invalid_mass_in_lower_tail_v1"
    )


def test_bootstrap_maps_sub_tail_failure_mass_into_valid_quantiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _transition(str(game), game, horizon, 1.0, current=True)
        for game in range(4)
        for horizon in (100, 150)
    ]
    fold_ids = _evaluation_fold_ids(rows, 2)
    values = iter([None, *(float(index) for index in range(99))])
    monkeypatch.setattr(
        controller_module, "_resample_source_game_clusters", lambda *_args: rows,
    )
    monkeypatch.setattr(
        controller_module, "_evaluation_fold_ids", lambda *_args: fold_ids,
    )
    monkeypatch.setattr(
        controller_module,
        "_refit_fold_predictions",
        lambda *_args, **_kwargs: np.zeros(len(rows)),
    )
    monkeypatch.setattr(
        controller_module,
        "_minimum_reachable_rung_gain_delta",
        lambda *_args, **_kwargs: next(values),
    )

    result = cluster_bootstrap_delta(
        rows,
        allocation_fraction=0.5,
        samples=100,
        seed=0,
        n_folds=2,
    )

    expected_quantile = (0.025 - 0.01) / 0.99
    assert result["lower_tail_failure_fraction"] == 0.01
    assert result["lower_95"] == pytest.approx(float(np.quantile(
        np.arange(99, dtype=np.float64), expected_quantile,
    )))


def test_nine_game_five_fold_global_draws_retain_canonical_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _transition(str(game), game, horizon, 0.0, current=True)
        for game in range(9)
        for horizon in (100, 150, 200)
    ]
    monkeypatch.setattr(
        controller_module,
        "_refit_fold_predictions",
        lambda sampled, _fold_ids, *, model, n_folds: np.zeros(len(sampled)),
    )
    monkeypatch.setattr(
        controller_module,
        "_minimum_reachable_rung_gain_delta",
        lambda *_args, **_kwargs: 1.0,
    )

    result = cluster_bootstrap_delta(
        rows,
        allocation_fraction=0.2,
        samples=2000,
        seed=0,
        n_folds=5,
    )

    assert result["requested_samples"] == 2000
    assert result["valid_samples"] == 2000
    assert result["valid_fraction"] == 1.0


def test_bootstrap_refits_exclude_the_target_fold_and_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _transition(str(game), game, horizon, game / horizon, current=True)
        for game in range(9)
        for horizon in (100, 150, 200)
    ]
    fold_ids = _evaluation_fold_ids(rows, 3)
    group_fold = {
        row.group_id: int(fold_ids[index]) for index, row in enumerate(rows)
    }
    seen = 0

    def checked_alpha(
        train_rows: list[Transition], model: str, n_folds: int,
    ) -> float:
        nonlocal seen
        del model, n_folds
        seen += 1
        assert len({row.horizon for row in train_rows}) == 2
        assert len({group_fold[row.group_id] for row in train_rows}) == 2
        return 1.0

    monkeypatch.setattr(controller_module, "_inner_alpha", checked_alpha)
    predictions = controller_module._refit_fold_predictions(
        rows, fold_ids, model="M0", n_folds=3,
    )

    assert np.isfinite(predictions).all()
    assert seen == 9


def _synthetic_tablebase_directory(
    path: str, *, wdl_count: int, dtz_count: int, inode_offset: int,
) -> dict[str, Any]:
    root_identity = {
        "device": 1,
        "inode": inode_offset,
        "mtime_ns": 1,
        "ctime_ns": 1,
    }
    current_path = Path(os.sep)
    path_components: list[dict[str, int | str]] = [{
        "path": os.sep,
        "device": 1,
        "inode": 1,
        "mtime_ns": 1,
        "ctime_ns": 1,
    }]
    for index, component in enumerate(Path(path).parts[1:], start=1):
        current_path /= component
        identity = (
            root_identity
            if current_path == Path(path)
            else {
                "device": 1,
                "inode": index + 1,
                "mtime_ns": 1,
                "ctime_ns": 1,
            }
        )
        path_components.append({"path": str(current_path), **identity})
    identities = [
        [f"w{index:03d}.rtbw", 1, 1, 1, 1, inode_offset + index + 1]
        for index in range(wdl_count)
    ] + [
        [
            f"z{index:03d}.rtbz", 1, 1, 1, 1,
            inode_offset + wdl_count + index + 1,
        ]
        for index in range(dtz_count)
    ]
    identity_fields = [
        "name", "size", "mtime_ns", "ctime_ns", "device", "inode",
    ]
    portable_digest = filename_size_sha256(
        (str(row[0]), int(row[1])) for row in identities
    )
    identity_document = json.dumps(
        {
            "root_identity": root_identity,
            "path_component_identity_fields": [
                "path", "device", "inode", "mtime_ns", "ctime_ns",
            ],
            "path_components": path_components,
            "file_identity_fields": identity_fields,
            "file_identities": identities,
        },
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "path": path,
        "root_identity": root_identity,
        "path_component_identity_fields": [
            "path", "device", "inode", "mtime_ns", "ctime_ns",
        ],
        "path_components": path_components,
        "rtbw_count": wdl_count,
        "rtbz_count": dtz_count,
        "file_identity_count": len(identities),
        "file_identity_fields": identity_fields,
        "file_identities": identities,
        "total_bytes": len(identities),
        "approved_layout": {
            "schema": "deepfin.approved_syzygy_layout.v1",
            "component": Path(path).name,
            "canonical_encoding": (
                "sorted_compact_ascii_json_array_of_filename_and_decimal_size"
            ),
            "rtbw_count": wdl_count,
            "rtbz_count": dtz_count,
            "file_count": len(identities),
            "total_bytes": len(identities),
            "filename_size_sha256": portable_digest,
            "passed": True,
        },
        "inventory_sha256": hashlib.sha256(identity_document).hexdigest(),
    }


def _synthetic_approved_syzygy_contract() -> tuple[
    tuple[ApprovedSyzygyComponent, ...], tuple[tuple[str, str], ...],
]:
    directories = [
        _synthetic_tablebase_directory(
            "/tb/syzygy_3-4-5", wdl_count=510, dtz_count=145,
            inode_offset=10,
        ),
        _synthetic_tablebase_directory(
            "/tb/syzygy_6", wdl_count=365, dtz_count=365,
            inode_offset=10_000,
        ),
    ]
    components = tuple(
        ApprovedSyzygyComponent(
            directory_name=str(row["approved_layout"]["component"]),
            rtbw_count=int(row["rtbw_count"]),
            rtbz_count=int(row["rtbz_count"]),
            file_count=int(row["file_identity_count"]),
            total_bytes=int(row["total_bytes"]),
            filename_size_sha256=str(
                row["approved_layout"]["filename_size_sha256"]
            ),
        )
        for row in directories
    )
    names = sorted({
        str(identity[0])
        for directory in directories
        for identity in directory["file_identities"]
    })
    catalog_rows = tuple(
        (
            name,
            hashlib.md5(name.encode("ascii"), usedforsecurity=False).hexdigest(),
        )
        for name in names
    )
    return components, catalog_rows


def _synthetic_syzygy_inventory() -> dict[str, Any]:
    directories = [
        _synthetic_tablebase_directory(
            "/tb/syzygy_3-4-5", wdl_count=510, dtz_count=145,
            inode_offset=10,
        ),
        _synthetic_tablebase_directory(
            "/tb/syzygy_6", wdl_count=365, dtz_count=365,
            inode_offset=10_000,
        ),
    ]
    _components, catalog_rows = _synthetic_approved_syzygy_contract()
    catalog = {
        "schema": "deepfin.syzygy_checksum_catalog.v1",
        "component": "syzygy_6",
        "name": "3-4-5-6.md5",
        "size": controller_module.APPROVED_SYZYGY_CHECKSUM_CATALOG_SIZE,
        "mtime_ns": 1,
        "ctime_ns": 1,
        "device": 1,
        "inode": 20_000,
        "raw_sha256": (
            controller_module.APPROVED_SYZYGY_CHECKSUM_CATALOG_RAW_SHA256
        ),
        "algorithm": "md5",
        "entry_count": len(catalog_rows),
        "rtbw_count": sum(
            name.endswith(".rtbw") for name, _digest in catalog_rows
        ),
        "rtbz_count": sum(
            name.endswith(".rtbz") for name, _digest in catalog_rows
        ),
        "canonical_entries_sha256": checksum_catalog_entries_sha256(catalog_rows),
        "entries": [list(row) for row in catalog_rows],
        "approved": True,
    }
    expected_md5 = dict(catalog_rows)
    verification_rows = [
        [
            str(directory["approved_layout"]["component"]),
            str(identity[0]),
            int(identity[1]),
            int(identity[2]),
            int(identity[3]),
            int(identity[4]),
            int(identity[5]),
            expected_md5[str(identity[0])],
        ]
        for directory in directories
        for identity in directory["file_identities"]
    ]
    verification_document = json.dumps(
        verification_rows, ensure_ascii=True, separators=(",", ":"),
    ).encode("utf-8")
    content_verification = {
        "schema": "deepfin.syzygy_content_verification.v1",
        "method": "single_pass_md5_against_approved_catalog",
        "identity_binding_fields": [
            "component", "name", "size", "mtime_ns", "ctime_ns", "device",
            "inode", "approved_md5",
        ],
        "file_count": sum(
            int(directory["file_identity_count"]) for directory in directories
        ),
        "bytes_hashed": sum(int(directory["total_bytes"]) for directory in directories),
        "file_identity_checksum_sha256": hashlib.sha256(
            verification_document
        ).hexdigest(),
        "passed": True,
    }
    result = {
        "schema": "deepfin.syzygy_inventory.v4",
        "identity_method": (
            "approved_filename_size_plus_no_follow_path_components_and_"
            "file_device_inode_size_mtime_ctime"
        ),
        "path_anchor_semantics": (
            "absolute_root_and_each_lexical_directory_component_no_follow"
        ),
        "path": "/tb/syzygy_3-4-5:/tb/syzygy_6",
        "rtbw_count": 875,
        "rtbz_count": 510,
        "directories": directories,
        "approved_layout_schema": "deepfin.approved_syzygy_layout.v1",
        "approved_component_order": ["syzygy_3-4-5", "syzygy_6"],
        "approved_layout_passed": True,
        "checksum_catalog": catalog,
        "checksum_catalog_covers_logical_table_names": True,
        "content_verification": content_verification,
    }
    inventory_document = json.dumps(
        result, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
    ).encode("utf-8")
    result["inventory_sha256"] = hashlib.sha256(inventory_document).hexdigest()
    return result


def _refresh_synthetic_syzygy_integrity(syzygy: dict[str, Any]) -> None:
    for directory in syzygy["directories"]:
        identities = directory["file_identities"]
        wdl_count = sum(str(row[0]).endswith(".rtbw") for row in identities)
        dtz_count = sum(str(row[0]).endswith(".rtbz") for row in identities)
        total_bytes = sum(int(row[1]) for row in identities)
        directory["rtbw_count"] = wdl_count
        directory["rtbz_count"] = dtz_count
        directory["file_identity_count"] = len(identities)
        directory["total_bytes"] = total_bytes
        layout_digest = filename_size_sha256(
            (str(row[0]), int(row[1])) for row in identities
        )
        directory["approved_layout"].update({
            "rtbw_count": wdl_count,
            "rtbz_count": dtz_count,
            "file_count": len(identities),
            "total_bytes": total_bytes,
            "filename_size_sha256": layout_digest,
            "passed": True,
        })
        identity_document = json.dumps(
            {
                "root_identity": directory["root_identity"],
                "path_component_identity_fields": directory[
                    "path_component_identity_fields"
                ],
                "path_components": directory["path_components"],
                "file_identity_fields": directory["file_identity_fields"],
                "file_identities": identities,
            },
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        directory["inventory_sha256"] = hashlib.sha256(
            identity_document
        ).hexdigest()
    expected_md5 = {
        str(row[0]): str(row[1]) for row in syzygy["checksum_catalog"]["entries"]
    }
    verification_rows = [
        [
            str(directory["approved_layout"]["component"]),
            str(identity[0]),
            int(identity[1]),
            int(identity[2]),
            int(identity[3]),
            int(identity[4]),
            int(identity[5]),
            expected_md5[str(identity[0])],
        ]
        for directory in syzygy["directories"]
        for identity in directory["file_identities"]
    ]
    verification_document = json.dumps(
        verification_rows, ensure_ascii=True, separators=(",", ":"),
    ).encode("utf-8")
    syzygy["content_verification"].update({
        "file_count": sum(
            int(row["file_identity_count"]) for row in syzygy["directories"]
        ),
        "bytes_hashed": sum(
            int(row["total_bytes"]) for row in syzygy["directories"]
        ),
        "file_identity_checksum_sha256": hashlib.sha256(
            verification_document
        ).hexdigest(),
    })
    payload = dict(syzygy)
    payload.pop("inventory_sha256", None)
    syzygy["inventory_sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _write_bank(path: Path, *, correct_gap: bool) -> Path:
    board = chess.Board()
    _ucis, legal_actions = legal_full_indices(board)
    actions = [int(action) for action in legal_actions]
    emitted = int(move_to_index(chess.Move.from_uci("a2a3"), board))
    alternative = int(move_to_index(chess.Move.from_uci("b2b3"), board))
    reference_best = int(move_to_index(chess.Move.from_uci("e2e4"), board))
    emitted_index = actions.index(emitted)
    alternative_index = actions.index(alternative)
    visits = [0] * len(actions)
    visits[emitted_index] = 20
    visits[alternative_index] = 30
    shares = [float(visit) / 50.0 for visit in visits]
    child_q = [0.2] * len(actions)
    child_q[emitted_index] = 0.1
    child_q_observed = [visit > 0 for visit in visits]
    action_regret = [20.0] * len(actions)
    action_regret[emitted_index] = 10.0
    action_regret[actions.index(reference_best)] = 0.0
    action_reference = [80.0] * len(actions)
    action_reference[emitted_index] = 90.0
    action_reference[actions.index(reference_best)] = 100.0
    deep_reference_move_cp = {
        "e2e4": 100.0,
        "a2a3": 90.0,
        "b2b3": 80.0,
        "b2b4": 80.0,
        "c2c3": 80.0,
        "c2c4": 80.0,
        "d2d3": 80.0,
        "d2d4": 80.0,
        "g1f3": 80.0,
        "b1c3": 80.0,
    }
    action_listed = [False] * len(actions)
    for uci in deep_reference_move_cp:
        action = int(move_to_index(chess.Move.from_uci(uci), board))
        action_listed[actions.index(action)] = True
    entropy = float(-(0.4 * np.log(0.4) + 0.6 * np.log(0.6)))
    best_cp = 100.0
    regret_cp = 10.0
    regret_score = (
        1.0 / (1.0 + 10.0 ** (-best_cp / 300.0))
        - 1.0 / (1.0 + 10.0 ** (-(best_cp - regret_cp) / 300.0))
    )
    rows = [
        {
            "schema": "deepfin.chunk_trajectory.v6",
            "key": position_key(board), "source_dir": "/snapshot", "shard": "s0.zarr",
            "fen": board.fen(),
            "game_id": 3, "group_id": "/snapshot\0" + "3",
            "chunk": chunk, "nodes": chunk * 50,
            "elapsed_ms": float(chunk), "regret_cp": regret_cp,
            "regret_score": regret_score, "regret_vs_final_cp": 0.0,
            "deep_reference_nodes": 1_000_000,
            "deep_reference_depth": 30,
            "deep_reference_scored_multipv": len(deep_reference_move_cp),
            "deep_reference_best_cp": best_cp,
            "deep_reference_move_cp": deep_reference_move_cp,
            "visit_gap": -0.2 if correct_gap else 0.2,
            "root_actions": actions,
            "root_visits": [visit * chunk for visit in visits],
            "root_visit_shares": shares,
            "root_child_q": child_q,
            "root_child_q_observed": child_q_observed,
            "root_action_regret_cp": action_regret,
            "root_action_reference_cp": action_reference,
            "root_action_reference_listed": action_listed,
            "emitted_reference_listed": True,
            "emitted_action": emitted, "uci": "a2a3", "bestmove_flip": False,
            "pv_actions": [emitted], "pv_uci": ["a2a3"],
            "stable_chunks": 0, "visit_entropy": entropy, "q_gap": -0.1,
            "complexity_predicate_continue": True,
            "q_drift": None if chunk == 1 else 0.0,
            "visit_churn": None if chunk == 1 else 0.0, "root_q": 0.0,
            "changes_to_final": False,
            "phase": phase_bucket(32), "source": 0,
            "piece_count": 32, "legal_move_count": 20,
            "tb_probes": 0, "tb_hits": 0,
        }
        for chunk in (1, 2, 3, 4)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = Path(str(path) + ".meta.json")
    producer_source_paths = {
        "producer_script": "scripts/backtest_chunk_trajectory.py",
        "scripts.chunk_trajectory_publication": (
            "scripts/chunk_trajectory_publication.py"
        ),
        "scripts.analyze_chunk_controller": "scripts/analyze_chunk_controller.py",
        "scripts.repo_output_guard": "scripts/repo_output_guard.py",
        "scripts.match_audit_rows": "scripts/match_audit_rows.py",
        "scripts.approved_syzygy": "scripts/approved_syzygy.py",
        "scripts.source_only_import": "scripts/source_only_import.py",
        "chess_anti_engine.eval.audit": "chess_anti_engine/eval/audit.py",
        "chess_anti_engine.mcts.search_options": (
            "chess_anti_engine/mcts/search_options.py"
        ),
        "chess_anti_engine.uci.search": "chess_anti_engine/uci/search.py",
        "chess_anti_engine.uci.model_loader": (
            "chess_anti_engine/uci/model_loader.py"
        ),
    }
    producer_sources: dict[str, dict[str, Any]] = {}
    for name, relative_path in producer_source_paths.items():
        source = f"synthetic source for {name}\n".encode()
        _TEST_GIT_FILES[relative_path] = source
        producer_sources[name] = {
            "path": "/producer-checkout/" + relative_path,
            "repo_relative_path": relative_path,
            "matches_producer_git_revision": True,
            "matches_preimport_snapshot": True,
            "source_only_import_verified": True,
            "source_execution": (
                "entrypoint_trust_boundary"
                if name == "producer_script" else (
                    "compiled_authenticated_bootstrap_source_bytes"
                    if name == "scripts.source_only_import"
                    else "compiled_authenticated_source_bytes"
                )
            ),
            "size": len(source),
            "mtime_ns": 1,
            "sha256": hashlib.sha256(source).hexdigest(),
        }
    native_builds: dict[str, dict[str, Any]] = {}
    for module in controller_module._NATIVE_MODULES:
        dependency_bytes = {
            relative_path: f"synthetic native input {relative_path}\n".encode()
            for relative_path in extension_spec(module).dependencies
        }
        _TEST_GIT_FILES.update(dependency_bytes)
        native_builds[module] = {
            **controller_module.native_build_attestation(
                module, "a" * 40, dependency_bytes,
            ),
            "current_inputs_match_revision": True,
            "matches_producer_revision": True,
        }
    audit_set_sha256 = controller_module._APPROVED_AUDIT_SET_SHA256
    deep_reference_evidence = controller_module._deep_reference_evidence_summary(
        [rows[0]], audit_set_sha256=audit_set_sha256,
    )
    python_preimport_files: dict[str, dict[str, Any]] = {}
    for index, (relative_path, source) in enumerate(sorted(
        (path, content) for path, content in _TEST_GIT_FILES.items()
        if path.endswith(".py")
    ), start=1):
        blob_oid = hashlib.sha1(
            f"blob {len(source)}\0".encode("ascii") + source
        ).hexdigest()
        python_preimport_files[relative_path] = {
            "path": "/producer-checkout/" + relative_path,
            "size": len(source),
            "mtime_ns": 1,
            "ctime_ns": 1,
            "device": 1,
            "inode": index,
            "sha256": hashlib.sha256(source).hexdigest(),
            "git_blob_oid": blob_oid,
            "observed_git_blob_oid": blob_oid,
            "stable_read": True,
            "matches_git_revision": True,
            "git_mode": "100644",
            "git_kind": "blob",
        }
    surface_rows = [
        [relative_path, artifact["git_blob_oid"], artifact["sha256"]]
        for relative_path, artifact in sorted(python_preimport_files.items())
    ]
    surface_digest = hashlib.sha256(json.dumps(
        surface_rows, separators=(",", ":"), ensure_ascii=True,
    ).encode()).hexdigest()
    python_check = {
        "passed": True,
        "changed": [],
        "git_sha": "a" * 40,
        "tracked_python_file_count": len(python_preimport_files),
        "tracked_python_surface_sha256": surface_digest,
    }
    native_import_artifacts = {
        "chess_anti_engine.encoding._features_ext": {
            "path": "/_features_ext.so", "lexical_path": "/_features_ext.so", "size": 1,
            "mtime_ns": 1, "ctime_ns": 1, "device": 1, "inode": 101,
            "sha256": "4" * 64, "stable_read": True,
        },
        "chess_anti_engine.encoding._lc0_ext": {
            "path": "/_lc0_ext.so", "lexical_path": "/_lc0_ext.so", "size": 1,
            "mtime_ns": 1, "ctime_ns": 1, "device": 1, "inode": 102,
            "sha256": "2" * 64, "stable_read": True,
        },
        "chess_anti_engine.mcts._mcts_tree": {
            "path": "/_mcts_tree.so", "lexical_path": "/_mcts_tree.so", "size": 1,
            "mtime_ns": 1, "ctime_ns": 1, "device": 1, "inode": 103,
            "sha256": "e" * 64, "stable_read": True,
        },
    }
    verified_source_modules = {
        name: {
            "repo_relative_path": relative_path,
            "sha256": python_preimport_files[relative_path]["sha256"],
            "execution": (
                "compiled_authenticated_bootstrap_source_bytes"
                if name == "scripts.source_only_import"
                else "compiled_authenticated_source_bytes"
            ),
            "bytecode_cache_read": False,
        }
        for name, relative_path in producer_source_paths.items()
        if name != "producer_script"
    }
    python_preimport = {
        "schema": "deepfin.python_preimport.v1",
        "git_sha": "a" * 40,
        "final_git_sha": "a" * 40,
        "git_object_format": "sha1",
        "repo_root": "/producer-checkout",
        "entrypoint": "/producer-checkout/scripts/backtest_chunk_trajectory.py",
        "snapshot_stage": "before_project_or_third_party_imports",
        "trust_boundary": "already_executing_entry_script_and_python_process",
        "preexisting_project_modules": [],
        "tracked_python_file_count": len(python_preimport_files),
        "tracked_python_surface_sha256": surface_digest,
        "source_tree_matches_revision": True,
        "files": python_preimport_files,
        "passed": True,
        "start_check": dict(python_check),
        "post_import_check": dict(python_check),
        "post_run_check": dict(python_check),
        "source_only_import": {
            "schema": "deepfin.source_only_import.v2",
            "active": True,
            "installed": True,
            "first_finder": True,
            "git_sha": "a" * 40,
            "tracked_python_surface_sha256": surface_digest,
            "project_scope": ["chess_anti_engine", "scripts"],
            "execution": "compile_authenticated_source_bytes",
            "bytecode_cache_reads": False,
            "native_extension_loading": (
                "default_deny_exact_preimport_artifact_authenticated_loader"
            ),
            "permitted_native_modules": list(controller_module._NATIVE_MODULES),
            "authorized_native_modules": list(controller_module._NATIVE_MODULES),
            "authorized_native_artifacts": native_import_artifacts,
            "verified_native_modules": {
                name: {
                    **artifact,
                    "execution": "authenticated_canonical_extension_loader",
                    "preimport_artifact_authenticated": True,
                }
                for name, artifact in native_import_artifacts.items()
            },
            "verified_modules": verified_source_modules,
            "loaded_project_modules": {
                "passed": True,
                "loaded_modules": sorted(
                    set(verified_source_modules) | set(controller_module._NATIVE_MODULES)
                ),
                "unverified_modules": [],
            },
            "failures": [],
        },
    }
    params_candidate_payload = {
        "schema": "deepfin.params_candidate_inventory.v1",
        "search_limit": 6,
        "selection_policy": "first_is_file_in_checkpoint_ancestor_order",
        "trainer_pt": "/trainer.pt",
        "candidates": [{
            "index": 0,
            "path": "/params.json",
            "state": "absent",
            "resolves_to_regular_file": False,
            "identity": None,
            "parent_path_components": [{
                "path": "/", "kind": "directory", "mode": 16877,
                "size": 1, "mtime_ns": 1, "ctime_ns": 1,
                "device": 1, "inode": 1,
            }],
        }],
        "selected_index": None,
        "selected_path": None,
    }
    params_candidate_inventory = {
        **params_candidate_payload,
        "inventory_sha256": hashlib.sha256(json.dumps(
            params_candidate_payload,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode()).hexdigest(),
    }
    manifest: dict[str, Any] = {
        "schema": "deepfin.chunk_trajectory.v6",
        "complete": True,
        "decision_grade": True,
        "analysis_scope": "fixed_node_horizons_only",
        "clock_conditioning_available": False,
        "reference_censoring": {
            "kind": "finite_multipv_unlisted_emitted_move",
            "scope": "completed_trajectory_decision_labels",
            "decision_labels_require_listed_emitted_moves": True,
            "passed": True,
            "affected_trajectory_count": 0,
            "unlisted_emitted_row_count": 0,
            "censored_transition_count": 0,
            "affected_trajectories": [],
        },
        "elapsed_measurement": {
            "kind": "callback_instrumented_wall_time",
            "usable_for_controller_or_cost_analysis": False,
        },
        "root_position_history": "fen_only_from_audit_fen",
        "root_tree_state": "fresh_per_position_no_cross_move_reuse",
        "game_group_kind": "source_dir:game_id",
        "deep_reference_evidence": deep_reference_evidence,
        "panel_selection": {
            "strategy": "joint_audit_source_phase_piece_round_robin_v1",
            "selection_mode": "full_set",
            "stratum_fields": ["source", "phase", "piece_bucket"],
            "piece_bucket_definition": "clamp_2_32_then_floor_divide_by_4",
            "within_stratum_order": "sha256_position_key_then_position_key",
            "source_order": [0, 1],
            "requested_max_positions": 2,
            "available_position_count": 1,
            "selected_position_count": 1,
            "available_position_keys_sha256": controller_module._panel_key_digest(
                [position_key(chess.Board())]
            ),
            "selected_position_keys_sha256": controller_module._panel_key_digest(
                [position_key(chess.Board())]
            ),
            "available_keys_unique": True,
            "source_domain_passed": True,
            "phase_morphology_passed": True,
            "available_source_counts": [
                {"source": 0, "count": 1}, {"source": 1, "count": 0},
            ],
            "selected_source_counts": [
                {"source": 0, "count": 1}, {"source": 1, "count": 0},
            ],
            "available_stratum_counts": [
                {"source": 0, "phase": 2, "piece_bucket": 8, "count": 1},
            ],
            "selected_stratum_counts": [
                {"source": 0, "phase": 2, "piece_bucket": 8, "count": 1},
            ],
            "source_balance": {
                "maximum_difference": 1, "observed_difference": 1, "passed": True,
            },
            "decision_grade_passed": True,
        },
        "complexity_predicate": {
            "kind": "clock_free_visit_gap_and_stability",
            "minimum_stable_chunks": 2,
            "minimum_visit_gap": 0.25,
            "single_legal_move_is_decided": True,
        },
        "producer_git_sha": "a" * 40,
        "producer_git_dirty": False,
        "python_preimport": python_preimport,
        "producer_script": producer_sources["producer_script"],
        "publication_helper": producer_sources[
            "scripts.chunk_trajectory_publication"
        ],
        "producer_sources": producer_sources,
        "checkpoint": {
            "path": "/trainer.pt", "lexical_path": "/trainer.pt",
            "size": 1, "mtime_ns": 1, "ctime_ns": 1,
            "device": 1, "inode": 201, "sha256": "b" * 64,
            "stable_read": True,
            "consumption": "torch_load_from_same_open_file_description",
        },
        "checkpoint_params": None,
        "params_candidate_inventory": params_candidate_inventory,
        "model_input_consumption": {
            "schema": "deepfin.model_input_consumption.v2",
            "checkpoint_open": "absolute_lexical_path_o_nofollow",
            "checkpoint": "torch_load_from_same_open_file_description",
            "checkpoint_path_reopened_by_loader": False,
            "checkpoint_identity_verified_before_search": True,
            "checkpoint_sha256_streamed_from_same_open_file_description": True,
            "params": "no_params_json",
            "params_open": "no_params_json",
            "params_path_reopened_by_loader": False,
            "params_identity_verified_before_search": True,
            "params_selection": "first_is_file_in_checkpoint_ancestor_order",
            "params_candidate_inventory_schema": (
                "deepfin.params_candidate_inventory.v1"
            ),
            "params_candidate_inventory_sha256": params_candidate_inventory[
                "inventory_sha256"
            ],
            "params_candidate_inventory_verified_before_load": True,
            "params_candidate_inventory_verified_after_load": True,
            "params_selected_index": None,
            "params_selected_path": None,
            "passed": True,
        },
        "audit_set": {
            "path": "/audit.jsonl", "size": 1, "mtime_ns": 1,
            "sha256": audit_set_sha256,
        },
        "matched_rows": {
            "path": "/matched.npz", "size": 1, "mtime_ns": 1, "sha256": "d" * 64,
        },
        "matched_rows_report": {
            "path": "/matched.npz.report.json", "size": 1,
            "mtime_ns": 1, "sha256": "8" * 64,
        },
        "matched_row_origin_verification": {
            "schema": "deepfin.matched_audit_rows.v1",
            "passed": True,
            "report": {
                "path": "/matched.npz.report.json", "size": 1,
                "mtime_ns": 1, "sha256": "8" * 64,
            },
            "snapshot_inventory": {
                "path": "/snapshot",
                "root_identity": {
                    "device": 1, "inode": 1, "mtime_ns": 1, "ctime_ns": 1,
                },
                "shard_count": 1,
                "shards": [{
                    "name": "s0.zarr", "device": 1, "inode": 1,
                    "mtime_ns": 1, "ctime_ns": 1, "file_count": 1,
                    "total_bytes": 1, "entries_identity_sha256": "5" * 64,
                }],
                "inventory_sha256": "7" * 64,
            },
            "selected_position_count": 1,
            "selected_position_keys_sha256": hashlib.sha256(json.dumps(
                [position_key(board)], ensure_ascii=True, separators=(",", ":"),
            ).encode()).hexdigest(),
            "rows": [{
                "key": position_key(board),
                "source_dir": "/snapshot",
                "selected_origin": {
                    "shard": "s0.zarr", "row": 0,
                    "position_key": position_key(board),
                    "stored_x_sha256": "6" * 64,
                    "has_game_id": True, "game_id": 3,
                },
                "duplicate_count": 1,
                "source_cluster_ambiguous": False,
                "source_cluster_unique": True,
                "occurrences": [{
                    "shard": "s0.zarr", "row": 0,
                    "position_key": position_key(board),
                    "stored_x_sha256": "6" * 64,
                    "has_game_id": True, "game_id": 3,
                }],
            }],
        },
        "features_extension": {
            **native_import_artifacts["chess_anti_engine.encoding._features_ext"],
            "build_attestation": native_builds[
                "chess_anti_engine.encoding._features_ext"
            ],
            "freshness_check": {
                "modules": list(controller_module._NATIVE_MODULES),
                "minimum_gcc_major": 15,
                "production_recipe_required": True,
                "passed": True, "issues": [],
            },
        },
        "mcts_extension": {
            **native_import_artifacts["chess_anti_engine.mcts._mcts_tree"],
            "abi_version": 9, "required_abi_version": 9,
            "gss_halving_rev": 3,
            "build_attestation": native_builds[
                "chess_anti_engine.mcts._mcts_tree"
            ],
            "freshness_check": {
                "modules": list(controller_module._NATIVE_MODULES),
                "minimum_gcc_major": 15,
                "production_recipe_required": True,
                "passed": True, "issues": [],
            },
        },
        "lc0_extension": {
            **native_import_artifacts["chess_anti_engine.encoding._lc0_ext"],
            "cboard_encode_full": True,
            "build_attestation": native_builds[
                "chess_anti_engine.encoding._lc0_ext"
            ],
            "freshness_check": {
                "modules": list(controller_module._NATIVE_MODULES),
                "minimum_gcc_major": 15,
                "production_recipe_required": True,
                "passed": True, "issues": [],
            },
        },
        "artifact_stability": {
            "passed": True, "changed": [], "final_git_sha": "a" * 40,
            "final_git_dirty": False,
        },
        "syzygy": _synthetic_syzygy_inventory(),
        "row_count": 4,
        "chunk_count": 4,
        "position_count": 1,
        "requested_position_count": 1,
        "requested_max_positions": 2,
        "excluded_position_count": 0,
        "excluded_positions": [],
        "incomplete_exclusion_count": 0,
        "source_game_group_count": 9,
        "minimum_decision_grade_source_games": 9,
        "source_group_resolution_passed": True,
        "requested_search": {
            "device": "cuda", "active_path": "walker_puct",
            "walkers": 2, "chunk_sims": 50, "max_chunks": 4,
            "active_parameters": {
                "c_puct": 1.75, "cpuct_factor": 3.89, "cpuct_base": 38739.0,
                "fpu_reduction": 0.33, "vloss_weight": 3,
                "walker_gather": 1,
            },
        },
        "requested_model_search_contract": {
            "model_input_history_encoding": "legacy",
            "model_input_extra_features": "v1",
            "model_policy_encoding": "lc0_1858",
            "model_compute_relations": False,
            "search_input_history_encoding": "legacy",
            "search_input_extra_features": "v1",
            "search_policy_encoding": "lc0_1858",
            "search_compute_relations": False,
            "evaluator_input_planes": 146,
            "walker_input_planes": 146,
            "walker_compute_relations": False,
        },
        "realized_model_search_contract": {
            "model_input_history_encoding": "legacy",
            "model_input_extra_features": "v1",
            "model_policy_encoding": "lc0_1858",
            "model_compute_relations": False,
            "search_input_history_encoding": "legacy",
            "search_input_extra_features": "v1",
            "search_policy_encoding": "lc0_1858",
            "search_compute_relations": False,
            "evaluator_input_planes": 146,
            "walker_input_planes": 146,
            "walker_compute_relations": False,
        },
        "requested_evaluator": {
            "stack": (
                "BatchCoalescingDispatcher("
                "ThreadSafeGPUDispatcher(DirectGPUEvaluator))"
            ),
            "direct_max_batch": 256, "outer_max_batch": 256, "n_slots": 2,
            "input_bf16": False, "legal_bf16": False,
            "compiled": True, "model_wrapper_type": "OptimizedModule",
        },
        "realized_evaluator": {
            "stack": (
                "BatchCoalescingDispatcher("
                "ThreadSafeGPUDispatcher(DirectGPUEvaluator))"
            ),
            "direct_max_batch": 256, "outer_max_batch": 256, "n_slots": 2,
            "input_bf16": False, "legal_bf16": False,
            "compiled": True, "model_wrapper_type": "OptimizedModule",
        },
        "compile": {
            "enabled": True, "mode": "max-autotune", "cache_dir": "/cache",
            "torchinductor_cache_dir": "/cache/torch", "triton_cache_dir": "/cache/triton",
        },
        "runtime": {
            "python_version": "3.10", "python_implementation": "CPython",
            "python_executable": "/usr/bin/python3", "numpy_version": "2.x",
            "python_chess_version": "1.x", "platform": "Linux-test",
            "machine": "x86_64",
            "torch_version": "2.x", "torch_cuda_version": "13.0",
            "cudnn_version": 90000, "nvidia_driver_version": "600.1",
            "requested_device": "cuda", "evaluator_device": "cuda",
            "model_parameter_devices": ["cuda:0"],
            "resolved_requested_device": "cuda:0",
            "resolved_evaluator_device": "cuda:0",
            "resolved_model_parameter_devices": ["cuda:0"],
            "cuda_device_name": "test GPU", "cuda_device_capability": [12, 0],
        },
        "realized_search": {
            "concurrency_mode": "walker_puct", "concurrency_workers": 2,
            "chunk_sims": 50, "c_puct": 1.75, "cpuct_factor": 3.89,
            "cpuct_base": 38739.0, "fpu_reduction": 0.33,
            "vloss_weight": 3, "walker_gather": 1,
        },
        "search_warmup": {
            "completed": True, "requested_nodes": 256, "realized_nodes": 256,
            "excluded_from_timing": True, "tree_reset_after": True,
            "tablebase_counters_reset_after": True,
        },
        "realized_tablebase": {
            "installed": True, "cursed_as_draw": True,
            "n_wdl": 510, "n_dtz": 510, "max_pieces": 6,
            "root_probe_active": True, "leaf_probe_active": False,
            "positive_control": {
                "fen": "7k/8/8/8/8/8/8/KQ6 w - - 0 1",
                "probes": 1, "hits": 1, "apply_return": 1, "passed": True,
            },
            "root_shortcut_positive_control": {
                "fen": "7k/8/8/8/8/8/8/KQ6 w - - 0 1",
                "bestmove_uci": "b1b7", "nodes": 1, "tbhits": 1,
                "root_declined": None, "tree_created": False, "passed": True,
            },
        },
        "output": {
            "path": str(path.resolve()),
            "sha256": digest,
            "size": path.stat().st_size,
        },
    }
    matched_builder_script = dict(producer_sources["scripts.match_audit_rows"])
    matched_builder_script["matches_git_revision"] = True
    manifest["matched_row_origin_verification"].update({
        "report_builder": {
            "git_sha": "a" * 40,
            "git_dirty": False,
            "script": matched_builder_script,
        },
        "report_audit_set": dict(manifest["audit_set"]),
        "report_output": dict(manifest["matched_rows"]),
        "report_input_stability": {
            "audit_set_unchanged": True,
            "snapshot_unchanged": True,
            "builder_checkout_unchanged": True,
        },
    })
    synthetic_inventory = manifest["matched_row_origin_verification"][
        "snapshot_inventory"
    ]
    synthetic_inventory["inventory_sha256"] = hashlib.sha256(json.dumps({
        "root_identity": synthetic_inventory["root_identity"],
        "shards": synthetic_inventory["shards"],
    }, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
    preregistration = {
        "schema": "deepfin.chunk_controller_preregistration.v3",
        "producer": {
            "source_git_sha": "9" * 40,
            "checkpoint_sha256": manifest["checkpoint"]["sha256"],
            "checkpoint_params_sha256": None,
            "model_input_consumption": manifest["model_input_consumption"],
            "params_candidate_inventory_sha256": manifest[
                "params_candidate_inventory"
            ]["inventory_sha256"],
            "audit_set_sha256": manifest["audit_set"]["sha256"],
            "matched_rows_sha256": manifest["matched_rows"]["sha256"],
            "matched_rows_report_sha256": manifest["matched_rows_report"]["sha256"],
            "matched_rows_snapshot_inventory_sha256": manifest[
                "matched_row_origin_verification"
            ]["snapshot_inventory"]["inventory_sha256"],
            "max_positions": manifest["requested_max_positions"],
            "panel_selection": manifest["panel_selection"],
            "deep_reference_evidence": manifest["deep_reference_evidence"],
            "requested_search": manifest["requested_search"],
            "requested_model_search_contract": manifest[
                "requested_model_search_contract"
            ],
            "requested_evaluator": manifest["requested_evaluator"],
            "compile": {
                "enabled": manifest["compile"]["enabled"],
                "mode": manifest["compile"]["mode"],
            },
            "syzygy": manifest["syzygy"],
        },
        "analysis": {
            "folds": 5,
            "bootstrap_samples": 2000,
            "seed": 0,
            "allocation_fraction": 0.2,
            "min_capture_gain": 0.05,
            "min_oracle_headroom": 1e-4,
            "min_bootstrap_valid_fraction": 0.95,
            "reachable_selection_semantics": "fold_local_nested_prefix_no_reentry",
            "reachable_oracle_semantics": (
                "exact_fold_local_nested_stop_depth_assignment"
            ),
            "bootstrap_resampling_semantics": (
                "global_source_game_clusters_with_recomputed_evaluation_folds"
            ),
            "bootstrap_interval_semantics": (
                "unconditional_requested_replicates_with_invalid_mass_in_lower_tail_v1"
            ),
        },
    }
    document = json.dumps(preregistration, sort_keys=True, separators=(",", ":")) + "\n"
    relative_path = "tests/fixtures/chunk_controller_preregistration.json"
    manifest["preregistration"] = {
        "path": "/repo/" + relative_path,
        "repo_relative_path": relative_path,
        "size": len(document.encode()),
        "mtime_ns": 1,
        "sha256": hashlib.sha256(document.encode()).hexdigest(),
    }
    manifest["preregistration_document"] = document
    _TEST_GIT_FILES[relative_path] = document.encode()
    meta.write_text(json.dumps(manifest))
    os.link(path, controller_module._pending_output_path(path))
    os.link(meta, controller_module._pending_manifest_path(meta))
    return meta


def _rewrite_bank(bank: Path, meta: Path, rows: list[dict[str, object]]) -> None:
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads(meta.read_text())
    manifest["output"] = {
        "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        "size": bank.stat().st_size,
    }
    meta.write_text(json.dumps(manifest))


def _sync_test_panel_selection(
    manifest: dict[str, Any], position_rows: list[dict[str, Any]],
) -> None:
    source_counts = {0: 0, 1: 0}
    stratum_counts: dict[tuple[int, int, int], int] = {}
    for row in position_rows:
        source = int(row["source"])
        phase = int(row["phase"])
        piece_bucket = min(32, max(2, int(row["piece_count"]))) // 4
        source_counts[source] = source_counts.get(source, 0) + 1
        stratum = (source, phase, piece_bucket)
        stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
    keys = [str(row["key"]) for row in position_rows]
    difference = abs(source_counts[0] - source_counts[1])
    selection = {
        "strategy": "joint_audit_source_phase_piece_round_robin_v1",
        "selection_mode": "full_set",
        "stratum_fields": ["source", "phase", "piece_bucket"],
        "piece_bucket_definition": "clamp_2_32_then_floor_divide_by_4",
        "within_stratum_order": "sha256_position_key_then_position_key",
        "source_order": [0, 1],
        "requested_max_positions": manifest["requested_max_positions"],
        "available_position_count": len(position_rows),
        "selected_position_count": len(position_rows),
        "available_position_keys_sha256": controller_module._panel_key_digest(keys),
        "selected_position_keys_sha256": controller_module._panel_key_digest(keys),
        "available_keys_unique": len(keys) == len(set(keys)),
        "source_domain_passed": set(source_counts).issubset({0, 1}),
        "phase_morphology_passed": True,
        "available_source_counts": [
            {"source": source, "count": count}
            for source, count in sorted(source_counts.items())
        ],
        "selected_source_counts": [
            {"source": source, "count": count}
            for source, count in sorted(source_counts.items())
        ],
        "available_stratum_counts": [
            {
                "source": source, "phase": phase,
                "piece_bucket": piece_bucket, "count": count,
            }
            for (source, phase, piece_bucket), count in sorted(stratum_counts.items())
        ],
        "selected_stratum_counts": [
            {
                "source": source, "phase": phase,
                "piece_bucket": piece_bucket, "count": count,
            }
            for (source, phase, piece_bucket), count in sorted(stratum_counts.items())
        ],
        "source_balance": {
            "maximum_difference": 1,
            "observed_difference": difference,
            "passed": difference <= 1,
        },
        "decision_grade_passed": bool(
            position_rows and len(keys) == len(set(keys)) and difference <= 1
        ),
    }
    manifest["panel_selection"] = selection
    manifest["deep_reference_evidence"] = (
        controller_module._deep_reference_evidence_summary(
            position_rows,
            audit_set_sha256=manifest["audit_set"]["sha256"],
        )
    )
    origin_rows = []
    for row in position_rows:
        origin = {
            "shard": str(row["shard"]),
            "row": 0,
            "position_key": str(row["key"]),
            "stored_x_sha256": "6" * 64,
            "has_game_id": True,
            "game_id": int(row["game_id"]),
        }
        origin_rows.append({
            "key": str(row["key"]),
            "source_dir": str(row["source_dir"]),
            "selected_origin": origin,
            "duplicate_count": 1,
            "source_cluster_ambiguous": False,
            "source_cluster_unique": True,
            "occurrences": [dict(origin)],
        })
    verification = manifest["matched_row_origin_verification"]
    verification["selected_position_count"] = len(origin_rows)
    verification["selected_position_keys_sha256"] = hashlib.sha256(json.dumps(
        sorted(str(row["key"]) for row in position_rows),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()).hexdigest()
    verification["rows"] = origin_rows
    preregistration = json.loads(manifest["preregistration_document"])
    preregistration["producer"]["panel_selection"] = selection
    preregistration["producer"]["deep_reference_evidence"] = manifest[
        "deep_reference_evidence"
    ]
    document = json.dumps(
        preregistration, sort_keys=True, separators=(",", ":"),
    ) + "\n"
    manifest["preregistration_document"] = document
    manifest["preregistration"]["size"] = len(document.encode())
    manifest["preregistration"]["sha256"] = hashlib.sha256(document.encode()).hexdigest()
    relative_path = manifest["preregistration"]["repo_relative_path"]
    _TEST_GIT_FILES[relative_path] = document.encode()


def test_deep_reference_evidence_pins_the_approved_frozen_ruler() -> None:
    forced_board = chess.Board(
        "rn3bnr/ppp1k2p/5pp1/Pb1pp3/1P3PPq/B7/N1PPP1BP/R2QK1NR w KQ - 4 14"
    )
    forced_move = next(iter(forced_board.legal_moves)).uci()
    assert forced_board.legal_moves.count() == 1
    position: dict[str, Any] = {
        "key": position_key(forced_board),
        "fen": forced_board.fen(),
        "deep_reference_nodes": 1_522,
        "deep_reference_depth": 245,
        "deep_reference_scored_multipv": 1,
        "deep_reference_best_cp": 100_000.0,
        "deep_reference_move_cp": {forced_move: 100_000.0},
    }

    approved = controller_module._deep_reference_evidence_summary(
        [position],
        audit_set_sha256=controller_module._APPROVED_AUDIT_SET_SHA256,
    )
    substituted = controller_module._deep_reference_evidence_summary(
        [position], audit_set_sha256="0" * 64,
    )

    assert approved["passed"] is True
    assert approved["positions_below_requested_nodes"] == [position["key"]]
    assert substituted["audit_set_identity_passed"] is False
    assert substituted["passed"] is False


@pytest.mark.parametrize("mutation", ["move_key", "move_cp", "best_cp"])
def test_deep_reference_evidence_digest_covers_full_ruler_values(
    mutation: str,
) -> None:
    board = chess.Board()
    position = {
        "key": position_key(board),
        "fen": board.fen(),
        "deep_reference_nodes": 1_000_000,
        "deep_reference_depth": 30,
        "deep_reference_scored_multipv": 10,
        "deep_reference_best_cp": 100.0,
        "deep_reference_move_cp": {
            "e2e4": 100.0,
            "d2d4": 90.0,
            "g1f3": 80.0,
            "c2c4": 70.0,
            "b1c3": 60.0,
            "c2c3": 50.0,
            "g2g3": 40.0,
            "b2b3": 30.0,
            "f2f4": 20.0,
            "a2a3": 10.0,
        },
    }
    original = controller_module._deep_reference_evidence_summary(
        [position],
        audit_set_sha256=controller_module._APPROVED_AUDIT_SET_SHA256,
    )
    changed = json.loads(json.dumps(position))
    if mutation == "move_key":
        changed["deep_reference_move_cp"]["h2h3"] = (
            changed["deep_reference_move_cp"].pop("a2a3")
        )
    elif mutation == "move_cp":
        changed["deep_reference_move_cp"]["a2a3"] = 11.0
    else:
        changed["deep_reference_best_cp"] = 101.0

    mutated = controller_module._deep_reference_evidence_summary(
        [changed],
        audit_set_sha256=controller_module._APPROVED_AUDIT_SET_SHA256,
    )

    assert mutated["position_evidence_sha256"] != original[
        "position_evidence_sha256"
    ]


@pytest.mark.parametrize("field", ["deep_reference_best_cp", "deep_reference_move_cp"])
def test_deep_reference_evidence_rejects_nonfinite_scores(field: str) -> None:
    board = chess.Board()
    position: dict[str, Any] = {
        "key": position_key(board),
        "fen": board.fen(),
        "deep_reference_nodes": 1_000_000,
        "deep_reference_depth": 30,
        "deep_reference_scored_multipv": 1,
        "deep_reference_best_cp": 10.0,
        "deep_reference_move_cp": {"e2e4": 10.0},
    }
    if field == "deep_reference_move_cp":
        position[field]["e2e4"] = float("nan")
    else:
        position[field] = float("inf")

    with pytest.raises(ValueError, match="must be a finite number"):
        controller_module._deep_reference_evidence_summary(
            [position],
            audit_set_sha256=controller_module._APPROVED_AUDIT_SET_SHA256,
        )


def test_audit_snapshot_hashes_and_parses_the_same_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chess_anti_engine.eval.audit import parse_audit_record as real_parse
    from scripts import backtest_chunk_trajectory as producer

    board = chess.Board()
    record = {
        "key": position_key(board),
        "fen": board.fen(),
        "phase": phase_bucket(32),
        "source": 0,
        "multipv": [{"move": "e2e4", "cp": 42}],
        "wdl": [400, 400, 200],
        "nodes": 1_000_000,
        "depth": 30,
    }
    audit_path = tmp_path / "audit.jsonl"
    original_bytes = (json.dumps(record) + "\n").encode()
    audit_path.write_bytes(original_bytes)
    original_digest = hashlib.sha256(original_bytes).hexdigest()
    monkeypatch.setattr(producer, "_APPROVED_AUDIT_SET_SHA256", original_digest)
    substituted = {**record, "multipv": [{"move": "d2d4", "cp": -999}]}
    substituted_bytes = (json.dumps(substituted) + "\n").encode()
    parse_calls = 0

    def replace_restore_while_parsing(line: str) -> AuditPosition:
        nonlocal parse_calls
        parse_calls += 1
        audit_path.write_bytes(substituted_bytes)
        try:
            return real_parse(line)
        finally:
            audit_path.write_bytes(original_bytes)

    monkeypatch.setattr(producer, "parse_audit_record", replace_restore_while_parsing)
    positions, artifact = producer._load_audit_set_snapshot(
        audit_path, require_approved=True,
    )

    assert parse_calls == 1
    assert positions[0].move_cp == {"e2e4": 42.0}
    assert artifact["sha256"] == original_digest
    assert artifact["consumption"] == (
        "sha256_and_positions_from_same_immutable_byte_snapshot"
    )


def test_audit_snapshot_rejects_unapproved_bytes_before_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import backtest_chunk_trajectory as producer

    audit_path = tmp_path / "substituted.jsonl"
    audit_path.write_text("{}\n")
    monkeypatch.setattr(
        producer,
        "parse_audit_record",
        lambda _line: pytest.fail("unapproved bytes reached the audit parser"),
    )

    with pytest.raises(SystemExit, match="approved frozen audit set SHA256"):
        producer._load_audit_set_snapshot(audit_path, require_approved=True)


def test_loader_rejects_a_shallow_one_move_reference(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    for row in rows:
        row.update({
            "deep_reference_nodes": 1,
            "deep_reference_depth": 1,
            "deep_reference_scored_multipv": 1,
            "deep_reference_best_cp": 90.0,
            "deep_reference_move_cp": {"a2a3": 90.0},
        })
    _rewrite_bank(bank, meta, rows)

    with pytest.raises(ValueError, match="frozen ruler requires 10"):
        load_transitions(bank)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("deep_reference_nodes", 0, "observed nodes must be positive"),
        ("deep_reference_depth", 0, "depth must be a positive integer"),
        ("deep_reference_scored_multipv", 9, "count disagrees"),
    ],
)
def test_loader_rejects_tampered_deep_reference_evidence(
    tmp_path: Path, field: str, value: int, match: str,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    rows[0][field] = value
    _rewrite_bank(bank, meta, rows)

    with pytest.raises(ValueError, match=match):
        load_transitions(bank)


def test_loader_recomputes_positive_deep_reference_evidence_tampering(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    rows[0]["deep_reference_nodes"] = 999_999
    _rewrite_bank(bank, meta, rows)

    with pytest.raises(ValueError, match="evidence disagrees with raw selected-position"):
        load_transitions(bank)


def test_manifest_rejects_substituted_audit_set_identity(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["deep_reference_evidence"]["audit_set_sha256"] = "0" * 64
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="deep-reference ruler evidence"):
        load_transitions(bank)


def test_loader_refuses_leader_gap_for_nonleading_emitted_action(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    _write_bank(bank, correct_gap=False)

    with pytest.raises(ValueError, match="emitted action's gap"):
        load_transitions(bank)


def test_loader_requires_provenance_but_allows_explicit_legacy_smoke(tmp_path: Path) -> None:
    bank = tmp_path / "legacy.jsonl"
    bank.write_text("".join([
        json.dumps({
            "key": "k", "chunk": 1, "nodes": 50, "regret_cp": 20,
            "visit_gap": 0.1, "uci": "a2a3",
        }) + "\n",
        json.dumps({
            "key": "k", "chunk": 2, "nodes": 100, "regret_cp": 10,
            "visit_gap": 0.2, "uci": "a2a3",
        }) + "\n",
    ]))

    with pytest.raises(SystemExit, match="unexpected hard links"):
        load_transitions(bank)
    transitions, info = load_transitions(bank, methodology_smoke=True)
    assert transitions[0].gain == 10.0
    assert info["decision_grade"] is False
    assert info["metric"] == "regret_cp"


def test_loader_accepts_verified_emitted_action_gap(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    _write_bank(bank, correct_gap=True)
    transitions, info = load_transitions(bank)
    assert transitions[0].state["visit_gap"] == pytest.approx(-0.2)
    assert info["decision_grade"] is True
    assert info["preregistered_design"] is True
    assert info["analysis_scope"] == "fresh_tree_fixed_node_horizons_only"
    assert info["cross_move_tree_reuse_tested"] is False


def test_loader_rejects_missing_matched_row_report(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    del manifest["matched_rows_report"]
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="matched_rows_report"):
        load_transitions(bank)


def test_loader_rejects_fabricated_unique_game_id_against_origin_proof(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    for row in rows:
        row["game_id"] = 99
        row["group_id"] = "/snapshot\0" + "99"
    _rewrite_bank(bank, meta, rows)

    with pytest.raises(ValueError, match="disagrees with row readback"):
        load_transitions(bank)


def test_loader_rejects_one_field_origin_proof_mutation(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["matched_row_origin_verification"]["rows"][0][
        "selected_origin"
    ]["row"] = 1
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="selected-origin proof is inconsistent"):
        load_transitions(bank)


def test_loader_rejects_panel_source_counts_that_disagree_with_rows(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    for row in rows:
        row["source"] = 1
    _rewrite_bank(bank, meta, rows)

    with pytest.raises(ValueError, match="source/phase counts disagree"):
        load_transitions(bank)


def test_loader_rejects_tampered_panel_selection_provenance(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["panel_selection"]["source_balance"]["observed_difference"] = 0
    meta.write_text(json.dumps(manifest))

    with pytest.raises(
        ValueError,
        match=r"preregistered design.*audit panel selection provenance",
    ) as exc_info:
        load_transitions(bank)
    assert "does not match the preregistered design" in str(exc_info.value)
    assert "audit panel selection provenance" in str(exc_info.value)


def test_loader_rejects_decision_grade_claim_with_insufficient_source_games(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["source_game_group_count"] = 8
    manifest["source_group_resolution_passed"] = False
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="insufficient source-game resolution"):
        load_transitions(bank)

    _transitions, info = load_transitions(bank, methodology_smoke=True)
    assert info["decision_grade"] is False


def test_loader_requires_publication_helper_provenance(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    del manifest["publication_helper"]
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="publication_helper artifact provenance"):
        load_transitions(bank)


def test_loader_rejects_publication_helper_not_bound_to_producer_revision(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    foreign_source = b"foreign publication helper\n"
    manifest["publication_helper"].update({
        "path": "/foreign-worktree/scripts/chunk_trajectory_publication.py",
        "size": len(foreign_source),
        "sha256": hashlib.sha256(foreign_source).hexdigest(),
    })
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="not bound to the producer Git revision"):
        load_transitions(bank)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("passed", False),
        ("preexisting_project_modules", ["chess_anti_engine.eval.audit"]),
        ("snapshot_stage", "after_imports"),
    ],
)
def test_loader_rejects_invalid_producer_preimport_proof(
    tmp_path: Path, field: str, value: Any,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["python_preimport"][field] = value
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="pre-import Python source provenance"):
        load_transitions(bank, meta_path=meta)


def test_loader_rejects_bytecode_enabled_producer_import_proof(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["python_preimport"]["source_only_import"][
        "bytecode_cache_reads"
    ] = True
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="pre-import Python source provenance"):
        load_transitions(bank, meta_path=meta)


def test_loader_rejects_foreign_loaded_producer_python_module(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    foreign_source = b"foreign search implementation\n"
    manifest["producer_sources"]["chess_anti_engine.uci.search"].update({
        "path": "/foreign/chess_anti_engine/uci/search.py",
        "size": len(foreign_source),
        "sha256": hashlib.sha256(foreign_source).hexdigest(),
    })
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="producer Python sources"):
        load_transitions(bank)


@pytest.mark.parametrize("tamper", ["omit", "wrong_path"])
def test_loader_requires_exact_approved_syzygy_source_provenance(
    tmp_path: Path, tamper: str,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    sources = manifest["producer_sources"]
    if tamper == "omit":
        sources.pop("scripts.approved_syzygy")
    else:
        sources["scripts.approved_syzygy"]["repo_relative_path"] = (
            "scripts/other_syzygy.py"
        )
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="producer Python sources"):
        load_transitions(bank)


def test_loader_rejects_a_collection_outside_the_preregistered_design(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["requested_max_positions"] = 3
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="does not match the preregistered design"):
        load_transitions(bank)


def test_loader_authenticates_preregistration_against_the_producer_commit(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    payload = json.loads(manifest["preregistration_document"])
    payload["analysis"]["seed"] = 1
    document = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    manifest["preregistration_document"] = document
    manifest["preregistration"]["size"] = len(document.encode())
    manifest["preregistration"]["sha256"] = hashlib.sha256(document.encode()).hexdigest()
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="not tracked verbatim"):
        load_transitions(bank)


def test_loader_requires_fresh_tree_scope_provenance(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["root_tree_state"] = "production_reused_tree"
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="root_tree_state"):
        load_transitions(bank)


def test_loader_exposes_missing_drift_and_churn_to_M0(tmp_path: Path) -> None:
    from scripts.analyze_chunk_controller import _design

    bank = tmp_path / "bank.jsonl"
    _write_bank(bank, correct_gap=True)

    transitions, _ = load_transitions(bank)

    assert transitions[0].state["q_drift_missing"] == 1.0
    assert transitions[0].state["visit_churn_missing"] == 1.0
    assert transitions[1].state["q_drift_missing"] == 0.0
    assert transitions[1].state["visit_churn_missing"] == 0.0
    assert _design(transitions, "M0")[0].tolist() != _design(transitions, "M0")[1].tolist()


def test_loader_rejects_dirty_or_incomplete_producer_provenance(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["producer_git_dirty"] = True
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="producer checkout was dirty"):
        load_transitions(bank)


def test_loader_rejects_requested_parameter_that_did_not_take_effect(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["realized_search"]["c_puct"] = 2.5
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="realized active parameters"):
        load_transitions(bank)


def test_loader_rejects_unrealized_walker_gather(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["requested_search"]["active_parameters"]["walker_gather"] = 999
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="realized active parameters"):
        load_transitions(bank)


@pytest.mark.parametrize(
    ("field", "value"),
    [("max_chunks", 3), ("chunk_sims", 31), ("walkers", 1)],
)
def test_loader_rejects_invalid_requested_search_shape(
    tmp_path: Path, field: str, value: int,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["requested_search"][field] = value
    if field in {"chunk_sims", "walkers"}:
        realized_field = "chunk_sims" if field == "chunk_sims" else "concurrency_workers"
        manifest["realized_search"][realized_field] = value
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="requested search"):
        load_transitions(bank)


def test_loader_rejects_classic_gumbel_as_decision_grade(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["requested_search"].update({
        "active_path": "gumbel",
        "walkers": 1,
        "active_parameters": {
            "c_scale": 0.1,
            "c_visit": 50.0,
            "c_visit_root": 50.0,
            "c_scale_root": 1.0,
            "q_visit_exp_root": 1.0,
            "topk": 16,
            "policy_temp": 1.0,
            "halving_div": 2,
            "root_noise_scale": 1.0,
            "vloss_weight": 3,
            "minibatch_size": 32,
        },
    })
    manifest["realized_search"] = {
        **manifest["requested_search"]["active_parameters"],
        "concurrency_mode": "gumbel",
        "concurrency_workers": 1,
        "chunk_sims": manifest["requested_search"]["chunk_sims"],
    }
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="requested search"):
        load_transitions(bank)


def test_loader_requires_realized_syzygy_and_native_binary(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["realized_tablebase"]["installed"] = False
    manifest["mcts_extension"]["sha256"] = "not-a-hash"
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=r"MCTS.*Syzygy|Syzygy.*MCTS"):
        load_transitions(bank)


def test_loader_requires_feature_encoder_binary_provenance(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["features_extension"]["sha256"] = "not-a-hash"
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="feature encoding extension"):
        load_transitions(bank)


@pytest.mark.parametrize(
    ("section", "module", "message"),
    [
        (
            "features_extension",
            "chess_anti_engine.encoding._features_ext",
            "feature encoding extension",
        ),
        (
            "lc0_extension",
            "chess_anti_engine.encoding._lc0_ext",
            "CBoard encoding extension",
        ),
        (
            "mcts_extension",
            "chess_anti_engine.mcts._mcts_tree",
            "MCTS extension",
        ),
    ],
)
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_git_sha", "b" * 40),
        ("input_sha256", "c" * 64),
        ("matches_producer_revision", False),
    ],
)
def test_loader_rejects_mismatched_embedded_native_build_attestation(
    tmp_path: Path,
    section: str,
    module: str,
    message: str,
    field: str,
    value: object,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    assert manifest[section]["build_attestation"]["module"] == module
    manifest[section]["build_attestation"][field] = value
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        load_transitions(bank)


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ("features_extension", "feature encoding extension"),
        ("lc0_extension", "CBoard encoding extension"),
        ("mcts_extension", "MCTS extension"),
    ],
)
def test_loader_rejects_native_build_dependency_not_at_producer_revision(
    tmp_path: Path, section: str, message: str,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    dependency = next(iter(manifest[section]["build_attestation"]["dependencies"]))
    _TEST_GIT_FILES[dependency] += b" changed after foreign build"

    with pytest.raises(ValueError, match=message):
        load_transitions(bank)


def test_historical_native_attestation_uses_its_versioned_dependency_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import check_c_extensions_fresh as freshness

    module = "chess_anti_engine.encoding._features_ext"
    schema = "deepfin.native_build.v1"
    dependency_bytes = {
        relative_path: f"historical:{relative_path}\n".encode()
        for relative_path in freshness.native_build_dependency_paths(schema, module)
    }
    _TEST_GIT_FILES.update(dependency_bytes)
    artifact = {
        "build_attestation": {
            **freshness.native_build_attestation(
                module, "a" * 40, dependency_bytes, schema=schema,
            ),
            "current_inputs_match_revision": True,
            "matches_producer_revision": True,
        },
    }
    grown = freshness.ExtensionSpec(
        module,
        (*extension_spec(module).dependencies, "chess_anti_engine/future_header.h"),
    )
    monkeypatch.setattr(
        freshness,
        "EXTENSION_SPECS",
        tuple(
            grown if spec.module == module else spec
            for spec in freshness.EXTENSION_SPECS
        ),
    )

    assert controller_module._native_build_matches_revision(
        artifact, "a" * 40, module,
    ) is True
    with pytest.raises(RuntimeError, match="publish a new schema"):
        freshness.require_current_native_build_attestation_schema()


@pytest.mark.parametrize("tamper", ["unknown_schema", "added_dependency"])
def test_loader_rejects_native_attestation_schema_or_dependency_set_tamper(
    tmp_path: Path, tamper: str,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    build = manifest["features_extension"]["build_attestation"]
    if tamper == "unknown_schema":
        build["schema"] = "deepfin.native_build.unknown"
    else:
        build["dependencies"]["chess_anti_engine/unexpected.h"] = "f" * 64
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="feature encoding extension"):
        load_transitions(bank)


@pytest.mark.parametrize(
    ("field", "value"),
    [("root_probe_active", False), ("leaf_probe_active", True)],
)
def test_loader_records_actual_walker_tablebase_semantics(
    tmp_path: Path, field: str, value: bool,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["realized_tablebase"][field] = value
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="production Syzygy probe"):
        load_transitions(bank)


def test_tablebase_inventory_binds_restored_mtime_content_mutation(
    tmp_path: Path,
) -> None:
    tablebases = tmp_path / "tablebases"
    tablebases.mkdir()
    table_file = tablebases / "KQvK.rtbw"
    table_file.write_bytes(b"first identity")
    before_stat = table_file.stat()
    before = _tablebase_inventory(str(tablebases))

    table_file.write_bytes(b"other identity")
    os.utime(
        table_file,
        ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns),
        follow_symlinks=False,
    )
    after_stat = table_file.stat()
    after = _tablebase_inventory(str(tablebases))

    assert after_stat.st_size == before_stat.st_size
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
    assert after_stat.st_ctime_ns != before_stat.st_ctime_ns
    assert after != before
    assert (
        after["directories"][0]["file_identities"][0][3]
        != before["directories"][0]["file_identities"][0][3]
    )


def test_tablebase_inventory_hashes_approved_contents_once_and_reuses_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "syzygy_3-4-5"
    second = tmp_path / "syzygy_6"
    first.mkdir()
    second.mkdir()
    wdl_bytes = b"approved wdl"
    dtz_bytes = b"approved dtz"
    (first / "KQvK.rtbw").write_bytes(wdl_bytes)
    (second / "KQvK.rtbw").write_bytes(wdl_bytes)
    dtz_path = second / "KQvK.rtbz"
    dtz_path.write_bytes(dtz_bytes)
    catalog_rows = (
        (
            "KQvK.rtbw",
            hashlib.md5(wdl_bytes, usedforsecurity=False).hexdigest(),
        ),
        (
            "KQvK.rtbz",
            hashlib.md5(dtz_bytes, usedforsecurity=False).hexdigest(),
        ),
    )
    catalog_bytes = "".join(
        f"{digest}  {name}\n" for name, digest in catalog_rows
    ).encode("ascii")
    (second / "3-4-5-6.md5").write_bytes(catalog_bytes)

    components = tuple(
        ApprovedSyzygyComponent(
            directory_name=directory.name,
            rtbw_count=sum(path.suffix == ".rtbw" for path in directory.iterdir()),
            rtbz_count=sum(path.suffix == ".rtbz" for path in directory.iterdir()),
            file_count=len([
                path for path in directory.iterdir()
                if path.suffix in (".rtbw", ".rtbz")
            ]),
            total_bytes=sum(
                path.stat().st_size for path in directory.iterdir()
                if path.suffix in (".rtbw", ".rtbz")
            ),
            filename_size_sha256=filename_size_sha256(
                (path.name, path.stat().st_size) for path in directory.iterdir()
                if path.suffix in (".rtbw", ".rtbz")
            ),
        )
        for directory in (first, second)
    )
    monkeypatch.setattr(
        trajectory_module, "_APPROVED_SYZYGY_COMPONENTS", components,
    )
    monkeypatch.setattr(
        trajectory_module, "APPROVED_SYZYGY_CHECKSUM_CATALOG_SIZE",
        len(catalog_bytes),
    )
    monkeypatch.setattr(
        trajectory_module, "APPROVED_SYZYGY_CHECKSUM_CATALOG_RAW_SHA256",
        hashlib.sha256(catalog_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        trajectory_module, "APPROVED_SYZYGY_CHECKSUM_CATALOG_ENTRY_COUNT", 2,
    )
    monkeypatch.setattr(
        trajectory_module, "APPROVED_SYZYGY_CHECKSUM_CATALOG_WDL_COUNT", 1,
    )
    monkeypatch.setattr(
        trajectory_module, "APPROVED_SYZYGY_CHECKSUM_CATALOG_DTZ_COUNT", 1,
    )
    monkeypatch.setattr(
        trajectory_module, "APPROVED_SYZYGY_CHECKSUM_CATALOG_ENTRIES_SHA256",
        checksum_catalog_entries_sha256(catalog_rows),
    )
    path_value = f"{first}{os.pathsep}{second}"
    initial = _tablebase_inventory(
        path_value, require_approved=True, verify_contents=True,
    )
    assert initial["content_verification"] == {
        **initial["content_verification"],
        "file_count": 3,
        "bytes_hashed": 2 * len(wdl_bytes) + len(dtz_bytes),
        "passed": True,
    }
    final = _tablebase_inventory(
        path_value,
        require_approved=True,
        prior_content_verification=initial["content_verification"],
    )
    assert final == initial

    before = dtz_path.stat()
    dtz_path.write_bytes(b"corrupt dtz!")
    assert dtz_path.stat().st_size == before.st_size
    os.utime(
        dtz_path, ns=(before.st_atime_ns, before.st_mtime_ns),
        follow_symlinks=False,
    )
    with pytest.raises(SystemExit, match="content checksum mismatch"):
        _tablebase_inventory(
            path_value, require_approved=True, verify_contents=True,
        )


def test_tablebase_inventory_rejects_noncanonical_parent_traversal(
    tmp_path: Path,
) -> None:
    lexical = tmp_path / "lexical"
    alternate = tmp_path / "alternate"
    lexical.mkdir()
    (alternate / "child").mkdir(parents=True)
    (lexical / "link").symlink_to(alternate / "child", target_is_directory=True)
    raw = f"{lexical}/link/../tablebases"

    with pytest.raises(SystemExit, match="canonical absolute paths"):
        _tablebase_inventory(raw)


def test_tablebase_inventory_rejects_double_leading_separator() -> None:
    with pytest.raises(SystemExit, match="canonical absolute paths"):
        _tablebase_inventory("//server/tables")


def test_tablebase_inventory_binds_ancestor_swap_and_restore(
    tmp_path: Path,
) -> None:
    slot = tmp_path / "slot"
    tablebases = slot / "tablebases"
    tablebases.mkdir(parents=True)
    (tablebases / "KQvK.rtbw").write_bytes(b"trusted table")
    alternate = tmp_path / "alternate"
    evil_tablebases = alternate / "tablebases"
    evil_tablebases.mkdir(parents=True)
    (evil_tablebases / "KQvK.rtbw").write_bytes(b"hostile table")
    parent_before = tmp_path.stat()
    before = _tablebase_inventory(str(tablebases))

    saved = tmp_path / "saved"
    slot.rename(saved)
    alternate.rename(slot)
    slot.rename(alternate)
    saved.rename(slot)
    os.utime(
        tmp_path,
        ns=(parent_before.st_atime_ns, parent_before.st_mtime_ns),
        follow_symlinks=False,
    )
    after = _tablebase_inventory(str(tablebases))

    before_directory = before["directories"][0]
    after_directory = after["directories"][0]
    assert after_directory["root_identity"] == before_directory["root_identity"]
    assert after_directory["file_identities"] == before_directory["file_identities"]
    assert after_directory["path_components"] != before_directory["path_components"]
    assert after != before


def test_tablebase_inventory_rejects_symlinked_root(tmp_path: Path) -> None:
    tablebases = tmp_path / "tablebases"
    tablebases.mkdir()
    (tablebases / "KQvK.rtbw").write_bytes(b"table")
    alias = tmp_path / "tablebase-alias"
    alias.symlink_to(tablebases, target_is_directory=True)

    with pytest.raises(SystemExit, match="symlink"):
        _tablebase_inventory(str(alias))


def test_tablebase_inventory_rejects_symlinked_parent_directory(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    tablebases = real_parent / "tablebases"
    tablebases.mkdir(parents=True)
    (tablebases / "KQvK.rtbw").write_bytes(b"table")
    alias = tmp_path / "parent-alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(SystemExit, match="symlink"):
        _tablebase_inventory(str(alias / "tablebases"))


@pytest.mark.parametrize("target_is_directory", [False, True])
def test_tablebase_inventory_rejects_symlinked_entries(
    tmp_path: Path, target_is_directory: bool,
) -> None:
    tablebases = tmp_path / "tablebases"
    tablebases.mkdir()
    if target_is_directory:
        target = tmp_path / "other-tablebases"
        target.mkdir()
        alias = tablebases / "other"
    else:
        target = tmp_path / "other.rtbw"
        target.write_bytes(b"table")
        alias = tablebases / "KQvK.rtbw"
    alias.symlink_to(target, target_is_directory=target_is_directory)

    with pytest.raises(SystemExit, match="symlink"):
        _tablebase_inventory(str(tablebases))


def test_tablebase_inventory_rejects_nonregular_entries(tmp_path: Path) -> None:
    tablebases = tmp_path / "tablebases"
    tablebases.mkdir()
    (tablebases / "nested").mkdir()

    with pytest.raises(SystemExit, match="regular files"):
        _tablebase_inventory(str(tablebases))


def test_tablebase_inventory_rejects_directory_change_during_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tablebases = tmp_path / "tablebases"
    tablebases.mkdir()
    (tablebases / "KQvK.rtbw").write_bytes(b"table")
    target_inode = tablebases.stat().st_ino
    real_fstat = os.fstat
    target_calls = 0

    def changing_fstat(file_descriptor: int) -> os.stat_result | SimpleNamespace:
        nonlocal target_calls
        observed = real_fstat(file_descriptor)
        if stat.S_ISDIR(observed.st_mode) and observed.st_ino == target_inode:
            target_calls += 1
            if target_calls == 3:
                return SimpleNamespace(
                    st_mode=observed.st_mode,
                    st_size=observed.st_size,
                    st_mtime_ns=observed.st_mtime_ns,
                    st_ctime_ns=observed.st_ctime_ns + 1,
                    st_dev=observed.st_dev,
                    st_ino=observed.st_ino,
                )
        return observed

    monkeypatch.setattr(os, "fstat", changing_fstat)

    with pytest.raises(SystemExit, match="changed while it was inventoried"):
        _tablebase_inventory(str(tablebases))


@pytest.mark.parametrize(
    "tamper",
    [
        "schema", "root_identity", "path_component", "file_identity",
        "directory_digest", "global_digest",
    ],
)
def test_loader_rejects_malformed_syzygy_identity_inventory(
    tmp_path: Path, tamper: str,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    syzygy = manifest["syzygy"]
    if tamper == "schema":
        syzygy["schema"] = "deepfin.syzygy_inventory.v1"
    elif tamper == "root_identity":
        syzygy["directories"][0]["root_identity"].pop("ctime_ns")
    elif tamper == "path_component":
        syzygy["directories"][0]["path_components"][0]["ctime_ns"] += 1
    elif tamper == "file_identity":
        syzygy["directories"][0]["file_identities"][0][3] += 1
    elif tamper == "directory_digest":
        syzygy["directories"][0]["inventory_sha256"] = "0" * 64
    else:
        syzygy["inventory_sha256"] = "0" * 64
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="production Syzygy provenance"):
        load_transitions(bank)


def test_loader_rejects_partial_syzygy_inventory(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["syzygy"]["directories"][0]["rtbw_count"] = 500
    manifest["syzygy"]["directories"][1]["rtbw_count"] = 375
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="production Syzygy provenance"):
        load_transitions(bank)


@pytest.mark.parametrize(
    "path_value",
    [
        "/tb/syzygy_3-4-5/../syzygy_3-4-5:/tb/syzygy_6",
        "/tb//syzygy_3-4-5:/tb/syzygy_6",
        "/tb/syzygy_3-4-5/:/tb/syzygy_6",
        "/tb/syzygy_3-4-5:/tb/./syzygy_6",
        "//tb/syzygy_3-4-5:/tb/syzygy_6",
    ],
)
def test_loader_rejects_noncanonical_resolving_equal_syzygy_path(
    tmp_path: Path, path_value: str,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["syzygy"]["path"] = path_value
    _refresh_synthetic_syzygy_integrity(manifest["syzygy"])
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="production Syzygy provenance"):
        load_transitions(bank)


def test_loader_rejects_self_consistent_wrong_syzygy_component_name(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    syzygy = manifest["syzygy"]
    first = syzygy["directories"][0]
    dtz = next(row for row in first["file_identities"] if row[0] == "z000.rtbz")
    dtz[0] = "z200.rtbz"
    _refresh_synthetic_syzygy_integrity(syzygy)
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="production Syzygy provenance"):
        load_transitions(bank)


def test_loader_rejects_self_consistent_wrong_syzygy_file_size(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    syzygy = manifest["syzygy"]
    syzygy["directories"][0]["file_identities"][0][1] += 1
    _refresh_synthetic_syzygy_integrity(syzygy)
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="production Syzygy provenance"):
        load_transitions(bank)


def test_loader_rejects_self_consistent_forged_syzygy_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = tmp_path / "bank.jsonl"
    _write_bank(bank, correct_gap=True)
    monkeypatch.setattr(
        controller_module, "_APPROVED_SYZYGY_COMPONENTS",
        APPROVED_SYZYGY_COMPONENTS,
    )
    monkeypatch.setattr(
        controller_module,
        "APPROVED_SYZYGY_CHECKSUM_CATALOG_RAW_SHA256",
        APPROVED_SYZYGY_CHECKSUM_CATALOG_RAW_SHA256,
    )
    monkeypatch.setattr(
        controller_module,
        "APPROVED_SYZYGY_CHECKSUM_CATALOG_ENTRIES_SHA256",
        APPROVED_SYZYGY_CHECKSUM_CATALOG_ENTRIES_SHA256,
    )

    with pytest.raises(ValueError, match="production Syzygy provenance"):
        load_transitions(bank)


def test_loader_requires_native_halving_semantic_revision(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["mcts_extension"].pop("gss_halving_rev")
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="native MCTS extension"):
        load_transitions(bank)


@pytest.mark.parametrize(
    ("section", "field", "value", "match"),
    [
        ("search_warmup", "completed", False, "search warmup"),
        ("search_warmup", "realized_nodes", 0, "search warmup"),
        (
            "elapsed_measurement", "usable_for_controller_or_cost_analysis", True,
            "elapsed-time instrumentation",
        ),
        ("mcts_extension", "freshness_check", {}, "native MCTS extension"),
    ],
)
def test_loader_requires_warmup_timing_and_native_freshness_provenance(
    tmp_path: Path, section: str, field: str, value: object, match: str,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest[section][field] = value
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=match):
        load_transitions(bank)


def test_loader_requires_resolved_checkpoint_params_provenance(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest.pop("checkpoint_params")
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="checkpoint architecture provenance"):
        load_transitions(bank)


def test_loader_requires_complete_params_candidate_inventory(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest.pop("params_candidate_inventory")
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="checkpoint or params consumption"):
        load_transitions(bank)


def test_loader_rejects_tampered_params_candidate_negative_evidence(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["params_candidate_inventory"]["candidates"][0]["state"] = "regular"
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="checkpoint or params consumption"):
        load_transitions(bank)


def test_loader_validates_the_final_rows_full_cluster_identity(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    rows[-1]["source_dir"] = "/other"
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads(meta.read_text())
    manifest["output"] = {
        "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        "size": bank.stat().st_size,
    }
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="group_id is not"):
        load_transitions(bank)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shard", "other.zarr"),
        ("game_id", 99),
        (
            "deep_reference_move_cp",
            {
                "e2e4": 100.0,
                "a2a3": 90.0,
                "b2b3": 80.0,
                "b2b4": 80.0,
                "c2c3": 85.0,
                "c2c4": 80.0,
                "d2d3": 80.0,
                "d2d4": 80.0,
                "g1f3": 80.0,
                "b1c3": 80.0,
            },
        ),
    ],
)
def test_loader_requires_trajectory_identity_and_labels_to_stay_fixed(
    tmp_path: Path, field: str, value: object,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    rows[-1][field] = value
    if field == "game_id":
        rows[-1]["group_id"] = "\0".join((
            str(rows[-1]["source_dir"]),
            str(rows[-1]["game_id"]),
        ))
    if field == "deep_reference_move_cp":
        assert isinstance(value, dict)
        board = chess.Board(str(rows[-1]["fen"]))
        newly_listed = int(move_to_index(chess.Move.from_uci("c2c3"), board))
        listed_index = rows[-1]["root_actions"].index(newly_listed)
        rows[-1]["root_action_reference_listed"][listed_index] = True
        rows[-1]["root_action_reference_cp"][listed_index] = value["c2c3"]
        rows[-1]["root_action_regret_cp"][listed_index] = 15.0
        rows[-1]["deep_reference_scored_multipv"] = len(value)
    _rewrite_bank(bank, meta, rows)

    with pytest.raises(ValueError, match="trajectory-invariant fields change"):
        load_transitions(bank)


def _mark_emitted_reference_unlisted(row: dict[str, object]) -> None:
    actions = row["root_actions"]
    listed = row["root_action_reference_listed"]
    references = row["root_action_reference_cp"]
    assert isinstance(actions, list)
    assert isinstance(listed, list)
    assert isinstance(references, list)
    emitted_index = actions.index(row["emitted_action"])
    listed[emitted_index] = False
    references[emitted_index] = min(float(value) for value in references)
    row["emitted_reference_listed"] = False
    move_cp = row["deep_reference_move_cp"]
    assert isinstance(move_cp, dict)
    move_cp.pop(str(row["uci"]))
    replacement_uci = "g1h3"
    move_cp[replacement_uci] = min(float(value) for value in move_cp.values())
    board = chess.Board(str(row["fen"]))
    replacement_action = int(move_to_index(chess.Move.from_uci(replacement_uci), board))
    replacement_index = actions.index(replacement_action)
    listed[replacement_index] = True
    row["deep_reference_scored_multipv"] = len(move_cp)


@pytest.mark.parametrize(
    ("listed", "expected_unlisted_rows"),
    [((True, False), 1), ((False, False), 2)],
    ids=("listed-unlisted", "unlisted-unlisted"),
)
def test_reference_censoring_marks_each_adjacent_unlisted_pair_once(
    listed: tuple[bool, bool], expected_unlisted_rows: int,
) -> None:
    rows = [
        {
            "chunk": chunk,
            "nodes": chunk * 50,
            "actions": [7],
            "action_reference_listed": [is_listed],
            "emitted_action": 7,
            "emitted_reference_listed": is_listed,
            "uci": "a2a3",
        }
        for chunk, is_listed in enumerate(listed, start=1)
    ]

    details = controller_module._trajectory_reference_censoring("position", rows)

    assert details is not None
    summary = controller_module._reference_censoring_summary([details])
    assert summary["unlisted_emitted_row_count"] == expected_unlisted_rows
    assert summary["censored_transition_count"] == 1
    assert details["censored_transitions"] == [{
        "from_chunk": 1, "to_chunk": 2, "horizon_nodes": 100,
    }]


@pytest.mark.parametrize(
    ("unlisted_indices", "expected_unlisted_rows", "expected_censored_transitions"),
    [
        ((1,), 1, 2),
        ((0, 1), 2, 2),
    ],
    ids=("listed-unlisted", "unlisted-unlisted"),
)
def test_finite_multipv_censoring_preserves_rows_but_disqualifies_the_bank(
    tmp_path: Path,
    unlisted_indices: tuple[int, ...],
    expected_unlisted_rows: int,
    expected_censored_transitions: int,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    for index in unlisted_indices:
        _mark_emitted_reference_unlisted(rows[index])
    details = controller_module._trajectory_reference_censoring(
        str(rows[0]["key"]), rows,
    )
    assert details is not None
    censoring = controller_module._reference_censoring_summary([details])
    assert censoring["unlisted_emitted_row_count"] == expected_unlisted_rows
    assert censoring["censored_transition_count"] == expected_censored_transitions
    assert censoring["passed"] is False
    _rewrite_bank(bank, meta, rows)
    manifest = json.loads(meta.read_text())
    manifest["decision_grade"] = False
    manifest["reference_censoring"] = censoring
    meta.write_text(json.dumps(manifest))

    transitions, info = load_transitions(bank, methodology_smoke=True)

    assert len(transitions) == 3
    assert info["decision_grade"] is False
    assert info["reference_censoring"] == censoring
    assert len(bank.read_text().splitlines()) == 4


def test_decision_grade_row_rejects_an_unlisted_emitted_move(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    _write_bank(bank, correct_gap=True)
    row = json.loads(bank.read_text().splitlines()[0])
    _mark_emitted_reference_unlisted(row)

    with pytest.raises(ValueError, match="decision label is censored"):
        controller_module._validate_decision_grade_row(
            row, 1, require_full_root_support=True,
        )


def test_decision_grade_manifest_rejects_finite_multipv_censoring(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["reference_censoring"] = {
        "kind": "finite_multipv_unlisted_emitted_move",
        "scope": "completed_trajectory_decision_labels",
        "decision_labels_require_listed_emitted_moves": True,
        "passed": False,
        "affected_trajectory_count": 1,
        "unlisted_emitted_row_count": 1,
        "censored_transition_count": 1,
        "affected_trajectories": [{
            "key": "position",
            "unlisted_emitted_rows": [{
                "chunk": 1, "nodes": 50, "emitted_action": 1, "uci": "a2a3",
            }],
            "censored_transitions": [{
                "from_chunk": 1, "to_chunk": 2, "horizon_nodes": 100,
            }],
        }],
    }
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="finite-MultiPV censoring"):
        load_transitions(bank)


def test_loader_validates_final_row_stability_semantics(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    rows[-1]["stable_chunks"] = 1
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads(meta.read_text())
    manifest["output"] = {
        "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        "size": bank.stat().st_size,
    }
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="stable_chunks disagrees"):
        load_transitions(bank)


def test_loader_requires_manifest_position_count_unique_trajectories(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    rows.extend(json.loads(json.dumps(row)) for row in rows[:])
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads(meta.read_text())
    manifest.update({"row_count": 8, "position_count": 2, "requested_position_count": 2})
    claimed_second_position = json.loads(json.dumps(rows[0]))
    claimed_second_position.update({"key": "claimed-second-position", "source": 1})
    _sync_test_panel_selection(manifest, [rows[0], claimed_second_position])
    manifest["output"] = {
        "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        "size": bank.stat().st_size,
    }
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="unique trajectory count"):
        load_transitions(bank)


def test_loader_requires_every_trajectory_to_have_all_manifest_chunks(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    original = [json.loads(line) for line in bank.read_text().splitlines()]
    rows = [json.loads(json.dumps(row)) for row in original]
    rows[-1] = json.loads(json.dumps(original[2]))
    other_board = chess.Board(
        "rn3bnr/ppp1k2p/5pp1/Pb1pp3/1P3PPq/B7/N1PPP1BP/R2QK1NR w KQ - 4 14"
    )
    other_move = next(iter(other_board.legal_moves))
    other_action = int(move_to_index(other_move, other_board))
    for chunk, template in enumerate(original, start=1):
        row = json.loads(json.dumps(template))
        row.update({
            "key": position_key(other_board), "fen": other_board.fen(),
            "game_id": 4,
            "group_id": "\0".join(("/snapshot", "4")),
            "chunk": chunk, "nodes": chunk * 50, "elapsed_ms": float(chunk),
            "root_actions": [other_action], "root_visits": [chunk * 50],
            "root_visit_shares": [1.0], "root_child_q": [0.1],
            "root_child_q_observed": [True],
            "root_action_regret_cp": [0.0],
            "root_action_reference_cp": [90.0],
            "root_action_reference_listed": [True],
            "deep_reference_best_cp": 90.0,
            "deep_reference_move_cp": {other_move.uci(): 90.0},
            "deep_reference_scored_multipv": 1,
            "regret_cp": 0.0, "regret_score": 0.0,
            "emitted_action": other_action, "uci": other_move.uci(),
            "pv_actions": [other_action], "pv_uci": [other_move.uci()],
            "visit_gap": 1.0, "visit_entropy": 0.0, "q_gap": None,
            "stable_chunks": chunk - 1,
            "complexity_predicate_continue": chunk < 3,
            "q_drift": None if chunk == 1 else 0.0,
            "visit_churn": None if chunk == 1 else 0.0,
            "phase": phase_bucket(chess.popcount(other_board.occupied)),
            "source": 1,
            "piece_count": chess.popcount(other_board.occupied),
            "legal_move_count": 1,
        })
        rows.append(row)
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads(meta.read_text())
    manifest.update({"row_count": 8, "position_count": 2, "requested_position_count": 2})
    _sync_test_panel_selection(manifest, [rows[0], rows[4]])
    manifest["output"] = {
        "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        "size": bank.stat().st_size,
    }
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="complete consecutive manifest horizon"):
        load_transitions(bank)


def test_loader_accepts_production_forced_move_stability_semantics(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    original = [json.loads(line) for line in bank.read_text().splitlines()]
    forced_board = chess.Board(
        "rn3bnr/ppp1k2p/5pp1/Pb1pp3/1P3PPq/B7/N1PPP1BP/R2QK1NR w KQ - 4 14"
    )
    forced_move = next(iter(forced_board.legal_moves))
    assert forced_board.legal_moves.count() == 1
    forced_action = int(move_to_index(forced_move, forced_board))
    rows = []
    for chunk in (1, 2, 3, 4):
        row = json.loads(json.dumps(original[min(chunk - 1, 1)]))
        row.update({
            "key": position_key(forced_board), "fen": forced_board.fen(),
            "chunk": chunk, "nodes": chunk * 50, "elapsed_ms": float(chunk),
            "root_actions": [forced_action],
            "root_visits": [chunk * 50], "root_visit_shares": [1.0],
            "root_child_q": [0.1], "root_child_q_observed": [True],
            "root_action_regret_cp": [0.0],
            "root_action_reference_cp": [90.0],
            "root_action_reference_listed": [True],
            "deep_reference_best_cp": 90.0,
            "deep_reference_move_cp": {forced_move.uci(): 90.0},
            "deep_reference_scored_multipv": 1,
            "regret_cp": 0.0, "regret_score": 0.0,
            "emitted_action": forced_action, "uci": forced_move.uci(),
            "pv_actions": [forced_action], "pv_uci": [forced_move.uci()],
            "visit_gap": 1.0, "visit_entropy": 0.0, "q_gap": None,
            "stable_chunks": chunk - 1,
            "complexity_predicate_continue": chunk < 3,
            "phase": phase_bucket(chess.popcount(forced_board.occupied)),
            "piece_count": chess.popcount(forced_board.occupied),
            "legal_move_count": 1,
        })
        rows.append(row)
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads(meta.read_text())
    manifest.update({"row_count": 4, "chunk_count": 4})
    manifest["requested_search"]["max_chunks"] = 4
    _sync_test_panel_selection(manifest, [rows[0]])
    manifest["output"] = {
        "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        "size": bank.stat().st_size,
    }
    meta.write_text(json.dumps(manifest))

    transitions, _ = load_transitions(bank)

    assert len(transitions) == 3
    assert transitions[-1].complexity_continue is False


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("root_child_q", [0.1, float("nan")], "root child Q"),
        ("root_action_regret_cp", [10.0, float("nan")], "root action regrets"),
        ("pv_actions", [0], "PV provenance"),
        ("pv_uci", ["b2b3"], "PV provenance"),
    ],
)
def test_loader_requires_reusable_root_q_and_pv_observations(
    tmp_path: Path, field: str, value: object, match: str,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    if field in ("root_child_q", "root_action_regret_cp"):
        assert isinstance(value, list)
        expanded = list(rows[-1][field])
        expanded[-1] = value[-1]
        value = expanded
    rows[-1][field] = value
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads(meta.read_text())
    manifest["output"] = {
        "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        "size": bank.stat().st_size,
    }
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=match):
        load_transitions(bank)


@pytest.mark.parametrize("location", ["root", "later_pv"])
def test_loader_strictly_rejects_undecodable_native_actions(
    tmp_path: Path, location: str,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    if location == "root":
        rows[-1]["root_actions"][1] = 99_999
    else:
        rows[-1]["pv_actions"].append(99_999)
        rows[-1]["pv_uci"].append("a7a6")
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads(meta.read_text())
    manifest["output"] = {
        "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        "size": bank.stat().st_size,
    }
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="action cannot be decoded"):
        load_transitions(bank)


def test_loader_requires_full_walker_root_support(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    for row in rows:
        zero_index = row["root_visits"].index(0)
        for field in (
            "root_actions", "root_visits", "root_visit_shares", "root_child_q",
            "root_child_q_observed", "root_action_regret_cp",
            "root_action_reference_cp", "root_action_reference_listed",
        ):
            row[field].pop(zero_index)
    _rewrite_bank(bank, meta, rows)

    with pytest.raises(ValueError, match="every legal action"):
        load_transitions(bank)


def test_loader_rejects_decreasing_accumulated_root_visits(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    row = rows[1]
    nonzero = [i for i, visit in enumerate(row["root_visits"]) if visit > 0]
    emitted_index = row["root_actions"].index(row["emitted_action"])
    other_index = next(index for index in nonzero if index != emitted_index)
    row["root_visits"][emitted_index] = 19
    row["root_visits"][other_index] = 81
    row["root_visit_shares"][emitted_index] = 0.19
    row["root_visit_shares"][other_index] = 0.81
    row["visit_gap"] = -0.62
    row["visit_entropy"] = float(-(0.19 * np.log(0.19) + 0.81 * np.log(0.81)))
    row["visit_churn"] = 0.21
    _rewrite_bank(bank, meta, rows)

    with pytest.raises(ValueError, match="root visits decrease"):
        load_transitions(bank)


def test_loader_requires_root_visits_to_account_for_completed_nodes(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    row = rows[-1]
    nonzero = [i for i, visit in enumerate(row["root_visits"]) if visit > 0]
    row["root_visits"][nonzero[0]] -= 1
    total = sum(row["root_visits"])
    row["root_visit_shares"] = [visit / total for visit in row["root_visits"]]
    emitted_index = row["root_actions"].index(row["emitted_action"])
    alternatives = [
        share for index, share in enumerate(row["root_visit_shares"])
        if index != emitted_index
    ]
    row["visit_gap"] = row["root_visit_shares"][emitted_index] - max(alternatives)
    positive = np.asarray([s for s in row["root_visit_shares"] if s > 0.0])
    row["visit_entropy"] = float(-(positive * np.log(positive)).sum())
    _rewrite_bank(bank, meta, rows)

    with pytest.raises(ValueError, match="completed nodes"):
        load_transitions(bank)


def test_gumbel_zero_visit_exception_requires_a_truly_forced_position(
    tmp_path: Path,
) -> None:
    from scripts.analyze_chunk_controller import _validate_decision_grade_row

    bank = tmp_path / "bank.jsonl"
    _write_bank(bank, correct_gap=True)
    row = json.loads(bank.read_text().splitlines()[0])
    emitted_index = row["root_actions"].index(row["emitted_action"])
    for field in (
        "root_actions", "root_child_q", "root_action_regret_cp",
        "root_action_reference_cp", "root_action_reference_listed",
    ):
        row[field] = [row[field][emitted_index]]
    row["root_visits"] = [0]
    row["root_visit_shares"] = [0.0]
    row["root_child_q_observed"] = [False]
    row["visit_gap"] = 0.0
    row["visit_entropy"] = 0.0
    row["q_gap"] = None

    with pytest.raises(ValueError, match="completed nodes"):
        _validate_decision_grade_row(row, 1, require_full_root_support=False)


def test_gumbel_zero_visit_exception_accepts_a_truly_forced_position(
    tmp_path: Path,
) -> None:
    from scripts.analyze_chunk_controller import _validate_decision_grade_row

    bank = tmp_path / "bank.jsonl"
    _write_bank(bank, correct_gap=True)
    row = json.loads(bank.read_text().splitlines()[0])
    board = chess.Board(
        "rn3bnr/ppp1k2p/5pp1/Pb1pp3/1P3PPq/B7/N1PPP1BP/R2QK1NR w KQ - 4 14"
    )
    move = next(iter(board.legal_moves))
    assert board.legal_moves.count() == 1
    action = int(move_to_index(move, board))
    row.update({
        "key": position_key(board), "fen": board.fen(),
        "phase": phase_bucket(chess.popcount(board.occupied)),
        "piece_count": chess.popcount(board.occupied), "legal_move_count": 1,
        "root_actions": [action], "root_visits": [0], "root_visit_shares": [0.0],
        "root_child_q": [0.0], "root_child_q_observed": [False],
        "root_action_regret_cp": [0.0], "root_action_reference_cp": [90.0],
        "root_action_reference_listed": [True],
        "deep_reference_best_cp": 90.0, "deep_reference_move_cp": {move.uci(): 90.0},
        "deep_reference_scored_multipv": 1,
        "regret_cp": 0.0, "regret_score": 0.0, "regret_vs_final_cp": 0.0,
        "emitted_action": action, "uci": move.uci(), "pv_actions": [action],
        "pv_uci": [move.uci()], "visit_gap": 0.0, "visit_entropy": 0.0,
        "q_gap": None, "root_q": 0.0,
    })

    _validate_decision_grade_row(row, 1, require_full_root_support=False)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("key", "invented", "position key disagrees"),
        ("phase", 1, "phase disagrees"),
        ("source", 7, "source must be"),
    ],
)
def test_loader_recomputes_position_identity_and_domains(
    tmp_path: Path, field: str, value: object, match: str,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    rows[0][field] = value
    _rewrite_bank(bank, meta, rows)

    with pytest.raises(ValueError, match=match):
        load_transitions(bank)


def test_loader_disqualifies_incomplete_search_exclusions(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest.update({
        "requested_position_count": 2,
        "excluded_position_count": 1,
        "incomplete_exclusion_count": 1,
        "excluded_positions": [{
            "key": "missing",
            "chunks_observed": 1,
            "chunks_required": 4,
            "reason": "incomplete_search",
            "search_result": {
                "nodes": 50, "tbhits": 0, "root_declined": "declined",
                "score_mate": None, "board_game_over": False,
            },
        }],
    })
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="accounting is inconsistent"):
        load_transitions(bank)


def test_loader_rejects_multinode_terminal_shortcut_exclusion(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest.update({
        "requested_position_count": 2,
        "excluded_position_count": 1,
        "excluded_positions": [{
            "key": "missing",
            "chunks_observed": 0,
            "chunks_required": 4,
            "reason": "production_terminal_shortcut",
            "search_result": {
                "nodes": 2, "tbhits": 0, "root_declined": None,
                "score_mate": 1, "board_game_over": False,
            },
        }],
    })
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="accounting is inconsistent"):
        load_transitions(bank)


def test_loader_rejects_source_balanced_panel_after_terminal_exclusion(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    terminal_board = chess.Board("7k/8/8/8/8/8/8/KQ6 w - - 0 1")
    assert not terminal_board.is_game_over()
    excluded = {
        "key": position_key(terminal_board),
        "fen": terminal_board.fen(),
        "source_dir": "/snapshot",
        "shard": "s1.zarr",
        "game_id": 4,
        "group_id": "/snapshot\0" + "4",
        "phase": phase_bucket(chess.popcount(terminal_board.occupied)),
        "source": 1,
        "piece_count": chess.popcount(terminal_board.occupied),
        "deep_reference_nodes": 1_000_000,
        "deep_reference_depth": 30,
        "deep_reference_best_cp": 0.0,
        "deep_reference_move_cp": {
            move.uci(): 0.0 for move in list(terminal_board.legal_moves)[:10]
        },
        "deep_reference_scored_multipv": 10,
        "chunks_observed": 0,
        "chunks_required": 4,
        "partial_observations": [],
        "reason": "production_terminal_shortcut",
        "search_result": {
            "nodes": 0,
            "tbhits": 1,
            "root_declined": None,
            "score_mate": None,
            "board_game_over": False,
        },
    }
    manifest = json.loads(meta.read_text())
    manifest.update({
        "requested_position_count": 2,
        "excluded_position_count": 1,
        "excluded_positions": [excluded],
    })
    _sync_test_panel_selection(manifest, [rows[0], excluded])
    meta.write_text(json.dumps(manifest))

    assert manifest["panel_selection"]["source_balance"] == {
        "maximum_difference": 1,
        "observed_difference": 0,
        "passed": True,
    }
    with pytest.raises(
        ValueError,
        match="decision-grade bank contains selected-position exclusions",
    ):
        load_transitions(bank)


@pytest.mark.parametrize(
    ("row_index", "field", "value", "match"),
    [
        (0, "visit_entropy", 0.1, "visit_entropy disagrees"),
        (0, "regret_cp", 11.0, "regret_cp disagrees"),
        (0, "regret_score", 0.5, "regret_score disagrees"),
        (0, "q_gap", 0.0, "q_gap disagrees"),
        (1, "bestmove_flip", True, "bestmove_flip disagrees"),
        (1, "q_drift", 0.1, "q_drift disagrees"),
        (1, "visit_churn", 0.1, "visit_churn disagrees"),
        (0, "changes_to_final", True, "changes_to_final disagrees"),
        (0, "regret_vs_final_cp", 99.0, "regret_vs_final_cp disagrees"),
        (0, "root_visits", [10, 40], "shares disagree"),
    ],
)
def test_loader_recomputes_derived_trajectory_fields(
    tmp_path: Path, row_index: int, field: str, value: object, match: str,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    if field == "root_visits":
        assert isinstance(value, list)
        assert len(value) == 2
        expanded = list(rows[row_index][field])
        nonzero = [i for i, visit in enumerate(expanded) if visit > 0]
        expanded[nonzero[0]], expanded[nonzero[1]] = value
        value = expanded
    rows[row_index][field] = value
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads(meta.read_text())
    manifest["output"] = {
        "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        "size": bank.stat().st_size,
    }
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=match):
        load_transitions(bank)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("regret_score", "bad"),
        ("regret_score", float("nan")),
        ("regret_score", float("inf")),
        ("regret_score", float("-inf")),
        ("root_q", "bad"),
        ("visit_entropy", float("nan")),
        ("q_gap", float("nan")),
        ("q_drift", float("inf")),
        ("visit_churn", float("-inf")),
    ],
)
def test_loader_rejects_nonfinite_or_nonnumeric_decision_fields(
    tmp_path: Path, field: str, value: object,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    target = rows[-1] if field == "regret_score" else rows[0]
    target[field] = value
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads(meta.read_text())
    manifest["output"] = {
        "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        "size": bank.stat().st_size,
    }
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=rf"{field} must be a finite number"):
        load_transitions(bank)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("regret_cp", -1.0, "audit regret domain"),
        ("regret_score", 1.1, r"outside \[0, 1\]"),
        ("visit_gap", 1.1, r"outside \[-1, 1\]"),
        ("root_q", -1.1, r"outside \[-1, 1\]"),
        ("q_gap", 2.1, r"outside \[-2, 2\]"),
        ("q_drift", 2.1, r"outside \[0, 2\]"),
        ("visit_churn", 1.1, r"outside \[0, 1\]"),
    ],
)
def test_loader_rejects_finite_values_outside_metric_domains(
    tmp_path: Path, field: str, value: float, match: str,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    target = rows[-1] if field in {"q_drift", "visit_churn"} else rows[0]
    target[field] = value
    _rewrite_bank(bank, meta, rows)

    with pytest.raises(ValueError, match=match):
        load_transitions(bank)


def test_output_path_cannot_replace_the_bank_or_manifest(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    checkpoint = tmp_path / "trainer.pt"
    matched_report = tmp_path / "matched.npz.report.json"
    snapshot = tmp_path / "snapshot"
    tablebases = tmp_path / "syzygy_6"
    _require_safe_output_path(bank, meta, tmp_path / "report.json")
    with pytest.raises(ValueError, match="must not overwrite"):
        _require_safe_output_path(bank, meta, bank)
    with pytest.raises(ValueError, match="must not overwrite"):
        _require_safe_output_path(bank, meta, meta)
    manifest = {
        "checkpoint": {"path": str(checkpoint)},
        "params_candidate_inventory": {
            "candidates": [{"path": str(tmp_path / "absent-params.json")}],
        },
        "matched_rows_report": {"path": str(matched_report)},
        "matched_row_origin_verification": {
            "snapshot_inventory": {"path": str(snapshot)},
        },
        "preregistration": {"path": str(tmp_path / "preregister.json")},
        "syzygy": {"directories": [{"path": str(tablebases)}]},
    }
    with pytest.raises(ValueError, match="consumed input artifact"):
        _require_safe_output_path(bank, meta, checkpoint, manifest=manifest)
    with pytest.raises(ValueError, match="consumed input artifact"):
        _require_safe_output_path(
            bank, meta, tmp_path / "absent-params.json", manifest=manifest,
        )
    with pytest.raises(ValueError, match="consumed input artifact"):
        _require_safe_output_path(
            bank, meta, tmp_path / "preregister.json", manifest=manifest,
        )
    with pytest.raises(ValueError, match="consumed input artifact"):
        _require_safe_output_path(bank, meta, matched_report, manifest=manifest)
    with pytest.raises(ValueError, match="replay-snapshot"):
        _require_safe_output_path(
            bank, meta, snapshot / "s0.zarr" / "analysis.json", manifest=manifest,
        )
    with pytest.raises(ValueError, match="Syzygy"):
        _require_safe_output_path(
            bank, meta, tablebases / "report.json", manifest=manifest,
        )


def test_producer_output_paths_cannot_replace_inputs_or_tablebases(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "audit.jsonl"
    tablebases = tmp_path / "tb"
    tablebases.mkdir()
    with pytest.raises(SystemExit, match="aliases"):
        _require_safe_output_paths(
            audit, tmp_path / "out.meta.json",
            protected_files=[audit], protected_directories=[tablebases],
        )
    with pytest.raises(SystemExit, match="Syzygy"):
        _require_safe_output_paths(
            tablebases / "bank.jsonl", tmp_path / "out.meta.json",
            protected_files=[audit], protected_directories=[tablebases],
        )


@pytest.mark.parametrize("overwrite", [False, True])
def test_producer_overwrite_cannot_replace_matched_report_or_snapshot(
    tmp_path: Path, overwrite: bool,
) -> None:
    """The evidence guard precedes and dominates the ordinary overwrite flag."""
    report = tmp_path / "matched.npz.report.json"
    report.write_text("frozen report\n")
    snapshot = tmp_path / "snapshot"
    nested = snapshot / "s0.zarr" / "trajectory.jsonl"

    def attempt(output: Path) -> None:
        meta = tmp_path / "trajectory.meta.json"
        _require_safe_output_paths(
            output,
            meta,
            protected_files=[report],
            protected_directories=[snapshot],
        )
        _require_new_output_pair(output, meta, overwrite=overwrite)

    with pytest.raises(SystemExit, match="aliases"):
        attempt(report)
    with pytest.raises(SystemExit, match="replay-snapshot"):
        attempt(nested)
    with pytest.raises(SystemExit, match="replay-snapshot"):
        _require_safe_output_paths(
            tmp_path / "trajectory.jsonl",
            snapshot / "s0.zarr" / "trajectory.meta.json",
            protected_files=[report],
            protected_directories=[snapshot],
        )
    assert report.read_text() == "frozen report\n"


def test_producer_preregistration_cannot_replace_origin_evidence(
    tmp_path: Path,
) -> None:
    report = tmp_path / "matched.npz.report.json"
    snapshot = tmp_path / "snapshot"
    with pytest.raises(SystemExit, match="aliases"):
        _require_safe_preregistration_path(
            report,
            protected_files=[report],
            protected_directories=[snapshot],
        )
    with pytest.raises(SystemExit, match="replay-snapshot"):
        _require_safe_preregistration_path(
            snapshot / "s0.zarr" / "plan.json",
            protected_files=[report],
            protected_directories=[snapshot],
        )


def test_output_guards_reject_linked_worktree_git_metadata(tmp_path: Path) -> None:
    from scripts.repo_output_guard import git_control_paths

    repo_root = Path(__file__).resolve().parents[1]
    git_directories = [path for path in git_control_paths(repo_root) if path.is_dir()]
    assert git_directories

    for git_directory in git_directories:
        target = git_directory / "HEAD"
        with pytest.raises(ValueError, match="repository-control"):
            _require_safe_output_path(
                tmp_path / "bank.jsonl",
                tmp_path / "bank.jsonl.meta.json",
                target,
            )
        with pytest.raises(SystemExit, match="repository-control"):
            _require_safe_output_paths(
                target,
                tmp_path / "out.meta.json",
                protected_files=[],
                protected_directories=[],
            )


def test_output_guard_fails_closed_on_fatal_git_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from scripts import repo_output_guard as guard

    monkeypatch.setattr(guard, "git_control_paths", lambda _root: ())
    monkeypatch.setattr(
        guard.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=128),
    )

    assert guard.repo_controlled_output(tmp_path / "report.json", tmp_path) is True


def test_producer_output_cannot_overwrite_an_imported_tracked_source(
    tmp_path: Path,
) -> None:
    from scripts import analyze_chunk_controller as controller

    with pytest.raises(SystemExit, match="tracked or repository-control"):
        _require_safe_output_paths(
            Path(controller.__file__),
            tmp_path / "out.meta.json",
            protected_files=[],
            protected_directories=[],
        )


def test_producer_output_lock_serializes_the_bank_manifest_pair(tmp_path: Path) -> None:
    output = tmp_path / "bank.jsonl"
    first = _acquire_output_lock(output)
    try:
        with pytest.raises(SystemExit, match="another producer"):
            _acquire_output_lock(output)
    finally:
        first.close()

    retry = _acquire_output_lock(output)
    retry.close()


def test_producer_output_locks_serialize_overlapping_pairs(tmp_path: Path) -> None:
    output = tmp_path / "bank.jsonl"
    meta = Path(str(output) + ".meta.json")
    first = _acquire_output_locks(output, meta)
    try:
        with pytest.raises(SystemExit, match="another producer holds"):
            _acquire_output_locks(meta, Path(str(meta) + ".meta.json"))
    finally:
        for handle in reversed(first):
            handle.close()

    retry = _acquire_output_locks(meta, Path(str(meta) + ".meta.json"))
    for handle in reversed(retry):
        handle.close()


def test_producer_never_overwrites_frozen_evidence_pair(tmp_path: Path) -> None:
    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    output.write_text("old bank\n")
    meta.write_text("old manifest\n")

    with pytest.raises(SystemExit, match="immutable or incomplete evidence"):
        _require_new_output_pair(output, meta, overwrite=False)
    with pytest.raises(SystemExit, match="overwrite is disabled"):
        _require_new_output_pair(output, meta, overwrite=True)

    assert output.read_text() == "old bank\n"
    assert meta.read_text() == "old manifest\n"


def test_producer_accepts_a_new_evidence_pair_path(tmp_path: Path) -> None:
    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"

    assert _require_new_output_pair(output, meta, overwrite=False) is False


def test_producer_recovers_bank_published_before_its_manifest(tmp_path: Path) -> None:
    from scripts import backtest_chunk_trajectory as producer

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = producer._pending_output_path(output)
    pending_meta = producer._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": producer._SCHEMA,
        "complete": True,
        "output": producer._prepared_output_artifact(pending_output, output),
    }
    producer._write_json_staged(pending_meta, manifest)
    producer._publish_output(pending_output, output)

    assert not meta.exists()
    assert pending_meta.exists()
    assert _require_new_output_pair(output, meta, overwrite=False) is True
    assert output.read_text() == "completed bank\n"
    assert json.loads(meta.read_text()) == manifest
    assert output.samefile(pending_output)
    assert meta.samefile(pending_meta)
    assert output.stat().st_nlink == 2
    assert meta.stat().st_nlink == 2


def _prepare_pending_manifest_recovery(
    tmp_path: Path, *, bank_published: bool,
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    from scripts import chunk_trajectory_publication as publication

    bank_dir = tmp_path / "bank"
    manifest_dir = tmp_path / "manifest"
    bank_dir.mkdir()
    manifest_dir.mkdir()
    output = bank_dir / "bank.jsonl"
    meta = manifest_dir / "bank.meta.json"
    pending_output = publication._pending_output_path(output)
    pending_meta = publication._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
        "complete": True,
        "output": publication._prepared_output_artifact(pending_output, output),
    }
    if bank_published:
        publication._publish_output(pending_output, output)
    # Model an interrupted staged write whose close made bytes readable from
    # cache but whose file and directory durability barriers never completed.
    pending_meta.write_text(json.dumps(manifest))
    return output, meta, pending_output, pending_meta, manifest


@pytest.mark.parametrize("bank_published", [True, False], ids=["bank-final", "bank-pending"])
def test_recovery_syncs_pending_manifest_before_any_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, bank_published: bool,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, _pending_output, pending_meta, manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=bank_published)
    )
    pending_identity = (pending_meta.stat().st_dev, pending_meta.stat().st_ino)
    parent_identity = (
        pending_meta.parent.stat().st_dev,
        pending_meta.parent.stat().st_ino,
    )
    real_fsync = os.fsync
    real_link = publication._link_open_file
    events: list[str] = []
    pending_file_synced = False
    pending_parent_synced = False

    def fsync_spy(descriptor: int) -> None:
        nonlocal pending_file_synced, pending_parent_synced
        descriptor_stat = os.fstat(descriptor)
        identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
        if stat.S_ISREG(descriptor_stat.st_mode) and identity == pending_identity:
            events.append("fsync:pending-manifest")
            pending_file_synced = True
        elif (
            stat.S_ISDIR(descriptor_stat.st_mode)
            and identity == parent_identity
            and pending_file_synced
            and not pending_parent_synced
        ):
            events.append("fsync:pending-parent")
            pending_parent_synced = True
        real_fsync(descriptor)

    def link_spy(file_fd: int, parent_fd: int, name: str) -> None:
        events.append(f"link:{name}")
        real_link(file_fd, parent_fd, name)

    monkeypatch.setattr(publication.os, "fsync", fsync_spy)
    monkeypatch.setattr(publication, "_link_open_file", link_spy)

    assert publication._require_new_output_pair(output, meta, overwrite=False) is True

    first_destination = meta if bank_published else output
    assert events[:3] == [
        "fsync:pending-manifest",
        "fsync:pending-parent",
        f"link:{first_destination.name}",
    ]
    assert json.loads(meta.read_text()) == manifest
    assert output.samefile(_pending_output)
    assert meta.samefile(pending_meta)


def test_recovery_syncs_retained_manifest_parent_during_path_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, _pending_output, pending_meta, manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=True)
    )
    parent = pending_meta.parent
    moved_parent = tmp_path / "moved-manifest"
    parent_identity = (parent.stat().st_dev, parent.stat().st_ino)
    pending_identity = (pending_meta.stat().st_dev, pending_meta.stat().st_ino)
    real_fsync = os.fsync
    directory_syncs: list[tuple[int, int]] = []
    swapped = False

    def fsync_spy(descriptor: int) -> None:
        nonlocal swapped
        descriptor_stat = os.fstat(descriptor)
        identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
        if stat.S_ISREG(descriptor_stat.st_mode) and identity == pending_identity:
            real_fsync(descriptor)
            parent.rename(moved_parent)
            parent.mkdir()
            swapped = True
            return
        if stat.S_ISDIR(descriptor_stat.st_mode) and swapped:
            directory_syncs.append(identity)
            parent.rmdir()
            moved_parent.rename(parent)
            swapped = False
        real_fsync(descriptor)

    monkeypatch.setattr(publication.os, "fsync", fsync_spy)

    assert publication._require_new_output_pair(output, meta, overwrite=False) is True

    assert directory_syncs[0] == parent_identity
    assert json.loads(meta.read_text()) == manifest
    assert meta.samefile(pending_meta)


@pytest.mark.parametrize("bank_published", [True, False], ids=["bank-final", "bank-pending"])
@pytest.mark.parametrize("failure_target", ["file", "parent"])
def test_pending_manifest_sync_failure_prevents_recovery_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    bank_published: bool,
    failure_target: str,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, pending_output, pending_meta, manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=bank_published)
    )
    pending_identity = (pending_meta.stat().st_dev, pending_meta.stat().st_ino)
    parent_identity = (
        pending_meta.parent.stat().st_dev,
        pending_meta.parent.stat().st_ino,
    )
    real_fsync = os.fsync
    real_link = publication._link_open_file
    real_unlink = os.unlink
    published_links: list[str] = []
    removed_names: list[str] = []
    pending_file_synced = False

    def fsync_spy(descriptor: int) -> None:
        nonlocal pending_file_synced
        descriptor_stat = os.fstat(descriptor)
        identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
        if (
            stat.S_ISREG(descriptor_stat.st_mode)
            and identity == pending_identity
        ):
            if failure_target == "file":
                raise OSError(errno.EIO, "pending manifest fsync failed")
            pending_file_synced = True
        elif (
            failure_target == "parent"
            and pending_file_synced
            and stat.S_ISDIR(descriptor_stat.st_mode)
            and identity == parent_identity
        ):
            raise OSError(errno.EIO, "pending manifest parent fsync failed")
        real_fsync(descriptor)

    def link_spy(file_fd: int, parent_fd: int, name: str) -> None:
        published_links.append(name)
        real_link(file_fd, parent_fd, name)

    def unlink_spy(path: str, *args: Any, **kwargs: Any) -> None:
        removed_names.append(path)
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(publication.os, "fsync", fsync_spy)
    monkeypatch.setattr(publication, "_link_open_file", link_spy)
    monkeypatch.setattr(publication.os, "unlink", unlink_spy)
    expected_error = (
        "pending manifest fsync failed"
        if failure_target == "file"
        else "pending manifest parent fsync failed"
    )
    with pytest.raises(OSError, match=expected_error):
        publication._require_new_output_pair(output, meta, overwrite=False)

    assert published_links == []
    assert removed_names == []
    assert output.exists() is bank_published
    assert pending_output.exists()
    assert not meta.exists()
    assert json.loads(pending_meta.read_text()) == manifest
    assert not publication._invalid_manifest_path(meta).exists()

    monkeypatch.setattr(publication.os, "fsync", real_fsync)
    assert publication._require_new_output_pair(output, meta, overwrite=False) is True
    assert output.samefile(pending_output)
    assert meta.samefile(pending_meta)


@pytest.mark.parametrize("bank_published", [True, False], ids=["bank-final", "bank-pending"])
@pytest.mark.parametrize("mutation", ["content", "inode"])
def test_recovery_rejects_pending_manifest_changed_after_durability_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    bank_published: bool,
    mutation: str,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, _pending_output, pending_meta, _manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=bank_published)
    )
    real_read = publication._read_pending_manifest_fd
    real_link = publication._link_open_file
    published_links: list[str] = []

    def read_then_mutate(file_fd: int, path: Path) -> dict[str, Any]:
        payload = real_read(file_fd, path)
        if mutation == "content":
            path.write_text(json.dumps({**payload, "tampered": True}))
        else:
            replacement = path.with_name(f"{path.name}.replacement")
            replacement.write_bytes(path.read_bytes())
            os.replace(replacement, path)
        return payload

    def link_spy(file_fd: int, parent_fd: int, name: str) -> None:
        published_links.append(name)
        real_link(file_fd, parent_fd, name)

    monkeypatch.setattr(publication, "_read_pending_manifest_fd", read_then_mutate)
    monkeypatch.setattr(publication, "_link_open_file", link_spy)
    with pytest.raises(SystemExit, match=r"changed|anchored regular file"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    assert published_links == []
    assert output.exists() is bank_published
    assert not meta.exists()
    assert pending_meta.exists()


@pytest.mark.parametrize("bank_published", [True, False], ids=["bank-final", "bank-pending"])
def test_recovery_rejects_static_pending_manifest_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, bank_published: bool,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, pending_output, pending_meta, manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=bank_published)
    )
    target = pending_meta.with_name("attacker-controlled-manifest.json")
    pending_meta.unlink()
    target.write_text(json.dumps(manifest))
    pending_meta.symlink_to(target)
    target_bytes = target.read_bytes()
    target_links = target.stat().st_nlink
    publication_mutations: list[str] = []

    def forbidden_link(_file_fd: int, _parent_fd: int, name: str) -> None:
        publication_mutations.append(f"link:{name}")

    def forbidden_unlink(path: str, *_args: Any, **_kwargs: Any) -> None:
        publication_mutations.append(f"unlink:{path}")

    monkeypatch.setattr(publication, "_link_open_file", forbidden_link)
    monkeypatch.setattr(publication.os, "unlink", forbidden_unlink)
    with pytest.raises(SystemExit, match="cannot safely open regular evidence file"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    assert publication_mutations == []
    assert not meta.exists()
    assert pending_meta.is_symlink()
    assert target.read_bytes() == target_bytes
    assert target.stat().st_nlink == target_links == 1
    assert output.exists() is bank_published
    assert pending_output.exists()


@pytest.mark.parametrize("bank_published", [True, False], ids=["bank-final", "bank-pending"])
def test_recovery_rejects_pending_manifest_with_an_extra_hard_link(
    tmp_path: Path, *, bank_published: bool,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, pending_output, pending_meta, _manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=bank_published)
    )
    extra_manifest = pending_meta.with_name("extra-manifest-link.json")
    os.link(pending_meta, extra_manifest)

    with pytest.raises(SystemExit, match="unexpected hard links"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    assert not meta.exists()
    assert pending_meta.samefile(extra_manifest)
    assert pending_meta.stat().st_nlink == 2
    assert output.exists() is bank_published
    assert pending_output.exists()


@pytest.mark.parametrize("pending_name", [False, True], ids=["final-only", "final-pending"])
def test_recovery_rejects_bank_with_an_extra_hard_link(
    tmp_path: Path, *, pending_name: bool,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, pending_output, pending_meta, _manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=not pending_name)
    )
    if pending_name:
        os.link(pending_output, output)
    extra_bank = output.with_name("extra-bank-link.jsonl")
    os.link(output, extra_bank)

    with pytest.raises(SystemExit, match="unexpected hard links"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    assert not meta.exists()
    assert pending_meta.exists()
    assert output.samefile(extra_bank)
    assert pending_output.exists()
    assert output.stat().st_nlink == 3


@pytest.mark.parametrize("mutation", ["content", "hard-link"])
def test_final_bank_recovery_guards_bank_through_manifest_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, mutation: str,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, _pending_output, pending_meta, _manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=True)
    )
    extra_bank = output.with_name("extra-bank-link.jsonl")
    real_publish = publication._publish_no_replace

    def mutate_at_manifest_boundary(
        tmp: Path, destination: Path, **kwargs: Any,
    ) -> None:
        if tmp == pending_meta:
            if mutation == "content":
                output.write_text("mutated bank\n")
            else:
                os.link(output, extra_bank)
        real_publish(tmp, destination, **kwargs)

    monkeypatch.setattr(publication, "_publish_no_replace", mutate_at_manifest_boundary)
    with pytest.raises(SystemExit, match=r"changed|unexpected hard links"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    assert pending_meta.exists()
    assert not meta.exists()
    if mutation == "hard-link":
        assert output.samefile(extra_bank)


@pytest.mark.parametrize("mutation", ["content", "hard-link"])
def test_normal_pair_guards_bank_through_manifest_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, mutation: str,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = publication._pending_output_path(output)
    pending_meta = publication._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
        "complete": True,
        "output": publication._prepared_output_artifact(pending_output, output),
    }
    extra_bank = output.with_name("extra-bank-link.jsonl")
    real_publish = publication._publish_no_replace

    def mutate_at_manifest_boundary(
        tmp: Path, destination: Path, **kwargs: Any,
    ) -> None:
        if tmp == pending_meta:
            if mutation == "content":
                output.write_text("mutated bank\n")
            else:
                os.link(output, extra_bank)
        real_publish(tmp, destination, **kwargs)

    monkeypatch.setattr(publication, "_publish_no_replace", mutate_at_manifest_boundary)
    with pytest.raises(SystemExit, match=r"changed|unexpected hard links"):
        publication._publish_evidence_pair(
            pending_output, output, pending_meta, meta, manifest,
        )

    assert output.exists()
    assert pending_meta.exists()
    assert not meta.exists()
    if mutation == "hard-link":
        assert output.samefile(extra_bank)


def test_normal_pair_retains_requested_manifest_through_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = publication._pending_output_path(output)
    pending_meta = publication._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
        "complete": True,
        "output": publication._prepared_output_artifact(pending_output, output),
    }
    real_publish = publication._publish_no_replace

    def mutate_manifest_at_publication(
        tmp: Path, destination: Path, **kwargs: Any,
    ) -> None:
        if tmp == pending_meta:
            pending_meta.write_text(json.dumps({**manifest, "tampered": True}))
        real_publish(tmp, destination, **kwargs)

    monkeypatch.setattr(
        publication, "_publish_no_replace", mutate_manifest_at_publication,
    )
    with pytest.raises(SystemExit, match=r"staged evidence changed|changed during"):
        publication._publish_evidence_pair(
            pending_output, output, pending_meta, meta, manifest,
        )

    assert output.read_text() == "completed bank\n"
    assert json.loads(pending_meta.read_text()) == {**manifest, "tampered": True}
    assert not meta.exists()
    invalid = publication._invalid_manifest_path(meta)
    assert json.loads(invalid.read_text())["invalid"] is True

    monkeypatch.setattr(publication, "_publish_no_replace", real_publish)
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    assert not meta.exists()
    assert json.loads(pending_meta.read_text()) == {**manifest, "tampered": True}


def test_normal_pair_rejects_substituted_pending_bank_witness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = publication._pending_output_path(output)
    pending_meta = publication._pending_manifest_path(meta)
    hidden = tmp_path / "authentic-pending-bank"
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
        "complete": True,
        "output": publication._prepared_output_artifact(pending_output, output),
    }
    real_publish = publication._publish_no_replace

    def publish_then_substitute(
        source_path: Path, destination: Path, **kwargs: Any,
    ) -> None:
        real_publish(source_path, destination, **kwargs)
        if source_path == pending_output:
            pending_output.rename(hidden)
            pending_output.write_text("completed bank\n")

    monkeypatch.setattr(publication, "_publish_no_replace", publish_then_substitute)
    with pytest.raises(
        SystemExit, match=r"anchored regular file|unexpected hard links|not the same",
    ):
        publication._publish_evidence_pair(
            pending_output, output, pending_meta, meta, manifest,
        )

    assert hidden.samefile(output)
    assert pending_output.read_text() == "completed bank\n"
    assert not pending_output.samefile(output)
    assert publication._invalid_manifest_path(meta).exists()
    assert not meta.exists()


def test_normal_pair_rejects_substituted_pending_manifest_witness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = publication._pending_output_path(output)
    pending_meta = publication._pending_manifest_path(meta)
    hidden = tmp_path / "authentic-pending-manifest"
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
        "complete": True,
        "output": publication._prepared_output_artifact(pending_output, output),
    }
    real_publish = publication._publish_no_replace

    def publish_then_substitute(
        source_path: Path, destination: Path, **kwargs: Any,
    ) -> None:
        real_publish(source_path, destination, **kwargs)
        if source_path == pending_meta:
            pending_meta.rename(hidden)
            pending_meta.write_text(json.dumps(manifest))

    monkeypatch.setattr(publication, "_publish_no_replace", publish_then_substitute)
    with pytest.raises(
        SystemExit, match=r"anchored regular file|unexpected hard links|not the same",
    ):
        publication._publish_evidence_pair(
            pending_output, output, pending_meta, meta, manifest,
        )

    assert hidden.samefile(meta)
    assert not pending_meta.samefile(meta)
    assert publication._invalid_manifest_path(meta).exists()
    assert output.samefile(pending_output)


def test_post_sync_manifest_name_swap_is_quarantined_across_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = publication._pending_output_path(output)
    pending_meta = publication._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
        "complete": True,
        "output": publication._prepared_output_artifact(pending_output, output),
    }
    attacker_manifest = {**manifest, "tampered": True}
    real_fsync = publication.os.fsync
    manifest_synced = False
    injected = False

    def replace_manifest_then_fail_parent_sync(descriptor: int) -> None:
        nonlocal manifest_synced, injected
        descriptor_stat = os.fstat(descriptor)
        if stat.S_ISREG(descriptor_stat.st_mode) and pending_meta.exists():
            pending_stat = pending_meta.stat()
            if (descriptor_stat.st_dev, descriptor_stat.st_ino) == (
                pending_stat.st_dev,
                pending_stat.st_ino,
            ):
                real_fsync(descriptor)
                pending_meta.unlink()
                pending_meta.write_text(json.dumps(attacker_manifest))
                manifest_synced = True
                return
        if manifest_synced and stat.S_ISDIR(descriptor_stat.st_mode) and not injected:
            injected = True
            raise OSError(errno.EIO, "manifest parent sync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(
        publication.os, "fsync", replace_manifest_then_fail_parent_sync,
    )
    with pytest.raises(SystemExit, match="anchored regular file"):
        publication._publish_evidence_pair(
            pending_output, output, pending_meta, meta, manifest,
        )

    monkeypatch.setattr(publication.os, "fsync", real_fsync)
    assert injected is True
    assert publication._invalid_manifest_path(meta).exists()
    assert json.loads(pending_meta.read_text()) == attacker_manifest
    assert not meta.exists()
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert not meta.exists()


def test_uncertain_manifest_create_is_quarantined_across_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = publication._pending_output_path(output)
    pending_meta = publication._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
        "complete": True,
        "output": publication._prepared_output_artifact(pending_output, output),
    }
    attacker_manifest = {**manifest, "attacker": True}
    attacker_bytes = json.dumps(attacker_manifest).encode()
    real_open = publication.os.open

    def create_attacker_then_fail(
        path: Any, flags: int, *args: Any, **kwargs: Any,
    ) -> int:
        if path == pending_meta.name and flags & os.O_CREAT:
            descriptor = real_open(path, flags, *args, **kwargs)
            os.write(descriptor, attacker_bytes)
            os.close(descriptor)
            raise OSError(errno.EIO, "uncertain staged create")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(publication.os, "open", create_attacker_then_fail)
    with pytest.raises(SystemExit, match="uncertain result"):
        publication._publish_evidence_pair(
            pending_output, output, pending_meta, meta, manifest,
        )

    monkeypatch.setattr(publication.os, "open", real_open)
    assert json.loads(pending_meta.read_text()) == attacker_manifest
    assert publication._invalid_manifest_path(meta).exists()
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert not output.exists()
    assert not meta.exists()


@pytest.mark.parametrize("operation", ["create", "recovery"])
def test_first_manifest_name_authentication_drift_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    operation: str,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    if operation == "recovery":
        output, meta, pending_output, pending_meta, manifest = (
            _prepare_pending_manifest_recovery(tmp_path, bank_published=False)
        )
    else:
        output = tmp_path / "bank.jsonl"
        meta = tmp_path / "bank.jsonl.meta.json"
        pending_output = publication._pending_output_path(output)
        pending_meta = publication._pending_manifest_path(meta)
        pending_output.write_text("completed bank\n")
        manifest = {
            "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
            "complete": True,
            "output": publication._prepared_output_artifact(
                pending_output, output,
            ),
        }
    attacker_manifest = {**manifest, "attacker": "longer"}
    real_open = publication.os.open
    real_stat = publication.os.stat
    manifest_opened = False
    injected = False

    def track_manifest_open(
        path: Any, flags: int, *args: Any, **kwargs: Any,
    ) -> int:
        nonlocal manifest_opened
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == pending_meta.name and kwargs.get("dir_fd") is not None:
            manifest_opened = True
        return descriptor

    def mutate_on_first_name_authentication(
        path: Any, *args: Any, **kwargs: Any,
    ) -> os.stat_result:
        nonlocal injected
        if (
            manifest_opened
            and not injected
            and path == pending_meta.name
            and kwargs.get("dir_fd") is not None
        ):
            injected = True
            pending_meta.write_text(json.dumps(attacker_manifest))
            raise OSError(errno.EIO, "first manifest name authentication failed")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(publication.os, "open", track_manifest_open)
    monkeypatch.setattr(publication.os, "stat", mutate_on_first_name_authentication)
    def attempt() -> Any:
        if operation == "recovery":
            return publication._require_new_output_pair(
                output, meta, overwrite=False,
            )
        return publication._publish_evidence_pair(
            pending_output, output, pending_meta, meta, manifest,
        )
    with pytest.raises(SystemExit, match="changed during authentication"):
        attempt()

    monkeypatch.setattr(publication.os, "open", real_open)
    monkeypatch.setattr(publication.os, "stat", real_stat)
    assert injected is True
    assert json.loads(pending_meta.read_text()) == attacker_manifest
    assert publication._invalid_manifest_path(meta).exists()
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert not meta.exists()


@pytest.mark.parametrize("operation", ["create", "recovery"])
def test_first_manifest_fstat_drift_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    operation: str,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    if operation == "recovery":
        output, meta, pending_output, pending_meta, manifest = (
            _prepare_pending_manifest_recovery(tmp_path, bank_published=False)
        )
    else:
        output = tmp_path / "bank.jsonl"
        meta = tmp_path / "bank.jsonl.meta.json"
        pending_output = publication._pending_output_path(output)
        pending_meta = publication._pending_manifest_path(meta)
        pending_output.write_text("completed bank\n")
        manifest = {
            "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
            "complete": True,
            "output": publication._prepared_output_artifact(
                pending_output, output,
            ),
        }
    attacker_manifest = {**manifest, "attacker": "longer"}
    real_open = publication.os.open
    real_fstat = publication.os.fstat
    manifest_fd = -1
    injected = False

    def track_manifest_open(
        path: Any, flags: int, *args: Any, **kwargs: Any,
    ) -> int:
        nonlocal manifest_fd
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == pending_meta.name and kwargs.get("dir_fd") is not None:
            manifest_fd = descriptor
        return descriptor

    def mutate_on_first_descriptor_snapshot(descriptor: int) -> os.stat_result:
        nonlocal injected
        if descriptor == manifest_fd and not injected:
            injected = True
            pending_meta.write_text(json.dumps(attacker_manifest))
            raise OSError(errno.EIO, "first manifest descriptor snapshot failed")
        return real_fstat(descriptor)

    def attempt() -> Any:
        if operation == "recovery":
            return publication._require_new_output_pair(
                output, meta, overwrite=False,
            )
        return publication._publish_evidence_pair(
            pending_output, output, pending_meta, meta, manifest,
        )

    monkeypatch.setattr(publication.os, "open", track_manifest_open)
    monkeypatch.setattr(publication.os, "fstat", mutate_on_first_descriptor_snapshot)
    with pytest.raises(RuntimeError, match="snapshot newly opened evidence"):
        attempt()

    monkeypatch.setattr(publication.os, "open", real_open)
    monkeypatch.setattr(publication.os, "fstat", real_fstat)
    assert injected is True
    assert json.loads(pending_meta.read_text()) == attacker_manifest
    assert publication._invalid_manifest_path(meta).exists()
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert not meta.exists()


def test_manifest_artifact_first_fstat_drift_is_quarantined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, _pending_output, pending_meta, manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=False)
    )
    attacker_manifest = {**manifest, "attacker": "longer"}
    real_open = publication.os.open
    real_fstat = publication.os.fstat
    manifest_fd = -1
    manifest_fstats = 0
    injected = False

    def track_manifest_open(
        path: Any, flags: int, *args: Any, **kwargs: Any,
    ) -> int:
        nonlocal manifest_fd
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == pending_meta.name and kwargs.get("dir_fd") is not None:
            manifest_fd = descriptor
        return descriptor

    def mutate_on_artifact_snapshot(descriptor: int) -> os.stat_result:
        nonlocal manifest_fstats, injected
        if descriptor == manifest_fd:
            manifest_fstats += 1
            if manifest_fstats == 3:
                injected = True
                pending_meta.write_text(json.dumps(attacker_manifest))
                raise OSError(errno.EIO, "manifest artifact snapshot failed")
        return real_fstat(descriptor)

    monkeypatch.setattr(publication.os, "open", track_manifest_open)
    monkeypatch.setattr(publication.os, "fstat", mutate_on_artifact_snapshot)
    with pytest.raises(SystemExit, match="changed during failed snapshot"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    monkeypatch.setattr(publication.os, "open", real_open)
    monkeypatch.setattr(publication.os, "fstat", real_fstat)
    assert injected is True
    assert publication._invalid_manifest_path(meta).exists()
    assert json.loads(pending_meta.read_text()) == attacker_manifest
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert not meta.exists()


def test_successful_manifest_open_drift_is_quarantined_same_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, _pending_output, pending_meta, manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=False)
    )
    attacker_manifest = {**manifest, "attacker": "successful-open-rewrite"}
    real_open = publication.os.open
    real_fstat = publication.os.fstat
    manifest_fd = -1
    manifest_fstats = 0
    injected = False

    def track_manifest_open(
        path: Any, flags: int, *args: Any, **kwargs: Any,
    ) -> int:
        nonlocal manifest_fd
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == pending_meta.name and kwargs.get("dir_fd") is not None:
            manifest_fd = descriptor
        return descriptor

    def mutate_between_open_snapshots(descriptor: int) -> os.stat_result:
        nonlocal manifest_fstats, injected
        if descriptor == manifest_fd:
            manifest_fstats += 1
            if manifest_fstats == 2:
                injected = True
                pending_meta.write_text(json.dumps(attacker_manifest))
        return real_fstat(descriptor)

    monkeypatch.setattr(publication.os, "open", track_manifest_open)
    monkeypatch.setattr(publication.os, "fstat", mutate_between_open_snapshots)
    with pytest.raises(SystemExit, match="changed during authentication"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    monkeypatch.setattr(publication.os, "open", real_open)
    monkeypatch.setattr(publication.os, "fstat", real_fstat)
    assert injected is True
    assert publication._invalid_manifest_path(meta).exists()
    assert json.loads(pending_meta.read_text()) == attacker_manifest
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert not meta.exists()


def test_stable_manifest_snapshot_fstat_error_is_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, pending_output, pending_meta, manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=False)
    )
    real_artifact_from_fd = publication._artifact_from_fd
    real_fstat = publication.os.fstat
    snapshot_fd = -1
    injected = False

    def track_unbased_manifest_snapshot(
        file_fd: int,
        path: Path,
        *,
        before: os.stat_result | None = None,
    ) -> dict[str, Any]:
        nonlocal snapshot_fd
        if path == pending_meta and before is None and snapshot_fd < 0:
            snapshot_fd = file_fd
        return real_artifact_from_fd(file_fd, path, before=before)

    def fail_stable_snapshot_once(descriptor: int) -> os.stat_result:
        nonlocal injected
        if descriptor == snapshot_fd and not injected:
            injected = True
            raise OSError(errno.EIO, "stable manifest snapshot failed")
        return real_fstat(descriptor)

    monkeypatch.setattr(
        publication, "_artifact_from_fd", track_unbased_manifest_snapshot,
    )
    monkeypatch.setattr(publication.os, "fstat", fail_stable_snapshot_once)
    with pytest.raises(OSError, match="stable manifest snapshot failed"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    monkeypatch.setattr(publication, "_artifact_from_fd", real_artifact_from_fd)
    monkeypatch.setattr(publication.os, "fstat", real_fstat)
    assert injected is True
    assert not publication._invalid_manifest_path(meta).exists()
    assert pending_output.exists()
    assert json.loads(pending_meta.read_text()) == manifest
    assert publication._require_new_output_pair(output, meta, overwrite=False) is True
    assert output.samefile(pending_output)
    assert meta.samefile(pending_meta)


def test_recovery_output_snapshots_reuse_authenticated_stat_baselines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, pending_output, pending_meta, _manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=False)
    )
    real_artifact_from_fd = publication._artifact_from_fd
    output_snapshots = 0

    def require_output_snapshot_baseline(
        file_fd: int,
        path: Path,
        *,
        before: os.stat_result | None = None,
    ) -> dict[str, Any]:
        nonlocal output_snapshots
        if path == output:
            output_snapshots += 1
            assert before is not None
        return real_artifact_from_fd(file_fd, path, before=before)

    monkeypatch.setattr(
        publication, "_artifact_from_fd", require_output_snapshot_baseline,
    )
    assert publication._require_new_output_pair(output, meta, overwrite=False) is True
    assert output_snapshots > 0
    assert output.samefile(pending_output)
    assert meta.samefile(pending_meta)


def test_same_inode_staged_manifest_drift_is_quarantined_across_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = publication._pending_output_path(output)
    pending_meta = publication._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
        "complete": True,
        "output": publication._prepared_output_artifact(pending_output, output),
    }
    attacker_manifest = {**manifest, "attacker": "longer"}
    real_fsync = publication.os.fsync
    manifest_synced = False
    injected = False

    def mutate_manifest_then_fail_parent_sync(descriptor: int) -> None:
        nonlocal manifest_synced, injected
        descriptor_stat = os.fstat(descriptor)
        if stat.S_ISREG(descriptor_stat.st_mode) and pending_meta.exists():
            pending_stat = pending_meta.stat()
            if (descriptor_stat.st_dev, descriptor_stat.st_ino) == (
                pending_stat.st_dev,
                pending_stat.st_ino,
            ):
                real_fsync(descriptor)
                manifest_synced = True
                return
        if manifest_synced and not injected and stat.S_ISDIR(descriptor_stat.st_mode):
            injected = True
            pending_meta.write_text(json.dumps(attacker_manifest))
            raise OSError(errno.EIO, "staged parent sync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(
        publication.os, "fsync", mutate_manifest_then_fail_parent_sync,
    )
    with pytest.raises(SystemExit, match="differs from requested payload"):
        publication._publish_evidence_pair(
            pending_output, output, pending_meta, meta, manifest,
        )

    monkeypatch.setattr(publication.os, "fsync", real_fsync)
    assert json.loads(pending_meta.read_text()) == attacker_manifest
    assert publication._invalid_manifest_path(meta).exists()
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert not meta.exists()


def test_same_inode_recovery_manifest_drift_is_quarantined_across_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, _pending_output, pending_meta, manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=False)
    )
    attacker_manifest = {**manifest, "attacker": "longer"}
    real_fsync = publication.os.fsync
    manifest_synced = False
    injected = False

    def mutate_manifest_then_fail_parent_sync(descriptor: int) -> None:
        nonlocal manifest_synced, injected
        descriptor_stat = os.fstat(descriptor)
        if stat.S_ISREG(descriptor_stat.st_mode):
            pending_stat = pending_meta.stat()
            if (descriptor_stat.st_dev, descriptor_stat.st_ino) == (
                pending_stat.st_dev,
                pending_stat.st_ino,
            ):
                real_fsync(descriptor)
                manifest_synced = True
                return
        if manifest_synced and not injected and stat.S_ISDIR(descriptor_stat.st_mode):
            injected = True
            pending_meta.write_text(json.dumps(attacker_manifest))
            raise OSError(errno.EIO, "recovery parent sync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(
        publication.os, "fsync", mutate_manifest_then_fail_parent_sync,
    )
    with pytest.raises(SystemExit, match="changed during durability barrier"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    monkeypatch.setattr(publication.os, "fsync", real_fsync)
    assert json.loads(pending_meta.read_text()) == attacker_manifest
    assert publication._invalid_manifest_path(meta).exists()
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert not meta.exists()


@pytest.mark.parametrize("failing_read", [1, 7], ids=["initial-snapshot", "json-body"])
def test_manifest_read_failure_with_same_inode_drift_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    failing_read: int,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, _pending_output, pending_meta, manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=False)
    )
    attacker_manifest = {**manifest, "attacker": "longer"}
    real_pread = publication.os.pread
    reads = 0
    injected = False

    def mutate_then_fail(fd: int, size: int, offset: int) -> bytes:
        nonlocal reads, injected
        reads += 1
        if reads == failing_read:
            injected = True
            pending_meta.write_text(json.dumps(attacker_manifest))
            raise OSError(errno.EIO, "manifest read failed after mutation")
        return real_pread(fd, size, offset)

    monkeypatch.setattr(publication.os, "pread", mutate_then_fail)
    with pytest.raises(SystemExit, match="changed during"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    monkeypatch.setattr(publication.os, "pread", real_pread)
    assert injected is True
    assert publication._invalid_manifest_path(meta).exists()
    assert json.loads(pending_meta.read_text()) == attacker_manifest
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert not meta.exists()


def test_manifest_quarantine_uses_retained_parent_during_decoy_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    parent = tmp_path / "evidence"
    moved_parent = tmp_path / "moved-evidence"
    parent.mkdir()
    output = parent / "bank.jsonl"
    meta = parent / "bank.jsonl.meta.json"
    pending_output = publication._pending_output_path(output)
    pending_meta = publication._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
        "complete": True,
        "output": publication._prepared_output_artifact(pending_output, output),
    }
    real_publish = publication._publish_no_replace

    def install_parent_decoy(
        tmp: Path, destination: Path, **kwargs: Any,
    ) -> None:
        if tmp == pending_meta:
            parent.rename(moved_parent)
            parent.mkdir()
            pending_meta.write_text(json.dumps({**manifest, "tampered": True}))
        real_publish(tmp, destination, **kwargs)

    monkeypatch.setattr(publication, "_publish_no_replace", install_parent_decoy)
    with pytest.raises(
        SystemExit,
        match=r"containing directory changed|cannot revalidate evidence artifact",
    ):
        publication._publish_evidence_pair(
            pending_output, output, pending_meta, meta, manifest,
        )

    invalid_name = publication._invalid_manifest_path(meta).name
    assert (moved_parent / invalid_name).exists()
    for child in parent.iterdir():
        child.unlink()
    parent.rmdir()
    moved_parent.rename(parent)
    monkeypatch.setattr(publication, "_publish_no_replace", real_publish)

    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    assert not meta.exists()
    assert publication._pending_manifest_path(meta).exists()


def test_manifest_quarantine_rejects_decoy_only_success_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    parent = tmp_path / "evidence"
    moved_parent = tmp_path / "moved-evidence"
    parent.mkdir()
    output = parent / "bank.jsonl"
    meta = parent / "bank.jsonl.meta.json"
    pending_output = publication._pending_output_path(output)
    pending_meta = publication._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
        "complete": True,
        "output": publication._prepared_output_artifact(pending_output, output),
    }
    real_stage = publication._write_json_staged

    def stage_then_install_decoy(
        path: Path, payload: dict[str, Any], **kwargs: Any,
    ) -> None:
        real_stage(path, payload, **kwargs)
        if path == pending_meta:
            parent.rename(moved_parent)
            parent.mkdir()
            (moved_parent / pending_output.name).rename(parent / pending_output.name)
            (moved_parent / pending_meta.name).rename(parent / pending_meta.name)

    monkeypatch.setattr(publication, "_write_json_staged", stage_then_install_decoy)
    with pytest.raises(
        SystemExit,
        match=r"containing directory changed|cannot revalidate evidence artifact",
    ):
        publication._publish_evidence_pair(
            pending_output, output, pending_meta, meta, manifest,
        )

    invalid_name = publication._invalid_manifest_path(meta).name
    assert (moved_parent / invalid_name).exists()
    assert not output.exists()
    assert not meta.exists()
    for child in parent.iterdir():
        child.unlink()
    parent.rmdir()
    moved_parent.rename(parent)
    monkeypatch.setattr(publication, "_write_json_staged", real_stage)

    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    assert not output.exists()
    assert not meta.exists()


def test_final_manifest_recovery_rejects_an_extra_hard_link(tmp_path: Path) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, _pending_output, pending_meta, _manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=True)
    )
    os.link(pending_meta, meta)
    extra_manifest = tmp_path / "extra-manifest-link.json"
    os.link(meta, extra_manifest)

    with pytest.raises(SystemExit, match="unexpected hard links"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    assert pending_meta.samefile(meta)
    assert meta.samefile(extra_manifest)
    assert meta.stat().st_nlink == 3
    assert output.read_text() == "completed bank\n"


def test_producer_prepares_both_artifacts_before_pair_publication(tmp_path: Path) -> None:
    from scripts import backtest_chunk_trajectory as producer

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = producer._pending_output_path(output)
    pending_meta = producer._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": producer._SCHEMA,
        "complete": True,
        "output": producer._prepared_output_artifact(pending_output, output),
    }
    retained_output_fd = os.open(pending_output, os.O_RDONLY | os.O_CLOEXEC)
    retained_output_parent_fd = os.open(
        pending_output.parent,
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
    )
    producer._ACTIVE_PENDING_EVIDENCE = {
        "collection_complete": True,
        "pending_output": pending_output,
        "output": output,
        "manifest": meta,
        "pending_manifest": pending_meta,
        "output_artifact": manifest["output"],
        "retained_output_fd": retained_output_fd,
        "retained_output_parent_fd": retained_output_parent_fd,
    }
    try:
        producer._publish_collected_evidence_pair(
            pending_output, output, pending_meta, meta, manifest,
        )
        os.fstat(retained_output_fd)
        os.fstat(retained_output_parent_fd)
    finally:
        os.close(retained_output_fd)
        os.close(retained_output_parent_fd)
        producer._ACTIVE_PENDING_EVIDENCE = None

    assert output.read_text() == "completed bank\n"
    assert json.loads(meta.read_text()) == manifest
    assert output.samefile(pending_output)
    assert meta.samefile(pending_meta)
    assert output.stat().st_nlink == 2
    assert meta.stat().st_nlink == 2


def test_producer_retained_publication_handoff_rejects_parent_decoy(
    tmp_path: Path,
) -> None:
    from scripts import backtest_chunk_trajectory as producer
    from scripts import chunk_trajectory_publication as publication

    parent = tmp_path / "evidence"
    held_parent = tmp_path / "held-evidence"
    decoy_parent = tmp_path / "decoy-evidence"
    parent.mkdir()
    output = parent / "bank.jsonl"
    meta = parent / "bank.jsonl.meta.json"
    pending_output = producer._pending_output_path(output)
    pending_meta = producer._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    output_artifact = producer._prepared_output_artifact(pending_output, output)
    manifest = {
        "schema": producer._SCHEMA,
        "complete": True,
        "output": output_artifact,
    }
    retained_output_fd = os.open(pending_output, os.O_RDONLY | os.O_CLOEXEC)
    retained_output_parent_fd = os.open(
        parent,
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
    )
    producer._ACTIVE_PENDING_EVIDENCE = {
        "collection_complete": True,
        "pending_output": pending_output,
        "output": output,
        "manifest": meta,
        "pending_manifest": pending_meta,
        "output_artifact": output_artifact,
        "retained_output_fd": retained_output_fd,
        "retained_output_parent_fd": retained_output_parent_fd,
    }
    parent.rename(held_parent)
    parent.mkdir()
    (held_parent / pending_output.name).rename(parent / pending_output.name)
    try:
        with pytest.raises(SystemExit, match="containing directory changed"):
            producer._publish_collected_evidence_pair(
                pending_output, output, pending_meta, meta, manifest,
            )
        os.fstat(retained_output_fd)
        os.fstat(retained_output_parent_fd)
    finally:
        os.close(retained_output_fd)
        os.close(retained_output_parent_fd)
        producer._ACTIVE_PENDING_EVIDENCE = None

    parent.rename(decoy_parent)
    held_parent.rename(parent)
    assert publication._invalid_manifest_path(meta).exists()
    assert not output.exists()
    assert not meta.exists()
    assert (decoy_parent / pending_output.name).read_text() == "completed bank\n"
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)


@pytest.mark.parametrize("failed_link", ["bank", "manifest"])
def test_procfs_link_failure_keeps_pair_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    failed_link: str,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = publication._pending_output_path(output)
    pending_meta = publication._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
        "complete": True,
        "output": publication._prepared_output_artifact(pending_output, output),
    }
    failed_call = 1 if failed_link == "bank" else 2
    procfs_stats = 0
    real_stat = Path.stat

    def fail_one_procfs_stat(path: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        nonlocal procfs_stats
        if path.parent == Path("/proc/self/fd"):
            procfs_stats += 1
            if procfs_stats == failed_call:
                raise FileNotFoundError(errno.ENOENT, "transient procfs failure", path)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fail_one_procfs_stat)
    with pytest.raises(FileNotFoundError, match="transient procfs failure"):
        publication._publish_evidence_pair(
            pending_output, output, pending_meta, meta, manifest,
        )

    assert not publication._invalid_manifest_path(meta).exists()
    assert pending_meta.exists()
    assert output.exists() is (failed_link == "manifest")
    assert pending_output.exists()
    assert not meta.exists()

    monkeypatch.setattr(Path, "stat", real_stat)
    assert publication._require_new_output_pair(output, meta, overwrite=False) is True
    assert output.read_text() == "completed bank\n"
    assert json.loads(meta.read_text()) == manifest
    assert output.samefile(pending_output)
    assert meta.samefile(pending_meta)
    assert not publication._invalid_manifest_path(meta).exists()


def test_pending_bank_sync_failure_keeps_exact_pair_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = publication._pending_output_path(output)
    pending_meta = publication._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
        "complete": True,
        "output": publication._prepared_output_artifact(pending_output, output),
    }
    pending_identity = (pending_output.stat().st_dev, pending_output.stat().st_ino)
    real_fsync = publication.os.fsync
    bank_synced = False

    def fail_bank_parent_sync(descriptor: int) -> None:
        nonlocal bank_synced
        descriptor_stat = os.fstat(descriptor)
        identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
        if stat.S_ISREG(descriptor_stat.st_mode) and identity == pending_identity:
            real_fsync(descriptor)
            bank_synced = True
            return
        if bank_synced and stat.S_ISDIR(descriptor_stat.st_mode):
            raise OSError(errno.EIO, "pending bank parent sync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(publication.os, "fsync", fail_bank_parent_sync)
    with pytest.raises(OSError, match="pending bank parent sync failure"):
        publication._publish_evidence_pair(
            pending_output, output, pending_meta, meta, manifest,
        )

    monkeypatch.setattr(publication.os, "fsync", real_fsync)
    assert not publication._invalid_manifest_path(meta).exists()
    assert pending_output.read_text() == "completed bank\n"
    assert json.loads(pending_meta.read_text()) == manifest
    assert not output.exists()
    assert not meta.exists()
    assert publication._require_new_output_pair(output, meta, overwrite=False) is True
    assert output.samefile(pending_output)
    assert meta.samefile(pending_meta)


@pytest.mark.parametrize("failure", ["manifest-open", "parent-stat", "manifest-pread"])
def test_manifest_inspection_io_failure_keeps_pair_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure: str,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, pending_output, pending_meta, manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=False)
    )
    if failure == "manifest-open":
        real_open = publication.os.open
        injected = False

        def fail_manifest_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
            nonlocal injected
            if not injected and path == pending_meta.name and kwargs.get("dir_fd") is not None:
                injected = True
                raise OSError(errno.EIO, "transient manifest open failure")
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(publication.os, "open", fail_manifest_open)
    elif failure == "parent-stat":
        real_stat = Path.stat
        parent_stats = 0

        def fail_second_parent_stat(
            path: Path, *args: Any, **kwargs: Any,
        ) -> os.stat_result:
            nonlocal parent_stats
            if path == pending_meta.parent:
                parent_stats += 1
                if parent_stats == 2:
                    raise OSError(errno.EIO, "transient parent stat failure")
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", fail_second_parent_stat)
    else:
        real_pread = publication.os.pread
        manifest_reads = 0

        def fail_manifest_body_read(fd: int, size: int, offset: int) -> bytes:
            nonlocal manifest_reads
            manifest_reads += 1
            if manifest_reads == 7:
                raise OSError(errno.EIO, "transient manifest read failure")
            return real_pread(fd, size, offset)

        monkeypatch.setattr(publication.os, "pread", fail_manifest_body_read)

    with pytest.raises(OSError, match="transient"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    monkeypatch.undo()
    assert not publication._invalid_manifest_path(meta).exists()
    assert pending_output.read_text() == "completed bank\n"
    assert json.loads(pending_meta.read_text()) == manifest
    assert not output.exists()
    assert not meta.exists()

    assert publication._require_new_output_pair(output, meta, overwrite=False) is True
    assert output.read_text() == "completed bank\n"
    assert json.loads(meta.read_text()) == manifest


def test_final_manifest_name_stat_io_failure_keeps_pair_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, _pending_output, pending_meta, manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=True)
    )
    os.link(pending_meta, meta)
    real_stat = publication.os.stat
    injected = False

    def fail_pending_name_stat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        nonlocal injected
        if not injected and path == pending_meta.name and kwargs.get("dir_fd") is not None:
            injected = True
            raise OSError(errno.EIO, "transient pending-name stat failure")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(publication.os, "stat", fail_pending_name_stat)
    with pytest.raises(OSError, match="transient pending-name stat failure"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    monkeypatch.setattr(publication.os, "stat", real_stat)
    assert not publication._invalid_manifest_path(meta).exists()
    assert pending_meta.samefile(meta)
    assert publication._require_new_output_pair(output, meta, overwrite=False) is True
    assert json.loads(meta.read_text()) == manifest
    assert pending_meta.samefile(meta)


def test_disappeared_output_parent_during_acquisition_is_quarantined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, pending_output, pending_meta, manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=False)
    )
    output_parent = output.parent
    moved_parent = tmp_path / "moved-bank"
    real_open_parent = publication._open_parent
    injected = False

    def disappear_before_output_parent_open(
        path: Path, **kwargs: Any,
    ) -> int:
        nonlocal injected
        if not injected and path == output:
            injected = True
            output_parent.rename(moved_parent)
        return real_open_parent(path, **kwargs)

    monkeypatch.setattr(publication, "_open_parent", disappear_before_output_parent_open)
    with pytest.raises(SystemExit, match="cannot open containing directory"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    invalid = publication._invalid_manifest_path(meta)
    assert invalid.exists()
    moved_parent.rename(output_parent)
    attacker_manifest = {**manifest, "tampered": True}
    pending_meta.write_text(json.dumps(attacker_manifest))
    monkeypatch.setattr(publication, "_open_parent", real_open_parent)

    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    assert pending_output.exists()
    assert not output.exists()
    assert not meta.exists()
    assert json.loads(pending_meta.read_text()) == attacker_manifest


def test_manifest_parent_initial_authentication_decoy_is_quarantined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    parent = tmp_path / "evidence"
    held_parent = tmp_path / "held-evidence"
    decoy_parent = tmp_path / "decoy-evidence"
    parent.mkdir()
    decoy_parent.mkdir()
    output = parent / "bank.jsonl"
    meta = parent / "bank.jsonl.meta.json"

    def write_complete_pair(directory: Path, *, attacker: bool) -> None:
        pending_output = directory / publication._pending_output_path(output).name
        pending_meta = directory / publication._pending_manifest_path(meta).name
        final_output = directory / output.name
        final_meta = directory / meta.name
        pending_output.write_text("attacker bank\n" if attacker else "authentic bank\n")
        manifest = {
            "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
            "complete": True,
            "attacker": attacker,
            "output": publication._prepared_output_artifact(
                pending_output, output,
            ),
        }
        os.link(pending_output, final_output)
        pending_meta.write_text(json.dumps(manifest))
        os.link(pending_meta, final_meta)

    write_complete_pair(parent, attacker=False)
    write_complete_pair(decoy_parent, attacker=True)
    real_stat = Path.stat
    parent_stats = 0
    injected = False

    def install_decoy_on_initial_parent_authentication(
        path: Path, *args: Any, **kwargs: Any,
    ) -> os.stat_result:
        nonlocal parent_stats, injected
        if path == parent:
            parent_stats += 1
            if parent_stats == 2:
                parent.rename(held_parent)
                decoy_parent.rename(parent)
                injected = True
                raise OSError(errno.EIO, "initial parent authentication failed")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", install_decoy_on_initial_parent_authentication)
    with pytest.raises(SystemExit, match="containing directory changed"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    monkeypatch.setattr(Path, "stat", real_stat)
    assert injected is True
    invalid_name = publication._invalid_manifest_path(meta).name
    assert (held_parent / invalid_name).exists()
    assert (parent / invalid_name).exists()
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert json.loads((parent / meta.name).read_text())["attacker"] is True


def test_manifest_parent_first_fstat_decoy_is_quarantined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    parent = tmp_path / "evidence"
    held_parent = tmp_path / "held-evidence"
    decoy_parent = tmp_path / "decoy-evidence"
    parent.mkdir()
    decoy_parent.mkdir()
    output = parent / "bank.jsonl"
    meta = parent / "bank.jsonl.meta.json"

    def write_complete_pair(directory: Path, *, attacker: bool) -> None:
        pending_output = directory / publication._pending_output_path(output).name
        pending_meta = directory / publication._pending_manifest_path(meta).name
        pending_output.write_text("attacker bank\n" if attacker else "authentic bank\n")
        manifest = {
            "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
            "complete": True,
            "attacker": attacker,
            "output": publication._prepared_output_artifact(
                pending_output, output,
            ),
        }
        os.link(pending_output, directory / output.name)
        pending_meta.write_text(json.dumps(manifest))
        os.link(pending_meta, directory / meta.name)

    write_complete_pair(parent, attacker=False)
    write_complete_pair(decoy_parent, attacker=True)
    real_fstat = publication.os.fstat
    directory_fstats = 0
    injected = False

    def install_decoy_on_first_retained_parent_fstat(
        descriptor: int,
    ) -> os.stat_result:
        nonlocal directory_fstats, injected
        descriptor_stat = real_fstat(descriptor)
        if stat.S_ISDIR(descriptor_stat.st_mode):
            directory_fstats += 1
            if directory_fstats == 2:
                parent.rename(held_parent)
                decoy_parent.rename(parent)
                injected = True
                raise OSError(errno.EIO, "initial parent descriptor snapshot failed")
        return descriptor_stat

    monkeypatch.setattr(
        publication.os, "fstat", install_decoy_on_first_retained_parent_fstat,
    )
    with pytest.raises(RuntimeError, match="snapshot newly opened evidence parent"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    monkeypatch.setattr(publication.os, "fstat", real_fstat)
    assert injected is True
    invalid_name = publication._invalid_manifest_path(meta).name
    assert (held_parent / invalid_name).exists()
    assert (parent / invalid_name).exists()
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert json.loads((parent / meta.name).read_text())["attacker"] is True


def test_manifest_parent_fstat_failure_does_not_leak_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, _pending_output, _pending_meta, _manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=False)
    )
    real_fstat = publication.os.fstat

    def fail_directory_fstat(descriptor: int) -> os.stat_result:
        descriptor_stat = real_fstat(descriptor)
        if stat.S_ISDIR(descriptor_stat.st_mode):
            raise OSError(errno.EIO, "parent descriptor snapshot failed")
        return descriptor_stat

    before = len(os.listdir("/proc/self/fd"))
    monkeypatch.setattr(publication.os, "fstat", fail_directory_fstat)
    for _attempt in range(20):
        with pytest.raises((OSError, RuntimeError), match=r"descriptor|quarantine"):
            publication._require_new_output_pair(output, meta, overwrite=False)
    monkeypatch.setattr(publication.os, "fstat", real_fstat)
    after = len(os.listdir("/proc/self/fd"))

    assert after == before


def test_namespace_quarantine_attempts_current_alias_after_retained_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    parent = tmp_path / "evidence"
    held_parent = tmp_path / "held-evidence"
    decoy_parent = tmp_path / "decoy-evidence"
    parent.mkdir()
    decoy_parent.mkdir()
    output = parent / "bank.jsonl"
    meta = parent / "bank.jsonl.meta.json"
    pending_output = decoy_parent / publication._pending_output_path(output).name
    pending_meta = decoy_parent / publication._pending_manifest_path(meta).name
    pending_output.write_text("attacker bank\n")
    attacker_manifest = {
        "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
        "complete": True,
        "attacker": True,
        "output": publication._prepared_output_artifact(
            pending_output, output,
        ),
    }
    os.link(pending_output, decoy_parent / output.name)
    pending_meta.write_text(json.dumps(attacker_manifest))
    os.link(pending_meta, decoy_parent / meta.name)
    invalid = publication._invalid_manifest_path(meta)
    invalid.write_text("invalidated\n")
    retained_parent_fd = os.open(
        parent, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
    )
    parent.rename(held_parent)
    decoy_parent.rename(parent)
    real_stat = publication.os.stat
    injected = False

    def fail_retained_marker_stat_once(
        path: Any, *args: Any, **kwargs: Any,
    ) -> os.stat_result:
        nonlocal injected
        if (
            not injected
            and path == invalid.name
            and kwargs.get("dir_fd") == retained_parent_fd
        ):
            injected = True
            raise OSError(errno.EIO, "retained marker stat failed")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(publication.os, "stat", fail_retained_marker_stat_once)
    try:
        with pytest.raises(OSError, match="retained marker stat failed"):
            publication._mark_manifest_namespace_invalid(
                meta, parent_fd=retained_parent_fd,
            )
    finally:
        os.close(retained_parent_fd)

    monkeypatch.setattr(publication.os, "stat", real_stat)
    assert injected is True
    assert (held_parent / invalid.name).exists()
    assert (parent / invalid.name).exists()
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert json.loads((parent / meta.name).read_text())["attacker"] is True


def test_namespace_quarantine_retries_failed_marker_on_same_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, pending_output, pending_meta, manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=False)
    )
    pending_meta.write_text(json.dumps({**manifest, "attacker": True}))
    invalid = publication._invalid_manifest_path(meta)
    parent_fd = os.open(
        meta.parent,
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
    )
    real_open = publication.os.open
    failures = 0

    def fail_first_four_marker_creates(
        path: Any, flags: int, *args: Any, **kwargs: Any,
    ) -> int:
        nonlocal failures
        if path == invalid.name and flags & os.O_CREAT and failures < 4:
            failures += 1
            raise OSError(errno.EIO, "marker create failed")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(publication.os, "open", fail_first_four_marker_creates)
    try:
        with pytest.raises(RuntimeError, match="cannot retain invalid-recovery marker"):
            publication._mark_manifest_namespace_invalid(
                meta, parent_fd=parent_fd,
            )
    finally:
        os.close(parent_fd)

    monkeypatch.setattr(publication.os, "open", real_open)
    assert failures == 4
    assert invalid.exists()
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert pending_output.exists()
    assert not output.exists()


def test_namespace_quarantine_retries_current_alias_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    parent = tmp_path / "evidence"
    held_parent = tmp_path / "held-evidence"
    decoy_parent = tmp_path / "decoy-evidence"
    parent.mkdir()
    decoy_parent.mkdir()
    output = parent / "bank.jsonl"
    meta = parent / "bank.jsonl.meta.json"
    pending_output = decoy_parent / publication._pending_output_path(output).name
    pending_meta = decoy_parent / publication._pending_manifest_path(meta).name
    pending_output.write_text("attacker bank\n")
    attacker_manifest = {
        "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
        "complete": True,
        "attacker": True,
        "output": publication._prepared_output_artifact(
            pending_output, output,
        ),
    }
    os.link(pending_output, decoy_parent / output.name)
    pending_meta.write_text(json.dumps(attacker_manifest))
    os.link(pending_meta, decoy_parent / meta.name)
    retained_parent_fd = os.open(
        parent, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
    )
    parent.rename(held_parent)
    decoy_parent.rename(parent)
    real_stat = Path.stat
    injected = False

    def fail_current_alias_open_once(
        path: Path, *args: Any, **kwargs: Any,
    ) -> os.stat_result:
        nonlocal injected
        if not injected and path == parent:
            injected = True
            raise OSError(errno.EIO, "current alias open failed")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fail_current_alias_open_once)
    try:
        publication._mark_manifest_namespace_invalid(
            meta, parent_fd=retained_parent_fd,
        )
    finally:
        os.close(retained_parent_fd)

    monkeypatch.setattr(Path, "stat", real_stat)
    assert injected is True
    invalid_name = publication._invalid_manifest_path(meta).name
    assert (held_parent / invalid_name).exists()
    assert (parent / invalid_name).exists()
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert json.loads((parent / meta.name).read_text())["attacker"] is True


def test_namespace_quarantine_converges_after_second_parent_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    parent = tmp_path / "evidence"
    held_parent = tmp_path / "held-evidence"
    held_second = tmp_path / "held-second"
    decoys = [tmp_path / "decoy-one", tmp_path / "decoy-two"]
    parent.mkdir()
    for decoy in decoys:
        decoy.mkdir()
    output = parent / "bank.jsonl"
    meta = parent / "bank.jsonl.meta.json"

    def write_complete_pair(directory: Path, label: str) -> None:
        pending_output = directory / publication._pending_output_path(output).name
        pending_meta = directory / publication._pending_manifest_path(meta).name
        pending_output.write_text(f"{label} bank\n")
        manifest = {
            "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
            "complete": True,
            "attacker": label,
            "output": publication._prepared_output_artifact(
                pending_output, output,
            ),
        }
        os.link(pending_output, directory / output.name)
        pending_meta.write_text(json.dumps(manifest))
        os.link(pending_meta, directory / meta.name)

    write_complete_pair(decoys[0], "first")
    write_complete_pair(decoys[1], "second")
    retained_parent_fd = os.open(
        parent, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
    )
    parent.rename(held_parent)
    decoys[0].rename(parent)
    real_mark = publication._mark_manifest_recovery_invalid
    marks = 0
    swapped = False

    def swap_before_marking_first_alias(
        path: Path, *, parent_fd: int, diagnostic: dict[str, Any] | None = None,
    ) -> None:
        nonlocal marks, swapped
        marks += 1
        if marks == 2:
            parent.rename(held_second)
            decoys[1].rename(parent)
            swapped = True
        real_mark(path, parent_fd=parent_fd, diagnostic=diagnostic)

    monkeypatch.setattr(
        publication,
        "_mark_manifest_recovery_invalid",
        swap_before_marking_first_alias,
    )
    try:
        publication._mark_manifest_namespace_invalid(
            meta, parent_fd=retained_parent_fd,
        )
    finally:
        os.close(retained_parent_fd)

    monkeypatch.setattr(publication, "_mark_manifest_recovery_invalid", real_mark)
    invalid_name = publication._invalid_manifest_path(meta).name
    assert swapped is True
    assert (held_parent / invalid_name).exists()
    assert (held_second / invalid_name).exists()
    assert (parent / invalid_name).exists()
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert json.loads((parent / meta.name).read_text())["attacker"] == "second"


def test_output_parent_swap_at_recovery_exit_is_quarantined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, _pending_output, _pending_meta, manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=False)
    )
    output_parent = output.parent
    moved_parent = tmp_path / "moved-bank"
    real_recover = publication._require_new_output_pair_at_parent

    def recover_then_swap_output_parent(*args: Any, **kwargs: Any) -> bool:
        recovered = real_recover(*args, **kwargs)
        output_parent.rename(moved_parent)
        output_parent.mkdir()
        return recovered

    monkeypatch.setattr(
        publication, "_require_new_output_pair_at_parent", recover_then_swap_output_parent,
    )
    with pytest.raises(SystemExit, match="containing directory changed"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    assert publication._invalid_manifest_path(meta).exists()
    assert not output.exists()
    assert (moved_parent / output.name).read_text() == "completed bank\n"
    assert json.loads(meta.read_text()) == manifest
    output_parent.rmdir()
    moved_parent.rename(output_parent)
    monkeypatch.setattr(
        publication, "_require_new_output_pair_at_parent", real_recover,
    )

    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)


def test_invalid_marker_stat_io_failure_cannot_publish_pending_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, pending_output, pending_meta, manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=False)
    )
    attacker_manifest = {**manifest, "tampered": True}
    pending_meta.write_text(json.dumps(attacker_manifest))
    invalid = publication._invalid_manifest_path(meta)
    invalid.write_text("invalidated\n")
    real_stat = publication.os.stat
    injected = False

    def fail_marker_stat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        nonlocal injected
        if not injected and path == invalid.name and kwargs.get("dir_fd") is not None:
            injected = True
            invalid.unlink()
            raise OSError(errno.EIO, "invalid marker stat failure")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(publication.os, "stat", fail_marker_stat)
    with pytest.raises(OSError, match="invalid marker stat failure"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    monkeypatch.setattr(publication.os, "stat", real_stat)
    assert invalid.exists()
    assert pending_output.exists()
    assert not output.exists()
    assert not meta.exists()
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert json.loads(pending_meta.read_text()) == attacker_manifest


def test_invalid_marker_enoent_after_removal_is_restored_before_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, pending_output, pending_meta, manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=False)
    )
    attacker_manifest = {**manifest, "tampered": True}
    pending_meta.write_text(json.dumps(attacker_manifest))
    invalid = publication._invalid_manifest_path(meta)
    invalid.write_text("invalidated\n")
    real_stat = publication.os.stat
    injected = False

    def remove_marker_then_report_absent(
        path: Any, *args: Any, **kwargs: Any,
    ) -> os.stat_result:
        nonlocal injected
        if not injected and path == invalid.name and kwargs.get("dir_fd") is not None:
            injected = True
            invalid.unlink()
            raise FileNotFoundError(errno.ENOENT, "marker disappeared")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(publication.os, "stat", remove_marker_then_report_absent)
    with pytest.raises(SystemExit, match="marker absence changed"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    monkeypatch.setattr(publication.os, "stat", real_stat)
    assert injected is True
    assert invalid.exists()
    assert pending_output.exists()
    assert not output.exists()
    assert not meta.exists()
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert json.loads(pending_meta.read_text()) == attacker_manifest


def test_parent_open_fstat_error_detects_marker_namespace_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, pending_output, pending_meta, manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=False)
    )
    attacker_manifest = {**manifest, "tampered": True}
    pending_meta.write_text(json.dumps(attacker_manifest))
    invalid = publication._invalid_manifest_path(meta)
    invalid.write_text("invalidated\n")
    real_fstat = publication.os.fstat
    injected = False

    def remove_marker_on_first_parent_fstat(descriptor: int) -> os.stat_result:
        nonlocal injected
        descriptor_stat = real_fstat(descriptor)
        if (
            not injected
            and stat.S_ISDIR(descriptor_stat.st_mode)
            and invalid.exists()
        ):
            injected = True
            invalid.unlink()
            raise OSError(errno.EIO, "parent descriptor stat failed")
        return descriptor_stat

    monkeypatch.setattr(publication.os, "fstat", remove_marker_on_first_parent_fstat)
    with pytest.raises(SystemExit, match="containing directory changed while being opened"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    monkeypatch.setattr(publication.os, "fstat", real_fstat)
    assert injected is True
    assert invalid.exists()
    assert pending_output.exists()
    assert not output.exists()
    assert not meta.exists()
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert json.loads(pending_meta.read_text()) == attacker_manifest


def test_parent_open_success_detects_marker_namespace_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, pending_output, pending_meta, manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=False)
    )
    attacker_manifest = {**manifest, "tampered": True}
    pending_meta.write_text(json.dumps(attacker_manifest))
    invalid = publication._invalid_manifest_path(meta)
    invalid.write_text("invalidated\n")
    real_open = publication.os.open
    injected = False

    def remove_marker_after_parent_open(
        path: Any, flags: int, *args: Any, **kwargs: Any,
    ) -> int:
        nonlocal injected
        descriptor = real_open(path, flags, *args, **kwargs)
        if not injected and path == meta.parent and invalid.exists():
            injected = True
            invalid.unlink()
        return descriptor

    monkeypatch.setattr(publication.os, "open", remove_marker_after_parent_open)
    with pytest.raises(SystemExit, match="containing directory changed while being opened"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    monkeypatch.setattr(publication.os, "open", real_open)
    assert injected is True
    assert invalid.exists()
    assert pending_output.exists()
    assert not output.exists()
    assert not meta.exists()
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert json.loads(pending_meta.read_text()) == attacker_manifest


def test_parent_authentication_success_detects_marker_namespace_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, pending_output, pending_meta, manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=False)
    )
    attacker_manifest = {**manifest, "tampered": True}
    pending_meta.write_text(json.dumps(attacker_manifest))
    invalid = publication._invalid_manifest_path(meta)
    invalid.write_text("invalidated\n")
    real_fstat = publication.os.fstat
    parent_identity = (meta.parent.stat().st_dev, meta.parent.stat().st_ino)
    parent_fstats = 0
    injected = False

    def remove_marker_during_parent_authentication(
        descriptor: int,
    ) -> os.stat_result:
        nonlocal parent_fstats, injected
        descriptor_stat = real_fstat(descriptor)
        if (
            stat.S_ISDIR(descriptor_stat.st_mode)
            and (descriptor_stat.st_dev, descriptor_stat.st_ino) == parent_identity
        ):
            parent_fstats += 1
            if parent_fstats == 3 and invalid.exists():
                injected = True
                invalid.unlink()
        return descriptor_stat

    monkeypatch.setattr(
        publication.os, "fstat", remove_marker_during_parent_authentication,
    )
    with pytest.raises(SystemExit, match="newly opened evidence parent changed"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    monkeypatch.setattr(publication.os, "fstat", real_fstat)
    assert injected is True
    assert invalid.exists()
    assert pending_output.exists()
    assert not output.exists()
    assert not meta.exists()
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert json.loads(pending_meta.read_text()) == attacker_manifest


def test_invalid_marker_recreates_blocking_name_after_final_check_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, pending_output, pending_meta, manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=False)
    )
    pending_meta.write_text(json.dumps({**manifest, "tampered": True}))
    invalid = publication._invalid_manifest_path(meta)
    parent_fd = os.open(
        meta.parent,
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
    )
    real_fsync = publication.os.fsync
    removed = False

    def remove_marker_after_parent_sync(descriptor: int) -> None:
        nonlocal removed
        real_fsync(descriptor)
        if (
            not removed
            and stat.S_ISDIR(os.fstat(descriptor).st_mode)
            and invalid.exists()
        ):
            invalid.unlink()
            removed = True

    monkeypatch.setattr(publication.os, "fsync", remove_marker_after_parent_sync)
    try:
        with pytest.raises(SystemExit, match="revalidate evidence artifact"):
            publication._mark_manifest_recovery_invalid(meta, parent_fd=parent_fd)
    finally:
        os.close(parent_fd)

    monkeypatch.setattr(publication.os, "fsync", real_fsync)
    assert removed is True
    assert invalid.exists()
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert pending_output.exists()
    assert not output.exists()


def test_invalid_marker_repair_survives_remove_and_io_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, pending_output, pending_meta, manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=False)
    )
    pending_meta.write_text(json.dumps({**manifest, "tampered": True}))
    invalid = publication._invalid_manifest_path(meta)
    parent_fd = os.open(
        meta.parent,
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
    )
    real_fsync = publication.os.fsync
    real_stat = publication.os.stat
    marker_sync_failed = False
    repair_stat_failed = False

    def fail_initial_marker_sync(descriptor: int) -> None:
        nonlocal marker_sync_failed
        descriptor_stat = os.fstat(descriptor)
        if not marker_sync_failed and stat.S_ISREG(descriptor_stat.st_mode):
            marker_sync_failed = True
            real_fsync(descriptor)
            raise OSError(errno.EIO, "marker file sync failed")
        real_fsync(descriptor)

    def remove_marker_during_repair(
        path: Any, *args: Any, **kwargs: Any,
    ) -> os.stat_result:
        nonlocal repair_stat_failed
        if (
            marker_sync_failed
            and not repair_stat_failed
            and path == invalid.name
            and kwargs.get("dir_fd") == parent_fd
        ):
            repair_stat_failed = True
            invalid.unlink()
            raise OSError(errno.EIO, "marker repair stat failed")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(publication.os, "fsync", fail_initial_marker_sync)
    monkeypatch.setattr(publication.os, "stat", remove_marker_during_repair)
    try:
        with pytest.raises(OSError, match="marker file sync failed"):
            publication._mark_manifest_recovery_invalid(
                meta, parent_fd=parent_fd,
            )
    finally:
        os.close(parent_fd)

    monkeypatch.setattr(publication.os, "fsync", real_fsync)
    monkeypatch.setattr(publication.os, "stat", real_stat)
    assert marker_sync_failed is True
    assert repair_stat_failed is True
    assert invalid.exists()
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert pending_output.exists()
    assert not output.exists()


def test_parent_swap_during_state_classification_is_quarantined_across_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, pending_output, pending_meta, manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=False)
    )
    manifest_parent = meta.parent
    moved_parent = tmp_path / "moved-manifest"
    invalid = publication._invalid_manifest_path(meta)
    real_invalid_check = publication._manifest_recovery_is_invalid
    invalid_checks = 0

    def install_empty_parent_after_marker_check(
        path: Path, *, parent_fd: int,
    ) -> bool:
        nonlocal invalid_checks
        result = real_invalid_check(path, parent_fd=parent_fd)
        invalid_checks += 1
        if invalid_checks == 2:
            manifest_parent.rename(moved_parent)
            manifest_parent.mkdir()
        return result

    monkeypatch.setattr(
        publication,
        "_manifest_recovery_is_invalid",
        install_empty_parent_after_marker_check,
    )
    with pytest.raises(SystemExit, match="containing directory changed"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    assert (moved_parent / invalid.name).exists()
    assert (manifest_parent / invalid.name).exists()
    attacker_manifest = {**manifest, "tampered": True}
    (moved_parent / pending_meta.name).write_text(json.dumps(attacker_manifest))
    (manifest_parent / invalid.name).unlink()
    manifest_parent.rmdir()
    moved_parent.rename(manifest_parent)
    monkeypatch.setattr(
        publication, "_manifest_recovery_is_invalid", real_invalid_check,
    )

    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    assert pending_output.exists()
    assert not output.exists()
    assert not meta.exists()
    assert json.loads(pending_meta.read_text()) == attacker_manifest


@pytest.mark.parametrize("replacement", ["regular-file", "symlink-loop"])
def test_invalid_parent_type_during_state_classification_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    replacement: str,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    parent = tmp_path / "evidence"
    moved_parent = tmp_path / "moved-evidence"
    parent.mkdir()
    output = parent / "bank.jsonl"
    meta = parent / "bank.jsonl.meta.json"
    pending_output = publication._pending_output_path(output)
    pending_meta = publication._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
        "complete": True,
        "output": publication._prepared_output_artifact(pending_output, output),
    }
    pending_meta.write_text(json.dumps(manifest))
    invalid = publication._invalid_manifest_path(meta)
    real_invalid_check = publication._manifest_recovery_is_invalid
    invalid_checks = 0

    def replace_parent_after_marker_check(
        path: Path, *, parent_fd: int,
    ) -> bool:
        nonlocal invalid_checks
        result = real_invalid_check(path, parent_fd=parent_fd)
        invalid_checks += 1
        if invalid_checks == 2:
            parent.rename(moved_parent)
            if replacement == "regular-file":
                parent.write_text("not a directory\n")
            else:
                parent.symlink_to(parent.name)
        return result

    monkeypatch.setattr(
        publication,
        "_manifest_recovery_is_invalid",
        replace_parent_after_marker_check,
    )
    with pytest.raises(
        (SystemExit, RuntimeError),
        match=r"cannot revalidate|containing directory changed|cannot quarantine",
    ):
        publication._require_new_output_pair(output, meta, overwrite=False)

    assert (moved_parent / invalid.name).exists()
    attacker_manifest = {**manifest, "tampered": True}
    (moved_parent / pending_meta.name).write_text(json.dumps(attacker_manifest))
    parent.unlink()
    moved_parent.rename(parent)
    monkeypatch.setattr(
        publication, "_manifest_recovery_is_invalid", real_invalid_check,
    )

    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    assert pending_output.exists()
    assert not output.exists()
    assert not meta.exists()
    assert json.loads(pending_meta.read_text()) == attacker_manifest


def test_parent_aba_during_state_inventory_cannot_create_a_retry_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    parent = tmp_path / "evidence"
    moved_parent = tmp_path / "moved-evidence"
    parent.mkdir()
    output = parent / "bank.jsonl"
    meta = parent / "bank.jsonl.meta.json"
    pending_output = publication._pending_output_path(output)
    pending_meta = publication._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
        "complete": True,
        "output": publication._prepared_output_artifact(pending_output, output),
    }
    pending_meta.write_text(json.dumps(manifest))
    real_invalid_check = publication._manifest_recovery_is_invalid
    real_name_exists = publication._entry_name_exists
    invalid_checks = 0
    inventory_checks = 0
    inventory_parent_fds: set[int] = set()

    def install_empty_parent_after_marker_check(
        path: Path, *, parent_fd: int,
    ) -> bool:
        nonlocal invalid_checks
        result = real_invalid_check(path, parent_fd=parent_fd)
        invalid_checks += 1
        if invalid_checks == 2:
            parent.rename(moved_parent)
            parent.mkdir()
        return result

    def restore_parent_after_inventory(path: Path, *, parent_fd: int) -> bool:
        nonlocal inventory_checks
        inventory_parent_fds.add(parent_fd)
        result = real_name_exists(path, parent_fd=parent_fd)
        inventory_checks += 1
        if inventory_checks == 4:
            parent.rmdir()
            moved_parent.rename(parent)
        return result

    monkeypatch.setattr(
        publication,
        "_manifest_recovery_is_invalid",
        install_empty_parent_after_marker_check,
    )
    monkeypatch.setattr(publication, "_entry_name_exists", restore_parent_after_inventory)
    first_result = publication._require_new_output_pair(output, meta, overwrite=False)
    if not first_result:
        pending_meta.write_text(json.dumps({**manifest, "tampered": True}))
        publication._require_new_output_pair(output, meta, overwrite=False)

    assert first_result is True
    assert len(inventory_parent_fds) == 1
    assert json.loads(meta.read_text()) == manifest
    assert output.samefile(pending_output)
    assert meta.samefile(pending_meta)
    assert not publication._invalid_manifest_path(meta).exists()


def test_complete_pair_never_reopens_parent_during_nested_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    live = tmp_path / "live"
    decoy = tmp_path / "decoy"
    moved = tmp_path / "moved-live"
    live.mkdir()
    decoy.mkdir()

    def prepare(parent: Path) -> tuple[Path, Path]:
        output = parent / "bank.jsonl"
        meta = parent / "bank.meta.json"
        pending_output = publication._pending_output_path(output)
        pending_meta = publication._pending_manifest_path(meta)
        pending_output.write_text("valid bank\n")
        manifest = {
            "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
            "complete": True,
            "output": publication._prepared_output_artifact(pending_output, output),
        }
        publication._publish_evidence_pair(
            pending_output, output, pending_meta, meta, manifest,
        )
        return output, meta

    output, meta = prepare(live)
    decoy_output, decoy_meta = prepare(decoy)
    decoy_manifest = json.loads(decoy_meta.read_text())
    decoy_manifest["output"]["path"] = str(output.resolve())
    decoy_meta.write_text(json.dumps(decoy_manifest))
    assert decoy_output.samefile(publication._pending_output_path(decoy_output))
    output.write_text("attacker bank\n")
    real_open_parent = publication._open_parent
    real_read_manifest = publication._read_pending_manifest_fd
    open_calls = 0
    manifest_reads = 0
    swapped = False

    def swap_on_nested_parent_open(path: Path, **kwargs: Any) -> int:
        nonlocal open_calls, swapped
        open_calls += 1
        if open_calls == 2:
            live.rename(moved)
            decoy.rename(live)
            swapped = True
        return real_open_parent(path, **kwargs)

    def restore_before_final_manifest_read(
        file_fd: int, path: Path,
    ) -> dict[str, Any]:
        nonlocal manifest_reads, swapped
        manifest_reads += 1
        if manifest_reads == 2 and swapped:
            live.rename(decoy)
            moved.rename(live)
            swapped = False
        return real_read_manifest(file_fd, path)

    monkeypatch.setattr(publication, "_open_parent", swap_on_nested_parent_open)
    monkeypatch.setattr(
        publication, "_read_pending_manifest_fd", restore_before_final_manifest_read,
    )
    with pytest.raises(SystemExit, match="pending trajectory bank does not match"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    # Validation uses the retained parent; the later opens are the fail-closed
    # quarantine convergence check after the bank mismatch is detected.
    assert open_calls >= 2
    assert swapped is True
    invalid_name = publication._invalid_manifest_path(meta).name
    assert (moved / invalid_name).exists()
    assert (live / invalid_name).exists()
    live.rename(decoy)
    moved.rename(live)
    assert publication._invalid_manifest_path(meta).exists()
    assert output.read_text() == "attacker bank\n"


def test_state_presence_io_failure_keeps_pending_pair_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, pending_output, pending_meta, manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=False)
    )
    real_stat = publication.os.stat
    injected = False

    def fail_output_presence_stat(
        path: Any, *args: Any, **kwargs: Any,
    ) -> os.stat_result:
        nonlocal injected
        if (
            not injected
            and path == output.name
            and kwargs.get("follow_symlinks") is False
            and kwargs.get("dir_fd") is not None
        ):
            injected = True
            raise OSError(errno.EIO, "state presence stat failure")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(publication.os, "stat", fail_output_presence_stat)
    with pytest.raises(OSError, match="state presence stat failure"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    monkeypatch.setattr(publication.os, "stat", real_stat)
    assert not publication._invalid_manifest_path(meta).exists()
    assert pending_output.read_text() == "completed bank\n"
    assert json.loads(pending_meta.read_text()) == manifest
    assert not output.exists()
    assert not meta.exists()

    assert publication._require_new_output_pair(output, meta, overwrite=False) is True
    assert output.read_text() == "completed bank\n"
    assert json.loads(meta.read_text()) == manifest


def test_completed_pending_bank_is_synced_before_its_identity_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    pending = tmp_path / ".bank.jsonl.tmp-pending"
    output = tmp_path / "bank.jsonl"
    events: list[str] = []
    real_fsync = os.fsync
    real_artifact = publication._artifact_from_fd

    def fsync_spy(descriptor: int) -> None:
        descriptor_stat = os.fstat(descriptor)
        events.append("fsync:directory" if stat.S_ISDIR(descriptor_stat.st_mode) else "fsync:file")
        real_fsync(descriptor)

    def artifact_spy(
        file_fd: int,
        path: Path,
        *,
        before: os.stat_result | None = None,
    ) -> dict[str, Any]:
        events.append("snapshot")
        return real_artifact(file_fd, path, before=before)

    class FlushSpy:
        def __init__(self, handle: Any) -> None:
            self.handle = handle

        def flush(self) -> None:
            events.append("flush")
            self.handle.flush()

        def fileno(self) -> int:
            return int(self.handle.fileno())

    monkeypatch.setattr(publication.os, "fsync", fsync_spy)
    monkeypatch.setattr(publication, "_artifact_from_fd", artifact_spy)
    with pending.open("w") as handle:
        handle.write("completed bank\n")
        artifact = publication._durably_prepare_output_artifact(
            FlushSpy(handle), pending, output,
        )

    assert events == ["flush", "fsync:file", "fsync:directory", "snapshot"]
    assert artifact["sha256"] == hashlib.sha256(b"completed bank\n").hexdigest()


def test_completed_bank_reader_baseline_rejects_same_inode_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    pending = tmp_path / ".bank.jsonl.tmp-pending"
    output = tmp_path / "bank.jsonl"
    real_open = publication.os.open
    real_fstat = publication.os.fstat
    reader_fd = -1
    writer_fd = -1
    injected = False

    def track_reader_open(
        path: Any, flags: int, *args: Any, **kwargs: Any,
    ) -> int:
        nonlocal reader_fd
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == pending.name and kwargs.get("dir_fd") is not None:
            reader_fd = descriptor
        return descriptor

    def rewrite_before_writer_comparison(descriptor: int) -> os.stat_result:
        nonlocal injected
        if descriptor == writer_fd and reader_fd >= 0 and not injected:
            injected = True
            pending.write_text("attacker observations are longer\n")
        return real_fstat(descriptor)

    monkeypatch.setattr(publication.os, "open", track_reader_open)
    monkeypatch.setattr(publication.os, "fstat", rewrite_before_writer_comparison)
    with pending.open("w") as handle:
        handle.write("completed bank\n")
        writer_fd = handle.fileno()
        with pytest.raises(SystemExit, match="pending trajectory bank changed"):
            publication._durably_prepare_output_artifact(handle, pending, output)

    assert injected is True
    assert pending.read_text() == "attacker observations are longer\n"


def test_completed_bank_reader_open_rejects_same_inode_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    pending = tmp_path / ".bank.jsonl.tmp-pending"
    output = tmp_path / "bank.jsonl"
    real_open = publication.os.open
    real_stat = publication.os.stat
    reader_opened = False
    injected = False

    def track_reader_open(
        path: Any, flags: int, *args: Any, **kwargs: Any,
    ) -> int:
        nonlocal reader_opened
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == pending.name and kwargs.get("dir_fd") is not None:
            reader_opened = True
        return descriptor

    def rewrite_on_first_reader_name_check(
        path: Any, *args: Any, **kwargs: Any,
    ) -> os.stat_result:
        nonlocal injected
        if (
            reader_opened
            and not injected
            and path == pending.name
            and kwargs.get("dir_fd") is not None
        ):
            injected = True
            pending.write_text("attacker observations are longer\n")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(publication.os, "open", track_reader_open)
    monkeypatch.setattr(publication.os, "stat", rewrite_on_first_reader_name_check)
    with pending.open("w") as handle:
        handle.write("completed bank\n")
        with pytest.raises(SystemExit, match="changed during authentication"):
            publication._durably_prepare_output_artifact(handle, pending, output)

    assert injected is True
    assert pending.read_text() == "attacker observations are longer\n"


def test_completed_bank_fsync_rejects_same_inode_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    pending = tmp_path / ".bank.jsonl.tmp-pending"
    output = tmp_path / "bank.jsonl"
    real_fsync = publication.os.fsync
    writer_fd = -1
    injected = False

    def rewrite_after_writer_fsync(descriptor: int) -> None:
        nonlocal injected
        real_fsync(descriptor)
        if descriptor == writer_fd and not injected:
            injected = True
            pending.write_text("attacker observations are longer\n")

    monkeypatch.setattr(publication.os, "fsync", rewrite_after_writer_fsync)
    with pending.open("w") as handle:
        handle.write("completed bank\n")
        writer_fd = handle.fileno()
        with pytest.raises(SystemExit, match="changed during durability barrier"):
            publication._durably_prepare_output_artifact(handle, pending, output)

    assert injected is True
    assert pending.read_text() == "attacker observations are longer\n"


def test_completed_bank_flush_is_bound_to_intended_bytes(
    tmp_path: Path,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    pending = tmp_path / ".bank.jsonl.tmp-pending"
    output = tmp_path / "bank.jsonl"
    intended_bytes = b"completed bank\n"
    attacker_bytes = b"attacker observations are longer\n"
    injected = False

    class RewriteAfterFlush:
        def __init__(self, handle: Any) -> None:
            self.handle = handle

        def flush(self) -> None:
            nonlocal injected
            self.handle.flush()
            if not injected:
                injected = True
                pending.write_bytes(attacker_bytes)

        def fileno(self) -> int:
            return int(self.handle.fileno())

    with pending.open("w", encoding="utf-8", newline="") as handle:
        handle.write(intended_bytes.decode("utf-8"))
        with pytest.raises(SystemExit, match="differs from intended bytes"):
            publication._durably_prepare_output_artifact(
                RewriteAfterFlush(handle),
                pending,
                output,
                expected_size=len(intended_bytes),
                expected_sha256=hashlib.sha256(intended_bytes).hexdigest(),
            )

    assert injected is True
    assert pending.read_bytes() == attacker_bytes


def test_pending_bank_sync_uses_parent_retained_from_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    parent = tmp_path / "bank"
    moved_parent = tmp_path / "moved-bank"
    parent.mkdir()
    pending = parent / ".bank.jsonl.tmp-pending"
    output = parent / "bank.jsonl"
    real_fsync = os.fsync
    directory_syncs: list[tuple[int, int]] = []
    swapped = False

    with publication._open_staged_output_file(pending) as (handle, parent_fd):
        handle.write("completed bank\n")
        parent_stat = os.fstat(parent_fd)
        pending_stat = os.fstat(handle.fileno())
        parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
        pending_identity = (pending_stat.st_dev, pending_stat.st_ino)

        def fsync_spy(descriptor: int) -> None:
            nonlocal swapped
            descriptor_stat = os.fstat(descriptor)
            identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
            if stat.S_ISREG(descriptor_stat.st_mode) and identity == pending_identity:
                real_fsync(descriptor)
                parent.rename(moved_parent)
                parent.mkdir()
                swapped = True
                return
            if stat.S_ISDIR(descriptor_stat.st_mode) and swapped:
                directory_syncs.append(identity)
                parent.rmdir()
                moved_parent.rename(parent)
                swapped = False
            real_fsync(descriptor)

        monkeypatch.setattr(publication.os, "fsync", fsync_spy)
        artifact = publication._durably_prepare_output_artifact(
            handle, pending, output, parent_fd=parent_fd,
        )

    assert directory_syncs == [parent_identity]
    assert artifact["sha256"] == hashlib.sha256(b"completed bank\n").hexdigest()
    assert pending.read_text() == "completed bank\n"


def test_collection_error_anchor_rejects_renamed_parent_decoy(tmp_path: Path) -> None:
    from scripts import chunk_trajectory_publication as publication

    parent = tmp_path / "bank"
    moved_parent = tmp_path / "moved-bank"
    parent.mkdir()
    pending = parent / ".bank.jsonl.tmp-pending"
    output = parent / "bank.jsonl"
    retained_file_fd = -1
    retained_parent_fd = -1
    try:
        with publication._open_staged_output_file(pending) as (handle, parent_fd):
            handle.write("original observations\n")
            retained_file_fd = os.dup(handle.fileno())
            retained_parent_fd = os.dup(parent_fd)
        parent.rename(moved_parent)
        parent.mkdir()
        decoy = parent / pending.name
        decoy.write_text("decoy observations\n")

        with pytest.raises(SystemExit, match="containing directory changed"):
            publication._durably_prepare_anchored_output_artifact(
                retained_file_fd, retained_parent_fd, pending, output,
            )

        assert decoy.read_text() == "decoy observations\n"
        assert (moved_parent / pending.name).read_text() == "original observations\n"
    finally:
        if retained_file_fd >= 0:
            os.close(retained_file_fd)
        if retained_parent_fd >= 0:
            os.close(retained_parent_fd)


def test_collection_failure_diagnostic_stays_with_retained_bank_parent(
    tmp_path: Path,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    parent = tmp_path / "bank"
    moved_parent = tmp_path / "moved-bank"
    parent.mkdir()
    pending = parent / ".bank.jsonl.tmp-pending"
    output = parent / "bank.jsonl"
    meta = parent / "bank.jsonl.meta.json"
    pending.write_text("partial observations\n")
    file_fd = os.open(pending, os.O_RDONLY | os.O_CLOEXEC)
    parent_fd = os.open(
        parent, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        parent.rename(moved_parent)
        parent.mkdir()
        publication._write_invalid_recovery_diagnostic(
            pending,
            output,
            meta,
            {"failure_stage": "trajectory_collection"},
            file_fd=file_fd,
            parent_fd=parent_fd,
        )
    finally:
        os.close(file_fd)
        os.close(parent_fd)

    invalid = publication._invalid_manifest_path(meta)
    assert not invalid.exists()
    retained_invalid = moved_parent / invalid.name
    diagnostic = json.loads(retained_invalid.read_text())["diagnostic"]
    assert diagnostic["failure_stage"] == "trajectory_collection"
    assert (moved_parent / pending.name).read_text() == "partial observations\n"
    assert list(parent.iterdir()) == []


def test_collection_failure_manifest_substitution_cannot_seed_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    pending = tmp_path / ".bank.jsonl.tmp-pending"
    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_meta = publication._pending_manifest_path(meta)
    invalid = publication._invalid_manifest_path(meta)
    pending.write_text("partial observations\n")
    output_artifact = publication._prepared_output_artifact(pending, output)
    attacker_manifest = {
        "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
        "complete": True,
        "tampered": True,
        "output": output_artifact,
    }
    file_fd = os.open(pending, os.O_RDONLY | os.O_CLOEXEC)
    parent_fd = os.open(
        tmp_path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
    )
    real_fsync = publication.os.fsync
    replaced = False

    def replace_diagnostic_after_file_sync(descriptor: int) -> None:
        nonlocal replaced
        descriptor_stat = os.fstat(descriptor)
        if (
            not replaced
            and stat.S_ISREG(descriptor_stat.st_mode)
            and invalid.exists()
            and (descriptor_stat.st_dev, descriptor_stat.st_ino)
            == (invalid.stat().st_dev, invalid.stat().st_ino)
        ):
            real_fsync(descriptor)
            invalid.unlink()
            invalid.write_text(json.dumps(attacker_manifest))
            replaced = True
            return
        real_fsync(descriptor)

    monkeypatch.setattr(publication.os, "fsync", replace_diagnostic_after_file_sync)
    try:
        with pytest.raises(SystemExit, match="anchored regular file"):
            publication._write_invalid_recovery_diagnostic(
                pending,
                output,
                meta,
                {"failure_stage": "trajectory_collection"},
                file_fd=file_fd,
                parent_fd=parent_fd,
            )
    finally:
        os.close(file_fd)
        os.close(parent_fd)

    monkeypatch.setattr(publication.os, "fsync", real_fsync)
    assert replaced is True
    assert not pending_meta.exists()
    assert json.loads(invalid.read_text()) == attacker_manifest
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert not output.exists()


def test_collection_failure_stat_error_cannot_seed_pending_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    pending = tmp_path / ".bank.jsonl.tmp-pending"
    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_meta = publication._pending_manifest_path(meta)
    pending.write_text("partial observations\n")
    output_artifact = publication._prepared_output_artifact(pending, output)
    attacker_manifest = {
        "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
        "complete": True,
        "attacker": True,
        "output": output_artifact,
    }
    file_fd = os.open(pending, os.O_RDONLY | os.O_CLOEXEC)
    parent_fd = os.open(
        tmp_path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
    )
    real_stat = publication.os.stat
    injected = False

    def create_manifest_then_fail_bank_stat(
        path: Any, *args: Any, **kwargs: Any,
    ) -> os.stat_result:
        nonlocal injected
        if (
            not injected
            and path == pending.name
            and kwargs.get("dir_fd") == parent_fd
        ):
            injected = True
            pending_meta.write_text(json.dumps(attacker_manifest))
            raise OSError(errno.EIO, "collection diagnostic bank stat failed")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(publication.os, "stat", create_manifest_then_fail_bank_stat)
    try:
        with pytest.raises(OSError, match="collection diagnostic bank stat failed"):
            publication._write_invalid_recovery_diagnostic(
                pending,
                output,
                meta,
                {"failure_stage": "trajectory_collection"},
                file_fd=file_fd,
                parent_fd=parent_fd,
            )
    finally:
        os.close(file_fd)
        os.close(parent_fd)

    monkeypatch.setattr(publication.os, "stat", real_stat)
    assert injected is True
    assert publication._invalid_manifest_path(meta).exists()
    assert json.loads(pending_meta.read_text()) == attacker_manifest
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert not output.exists()


def test_collection_failure_bank_substitution_invalidates_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    pending = tmp_path / ".bank.jsonl.tmp-pending"
    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending.write_text("partial observations\n")
    file_fd = os.open(pending, os.O_RDONLY | os.O_CLOEXEC)
    parent_fd = os.open(
        tmp_path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
    )
    retained_identity = (os.fstat(file_fd).st_dev, os.fstat(file_fd).st_ino)
    real_fsync = publication.os.fsync
    replaced = False

    def replace_bank_after_file_sync(descriptor: int) -> None:
        nonlocal replaced
        descriptor_stat = os.fstat(descriptor)
        if (
            not replaced
            and (descriptor_stat.st_dev, descriptor_stat.st_ino) == retained_identity
        ):
            real_fsync(descriptor)
            pending.unlink()
            pending.write_text("attacker observations\n")
            replaced = True
            return
        real_fsync(descriptor)

    monkeypatch.setattr(publication.os, "fsync", replace_bank_after_file_sync)
    try:
        with pytest.raises(SystemExit, match="anchored regular file"):
            publication._write_invalid_recovery_diagnostic(
                pending,
                output,
                meta,
                {"failure_stage": "trajectory_collection"},
                file_fd=file_fd,
                parent_fd=parent_fd,
            )
    finally:
        os.close(file_fd)
        os.close(parent_fd)

    assert replaced is True
    assert publication._invalid_manifest_path(meta).exists()
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert pending.read_text() == "attacker observations\n"


def test_completed_pending_bank_fsync_failure_prevents_identity_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    pending = tmp_path / ".bank.jsonl.tmp-pending"
    output = tmp_path / "bank.jsonl"
    snapshot_called = False

    def fail_fsync(_descriptor: int) -> None:
        raise OSError(errno.EIO, "bank fsync failed")

    def forbidden_snapshot(
        _file_fd: int, _path: Path, **_kwargs: Any,
    ) -> dict[str, Any]:
        nonlocal snapshot_called
        snapshot_called = True
        raise AssertionError("unsynced bank reached identity snapshot")

    monkeypatch.setattr(publication.os, "fsync", fail_fsync)
    monkeypatch.setattr(publication, "_artifact_from_fd", forbidden_snapshot)
    with pending.open("w") as handle:
        handle.write("completed bank\n")
        with pytest.raises(OSError, match="bank fsync failed"):
            publication._durably_prepare_output_artifact(handle, pending, output)

    assert snapshot_called is False
    assert pending.read_text() == "completed bank\n"


def test_staged_manifest_syncs_bytes_then_its_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    events: list[str] = []
    real_fsync = os.fsync

    def fsync_spy(descriptor: int) -> None:
        descriptor_stat = os.fstat(descriptor)
        events.append("directory" if stat.S_ISDIR(descriptor_stat.st_mode) else "file")
        real_fsync(descriptor)

    monkeypatch.setattr(publication.os, "fsync", fsync_spy)
    pending_meta = tmp_path / ".bank.meta.json.tmp-pending"
    publication._write_json_staged(pending_meta, {"complete": True})

    assert events == ["file", "directory"]
    assert json.loads(pending_meta.read_text()) == {"complete": True}


def test_staged_manifest_file_sync_failure_retains_exact_recovery_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    pending_meta = tmp_path / ".bank.meta.json.tmp-pending"
    real_fsync = os.fsync

    def fsync_spy(descriptor: int) -> None:
        descriptor_stat = os.fstat(descriptor)
        if stat.S_ISDIR(descriptor_stat.st_mode):
            real_fsync(descriptor)
            return
        raise OSError(errno.EIO, "manifest fsync failed")

    monkeypatch.setattr(publication.os, "fsync", fsync_spy)
    with pytest.raises(OSError, match="manifest fsync failed"):
        publication._write_json_staged(pending_meta, {"complete": True})

    assert json.loads(pending_meta.read_text()) == {"complete": True}


def test_staged_manifest_directory_sync_failure_keeps_recovery_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    pending_meta = tmp_path / ".bank.meta.json.tmp-pending"
    real_fsync = os.fsync

    def fsync_spy(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError(errno.EIO, "manifest directory fsync failed")
        real_fsync(descriptor)

    monkeypatch.setattr(publication.os, "fsync", fsync_spy)
    with pytest.raises(OSError, match="manifest directory fsync failed"):
        publication._write_json_staged(pending_meta, {"complete": True})

    assert json.loads(pending_meta.read_text()) == {"complete": True}


def test_hard_link_publication_syncs_destination_and_retains_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    source_dir = tmp_path / "staging"
    destination_dir = tmp_path / "published"
    source_dir.mkdir()
    destination_dir.mkdir()
    pending = source_dir / ".bank.jsonl.tmp-pending"
    output = destination_dir / "bank.jsonl"
    pending.write_text("completed bank\n")
    events: list[str] = []
    real_fsync = os.fsync
    real_link = publication._link_open_file
    directory_labels = {
        (source_dir.stat().st_dev, source_dir.stat().st_ino): "source",
        (destination_dir.stat().st_dev, destination_dir.stat().st_ino): "destination",
    }

    def fsync_spy(descriptor: int) -> None:
        descriptor_stat = os.fstat(descriptor)
        label = directory_labels.get((descriptor_stat.st_dev, descriptor_stat.st_ino))
        if label is not None:
            events.append(f"fsync:{label}")
        real_fsync(descriptor)

    def link_spy(file_fd: int, parent_fd: int, name: str) -> None:
        events.append("link")
        real_link(file_fd, parent_fd, name)

    monkeypatch.setattr(publication.os, "fsync", fsync_spy)
    monkeypatch.setattr(publication, "_link_open_file", link_spy)
    publication._publish_no_replace(pending, output)

    assert events == ["link", "fsync:destination"]
    assert output.read_text() == "completed bank\n"
    assert output.samefile(pending)
    assert output.stat().st_nlink == 2


def test_same_directory_publication_keeps_destination_sync_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    pending = tmp_path / ".bank.jsonl.tmp-pending"
    output = tmp_path / "bank.jsonl"
    pending.write_text("completed bank\n")
    events: list[str] = []
    real_fsync = os.fsync
    real_link = publication._link_open_file

    def fsync_spy(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            events.append("fsync")
        real_fsync(descriptor)

    def link_spy(file_fd: int, parent_fd: int, name: str) -> None:
        events.append("link")
        real_link(file_fd, parent_fd, name)

    monkeypatch.setattr(publication.os, "fsync", fsync_spy)
    monkeypatch.setattr(publication, "_link_open_file", link_spy)
    publication._publish_no_replace(pending, output)

    assert events == ["link", "fsync"]
    assert output.samefile(pending)


def test_publication_rehashes_after_destination_directory_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    pending = tmp_path / ".bank.jsonl.tmp-pending"
    output = tmp_path / "bank.jsonl"
    pending.write_text("completed bank\n")
    real_fsync = os.fsync
    directory_syncs = 0

    def fsync_spy(descriptor: int) -> None:
        nonlocal directory_syncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_syncs += 1
            if directory_syncs == 1:
                output.write_text("mutated bank\n")
        real_fsync(descriptor)

    monkeypatch.setattr(publication.os, "fsync", fsync_spy)
    with pytest.raises(SystemExit, match="evidence artifact changed"):
        publication._publish_no_replace(pending, output)

    assert directory_syncs == 1
    assert pending.samefile(output)
    assert output.read_text() == "mutated bank\n"
    assert output.stat().st_nlink == 2


def test_destination_directory_sync_failure_retains_both_publication_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    source_dir = tmp_path / "staging"
    destination_dir = tmp_path / "published"
    source_dir.mkdir()
    destination_dir.mkdir()
    pending = source_dir / ".bank.jsonl.tmp-pending"
    output = destination_dir / "bank.jsonl"
    pending.write_text("completed bank\n")
    real_fsync = os.fsync
    unlinked = False

    def fsync_spy(descriptor: int) -> None:
        descriptor_stat = os.fstat(descriptor)
        if (
            descriptor_stat.st_dev == destination_dir.stat().st_dev
            and descriptor_stat.st_ino == destination_dir.stat().st_ino
        ):
            raise OSError(errno.EIO, "destination directory fsync failed")
        real_fsync(descriptor)

    def forbidden_unlink(_path: str, *_args: Any, **_kwargs: Any) -> None:
        nonlocal unlinked
        unlinked = True
        raise AssertionError("recovery name consumed after destination sync failure")

    monkeypatch.setattr(publication.os, "fsync", fsync_spy)
    monkeypatch.setattr(publication.os, "unlink", forbidden_unlink)
    with pytest.raises(OSError, match="destination directory fsync failed"):
        publication._publish_no_replace(pending, output)

    assert unlinked is False
    assert output.read_text() == "completed bank\n"
    assert pending.read_text() == "completed bank\n"


def test_final_manifest_directory_sync_failure_restarts_to_complete_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = publication._pending_output_path(output)
    pending_meta = publication._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
        "complete": True,
        "output": publication._prepared_output_artifact(pending_output, output),
    }
    real_fsync = os.fsync
    real_link = publication._link_open_file
    final_manifest_linked = False
    injected = False

    def link_spy(file_fd: int, parent_fd: int, name: str) -> None:
        nonlocal final_manifest_linked
        real_link(file_fd, parent_fd, name)
        if name == meta.name:
            final_manifest_linked = True

    def fsync_spy(descriptor: int) -> None:
        nonlocal injected
        if final_manifest_linked and stat.S_ISDIR(os.fstat(descriptor).st_mode):
            injected = True
            raise OSError(errno.EIO, "final manifest directory fsync failed")
        real_fsync(descriptor)

    monkeypatch.setattr(publication, "_link_open_file", link_spy)
    monkeypatch.setattr(publication.os, "fsync", fsync_spy)
    with pytest.raises(OSError, match="final manifest directory fsync failed"):
        publication._publish_evidence_pair(
            pending_output, output, pending_meta, meta, manifest,
        )

    assert injected is True
    assert output.read_text() == "completed bank\n"
    assert json.loads(meta.read_text()) == manifest
    assert output.samefile(pending_output)
    assert pending_meta.samefile(meta)

    monkeypatch.setattr(publication, "_link_open_file", real_link)
    monkeypatch.setattr(publication.os, "fsync", real_fsync)
    assert publication._require_new_output_pair(output, meta, overwrite=False) is True
    assert output.read_text() == "completed bank\n"
    assert json.loads(meta.read_text()) == manifest
    assert output.samefile(pending_output)
    assert meta.samefile(pending_meta)


def test_manifest_publication_io_failure_quarantines_named_manifest_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = publication._pending_output_path(output)
    pending_meta = publication._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
        "complete": True,
        "output": publication._prepared_output_artifact(pending_output, output),
    }
    attacker_manifest = {**manifest, "tampered": True}
    attacker = tmp_path / "attacker-manifest.json"
    attacker.write_text(json.dumps(attacker_manifest))
    real_fsync = publication.os.fsync
    injected = False

    def mutate_manifest_then_fail_sync(descriptor: int) -> None:
        nonlocal injected
        if (
            not injected
            and stat.S_ISDIR(os.fstat(descriptor).st_mode)
            and meta.exists()
            and pending_meta.exists()
        ):
            meta.unlink()
            pending_meta.unlink()
            os.link(attacker, meta)
            os.link(attacker, pending_meta)
            attacker.unlink()
            injected = True
            raise OSError(errno.EIO, "linked manifest sync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(publication.os, "fsync", mutate_manifest_then_fail_sync)
    with pytest.raises(SystemExit, match=r"anchored regular file|changed"):
        publication._publish_evidence_pair(
            pending_output, output, pending_meta, meta, manifest,
        )

    monkeypatch.setattr(publication.os, "fsync", real_fsync)
    assert injected is True
    assert publication._invalid_manifest_path(meta).exists()
    assert output.read_text() == "completed bank\n"
    assert json.loads(meta.read_text()) == attacker_manifest
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)


def test_final_manifest_recovery_rejects_identical_different_inode_manifest(
    tmp_path: Path,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = publication._pending_output_path(output)
    pending_meta = publication._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    os.link(pending_output, output)
    manifest = {
        "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
        "complete": True,
        "output": publication._prepared_output_artifact(pending_output, output),
    }
    publication._write_json_staged(pending_meta, manifest)
    meta.write_bytes(pending_meta.read_bytes())
    os.link(pending_meta, tmp_path / "pending-manifest-extra-link")
    os.link(meta, tmp_path / "final-manifest-extra-link")
    assert not pending_meta.samefile(meta)

    with pytest.raises(SystemExit, match="not the same recovery hard link"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    assert json.loads(meta.read_text()) == manifest
    assert json.loads(pending_meta.read_text()) == manifest
    assert output.read_text() == "completed bank\n"


def test_final_manifest_recovery_rejects_bank_that_differs_from_manifest(
    tmp_path: Path,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output = tmp_path / "bank.jsonl"
    pending_output = publication._pending_output_path(output)
    expected = tmp_path / ".expected-bank"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_meta = publication._pending_manifest_path(meta)
    expected.write_text("expected bank\n")
    pending_output.write_text("substituted bank\n")
    os.link(pending_output, output)
    manifest = {
        "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
        "complete": True,
        "output": publication._prepared_output_artifact(expected, output),
    }
    publication._write_json_staged(pending_meta, manifest)
    os.link(pending_meta, meta)

    with pytest.raises(SystemExit, match="pending trajectory bank does not match"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    assert output.read_text() == "substituted bank\n"
    assert pending_meta.samefile(meta)


def test_final_manifest_recovery_sync_failure_retains_pending_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output, meta, _pending_output, pending_meta, _manifest = (
        _prepare_pending_manifest_recovery(tmp_path, bank_published=True)
    )
    os.link(pending_meta, meta)
    manifest_inode = meta.stat().st_ino
    real_fsync = os.fsync
    manifest_file_synced = False

    def fsync_spy(descriptor: int) -> None:
        nonlocal manifest_file_synced
        descriptor_stat = os.fstat(descriptor)
        if stat.S_ISREG(descriptor_stat.st_mode) and descriptor_stat.st_ino == manifest_inode:
            manifest_file_synced = True
        elif manifest_file_synced and stat.S_ISDIR(descriptor_stat.st_mode):
            raise OSError(errno.EIO, "manifest recovery directory fsync failed")
        real_fsync(descriptor)

    monkeypatch.setattr(publication.os, "fsync", fsync_spy)
    with pytest.raises(OSError, match="manifest recovery directory fsync failed"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    assert manifest_file_synced is True
    assert pending_meta.samefile(meta)
    assert output.read_text() == "completed bank\n"


def test_final_manifest_recovery_rejects_unexpected_pending_bank(tmp_path: Path) -> None:
    from scripts import chunk_trajectory_publication as publication

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = publication._pending_output_path(output)
    pending_meta = publication._pending_manifest_path(meta)
    output.write_text("completed bank\n")
    pending_output.write_text("unexpected pending bank\n")
    manifest = {
        "schema": publication.CHUNK_TRAJECTORY_SCHEMA,
        "complete": True,
        "output": publication._prepared_output_artifact(output, output),
    }
    publication._write_json_staged(pending_meta, manifest)
    os.link(pending_meta, meta)

    with pytest.raises(SystemExit, match="unexpected hard links"):
        publication._require_new_output_pair(output, meta, overwrite=False)

    assert pending_output.read_text() == "unexpected pending bank\n"
    assert pending_meta.samefile(meta)


def test_pair_publication_preserves_preexisting_pending_manifest(tmp_path: Path) -> None:
    from scripts import backtest_chunk_trajectory as producer
    from scripts import chunk_trajectory_publication as publication

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = producer._pending_output_path(output)
    pending_meta = producer._pending_manifest_path(meta)
    pending_output.write_text("new bank\n")
    pending_meta.write_text("existing pending manifest\n")
    manifest = {
        "schema": producer._SCHEMA,
        "complete": True,
        "output": producer._prepared_output_artifact(pending_output, output),
    }

    with pytest.raises(FileExistsError):
        producer._publish_evidence_pair(
            pending_output, output, pending_meta, meta, manifest,
        )

    assert pending_output.read_text() == "new bank\n"
    assert pending_meta.read_text() == "existing pending manifest\n"
    assert not output.exists()
    assert not meta.exists()
    assert publication._invalid_manifest_path(meta).exists()

    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        producer._require_new_output_pair(output, meta, overwrite=False)

    assert pending_output.read_text() == "new bank\n"
    assert pending_meta.read_text() == "existing pending manifest\n"
    assert not output.exists()
    assert not meta.exists()


def test_producer_atomic_json_preserves_preexisting_unique_staging_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import backtest_chunk_trajectory as producer
    from scripts import chunk_trajectory_publication as publication

    output = tmp_path / "manifest.json"
    monkeypatch.setattr(publication.secrets, "token_hex", lambda _size: "fixed")
    staging = output.with_name(f".{output.name}.tmp-fixed")
    staging.write_text("other process staging\n")

    with pytest.raises(FileExistsError):
        producer._write_json_atomic(output, {"new": True})

    assert staging.read_text() == "other process staging\n"
    assert not output.exists()


def test_producer_atomic_json_retains_requested_payload_through_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    output = tmp_path / "preregistration.json"
    monkeypatch.setattr(publication.secrets, "token_hex", lambda _size: "fixed")
    staging = output.with_name(f".{output.name}.tmp-fixed")
    payload = {"schema": "preregistration", "complete": True}
    attacker_payload = {**payload, "tampered": True}
    real_publish = publication._publish_no_replace

    def mutate_at_publication(
        tmp: Path, destination: Path, **kwargs: Any,
    ) -> None:
        if tmp == staging:
            staging.write_text(json.dumps(attacker_payload))
        real_publish(tmp, destination, **kwargs)

    monkeypatch.setattr(publication, "_publish_no_replace", mutate_at_publication)
    with pytest.raises(SystemExit, match=r"staged evidence changed|changed during"):
        publication._write_json_atomic(output, payload)

    assert json.loads(staging.read_text()) == attacker_payload
    assert not output.exists()


def test_producer_atomic_json_retains_one_parent_across_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    live = tmp_path / "live"
    held = tmp_path / "held-live"
    decoy = tmp_path / "decoy"
    live.mkdir()
    decoy.mkdir()
    output = live / "preregistration.json"
    staging = live / ".preregistration.json.tmp-fixed"
    payload = {"schema": "preregistration", "complete": True}
    real_open_parent = publication._open_parent
    real_require_parent = publication._require_parent
    open_calls = 0
    swapped = False

    def swap_on_third_parent_open(path: Path) -> int:
        nonlocal open_calls, swapped
        open_calls += 1
        if open_calls == 3:
            live.rename(held)
            decoy.rename(live)
            swapped = True
        return real_open_parent(path)

    def restore_before_source_revalidation(path: Path, parent_fd: int) -> None:
        nonlocal swapped
        if swapped and path.name == staging.name:
            live.rename(decoy)
            held.rename(live)
            swapped = False
        real_require_parent(path, parent_fd)

    monkeypatch.setattr(publication.secrets, "token_hex", lambda _size: "fixed")
    monkeypatch.setattr(publication, "_open_parent", swap_on_third_parent_open)
    monkeypatch.setattr(
        publication, "_require_parent", restore_before_source_revalidation,
    )

    publication._write_json_atomic(output, payload)

    assert open_calls == 1
    assert swapped is False
    assert json.loads(output.read_text()) == payload
    assert output.samefile(staging)
    assert not (decoy / output.name).exists()


def test_analyzer_anonymous_json_preserves_preexisting_staging_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "analysis.json"
    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    staging.write_text("other process staging\n")

    with controller_module._anchored_output_target(
        tmp_path / "bank.jsonl",
        tmp_path / "bank.jsonl.meta.json",
        output,
    ) as target:
        controller_module._write_json_atomic(target, "{}")

    assert staging.read_text() == "other process staging\n"
    assert output.read_text() == "{}\n"


def test_analyzer_rejects_same_inode_anonymous_report_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "analysis.json"
    real_fsync = controller_module.os.fsync
    rewritten = False

    def rewrite_anonymous_inode_after_fsync(fd: int) -> None:
        nonlocal rewritten
        real_fsync(fd)
        if not rewritten and os.fstat(fd).st_nlink == 0:
            os.ftruncate(fd, 0)
            os.pwrite(fd, b"attacker report\n", 0)
            real_fsync(fd)
            rewritten = True

    with controller_module._anchored_output_target(
        tmp_path / "bank.jsonl",
        tmp_path / "bank.jsonl.meta.json",
        output,
    ) as target:
        monkeypatch.setattr(
            controller_module.os, "fsync", rewrite_anonymous_inode_after_fsync,
        )
        with pytest.raises(RuntimeError, match="differs from rendered report"):
            controller_module._write_json_atomic(target, "{}")

    assert rewritten is True
    assert not output.exists()


def test_analyzer_revalidates_separate_report_parent_on_context_exit(
    tmp_path: Path,
) -> None:
    report_parent = tmp_path / "reports"
    report_parent.mkdir()
    displaced_parent = tmp_path / "displaced-reports"
    output = report_parent / "analysis.json"

    def publish_then_swap_parent() -> None:
        with controller_module._anchored_output_target(
            tmp_path / "bank.jsonl",
            tmp_path / "bank.jsonl.meta.json",
            output,
        ) as target:
            controller_module._write_json_atomic(target, "{}")
            report_parent.rename(displaced_parent)
            report_parent.mkdir()
            output.write_text("attacker report\n")

    with pytest.raises(RuntimeError, match="output parent changed"):
        publish_then_swap_parent()

    assert output.read_text() == "attacker report\n"
    assert (displaced_parent / output.name).read_text() == "{}\n"


def test_analyzer_revalidates_report_leaf_on_context_exit(tmp_path: Path) -> None:
    output = tmp_path / "analysis.json"
    displaced = tmp_path / "authentic-analysis.json"

    def publish_then_swap_leaf() -> None:
        with controller_module._anchored_output_target(
            tmp_path / "bank.jsonl",
            tmp_path / "bank.jsonl.meta.json",
            output,
        ) as target:
            controller_module._write_json_atomic(target, "{}")
            output.rename(displaced)
            output.write_text("attacker report\n")

    with pytest.raises(SystemExit, match="anchored regular file"):
        publish_then_swap_leaf()

    assert output.read_text() == "attacker report\n"
    assert displaced.read_text() == "{}\n"


def test_analyzer_atomic_json_requires_fresh_ordinary_output(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "new" / "nested"
    alias = tmp_path / "output-parent"
    output_directory.mkdir(parents=True)
    alias.symlink_to(output_directory, target_is_directory=True)
    output = alias / "analysis.json"

    with controller_module._anchored_output_target(
        tmp_path / "bank.jsonl",
        tmp_path / "bank.jsonl.meta.json",
        output,
    ) as target:
        controller_module._write_json_atomic(target, '{"generation": 1}')

    with (
        pytest.raises(FileExistsError, match="choose a fresh --out path"),
        controller_module._anchored_output_target(
            tmp_path / "bank.jsonl",
            tmp_path / "bank.jsonl.meta.json",
            output,
        ),
    ):
        pytest.fail("existing analyzer report reached publication")

    assert output.read_text() == '{"generation": 1}\n'
    assert not output.with_name(f".{output.name}.tmp-{os.getpid()}").exists()
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(target.parent_fd)


def test_analyzer_no_clobber_when_consumed_bank_appears_at_output_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_directory = tmp_path / "evidence"
    output_directory = tmp_path / "reports"
    evidence_directory.mkdir()
    output_directory.mkdir()
    bank = evidence_directory / "bank.jsonl"
    bank.write_bytes(b"authenticated trajectory bank\n")
    bank_bytes, consumed = controller_module._read_consumed_artifact(
        bank, role="trajectory_bank",
    )
    output = output_directory / "analysis.json"
    real_publish = controller_module._link_anonymous_file_no_replace

    def move_bank_before_publish(
        source_fd: int, parent_fd: int, name: str,
    ) -> None:
        bank.rename(output)
        real_publish(source_fd, parent_fd, name)

    monkeypatch.setattr(
        controller_module,
        "_link_anonymous_file_no_replace",
        move_bank_before_publish,
    )
    targets: list[controller_module._AnchoredOutputTarget] = []

    def attempt_publication() -> None:
        with controller_module._anchored_output_target(
            bank,
            tmp_path / "bank.jsonl.meta.json",
            output,
            consumed_artifacts=(consumed,),
        ) as target:
            targets.append(target)
            with pytest.raises(FileExistsError, match="appeared during publication"):
                controller_module._write_json_atomic(target, "{}")

    with pytest.raises(ValueError, match="alias of a consumed input artifact"):
        attempt_publication()

    assert not bank.exists()
    assert output.read_bytes() == bank_bytes
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(targets[0].parent_fd)


def test_analyzer_publication_never_name_unlinks_consumed_or_foreign_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_directory = tmp_path / "evidence"
    output_directory = tmp_path / "reports"
    evidence_directory.mkdir()
    output_directory.mkdir()
    bank = evidence_directory / "bank.jsonl"
    bank.write_bytes(b"authenticated trajectory bank\n")
    bank_bytes, consumed = controller_module._read_consumed_artifact(
        bank, role="trajectory_bank",
    )
    output = output_directory / "analysis.json"
    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    staging.write_bytes(b"foreign staging entry\n")
    real_unlink = controller_module.os.unlink
    unlink_calls = 0

    def destroy_bank_at_unlink(path: str | bytes, **kwargs: Any) -> None:
        nonlocal unlink_calls
        unlink_calls += 1
        bank.replace(staging)
        real_unlink(path, **kwargs)

    with controller_module._anchored_output_target(
        bank,
        tmp_path / "bank.jsonl.meta.json",
        output,
        consumed_artifacts=(consumed,),
    ) as target:
        monkeypatch.setattr(controller_module.os, "unlink", destroy_bank_at_unlink)
        controller_module._write_json_atomic(target, "{}")

    assert unlink_calls == 0
    assert bank.read_bytes() == bank_bytes
    assert staging.read_bytes() == b"foreign staging entry\n"
    assert output.read_text() == "{}\n"
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(target.parent_fd)


def test_analyzer_fails_closed_without_anonymous_tmpfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "analysis.json"
    monkeypatch.delattr(controller_module.os, "O_TMPFILE")

    with controller_module._anchored_output_target(
        tmp_path / "bank.jsonl",
        tmp_path / "bank.jsonl.meta.json",
        output,
    ) as target, pytest.raises(RuntimeError, match="requires Linux O_TMPFILE"):
        controller_module._write_json_atomic(target, "{}")

    assert not output.exists()


def test_analyzer_fails_closed_without_linkat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "analysis.json"
    monkeypatch.setattr(
        controller_module.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    with controller_module._anchored_output_target(
        tmp_path / "bank.jsonl",
        tmp_path / "bank.jsonl.meta.json",
        output,
    ) as target, pytest.raises(RuntimeError, match="requires Linux linkat"):
        controller_module._write_json_atomic(target, "{}")

    assert not output.exists()


def test_analyzer_anonymous_tmpfile_fd_closes_on_publication_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "analysis.json"
    anonymous_fds: list[int] = []

    def fail_publication(source_fd: int, parent_fd: int, name: str) -> None:
        del parent_fd, name
        anonymous_fds.append(source_fd)
        raise RuntimeError("injected anonymous publication failure")

    monkeypatch.setattr(
        controller_module,
        "_link_anonymous_file_no_replace",
        fail_publication,
    )
    with controller_module._anchored_output_target(
        tmp_path / "bank.jsonl",
        tmp_path / "bank.jsonl.meta.json",
        output,
    ) as target, pytest.raises(RuntimeError, match="injected anonymous"):
        controller_module._write_json_atomic(target, "{}")

    assert not output.exists()
    assert anonymous_fds
    for anonymous_fd in anonymous_fds:
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(anonymous_fd)
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(target.parent_fd)


def test_analyzer_parent_symlink_swap_cannot_replace_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    safe_parent = tmp_path / "safe"
    protected_parent = tmp_path / "checkpoint"
    safe_parent.mkdir()
    protected_parent.mkdir()
    checkpoint = protected_parent / "model.pt"
    checkpoint.write_bytes(b"authenticated checkpoint\n")
    alias = tmp_path / "output-parent"
    alias.symlink_to(safe_parent, target_is_directory=True)
    output = alias / checkpoint.name
    manifest = {"checkpoint": {"path": str(checkpoint)}}
    real_publish = controller_module._link_anonymous_file_no_replace

    def swap_parent_before_publish(
        source_fd: int, parent_fd: int, name: str,
    ) -> None:
        alias.unlink()
        alias.symlink_to(protected_parent, target_is_directory=True)
        real_publish(source_fd, parent_fd, name)

    with controller_module._anchored_output_target(
        tmp_path / "bank.jsonl",
        tmp_path / "bank.jsonl.meta.json",
        output,
        manifest=manifest,
    ) as target:
        monkeypatch.setattr(
            controller_module,
            "_link_anonymous_file_no_replace",
            swap_parent_before_publish,
        )
        with pytest.raises(RuntimeError, match="output parent changed"):
            controller_module._write_json_atomic(target, "{}")

    assert checkpoint.read_bytes() == b"authenticated checkpoint\n"
    assert (safe_parent / checkpoint.name).read_text() == "{}\n"


def test_analyzer_parent_symlink_swap_cannot_replace_analyzer_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyzer_source = Path(controller_module.__file__).resolve()
    original_source = analyzer_source.read_bytes()
    safe_parent = tmp_path / "safe"
    safe_parent.mkdir()
    alias = tmp_path / "output-parent"
    alias.symlink_to(safe_parent, target_is_directory=True)
    output = alias / analyzer_source.name
    real_publish = controller_module._link_anonymous_file_no_replace

    def swap_parent_before_publish(
        source_fd: int, parent_fd: int, name: str,
    ) -> None:
        alias.unlink()
        alias.symlink_to(analyzer_source.parent, target_is_directory=True)
        real_publish(source_fd, parent_fd, name)

    with controller_module._anchored_output_target(
        tmp_path / "bank.jsonl",
        tmp_path / "bank.jsonl.meta.json",
        output,
    ) as target:
        monkeypatch.setattr(
            controller_module,
            "_link_anonymous_file_no_replace",
            swap_parent_before_publish,
        )
        with pytest.raises(RuntimeError, match="output parent changed"):
            controller_module._write_json_atomic(target, "{}")

    assert analyzer_source.read_bytes() == original_source
    assert (safe_parent / analyzer_source.name).read_text() == "{}\n"


def test_analyzer_pre_anchor_bank_symlink_swap_keeps_consumed_target_protected(
    tmp_path: Path,
) -> None:
    original_directory = tmp_path / "original"
    original_directory.mkdir()
    bank = original_directory / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    bank_bytes = bank.read_bytes()
    alias = tmp_path / "bank-link.jsonl"
    alias.symlink_to(bank)

    _transitions, info = load_transitions(
        alias, meta_path=meta, methodology_smoke=True,
    )
    consumed = info["analyzer_consumed_inputs"]
    manifest = dict(info["manifest"])
    manifest["output"] = dict(manifest["output"])
    manifest["output"].pop("path", None)
    decoy = tmp_path / "decoy.jsonl"
    decoy.write_text("decoy\n")
    alias.unlink()
    alias.symlink_to(decoy)

    with (
        pytest.raises(ValueError, match="consumed input artifact"),
        controller_module._anchored_output_target(
            alias,
            meta,
            bank,
            manifest=manifest,
            consumed_artifacts=consumed,
        ),
    ):
        pytest.fail("consumed bank target reached publication")

    assert bank.read_bytes() == bank_bytes
    assert decoy.read_text() == "decoy\n"


def test_analyzer_pre_anchor_meta_symlink_swap_keeps_consumed_target_protected(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    meta_bytes = meta.read_bytes()
    alias = tmp_path / "manifest-link.json"
    alias.symlink_to(meta)

    _transitions, info = load_transitions(
        bank, meta_path=alias, methodology_smoke=True,
    )
    consumed = info["analyzer_consumed_inputs"]
    decoy = tmp_path / "decoy.json"
    decoy.write_text("{}\n")
    alias.unlink()
    alias.symlink_to(decoy)

    with (
        pytest.raises(ValueError, match="consumed input artifact"),
        controller_module._anchored_output_target(
            bank,
            alias,
            meta,
            manifest=info["manifest"],
            consumed_artifacts=consumed,
        ),
    ):
        pytest.fail("consumed manifest target reached publication")

    assert meta.read_bytes() == meta_bytes
    assert decoy.read_text() == "{}\n"


def test_analyzer_output_protects_manifest_recorded_bank_path(tmp_path: Path) -> None:
    published_bank = tmp_path / "published-bank.jsonl"

    with pytest.raises(ValueError, match="consumed input artifact"):
        _require_safe_output_path(
            tmp_path / "relocated-bank.jsonl",
            tmp_path / "relocated-bank.jsonl.meta.json",
            published_bank,
            manifest={"output": {"path": str(published_bank)}},
        )


def test_analyzer_output_rejects_renamed_checkpoint_by_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"authenticated checkpoint\n")
    checkpoint_stat = checkpoint.stat()
    manifest = {
        "checkpoint": {
            "path": str(checkpoint),
            "device": checkpoint_stat.st_dev,
            "inode": checkpoint_stat.st_ino,
        },
    }
    renamed = tmp_path / "apparently-ordinary-report.json"
    checkpoint.rename(renamed)

    with pytest.raises(ValueError, match="alias of a consumed input artifact"):
        _require_safe_output_path(
            tmp_path / "bank.jsonl",
            tmp_path / "bank.jsonl.meta.json",
            renamed,
            manifest=manifest,
        )

    assert renamed.read_bytes() == b"authenticated checkpoint\n"


def test_analyzer_full_manifest_allows_output_outside_syzygy_roots(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    _transitions, info = load_transitions(bank, meta_path=meta)
    assert info["decision_grade"] is True
    manifest = info["manifest"]
    filesystem_root = Path(os.sep).stat()
    for directory in manifest["syzygy"]["directories"]:
        directory["path_components"][0].update({
            "device": filesystem_root.st_dev,
            "inode": filesystem_root.st_ino,
            "mtime_ns": filesystem_root.st_mtime_ns,
            "ctime_ns": filesystem_root.st_ctime_ns,
        })

    _require_safe_output_path(
        bank,
        meta,
        tmp_path / "ordinary" / "analysis.json",
        manifest=manifest,
        consumed_artifacts=info["analyzer_consumed_inputs"],
    )


@pytest.mark.parametrize("protected_kind", ["syzygy", "snapshot"])
def test_analyzer_descriptor_ancestry_rejects_protected_root_after_procfs_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protected_kind: str,
) -> None:
    output_parent = tmp_path / "output-parent"
    protected_root = tmp_path / "protected-root"
    displaced_output_parent = tmp_path / "displaced-output-parent"
    output_parent.mkdir()
    protected_root.mkdir()
    evidence = protected_root / "evidence.bin"
    evidence.write_bytes(b"authenticated protected evidence\n")
    protected_stat = protected_root.stat()
    protected_identity = {
        "device": protected_stat.st_dev,
        "inode": protected_stat.st_ino,
    }
    if protected_kind == "syzygy":
        manifest = {
            "syzygy": {
                "directories": [{
                    "path": str(protected_root),
                    "root_identity": protected_identity,
                }],
            },
        }
    else:
        manifest = {
            "matched_row_origin_verification": {
                "snapshot_inventory": {
                    "path": str(protected_root),
                    "root_identity": protected_identity,
                },
            },
        }
    output = output_parent / "analysis.json"
    real_open_directory = controller_module._open_directory_anchored
    real_strict_descriptor_path = controller_module._strict_descriptor_path
    opened_protected_root = False
    restored_decoy = False

    def open_protected_root(path: Path, *, create: bool) -> int:
        nonlocal opened_protected_root
        assert path == output_parent
        assert create is True
        output_parent.rename(displaced_output_parent)
        protected_root.rename(output_parent)
        opened_protected_root = True
        return real_open_directory(path, create=False)

    def resolve_then_restore_decoy(fd: int, *, kind: str) -> Path:
        nonlocal restored_decoy
        resolved = real_strict_descriptor_path(fd, kind=kind)
        output_parent.rename(protected_root)
        displaced_output_parent.rename(output_parent)
        restored_decoy = True
        return resolved

    monkeypatch.setattr(
        controller_module, "_open_directory_anchored", open_protected_root,
    )
    monkeypatch.setattr(
        controller_module, "_strict_descriptor_path", resolve_then_restore_decoy,
    )

    with (
        pytest.raises(ValueError, match="inside a consumed protected directory"),
        controller_module._anchored_output_target(
            tmp_path / "bank.jsonl",
            tmp_path / "bank.jsonl.meta.json",
            output,
            manifest=manifest,
        ),
    ):
        pytest.fail("protected root reached analyzer publication")

    assert opened_protected_root is True
    assert restored_decoy is True
    assert evidence.read_bytes() == b"authenticated protected evidence\n"
    assert not output.exists()


def test_analyzer_descriptor_ancestry_depth_failure_closes_walk_fds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_parent = tmp_path / "output-parent"
    output_parent.mkdir()
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_fd = os.open(output_parent, flags)
    duplicated_fds: list[int] = []
    real_dup = controller_module.os.dup

    def tracked_dup(fd: int) -> int:
        duplicated = real_dup(fd)
        duplicated_fds.append(duplicated)
        return duplicated

    monkeypatch.setattr(controller_module.os, "dup", tracked_dup)
    monkeypatch.setattr(controller_module, "_MAX_OUTPUT_ANCESTOR_DEPTH", 1)
    try:
        with pytest.raises(RuntimeError, match="ancestry is too deep"):
            controller_module._descriptor_ancestor_identities(parent_fd)
        assert stat.S_ISDIR(os.fstat(parent_fd).st_mode)
        assert duplicated_fds
        for duplicated in duplicated_fds:
            with pytest.raises(OSError, match="Bad file descriptor"):
                os.fstat(duplicated)
    finally:
        os.close(parent_fd)


def test_analyzer_output_rejects_renamed_syzygy_directory_by_identity(
    tmp_path: Path,
) -> None:
    tablebases = tmp_path / "syzygy"
    tablebases.mkdir()
    tablebase_stat = tablebases.stat()
    manifest = {
        "syzygy": {
            "directories": [{
                "path": str(tablebases),
                "root_identity": {
                    "device": tablebase_stat.st_dev,
                    "inode": tablebase_stat.st_ino,
                },
            }],
        },
    }
    renamed = tmp_path / "ordinary-directory"
    tablebases.rename(renamed)
    output = renamed / "analysis.json"

    with pytest.raises(ValueError, match="inside a consumed protected directory"):
        _require_safe_output_path(
            tmp_path / "bank.jsonl",
            tmp_path / "bank.jsonl.meta.json",
            output,
            manifest=manifest,
        )

    assert not output.exists()


def test_analyzer_output_rejects_renamed_snapshot_root_by_identity(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    snapshot_stat = snapshot.stat()
    manifest = {
        "matched_row_origin_verification": {
            "snapshot_inventory": {
                "path": str(snapshot),
                "root_identity": {
                    "device": snapshot_stat.st_dev,
                    "inode": snapshot_stat.st_ino,
                },
            },
        },
    }
    renamed = tmp_path / "ordinary-directory"
    snapshot.rename(renamed)
    output = renamed / "analysis.json"

    with pytest.raises(ValueError, match="inside a consumed protected directory"):
        _require_safe_output_path(
            tmp_path / "bank.jsonl",
            tmp_path / "bank.jsonl.meta.json",
            output,
            manifest=manifest,
        )

    assert not output.exists()


def test_analyzer_output_rejects_renamed_snapshot_shard_by_identity(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    shard = snapshot / "s0.zarr"
    shard.mkdir(parents=True)
    internal_data = shard / "positions.bin"
    internal_data.write_bytes(b"authenticated replay data\n")
    snapshot_stat = snapshot.stat()
    shard_stat = shard.stat()
    manifest = {
        "matched_row_origin_verification": {
            "snapshot_inventory": {
                "path": str(snapshot),
                "root_identity": {
                    "device": snapshot_stat.st_dev,
                    "inode": snapshot_stat.st_ino,
                },
                "shards": [{
                    "name": shard.name,
                    "device": shard_stat.st_dev,
                    "inode": shard_stat.st_ino,
                }],
            },
        },
    }
    renamed = tmp_path / "apparently-ordinary.zarr"
    shard.rename(renamed)
    output = renamed / "analysis.json"

    with pytest.raises(ValueError, match="inside a consumed protected directory"):
        _require_safe_output_path(
            tmp_path / "bank.jsonl",
            tmp_path / "bank.jsonl.meta.json",
            output,
            manifest=manifest,
        )

    assert (renamed / internal_data.name).read_bytes() == (
        b"authenticated replay data\n"
    )
    assert not output.exists()


def test_analyzer_output_rejects_renamed_tablebase_file_by_identity(
    tmp_path: Path,
) -> None:
    tablebases = tmp_path / "syzygy"
    tablebases.mkdir()
    table = tablebases / "KQvK.rtbw"
    table.write_bytes(b"authenticated tablebase\n")
    table_stat = table.stat()
    manifest = {
        "syzygy": {
            "directories": [{
                "path": str(tablebases),
                "file_identities": [[
                    table.name,
                    table_stat.st_size,
                    table_stat.st_mtime_ns,
                    table_stat.st_ctime_ns,
                    table_stat.st_dev,
                    table_stat.st_ino,
                ]],
            }],
        },
    }
    renamed = tmp_path / "apparently-ordinary-report.json"
    table.rename(renamed)

    with pytest.raises(ValueError, match="alias of a consumed input artifact"):
        _require_safe_output_path(
            tmp_path / "bank.jsonl",
            tmp_path / "bank.jsonl.meta.json",
            renamed,
            manifest=manifest,
        )

    assert renamed.read_bytes() == b"authenticated tablebase\n"


def test_analyzer_missing_procfs_descriptor_mapping_fails_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = tmp_path / "bank.jsonl"
    bank.write_text("bank evidence\n")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint evidence\n")
    output = tmp_path / "safe" / "analysis.json"
    _bytes, consumed = controller_module._read_consumed_artifact(
        bank, role="trajectory_bank",
    )
    monkeypatch.setattr(controller_module, "_PROC_SELF_FD", tmp_path / "missing-proc")

    with (
        pytest.raises(RuntimeError, match="procfs descriptor mapping"),
        controller_module._anchored_output_target(
            bank,
            tmp_path / "bank.jsonl.meta.json",
            output,
            manifest={"checkpoint": {"path": str(checkpoint)}},
            consumed_artifacts=(consumed,),
        ),
    ):
        pytest.fail("missing procfs mapping reached publication")

    assert bank.read_text() == "bank evidence\n"
    assert checkpoint.read_bytes() == b"checkpoint evidence\n"
    assert not output.exists()
    assert not output.with_name(f".{output.name}.tmp-{os.getpid()}").exists()


def test_analyzer_misdirected_procfs_descriptor_mapping_fails_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = tmp_path / "bank.jsonl"
    bank.write_text("bank evidence\n")
    output = tmp_path / "safe" / "analysis.json"
    wrong_parent = tmp_path / "wrong"
    fake_proc = tmp_path / "fake-proc"
    wrong_parent.mkdir()
    fake_proc.mkdir()
    _bytes, consumed = controller_module._read_consumed_artifact(
        bank, role="trajectory_bank",
    )
    real_open_directory = controller_module._open_directory_anchored

    def open_with_misdirected_proc(path: Path, *, create: bool) -> int:
        fd = real_open_directory(path, create=create)
        (fake_proc / str(fd)).symlink_to(wrong_parent, target_is_directory=True)
        return fd

    monkeypatch.setattr(controller_module, "_PROC_SELF_FD", fake_proc)
    monkeypatch.setattr(
        controller_module, "_open_directory_anchored", open_with_misdirected_proc,
    )

    with (
        pytest.raises(RuntimeError, match="mapping disagrees with fstat"),
        controller_module._anchored_output_target(
            bank,
            tmp_path / "bank.jsonl.meta.json",
            output,
            consumed_artifacts=(consumed,),
        ),
    ):
        pytest.fail("misdirected procfs mapping reached publication")

    assert bank.read_text() == "bank evidence\n"
    assert not output.exists()
    assert not output.with_name(f".{output.name}.tmp-{os.getpid()}").exists()


def test_analyzer_procfs_resolved_path_must_name_the_open_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumed = tmp_path / "consumed.jsonl"
    consumed.write_text("authenticated input\n")
    wrong = tmp_path / "wrong.jsonl"
    wrong.write_text("different input\n")
    fd = os.open(consumed, os.O_RDONLY | os.O_CLOEXEC)
    descriptor_link = controller_module._PROC_SELF_FD / str(fd)
    real_resolve = Path.resolve

    def misdirect_resolved_path(path: Path, strict: bool = False) -> Path:
        if path == descriptor_link:
            return wrong
        return real_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", misdirect_resolved_path)
    try:
        with pytest.raises(RuntimeError, match="mapping disagrees with fstat"):
            controller_module._strict_descriptor_path(fd, kind="file")
    finally:
        os.close(fd)

    assert consumed.read_text() == "authenticated input\n"
    assert wrong.read_text() == "different input\n"


def test_producer_recovers_fully_staged_pair_before_either_publish(tmp_path: Path) -> None:
    from scripts import backtest_chunk_trajectory as producer

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = producer._pending_output_path(output)
    pending_meta = producer._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": producer._SCHEMA,
        "complete": True,
        "output": producer._prepared_output_artifact(pending_output, output),
    }
    producer._write_json_staged(pending_meta, manifest)

    assert _require_new_output_pair(output, meta, overwrite=False) is True
    assert output.read_text() == "completed bank\n"
    assert json.loads(meta.read_text()) == manifest
    assert output.samefile(pending_output)
    assert meta.samefile(pending_meta)


def test_recovery_cli_does_not_require_search_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import backtest_chunk_trajectory as producer

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = producer._pending_output_path(output)
    pending_meta = producer._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": producer._SCHEMA,
        "complete": True,
        "output": producer._prepared_output_artifact(pending_output, output),
    }
    producer._write_json_staged(pending_meta, manifest)
    monkeypatch.setattr(
        sys,
        "argv",
        ["backtest_chunk_trajectory", "--recover-publication", "--out", str(output)],
    )

    producer.main()

    assert output.read_text() == "completed bank\n"
    assert json.loads(meta.read_text()) == manifest


def test_recovery_cli_runs_before_project_or_native_imports(tmp_path: Path) -> None:
    from scripts import backtest_chunk_trajectory as producer

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    pending_output = producer._pending_output_path(output)
    pending_meta = producer._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": producer._SCHEMA,
        "complete": True,
        "output": producer._prepared_output_artifact(pending_output, output),
    }
    producer._write_json_staged(pending_meta, manifest)

    shadow = tmp_path / "shadow"
    shadow_package = shadow / "chess_anti_engine"
    shadow_package.mkdir(parents=True)
    (shadow_package / "__init__.py").write_text(
        'raise RuntimeError("search runtime imported")\n'
    )
    repo_root = Path(producer.__file__).resolve().parents[1]
    python_path = [str(shadow), str(repo_root)]
    inherited_python_path = os.environ.get("PYTHONPATH")
    if inherited_python_path:
        python_path.append(inherited_python_path)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(python_path)}

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(producer.__file__).resolve()),
            "--recover-publication",
            "--out", str(output),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "search runtime imported" not in completed.stderr
    assert output.read_text() == "completed bank\n"
    assert json.loads(meta.read_text()) == manifest
    assert output.samefile(pending_output)
    assert meta.samefile(pending_meta)


def test_recovery_cli_rejects_unknown_options_without_publishing(tmp_path: Path) -> None:
    from scripts import backtest_chunk_trajectory as producer

    default_output = tmp_path / "runs/backtest/chunk_trajectory.jsonl"
    default_output.parent.mkdir(parents=True)
    default_meta = Path(str(default_output) + ".meta.json")
    pending_output = producer._pending_output_path(default_output)
    pending_meta = producer._pending_manifest_path(default_meta)
    pending_output.write_text("completed default bank\n")
    manifest = {
        "schema": producer._SCHEMA,
        "complete": True,
        "output": producer._prepared_output_artifact(pending_output, default_output),
    }
    producer._write_json_staged(pending_meta, manifest)
    intended_output = tmp_path / "intended/bank.jsonl"
    repo_root = Path(producer.__file__).resolve().parents[1]
    python_path = [str(repo_root)]
    inherited_python_path = os.environ.get("PYTHONPATH")
    if inherited_python_path:
        python_path.append(inherited_python_path)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(python_path)}

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(producer.__file__).resolve()),
            "--recover-publication",
            "--ouut", str(intended_output),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "unrecognized arguments: --ouut" in completed.stderr
    assert pending_output.read_text() == "completed default bank\n"
    assert json.loads(pending_meta.read_text()) == manifest
    assert not default_output.exists()
    assert not default_meta.exists()
    assert not intended_output.exists()


def test_import_never_dispatches_recovery_from_host_argv(tmp_path: Path) -> None:
    from scripts import backtest_chunk_trajectory as producer

    output = tmp_path / "bank.jsonl"
    meta = Path(str(output) + ".meta.json")
    pending_output = producer._pending_output_path(output)
    pending_meta = producer._pending_manifest_path(meta)
    pending_output.write_text("completed bank\n")
    manifest = {
        "schema": producer._SCHEMA,
        "complete": True,
        "output": producer._prepared_output_artifact(pending_output, output),
    }
    producer._write_json_staged(pending_meta, manifest)
    repo_root = Path(producer.__file__).resolve().parents[1]
    python_path = [str(repo_root)]
    inherited_python_path = os.environ.get("PYTHONPATH")
    if inherited_python_path:
        python_path.append(inherited_python_path)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(python_path)}
    program = (
        "import sys; "
        f"sys.argv = ['host', '--recover-publication', '--out', {str(output)!r}]; "
        "import scripts.backtest_chunk_trajectory"
    )

    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert pending_output.read_text() == "completed bank\n"
    assert json.loads(pending_meta.read_text()) == manifest
    assert not output.exists()
    assert not meta.exists()


def test_producer_refuses_unprepared_pending_bank(tmp_path: Path) -> None:
    from scripts import backtest_chunk_trajectory as producer

    output = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"
    producer._pending_output_path(output).write_text("partial bank\n")

    with pytest.raises(SystemExit, match="incomplete evidence"):
        _require_new_output_pair(output, meta, overwrite=False)


def test_git_ignored_output_guard_distinguishes_repository_paths(tmp_path: Path) -> None:
    from scripts import backtest_chunk_trajectory as producer

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("/ignored/\n")

    assert producer._git_ignored_or_outside(repo / "ignored/bank.jsonl", repo)
    assert not producer._git_ignored_or_outside(repo / "bank.jsonl", repo)
    assert producer._git_ignored_or_outside(tmp_path / "outside.jsonl", repo)


def test_producer_rejects_output_artifacts_that_would_dirty_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import backtest_chunk_trajectory as producer

    monkeypatch.setattr(
        producer.publication_module,
        "_git_ignored_or_outside",
        lambda _path, _root: False,
    )

    with pytest.raises(SystemExit, match="must be Git-ignored or outside"):
        _require_safe_output_paths(
            tmp_path / "bank.jsonl",
            tmp_path / "bank.jsonl.meta.json",
            protected_files=[],
            protected_directories=[],
        )


def test_producer_requires_enough_source_games_for_canonical_bootstrap() -> None:
    from scripts import backtest_chunk_trajectory as producer

    with pytest.raises(SystemExit, match="at least 9 distinct source games"):
        producer._require_analyzable_source_groups(
            {f"position-{game}": f"snapshot\0game-{game}" for game in range(8)},
            methodology_smoke=False,
        )

    producer._require_analyzable_source_groups(
        {f"position-{game}": f"snapshot\0game-{game}" for game in range(9)},
        methodology_smoke=False,
    )
    producer._require_analyzable_source_groups(
        {"a": None}, methodology_smoke=True,
    )


def test_producer_truncation_balances_source_blocks_and_morphology() -> None:
    from scripts import backtest_chunk_trajectory as producer

    opening = chess.Board().fen()
    endgame = "8/8/8/8/8/8/4K3/7k w - - 0 1"

    def position(source: int, index: int) -> AuditPosition:
        fen = opening if index % 2 == 0 else endgame
        pieces = chess.popcount(chess.Board(fen).occupied)
        return AuditPosition(
            key=f"source-{source}-position-{index}",
            fen=fen,
            phase=phase_bucket(pieces),
            source=source,
            move_cp={"a2a3": 0.0},
            best_cp=0.0,
            deep_wdl=(0.0, 1.0, 0.0),
            sf_nodes=1_000_000,
            sf_depth=1,
        )

    source_blocked = [
        position(source, index)
        for source in (0, 1)
        for index in range(24)
    ]
    selected, evidence = producer._select_audit_panel(source_blocked, 20)
    reverse_selected, reverse_evidence = producer._select_audit_panel(
        list(reversed(source_blocked)), 20,
    )

    assert [row["count"] for row in evidence["selected_source_counts"]] == [10, 10]
    assert sorted(row.source for row in selected) == [0] * 10 + [1] * 10
    assert evidence["source_balance"] == {
        "maximum_difference": 1, "observed_difference": 0, "passed": True,
    }
    assert evidence["selected_stratum_counts"] == [
        {"source": 0, "phase": 0, "piece_bucket": 0, "count": 5},
        {"source": 0, "phase": 2, "piece_bucket": 8, "count": 5},
        {"source": 1, "phase": 0, "piece_bucket": 0, "count": 5},
        {"source": 1, "phase": 2, "piece_bucket": 8, "count": 5},
    ]
    assert [row.key for row in reverse_selected] == [row.key for row in selected]
    assert reverse_evidence == evidence


def test_producer_panel_preserves_full_set_order_and_fails_closed_on_imbalance() -> None:
    from scripts import backtest_chunk_trajectory as producer

    board = chess.Board()
    positions = [
        AuditPosition(
            key=f"position-{index}",
            fen=board.fen(),
            phase=phase_bucket(chess.popcount(board.occupied)),
            source=0,
            move_cp={"a2a3": 0.0},
            best_cp=0.0,
            deep_wdl=(0.0, 1.0, 0.0),
            sf_nodes=1_000_000,
            sf_depth=1,
        )
        for index in range(9)
    ]

    selected, evidence = producer._select_audit_panel(positions, 0)

    assert selected is positions
    assert evidence["selection_mode"] == "full_set"
    assert evidence["source_balance"]["passed"] is False
    with pytest.raises(SystemExit, match="source-balanced audit panel"):
        producer._require_decision_grade_panel_selection(
            evidence, methodology_smoke=False,
        )
    producer._require_decision_grade_panel_selection(
        evidence, methodology_smoke=True,
    )


def test_producer_marks_post_collection_group_loss_non_decision_grade() -> None:
    from scripts import backtest_chunk_trajectory as producer

    requested_groups = {
        f"position-{game}": f"snapshot\0game-{game}" for game in range(9)
    }
    completed_groups = {
        f"snapshot\0game-{game}": f"snapshot\0game-{game}" for game in range(8)
    }

    producer._require_analyzable_source_groups(
        requested_groups, methodology_smoke=False,
    )
    assert producer._source_game_group_count(completed_groups) == 8
    assert producer._source_group_resolution_passed(completed_groups) is False


def test_excluded_position_evidence_preserves_partial_raw_snapshots() -> None:
    from scripts import backtest_chunk_trajectory as producer

    position = SimpleNamespace(
        key="position",
        fen=chess.Board().fen(),
        phase=1,
        source=2,
        best_cp=42,
        move_cp={"e2e4": 42, "d2d4": 31},
        sf_nodes=1_000_000,
        sf_depth=30,
    )
    snapshots = [{
        "nodes": 50,
        "actions": [1, 2],
        "visits": [30, 20],
        "child_q": [0.2, 0.1],
        "pv_actions": [1],
        "pv_uci": ["e2e4"],
    }]

    evidence = producer._excluded_position_evidence(
        position,
        source_dir="/snapshot",
        source_shard="s0.zarr",
        game_id=7,
        group_id="/snapshot\0" + "7",
        chunks_required=4,
        snapshots=snapshots,
        reason="incomplete_search",
        search_result={"nodes": 50},
    )

    assert evidence["chunks_observed"] == 1
    assert evidence["partial_observations"] == snapshots
    assert evidence["deep_reference_move_cp"] == {"e2e4": 42.0, "d2d4": 31.0}
    assert evidence["deep_reference_nodes"] == 1_000_000
    assert evidence["deep_reference_depth"] == 30
    assert evidence["deep_reference_scored_multipv"] == 2
    assert evidence["search_result"] == {"nodes": 50}


def test_post_collection_failure_preserves_complete_bank_with_invalid_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import backtest_chunk_trajectory as producer

    output = tmp_path / "bank.jsonl"
    pending_output = producer._pending_output_path(output)
    meta = Path(str(output) + ".meta.json")
    pending_meta = producer._pending_manifest_path(meta)
    pending_output.write_text("row one\nrow two\n")
    censoring = controller_module._reference_censoring_summary([])

    def fail_after_collection() -> None:
        retained_output_fd = os.open(pending_output, os.O_RDONLY | os.O_CLOEXEC)
        retained_output_parent_fd = os.open(
            pending_output.parent,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
        )
        producer._ACTIVE_PENDING_EVIDENCE = {
            "collection_complete": True,
            "pending_output": pending_output,
            "output": output,
            "manifest": meta,
            "pending_manifest": pending_meta,
            "output_artifact": producer._prepared_output_artifact(
                pending_output, output,
            ),
            "retained_output_fd": retained_output_fd,
            "retained_output_parent_fd": retained_output_parent_fd,
            "output_locks": (),
            "provenance": {"schema": producer._SCHEMA},
            "row_count": 2,
            "position_count": 1,
            "requested_position_count": 1,
            "requested_max_positions": 1,
            "excluded_positions": [],
            "reference_censoring_details": [],
        }
        raise RuntimeError("final provenance snapshot failed")

    monkeypatch.setattr(producer, "_main", fail_after_collection)

    with pytest.raises(RuntimeError, match="final provenance snapshot failed"):
        producer.main()

    assert pending_output.read_text() == "row one\nrow two\n"
    assert not pending_meta.exists()
    invalid = producer._invalid_manifest_path(meta)
    manifest = json.loads(invalid.read_text())["diagnostic"]
    assert manifest["decision_grade"] is False
    assert manifest["complete"] is False
    assert manifest["trajectory_collection_complete"] is True
    assert manifest["failure_stage"] == "post_collection_finalization"
    assert manifest["finalization_error"] == {
        "type": "RuntimeError",
        "message": "final provenance snapshot failed",
    }
    assert manifest["raw_observations_preserved"] is True
    assert manifest["reference_censoring"] == censoring
    assert manifest["output"] == producer._prepared_output_artifact(
        pending_output, output,
    )


def test_post_collection_diagnostic_substitution_cannot_seed_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import backtest_chunk_trajectory as producer
    from scripts import chunk_trajectory_publication as publication

    output = tmp_path / "bank.jsonl"
    pending_output = producer._pending_output_path(output)
    meta = Path(str(output) + ".meta.json")
    pending_meta = producer._pending_manifest_path(meta)
    invalid = producer._invalid_manifest_path(meta)
    pending_output.write_text("row one\nrow two\n")
    output_artifact = producer._prepared_output_artifact(pending_output, output)
    attacker_manifest = {
        "schema": producer._SCHEMA,
        "complete": True,
        "tampered": True,
        "output": output_artifact,
    }

    def fail_after_collection() -> None:
        retained_output_fd = os.open(pending_output, os.O_RDONLY | os.O_CLOEXEC)
        retained_output_parent_fd = os.open(
            pending_output.parent,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
        )
        producer._ACTIVE_PENDING_EVIDENCE = {
            "collection_complete": True,
            "pending_output": pending_output,
            "output": output,
            "manifest": meta,
            "pending_manifest": pending_meta,
            "output_artifact": output_artifact,
            "retained_output_fd": retained_output_fd,
            "retained_output_parent_fd": retained_output_parent_fd,
            "output_locks": (),
            "provenance": {"schema": producer._SCHEMA},
            "row_count": 2,
            "position_count": 1,
            "requested_position_count": 1,
            "requested_max_positions": 1,
            "excluded_positions": [],
            "reference_censoring_details": [],
        }
        raise RuntimeError("final provenance snapshot failed")

    real_fsync = publication.os.fsync
    replaced = False

    def replace_diagnostic_after_file_sync(descriptor: int) -> None:
        nonlocal replaced
        descriptor_stat = os.fstat(descriptor)
        if (
            not replaced
            and stat.S_ISREG(descriptor_stat.st_mode)
            and invalid.exists()
            and (descriptor_stat.st_dev, descriptor_stat.st_ino)
            == (invalid.stat().st_dev, invalid.stat().st_ino)
        ):
            real_fsync(descriptor)
            invalid.unlink()
            invalid.write_text(json.dumps(attacker_manifest))
            replaced = True
            return
        real_fsync(descriptor)

    monkeypatch.setattr(producer, "_main", fail_after_collection)
    monkeypatch.setattr(publication.os, "fsync", replace_diagnostic_after_file_sync)
    with pytest.raises(RuntimeError, match="final provenance snapshot failed"):
        producer.main()

    monkeypatch.setattr(publication.os, "fsync", real_fsync)
    assert replaced is True
    assert pending_output.read_text() == "row one\nrow two\n"
    assert not pending_meta.exists()
    assert json.loads(invalid.read_text()) == attacker_manifest
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert not output.exists()


def test_post_collection_stat_error_cannot_seed_pending_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import backtest_chunk_trajectory as producer
    from scripts import chunk_trajectory_publication as publication

    output = tmp_path / "bank.jsonl"
    pending_output = producer._pending_output_path(output)
    meta = Path(str(output) + ".meta.json")
    pending_meta = producer._pending_manifest_path(meta)
    pending_output.write_text("row one\nrow two\n")
    output_artifact = producer._prepared_output_artifact(pending_output, output)
    attacker_manifest = {
        "schema": producer._SCHEMA,
        "complete": True,
        "attacker": True,
        "output": output_artifact,
    }

    def fail_after_collection() -> None:
        retained_output_fd = os.open(pending_output, os.O_RDONLY | os.O_CLOEXEC)
        retained_output_parent_fd = os.open(
            pending_output.parent,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
        )
        producer._ACTIVE_PENDING_EVIDENCE = {
            "collection_complete": True,
            "pending_output": pending_output,
            "output": output,
            "manifest": meta,
            "pending_manifest": pending_meta,
            "output_artifact": output_artifact,
            "retained_output_fd": retained_output_fd,
            "retained_output_parent_fd": retained_output_parent_fd,
            "output_locks": (),
            "provenance": {"schema": producer._SCHEMA},
            "row_count": 2,
            "position_count": 1,
            "requested_position_count": 1,
            "requested_max_positions": 1,
            "excluded_positions": [],
            "reference_censoring_details": [],
        }
        raise RuntimeError("final provenance snapshot failed")

    real_stat = publication.os.stat
    bank_stats = 0
    injected = False

    def create_manifest_during_diagnostic(
        path: Any, *args: Any, **kwargs: Any,
    ) -> os.stat_result:
        nonlocal bank_stats, injected
        if path == pending_output.name and kwargs.get("dir_fd") is not None:
            bank_stats += 1
            if bank_stats == 2:
                injected = True
                pending_meta.write_text(json.dumps(attacker_manifest))
                raise OSError(errno.EIO, "post-collection bank stat failed")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(producer, "_main", fail_after_collection)
    monkeypatch.setattr(publication.os, "stat", create_manifest_during_diagnostic)
    with pytest.raises(RuntimeError, match="final provenance snapshot failed"):
        producer.main()

    monkeypatch.setattr(publication.os, "stat", real_stat)
    assert injected is True
    assert publication._invalid_manifest_path(meta).exists()
    assert json.loads(pending_meta.read_text()) == attacker_manifest
    with pytest.raises(SystemExit, match="manifest recovery was invalidated"):
        publication._require_new_output_pair(output, meta, overwrite=False)
    assert not output.exists()


def test_producer_rejects_foreign_loaded_python_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import backtest_chunk_trajectory as producer

    foreign = tmp_path / "foreign_search.py"
    foreign.write_text("FOREIGN = True\n")
    monkeypatch.setitem(
        sys.modules,
        "chess_anti_engine.foreign_review_fixture",
        SimpleNamespace(__file__=str(foreign)),
    )
    repo_root = Path(producer.__file__).resolve().parents[1]
    monkeypatch.setattr(
        producer,
        "_producer_git_file_at_commit",
        lambda _commit, relative: (
            (repo_root / relative).read_bytes()
            if (repo_root / relative).is_file() else None
        ),
    )

    with pytest.raises(
        SystemExit, match=r"chess_anti_engine\.foreign_review_fixture",
    ):
        producer._producer_python_source_artifacts("a" * 40, require_tracked=True)


@pytest.mark.parametrize("module_name", controller_module._NATIVE_MODULES)
def test_producer_rejects_native_binary_built_from_another_revision(
    module_name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import backtest_chunk_trajectory as producer

    repo_root = Path(producer.__file__).resolve().parents[1]
    dependencies = {
        relative_path: (repo_root / relative_path).read_bytes()
        for relative_path in extension_spec(module_name).dependencies
    }
    copied = controller_module.native_build_attestation(
        module_name, "b" * 40, dependencies,
    )
    loaded = SimpleNamespace(
        BUILD_ATTESTATION_SCHEMA=copied["schema"],
        BUILD_MODULE_NAME=copied["module"],
        BUILD_SOURCE_GIT_SHA=copied["source_git_sha"],
        BUILD_INPUT_SHA256=copied["input_sha256"],
    )
    monkeypatch.setattr(
        producer,
        "_producer_git_file_at_commit",
        lambda _commit, relative: dependencies.get(relative),
    )

    observed = producer._loaded_native_build_attestation(
        loaded, module_name, "a" * 40,
    )

    assert observed["current_inputs_match_revision"] is True
    assert observed["matches_producer_revision"] is False


@pytest.mark.parametrize("module_name", controller_module._NATIVE_MODULES)
def test_producer_rejects_native_binary_built_from_altered_then_restored_inputs(
    module_name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import backtest_chunk_trajectory as producer

    repo_root = Path(producer.__file__).resolve().parents[1]
    dependencies = {
        relative_path: (repo_root / relative_path).read_bytes()
        for relative_path in extension_spec(module_name).dependencies
    }
    built_from = dict(dependencies)
    changed_path = next(iter(built_from))
    built_from[changed_path] += b" locally modified for build"
    altered = controller_module.native_build_attestation(
        module_name, "a" * 40, built_from,
    )
    loaded = SimpleNamespace(
        BUILD_ATTESTATION_SCHEMA=altered["schema"],
        BUILD_MODULE_NAME=altered["module"],
        BUILD_SOURCE_GIT_SHA=altered["source_git_sha"],
        BUILD_INPUT_SHA256=altered["input_sha256"],
    )
    monkeypatch.setattr(
        producer,
        "_producer_git_file_at_commit",
        lambda _commit, relative: dependencies.get(relative),
    )

    observed = producer._loaded_native_build_attestation(
        loaded, module_name, "a" * 40,
    )

    assert observed["current_inputs_match_revision"] is True
    assert observed["matches_producer_revision"] is False


@pytest.mark.parametrize("module_name", controller_module._NATIVE_MODULES)
def test_producer_rejects_missing_native_build_attestation(
    module_name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import backtest_chunk_trajectory as producer

    repo_root = Path(producer.__file__).resolve().parents[1]
    dependencies = {
        relative_path: (repo_root / relative_path).read_bytes()
        for relative_path in extension_spec(module_name).dependencies
    }
    monkeypatch.setattr(
        producer,
        "_producer_git_file_at_commit",
        lambda _commit, relative: dependencies.get(relative),
    )

    observed = producer._loaded_native_build_attestation(
        SimpleNamespace(), module_name, "a" * 40,
    )

    assert observed["matches_producer_revision"] is False


def test_real_producer_source_inventory_round_trips_through_analyzer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import backtest_chunk_trajectory as producer

    importlib.import_module("chess_anti_engine.uci.model_loader")
    repo_root = Path(producer.__file__).resolve().parents[1]

    def tracked_bytes(_commit: str, relative: str) -> bytes | None:
        path = repo_root / relative
        return path.read_bytes() if path.is_file() else None

    monkeypatch.setattr(producer, "_producer_git_file_at_commit", tracked_bytes)
    monkeypatch.setattr(controller_module, "_git_file_at_commit", tracked_bytes)
    monkeypatch.setattr(
        producer, "_SOURCE_ONLY_IMPORT_GUARD",
        SimpleNamespace(
            verified_modules={},
            verified_native_modules={},
            module_verified=lambda _name, _relative: True,
            status=lambda: {
                "active": True,
                "first_finder": True,
                "loaded_project_modules": {
                    "passed": True, "unverified_modules": [],
                },
            },
        ),
    )

    sources = producer._producer_python_source_artifacts(
        "a" * 40, require_tracked=True,
    )

    assert sources["scripts"]["repo_relative_path"] == "scripts/__init__.py"
    assert sources["scripts"]["size"] == 0
    assert sources["scripts.approved_syzygy"]["repo_relative_path"] == (
        "scripts/approved_syzygy.py"
    )
    assert controller_module._producer_sources_match_revision(sources, "a" * 40)


def test_producer_requires_driver_provenance_before_decision_grade_search() -> None:
    from scripts import backtest_chunk_trajectory as producer

    with pytest.raises(RuntimeError, match="NVIDIA driver provenance"):
        producer._require_nvidia_driver_provenance(None, methodology_smoke=False)
    producer._require_nvidia_driver_provenance("600.1", methodology_smoke=False)
    producer._require_nvidia_driver_provenance(None, methodology_smoke=True)


@pytest.mark.parametrize(
    "name",
    [
        ".bank.jsonl.lock",
        ".bank.jsonl.tmp-12345",
        "..tmp-bank.tmp-12345",
        ".bank.meta.json.invalid-recovery",
    ],
)
def test_producer_output_rejects_internal_output_namespaces(
    tmp_path: Path, name: str,
) -> None:
    with pytest.raises(SystemExit, match="lock/staging namespace"):
        _require_safe_output_paths(
            tmp_path / name,
            tmp_path / "out.meta.json",
            protected_files=[],
            protected_directories=[],
        )


def test_analyzer_output_cannot_replace_producer_staging_file(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = tmp_path / "bank.jsonl.meta.json"

    with pytest.raises(ValueError, match="lock/staging namespace"):
        _require_safe_output_path(bank, meta, tmp_path / ".bank.jsonl.tmp-12345")


def test_hidden_output_staging_name_is_reserved(tmp_path: Path) -> None:
    from scripts.repo_output_guard import reserved_output_path

    output = tmp_path / ".tmp-bank"
    staging = output.with_name(f".{output.name}.tmp-12345")

    assert staging.name == "..tmp-bank.tmp-12345"
    assert reserved_output_path(staging)


def test_internal_namespace_guard_checks_symlink_lexical_name(tmp_path: Path) -> None:
    target = tmp_path / "ordinary-name"
    target.write_text("existing\n")
    alias = tmp_path / ".bank.jsonl.tmp-12345"
    alias.symlink_to(target)

    with pytest.raises(SystemExit, match="lock/staging namespace"):
        _require_safe_output_paths(
            alias,
            tmp_path / "out.meta.json",
            protected_files=[],
            protected_directories=[],
        )
    with pytest.raises(ValueError, match="lock/staging namespace"):
        _require_safe_output_path(
            tmp_path / "bank.jsonl",
            tmp_path / "bank.jsonl.meta.json",
            alias,
        )


def test_publish_output_detects_replacement_before_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import chunk_trajectory_publication as publication

    private = tmp_path / ".bank.tmp"
    output = tmp_path / "bank.jsonl"
    private.write_text("producer bytes\n")
    real_link = publication._link_open_file

    def replacing_publish(file_fd: int, parent_fd: int, name: str) -> None:
        real_link(file_fd, parent_fd, name)
        replacement_fd = os.open(name, os.O_WRONLY | os.O_TRUNC, dir_fd=parent_fd)
        try:
            os.write(replacement_fd, b"other producer bytes\n")
        finally:
            os.close(replacement_fd)

    monkeypatch.setattr(publication, "_link_open_file", replacing_publish)

    with pytest.raises((RuntimeError, SystemExit), match=r"changed|differs"):
        _publish_output(private, output)


def test_publish_output_never_clobbers_a_destination_that_appeared(
    tmp_path: Path,
) -> None:
    private = tmp_path / ".bank.tmp"
    output = tmp_path / "bank.jsonl"
    private.write_text("new bank\n")
    output.write_text("existing bank\n")

    with pytest.raises(RuntimeError, match="immutable evidence"):
        _publish_output(private, output)

    assert output.read_text() == "existing bank\n"
    assert private.read_text() == "new bank\n"


def test_manifest_publication_never_clobbers_an_existing_manifest(
    tmp_path: Path,
) -> None:
    from scripts import backtest_chunk_trajectory as producer

    manifest = tmp_path / "bank.jsonl.meta.json"
    manifest.write_text("old manifest\n")

    with pytest.raises(RuntimeError, match="immutable evidence"):
        producer._write_json_atomic(manifest, {"new": True})

    assert manifest.read_text() == "old manifest\n"


def test_producer_applies_shared_uci_search_ranges() -> None:
    _validate_registry_search_values(
        "walker_puct", {"chunk_sims": 32, "c_puct": 1.75},
    )
    with pytest.raises(SystemExit, match="chunk_sims"):
        _validate_registry_search_values(
            "walker_puct", {"chunk_sims": 1, "c_puct": 1.75},
        )


def test_producer_rejects_chunk_size_before_checkpoint_or_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import backtest_chunk_trajectory as producer

    monkeypatch.setattr(
        sys,
        "argv",
        ["backtest_chunk_trajectory", "--checkpoint", "unused", "--chunk-sims", "1"],
    )
    monkeypatch.setattr(
        producer,
        "_checkpoint_file",
        lambda _path: pytest.fail("checkpoint resolution ran before range validation"),
    )

    with pytest.raises(SystemExit, match="production UCI registry"):
        producer.main()


def test_decision_grade_requires_enough_bootstrap_samples() -> None:
    _require_bootstrap_resolution(1, methodology_smoke=True)
    _require_bootstrap_resolution(1000, methodology_smoke=False)
    with pytest.raises(ValueError, match="at least 1000"):
        _require_bootstrap_resolution(999, methodology_smoke=False)


def test_evidence_verdict_is_scoped_to_authenticated_live_evidence(
    tmp_path: Path,
) -> None:
    from scripts.analyze_chunk_controller import _evidence_verdict, _is_canonical_decision_rule

    assert _is_canonical_decision_rule(
        n_folds=5, bootstrap_samples=2000, seed=0,
        allocation_fraction=0.2, min_capture_gain=0.05,
        min_oracle_headroom=1e-4, min_bootstrap_valid_fraction=0.95,
    ) is True
    assert _is_canonical_decision_rule(
        n_folds=5, bootstrap_samples=2000, seed=1,
        allocation_fraction=0.2, min_capture_gain=0.05,
        min_oracle_headroom=1e-4, min_bootstrap_valid_fraction=0.95,
    ) is False

    analyzer = {"decision_grade": True}
    assert _evidence_verdict(
        bank_decision_grade=True,
        analyzer_provenance=analyzer,
        evidence_guard=None,
        canonical_preregistered_rule=True,
        source_group_resolution_passed=False,
        statistical_gate_passed=True,
    ) == "METHODOLOGY_SMOKE_ONLY"

    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    with controller_module._retained_decision_grade_evidence(
        bank, meta,
    ) as evidence_guard:
        assert _evidence_verdict(
            bank_decision_grade=True,
            analyzer_provenance=analyzer,
            evidence_guard=evidence_guard,
            canonical_preregistered_rule=True,
            source_group_resolution_passed=True,
            statistical_gate_passed=True,
        ) == "ADVANCE_TO_CLOCK_HISTORY_REUSED_TREE_BANK"
        assert _evidence_verdict(
            bank_decision_grade=True,
            analyzer_provenance=analyzer,
            evidence_guard=evidence_guard,
            canonical_preregistered_rule=True,
            source_group_resolution_passed=True,
            statistical_gate_passed=False,
        ) == "NO_ADVANCE_FROM_FRESH_TREE_FIXED_NODE_SCREEN"
        assert _evidence_verdict(
            bank_decision_grade=True,
            analyzer_provenance=analyzer,
            evidence_guard=evidence_guard,
            canonical_preregistered_rule=False,
            source_group_resolution_passed=False,
            statistical_gate_passed=True,
        ) == "NONCANONICAL_RULE_DIAGNOSTIC_ONLY"
        assert _evidence_verdict(
            bank_decision_grade=True,
            analyzer_provenance=analyzer,
            evidence_guard=evidence_guard,
            canonical_preregistered_rule=True,
            source_group_resolution_passed=False,
            statistical_gate_passed=False,
        ) == "INSUFFICIENT_SOURCE_GAME_GROUPS"
    assert _evidence_verdict(
        bank_decision_grade=True,
        analyzer_provenance=analyzer,
        evidence_guard=evidence_guard,
        canonical_preregistered_rule=True,
        source_group_resolution_passed=True,
        statistical_gate_passed=True,
    ) == "METHODOLOGY_SMOKE_ONLY"


def test_analyzer_main_skips_grouped_analysis_for_undersized_bank(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    one_game = [_transition("position", 1, 100, 0.0, current=True)]
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    info = {
        "decision_grade": True,
        "preregistered_design": True,
        "manifest": {"producer_git_sha": "a" * 40},
    }
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_chunk_controller",
            "--in", str(bank),
            "--meta", str(meta),
        ],
    )
    monkeypatch.setattr(controller_module, "_analyzer_source_artifacts", dict)
    monkeypatch.setattr(controller_module, "_git_state", lambda: ("b" * 40, False))
    monkeypatch.setattr(
        controller_module, "_PREIMPORT_PYTHON_SOURCES", {"passed": True},
    )
    monkeypatch.setattr(
        controller_module, "_preimport_python_surface_status",
        lambda _snapshot: {"passed": True},
    )
    monkeypatch.setattr(controller_module, "_require_safe_output_path", lambda *_a, **_k: None)
    monkeypatch.setattr(
        controller_module,
        "load_transitions",
        lambda *_a, **_k: (one_game, info),
    )
    monkeypatch.setattr(
        controller_module,
        "analyze",
        lambda *_a, **_k: pytest.fail("undersized bank entered grouped analysis"),
    )
    monkeypatch.setattr(
        controller_module,
        "_analyzer_provenance",
        lambda *_a, **_k: {
            "decision_grade": True,
            "git_sha": "b" * 40,
            "final_git_sha": "b" * 40,
        },
    )

    controller_module.main()

    payload = json.loads(capsys.readouterr().out)
    analysis = payload["analysis"]
    assert analysis["verdict"] == "INSUFFICIENT_SOURCE_GAME_GROUPS"
    assert analysis["analysis_skipped"] == "insufficient_source_game_groups"
    assert analysis["source_game_group_count"] == 1
    assert analysis["grouped_analysis_possible"] is False
    assert analysis["source_group_resolution_passed"] is False
    assert analysis["evidence_decision_grade"] is False


def test_analyzer_main_revalidates_witnesses_after_report_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    report = tmp_path / "analysis.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_chunk_controller",
            "--in", str(bank),
            "--meta", str(meta),
            "--out", str(report),
        ],
    )
    monkeypatch.setattr(controller_module, "_analyzer_source_artifacts", dict)
    monkeypatch.setattr(controller_module, "_git_state", lambda: ("b" * 40, False))
    monkeypatch.setattr(
        controller_module, "_PREIMPORT_PYTHON_SOURCES", {"passed": True},
    )
    monkeypatch.setattr(
        controller_module,
        "_preimport_python_surface_status",
        lambda _snapshot: {"passed": True},
    )
    monkeypatch.setattr(
        controller_module,
        "_analyzer_provenance",
        lambda *_a, **_k: {
            "decision_grade": True,
            "git_sha": "b" * 40,
            "final_git_sha": "b" * 40,
        },
    )
    real_write = controller_module._write_json_atomic

    def write_report_then_change_bank(
        path: controller_module._AnchoredOutputTarget,
        payload: str,
        *,
        evidence_check: Any = None,
    ) -> None:
        real_write(path, payload, evidence_check=evidence_check)
        bank.write_text('{"tampered":true}\n')

    monkeypatch.setattr(
        controller_module, "_write_json_atomic", write_report_then_change_bank,
    )

    with pytest.raises(SystemExit, match="changed"):
        controller_module.main()

    assert report.exists()


def test_analyzer_main_runs_grouped_analysis_for_two_game_smoke_bank(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    two_games = [
        _transition(key, game, horizon, float(game), current=True)
        for key, game in (("first", 1), ("second", 2))
        for horizon in (100, 150, 200)
    ]
    info = {
        "decision_grade": False,
        "preregistered_design": False,
        "manifest": {"producer_git_sha": "a" * 40},
    }
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_chunk_controller",
            "--in", str(tmp_path / "bank.jsonl"),
            "--meta", str(tmp_path / "bank.jsonl.meta.json"),
            "--methodology-smoke",
            "--folds", "2",
            "--bootstrap-samples", "1",
            "--allocation-fraction", "0.5",
        ],
    )
    monkeypatch.setattr(controller_module, "_analyzer_source_artifacts", dict)
    monkeypatch.setattr(controller_module, "_git_state", lambda: ("b" * 40, False))
    monkeypatch.setattr(controller_module, "_require_safe_output_path", lambda *_a, **_k: None)
    monkeypatch.setattr(
        controller_module,
        "load_transitions",
        lambda *_a, **_k: (two_games, info),
    )
    monkeypatch.setattr(
        controller_module,
        "_analyzer_provenance",
        lambda *_a, **_k: {
            "decision_grade": True,
            "git_sha": "b" * 40,
            "final_git_sha": "b" * 40,
        },
    )

    controller_module.main()

    analysis = json.loads(capsys.readouterr().out)["analysis"]
    assert analysis["verdict"] == "METHODOLOGY_SMOKE_ONLY"
    assert "reachable_rollout" in analysis
    assert analysis["source_game_group_count"] == 2
    assert analysis["grouped_analysis_possible"] is True
    assert analysis["source_group_resolution_passed"] is False


def test_grouped_analysis_requires_multiple_held_horizon_training_rows() -> None:
    two_games_one_horizon = [
        _transition("first", 1, 100, 0.0, current=True),
        _transition("second", 2, 100, 0.0, current=True),
    ]

    assert controller_module._grouped_analysis_possible(
        two_games_one_horizon, 2,
    ) is False


def test_analyzer_main_reports_nonrectangular_smoke_bank_without_entering_rollout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    uneven = [
        _transition(key, game, horizon, float(game), current=True)
        for key, game, horizons in (
            ("first", 1, (100, 150, 200)),
            ("second", 2, (100, 150, 200)),
            ("short", 3, (100, 150)),
        )
        for horizon in horizons
    ]
    info = {
        "decision_grade": False,
        "preregistered_design": False,
        "manifest": {"producer_git_sha": "a" * 40},
    }
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_chunk_controller",
            "--in", str(tmp_path / "bank.jsonl"),
            "--meta", str(tmp_path / "bank.jsonl.meta.json"),
            "--methodology-smoke",
            "--folds", "2",
            "--bootstrap-samples", "1",
            "--allocation-fraction", "0.5",
        ],
    )
    monkeypatch.setattr(controller_module, "_analyzer_source_artifacts", dict)
    monkeypatch.setattr(controller_module, "_git_state", lambda: ("b" * 40, False))
    monkeypatch.setattr(controller_module, "_require_safe_output_path", lambda *_a, **_k: None)
    monkeypatch.setattr(
        controller_module,
        "load_transitions",
        lambda *_a, **_k: (uneven, info),
    )
    monkeypatch.setattr(
        controller_module,
        "analyze",
        lambda *_a, **_k: pytest.fail("nonrectangular bank entered grouped rollout"),
    )
    monkeypatch.setattr(
        controller_module,
        "_analyzer_provenance",
        lambda *_a, **_k: {
            "decision_grade": True,
            "git_sha": "b" * 40,
            "final_git_sha": "b" * 40,
        },
    )

    controller_module.main()

    analysis = json.loads(capsys.readouterr().out)["analysis"]
    assert analysis["analysis_skipped"] == "nonrectangular_key_by_horizon_layout"
    assert analysis["grouped_analysis_possible"] is False
    assert analysis["grouped_analysis_preflight"] == {
        "passed": False,
        "reasons": ["nonrectangular_key_by_horizon_layout"],
        "source_game_group_count": 3,
        "horizon_count": 3,
    }


def test_analyze_cannot_advance_with_an_undersampled_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import analyze_chunk_controller as controller

    rows = [
        _transition(key, game, horizon, gain, current=gain > 0.0)
        for key, game, gain in (
            ("a", 1, -1.0),
            ("b", 2, -1.0),
            ("c", 3, 1.0),
            ("d", 4, 1.0),
        )
        for horizon in (100, 150)
    ]
    m0 = np.zeros(len(rows), dtype=np.float64)
    m1 = np.asarray([row.gain for row in rows], dtype=np.float64)

    def fake_predictions(
        transitions: list[Transition], model: str, *, n_folds: int = 5,
    ) -> tuple[np.ndarray, list[dict[str, object]]]:
        del transitions, n_folds
        return (m0 if model == "M0" else m1), []

    monkeypatch.setattr(controller, "held_horizon_predictions", fake_predictions)
    monkeypatch.setattr(
        controller,
        "cluster_bootstrap_delta",
        lambda *_args, **_kwargs: {
            "requested_samples": 1, "valid_samples": 1, "valid_fraction": 1.0,
            "mean": 1.0, "lower_95": 1.0, "upper_95": 1.0,
        },
    )

    result = analyze(
        rows,
        n_folds=2,
        bootstrap_samples=1,
        seed=0,
        allocation_fraction=0.5,
        min_capture_gain=0.0,
        min_oracle_headroom=0.0,
        min_bootstrap_valid_fraction=0.5,
    )

    assert result["bootstrap_resolution_passed"] is False
    assert result["statistical_gate_passed"] is False
    assert "verdict" not in result

    positive = analyze(
        rows,
        n_folds=2,
        bootstrap_samples=1000,
        seed=0,
        allocation_fraction=0.5,
        min_capture_gain=0.0,
        min_oracle_headroom=0.0,
        min_bootstrap_valid_fraction=0.5,
    )

    assert positive["statistical_gate_passed"] is True
    assert positive["evaluated_rule"]["grouped_cv_folds"] == 2
    assert positive["evaluated_rule"]["bootstrap_seed"] == 0
    assert "verdict" not in positive


def test_analyze_requires_M1_improvement_at_every_reachable_rung(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import analyze_chunk_controller as controller

    rows = [
        _transition("a", 1, 100, 0.0, current=False),
        _transition("a", 1, 150, 0.0, current=False),
        _transition("b", 2, 100, 0.0, current=False),
        _transition("b", 2, 150, 0.0, current=False),
        _transition("c", 3, 100, 0.0, current=False),
        _transition("c", 3, 150, -1.0, current=True),
        _transition("d", 4, 100, 10.0, current=True),
        _transition("d", 4, 150, 0.0, current=False),
    ]
    m0 = np.zeros(len(rows), dtype=np.float64)
    m1 = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 10.0, 0.0])

    monkeypatch.setattr(
        controller,
        "held_horizon_predictions",
        lambda _rows, model, *, n_folds=5: (m0 if model == "M0" else m1, []),
    )
    monkeypatch.setattr(
        controller,
        "cluster_bootstrap_delta",
        lambda *_args, **_kwargs: {
            "requested_samples": 1000, "valid_samples": 1000,
            "valid_fraction": 1.0, "mean": 1.0, "lower_95": 1.0,
            "upper_95": 1.0,
        },
    )

    result = analyze(
        rows, n_folds=2, bootstrap_samples=1000, seed=4,
        allocation_fraction=0.5, min_capture_gain=0.0,
        min_oracle_headroom=0.0, min_bootstrap_valid_fraction=0.5,
    )

    assert result["M1_minus_M0_oracle_capture"] > 0.0
    assert result["reachable_stage_rule_passed"] is False
    assert result["statistical_gate_passed"] is False


def test_analyzer_provenance_requires_a_stable_clean_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import analyze_chunk_controller as controller

    repo_root = Path(controller.__file__).resolve().parents[1]
    sources = {
        name: artifact
        for name, artifact in controller._analyzer_source_artifacts().items()
        if (
            name == "analyzer"
            or (
                isinstance(artifact.get("path"), str)
                and str(artifact["path"]).endswith(".py")
            )
        )
    }
    for artifact in sources.values():
        artifact["matches_preimport_snapshot"] = True
        artifact["source_only_import_verified"] = True
    monkeypatch.setattr(controller, "_analyzer_source_artifacts", lambda: sources)
    monkeypatch.setattr(controller, "_git_state", lambda: ("b" * 40, False))
    monkeypatch.setattr(
        controller, "_PREIMPORT_PYTHON_SOURCES",
        {"passed": True, "git_sha": "b" * 40},
    )
    monkeypatch.setattr(
        controller, "_preimport_python_surface_status",
        lambda _snapshot: {"passed": True, "changed": [], "git_sha": "b" * 40},
    )
    monkeypatch.setattr(
        controller, "_SOURCE_ONLY_IMPORT_GUARD",
        SimpleNamespace(status=lambda: {
            "schema": "deepfin.source_only_import.v2",
            "active": True,
            "installed": True,
            "first_finder": True,
            "git_sha": "b" * 40,
            "tracked_python_surface_sha256": None,
            "project_scope": ["chess_anti_engine", "scripts"],
            "execution": "compile_authenticated_source_bytes",
            "bytecode_cache_reads": False,
            "native_extension_loading": (
                "default_deny_exact_preimport_artifact_authenticated_loader"
            ),
            "permitted_native_modules": list(controller._NATIVE_MODULES),
            "authorized_native_modules": [],
            "authorized_native_artifacts": {},
            "verified_native_modules": {},
            "loaded_project_modules": {
                "passed": True, "loaded_modules": sorted(sources),
                "unverified_modules": [],
            },
            "failures": [],
        }),
    )
    monkeypatch.setattr(
        controller,
        "_git_file_at_commit",
        lambda _commit, relative_path: (repo_root / relative_path).read_bytes(),
    )

    clean = controller._analyzer_provenance(sources, "b" * 40, False)
    dirty = controller._analyzer_provenance(sources, "b" * 40, True)

    assert clean["decision_grade"] is True
    assert clean["sources_match_git_revision"] is True
    assert clean["numpy_version"]
    assert clean["python_chess_version"]
    assert dirty["decision_grade"] is False


def test_analyzer_provenance_rejects_helper_from_another_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import analyze_chunk_controller as controller

    repo_root = Path(controller.__file__).resolve().parents[1]
    sources = {
        name: artifact
        for name, artifact in controller._analyzer_source_artifacts().items()
        if (
            name == "analyzer"
            or (
                isinstance(artifact.get("path"), str)
                and str(artifact["path"]).endswith(".py")
            )
        )
    }
    for artifact in sources.values():
        artifact["matches_preimport_snapshot"] = True
        artifact["source_only_import_verified"] = True
    foreign_oracle = tmp_path / "reachable_oracle.py"
    foreign_oracle.write_bytes(
        Path(controller.solve_reachable_oracle.__code__.co_filename).read_bytes()
    )
    sources["scripts.reachable_oracle"] = controller._artifact_snapshot(foreign_oracle)
    monkeypatch.setattr(controller, "_analyzer_source_artifacts", lambda: sources)
    monkeypatch.setattr(controller, "_git_state", lambda: ("b" * 40, False))
    monkeypatch.setattr(
        controller, "_PREIMPORT_PYTHON_SOURCES",
        {"passed": True, "git_sha": "b" * 40},
    )
    monkeypatch.setattr(
        controller, "_preimport_python_surface_status",
        lambda _snapshot: {"passed": True, "changed": [], "git_sha": "b" * 40},
    )
    monkeypatch.setattr(
        controller, "_SOURCE_ONLY_IMPORT_GUARD",
        SimpleNamespace(status=lambda: {
            "schema": "deepfin.source_only_import.v2",
            "active": True,
            "installed": True,
            "first_finder": True,
            "git_sha": "b" * 40,
            "tracked_python_surface_sha256": None,
            "project_scope": ["chess_anti_engine", "scripts"],
            "execution": "compile_authenticated_source_bytes",
            "bytecode_cache_reads": False,
            "native_extension_loading": (
                "default_deny_exact_preimport_artifact_authenticated_loader"
            ),
            "permitted_native_modules": list(controller._NATIVE_MODULES),
            "authorized_native_modules": [],
            "authorized_native_artifacts": {},
            "verified_native_modules": {},
            "loaded_project_modules": {
                "passed": True, "loaded_modules": sorted(sources),
                "unverified_modules": [],
            },
            "failures": [],
        }),
    )
    monkeypatch.setattr(
        controller,
        "_git_file_at_commit",
        lambda _commit, relative_path: (repo_root / relative_path).read_bytes(),
    )

    provenance = controller._analyzer_provenance(sources, "b" * 40, False)

    assert provenance["sources_stable"] is True
    assert provenance["sources_match_git_revision"] is False
    assert provenance["decision_grade"] is False
    assert provenance["source_revision_bindings"]["scripts.reachable_oracle"] == {
        "repo_relative_path": None,
        "matches_reported_git_revision": False,
        "matches_preimport_snapshot": False,
        "source_only_import_verified": False,
        "passed": False,
    }


def test_analyzer_revision_is_authenticated_independently_of_bank_producer(
    tmp_path: Path,
) -> None:
    analyzer = {
        "decision_grade": True,
        "git_sha": "b" * 40,
        "sources": {"analyzer": {"sha256": "c" * 64}},
    }

    assert not controller_module._decision_grade_evidence_inputs(
        bank_decision_grade=True,
        analyzer_provenance=analyzer,
        evidence_guard=None,
    )
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    with controller_module._retained_decision_grade_evidence(
        bank, meta,
    ) as evidence_guard:
        assert controller_module._decision_grade_evidence_inputs(
            bank_decision_grade=True,
            analyzer_provenance=analyzer,
            evidence_guard=evidence_guard,
        ) is True


def test_decision_grade_loader_records_both_retained_witnesses(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    _write_bank(bank, correct_gap=True)

    _transitions, info = load_transitions(bank)

    artifacts = info["analyzer_consumed_inputs"]
    assert [artifact["role"] for artifact in artifacts] == [
        "trajectory_bank",
        "trajectory_bank_witness",
        "trajectory_manifest",
        "trajectory_manifest_witness",
    ]
    assert all(artifact["descriptor_authenticated"] for artifact in artifacts)
    assert [artifact["retained_witness"] for artifact in artifacts] == [
        False, True, False, True,
    ]


def test_methodology_smoke_explicitly_allows_legacy_final_only_inputs(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    controller_module._pending_output_path(bank).unlink()
    controller_module._pending_manifest_path(meta).unlink()

    transitions, info = load_transitions(bank, methodology_smoke=True)

    assert len(transitions) == 3
    assert info["decision_grade"] is False


def test_decision_grade_loader_rejects_invalid_recovery_marker(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    controller_module._invalid_manifest_path(meta).write_text("invalid\n")

    with pytest.raises(SystemExit, match="invalidated"):
        load_transitions(bank)


def test_decision_grade_loader_restores_marker_removed_during_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    marker = controller_module._invalid_manifest_path(meta)
    marker.write_text("invalid\n")
    real_stat = publication_module.os.stat
    removed = False

    def remove_marker_then_report_absent(
        path: Any, *args: Any, **kwargs: Any,
    ) -> os.stat_result:
        nonlocal removed
        if not removed and path == marker.name and kwargs.get("dir_fd") is not None:
            removed = True
            marker.unlink()
            raise FileNotFoundError(errno.ENOENT, "removed marker", marker.name)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(publication_module.os, "stat", remove_marker_then_report_absent)

    with pytest.raises(SystemExit, match="invalid-recovery marker absence changed"):
        load_transitions(bank)

    assert marker.exists()


@pytest.mark.parametrize("witness_kind", ["bank", "manifest"])
def test_decision_grade_loader_rejects_missing_retained_witness(
    tmp_path: Path, witness_kind: str,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    witness = (
        controller_module._pending_output_path(bank)
        if witness_kind == "bank"
        else controller_module._pending_manifest_path(meta)
    )
    witness.unlink()

    with pytest.raises(SystemExit, match=r"(hard links|safely open)"):
        load_transitions(bank)


@pytest.mark.parametrize("witness_kind", ["bank", "manifest"])
def test_decision_grade_loader_rejects_foreign_retained_witness(
    tmp_path: Path, witness_kind: str,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    witness = (
        controller_module._pending_output_path(bank)
        if witness_kind == "bank"
        else controller_module._pending_manifest_path(meta)
    )
    witness.unlink()
    witness.write_text("foreign\n")

    with pytest.raises(SystemExit, match=r"(hard links|retained hard link)"):
        load_transitions(bank)


def test_decision_grade_loader_rejects_extra_hard_link(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    _write_bank(bank, correct_gap=True)
    os.link(bank, tmp_path / "extra-bank-link")

    with pytest.raises(SystemExit, match="unexpected hard links"):
        load_transitions(bank)


def test_decision_grade_loader_rejects_bank_change_during_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = tmp_path / "bank.jsonl"
    _write_bank(bank, correct_gap=True)
    real_read = controller_module._read_stable_bytes_fd
    changed = False

    def read_then_change(
        file_fd: int, path: Path, *, before: os.stat_result | None = None,
    ) -> bytes:
        nonlocal changed
        content = real_read(file_fd, path, before=before)
        if not changed and path == bank:
            changed = True
            bank.write_text('{"tampered":true}\n')
        return content

    monkeypatch.setattr(controller_module, "_read_stable_bytes_fd", read_then_change)

    with pytest.raises(SystemExit, match="changed"):
        load_transitions(bank)


def test_walker_manifest_omits_gumbel_only_minibatch_setting(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())

    assert "minibatch_size" not in manifest["requested_search"]["active_parameters"]
    assert "walker_gather" in manifest["requested_search"]["active_parameters"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stack", "ThreadSafeGPUDispatcher(DirectGPUEvaluator)"),
        ("direct_max_batch", 999),
        ("input_bf16", True),
    ],
)
def test_requested_only_evaluator_changes_fail_closed(
    tmp_path: Path, field: str, value: object,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["requested_evaluator"][field] = value
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="realized evaluator"):
        load_transitions(bank)


def test_loader_rejects_internally_contradictory_model_search_contract(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    for name in ("requested_model_search_contract", "realized_model_search_contract"):
        manifest[name]["model_input_history_encoding"] = "lc0_root"
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="model/search encoding contract"):
        load_transitions(bank)


@pytest.mark.parametrize(
    ("field", "value"),
    [("passed", False), ("changed", ["checkpoint"]), ("final_git_dirty", True)],
)
def test_loader_requires_stable_consumed_artifacts_and_checkout(
    tmp_path: Path, field: str, value: object,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["artifact_stability"][field] = value
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="changed during collection"):
        load_transitions(bank)


def test_loader_authenticates_and_parses_one_bank_buffer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    original_read_artifact = controller_module._read_consumed_artifact
    reads: dict[Path, int] = {}

    def racing_read_artifact(
        path: Path, *, role: str,
    ) -> tuple[bytes, dict[str, Any]]:
        resolved = path.resolve()
        reads[resolved] = reads.get(resolved, 0) + 1
        payload = original_read_artifact(path, role=role)
        if resolved == bank.resolve():
            path.write_text('{"replacement": true}\n')
        return payload

    monkeypatch.setattr(
        controller_module, "_read_consumed_artifact", racing_read_artifact,
    )

    transitions, _ = load_transitions(bank, methodology_smoke=True)

    assert len(transitions) == 3
    assert reads[bank.resolve()] == 1
    assert reads[meta.resolve()] == 1


def test_smoke_loader_never_reopens_manifest_that_appears_after_absent_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest_bytes = meta.read_bytes()
    meta.unlink()
    original_read_artifact = controller_module._read_consumed_artifact

    def manifest_appears_after_failed_open(
        path: Path, *, role: str,
    ) -> tuple[bytes, dict[str, Any]]:
        if role == "trajectory_manifest":
            path.write_bytes(manifest_bytes)
            raise FileNotFoundError(path)
        return original_read_artifact(path, role=role)

    monkeypatch.setattr(
        controller_module,
        "_read_consumed_artifact",
        manifest_appears_after_failed_open,
    )

    transitions, info = load_transitions(
        bank, meta_path=meta, methodology_smoke=True,
    )

    assert len(transitions) == 3
    assert info["manifest"] == {}
    assert meta.read_bytes() == manifest_bytes


def test_loader_requires_loaded_cboard_native_provenance(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["lc0_extension"]["cboard_encode_full"] = False
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="CBoard encoding extension"):
        load_transitions(bank)


def test_analyzer_output_cannot_alias_loaded_cboard_extension(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())

    with pytest.raises(ValueError, match="consumed input artifact"):
        _require_safe_output_path(
            bank, meta, Path("/_lc0_ext.so"), manifest=manifest,
        )


def test_analyzer_output_cannot_overwrite_imported_oracle_source(tmp_path: Path) -> None:
    from scripts import reachable_oracle

    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())

    with pytest.raises(ValueError, match="tracked or repository-control"):
        _require_safe_output_path(
            bank, meta, Path(reachable_oracle.__file__), manifest=manifest,
        )


def test_loader_requires_exact_resolved_cuda_device_identity(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["runtime"]["resolved_model_parameter_devices"] = ["cuda:1"]
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="CUDA runtime/device provenance"):
        load_transitions(bank)


def test_loader_rejects_raw_devices_that_contradict_resolved_identity(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["runtime"]["evaluator_device"] = "cuda:1"
    manifest["runtime"]["model_parameter_devices"] = ["cuda:1"]
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="CUDA runtime/device provenance"):
        load_transitions(bank)


@pytest.mark.parametrize(
    ("field", "value"),
    [("enabled", False), ("mode", "reduce-overhead"), ("cache_dir", "")],
)
def test_loader_requires_production_compile_manifest(
    tmp_path: Path, field: str, value: object,
) -> None:
    bank = tmp_path / "bank.jsonl"
    meta = _write_bank(bank, correct_gap=True)
    manifest = json.loads(meta.read_text())
    manifest["compile"][field] = value
    meta.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="realized evaluator"):
        load_transitions(bank)


def test_cluster_bootstrap_refits_models_inside_resamples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import analyze_chunk_controller as controller

    rows = [
        _transition(
            str(game), game, horizon, (game - 5) / horizon,
            current=game % 2 == 0,
        )
        for game in range(12)
        for horizon in (100, 150)
    ]
    original = controller._fit_ridge
    calls = 0
    alpha_calls = 0

    def counted_fit(x: np.ndarray, y: np.ndarray, alpha: float):
        nonlocal calls
        calls += 1
        return original(x, y, alpha)

    original_inner_alpha = controller._inner_alpha

    def counted_inner_alpha(
        transitions: list[Transition], model: str, n_folds: int,
    ) -> float:
        nonlocal alpha_calls
        alpha_calls += 1
        return original_inner_alpha(transitions, model, n_folds)

    monkeypatch.setattr(controller, "_fit_ridge", counted_fit)
    monkeypatch.setattr(controller, "_inner_alpha", counted_inner_alpha)
    controller.cluster_bootstrap_delta(
        rows,
        allocation_fraction=0.5,
        samples=5,
        seed=11,
        n_folds=3,
    )

    assert calls >= 4, "each usable replicate must refit both models"
    assert alpha_calls >= 4, "each usable replicate must reselect alpha in-bag"


def test_bootstrap_blas_limit_restores_after_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threadpoolctl
    from scripts import analyze_chunk_controller as controller

    rows = [
        _transition(str(game), game, horizon, game / horizon, current=True)
        for game in range(6)
        for horizon in (100, 150)
    ]
    active = False
    calls: list[tuple[int, str]] = []

    class FakeLimit:
        def __enter__(self) -> None:
            nonlocal active
            assert not active
            active = True

        def __exit__(self, *_args: object) -> None:
            nonlocal active
            assert active
            active = False

    def fake_limit(*, limits: int, user_api: str) -> FakeLimit:
        calls.append((limits, user_api))
        return FakeLimit()

    def checked_refit(
        transitions: list[Transition], _fold_ids: np.ndarray,
        *, model: str, n_folds: int,
    ) -> np.ndarray:
        del model, n_folds
        assert active
        return np.zeros(len(transitions), dtype=np.float64)

    monkeypatch.setattr(threadpoolctl, "threadpool_limits", fake_limit)
    monkeypatch.setattr(controller, "_refit_fold_predictions", checked_refit)
    monkeypatch.setattr(
        controller,
        "_minimum_reachable_rung_gain_delta",
        lambda *_args, **_kwargs: 1.0,
    )

    result = controller.cluster_bootstrap_delta(
        rows, allocation_fraction=0.5, samples=2, seed=3, n_folds=3,
    )

    assert calls == [(1, "blas")]
    assert active is False
    assert result["valid_samples"] == 2


def test_bootstrap_blas_limit_preserves_duplicate_cluster_near_tie_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import analyze_chunk_controller as controller

    rows = [
        _transition(
            f"{game}:{copy}", game, horizon,
            (game - 2.5) * 1e-12 + copy * 1e-15 + horizon * 1e-18,
            current=(game + copy) % 2 == 0,
            value=(game * 2 + copy) * 1e-8,
        )
        for game in range(6)
        for copy in range(2)
        for horizon in (100, 150, 200)
    ]
    scoped_limit = controller._bootstrap_blas_limit
    monkeypatch.setattr(controller, "_bootstrap_blas_limit", nullcontext)
    unrestricted = controller.cluster_bootstrap_delta(
        rows, allocation_fraction=0.5, samples=3, seed=9, n_folds=3,
    )
    monkeypatch.setattr(controller, "_bootstrap_blas_limit", scoped_limit)
    scoped = controller.cluster_bootstrap_delta(
        rows, allocation_fraction=0.5, samples=3, seed=9, n_folds=3,
    )

    assert scoped == unrestricted


def test_bootstrap_missing_analysis_dependency_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _transition(str(game), game, horizon, 0.0, current=True)
        for game in range(3)
        for horizon in (100, 150)
    ]
    monkeypatch.setitem(sys.modules, "threadpoolctl", None)

    with pytest.raises(RuntimeError, match=r'pip install -e "\.\[analysis\]"'):
        cluster_bootstrap_delta(
            rows, allocation_fraction=0.5, samples=1, seed=0, n_folds=2,
        )


def test_trajectory_producer_import_does_not_need_analysis_dependency() -> None:
    code = """
import builtins

real_import = builtins.__import__

def blocked_import(name, *args, **kwargs):
    if name == "threadpoolctl" or name.startswith("threadpoolctl."):
        raise ModuleNotFoundError("blocked analysis dependency", name="threadpoolctl")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
import scripts.backtest_chunk_trajectory
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_bootstrap_fails_closed_when_oracle_headroom_is_undefined() -> None:
    rows = [
        _transition(str(game), game, horizon, 0.0, current=True)
        for game in range(6)
        for horizon in (100, 150)
    ]
    result = cluster_bootstrap_delta(
        rows,
        allocation_fraction=0.5,
        samples=25,
        seed=3,
        n_folds=3,
        min_oracle_headroom=1e-4,
    )

    assert result["requested_samples"] == 25
    assert result["valid_samples"] == 0
    assert result["valid_fraction"] == 0.0
    assert result["lower_95"] is None
