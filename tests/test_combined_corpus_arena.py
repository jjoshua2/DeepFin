"""Fixed combined settings reach the actual arena command and strict reader."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import combined_corpus_arena as tool


def manifest(tmp_path) -> dict[str, Any]:
    return {
        "profile": tool.PROFILE,
        "state": str(tmp_path / "state"),
        "output": str(tmp_path / "arena"),
        "candidate": {
            "role": "Combined35M_V50",
            "path": "/models/v50.pt",
            "sha256": "a" * 64,
        },
        "reference": {
            "role": "Combined35M_SF100",
            "path": "/models/sf100.pt",
            "sha256": "b" * 64,
        },
        "candidate_training": {"path": "/receipts/v50.json", "sha256": "c" * 64},
        "reference_training": {"path": "/receipts/sf100.json", "sha256": "d" * 64},
        "opening_panel": {"path": "/panel.json", "sha256": tool.combined.PANEL_SHA},
        "book": {"path": "/book.zip", "sha256": tool.owned.BOOK_SHA},
    }


def settings(m):
    search = {"shape": "training", "gumbel": {"policy_temp": 1.0}, "tree_reuse": "cold"}
    return {
        "mode": "matched_sims",
        "candidate": m["candidate"]["path"],
        "reference": m["reference"]["path"],
        "games": 512,
        "seed": 20260913,
        "sims_candidate": 400,
        "sims_reference": 400,
        "ms_per_move": None,
        "openings": "/book.zip",
        "openings_kind": "book",
        "opening_plies": 16,
        "max_plies": 300,
        "temperature": 0.1,
        "gumbel_add_noise": True,
        "volatility_candidate": None,
        "uci_args": "",
        "syzygy": "",
        "syzygy_max_pieces": 0,
        "search_candidate": search,
        "search_reference": deepcopy(search),
    }


@pytest.fixture
def pair(tmp_path, monkeypatch):
    m = manifest(tmp_path)
    verified = []
    monkeypatch.setattr(tool.package.combined, "matched_training_pair", verified.append)
    monkeypatch.setattr(
        tool.package, "read_json", lambda ref: {"opening_panel": m["opening_panel"]}
    )
    return m, verified


def test_exact_arena_command_and_combined_reader(pair):
    m, verified = pair
    contract = tool.contract_for(m, settings(m))
    cmd = tool.command(contract, "/usr/bin/python3")
    tool.package.validate(contract)
    tool.package.command_check(cmd, contract)
    assert verified[0]["candidate_training"] == m["candidate_training"]
    assert verified[0]["reference_training"] == m["reference_training"]
    assert cmd[cmd.index("--games") + 1] == "512"
    assert cmd[cmd.index("--seed") + 1] == "20260913"
    assert cmd[cmd.index("--max-seconds") + 1] == "7140.0"
    assert "--no-rolling" not in cmd
    assert tool.owned.timeout_command(cmd, 7200)[:5] == [
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=30s",
        "7170s",
        "/usr/bin/python3",
    ]


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--seed", "42"), ("--seed", "20260909"), ("--games", "256"), ("--sims", "100")],
)
def test_actual_command_rejects_old_defaults(pair, flag, value):
    m, _ = pair
    contract = tool.contract_for(m, settings(m))
    cmd = tool.command(contract, "/usr/bin/python3")
    cmd[cmd.index(flag) + 1] = value
    with pytest.raises(ValueError, match=r"command|setting|profile|combined|fields"):
        tool.package.command_check(cmd, contract)


@pytest.mark.parametrize(
    ("key", "value"),
    [("seed", 42), ("seed", 20260909), ("pairs", 128), ("profile", "old_corpus")],
)
def test_reader_rejects_old_contract(pair, key, value):
    m, _ = pair
    contract = tool.contract_for(m, settings(m))
    contract[key] = value
    with pytest.raises(ValueError, match=r"command|setting|profile|combined|fields"):
        tool.package.validate(contract)


def test_training_verifier_uses_bound_original_path_and_restores(tmp_path, monkeypatch):
    m = manifest(tmp_path)
    p = tmp_path / "original.py"
    p.write_text('def matched_training_pair(m):\n m["verified"] = True\n')
    sha = tool.owned.sha(p)
    m["training_verifier"] = {"path": str(p), "sha256": sha}
    # Use the actual producer to establish the completion schema, not a guessed
    # predecessor field (which exists only in the launch manifest).
    producer_m = dict(
        m,
        role="Combined35M_V50",
        run="/fake/run",
        state="/fake/state",
        code_pins={str(p): sha},
        previous_training=m["reference_training"],
    )
    for key in (
        "corpus_manifest",
        "prospective",
        "opening_panel",
        "preregistration",
        "runtime_manifest",
        "selected_subset_qualification",
    ):
        producer_m.setdefault(key, {})
    with monkeypatch.context() as context:
        context.setattr(tool.combined, "summary_contract", lambda *a: None)
        context.setattr(tool.combined, "selected_subset", lambda *a: {})
        context.setattr(
            tool.combined,
            "verify_actual_columns",
            lambda *a: {"actual_staging_sha256": "staging", "actual_game_columns": []},
        )
        context.setattr(
            tool.combined, "pin", lambda path: {"path": str(path), "sha256": "digest"}
        )
        context.setattr(tool.combined.owned, "sha", lambda path: "digest")
        context.setattr(
            tool.combined.owned,
            "read",
            lambda path: {
                "checkpoints": [
                    {
                        "role": "last",
                        "path": "/fake/run/checkpoint.pt",
                        "sha256": "digest",
                    }
                ],
                "valid_control": False,
                "validity_problems": [],
            },
        )
        receipt = tool.combined.completed_training(
            producer_m,
            {
                "arms": {
                    "V50": {
                        "canonical_plan_sha256": "canonical",
                        "physical_plan": {"plan_sha256": "physical"},
                    }
                }
            },
            {"gpu_seconds": 1.0},
        )
    assert "previous_training" not in receipt
    monkeypatch.setattr(tool, "combined", SimpleNamespace(__file__=str(p)))
    monkeypatch.setattr(
        tool.reader,
        "read_json",
        lambda ref: receipt,
    )
    previous = tool.package.combined

    def fail_inside():
        with tool.training_verifier(m):
            assert m["verified"]
            assert tool.package.combined.__file__ == str(p)
            raise RuntimeError("downstream fail")

    with pytest.raises(RuntimeError, match="downstream"):
        fail_inside()
    assert tool.package.combined is previous
    monkeypatch.setattr(
        tool.reader,
        "read_json",
        lambda ref: {
            "code_pins": {str(p): "wrong"},
        },
    )
    with pytest.raises(ValueError, match="binding"), tool.training_verifier(m):
        pass


def test_stat_watch_does_not_reread_receipts(tmp_path):
    p = tmp_path / "receipt.json"
    p.write_text("{}")
    before = tool.stat_paths([str(p)])
    p.write_text('{"changed":true}')
    assert tool.stat_paths(before) != before


@pytest.mark.parametrize("change_receipt", [False, True])
def test_execute_uses_owned_stage_guard_and_preserves_failure(
    tmp_path, monkeypatch, change_receipt
):
    from contextlib import nullcontext
    import time

    m = manifest(tmp_path)
    m["stop_paths"] = []
    state = Path(m["state"])
    state.mkdir()
    (tmp_path / "scratchpad").mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    rt = {"executable": "/usr/bin/python3"}
    contract = tool.contract_for(m, settings(m))
    manifest_pin = tool.recipe.write(state / "manifest.json", m)
    observed = tool.recipe.write(state / "observed.json", {})
    prepared = {
        "status": "CPU_PREPARED_NOT_LAUNCHED",
        "manifest": manifest_pin,
        "observed": observed,
        "request": observed,
        "runtime_root": str(runtime),
        "runtime": rt,
        "contract_template": contract,
        "command": tool.command(contract, rt["executable"]),
        "input_stamps": {},
    }
    prepared_pin = tool.recipe.write(state / "prepared.json", prepared)
    monkeypatch.setattr(tool, "static", lambda m: None)
    monkeypatch.setattr(tool, "training_verifier", lambda m: nullcontext())
    monkeypatch.setattr(tool.recipe, "qualified_runtime", lambda m: (runtime, rt))
    monkeypatch.setattr(tool.package, "validate", lambda c: None)
    monkeypatch.setattr(tool.memory, "require_available", lambda gib: None)
    monkeypatch.setattr(tool, "guard", lambda m, deadline: None)
    monkeypatch.setattr(tool.owned, "ROOT", tmp_path)
    monkeypatch.setattr(tool.subprocess, "check_output", lambda *args, **kwargs: "")
    calls = []

    def run(cmd, _out, seconds, lease, stage, _metadata, **kwargs):
        calls.append((cmd, seconds, lease, stage, kwargs))
        if change_receipt:
            Path(prepared_pin["path"]).write_text("changed")
        kwargs["guard"]()
        raise RuntimeError("owned-stage failure")

    monkeypatch.setattr(tool.owned, "run_owned_stage", run)
    error_type = ValueError if change_receipt else RuntimeError
    error_message = (
        "preparation evidence changed" if change_receipt else "owned-stage failure"
    )
    with pytest.raises(error_type, match=error_message):
        tool.execute(m, prepared_pin, time.time() + 12000)
    assert calls[0][1] == 7200
    assert calls[0][2] is not None
    assert calls[0][3] == "arena"
    tool.package.command_check(calls[0][0], contract)
    failure = tool.owned.read(state / "arena_operator/failed.json")
    assert failure["status"] == "FAILED_NO_AUTOMATIC_RETRY"
    assert not (state / "arena_operator/completed.json").exists()


def test_cpu_probe_real_settings_use_new_seed_and_full_pair_count(
    tmp_path, monkeypatch
):
    import json
    import sys
    import chess
    import numpy as np
    import torch
    from scripts import combined_corpus_arena_probe as probe

    runtime = Path(probe.arena.__file__).resolve().parent.parent
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    boards = [chess.Board()] * 256
    calls = []

    def openings(_path, *, n_pairs, max_plies, rng):
        calls.append((n_pairs, max_plies))
        assert (
            rng.bit_generator.state
            == np.random.default_rng(20260913).bit_generator.state
        )
        return boards

    monkeypatch.setattr(probe.arena, "load_paired_openings", openings)
    model = SimpleNamespace(
        state_dict=lambda: {"w": SimpleNamespace(shape=(1,))},
        parameters=lambda: [SimpleNamespace(numel=lambda: 61444448)],
    )
    monkeypatch.setattr(probe, "load_model_from_checkpoint", lambda path, device: model)
    panel = [{"root_fen": b.fen(), "moves": [], "fen": b.fen()} for b in boards]
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(json.dumps(panel))
    m = manifest(tmp_path)
    request = {
        "runtime_root": str(runtime),
        "runtime": {"native_extensions": {}},
        "arena_seed": 20260913,
        "book": {"path": "/fake/book.zip"},
        "panel": {"path": str(panel_path)},
        "packages": {m[s]["role"]: m[s] for s in ("candidate", "reference")},
        "cells": [
            {
                "name": "combined_value",
                "candidate": "Combined35M_V50",
                "reference": "Combined35M_SF100",
                "priors": [1.0, 1.0],
            }
        ],
    }
    req = tmp_path / "request.json"
    req.write_text(json.dumps(request))
    output = tmp_path / "observed.json"
    monkeypatch.setattr(sys, "argv", ["probe", str(req), str(output)])
    probe.main()
    observed = json.loads(output.read_text())
    assert calls == [(256, 16)]
    contract = tool.contract_for(m, observed["cells"]["combined_value"]["settings"])
    tool.package.command_check(tool.command(contract, sys.executable), contract)
    assert contract["settings"]["games"] == 512
    assert contract["settings"]["seed"] == 20260913
    assert observed["cuda_initialized"] is False


@pytest.mark.parametrize('defect', ['none', 'old_seed', 'old_pairs', 'wrong_role', 'wrong_prior', 'wrong_panel', 'workers16', 'missing_schedule'])
def test_ceres_endpoint_original_receipt_and_fixed_settings(tmp_path, monkeypatch, defect):
    from tests.test_matched_recipe_pair import completed_training, put
    from scripts import bt4_one_epoch_screen as epoch

    m = manifest(tmp_path)
    m['profile'] = tool.package.CERES_PROFILE
    for side, role in zip(('candidate', 'reference'), tool.package.CERES_ROLES):
        m[side] = {'role': role, 'path': str(tmp_path / role / 'checkpoint.pt'),
                   'sha256': ('a' if side == 'candidate' else 'b') * 64}
        m[side + '_training'] = completed_training(tmp_path, m[side])
    # Real original completed-receipt shape; no combined-only keys are supplied.
    pin = m['candidate_training']
    receipt = tool.reader.read_json(pin)
    corpus = epoch.CORPORA['Ceres100']
    summary_path = Path(receipt['run']) / 'summary.json'
    summary = tool.owned.read(summary_path)
    summary['corpus']['shard_dirs'] = [str(corpus)]
    summary['sampling'].update(plan_workers=2, load_workers=2)
    if defect == 'workers16':
        summary['sampling'].update(plan_workers=16, load_workers=16)
    receipt['summary_sha256'] = put(summary_path, summary)['sha256']
    schedule = tool.reader.read_json(receipt['schedule'])
    schedule['arms']['Ceres100'].update(corpus=str(corpus), summary_sha256=receipt['summary_sha256'])
    receipt['schedule'] = put(Path(receipt['schedule']['path']), schedule)
    if defect == 'missing_schedule':
        receipt.pop('schedule')
    m['candidate_training'] = put(Path(pin['path']), receipt)
    # Endpoint recipe behavior is exercised by test_matched_recipe_pair; this
    # integration isolates dispatch to the actual original completion checker.
    monkeypatch.setattr(tool.reader, 'matched_epoch_workers', lambda role, *_: 2 if role == 'Ceres100' else 16)
    contract = tool.contract_for(m, settings(m))
    if defect == 'old_seed':
        contract['seed'] = 42
    if defect == 'old_pairs':
        contract['pairs'] = 128
    if defect == 'wrong_role':
        contract['candidate']['role'] = 'CeresB50'
    if defect == 'wrong_prior':
        contract['candidate_prior_temperature'] = .7
    if defect == 'wrong_panel':
        contract['opening_panel']['sha256'] = '0' * 64
    if defect != 'none':
        with pytest.raises((ValueError, KeyError)):
            tool.package.validate(contract)
        return
    tool.package.validate(contract)
    cmd = tool.command(contract, '/usr/bin/python3')
    tool.package.command_check(cmd, contract)
    assert cmd[cmd.index('--seed') + 1] == '20260913'
    assert cmd[cmd.index('--games') + 1] == '512'
    assert cmd[cmd.index('--sims') + 1] == '400'
    m['training_verifier'] = {'path': tool.reader.__file__, 'sha256': tool.owned.sha(tool.reader.__file__)}
    with tool.training_verifier(m):
        pass


@pytest.mark.parametrize('roles', [('Ceres100', 'B100'), ('Combined35M_V50', 'Combined35M_SF100'), ('Ceres100', 'Combined35M_SF100'), ('CeresB50', 'B100')])
def test_probe_cell_dispatch_preserves_exact_registered_directions(roles):
    from scripts import combined_corpus_arena_probe as probe

    if roles not in (tool.package.CERES_ROLES, tool.ROLES):
        with pytest.raises(AssertionError):
            probe.registered_cell(dict.fromkeys(roles))
    else:
        cell = probe.registered_cell(dict.fromkeys(roles))
        assert (cell['candidate'], cell['reference']) == roles
        assert cell['priors'] == [1.0, 1.0]


@pytest.mark.parametrize('defect', ['none', 'old_profile', 'reverse', 'old_seed', 'old_pairs', 'prior', 'deadline'])
def test_native100_fixed_profile_reaches_exact_command(pair, defect):
    m, verified = pair
    m['profile'] = tool.package.V100_PROFILE
    m['candidate']['role'], m['reference']['role'] = tool.package.V100_ROLES
    contract = tool.contract_for(m, settings(m))
    if defect == 'old_profile':
        contract['profile'] = tool.PROFILE
    elif defect == 'reverse':
        contract['candidate']['role'], contract['reference']['role'] = reversed(tool.package.V100_ROLES)
    elif defect == 'old_seed':
        contract['seed'] = 42
    elif defect == 'old_pairs':
        contract['pairs'] = 128
    elif defect == 'prior':
        contract['candidate_prior_temperature'] = .5
    elif defect == 'deadline':
        contract['execution']['max_seconds'] = 5340.
    if defect != 'none':
        with pytest.raises(ValueError, match='differs'):
            tool.package.validate(contract)
        return
    tool.package.validate(contract)
    assert verified[0]['candidate']['role'] == 'Combined35M_V100'
    assert verified[0]['reference']['role'] == 'Combined35M_V50'
    command = tool.command(contract, '/usr/bin/python3')
    tool.package.command_check(command, contract)
    assert command[command.index('--games') + 1] == '512'
    assert command[command.index('--sims') + 1] == '400'


def test_native100_host_requires_current_verifier_only_in_candidate(tmp_path, monkeypatch):
    m = manifest(tmp_path)
    m['profile'] = tool.package.V100_PROFILE
    verifier = tmp_path / 'current.py'
    verifier.write_text('def matched_training_pair(m):\n m["bridge_called"] = True\n')
    digest = tool.owned.sha(verifier)
    m['training_verifier'] = {'path': str(verifier), 'sha256': digest}
    monkeypatch.setattr(tool, 'combined', SimpleNamespace(__file__=str(verifier)))
    receipts = {
        m['candidate_training']['path']: {'code_pins': {str(verifier): digest}},
        m['reference_training']['path']: {'code_pins': {'/old/verifier.py': 'old'}},
    }
    monkeypatch.setattr(tool.reader, 'read_json', lambda ref: receipts[ref['path']])
    previous = tool.package.combined
    with tool.training_verifier(m):
        assert m['bridge_called']
    assert tool.package.combined is previous
    receipts[m['candidate_training']['path']]['code_pins'] = {}
    with pytest.raises(ValueError, match='binding'), tool.training_verifier(m):
        pass


def test_native100_probe_cell_direction():
    from scripts import combined_corpus_arena_probe as probe

    cell = probe.registered_cell(dict.fromkeys(tool.package.V100_ROLES))
    assert cell == {'name': 'native100_value', 'candidate': 'Combined35M_V100',
                    'reference': 'Combined35M_V50', 'priors': [1.0, 1.0]}
    with pytest.raises(AssertionError):
        probe.registered_cell(dict.fromkeys(('Combined35M_V100', 'Combined35M_SF100')))
