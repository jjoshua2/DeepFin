from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import bt4_joint_readout as tool


def bank(
    root: Path, arm: str, sims: int, *, candidate: str | None = None,
    opening_suffix: str = "", hoist: str = "4096", reference_name: str = "S0.pt",
    prior_temperature: float = 1.0,
) -> dict:
    reference = root / reference_name
    settings = {
        "candidate": str(root / (candidate or f"{arm}.pt")), "reference": str(reference),
        "games": 1000, "mode": "matched_sims", "sims_candidate": sims,
        "sims_reference": sims, "seed": 42, "opening_plies": 16,
        "openings_kind": "book", "openings": "frozen-book.zip", "max_plies": 300,
        "temperature": 0.1, "gumbel_add_noise": True,
        "search_candidate": {"shape": "training", "gumbel": {"policy_temp": prior_temperature}},
        "search_reference": {"shape": "training", "gumbel": {"policy_temp": prior_temperature}},
    }
    records: list[dict[str, Any]] = [{"kind": "header", "driver": "arena_standard", "version": 1,
                "settings": settings, "fingerprint": tool.settings_fingerprint(settings)}]
    for pair in range(tool.PAIRS):
        total = (pair % 4) * 0.5  # Mean score .375: both arms lose, yet comparison is required.
        for half, score in enumerate((min(total, 1.0), max(0.0, total - 1.0))):
            white_score = score if half == 0 else 1 - score
            records.append({
                "kind": "game", "pair_id": pair, "half": half,
                "a_is_white": half == 0, "opening_index": pair,
                "opening_fen": f"opening{pair}{opening_suffix}",
                "start_fen": f"opening{pair}{opening_suffix}",
                "result": {0.0: "0-1", 0.5: "1/2-1/2", 1.0: "1-0"}[white_score],
                "score_candidate": score, "seed": 42, "loop": "chunked",
                "compile": "on", "eval_hoist": hoist,
            })
    path = root / f"{arm}.{sims}.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    return tool.read_arm(path, reference=reference, seed=42, sims=sims,
                         prior_temperature=prior_temperature)


def test_identical_correlated_curves_have_zero_interaction_interval(tmp_path: Path) -> None:
    cells = {(arm, sims): bank(tmp_path, arm, sims)
             for arm in tool.ARMS for sims in tool.BUDGETS}
    report = tool.build_report(cells)
    assert report["screen_complete"]
    assert len(report["cells"]) == 9
    for result in report["cells"].values():
        assert result["strength_interpretation"] == "LOSS"
    for contrast in [*report["search_interactions_25_to_400"].values(),
                     *report["global_arm_comparisons_by_budget"].values()]:
        assert contrast["status"] == "READ"
        assert contrast["score_advantage"] == 0.0
        assert contrast["ci95"] == [0.0, 0.0]
        assert contrast["pair_outcome_covariance"] > 0
    assert report["promotion"].startswith("NONE")


@pytest.mark.parametrize("mismatch", ["opening", "candidate", "execution", "shared_candidate"])
def test_cross_cell_identity_mismatches_are_refused(tmp_path: Path, mismatch: str) -> None:
    left = bank(tmp_path, "G20T1", 25)
    arm = "G20T05" if mismatch == "shared_candidate" else "G20T1"
    right = bank(tmp_path, arm, 400,
        candidate="changed.pt" if mismatch == "candidate" else
                  "G20T1.pt" if mismatch == "shared_candidate" else None,
        opening_suffix="changed" if mismatch == "opening" else "",
        hoist="2048" if mismatch == "execution" else "4096")
    reason = {"opening": "same opening sequence", "candidate": "differs across budgets",
              "execution": "execution differs", "shared_candidate": "same candidate"}[mismatch]
    with pytest.raises(ValueError, match=reason):
        tool.build_report({("G20T1", 25): left, (arm, 400): right})


def test_missing_cells_remain_unread(tmp_path: Path) -> None:
    report = tool.build_report({("G20T1", 25): bank(tmp_path, "G20T1", 25)})
    assert not report["screen_complete"]
    assert len(report["missing_cells"]) == 8
    assert report["cells"]["G20T05:100"] == {"status": "UNREAD"}
    assert report["search_interactions_25_to_400"]["G20T1"] == {
        "status": "UNREAD", "missing_cells": ["G20T1:400"],
    }
    assert all(result["status"] == "UNREAD"
               for result in report["global_arm_comparisons_by_budget"].values())
    assert len(tool.build_report({})["missing_cells"]) == 9


def test_paired_interaction_retains_a_constant_advantage() -> None:
    baseline = np.tile([0.0, 0.25, 0.5, 0.75], 125)
    result = tool.compare(baseline + 0.25, baseline, seed=20260903, samples=10000)
    assert result["score_advantage"] == 0.25
    assert result["ci95"] == [0.25, 0.25]


def test_sf_close_cli_reads_only_its_three_direct_e0_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    args = ["bt4_joint_readout.py", "--profile", "sf-close",
            "--reference", str(tmp_path / "E0.pt")]
    for sims in tool.BUDGETS:
        cell = bank(tmp_path, "C20T05", sims, reference_name="E0.pt")
        args += ["--cell", f"C20T05:{sims}={cell['path']}"]
    monkeypatch.setattr("sys.argv", args)
    tool.main()
    report = json.loads(capsys.readouterr().out)
    assert report["screen_complete"]
    assert not report["missing_cells"]
    assert len(report["cells"]) == 3
    assert all(cell["strength_interpretation"] == "LOSS" for cell in report["cells"].values())
    assert report["global_arm_comparisons_by_budget"] == {}
    interaction = report["search_interactions_25_to_400"]["C20T05"]
    assert interaction["contrast"] == "score400 - score25 against E0"
    assert interaction["ci95"] == [0.0, 0.0]
    assert report["promotion"].startswith("NONE")


def test_sf_close_partial_report_does_not_import_global_results(tmp_path: Path) -> None:
    cell = bank(tmp_path, "C20T05", 25, reference_name="E0.pt")
    report = tool.build_report({("C20T05", 25): cell}, profile="sf-close")
    assert not report["screen_complete"]
    assert report["missing_cells"] == ["C20T05:100", "C20T05:400"]
    with pytest.raises(ValueError, match="unexpected arm"):
        tool.build_report({("G20T1", 25): cell}, profile="sf-close")
    with pytest.raises(ValueError, match="reference is not"):
        tool.read_arm(Path(cell["path"]), reference=tmp_path / "S0.pt", seed=42, sims=25)


def test_legacy_softened_banks_require_explicit_temperature(tmp_path: Path) -> None:
    cell = bank(tmp_path, "E0", 100, prior_temperature=1.5)
    with pytest.raises(ValueError, match="realized prior temperature"):
        tool.read_arm(Path(cell["path"]), reference=tmp_path / "S0.pt", seed=42)
    with pytest.raises(ValueError, match="prior temperature differs"):
        tool.build_report({("E0", 100): cell})
    report = tool.build_report({("E0", 100): cell}, prior_temperature=1.5)
    assert report["prior_temperature"] == 1.5
    assert report["cells"]["E0:100"]["status"] == "READ"


def test_one_sided_temperature_override_cannot_enter_report(tmp_path: Path) -> None:
    cell = bank(tmp_path, "C20T05", 100, reference_name="E0.pt")
    path = Path(cell["path"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["settings"]["search_reference"]["gumbel"]["policy_temp"] = 1.5
    rows[0]["fingerprint"] = tool.settings_fingerprint(rows[0]["settings"])
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="candidate and reference search settings differ"):
        tool.read_arm(path, reference=tmp_path / "E0.pt", seed=42)


def calibration_bank(root: Path, arm: str) -> Path:
    cell = bank(root, arm, 100, reference_name="temporary-control.pt")
    path = Path(cell['path'])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    settings = rows[0]['settings']
    settings['reference'] = settings['candidate']
    settings['search_reference']['gumbel']['policy_temp'] = 1.5
    # Frozen apply_search_overrides includes the literal CLI override in provenance.
    for side, temperature in (('candidate', '1.0'), ('reference', '1.5')):
        settings[f'search_{side}']['source'] = (
            f'pbt2_small.yaml -> reco -> worker SearchConfig + CLI(policy_temp={temperature})'
        )
    rows[0]['fingerprint'] = tool.settings_fingerprint(settings)
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    return path


def test_calibration_cli_separates_two_same_net_effects(tmp_path, monkeypatch, capsys):
    args = ['reader', '--profile', 'calibration']
    for arm in ('S0', 'E0'):
        path = calibration_bank(tmp_path, arm)
        args += ['--checkpoint', f'{arm}={tmp_path / (arm + ".pt")}', '--cell', f'{arm}:100={path}']
    monkeypatch.setattr('sys.argv', args)
    tool.main()
    report = json.loads(capsys.readouterr().out)
    assert report['screen_complete']
    assert set(report['cells']) == {'S0:100', 'E0:100'}
    assert all(c['result']['verdict'] == 'KILL' for c in report['cells'].values())
    assert 'global_arm_comparisons_by_budget' not in report


def test_calibration_checks_expected_checkpoint_and_non_temperature_search(tmp_path):
    path = calibration_bank(tmp_path, 'E0')
    expected = tmp_path / 'E0.pt'
    tool.read_arm(path, reference=expected, seed=42, calibration=True)
    with pytest.raises(ValueError, match='reference is not'):
        tool.read_arm(path, reference=tmp_path / 'wrong.pt', seed=42, calibration=True)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['settings']['search_reference']['gumbel']['topk'] = 12
    rows[0]['fingerprint'] = tool.settings_fingerprint(rows[0]['settings'])
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    with pytest.raises(ValueError, match='differ only'):
        tool.read_arm(path, reference=expected, seed=42, calibration=True)


def test_calibration_alignment_and_missing_cells(tmp_path):
    cells = {}
    for arm in ('S0', 'E0'):
        path = calibration_bank(tmp_path, arm)
        cells[arm, 100] = tool.read_arm(path, reference=tmp_path / f'{arm}.pt', seed=42, calibration=True)
    partial = tool.calibration_report({('S0', 100): cells['S0', 100]})
    assert partial['cells']['E0:100'] == {'status': 'UNREAD'}
    cells['E0', 100]['openings'][0] = 'different'
    with pytest.raises(ValueError, match='alignment'):
        tool.calibration_report(cells)


@pytest.fixture
def confirmation_fixture(tmp_path):
    import hashlib
    import random

    import chess

    rng = random.Random(817)
    lines, epds = [], set()
    while len(lines) < 500:
        board = chess.Board()
        for _ in range(16):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        if len(board.move_stack) != 16 or board.is_game_over() or board.legal_moves.count() < 2 or board.epd() in epds:
            continue
        epds.add(board.epd())
        lines.append(board.root().fen() + ' | ' + ' '.join(move.uci() for move in board.move_stack))
    opening_file = tmp_path / 'confirmation.fen'
    opening_file.write_text('\n'.join(lines) + '\n')
    digest = hashlib.sha256(opening_file.read_bytes()).hexdigest()
    evidence = tool.confirmation_input(opening_file, digest)
    cell = bank(tmp_path, 'G20T1', 100)
    path = Path(cell['path'])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    settings = rows[0]['settings']
    settings.update(openings=str(opening_file), openings_kind='fen', opening_plies=None)
    rows[0]['fingerprint'] = tool.settings_fingerprint(settings)
    for row in rows[1:]:
        row['opening_fen'] = row['start_fen'] = evidence['fens'][row['pair_id']]
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    return path, opening_file, digest, evidence


def test_explicit_confirmation_cli_validates_actual_history_format(confirmation_fixture, tmp_path, monkeypatch, capsys):
    path, openings, digest, _ = confirmation_fixture
    monkeypatch.setattr('sys.argv', ['reader', '--reference', str(tmp_path / 'S0.pt'),
                                    '--cell', f'G20T1:100={path}', '--confirmation-openings', str(openings),
                                    '--confirmation-sha256', digest])
    tool.main()
    report = json.loads(capsys.readouterr().out)
    evidence = report['cells']['G20T1:100']['confirmation_input']
    assert evidence['sha256'] == digest
    assert evidence['history_plies'] == 16
    assert 'endpoints only' in evidence['limitation']
    assert not report['screen_complete']
    assert report['cells']['G20T1:100']['result']['pairs'] == 500


def test_confirmation_refuses_wrong_identity_and_keeps_book_default_strict(confirmation_fixture, tmp_path):
    path, openings, _, evidence = confirmation_fixture
    with pytest.raises(ValueError, match='SHA256'):
        tool.confirmation_input(openings, '0' * 64)
    with pytest.raises(ValueError, match='off-protocol'):
        tool.read_arm(path, reference=tmp_path / 'S0.pt', seed=42)
    with pytest.raises(ValueError, match='calibration'):
        tool.read_arm(path, reference=tmp_path / 'S0.pt', seed=42, confirmation=evidence, calibration=True)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['settings']['openings'] = str(tmp_path / 'other.fen')
    rows[0]['fingerprint'] = tool.settings_fingerprint(rows[0]['settings'])
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    with pytest.raises(ValueError, match='opening path'):
        tool.read_arm(path, reference=tmp_path / 'S0.pt', seed=42, confirmation=evidence)


def test_confirmation_rejects_reordered_endpoints(confirmation_fixture, tmp_path):
    path, _, _, evidence = confirmation_fixture
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows[1:]:
        if row['pair_id'] in (0, 1):
            row['opening_fen'] = row['start_fen'] = evidence['fens'][1 - row['pair_id']]
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    with pytest.raises(ValueError, match='endpoint/order'):
        tool.read_arm(path, reference=tmp_path / 'S0.pt', seed=42, confirmation=evidence)


def test_confirmation_rejects_endpoint_only_file_even_with_matching_hash(confirmation_fixture):
    import hashlib

    _, openings, _, evidence = confirmation_fixture
    openings.write_text('\n'.join(evidence['fens']) + '\n')
    digest = hashlib.sha256(openings.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='16 history moves'):
        tool.confirmation_input(openings, digest)


@pytest.mark.parametrize(("candidate_role", "reference_role"), [("G20T1", "E0"), ("E0T05", "C20T05"), ("C20T05", "E0T05")])
def test_direct_confirmation_reports_actual_reference_without_screen_claims(
    confirmation_fixture, tmp_path, monkeypatch, capsys, candidate_role, reference_role,
):
    path, openings, digest, _ = confirmation_fixture
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['settings']['reference'] = str(tmp_path / f'{reference_role}-new-seed.pt')
    rows[0]['settings']['candidate'] = str(tmp_path / f'{candidate_role}-new-seed.pt')
    rows[0]['fingerprint'] = tool.settings_fingerprint(rows[0]['settings'])
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    monkeypatch.setattr('sys.argv', [
        'reader', '--profile', 'confirmation', '--candidate-role', candidate_role,
        '--reference-role', reference_role, '--candidate', str(tmp_path / f'{candidate_role}-new-seed.pt'),
        '--reference', str(tmp_path / f'{reference_role}-new-seed.pt'), '--confirmation-sims', '100',
        '--cell', f'{candidate_role}:100={path}', '--confirmation-openings', str(openings),
        '--confirmation-sha256', digest,
    ])
    tool.main()
    report = json.loads(capsys.readouterr().out)
    assert report['reference_role'] == reference_role
    assert report['contrast'] == f'{candidate_role} directly against {reference_role}'
    assert report['match_complete']
    assert report['cells'][f'{candidate_role}:100']['result']['games'] == 1000
    assert report['training_provenance_verified'] is False
    assert 'search_interactions_25_to_400' not in report
    assert 'global_arm_comparisons_by_budget' not in report
    assert 'S0' not in json.dumps(report)
    assert not any('historical runtime' in text for text in report['limitations'])


@pytest.mark.parametrize(('changed_flag', 'changed_value'), [
    ('--candidate', 'wrong.pt'), ('--confirmation-sims', '400'),
    ('--reference-role', 'G20T1'), ('--profile', 'global'),
    ('--bootstrap-samples', '10000'),
])
def test_direct_confirmation_refuses_wrong_identity_budget_or_profile(
    confirmation_fixture, tmp_path, monkeypatch, capsys, changed_flag, changed_value,
):
    path, openings, digest, _ = confirmation_fixture
    args = [
        'reader', '--profile', 'confirmation', '--candidate-role', 'G20T1',
        '--reference-role', 'S0', '--candidate', str(tmp_path / 'G20T1.pt'),
        '--reference', str(tmp_path / 'S0.pt'), '--confirmation-sims', '100',
        '--cell', f'G20T1:100={path}', '--confirmation-openings', str(openings),
        '--confirmation-sha256', digest,
    ]
    if changed_flag in args:
        args[args.index(changed_flag) + 1] = changed_value
    else:
        args.extend([changed_flag, changed_value])
    monkeypatch.setattr('sys.argv', args)
    with pytest.raises(SystemExit) as error:
        tool.main()
    assert error.value.code == 2
    assert not capsys.readouterr().out
