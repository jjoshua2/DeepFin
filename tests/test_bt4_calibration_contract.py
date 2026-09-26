"""Explicit calibration exercises the real arena settings writer, without play."""
from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import chess
import pytest

from chess_anti_engine.utils.game_log import GameLogWriter
from scripts import bt4_joint_readout as tool
from scripts.arena_standard import SideSearch, arena_game_log_settings


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save_contract(path: Path, contract: dict[str, Any]) -> None:
    path.write_text(json.dumps(contract) + "\n")


@pytest.fixture
def calibration(tmp_path: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"immutable checkpoint fixture; never loaded as a model")
    panel: list[dict[str, Any]] = []
    rng = random.Random(281)
    while len(panel) < 4:
        board = chess.Board()
        for _ in range(16):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        if len(board.move_stack) == 16 and not board.is_game_over():
            panel.append({"root_fen": board.root().fen(),
                          "moves": [move.uci() for move in board.move_stack], "fen": board.fen()})
    panel_path = tmp_path / "openings.json"
    panel_path.write_text(json.dumps(panel) + "\n")
    settings = arena_game_log_settings(
        mode="matched_sims", candidate=str(checkpoint), reference=str(checkpoint), games=8,
        seed=42, openings_path=str(tmp_path / "pinned-book.zip"), openings_kind="book",
        opening_plies=16, sims_candidate=200, sims_reference=200, ms_per_move=None,
        max_plies=300, temperature=0.1, gumbel_add_noise=True,
        search_candidate=SideSearch("training", "pinned config + CLI(policy_temp=0.7)",
                                    {"policy_temp": 0.7}, 1, 0),
        search_reference=SideSearch("training", "pinned config + CLI(policy_temp=1.3)",
                                    {"policy_temp": 1.3}, 1, 0),
        volatility_candidate=None, uci_args="", syzygy_path=None, tb_max_pieces=0,
    )
    bank = tmp_path / "games.jsonl"
    with GameLogWriter(bank, driver="arena_standard", settings=settings) as writer:
        for pair, opening in enumerate(panel):
            for half in (0, 1):
                writer.write_game({"pair_id": pair, "half": half, "a_is_white": half == 0,
                                   "opening_index": pair, "opening_fen": opening["fen"],
                                   "start_fen": opening["fen"], "result": "1/2-1/2",
                                   "score_candidate": 0.5, "seed": 42, "loop": "rolling",
                                   "compile": "on", "eval_hoist": "4096"})
    contract = {"schema": 1, "checkpoint": pin(checkpoint), "candidate_prior_temperature": 0.7,
                "reference_prior_temperature": 1.3, "sims": 200, "expected_pairs": 4,
                "seed": 42, "loop": "rolling", "expected_settings": settings,
                "expected_execution": ["on", "4096"], "opening_panel": pin(panel_path),
                "bank": pin(bank)}
    path = tmp_path / "contract.json"
    save_contract(path, contract)
    return path, contract, [json.loads(line) for line in bank.read_text().splitlines()]


def save_bank(path: Path, contract: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    bank = Path(contract["bank"]["path"])
    bank.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    contract["bank"] = pin(bank)
    save_contract(path, contract)


@pytest.mark.parametrize("loop", ["rolling", "chunked"])
def test_explicit_contract_cli_reports_only_complete_selected_temperature_contrast(
    calibration, loop: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    path, contract, rows = calibration
    contract["loop"] = loop
    for row in rows[1:]:
        row["loop"] = loop
    save_bank(path, contract, rows)
    option = [f"--calibration-contract={path}"] if loop == "chunked" else ["--calibration-contract", str(path)]
    monkeypatch.setattr("sys.argv", ["reader", *option])
    tool.main()
    result = json.loads(capsys.readouterr().out)
    assert result["bank_complete"]
    assert result["checkpoint_content_verified_now"]
    assert result["checkpoint"] == contract["checkpoint"]
    assert result["loop"] == loop
    assert result["candidate_prior_temperature"] == 0.7
    assert result["reference_prior_temperature"] == 1.3
    assert result["cell"]["settings"] == contract["expected_settings"]
    assert result["cell"]["result"]["pairs"] == 4
    assert result["cell"]["result"]["score"] == 0.5
    assert not result["launch_qualification_verified"]
    assert "training improvement" in " ".join(result["limitations"])


@pytest.mark.parametrize("mutation", ["candidate", "reference", "both_checkpoints", "candidate_prior",
                                     "reference_prior", "other_search", "other_search_type", "candidate_volatility", "fingerprint", "opening_index",
                                     "loop", "execution", "opening_order", "missing_pair", "orphan",
                                     "duplicate_pair", "sequential"])
def test_rebound_bank_still_refuses_wrong_protocol_or_incomplete_fixed_pairs(calibration, mutation: str) -> None:
    path, contract, rows = calibration
    settings = rows[0]["settings"]
    if mutation in ("candidate", "reference", "both_checkpoints"):
        for side in ("candidate", "reference") if mutation == "both_checkpoints" else (mutation,):
            settings[side] = str(path.parent / "wrong.pt")
    elif mutation.endswith("_prior"):
        side = mutation.removesuffix("_prior")
        settings[f"search_{side}"]["gumbel"]["policy_temp"] = 0.9
    elif mutation == "other_search":
        settings["search_reference"]["gumbel"]["c_scale"] = 2.0
    elif mutation == "other_search_type":
        settings["search_reference"]["vloss_weight"] = True
    elif mutation == "candidate_volatility":
        settings["volatility_candidate"] = {"volatility_q_scale": 0.5, "volatility_fpu": 0,
                                            "volatility_anchor": 0}
    elif mutation == "opening_index":
        rows[1]["opening_index"] = False
    elif mutation == "loop":
        rows[1]["loop"] = "chunked"
    elif mutation == "execution":
        for row in rows[1:]:
            row["compile"] = "off"
    elif mutation == "opening_order":
        rows[1]["opening_fen"] = rows[1]["start_fen"] = rows[3]["opening_fen"]
        rows[2]["opening_fen"] = rows[2]["start_fen"] = rows[3]["opening_fen"]
    elif mutation == "missing_pair":
        # Native fixed-N max_seconds can exit0 here. Bank completeness must
        # independently refuse even when a terminal process claims success.
        (path.parent / "process.json").write_text('{"returncode":0}\n')
        rows = rows[:-2]
    elif mutation == "orphan":
        rows = rows[:-1]
    elif mutation == "duplicate_pair":
        rows += copy.deepcopy(rows[1:3])
    elif mutation == "sequential":
        rows[0]["info"] = {"sprt": {"verdict": "H1"}}
    rows[0]["fingerprint"] = "wrong" if mutation == "fingerprint" else tool.settings_fingerprint(settings)
    contract["expected_settings"] = settings  # Keep pins/settings consistent to reach semantic checks.
    save_bank(path, contract, rows)
    reason = {
        "candidate": "calibration candidate is not", "reference": "reference is not",
        "both_checkpoints": "reference is not", "candidate_prior": "must differ only",
        "reference_prior": "must differ only", "other_search": "must differ only",
        "other_search_type": "must differ only",
        "candidate_volatility": "calibration setting volatility_candidate differs",
        "fingerprint": "header fingerprint", "opening_index": "calibration opening index differs",
        "loop": "seed/loop differs", "execution": "calibration execution differs",
        "opening_order": "calibration canonical opening sequence differs",
        "missing_pair": "exactly the fixed game count", "orphan": "exactly the fixed game count",
        "duplicate_pair": "exactly the fixed game count", "sequential": "sequential stopping",
    }[mutation]
    with pytest.raises(ValueError, match=reason):
        tool.read_calibration_contract(path)


@pytest.mark.parametrize(("field", "value"), [("candidate_prior_temperature", float("nan")),
                                              ("reference_prior_temperature", float("inf")),
                                              ("candidate_prior_temperature", True),
                                              ("reference_prior_temperature", 0.0),
                                              ("sims", True), ("expected_pairs", 1)])
def test_invalid_contract_numbers_refused(calibration, field: str, value: Any) -> None:
    path, contract, _ = calibration
    contract[field] = value
    save_contract(path, contract)
    with pytest.raises(ValueError, match=r"calibration priors|calibration requires positive"):
        tool.read_calibration_contract(path)


@pytest.mark.parametrize("changed", ["checkpoint", "bank", "panel_history", "torn_tail"])
def test_content_pins_and_full_history_are_checked(calibration, changed: str) -> None:
    path, contract, _ = calibration
    if changed == "checkpoint":
        Path(contract["checkpoint"]["path"]).write_bytes(b"different checkpoint")
    elif changed == "bank":
        contract["bank"]["sha256"] = "0" * 64
        save_contract(path, contract)
    elif changed == "panel_history":
        panel_path = Path(contract["opening_panel"]["path"])
        panel = json.loads(panel_path.read_text())
        panel[0]["moves"][0] = "a1a8"
        panel_path.write_text(json.dumps(panel))
        contract["opening_panel"] = pin(panel_path)
        save_contract(path, contract)
    else:
        bank = Path(contract["bank"]["path"])
        bank.write_text(bank.read_text() + '{"kind":')
        contract["bank"] = pin(bank)
        save_contract(path, contract)
    reason = {"checkpoint": "calibration checkpoint content differs", "bank": "changed pin",
              "panel_history": "illegal uci", "torn_tail": "complete game bank"}[changed]
    with pytest.raises(ValueError, match=reason):
        tool.read_calibration_contract(path)


@pytest.mark.parametrize("extra", [["--profile", "calibration"], ["--seed", "42"],
                                   ["--prior-temperature", "1.0"]])
def test_explicit_contract_does_not_relax_legacy_reader(
    calibration, extra: list[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, contract, _ = calibration
    with pytest.raises(ValueError, match="off-protocol settings"):
        tool.read_arm(Path(contract["bank"]["path"]), reference=Path(contract["checkpoint"]["path"]),
                      seed=42, calibration=True)
    monkeypatch.setattr("sys.argv", ["reader", "--calibration-contract", str(path), *extra])
    with pytest.raises(SystemExit):
        tool.main()


@pytest.mark.parametrize("kind", ["abbreviated", "duplicate"])
def test_explicit_contract_cli_rejects_ambiguous_selection(calibration, kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    path, _, _ = calibration
    option = ["--calibration-co", str(path)] if kind == "abbreviated" else ["--calibration-contract", str(path)] * 2
    monkeypatch.setattr("sys.argv", ["reader", *option])
    with pytest.raises(SystemExit):
        tool.main()
