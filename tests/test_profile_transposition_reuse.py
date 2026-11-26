"""The probe must preserve histories, expose its limits and run real C search."""
from __future__ import annotations

import ast
import json
from dataclasses import asdict
from types import SimpleNamespace
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.eval.model_provenance import sha256_file
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig, build_model
from chess_anti_engine.selfplay.network_turn import build_selfplay_search_shape
from chess_anti_engine.simple_rl.collect import CollectSettings, build_configs
from scripts import profile_transposition_reuse as probe


def _roots(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_root_reader_keeps_move_order_and_complete_history(tmp_path: Path) -> None:
    path = _roots(tmp_path / "roots.jsonl", [
        {"id": "knights-a", "moves": ["g1f3", "g8f6", "b1c3", "b8c6"]},
        {"id": "knights-b", "moves": ["b1c3", "b8c6", "g1f3", "g8f6"]},
    ])
    roots = probe.load_roots(path, 2)
    assert roots[0][1].fen() == roots[1][1].fen()
    assert roots[0][1].move_stack != roots[1][1].move_stack
    assert len(roots[0][1].move_stack) == 4
    assert [key for key, _ in probe.load_roots(path, 1)] == ["knights-a"]


@pytest.mark.parametrize("row", [
    {"id": "fen-only", "fen": chess.STARTING_FEN},
    {"id": "missing-history"},
    {"id": "null", "moves": ["0000"]},
    {"id": "illegal", "moves": ["e2e5"]},
    {"id": "wrong-type", "moves": [2]},
    {"id": "ended", "moves": ["f2f3", "e7e5", "g2g4", "d8h4"]},
    # Threefold repetition: claimable, not automatically over. Only claim_draw=True refuses it.
    {"id": "claimable-draw", "moves": ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]},
])
def test_root_reader_refuses_incomplete_or_illegal_roots(tmp_path: Path, row: dict) -> None:
    path = _roots(tmp_path / "roots.jsonl", [row])
    with pytest.raises(ValueError, match="line 1"):
        probe.load_roots(path, 2)


def test_counter_delta_does_not_reset_or_hide_unavailable_counts() -> None:
    before = {"probe_hits": 7, "reuse": 2}
    assert probe.counter_delta(before, {"probe_hits": 10, "reuse": 3}) == {"probe_hits": 3, "reuse": 1}
    assert before == {"probe_hits": 7, "reuse": 2}
    with pytest.raises(RuntimeError, match="decreased"):
        probe.counter_delta(before, {"probe_hits": 1, "reuse": 1})
    with pytest.raises(RuntimeError, match="schema"):
        probe.counter_delta(before, {"probe_hits": 10})


def test_profile_preserves_full_real_search_output() -> None:
    rep_fix.apply(False)
    arch = {"input_history_encoding": "lc0_root", "input_extra_features": "v2_threats"}
    configs = build_configs(CollectSettings(games=1, sims=64, vloss_weight=1), arch, Path("unused"))
    shape = build_selfplay_search_shape(search=configs["search"], game=configs["game"], simulations=64)
    board = chess.Board()
    board.push_uci("e2e4")
    expected = run_gumbel_root_many_c(
        None, [board], device="cpu", rng=np.random.default_rng(42),
        evaluator=probe.SyntheticEvaluator(), per_game_add_noise=[True],
        per_game_gumbel_scale=[1.0], allow_terminal_root_shortcuts=True,
        **shape.runner_kwargs(),
    )
    got = probe.profile_root(
        board, shape=shape, model=None, evaluator=probe.SyntheticEvaluator(),
        seed=42, noise_scale=1.0, max_cached_boards=8192,
    )
    output = got["search_output"]
    np.testing.assert_array_equal(output["policy"], expected[0][0])
    np.testing.assert_array_equal(output["legal_mask"], expected[3][0])
    assert output["action"] == expected[1][0]
    assert output["root_value"] == expected[2][0]
    assert got["tree_nodes"] == expected[4].node_count()
    assert got["census"]["cached_boards"] > 1
    assert got["census"]["complete"] is True
    assert got["realized_shape"]["vloss_weight"] == 1
    assert got["realized_shape"]["allow_terminal_root_shortcuts"] is True
    assert board.move_stack == [chess.Move.from_uci("e2e4")]


def test_synthetic_cli_is_explicit_and_reproducible(tmp_path: Path) -> None:
    paths = [tmp_path / "one.json", tmp_path / "two.json"]
    for path in paths:
        probe.main(["--synthetic-smoke", "--simulations", "32", "--output", str(path)])
    first, second = [json.loads(path.read_text()) for path in paths]
    assert first["model"] == {"kind": "synthetic_fixture", "production_evidence": False}
    assert first["runtime"]["tree_reuse"] is False
    assert first["runtime"]["eval_cache"] is False
    assert first["results"][0]["search_output_sha256"] == second["results"][0]["search_output_sha256"]
    assert first["results"][0]["census"] == second["results"][0]["census"]
    assert first["results"][0]["tt_counts"] == second["results"][0]["tt_counts"]
    assert len(first["native"]) == 2


def test_checkpoint_cli_loads_actual_model_and_pins_history_profile(tmp_path: Path) -> None:
    cfg = ModelConfig(kind="tiny", input_history_encoding="lc0_root_legacy_meta", history_rep_fix=True)
    model = build_model(cfg)
    checkpoint = tmp_path / "trainer.pt"
    torch.save({"model": model.state_dict(), "arch": {"_schema_version": ARCH_SCHEMA_VERSION, **asdict(cfg)}}, checkpoint)
    roots = _roots(tmp_path / "roots.jsonl", [{"id": "e4-e5", "moves": ["e2e4", "e7e5"]}])
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"games": 1, "vloss_weight": 1, "topk": 8, "c_scale": 0.2}))
    output = tmp_path / "report.json"
    probe.main(["--checkpoint", str(checkpoint), "--roots", str(roots), "--collect-settings", str(settings),
                "--simulations", "32", "--noise-scale", "0", "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["model"]["sha256"] == sha256_file(checkpoint)
    assert report["roots"]["sha256"] == sha256_file(roots)
    assert report["runtime"]["history_rep_fix"] is True
    row = report["results"][0]
    assert row["moves"] == ["e2e4", "e7e5"]
    assert row["realized_shape"]["cfg"]["input_history_encoding"] == "lc0_root_legacy_meta"
    assert row["realized_shape"]["cfg"]["c_scale"] == 0.2
    assert row["realized_shape"]["cfg"]["topk"] == 8
    assert row["census"]["input_history_mode"] == 2
    assert row["census"]["cached_boards"] > 1


@pytest.mark.parametrize("flag", ["use_dynamic_relations", "policy_dynamic_relations"])
def test_checkpoint_cli_refuses_to_drop_relation_inputs(
    tmp_path: Path, flag: str, capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint = tmp_path / "relations.pt"
    arch = asdict(ModelConfig())
    arch[flag] = True
    torch.save({"arch": arch}, checkpoint)
    roots = _roots(tmp_path / "roots.jsonl", [{"id": "initial", "moves": []}])
    settings = tmp_path / "settings.json"
    settings.write_text('{"games":1}')
    output = tmp_path / "report.json"
    with pytest.raises(SystemExit):
        probe.main(["--checkpoint", str(checkpoint), "--roots", str(roots),
                    "--collect-settings", str(settings), "--output", str(output)])
    assert "refusing to drop their auxiliary inputs" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("arguments", [
    ["--checkpoint", "missing.pt"],
    ["--synthetic-smoke", "--max-roots", "0"],
    ["--synthetic-smoke", "--simulations", "4097"],
    ["--synthetic-smoke", "--simulations", "32", "32"],
    ["--synthetic-smoke", "--noise-scale", "nan"],
])
def test_cli_refuses_unbounded_or_underspecified_run(tmp_path: Path, arguments: list[str]) -> None:
    output = tmp_path / "report.json"
    with pytest.raises(SystemExit):
        probe.main([*arguments, "--output", str(output)])
    assert not output.exists()


def _call_keyword_literals(path: Path, keyword: str) -> list[object]:
    """Values passed as `keyword` by every call in a source file (constants unwrapped).

    Matched by keyword, not by callee name: production calls the runner through a
    local alias (`gumbel_c_fn = cast(Any, _run_gumbel_root_many_c)`).
    """
    values: list[object] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        values.extend(
            kw.value.value if isinstance(kw.value, ast.Constant) else ast.dump(kw.value)
            for kw in node.keywords if kw.arg == keyword
        )
    return values


def test_root_shortcut_flag_matches_the_production_call_site() -> None:
    # Compare the driver with production's own literal, not with itself: if either side
    # changes, the driver no longer reproduces production's root behaviour.
    repo = Path(__file__).resolve().parents[1]
    production = _call_keyword_literals(
        repo / "chess_anti_engine/selfplay/network_turn.py", "allow_terminal_root_shortcuts")
    assert len(production) == 1, production
    assert isinstance(production[0], bool), production
    assert probe.ALLOW_TERMINAL_ROOT_SHORTCUTS is production[0]
    driver = _call_keyword_literals(
        repo / "scripts/profile_transposition_reuse.py", "allow_terminal_root_shortcuts")
    assert driver == [ast.dump(ast.Name(id="ALLOW_TERMINAL_ROOT_SHORTCUTS", ctx=ast.Load()))], driver


def _tiny_checkpoint(tmp_path: Path) -> Path:
    cfg = ModelConfig(kind="tiny", input_history_encoding="lc0_root_legacy_meta", history_rep_fix=True)
    checkpoint = tmp_path / "trainer.pt"
    torch.save({"model": build_model(cfg).state_dict(),
                "arch": {"_schema_version": ARCH_SCHEMA_VERSION, **asdict(cfg)}}, checkpoint)
    return checkpoint


def test_checkpoint_cli_refuses_a_tablebase_profile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    roots = _roots(tmp_path / "roots.jsonl", [{"id": "e4", "moves": ["e2e4"]}])
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"games": 1, "syzygy": str(tmp_path / "tb")}))
    output = tmp_path / "report.json"
    with pytest.raises(SystemExit):
        probe.main(["--checkpoint", str(_tiny_checkpoint(tmp_path)), "--roots", str(roots),
                    "--collect-settings", str(settings), "--simulations", "16", "--output", str(output)])
    assert "tablebase-backed collection is outside this CPU probe" in capsys.readouterr().err
    assert not output.exists()


def test_cli_refuses_to_overwrite_an_existing_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "report.json"
    output.write_text("earlier report\n")
    with pytest.raises(SystemExit):
        probe.main(["--synthetic-smoke", "--simulations", "16", "--output", str(output)])
    assert "output already exists" in capsys.readouterr().err
    assert output.read_text() == "earlier report\n"


def test_profile_refuses_when_tt_counters_move_during_the_census(monkeypatch: pytest.MonkeyPatch) -> None:
    rep_fix.apply(False)
    arch = {"input_history_encoding": "lc0_root", "input_extra_features": "v2_threats"}
    configs = build_configs(CollectSettings(games=1, sims=16, vloss_weight=1), arch, Path("unused"))
    shape = build_selfplay_search_shape(search=configs["search"], game=configs["game"], simulations=16)
    real = probe._mcts_tree
    calls = {"n": 0}

    def tt_stats() -> dict[str, int]:
        calls["n"] += 1
        stats = dict(real.tt_stats())
        if calls["n"] >= 3:  # before, after, then the post-census check: another caller moved them
            stats["probe_hits"] = stats["probe_hits"] + 1
        return stats

    monkeypatch.setattr(probe, "_mcts_tree", SimpleNamespace(MCTSTree=real.MCTSTree, tt_stats=tt_stats))
    with pytest.raises(RuntimeError, match="changed during census"):
        probe.profile_root(chess.Board(), shape=shape, model=None, evaluator=probe.SyntheticEvaluator(),
                           seed=1, noise_scale=1.0, max_cached_boards=8192)
    assert calls["n"] == 3
