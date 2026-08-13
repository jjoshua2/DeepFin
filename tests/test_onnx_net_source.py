"""The rulers' ``--onnx`` input path (scripts/net_source.py).

The defect this file is aimed at is NOT "does the code run" — it is a flag that
is accepted and then silently ignored. So the tests assert, in order:

* exactly one net source, with both/neither refused;
* the ONNX graph's tensor names are read (and ambiguity refused) rather than
  guessed;
* the ``--onnx`` path never falls back to a checkpoint;
* the policy the ruler ends up reading came through the BOARD-AWARE Leela
  remap, proven per legal move against the independent reference in
  ``moves/leela_index.py`` — a static table gives a different, non-crashing
  answer here, which is exactly how the O-O prior was once read 49-120x too
  small;
* ``scripts/value_regret.py --onnx`` scores that net end to end from its real
  ``main()``, on a frozen-audit-set file, with the checkpoint loader booby
  trapped.

The ONNX fixture is a tiny hand-built graph, not BT4: its policy output is
``[0, 1, ..., 1857]`` for every row, so "which Leela slot did the ruler read"
is directly observable in the returned logits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.encoding.model_inputs import encode_position_for_model
from chess_anti_engine.moves import COMPACT_POLICY_SIZE
from chess_anti_engine.moves.encode import move_to_index
from chess_anti_engine.moves.lc0_1858_movestrs import LC0_1858_UCI_TO_IDX
from chess_anti_engine.moves.leela_index import compact_index_for_move, leela_index_for_move
from scripts.net_source import (
    CPU_PROVIDER,
    CUDA_PROVIDER,
    NetSource,
    OnnxNetSpec,
    net_source_from_args,
    onnx_providers_for_device,
    resolve_onnx_spec,
)

LC0_PLANES = 112


class _IndexEchoNet(torch.nn.Module):
    """policy[b, i] = i; wdl = softmax of a position-dependent logit triple.

    The policy makes the Leela->ours permutation readable off the output. The
    WDL is deliberately emitted as PROBABILITIES, the way LC0/Ceres value heads
    do, so the adapter's probs->log-probs branch is the one under test.
    """

    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = planes.sum(dim=(1, 2, 3), keepdim=False)
        zero = rows * 0.0
        pol = torch.arange(COMPACT_POLICY_SIZE, dtype=torch.float32) + zero[:, None]
        wdl_logits = torch.stack([rows * 0.01, zero, -rows * 0.01], dim=-1)
        return pol, torch.softmax(wdl_logits, dim=-1)


class _TwoPolicyHeadNet(torch.nn.Module):
    """Two 1858-wide outputs: the ambiguity a resolver must refuse to guess at."""

    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pol, wdl = _IndexEchoNet().forward(planes)
        return pol, pol * 2.0, wdl


class _NoPolicyNet(torch.nn.Module):
    """A graph with no 1858-wide head at all."""

    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = planes.sum(dim=(1, 2, 3))
        return rows[:, None].expand(-1, 64), torch.softmax(
            torch.stack([rows, rows * 0.0, -rows], dim=-1), dim=-1,
        )


def _export(net: torch.nn.Module, path: Path, output_names: list[str]) -> Path:
    torch.onnx.export(
        net,
        (torch.zeros(2, LC0_PLANES, 8, 8),),
        str(path),
        input_names=["planes"],
        output_names=output_names,
        dynamic_axes={n: {0: "batch"} for n in ["planes", *output_names]},
        opset_version=17,
        dynamo=False,
    )
    return path


@pytest.fixture(scope="module")
def echo_onnx(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _export(
        _IndexEchoNet(), tmp_path_factory.mktemp("onnx") / "echo.onnx", ["policy", "wdl"],
    )


# --------------------------------------------------------------------------
# exactly one source, never a silent default
# --------------------------------------------------------------------------


def test_net_source_refuses_both_sources(echo_onnx: Path) -> None:
    spec = resolve_onnx_spec(echo_onnx)
    with pytest.raises(SystemExit, match="exactly one"):
        NetSource(checkpoint="ckpt", onnx=spec)


def test_net_source_refuses_neither_source() -> None:
    with pytest.raises(SystemExit, match="neither was given"):
        NetSource()


def test_args_with_both_flags_are_refused_before_the_graph_is_opened() -> None:
    args = _args(checkpoint="ckpt", onnx=Path("/nonexistent/never-opened.onnx"))
    with pytest.raises(SystemExit, match="exactly one"):
        net_source_from_args(args)


def test_a_name_override_without_onnx_is_refused_not_ignored() -> None:
    """The signature defect, in miniature: a value accepted and dropped."""
    args = _args(checkpoint="ckpt", onnx_policy_output="/output/policy")
    with pytest.raises(SystemExit, match="--onnx-policy-output given without --onnx"):
        net_source_from_args(args)


def test_a_missing_onnx_file_raises_rather_than_falling_back() -> None:
    with pytest.raises(SystemExit, match="no such file"):
        resolve_onnx_spec("/nonexistent/not-a-net.onnx")


def _args(**kw: object) -> argparse.Namespace:
    """A parsed-CLI stand-in carrying exactly the flags `add_net_source_args` adds."""
    defaults: dict[str, object] = {
        "checkpoint": None,
        "onnx": None,
        "onnx_input_name": None,
        "onnx_policy_output": None,
        "onnx_wdl_output": None,
    }
    return argparse.Namespace(**{**defaults, **kw})


# --------------------------------------------------------------------------
# graph resolution: read, don't guess
# --------------------------------------------------------------------------


def test_resolve_reads_the_names_off_the_graph(echo_onnx: Path) -> None:
    spec = resolve_onnx_spec(echo_onnx)
    assert (spec.input_name, spec.policy_output, spec.wdl_output) == (
        "planes", "policy", "wdl",
    )
    assert str(echo_onnx) in spec.label


def test_two_policy_heads_raise_instead_of_picking_one(tmp_path: Path) -> None:
    path = _export(
        _TwoPolicyHeadNet(), tmp_path / "two.onnx", ["policy", "policy2", "wdl"],
    )
    with pytest.raises(SystemExit, match="2 candidate policy outputs"):
        resolve_onnx_spec(path)
    # ...and naming one resolves it, so the guard is not just a blanket refusal.
    spec = resolve_onnx_spec(path, policy_output="policy2")
    assert spec.policy_output == "policy2"


def test_a_graph_without_an_1858_head_raises(tmp_path: Path) -> None:
    path = _export(_NoPolicyNet(), tmp_path / "nopolicy.onnx", ["notpolicy", "wdl"])
    with pytest.raises(SystemExit, match="no 1858-wide policy output"):
        resolve_onnx_spec(path)


def test_an_unknown_explicit_output_name_raises(echo_onnx: Path) -> None:
    with pytest.raises(SystemExit, match="is not an output of this graph"):
        resolve_onnx_spec(echo_onnx, policy_output="/output/policy")


def test_cpu_device_never_offers_ort_the_cuda_provider() -> None:
    """`--device cpu` must be unable to allocate on a training GPU, not merely
    prefer not to."""
    assert onnx_providers_for_device("cpu") == (CPU_PROVIDER,)
    assert onnx_providers_for_device("cuda:1") == (CUDA_PROVIDER, CPU_PROVIDER)


# --------------------------------------------------------------------------
# loading: the requested net, or nothing
# --------------------------------------------------------------------------


def test_the_onnx_path_never_loads_a_checkpoint(
    echo_onnx: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chess_anti_engine.uci import model_loader

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("--onnx fell back to the checkpoint loader")

    monkeypatch.setattr(model_loader, "load_model_from_checkpoint", _boom)
    model = NetSource(onnx=resolve_onnx_spec(echo_onnx)).load(device="cpu")
    # The adapter must DECLARE LC0's input contract, or every caller encodes
    # production planes and the 112-plane slice silently means something else.
    assert model.input_history_encoding == "lc0_root"
    assert model.input_extra_features == "v1"
    assert model.policy_encoding == "az_4672"
    assert not model.training


def test_the_checkpoint_path_still_goes_through_the_checkpoint_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chess_anti_engine.uci import model_loader

    seen: list[str] = []

    class _Stub(torch.nn.Module):
        pass

    def _fake(path: str, **_k: object) -> torch.nn.Module:
        seen.append(path)
        return _Stub()

    monkeypatch.setattr(model_loader, "load_model_from_checkpoint", _fake)
    model = NetSource(checkpoint="some/trainer.pt").load(device="cpu")
    assert seen == ["some/trainer.pt"]
    assert not model.training


# --------------------------------------------------------------------------
# the policy actually read is the board-aware remap
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fen",
    [
        # White to move, both castles legal, and a 7th-rank pawn plus a rook
        # that can slide onto the 8th rank -- the two families a static table
        # cannot map.
        "r3k2r/PP4pp/8/8/8/8/6PP/R3K2R w KQkq - 0 1",
        # Same, black to move: proves the orientation flip is applied too.
        "r3k2r/pp4pp/8/8/8/8/6PP/R3K2R b KQkq - 0 1",
    ],
)
def test_every_legal_move_reads_its_leela_slot(echo_onnx: Path, fen: str) -> None:
    """policy[i] == i in the graph, so the returned logit IS the Leela slot read.

    Compared per move against ``leela_index_for_move``, which derives the slot
    from the move itself rather than from the compact tables under test.
    """
    model = NetSource(onnx=resolve_onnx_spec(echo_onnx)).load(device="cpu")
    board = chess.Board(fen)
    x = encode_position_for_model(model, board)
    out = model(torch.from_numpy(np.asarray(x, dtype=np.float32))[None, ...])
    pol = out["policy_own"][0].numpy()

    checked = 0
    castles = 0
    for move in board.legal_moves:
        full = move_to_index(move, board)
        if full < 0:
            continue
        want = leela_index_for_move(board, move)
        assert want >= 0
        assert float(pol[full]) == pytest.approx(float(want)), (
            f"{move.uci()} read Leela slot {pol[full]}, expected {want}"
        )
        checked += 1
        castles += int(board.is_castling(move))
    # Both fens keep all four castling rights, so a remap that dropped the
    # castling context would have had to fail one of the assertions above.
    assert castles == 2
    assert checked >= 20


def test_castling_reads_leelas_king_takes_rook_slot_not_the_slide(echo_onnx: Path) -> None:
    """The measured 49-120x castling-prior error, as an assertion.

    Leela's table has BOTH an ``e1g1`` slide entry and the ``e1h1`` castling
    entry, so reading the wrong one is silent. Our compact slot index differs
    from both, which is what makes this a real cross-convention check.
    """
    model = NetSource(onnx=resolve_onnx_spec(echo_onnx)).load(device="cpu")
    board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    x = encode_position_for_model(model, board)
    pol = model(torch.from_numpy(np.asarray(x, dtype=np.float32))[None, ...])["policy_own"][0]

    oo = chess.Move.from_uci("e1g1")
    read = float(pol[move_to_index(oo, board)])
    assert read == pytest.approx(float(LC0_1858_UCI_TO_IDX["e1h1"]))
    assert read != pytest.approx(float(LC0_1858_UCI_TO_IDX["e1g1"]))
    assert read != pytest.approx(float(compact_index_for_move(board, oo)))


def test_the_wdl_head_is_returned_as_logits_not_raw_probabilities(
    echo_onnx: Path,
) -> None:
    """The search softmaxes ``wdl``; feeding it probabilities crushes them."""
    model = NetSource(onnx=resolve_onnx_spec(echo_onnx)).load(device="cpu")
    x = encode_position_for_model(model, chess.Board())
    wdl = model(torch.from_numpy(np.asarray(x, dtype=np.float32))[None, ...])["wdl"]
    assert float(wdl.sum()) < 0.0  # log-probs, not a row summing to 1
    np.testing.assert_allclose(
        torch.softmax(wdl, dim=-1).sum(dim=-1).numpy(), 1.0, atol=1e-5,
    )


# --------------------------------------------------------------------------
# end to end through the ruler's real main()
# --------------------------------------------------------------------------


def _audit_row(key: str, fen: str) -> str:
    board = chess.Board(fen)
    moves = [m.uci() for m in board.legal_moves][:6]
    return json.dumps({
        "key": key,
        "fen": fen,
        "phase": 1,
        "source": 0,
        "multipv": [
            {"move": m, "cp": 40 - 10 * i} for i, m in enumerate(moves)
        ],
        "wdl": [400, 400, 200],
        "nodes": 1_000_000,
        "depth": 40,
    })


@pytest.fixture
def mini_audit_set(tmp_path: Path) -> Path:
    path = tmp_path / "mini_audit.jsonl"
    path.write_text(
        "\n".join([
            _audit_row("a", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
            _audit_row("b", "r3k2r/pp3ppp/2n5/3q4/8/2N5/PPP2PPP/R3K2R w KQkq - 0 12"),
        ]) + "\n",
        encoding="utf-8",
    )
    return path


def test_value_regret_main_scores_the_onnx_net(
    echo_onnx: Path,
    mini_audit_set: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The whole CLI: --onnx in, a number out, and the checkpoint loader armed."""
    from chess_anti_engine.uci import model_loader
    from scripts import value_regret

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("value_regret --onnx loaded a checkpoint")

    dump = tmp_path / "per_position.jsonl"
    monkeypatch.setattr(model_loader, "load_model_from_checkpoint", _boom)
    monkeypatch.setattr("sys.argv", [
        "value_regret.py",
        "--onnx", str(echo_onnx),
        "--audit-set", str(mini_audit_set),
        "--device", "cpu",
        "--batch-size", "8",
        "--min-pieces", "0",
        "--dump-per-position", str(dump),
    ])
    value_regret.main()
    out = capsys.readouterr().out
    assert "OVERALL" in out
    assert "cp (n=2)" in out
    # ⚑ The RESULT HEADER must name the net, not merely some earlier progress
    # line: a header reading "@ <checkpoint>" over an ONNX number is the whole
    # failure mode. Checked on the header line itself, because an
    # `f"onnx:..." in out` assertion passes on the [net-source] echo alone
    # (measured: that version survived deleting `net.label` from the header).
    headers = [ln for ln in out.splitlines() if ln.startswith("=== value-head")]
    assert len(headers) == 1
    assert f"@ onnx:{echo_onnx}" in headers[0]
    # A dump is a report too: every row must carry the net that produced it.
    rows = [json.loads(ln) for ln in dump.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert {r["net"] for r in rows} == {f"onnx:{echo_onnx} "
                                        f"[in=planes policy=policy wdl=wdl]"}


def test_value_regret_main_refuses_a_missing_net(
    mini_audit_set: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import value_regret

    monkeypatch.setattr("sys.argv", [
        "value_regret.py", "--audit-set", str(mini_audit_set), "--device", "cpu",
    ])
    with pytest.raises(SystemExit, match="exactly one"):
        value_regret.main()


def test_both_rulers_resolve_their_net_before_any_scoring_work() -> None:
    """A late resolve would spend the audit set — or an hour of SF — first.

    ``_shallow_sf_records`` is the audit's Stockfish labelling pass; a bad
    ``--onnx`` surfacing after it is exactly the failure the other fail-fast
    guards in that ``main()`` were written for.
    """
    import inspect

    from scripts import audit_targets, value_regret

    expensive = {
        value_regret: "load_audit_set(",
        audit_targets: "_shallow_sf_records(",
    }
    for module, costly_call in expensive.items():
        src = inspect.getsource(module.main)
        assert "net_source_from_args(args)" in src
        parse_at = src.index("ap.parse_args()")
        resolve_at = src.index("net_source_from_args(args)")
        costly_at = src.index(costly_call)
        assert parse_at < resolve_at < costly_at


def test_audit_targets_net_candidates_takes_a_net_source_not_a_checkpoint() -> None:
    """No ``checkpoint=`` back door, and no default that could load something else."""
    import inspect

    from scripts import audit_targets

    params = inspect.signature(audit_targets._net_candidates).parameters
    assert "checkpoint" not in params
    assert params["net"].default is inspect.Parameter.empty
    assert params["net"].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_onnx_spec_label_names_every_tensor_it_resolved() -> None:
    spec = OnnxNetSpec(
        path=Path("/tmp/x.onnx"), input_name="i", policy_output="p", wdl_output="w",
    )
    assert spec.label == "onnx:/tmp/x.onnx [in=i policy=p wdl=w]"
