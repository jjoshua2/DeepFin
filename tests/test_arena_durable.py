"""CPU-only crash and identity checks for the opt-in paired arena."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import chess
import pytest

from chess_anti_engine.eval import arena_durable as durable
from chess_anti_engine.eval.arena_pgn import ArenaGame, ArenaPgnWriter
from chess_anti_engine.utils.game_log import GameLogWriter, read_game_log


def _write_half(
    pgn: ArenaPgnWriter, log: GameLogWriter, receipts: durable.DurablePairReceipts,
    pair_id: int, half: int,
) -> None:
    pgn_offset = pgn.path.stat().st_size
    pgn.write_game(ArenaGame(
        white="A" if half == 0 else "B",
        black="B" if half == 0 else "A",
        result="1-0" if half == 0 else "0-1",
        moves=(chess.Move.from_uci("e2e4"),),
        pair_id=pair_id, pair_half=half,
    ))
    pgn_span = durable.file_span(
        pgn.path, pgn_offset, pgn.path.stat().st_size - pgn_offset,
    )
    log_offset = log.path.stat().st_size
    log.write_game({
        "pair_id": pair_id, "half": half, "a_is_white": half == 0,
        "opening_fen": chess.Board().fen(),
        "result": "1-0" if half == 0 else "0-1",
        "durable_pgn_span": pgn_span,
    })
    log_span = durable.file_span(
        log.path, log_offset, log.path.stat().st_size - log_offset,
    )
    receipts.record_game(
        pair_id, half, jsonl_span=log_span, pgn_span=pgn_span,
    )


def test_kill_resume_pair_receipts_orphan_tail_and_union(tmp_path: Path) -> None:
    pgn_path = tmp_path / "games.pgn"
    log_path = tmp_path / "games.jsonl"
    receipt_path = tmp_path / "pairs"
    seal = "a" * 64
    receipts = durable.DurablePairReceipts(
        receipt_path, seal_sha256=seal, log=log_path, pgn=pgn_path,
    )
    with (
        ArenaPgnWriter(pgn_path, durable=True) as pgn,
        GameLogWriter(log_path, driver="arena_standard", settings={"seal": seal},
                      durable=True) as log,
    ):
        _write_half(pgn, log, receipts, 0, 0)
        assert not list(receipt_path.glob("pair_*.json"))
        _write_half(pgn, log, receipts, 0, 1)
        _write_half(pgn, log, receipts, 1, 0)  # orphan at kill
    receipts.close()
    assert [p.name for p in receipt_path.glob("pair_*.json")] == ["pair_000000.json"]
    with log_path.open("ab") as fh:
        fh.write(b'{"kind":"game","pair_id":1,"half":')
        fh.flush()
        os.fsync(fh.fileno())
    assert read_game_log(log_path).truncated_tail
    # A separate kill can land after PGN fsync but before JSONL fsync. This
    # PGN-only game is not eligible for scoring or a pair receipt.
    with ArenaPgnWriter(pgn_path, durable=True) as pgn:
        pgn.write_game(ArenaGame(
            white="A", black="B", result="1-0", pair_id=1, pair_half=0,
        ))
    from scripts.arena_standard import load_arena_resume

    openings = [chess.Board(), chess.Board()]
    interrupted = load_arena_resume(
        log_path, settings={"seal": seal}, openings=openings,
    )
    assert interrupted.complete_pair_ids == [0]
    assert interrupted.orphan_pair_ids == [1]

    resumed = durable.DurablePairReceipts(
        receipt_path, seal_sha256=seal, log=log_path, pgn=pgn_path,
    )
    resumed.recover_and_verify([0])
    with (
        ArenaPgnWriter(pgn_path, durable=True) as pgn,
        GameLogWriter(log_path, driver="arena_standard", settings={"seal": seal},
                      resuming=True, durable=True) as log,
    ):
        assert log.repaired_truncated_tail
        _write_half(pgn, log, resumed, 1, 0)
        _write_half(pgn, log, resumed, 1, 1)
    resumed.recover_and_verify([0, 1])
    assert sorted(p.name for p in receipt_path.glob("pair_*.json")) == [
        "pair_000000.json", "pair_000001.json",
    ]
    assert len(read_game_log(log_path).games) == 5  # orphan plus replay
    assert pgn_path.read_text().count('[PairId "1"]') == 4
    finished = load_arena_resume(
        log_path, settings={"seal": seal}, openings=openings,
    )
    assert finished.complete_pair_ids == [0, 1]
    assert finished.orphan_pair_ids == []
    assert finished.pair_scores == [2.0, 2.0]
    resumed.close()


def test_receipt_gap_recovered_only_from_durable_rows(tmp_path: Path) -> None:
    pgn_path, log_path, receipt_path = (
        tmp_path / "games.pgn", tmp_path / "games.jsonl", tmp_path / "pairs",
    )
    receipts = durable.DurablePairReceipts(
        receipt_path, seal_sha256="a" * 64, log=log_path, pgn=pgn_path,
    )
    with (
        ArenaPgnWriter(pgn_path, durable=True) as pgn,
        GameLogWriter(log_path, driver="arena_standard", settings={"k": 1},
                      durable=True) as log,
    ):
        _write_half(pgn, log, receipts, 0, 0)
        _write_half(pgn, log, receipts, 0, 1)
    receipts.close()
    receipt = receipt_path / "pair_000000.json"
    expected = receipt.read_bytes()
    receipt.unlink()  # crash after second JSONL fsync, before atomic receipt
    recovered = durable.DurablePairReceipts(
        receipt_path, seal_sha256="a" * 64, log=log_path, pgn=pgn_path,
    )
    recovered.recover_and_verify([0])
    assert receipt.read_bytes() == expected
    receipt.write_bytes(b"corrupt\n")
    with pytest.raises(ValueError, match="receipt changed"):
        recovered.recover_and_verify([0])
    recovered.close()


def test_changed_pgn_bytes_refuse_committed_pair(tmp_path: Path) -> None:
    pgn_path, log_path, receipt_path = (
        tmp_path / "games.pgn", tmp_path / "games.jsonl", tmp_path / "pairs",
    )
    receipts = durable.DurablePairReceipts(
        receipt_path, seal_sha256="a" * 64, log=log_path, pgn=pgn_path,
    )
    with (
        ArenaPgnWriter(pgn_path, durable=True) as pgn,
        GameLogWriter(log_path, driver="arena_standard", settings={"k": 1},
                      durable=True) as log,
    ):
        _write_half(pgn, log, receipts, 0, 0)
        _write_half(pgn, log, receipts, 0, 1)
    receipts.close()
    data = bytearray(pgn_path.read_bytes())
    data[5] ^= 1
    pgn_path.write_bytes(data)
    reopened = durable.DurablePairReceipts(
        receipt_path, seal_sha256="a" * 64, log=log_path, pgn=pgn_path,
    )
    with pytest.raises(ValueError, match="durable span changed"):
        reopened.recover_and_verify([0])
    reopened.close()


def test_durable_game_commit_fsyncs_pgn_before_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pgn_path, log_path, receipt_path = (
        tmp_path / "games.pgn", tmp_path / "games.jsonl", tmp_path / "pairs",
    )
    receipts = durable.DurablePairReceipts(
        receipt_path, seal_sha256="a" * 64, log=log_path, pgn=pgn_path,
    )
    with (
        ArenaPgnWriter(pgn_path, durable=True) as pgn,
        GameLogWriter(log_path, driver="arena_standard", settings={"k": 1},
                      durable=True) as log,
    ):
        fsynced: list[str] = []
        real_fsync = os.fsync

        def observe(fd: int) -> None:
            fsynced.append(Path(f"/proc/self/fd/{fd}").resolve().name)
            real_fsync(fd)

        monkeypatch.setattr(os, "fsync", observe)
        _write_half(pgn, log, receipts, 0, 0)
        assert fsynced == ["games.pgn", "games.jsonl"]
    receipts.close()


def test_sigkill_reuses_sealed_pair_and_replays_orphan(tmp_path: Path) -> None:
    """A different process dies after the durable commits, then we reopen."""
    root = Path(__file__).resolve().parents[1]
    pgn_path = tmp_path / "games.pgn"
    log_path = tmp_path / "games.jsonl"
    receipt_path = tmp_path / "pairs"
    ready = tmp_path / "ready"
    seal = "b" * 64
    child = """
import os, signal, sys
from pathlib import Path
from chess_anti_engine.eval.arena_durable import DurablePairReceipts
from chess_anti_engine.eval.arena_pgn import ArenaGame, ArenaPgnWriter
from chess_anti_engine.utils.game_log import GameLogWriter
from tests.test_arena_durable import _write_half
pgn_path, log_path, receipt_path, ready = map(Path, sys.argv[1:5])
seal = sys.argv[5]
receipts = DurablePairReceipts(receipt_path, seal_sha256=seal, log=log_path, pgn=pgn_path)
with ArenaPgnWriter(pgn_path, durable=True) as pgn, GameLogWriter(
    log_path, driver='arena_standard', settings={'seal': seal}, durable=True,
) as log:
    _write_half(pgn, log, receipts, 0, 0)
    _write_half(pgn, log, receipts, 0, 1)
    _write_half(pgn, log, receipts, 1, 0)
    pgn.write_game(ArenaGame(white='B', black='A', result='0-1', pair_id=1, pair_half=1))
    ready.write_text('kill after PGN fsync, before JSONL for second half')
    while True:
        signal.pause()
"""
    env = {**os.environ, "PYTHONPATH": str(root), "CUDA_VISIBLE_DEVICES": ""}
    process = subprocess.Popen(
        [sys.executable, "-c", child, str(pgn_path), str(log_path),
         str(receipt_path), str(ready), seal],
        cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 20
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists(), process.communicate(timeout=5)
        os.kill(process.pid, signal.SIGKILL)
        process.communicate(timeout=5)
        assert process.returncode == -signal.SIGKILL
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
    first_receipt = receipt_path / "pair_000000.json"
    before = first_receipt.read_bytes()
    before_mtime = first_receipt.stat().st_mtime_ns
    assert not (receipt_path / "pair_000001.json").exists()
    from scripts.arena_standard import load_arena_resume

    openings = [chess.Board(), chess.Board()]
    interrupted = load_arena_resume(
        log_path, settings={"seal": seal}, openings=openings,
    )
    assert interrupted.complete_pair_ids == [0]
    assert interrupted.orphan_pair_ids == [1]
    resumed = durable.DurablePairReceipts(
        receipt_path, seal_sha256=seal, log=log_path, pgn=pgn_path,
    )
    resumed.recover_and_verify([0])
    with (
        ArenaPgnWriter(pgn_path, durable=True) as pgn,
        GameLogWriter(log_path, driver="arena_standard", settings={"seal": seal},
                      resuming=True, durable=True) as log,
    ):
        _write_half(pgn, log, resumed, 1, 0)
        _write_half(pgn, log, resumed, 1, 1)
    resumed.recover_and_verify([0, 1])
    resumed.close()
    assert first_receipt.read_bytes() == before
    assert first_receipt.stat().st_mtime_ns == before_mtime
    finished = load_arena_resume(
        log_path, settings={"seal": seal}, openings=openings,
    )
    assert finished.complete_pair_ids == [0, 1]
    assert finished.orphan_pair_ids == []
    assert finished.pair_scores == [2.0, 2.0]


def test_run_arena_opt_in_wires_pair_receipts_and_refuses_changed_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the real orchestration/sink, with only the model/play/TB faked."""
    from chess_anti_engine.selfplay import match as match_helpers
    from chess_anti_engine import tablebase
    from chess_anti_engine.uci import model_loader
    from scripts import arena_standard as arena

    monkeypatch.setattr(match_helpers, "_HAS_GUMBEL_C", True)
    monkeypatch.setattr(
        arena, "load_fen_openings",
        lambda *_args, **_kwargs: [chess.Board() for _ in range(576)],
    )

    class FakeModel:
        use_dynamic_relations = False

    class FakeTablebase:
        def close(self) -> None:
            pass

    class FakeProbe:
        n_wdl = 1
        n_dtz = 1
        probes = 0
        hits = 0

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

    monkeypatch.setattr(model_loader, "load_model_from_checkpoint",
                        lambda *_args, **_kwargs: FakeModel())
    monkeypatch.setattr(tablebase, "open_strict_match_tablebase",
                        lambda *_args, **_kwargs: FakeTablebase())
    monkeypatch.setattr(tablebase, "SyzygyProbe", FakeProbe)

    def fake_play(
        _candidate: object, _reference: object, _openings: object,
        *, pgn_sink: object, pair_ids: list[int], **_kwargs: object,
    ) -> list[float]:
        pair_id = pair_ids[0]
        for half, result in ((0, "1-0"), (1, "0-1")):
            assert callable(pgn_sink)
            pgn_sink(
                pair_id=pair_id, half=half, a_is_white=half == 0,
                start_fen=chess.Board().fen(), moves=(), result=result,
                termination="checkmate", plies=0, duration_s=0.1,
            )
        return [2.0]

    monkeypatch.setattr(arena, "play_paired_games_matched_sims_rolling", fake_play)
    pgn_path = tmp_path / "arena.pgn"
    log_path = tmp_path / "arena.games.jsonl"
    receipts_path = tmp_path / "pairs"
    side = arena.SideSearch(
        shape="training", source="fixture", gumbel={},
        vloss_weight=1, target_batch=1,
    )
    def run(*, games: int = 1152, resume: bool = False,
            seal: str = "c" * 64) -> dict:
        return arena.run_arena(
            candidate="candidate.pt", reference="reference.pt",
            games=games, openings_path=None,
            openings_fen=tmp_path / "openings.fen",
            opening_plies=16, mode="matched_sims",
            sims_candidate=1, sims_reference=1, ms_per_move=0,
            max_plies=1, temperature=0.1, gumbel_add_noise=False,
            device="cpu", seed=121, out_path=None,
            pgn_out=pgn_path, pgn_candidate_name="A",
            pgn_reference_name="B", game_log_path=log_path,
            pair_receipts_dir=receipts_path, max_seconds=3200,
            syzygy_path=str(tmp_path / "tb"), tb_max_pieces=6,
            compile_models=False, eval_max_batch=0,
            max_concurrent_games=2,
            search_candidate=side, search_reference=side,
            durable_source_seal_sha256=seal, resume=resume,
        )
    with pytest.raises(SystemExit, match="durable paired arena requires"):
        run(games=4)
    assert not log_path.exists()
    first = run()
    assert first["pairs"] == 1
    assert first["complete_pair_receipts"] == 1
    receipt0 = receipts_path / "pair_000000.json"
    first_bytes = receipt0.read_bytes()
    with pytest.raises(SystemExit, match="--resume"):
        run(resume=True, seal="d" * 64)
    assert receipt0.read_bytes() == first_bytes
    second = run(resume=True)
    assert second["pairs"] == 2
    assert second["resumed_pairs"] == 1
    assert second["complete_pair_receipts"] == 2
    assert sorted(p.name for p in receipts_path.glob("pair_*.json")) == [
        "pair_000000.json", "pair_000001.json",
    ]


def test_source_seal_refuses_model_mutation_and_argv_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "source"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "tracked.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "tracked.py"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com",
         "commit", "-qm", "freeze"], cwd=repo, check=True,
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                   text=True).strip()
    monkeypatch.setattr(durable, "__file__", str(repo / "pkg" / "eval" / "arena_durable.py"))

    class _Spec:
        def __init__(self, name: str):
            self.origin = str(tmp_path / (name.rsplit(".", 1)[-1] + ".so"))

    monkeypatch.setattr(durable.importlib.util, "find_spec", _Spec)
    files: dict[str, dict[str, object]] = {}
    for role in ("candidate", "reference", "openings", "production_config",
                 "native_encoding", "native_features", "native_mcts"):
        path = tmp_path / ({"native_encoding": "_lc0_ext.so",
                            "native_features": "_features_ext.so",
                            "native_mcts": "_mcts_tree.so"}.get(role, role))
        path.write_bytes(role.encode())
        files[role] = {"path": str(path), "bytes": path.stat().st_size,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    tb_root = tmp_path / "tb"
    tb_root.mkdir()
    tb = tb_root / "KQvK.rtbw"
    tb.write_bytes(b"tablebase fixture bytes")
    (tb_root / "KQvK.rtbz").write_bytes(b"DTZ fixture bytes")
    catalog_path = tmp_path / "catalog.json"
    catalog_sha = durable.prepare_tablebase_inventory(catalog_path, str(tb_root))
    assert catalog_sha == hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    catalog = json.loads(catalog_path.read_text())
    assert all(set(entry) == {"path", "bytes", "mtime_ns"}
               for entry in catalog["files"])
    argv = ["--candidate", str(tmp_path / "candidate"), "--games", "1152"]
    seal = {
        "schema": durable.SEAL_SCHEMA, "argv": argv,
        "runtime": durable.runtime_record(),
        "profile": {"pairs": 576, "games": 1152, "sprt": False,
                    "syzygy_max_pieces": 6, "rule50_aware": True},
        "files": files, "source": {"root": str(repo), "git_head": head},
        "tablebase": {"path": str(catalog_path),
                      "sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest()},
    }
    seal_path = tmp_path / "seal.json"
    seal_path.write_text(json.dumps(seal))
    seal_sha = hashlib.sha256(seal_path.read_bytes()).hexdigest()
    def validate(path: Path, digest: str, frozen_argv: list[str]) -> str:
        return durable.validate_source_seal(
            path, digest, argv=frozen_argv,
            candidate=str(tmp_path / "candidate"),
            reference=str(tmp_path / "reference"),
            openings=tmp_path / "openings",
            config=tmp_path / "production_config", syzygy_path=str(tb_root),
        )

    assert validate(seal_path, seal_sha, argv) == seal_sha
    generated = tmp_path / "generated.json"
    generated_sha = durable.prepare_source_seal(
        generated, argv=argv, candidate=tmp_path / "candidate",
        reference=tmp_path / "reference", openings=tmp_path / "openings",
        config=tmp_path / "production_config", catalog_path=catalog_path,
        syzygy_path=str(tb_root),
    )
    assert validate(generated, generated_sha, argv) == generated_sha
    with pytest.raises(ValueError, match="argv"):
        validate(seal_path, seal_sha, [*argv, "--compile", "off"])
    (tmp_path / "candidate").write_bytes(b"Candidate")  # same path and byte count
    with pytest.raises(ValueError, match="sealed file bytes changed"):
        validate(seal_path, seal_sha, argv)
    (tmp_path / "candidate").write_bytes(b"candidate")
    original_tb = tb.read_bytes()
    tb_stat = tb.stat()
    tb.write_bytes(b"different tablebase fixture")
    with pytest.raises(ValueError, match="tablebase identity changed"):
        validate(seal_path, seal_sha, argv)
    tb.write_bytes(b"X" + original_tb[1:])
    os.utime(tb, ns=(tb_stat.st_atime_ns, tb_stat.st_mtime_ns))
    assert validate(seal_path, seal_sha, argv) == seal_sha, (
        "same-size, restored-mtime tablebase mutation is intentionally outside "
        "this metadata-only contract"
    )
