"""Chess-key producer contracts; ordinary pytest does not benchmark maps."""
from types import SimpleNamespace

import pytest

from native.bend_engine.u64_map_probe import chess_workloads as c
from native.bend_engine.u64_map_probe.benchmark import model
from native.bend_engine.u64_map_probe.run_probe import encode, expected


def test_full_key_bindings_preserve_distinct_high_halves() -> None:
    table: dict[int, tuple[int, ...]] = {}
    c.bind(table, 0, (1,))
    c.bind(table, 1 << 63, (2,))
    c.bind(table, 0, (1,))
    assert table == {0: (1,), 1 << 63: (2,)}


def test_position_hash_collision_is_not_a_transposition() -> None:
    table: dict[int, tuple[int, ...]] = {7: (1, 2)}
    with pytest.raises(ValueError, match='hash collision'):
        c.bind(table, 7, (2, 1))
    assert table == {7: (1, 2)}


@pytest.mark.parametrize('key', [-1, 1 << 64, True, 1.0])
def test_invalid_key_binding(key: int) -> None:
    with pytest.raises(ValueError, match='full-width'):
        c.bind({}, key, (1,))


def test_encounter_order_and_neutral_operation_tapes() -> None:
    keys = [(i << 32) | 7 for i in range(130)] + [7, (4 << 32) | 7]
    workloads, replay = c.from_keys('test', keys)
    assert len(workloads) == 3
    assert all(model(w)[1] == dict(w.initial) for w in workloads)
    assert all(len(w.initial) == 64 and len(w.ops) == 256 for w in workloads)
    assert [k for k, _ in workloads[0].initial] == keys[:64]
    assert replay.bits == 11
    assert len(replay.ops) == 3 * len(keys)
    assert [op.key for op in replay.ops[::3]] == keys
    assert expected(replay).endswith('g 4 7 value 4 130\nend 130\n')
    assert len(encode(replay)) < 100000


@pytest.mark.parametrize('keys', [list(range(127)), [7] * 256, list(range(521))])
def test_no_truncated_or_undersized_corpus(keys: list[int]) -> None:
    with pytest.raises(ValueError, match='corpus'):
        c.from_keys('bad', keys)


@pytest.mark.parametrize('bad', [-1, 1 << 64, True, 0.25])
def test_no_invalid_corpus_key(bad: int) -> None:
    with pytest.raises(ValueError, match='full-width'):
        c.from_keys('bad', [*range(128), bad])


def test_engine_identity_boundaries() -> None:
    checks = {r['name']: r for r in c.identity_checks()}
    assert len(checks) == 6
    assert checks['knight-cycle-history']['same_key'] is True
    assert checks['knight-cycle-history']['right_history_plies'] == 4
    assert checks['halfmove-clock']['same_key'] is True
    assert checks['capturable-ep']['same_key'] is False
    assert checks['capturable-ep']['same_raw_hash'] is True
    assert checks['noncapturable-ep']['same_key'] is True
    assert checks['pinned-ep-conservative-split']['same_key'] is False
    assert checks['castling-rights']['same_key'] is False


def test_engine_raw_position_hash_substitution_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from chess_anti_engine.encoding import _lc0_ext
    original = _lc0_ext.CBoard

    class RawHash:
        @staticmethod
        def from_board(board: object) -> SimpleNamespace:
            native = original.from_board(board)
            return SimpleNamespace(transposition_key=native.zobrist_hash, zobrist_hash=native.zobrist_hash)

    monkeypatch.setattr(_lc0_ext, 'CBoard', RawHash)
    with pytest.raises(ValueError, match='capturable-ep'):
        c.identity_checks()


def test_engine_generated_chess_workloads_are_reproducible() -> None:
    import chess
    from chess_anti_engine.encoding._lc0_ext import CBoard

    workloads, traces, provenance = c.collect()
    second = c.collect()
    assert (workloads, traces, provenance) == second
    assert len(workloads) == 12
    assert len(traces) == 4
    assert len(provenance['corpora']) == 4
    for corpus in provenance['corpora']:
        assert corpus['distinct_keys'] >= 128
        assert corpus['revisits'] >= 3  # Each independent walk starts at the same root.
        assert corpus['high_bit_keys'] > 0
        assert corpus['observations'] <= 520
        assert corpus['records_sha256'] == c.digest(corpus['records'])
        # Independently reconstruct each recorded path and check the producer's key.
        for row in corpus['records']:
            board = chess.Board(corpus['root_fen'])
            for uci in row['moves']:
                board.push_uci(uci)
            assert row['ply'] == len(row['moves'])
            assert board.fen(en_passant='fen') == row['fen']
            assert int(CBoard.from_board(board).transposition_key) == row['key']
    for w in workloads:
        assert model(w)[1] == dict(w.initial)
    for trace in traces:
        assert len(encode(trace)) < 100000
        assert expected(trace).startswith('begin 11\n')
