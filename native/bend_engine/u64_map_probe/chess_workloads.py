"""Legal-position key fixtures from DeepFin CBoard, not recorded search traffic."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import TYPE_CHECKING, Any

from .benchmark import Workload
from .run_probe import Case, Op

if TYPE_CHECKING:
    import chess

ROOT = Path(__file__).resolve().parents[3]
SEEDS = {
    'opening': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    'castling': 'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
    'ep': '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1',
    'promotion': '7k/P7/8/8/8/8/1p6/7K w - - 0 1',
}
WALKS = 4
PLIES = 64
ENTRIES = 64
BITS = 7
MASK = (1 << 32) - 1


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def structural(board: chess.Board) -> tuple[int, ...]:
    """Match the native graph's conservative pseudo-legal EP identity, not NN input."""
    rights = sum(int(board.has_kingside_castling_rights(color)) << bit |
                 int(board.has_queenside_castling_rights(color)) << (bit + 1)
                 for color, bit in ((True, 0), (False, 2)))
    ep = board.ep_square if board.has_pseudo_legal_en_passant() else None
    return (board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings,
            board.occupied_co[True], board.occupied_co[False], int(board.turn), rights,
            -1 if ep is None else ep)


def bind(table: dict[int, tuple[int, ...]], key: int, identity: tuple[int, ...]) -> None:
    if type(key) is not int or not 0 <= key < 1 << 64:
        raise ValueError('invalid full-width position key')
    if key in table and table[key] != identity:
        raise ValueError('position hash collision: numeric-only workload cannot merge identities')
    table[key] = identity


def from_keys(name: str, keys: list[int]) -> tuple[list[Workload], Case]:
    """Keep real encounter order; the hit/miss/churn scheduling is still synthetic."""
    unique = list(dict.fromkeys(keys))
    if len(unique) < ENTRIES * 2 or len(keys) > WALKS * (PLIES + 1):
        raise ValueError('chess corpus too small or exceeds bounded replay')
    if any(type(k) is not int or not 0 <= k < 1 << 64 for k in keys):
        raise ValueError('invalid full-width position key')
    initial = tuple((key, i) for i, key in enumerate(unique[:ENTRIES]))
    results = []
    for kind in ('hits', 'misses', 'churn'):
        ops: list[tuple[int, int, int]] = []
        if kind in ('hits', 'misses'):
            domain = unique[:ENTRIES] if kind == 'hits' else unique[ENTRIES:2 * ENTRIES]
            ops = [(0, domain[i % ENTRIES], 0) for i in range(256)]
        else:
            for i in range(32):
                key, value = initial[i]
                ops.extend(((1, unique[ENTRIES + i], 0), (1, key, value ^ MASK),
                            (0, key, 0), (1, key, value), (2, key, 0), (0, key, 0),
                            (1, key, value), (0, key, 0)))
        results.append(Workload('chess-' + name + '-' + kind, BITS, initial, tuple(ops)))
    # One value per unique key, while revisits exercise get/update/get in order.
    ids = {key: i for i, key in enumerate(unique)}
    replay = [op for key in keys for op in (Op('g', key), Op('p', key, ids[key]), Op('g', key))]
    # 1,024 buckets / 512-entry limit exceeds this bounded corpus without eviction.
    return results, Case('chess-' + name + '-encounters', 10, replay)


def identity_checks() -> list[dict[str, Any]]:
    import chess
    from chess_anti_engine.encoding._lc0_ext import CBoard

    origin = chess.Board()
    cycle = origin.copy()
    for uci in ('g1f3', 'g8f6', 'f3g1', 'f6g8'):
        cycle.push_uci(uci)
    advanced = origin.copy()
    advanced.halfmove_clock = 99
    ep = chess.Board(SEEDS['ep'])
    no_ep = ep.copy()
    no_ep.ep_square = None
    unreachable_ep = chess.Board()
    unreachable_ep.push_uci('e2e4')
    no_capturer = unreachable_ep.copy()
    no_capturer.ep_square = None
    pinned = chess.Board('4r1k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1')
    pinned_without = pinned.copy()
    pinned_without.ep_square = None
    castle = chess.Board(SEEDS['castling'])
    no_rights = castle.copy()
    no_rights.castling_rights = 0
    rows = []
    for name, left, right, same_key in (
        ('knight-cycle-history', origin, cycle, True),
        ('halfmove-clock', origin, advanced, True),
        ('capturable-ep', ep, no_ep, False),
        ('noncapturable-ep', unreachable_ep, no_capturer, True),
        ('pinned-ep-conservative-split', pinned, pinned_without, False),
        ('castling-rights', castle, no_rights, False),
    ):
        if not left.is_valid() or not right.is_valid():
            raise ValueError('invalid identity fixture')
        a, b = CBoard.from_board(left), CBoard.from_board(right)
        if (a.transposition_key == b.transposition_key) != same_key:
            raise ValueError('native structural-key contract changed: ' + name)
        if (structural(left) == structural(right)) != same_key:
            raise ValueError('structural identity mismatch: ' + name)
        rows.append({'name': name, 'same_key': same_key,
                     'left_key': a.transposition_key, 'right_key': b.transposition_key,
                     'same_raw_hash': a.zobrist_hash == b.zobrist_hash,
                     'left_fen': left.fen(en_passant='fen'), 'right_fen': right.fen(en_passant='fen'),
                     'left_history_plies': len(left.move_stack), 'right_history_plies': len(right.move_stack)})
    if not ep.has_legal_en_passant() or not pinned.has_pseudo_legal_en_passant() or pinned.has_legal_en_passant():
        raise ValueError('en-passant fixture no longer exercises intended legality')
    if rows[0]['left_history_plies'] == rows[0]['right_history_plies']:
        raise ValueError('history fixture did not retain the reversible cycle')
    return rows


def collect() -> tuple[list[Workload], list[Case], dict[str, Any]]:
    import chess
    from chess_anti_engine.encoding import _lc0_ext
    CBoard = _lc0_ext.CBoard
    extension = Path(_lc0_ext.__file__).resolve()
    if ROOT not in extension.parents:
        raise ValueError("CBoard extension is outside the inspected checkout")

    workloads: list[Workload] = []
    replays: list[Case] = []
    corpora = []
    bindings: dict[int, tuple[int, ...]] = {}
    for label, fen in SEEDS.items():
        rows = []
        for walk in range(WALKS):
            board = chess.Board(fen)
            if not board.is_valid():
                raise ValueError('invalid corpus seed: ' + label)
            rng = random.Random(20260924 + walk)
            moves: list[str] = []
            for ply in range(PLIES + 1):
                key = int(CBoard.from_board(board).transposition_key)
                bind(bindings, key, structural(board))
                rows.append({'walk': walk, 'ply': ply, 'moves': tuple(moves),
                             'fen': board.fen(en_passant='fen'), 'key': key})
                legal = sorted(board.legal_moves, key=lambda m: m.uci())
                if ply == PLIES or board.is_game_over() or not legal:
                    break
                move = legal[rng.randrange(len(legal))]
                board.push(move)
                moves.append(move.uci())
        keys = [row['key'] for row in rows]
        cases, replay = from_keys(label, keys)
        workloads.extend(cases)
        replays.append(replay)
        counts = Counter(keys)
        corpora.append({'name': label, 'root_fen': fen, 'walks': WALKS, 'max_plies': PLIES,
                        'observations': len(keys), 'distinct_keys': len(counts),
                        'revisits': len(keys) - len(counts), 'high_bit_keys': sum(k >> 63 for k in counts),
                        'records_sha256': digest(rows), 'records': rows})
    sources = ('chess_anti_engine/encoding/_cboard_impl.h', 'chess_anti_engine/encoding/_lc0_ext.c',
               'chess_anti_engine/mcts/_position_dag.h',
               'native/bend_engine/u64_map_probe/chess_workloads.py')
    return workloads, replays, {
        'scope': 'CBoard keys from deterministic legal walks; not selfplay/search traffic or NN cache identities',
        'python_chess_version': chess.__version__,
        'cboard_extension_sha256': hashlib.sha256(extension.read_bytes()).hexdigest(),
        'producer_source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        'identity_checks': identity_checks(), 'corpora': corpora,
    }
