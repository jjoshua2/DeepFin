"""Opt-in native optional-claim tests. No model, perft, GPU or strength claim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import chess

from . import run_probe as sessions
from .claims import claim_option
from .draw_probe import cycle_root
from .root_protocol import board_position, packed_move
from .search_draws import automatic_draw, draw_reply, reconstruct_leaf

ROOT = Path(__file__).resolve().parents[3]


def drive(binary: Path, oracle: sessions.Oracle, root: chess.Board, *, budget: int = 16,
          depth: int = 3, cap: int = 4096, prefer: tuple[str, ...] = (),
          claims: bool = True, fault: str = '', fault_at: int = 1,
          model_value: float = -0.75) -> tuple[dict[str, object], sessions.Reference]:
    ref = sessions.Reference(board_position(root), oracle, cap=cap, depth=depth, budget=budget)
    peer = sessions.Peer(binary, board_position(root))
    offered: list[dict[str, object]] = []
    paths = []
    try:
        peer.write(f'config 1 {budget:x} {cap:x} {depth:x}\n')
        while True:
            wanted = ref.next()
            line = peer.line()
            if wanted is None:
                result = sessions.numbers(line, 'result', 6)
                rows = [sessions.numbers(peer.line(), 'node', 30) for _ in ref.nodes]
                best = sessions.numbers(peer.line(), 'best', 1)[0]
                ref.check_snapshot(rows, result, best, 1)
                peer.expect('ready')
                peer.finish()
                return {'completed': ref.completed, 'nodes': len(ref.nodes), 'best': best,
                        'stop': ref.stop, 'offers': offered, 'requests': len(paths)}, ref
            header = sessions.numbers(line, 'eval', 4)
            if header[:3] != [1, ref.seq, wanted] or not 1 <= header[3] <= 256:
                raise AssertionError('claim request identity/count mismatch')
            supplied = sessions.position(sessions.numbers(peer.line(), 'board', 19))
            path = sessions.parse_path(peer.line())
            ancestors, i = [], wanted
            while i:
                ancestors.append(ref.nodes[i].key)
                i = ref.nodes[i].parent
            if path != ancestors[::-1] or supplied != ref.nodes[wanted].board:
                raise AssertionError('claim request has wrong board or path')
            actions = [sessions.numbers(peer.line(), 'action', 1)[0] for _ in range(header[3])]
            peer.expect('end_eval')
            board = reconstruct_leaf(root, path, supplied, actions)
            paths.append(path)
            if automatic_draw(board) is not None:
                peer.write(draw_reply(1, ref.seq, wanted))
                ref.accept_draw(wanted)
                continue
            option = claim_option(board) if claims else None
            priors = [sessions.f32(1 / len(actions))] * len(actions)
            if len(path) < len(prefer):
                move = chess.Move.from_uci(prefer[len(path)])
                if move not in board.legal_moves:
                    raise AssertionError('illegal preferred continuation')
                key = packed_move(board, move)
                priors = [float(k == key) for k in actions]
            # Root and ordinary leaves can be made pessimistic. Immediate mates
            # are decided by Bend, independent of these deliberately poor values.
            wdl = [max(0.0, model_value), 1 - abs(model_value), max(0.0, -model_value)]
            status = 4 if option else 0
            fields = [1, ref.seq, wanted, status, *(sessions.bits(v) for v in wdl),
                      len(priors), *(sessions.bits(p) for p in priors)]
            if fault and ref.seq == fault_at:
                if option is None:
                    raise AssertionError('fault must target an actual claim option')
                if fault == 'epoch':
                    fields[0] += 1
                elif fault == 'request':
                    fields[1] += 1
                elif fault == 'node':
                    fields[2] += 1
                elif fault == 'count':
                    fields[7] -= 1
                    fields.pop()
                elif fault == 'nan':
                    fields[4] = sessions.bits(float('nan'))
                elif fault == 'infinity':
                    fields[5] = sessions.bits(float('inf'))
                elif fault == 'negative':
                    fields[8] = sessions.bits(-1.0)
                elif fault == 'zero_policy':
                    fields[8:] = [0] * len(priors)
                elif fault == 'wdl':
                    fields[4:7] = [sessions.bits(1.0)] * 3
                else:
                    raise ValueError('unknown claim fault')
                ref.stop = 4
            else:
                ref.accept(wanted, actions, wdl, priors, claim=option is not None)
                if option and not ref.stop:
                    offered.append({'node': wanted, 'path': path, 'reason': option.reason,
                                    'intended_move': option.intended_move})
            peer.write('reply ' + ' '.join(f'{v:x}' for v in fields) + '\n')
    finally:
        peer.close()


def verify(binaries: dict[str, Path]) -> dict[str, object]:
    oracle = sessions.Oracle(binaries['reference'], with_python_chess=True)
    all_modes = []
    for mode, binary in binaries.items():
        if mode == 'reference':
            continue
        cases: dict[str, object] = {}
        current = cycle_root(2)
        prospective = cycle_root(1, ('g1f3', 'g8f6', 'f3g1'))
        fifty = chess.Board('4k3/8/8/8/8/8/8/R3K3 w - - 100 1')
        near = chess.Board('4k3/8/8/8/8/8/8/R3K3 w - - 99 1')
        for label, root in [('current-threefold', current), ('prospective-threefold', prospective),
                            ('current-fifty', fifty), ('prospective-fifty', near)]:
            before = root.fen(), list(root.move_stack)
            # One completed root evaluation suffices to expose the claim action,
            # but gives no sampled reason to prefer an unknown move over a draw.
            row, ref = drive(binary, oracle, root, budget=1)
            assert row['best'] == sessions.CLAIM_KEY
            real = {a.key for a in ref.nodes[1:] if a.key != sessions.CLAIM_KEY}
            assert real == {packed_move(root, m) for m in root.legal_moves}
            claim = ref.nodes[-1]
            assert (claim.key, claim.status, claim.value, claim.prior, claim.board) == (
                sessions.CLAIM_KEY, 2, 0.0, 0.0, board_position(root))
            assert (root.fen(), root.move_stack) == before
            assert ref.nodes[0].count == len(real) + 1
            cases[label] = row
        # With all legal continuations pessimistic at a depth-one frontier, the
        # known draw stays available without being the only retained action.
        row, ref = drive(binary, oracle, current, depth=1, model_value=0.0)
        assert row['best'] == sessions.CLAIM_KEY
        assert len(ref.nodes) == len(list(current.legal_moves)) + 2
        cases['tie-prefers-claim'] = row
        row, ref = drive(binary, oracle, current, depth=1, budget=32, model_value=0.75)
        assert row['best'] == sessions.CLAIM_KEY
        assert any(n.key == sessions.CLAIM_KEY and n.n for n in ref.nodes)
        cases['losing-continuations-choose-claim'] = row
        mate = chess.Board('k7/2Q5/2K5/8/8/8/8/8 w - - 100 1')
        row, ref = drive(binary, oracle, mate, budget=8, prefer=('c7b7',))
        mate_key = packed_move(mate, chess.Move.from_uci('c7b7'))
        assert row['best'] == mate_key
        winner = next(a for a in ref.nodes[1:] if a.key == mate_key)
        assert winner.status == 2
        assert winner.value == -1.0
        assert any(a.key == sessions.CLAIM_KEY for a in ref.nodes)
        cases['mate-beats-available-claim'] = row
        before_claim = chess.Board('4k3/8/8/8/8/8/8/R3K3 w - - 98 1')
        row, ref = drive(binary, oracle, before_claim, budget=3, prefer=('a1b1',), depth=3)
        a = next(n for n in ref.nodes[1:] if n.n)
        assert any(n.parent == ref.nodes.index(a) and n.key == sessions.CLAIM_KEY for n in ref.nodes)
        cases['below-root-option'] = row
        row, ref = drive(binary, oracle, before_claim, budget=3, prefer=('a1b1',), depth=1)
        cut = next(n for n in ref.nodes[1:] if n.n)
        assert (cut.status, cut.value, cut.count) == (3, 0.0, 0)
        cases['cutoff-keeps-zero-option'] = row
        row, ref = drive(binary, oracle, before_claim, budget=2, prefer=('a1b1',),
                         depth=1, model_value=0.75)
        assert next(n for n in ref.nodes[1:] if n.n).value == 0.75
        cases['cutoff-keeps-winning-estimate'] = row
        for label, root, enabled in [('historyless', chess.Board(current.fen()), True),
                                     ('option-disabled', current, False)]:
            row, ref = drive(binary, oracle, root, budget=1, claims=enabled)
            assert all(a.key != sessions.CLAIM_KEY for a in ref.nodes)
            assert row['best'] != sessions.CLAIM_KEY
            cases[label] = row
        # Extra node capacity is checked atomically after the claim is announced.
        for cap in (21, 22):
            row, ref = drive(binary, oracle, current, budget=1, cap=cap)
            if cap == 21:
                assert (ref.stop, ref.completed, len(ref.nodes), ref.seq) == (1, 0, 1, 1)
            else:
                assert (ref.stop, ref.completed, len(ref.nodes)) == (0, 1, 22)
            cases[f'capacity-{cap}'] = row
        for fault in ('epoch', 'request', 'node', 'count', 'nan', 'infinity', 'negative', 'zero_policy', 'wdl'):
            row, ref = drive(binary, oracle, current, fault=fault)
            assert (ref.stop, ref.completed, len(ref.nodes), ref.seq) == (4, 0, 1, 1)
            cases['invalid-' + fault] = row
        row, ref = drive(binary, oracle, before_claim, fault='count', fault_at=2, prefer=('a1b1',))
        assert (ref.stop, ref.completed) == (4, 1)
        assert not any(a.key == sessions.CLAIM_KEY for a in ref.nodes)
        cases['invalid-after-completed-simulation'] = row
        # Terminal auto outcomes take precedence and no optional child is added.
        row, ref = drive(binary, oracle, cycle_root(4))
        assert len(ref.nodes) == 1
        assert row['offers'] == []
        cases['automatic-before-optional'] = row
        all_modes.append({'mode': mode, 'case_count': len(cases), 'cases': cases})
    return {'scope': 'optional-claim actions, controlled evaluator; not trained strength or throughput',
            'results': all_modes, 'oracle_positions': len(oracle.cache)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-root', type=Path, default=ROOT / 'build/bend_u64_toolchain/source')
    parser.add_argument('--bun', default=shutil.which('bun'))
    parser.add_argument('--cc', default=shutil.which('clang'))
    parser.add_argument('--modes', nargs='+', choices=list(sessions.MODES), default=list(sessions.MODES))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if not args.bun or not args.cc:
        parser.error('Bun and Clang required')
    with tempfile.TemporaryDirectory(prefix='bend-claims-') as directory:
        binaries = sessions.build(args.compiler_root, Path(directory), args.bun, args.cc, args.modes)
        report = verify(binaries)
        report['original_sessions'] = sessions.verify(binaries, with_python_chess=True)
        from .draw_probe import verify as verify_draws
        report['automatic_draws'] = verify_draws(binaries)
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
