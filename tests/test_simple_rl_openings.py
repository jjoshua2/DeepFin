"""simple_rl opening split: disjoint, deterministic, and mirror/transposition-safe."""
from __future__ import annotations

import random

import chess

from chess_anti_engine.simple_rl.openings import BookLine, position_keys, split_lines

START = chess.STARTING_FEN

# A pool of distinct legal 4-ply openings.
_FIRST = ["e2e4", "d2d4", "c2c4", "g1f3", "b1c3", "e2e3", "d2d3", "g2g3", "b2b3", "f2f4"]
_REPLY = ["e7e5", "d7d5", "c7c5", "g8f6", "b8c6", "e7e6", "d7d6", "g7g6"]


def _pool() -> list[BookLine]:
    lines = []
    for a in _FIRST:
        for b in _REPLY:
            board = chess.Board()
            board.push_uci(a)
            board.push_uci(b)
            lines.extend(BookLine(book=1, root_fen=START, moves=(a, b, third.uci()))
                         for third in list(board.legal_moves)[:3])
    return lines


def test_split_is_disjoint_and_complete() -> None:
    lines = _pool()
    train, eval_set, counts = split_lines(lines, eval_lines=40, salt="t")
    train_text = {line.line for line in train}
    eval_text = {line.line for line in eval_set}
    assert len(eval_set) == 40
    assert counts["eval"] == 40
    assert not train_text & eval_text
    # No train FINAL POSITION may be an eval final position or a mirror of one.
    banned = set().union(*(position_keys(line.final_board()) for line in eval_set))
    assert not {line.final_board().epd() for line in train} & banned
    assert counts["train"] + counts["eval"] + counts["train_excluded_by_eval_position"] == counts["usable"]


def test_split_is_deterministic_and_order_independent() -> None:
    lines = _pool()
    first = split_lines(lines, eval_lines=25, salt="seed-a")
    shuffled = list(lines)
    random.Random(7).shuffle(shuffled)
    second = split_lines(shuffled, eval_lines=25, salt="seed-a")
    assert [x.line for x in first[1]] == [x.line for x in second[1]]
    assert [x.line for x in first[0]] == [x.line for x in second[0]]
    other_salt = split_lines(lines, eval_lines=25, salt="seed-b")
    assert [x.line for x in first[1]] != [x.line for x in other_salt[1]]


def _line_to(board: chess.Board) -> BookLine:
    return BookLine(book=1, root_fen=board.root().fen(), moves=tuple(m.uci() for m in board.move_stack))


def test_transpositions_and_both_mirrors_of_an_eval_line_are_excluded() -> None:
    # One eval line, then a train candidate for each forbidden relation.
    eval_line = BookLine(book=1, root_fen=START, moves=("g1f3", "g8f6"))
    transposition = BookLine(book=1, root_fen=START, moves=("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"))
    # Colour mirror of 1.Nf3 Nf6 is itself; use an asymmetric line instead.
    asym_eval = BookLine(book=1, root_fen=START, moves=("e2e4", "c7c5", "g1f3"))
    colour_mirror = chess.Board(asym_eval.final_board().mirror().fen())
    flip = asym_eval.final_board().transform(chess.flip_horizontal)
    candidates = [
        transposition,
        BookLine(book=1, root_fen=colour_mirror.fen(), moves=()),
        BookLine(book=1, root_fen=flip.fen(), moves=()),
    ]
    # Lines with no moves are legal FEN-only starts; give them a harmless
    # null-effect form by appending nothing (the grammar accepts it).
    filler = _pool()[:60]
    lines = [eval_line, asym_eval, *candidates, *filler]

    # Force the two eval lines by making them rank first under the salt.
    import hashlib

    def score(line: BookLine, salt: str) -> str:
        return hashlib.sha256(f"{salt}\n{line.line}".encode()).hexdigest()

    salt = next(s for s in (f"s{i}" for i in range(20000))
                if sorted(lines, key=lambda x: (score(x, s), x.line))[:2] in ([eval_line, asym_eval], [asym_eval, eval_line]))
    train, eval_set, counts = split_lines(lines, eval_lines=2, salt=salt)
    assert {x.line for x in eval_set} == {eval_line.line, asym_eval.line}
    train_text = {x.line for x in train}
    for forbidden in candidates:
        assert forbidden.line not in train_text, forbidden.line
    assert counts["train_excluded_by_eval_position"] == 3


def _force_eval(lines: list[BookLine], wanted: list[BookLine]) -> str:
    import hashlib

    def score(line: BookLine, salt: str) -> str:
        return hashlib.sha256(f"{salt}\n{line.line}".encode()).hexdigest()

    want = {w.line for w in wanted}
    return next(s for s in (f"s{i}" for i in range(200000))
                if {x.line for x in sorted(lines, key=lambda x: (score(x, s), x.line))[:len(wanted)]} == want)


def _leaks(train: list[BookLine], eval_set: list[BookLine]) -> dict[str, int]:
    """The reviewer's leak check (scratchpad pr1416/leak_check.py), orbit-closed."""
    from chess_anti_engine.simple_rl.openings import path_boards

    ev_final: set[str] = set()
    ev_path: set[str] = set()
    for line in eval_set:
        boards = list(path_boards(line))
        ev_final |= position_keys(boards[-1])
        for b in boards:
            ev_path |= position_keys(b)
    through = sum(1 for line in train if any(b.epd() in ev_final for b in list(path_boards(line))[:-1]))
    tr_final: set[str] = set()
    for line in train:
        tr_final |= position_keys(line.final_board())
    eval_through_train = sum(1 for line in eval_set
                             if any(b.epd() in tr_final for b in list(path_boards(line))[:-1]))
    on_path = sum(1 for line in train if line.final_board().epd() in ev_path)
    return {"train_through_eval_start": through, "eval_through_train_start": eval_through_train,
            "train_ends_on_eval_path": on_path}


def test_no_train_line_shares_a_path_with_an_eval_start() -> None:
    """Prefix leaks: a TRAIN line passing through an EVAL start, and an EVAL
    line passing through a TRAIN start, both directions and mirrored."""
    eval_short = BookLine(book=2, root_fen=START, moves=("e2e4", "e7e5"))
    eval_long = BookLine(book=1, root_fen=START, moves=("d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"))
    through = BookLine(book=1, root_fen=START, moves=("e2e4", "e7e5", "g1f3", "b8c6"))     # passes eval_short
    prefix = BookLine(book=2, root_fen=START, moves=("d2d4", "d7d5", "c2c4", "e7e6"))      # ends on eval_long's path
    mirrored = BookLine(book=1, root_fen=START, moves=("d2d4", "d7d5", "e2e4", "e7e5"))     # unrelated control
    flip_prefix = BookLine(book=2, root_fen=START, moves=("e2e4", "e7e5", "f1c4"))          # passes eval_short
    filler = _pool()[:40]
    lines = [eval_short, eval_long, through, prefix, mirrored, flip_prefix, *filler]
    salt = _force_eval(lines, [eval_short, eval_long])
    train, eval_set, counts = split_lines(lines, eval_lines=2, salt=salt)
    assert {x.line for x in eval_set} == {eval_short.line, eval_long.line}
    text = {x.line for x in train}
    assert through.line not in text
    assert prefix.line not in text
    assert flip_prefix.line not in text
    assert mirrored.line in text
    assert _leaks(train, eval_set) == {"train_through_eval_start": 0, "eval_through_train_start": 0,
                                       "train_ends_on_eval_path": 0}
    assert counts["train_excluded_passes_through_eval_start"] >= 2
    assert counts["train_excluded_ends_on_eval_path"] >= 1


def test_eval_slice_is_position_distinct_and_the_ban_is_orbit_closed() -> None:
    # Two eval candidates with the same final position by transposition: only one is held out.
    a = BookLine(book=1, root_fen=START, moves=("g1f3", "g8f6", "b1c3"))
    b = BookLine(book=1, root_fen=START, moves=("b1c3", "g8f6", "g1f3"))
    filler = _pool()[:30]
    lines = [a, b, *filler]
    salt = _force_eval(lines, [a, b])
    train, eval_set, counts = split_lines(lines, eval_lines=2, salt=salt)
    finals = [min(position_keys(x.final_board())) for x in eval_set]
    assert len(set(finals)) == 2
    assert counts["eval_skipped_duplicate_or_mirror"] == 1
    assert not {a.line, b.line} <= {x.line for x in eval_set}
    # O1: the reflection of the colour mirror is in the orbit too.
    board = a.final_board()
    assert board.mirror().transform(chess.flip_horizontal).epd() in position_keys(board)
    assert _leaks(train, eval_set)["train_through_eval_start"] == 0


def test_pgn_parse_errors_are_rejected_and_counted(tmp_path) -> None:
    from chess_anti_engine.simple_rl.openings import read_book

    pgn = tmp_path / "book.pgn"
    pgn.write_text('[Event "ok"]\n\n1. e4 e5 2. Nf3 *\n\n[Event "bad"]\n\n1. e4 e5 2. Ke3 Nc6 *\n\n')
    stats: dict[str, int] = {}
    lines = read_book(pgn, book=1, max_plies=8, stats=stats)
    assert [x.moves for x in lines] == [("e2e4", "e7e5", "g1f3")]
    assert stats["rejected_parse_errors"] == 1
