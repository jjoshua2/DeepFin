"""Property battery for the `searchmoves` token boundary.

⚑ WHY THIS EXISTS. The hand-curated malformed-token list in
`test_stockfish_searchmoves.py` contained `"e2e4\\nstop"` and passed — while
`"e2e4\\n"` slipped through, because Python's `$` also matches immediately
before a trailing newline. The list covered the LOUD half of newline injection
and missed the quiet half, and looked complete either way. Two independent
reviewers found the same anchor character on the same day.

A curated list can only assert the cases someone thought of. This file asserts
the PROPERTY instead:

    a token is accepted IFF the whole string is exactly one permitted token

built by mutating VALID moves with every separator/keyword/control that could
change how the `go` line parses. No new dependency — the corpus is generated
deterministically, so failures are reproducible by name.

The second property matters just as much: **rejection must happen before any
byte reaches the engine.** A raise from inside `_protocol_section` marks the
process desynced and costs a Stockfish restart, which is the very thing this
validation exists to prevent.
"""

from __future__ import annotations

import chess
import pytest

from chess_anti_engine.stockfish.uci import _validated_searchmoves

START = chess.STARTING_FEN

# Legal in the start position, and one promotion-shaped token that is not.
VALID_IN_START = ["e2e4", "d2d4", "g1f3", "b1c3", "a2a3", "h2h4"]

# Anything that could split or extend the `go` line. `searchmoves` consumes
# every remaining token, so a separator that survives validation silently
# redefines the rest of the command.
SEPARATORS = [
    "\n", "\r", "\r\n", "\t", " ", "\v", "\f", "\x00",
    "\x1b", "\x85", " ", " ", " ", "　",
]

# UCI keywords that change the meaning of a `go` line if injected.
KEYWORDS = ["stop", "quit", "movetime", "infinite", "ponder", "depth", "nodes"]


def _accepted(token: str) -> bool:
    try:
        _validated_searchmoves(START, [token])
    except ValueError:
        return False
    return True


def test_the_valid_corpus_is_accepted() -> None:
    """Guard on the guard: if these were rejected the battery would be vacuous."""
    for move in VALID_IN_START:
        assert _accepted(move), move


@pytest.mark.parametrize("sep", SEPARATORS)
@pytest.mark.parametrize("move", ["e2e4"])
def test_no_separator_survives_in_any_position(move: str, sep: str) -> None:
    """Trailing, leading, embedded, and doubled — all must be rejected."""
    for variant in (
        f"{move}{sep}",          # ⚑ the case `$` missed
        f"{sep}{move}",
        f"{move}{sep}{move}",
        f"{move[:2]}{sep}{move[2:]}",
        f"{move}{sep}{sep}",
    ):
        assert not _accepted(variant), repr(variant)


@pytest.mark.parametrize("keyword", KEYWORDS)
def test_no_uci_keyword_can_ride_along(keyword: str) -> None:
    for variant in (f"e2e4 {keyword}", f"e2e4\n{keyword}", f"{keyword} e2e4", keyword):
        assert not _accepted(variant), repr(variant)


@pytest.mark.parametrize("bad", ["k", "K", "p", "x", "Q", "1", "qq", "qn"])
def test_only_the_four_promotion_pieces_are_permitted(bad: str) -> None:
    assert not _accepted(f"e7e8{bad}")


def test_every_real_promotion_suffix_is_accepted() -> None:
    """A promotion position, so legality cannot mask a syntax pass/fail."""
    # ⚑ kings must not be adjacent, or python-chess calls the POSITION
    # invalid and rejects every move — which would have made this test
    # pass for the wrong reason had it been asserting rejection.
    fen = "8/4P3/8/8/8/8/8/4K2k w - - 0 1"
    for suffix in "qrbn":
        assert _validated_searchmoves(fen, [f"e7e8{suffix}"]) == [f"e7e8{suffix}"]


@pytest.mark.parametrize("bad", ["", " ", "e2", "e2e", "e2e4e5", "z2z4", "e0e4", "e9e4"])
def test_structurally_malformed_tokens_are_rejected(bad: str) -> None:
    assert not _accepted(bad)


def test_case_is_not_normalised_away() -> None:
    """UCI is lowercase on the wire; accepting uppercase would be a silent fix."""
    for variant in ("E2E4", "E2e4", "e2E4", "E7E8Q"):
        assert not _accepted(variant)


def test_rejection_happens_before_any_engine_byte() -> None:
    """The whole point: a bad token must not cost a Stockfish process.

    `_validated_searchmoves` is a pure function over (fen, tokens) — it holds no
    engine handle and cannot write. This pins that structurally, so a future
    refactor that moves validation inside the protocol section fails here.
    """
    import inspect

    src = inspect.getsource(_validated_searchmoves)
    for forbidden in ("_send", "_protocol_section", "self."):
        assert forbidden not in src, (
            f"{forbidden!r} appears in _validated_searchmoves — validation must "
            "stay outside the protocol section, or a caller's typo desyncs and "
            "costs an engine restart"
        )


def test_a_legal_move_for_a_different_position_is_still_rejected() -> None:
    """Syntax alone is not enough; the token must be legal in THIS fen."""
    assert not _accepted("e4e5")  # well-formed, not legal from the start position


def test_unparseable_fen_keeps_syntax_but_drops_legality() -> None:
    """The documented carve-out, pinned so it cannot silently widen."""
    bad_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - x 1"
    # legality is skipped ...
    assert _validated_searchmoves(bad_fen, ["e4e5"]) == ["e4e5"]
    # ... but syntax is NOT, which is what keeps the `go` line well formed.
    with pytest.raises(ValueError, match="malformed"):
        _validated_searchmoves(bad_fen, ["e2e4\n"])
