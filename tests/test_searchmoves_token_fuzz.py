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

# ⚑⚑ THE ORACLE NEEDS *BOTH* FENS, AND THE SECOND ONE IS LOAD-BEARING.
# Against a parseable FEN, `_validated_searchmoves` runs syntax THEN legality,
# and `chess.Move.from_uci` raises `InvalidMoveError` (a ValueError) on a
# malformed token — so a rejection proves nothing about WHICH gate fired. The
# first revision of this file tested only START and was therefore vacuous for
# the very `$`→`\Z` regression it was written for: reverting the anchor left
# the separator property 15/15 green.
#
# Against a FEN python-chess cannot parse, legality is SKIPPED by design, so a
# rejection can only have come from the SYNTAX gate. That isolates the regex and
# is what makes these properties discriminating.
BAD_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - x 1"
FENS = [START, BAD_FEN]

# All legal in the start position. (No promotion is possible there — an earlier
# comment claimed otherwise; promotions are exercised against their own FEN.)
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


def _accepted(token: str, fen: str = START) -> bool:
    try:
        _validated_searchmoves(fen, [token])
    except ValueError:
        return False
    return True


@pytest.mark.parametrize("fen", FENS)
@pytest.mark.parametrize("move", VALID_IN_START)
def test_the_valid_corpus_is_accepted(move: str, fen: str) -> None:
    """POSITIVE CONTROL: a reject-everything regex must fail the battery here."""
    assert _accepted(move, fen), move


@pytest.mark.parametrize("fen", FENS)
@pytest.mark.parametrize("sep", SEPARATORS)
@pytest.mark.parametrize(
    "shape",
    ["{m}{s}", "{s}{m}", "{m}{s}{m}", "{h}{s}{t}", "{m}{s}{s}"],
    ids=["trailing", "leading", "doubled-move", "embedded", "doubled-sep"],
)
def test_no_separator_survives_in_any_position(
    shape: str, sep: str, fen: str,
) -> None:
    """⚑ `{m}{s}` under BAD_FEN is the case `$` missed. One variant per test id,
    so a failure names the exact shape and separator rather than hiding in a loop.
    """
    m = "e2e4"
    variant = shape.format(m=m, s=sep, h=m[:2], t=m[2:])
    assert not _accepted(variant, fen), repr(variant)


@pytest.mark.parametrize("fen", FENS)
@pytest.mark.parametrize("keyword", KEYWORDS)
@pytest.mark.parametrize("shape", ["e2e4 {k}", "e2e4\n{k}", "{k} e2e4", "{k}"],
                         ids=["after", "newline", "before", "alone"])
def test_no_uci_keyword_can_ride_along(shape: str, keyword: str, fen: str) -> None:
    assert not _accepted(shape.format(k=keyword), fen)


@pytest.mark.parametrize("fen", FENS)
@pytest.mark.parametrize("bad", ["k", "K", "p", "x", "Q", "1", "qq", "qn"])
def test_only_the_four_promotion_pieces_are_permitted(bad: str, fen: str) -> None:
    """⚑ BAD_FEN is what makes this bite: widening `[qrbn]?` to `[a-z]?` left the
    START-only version fully green, because legality rejected the token instead.
    """
    assert not _accepted(f"e7e8{bad}", fen)


def test_every_real_promotion_suffix_is_accepted() -> None:
    """A promotion position, so legality cannot mask a syntax pass/fail."""
    # ⚑ kings must not be adjacent, or python-chess calls the POSITION
    # invalid and rejects every move — which would have made this test
    # pass for the wrong reason had it been asserting rejection.
    fen = "8/4P3/8/8/8/8/8/4K2k w - - 0 1"
    for suffix in "qrbn":
        assert _validated_searchmoves(fen, [f"e7e8{suffix}"]) == [f"e7e8{suffix}"]


@pytest.mark.parametrize("fen", FENS)
@pytest.mark.parametrize("bad", ["", " ", "e2", "e2e", "e2e4e5", "z2z4", "e0e4", "e9e4"])
def test_structurally_malformed_tokens_are_rejected(bad: str, fen: str) -> None:
    assert not _accepted(bad, fen)


@pytest.mark.parametrize("fen", FENS)
@pytest.mark.parametrize("variant", ["E2E4", "E2e4", "e2E4", "E7E8Q"])
def test_case_is_not_normalised_away(variant: str, fen: str) -> None:
    """UCI is lowercase on the wire; accepting uppercase would be a silent fix.

    ⚑ Also needs BAD_FEN: with `re.IGNORECASE` added, the START-only version
    stayed green because python-chess rejected the uppercase token downstream.
    """
    assert not _accepted(variant, fen)


# ⚑ DELETED: `test_rejection_happens_before_any_engine_byte`.
# It grepped `inspect.getsource(_validated_searchmoves)` for `_send`,
# `_protocol_section` and `self.` — and was THEATRE. An independent reviewer
# performed the exact refactor its docstring named (moving validation inside
# `_protocol_section`) and the battery stayed 43/43 green: the ordering lives in
# `search`, not in the validator's own body, so reading that body cannot see it.
# `"self."` is unfalsifiable there by construction — it can never appear in a
# module-level function. The property IS already covered, properly, by
# `test_stockfish_searchmoves.py::test_malformed_move_raises_naming_the_token`,
# which asserts on `engine.commands == []` and `desynced is False`. A guard that
# cannot fail is worse than no guard: it reads as assurance.


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
