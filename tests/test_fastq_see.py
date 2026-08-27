"""Static exchange evaluation for FastQ, against a brute-force oracle.

⚑⚑ THE ORACLE IS A DIFFERENT ALGORITHM, NOT A SECOND COPY OF THE SAME ONE.
`_oracle_see` plays out capture sequences with python-chess's LEGAL move
generator and takes a true minimax over every capture onto the target square.
The C evaluator does a least-valuable-attacker swap over bitboards with no
legality notion at all. They agree on ordinary positions because the answer is
the same, not because the code is — which is the only reason agreement is
evidence. A "reference implementation" that mirrored the swap loop in Python
would pass while sharing every wrong assumption, which is this repo's named
parity trap (internal_equivalence_cannot_find_a_shared_wrong_rule).

⚑ THE DIVERGENCES ARE THE POINT OF THE FILE, NOT AN EMBARRASSMENT IN IT. Static
SEE has no notion of move legality (`docs/fastq_design.md` §5 states the pin
half of this), so where legality binds the two MUST differ. Those rows are
asserted as expected divergences with BOTH values pinned, so the approximation
is measured rather than described: if someone teaches SEE about legality, or
breaks it in a way that happens to change these rows, the test fails and the
change has to be deliberate.

MEASURED on the 6,422-capture sweep below: 5,208 rows are compared bitwise with
no exemption at all and every one agrees; 24 rows (0.37%) diverge, and all 24
are legality-bound. The spec names pins; the corpus also found checks — a
recapture that would leave its own side in check does not exist, including under
a DISCOVERED check from a piece unrelated to the target square. Both are the
same underlying approximation and both have a crafted fixture.
"""

from __future__ import annotations

import random

import chess
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.nnue import _nnue_ext

#: Must mirror CAE_SEE_VALUE in chess_anti_engine/nnue/_fastq_see.h.
#:
#: ⚑ Duplicated deliberately and pinned by
#: test_the_python_value_table_matches_the_c_one_it_mirrors, which reads the
#: header. An unpinned copy of a constant table is how the oracle and the
#: subject silently stop scoring the same game.
SEE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


def _see(board: chess.Board, move: chess.Move) -> int:
    """The C evaluator, through the same entry point the search uses."""
    return _nnue_ext.see(
        CBoard.from_board(board),
        move.from_square,
        move.to_square,
        move.promotion or 0,
    )


def _immediate_gain(board: chess.Board, move: chess.Move) -> int:
    """Material this move wins outright: the victim, plus any promotion upgrade."""
    gain = 0
    if board.is_en_passant(move):
        gain += SEE_VALUE[chess.PAWN]
    else:
        victim = board.piece_at(move.to_square)
        if victim is not None:
            gain += SEE_VALUE[victim.piece_type]
    if move.promotion:
        gain += SEE_VALUE[move.promotion] - SEE_VALUE[chess.PAWN]
    return gain


def _best_continuation(board: chess.Board, target: int) -> int:
    """Best net gain for the side to move, who may always decline to recapture.

    A true maximum over every LEGAL capture onto `target` — not the least
    valuable attacker. Taking the cheapest attacker is itself one of static
    SEE's approximations, so an oracle that assumed it could not detect the
    cases where it is wrong.
    """
    # ⚑ list(), NOT the bare generator. `board.legal_moves` is a lazy
    # LegalMoveGenerator reading the LIVE board, and this loop pushes and pops
    # inside itself — iterating it directly silently walks a mutated position
    # and yields moves for the wrong side. It produced four plausible-looking
    # wrong oracle values before it was caught, which is the failure mode of an
    # oracle exactly: it does not crash, it just quietly stops being a reference.
    best = 0
    for move in list(board.legal_moves):
        if move.to_square != target:
            continue
        if not (board.is_capture(move) or move.promotion):
            continue
        gain = _immediate_gain(board, move)
        board.push(move)
        try:
            best = max(best, gain - _best_continuation(board, target))
        finally:
            board.pop()
    return best


def _oracle_see(board: chess.Board, move: chess.Move) -> int:
    """Brute-force SEE: the given move is forced, everything after it is optimal."""
    gain = _immediate_gain(board, move)
    after = board.copy(stack=False)
    after.push(move)
    return gain - _best_continuation(after, move.to_square)


# ===========================================================================
# Crafted fixtures, one property each
# ===========================================================================

#: (label, fen, uci, expected) — hand-computed, NOT read off the implementation.
AGREEING_FIXTURES = [
    # A free pawn is worth a pawn.
    ("undefended pawn", "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", 100),
    # Pawn takes pawn, knight takes back: even.
    ("defended pawn", "4k3/2n5/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", 0),
    # Knight takes a defended pawn: wins 100, loses 320.
    ("knight for pawn", "4k3/2n5/8/3p4/8/4N3/8/4K3 w - - 0 1", "e3d5", -220),
    # ⚑ X-RAY: the rook behind the rook joins the swap only because the first
    # one left the occupancy. A SEE that recomputed attacks against the ORIGINAL
    # occupancy scores this as losing a rook for a pawn.
    (
        "x-ray rook battery",
        "3rk3/3r4/8/3p4/8/8/3R4/3RK3 w - - 0 1",
        "d2d5",
        -400,
    ),
    (
        "x-ray queen behind rook",
        "3qk3/3r4/8/3p4/8/8/3R4/3QK3 w - - 0 1",
        "d2d5",
        -400,
    ),
    # ⚑⚑ THE TWO ROWS ABOVE ARE NAMED "x-ray" AND DO NOT TEST X-RAY. MEASURED:
    # with the reveal deleted (`occ &= ~sq_bit(lva_sq)` removed) both still
    # return -400, because the swap is already losing by the time the hidden
    # slider would matter — white wins a pawn and loses a rook either way, so
    # whether the piece behind ever joins changes nothing. They are correct rows
    # that happen to be blind to the property their name claims. Kept as valid
    # data; the two rows below are the ones that actually bind.
    #
    # Here the reveal DECIDES the sign: Rd2xd5 wins the pawn, Rd6 recaptures,
    # and the revealed Rd1 wins the rook back — +100 with x-ray, -400 without.
    (
        "x-ray reveal decides the sign",
        "4k3/8/3r4/3p4/8/8/3R4/3RK3 w - - 0 1",
        "d2d5",
        100,
    ),
    # ⚑ THREE DEEP, so a reveal must itself reveal another. Rd3xd5, Rd6 takes,
    # Rd2 (revealed) takes, Rd7 takes, Rd1 (revealed again) takes. An evaluator
    # that re-scanned occupancy once and then stopped gets this wrong even though
    # it passes the two-deep row above it.
    (
        "three-deep x-ray, each reveal uncovering the next",
        "4k3/3r4/3r4/3p4/8/3R4/3R4/3RK3 w - - 0 1",
        "d3d5",
        100,
    ),
    # En passant: the victim is not on the destination square.
    ("en passant", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", "e5d6", 100),
    # ⚑ EN PASSANT WHERE THE VICTIM WAS BLOCKING THE SWAP, WHICH THE ROW ABOVE
    # DOES NOT TEST. Removing the d5 pawn from the occupancy is what lets Rd1
    # reach d6 later in the sequence, so a SEE that scored the EP victim but
    # left it on the board reads this as 0 instead of 100. The simple row above
    # has no follow-up captures at all and survives that mutant untouched —
    # which it did, until this row was added.
    (
        "en passant unblocks the file behind it",
        "3r3k/8/8/3pP3/8/8/8/3RK3 w - d6 0 2",
        "e5d6",
        100,
    ),
    # Promotion on an empty square still gains the upgrade.
    ("quiet promotion", "4k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "b7b8q", 800),
    ("under-promotion", "4k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "b7b8n", 220),
    # Promotion capturing a rook, with the promoted queen then taken.
    (
        "capture-promotion, recaptured",
        "1r2k3/P7/2n5/8/8/8/8/4K3 w - - 0 1",
        "a7b8q",
        400,
    ),
    # A king declines to recapture into a defended square. ⚑ There is no guard
    # in the C for this — the optional-recapture fold produces it, because
    # winning a king at 20000 makes the continuation hugely negative. See the ⚑⚑
    # block in _fastq_see.h for the mutant that established the guard was
    # unobservable and removed it.
    (
        "king declines to recapture into a defended square",
        "4k3/3p4/8/8/8/8/3R4/3RK3 w - - 0 1",
        "d2d7",
        100,
    ),
    # Queen takes a pawn defended by a pawn.
    ("queen for pawn", "4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1", "d1d5", -800),
    # ⚑ A PAWN THAT RECAPTURES *ONTO* THE PROMOTION SQUARE PROMOTES TOO, AND
    # NOTHING ELSE IN THIS LIST TESTS IT. It needs two white pawns bearing on one
    # 8th-rank square, because only White promotes on rank 8 — a black pawn can
    # never recapture there, so the second promotion has to be White's as well:
    # axb8=Q wins the rook, Nxb8 takes the new queen, and cxb8=Q promotes a
    # SECOND time. Ignoring that upgrade scores the row 720 instead of 1300, and
    # every other promotion fixture here survives the mutant because in each of
    # them the promotion is the FIRST move, which is handled by separate code.
    (
        "a recapturing pawn promotes as well",
        "1r5k/P1Pn4/8/8/8/8/8/4K3 w - - 0 1",
        "a7b8q",
        1300,
    ),
]

#: (label, fen, uci, see_value, oracle_value) — rows where the two DISAGREE, and
#: must. Both numbers are asserted, so the size of the approximation is pinned.
#:
#: ⚑⚑ THE SPEC NAMES ONE BLIND SPOT; THE CORPUS FOUND TWO, AND THEY ARE THE SAME
#: ONE. §5 documents "pins are deliberately ignored", which is a special case of
#: the real property: static SEE has NO notion of legality at all. It assumes
#: every attacker of the square can actually capture, and a pin is only one of
#: the reasons that can be false. The other is a check: if a capture leaves the
#: recapturing side in check — including a DISCOVERED check from a piece that
#: has nothing to do with the target square — the recapture SEE counted on does
#: not exist. Both rows below are held here so the approximation is stated at its
#: true width rather than at the width the spec happened to mention.
EXPECTED_DIVERGENCES = [
    # ⚑ THE PIN MUST BE PERPENDICULAR TO THE CAPTURE, AND THE FIRST ATTEMPT AT
    # THIS FIXTURE WAS VACUOUS FOR MISSING IT. Pinning the d7 rook along the
    # D-FILE (king d8, white rook arriving on d5) leaves `is_pinned` reporting
    # True while Rxd5 stays perfectly legal — a piece pinned along a line may
    # still move ALONG that line, so SEE and the oracle agreed and the row
    # proved nothing. Here the rook is pinned along RANK 7 by Rh7 against Ka7,
    # so stepping off the rank to d5 is illegal and the divergence is real.
    (
        "pinned defender cannot recapture",
        "8/k2r3R/8/3p4/8/8/8/3RK3 w - - 0 1",
        "d1d5",
        -400,
        100,
    ),
    # ⚑ DISCOVERED CHECK, WHICH IS WHY THE RECAPTURER IS NOT THE CHECKING PIECE.
    # axb4 wins a knight; cxb4 recaptures AND opens the c-file, so Qc8 gives
    # check and White's Qb3xb4 becomes illegal — it neither blocks the file nor
    # takes the checker. SEE counts that queen recapture and scores the whole
    # sequence a knight; the oracle stops one ply earlier and scores knight
    # minus pawn. A fixture where the checking piece IS the one on the target
    # square would not reproduce this: capturing it would resolve the check.
    (
        "discovered check makes the recapture illegal",
        "2q4k/8/8/2p5/1n6/PQ6/8/2K5 w - - 0 1",
        "a3b4",
        320,
        220,
    ),
]


@pytest.mark.parametrize(
    ("label", "fen", "uci", "expected"),
    AGREEING_FIXTURES,
    ids=[f[0] for f in AGREEING_FIXTURES],
)
def test_crafted_fixtures_match_the_hand_computed_value(
    label: str, fen: str, uci: str, expected: int
) -> None:
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves, f"{label}: fixture move is not legal"
    assert _see(board, move) == expected, label


@pytest.mark.parametrize(
    ("label", "fen", "uci", "expected"),
    AGREEING_FIXTURES,
    ids=[f[0] for f in AGREEING_FIXTURES],
)
def test_crafted_fixtures_match_the_brute_force_oracle(
    label: str, fen: str, uci: str, expected: int
) -> None:
    """The same rows again, against the oracle rather than the hand computation.

    Both assertions earn their place: the hand-computed number catches an oracle
    that is wrong in the same direction as the implementation, and the oracle
    catches a hand-computed number that was quietly copied from a failing run.
    """
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    assert _oracle_see(board, move) == expected, f"{label}: oracle disagrees"
    assert _see(board, move) == _oracle_see(board, move), label


@pytest.mark.parametrize(
    ("label", "fen", "uci", "see_value", "oracle_value"),
    EXPECTED_DIVERGENCES,
    ids=[f[0] for f in EXPECTED_DIVERGENCES],
)
def test_legality_is_ignored_and_the_divergence_is_exactly_this_big(
    label: str, fen: str, uci: str, see_value: int, oracle_value: int
) -> None:
    """⚑ AN EXPECTED FAILURE, ASSERTED IN BOTH DIRECTIONS.

    `assert see != oracle` alone would be satisfied by SEE returning anything at
    all, including garbage. Pinning both values makes this a measurement of the
    approximation: it fails if SEE changes, if the oracle changes, or if they
    ever agree.
    """
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves, f"{label}: fixture move is not legal"
    assert _see(board, move) == see_value, f"{label}: SEE"
    assert _oracle_see(board, move) == oracle_value, f"{label}: oracle"
    assert see_value != oracle_value, f"{label}: fixture no longer diverges"


def test_the_python_value_table_matches_the_c_one_it_mirrors() -> None:
    """SEE_VALUE above is a copy; this is what stops it drifting from the header.

    A single trade tells the two tables apart, so each entry is read back through
    a position whose SEE is exactly that piece's value.
    """
    for piece, value in SEE_VALUE.items():
        if piece == chess.KING:
            continue  # a king is never won; its value has no such fixture
        symbol = chess.piece_symbol(piece).lower()
        # Black `piece` alone on d5, taken by an undefended white rook on d1.
        board = chess.Board(f"4k3/8/8/3{symbol}4/8/8/8/3RK3 w - - 0 1")
        move = chess.Move.from_uci("d1d5")
        assert move in board.legal_moves
        assert _see(board, move) == value, chess.piece_name(piece)


# ===========================================================================
# The sweep: every capture in a deterministic corpus
# ===========================================================================


def _corpus() -> list[chess.Board]:
    """Seeded legal-move play, sampled from ply 4 so captures actually exist."""
    out: list[chess.Board] = []
    rng = random.Random(20260826)
    for _game in range(40):
        board = chess.Board()
        for ply in range(48):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            if ply >= 4:
                out.append(board.copy(stack=False))
    return out


def _capture_rows() -> list[tuple[chess.Board, chess.Move]]:
    rows: list[tuple[chess.Board, chess.Move]] = []
    for board in _corpus():
        rows.extend(
            (board, move)
            for move in board.legal_moves
            if board.is_capture(move) or move.promotion
        )
    return rows


CAPTURE_ROWS = _capture_rows()


def _legality_binds_somewhere_in_the_swap(
    board: chess.Board, move: chess.Move
) -> bool:
    """Does move LEGALITY constrain the capture sequence anywhere it can reach?

    ⚑ THE CLASSIFIER WALKS THE WHOLE SWAP, NOT JUST ITS FIRST PLY, AND THAT
    MATTERS. An earlier version tested the root and the position after the first
    capture only. It left two corpus rows "unexplained" that turned out to be
    the ordinary case one ply deeper: a recapture opens a line, the recapturing
    side gives DISCOVERED check, and the reply SEE counted on is illegal. Two
    unexplained rows is exactly the size of residue that invites an
    allow-a-few-mismatches tolerance, which would then absorb a genuine bug —
    so the predicate was widened to the true cause instead.

    The cause is one thing wearing two hats: static SEE has no notion of
    legality. It assumes every attacker can capture. A pin and a check are both
    reasons that assumption fails.
    """
    target = move.to_square

    def bound(bd: chess.Board) -> bool:
        if bd.is_check():
            return True
        return any(
            bd.is_pinned(color, square)
            for color in (chess.WHITE, chess.BLACK)
            for square in bd.attackers(color, target)
        )

    def walk(bd: chess.Board) -> bool:
        if bound(bd):
            return True
        for candidate in list(bd.legal_moves):
            if candidate.to_square != target:
                continue
            if not (bd.is_capture(candidate) or candidate.promotion):
                continue
            bd.push(candidate)
            try:
                if walk(bd):
                    return True
            finally:
                bd.pop()
        return False

    if bound(board):
        return True
    after = board.copy(stack=False)
    after.push(move)
    return walk(after)


def test_the_sweep_corpus_is_big_and_really_contains_what_it_claims() -> None:
    """The sweep is a premise; assert it rather than assume it."""
    assert len(CAPTURE_ROWS) >= 2000
    assert sum(1 for _b, m in CAPTURE_ROWS if m.promotion) >= 5
    assert sum(1 for b, m in CAPTURE_ROWS if b.is_en_passant(m)) >= 3
    # Rows where a swap actually happens, rather than a free grab.
    contested = sum(
        1 for b, m in CAPTURE_ROWS if b.attackers(not b.turn, m.to_square)
    )
    assert contested >= 300, contested


def test_see_agrees_with_the_oracle_wherever_legality_does_not_bind() -> None:
    """⚑⚑ THE FILE'S PRIMARY ASSERTION: EXACT, NOT APPROXIMATE.

    Every capture in the corpus, bitwise against a brute-force minimax. The only
    rows allowed to differ are the ones static SEE is DOCUMENTED to get wrong,
    and each of those is classified by its cause rather than waved through by a
    tolerance — an "allow 2% mismatch" gate would pass a real defect the moment
    it stayed under the threshold.
    """
    mismatches: list[tuple[str, str, int, int]] = []
    for board, move in CAPTURE_ROWS:
        got = _see(board, move)
        want = _oracle_see(board, move)
        if got != want and not _legality_binds_somewhere_in_the_swap(board, move):
            mismatches.append((board.fen(), move.uci(), got, want))

    assert not mismatches, (
        f"{len(mismatches)} legality-unconstrained rows disagree with the oracle; "
        "first five:\n"
        + "\n".join(f"  {f} {u}: see={g} oracle={w}" for f, u, g, w in mismatches[:5])
    )


def test_the_legality_exemption_is_narrow_rather_than_a_blanket_excuse() -> None:
    """⚑ THE EXEMPTION ABOVE MUST NOT BE DOING THE WORK.

    `_legality_binds_somewhere_in_the_swap` is permissive by design — it fires
    whenever any attacker of the square is pinned or any reachable node is in
    check, whether or not that actually changed the answer. If it covered most
    of the corpus the primary test would be nearly vacuous, so this bounds how
    much it excuses AND requires the great majority of rows to agree outright.
    """
    exempt = [(b, m) for b, m in CAPTURE_ROWS if _legality_binds_somewhere_in_the_swap(b, m)]
    excused = [m for b, m in exempt if _see(b, m) != _oracle_see(b, m)]
    checked_exactly = len(CAPTURE_ROWS) - len(exempt)

    # The population the primary test compares bitwise is the bulk of the corpus.
    assert checked_exactly > (len(CAPTURE_ROWS) * 2) // 3, (
        f"only {checked_exactly}/{len(CAPTURE_ROWS)} rows are checked exactly"
    )
    # ⚑ AND THE MEASURE THAT ACTUALLY MATTERS: how many rows the exemption
    # EXCUSES, not how many it covers. The predicate is deliberately broad — a
    # check anywhere in the swap tree qualifies — so bounding its coverage would
    # only be measuring how often checks occur. What must stay small is the set
    # of rows where it is load-bearing, i.e. exempt AND actually disagreeing.
    assert len(excused) < len(CAPTURE_ROWS) // 100, (
        f"the exemption is excusing {len(excused)}/{len(CAPTURE_ROWS)} disagreements — "
        "static SEE's approximation is wider than this file claims"
    )
    # Most exempt rows agree anyway, which is what shows the exemption is not a
    # dumping ground for rows that would otherwise fail.
    assert len(excused) < len(exempt) // 10, (
        f"{len(excused)}/{len(exempt)} exempt rows disagree"
    )


# ===========================================================================
# The OTHER SEE — a tripwire against unifying them
# ===========================================================================

#: A capture whose victim stands on the destination square. Both implementations
#: see it, which is what makes the divergence row below a divergence rather than
#: a dead encoder.
_SHARED_FIXTURE = ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", chess.D5)

#: An en-passant capture. The victim is on e4/f5 — NOT on the destination square
#: — so a square-keyed evaluator cannot express this capture at all.
_EP_FIXTURE = ("8/8/8/8/4pP2/8/8/K3k3 b - f3 0 1", "e4f3", chess.F3)


def test_the_two_SEEs_in_this_repo_must_not_be_unified() -> None:
    """⚑⚑ A DIVERGENCE-IS-EXPECTED TEST, SO "UNIFY THEM" BREAKS SOMETHING.

    `feat_see_capture` (encoding/_features_impl.h) and `cae_see_capture`
    (nnue/_fastq_see.h) are both static exchange evaluators in this repo, and the
    reasons they are separate live in a comment at the top of each file. A
    comment is not a tripwire: someone who never opens either file can still
    delete one and route both callers at the other, and every existing test would
    pass — the feature planes have their own tests, FastQ's SEE has its own
    oracle, and neither notices that the OTHER one changed.

    So the difference is asserted. The two are observed through the only surface
    that shows both: FastQ's `see()` directly, and `feat_see_capture` through the
    v3_see planes it is the sole feeder of.

    ⚑ ANTI-VACUITY FIRST. The shared fixture proves the plane encoder is live and
    agrees on an ordinary capture, so the EP row below is a real disagreement and
    not two silent zeros.

    ⚑ THE PLANES ARE NOT PRODUCTION INPUT. v3_see is a separate 65-plane family
    from the 63-plane v2_threats `configs/pbt2_small.yaml` selects, so this test
    reads a research encoder, not the live one. That is deliberate and is the
    correction to an earlier claim in _fastq_see.h that said otherwise.
    """
    from chess_anti_engine.encoding import features

    fen, uci, square = _SHARED_FIXTURE
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    shared_planes = features.extra_feature_planes_c(board, version="v3_see")
    # [64] is "their pieces' SEE when WE initiate", in pawn units clipped to ±1.
    assert shared_planes[64].flatten()[square] == pytest.approx(0.125), (
        "the v3_see encoder no longer scores an ordinary hanging capture; the "
        "divergence assertion below would be comparing two dead zeros"
    )
    assert _see(board, move) == 100, "FastQ's SEE no longer scores it either"

    fen, uci, square = _EP_FIXTURE
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves
    assert board.is_en_passant(move)

    assert _see(board, move) == 100, (
        "FastQ's SEE must score the en-passant capture; it is move-based"
    )
    ep_planes = features.extra_feature_planes_c(board, version="v3_see")
    assert ep_planes[63].flatten()[square] == 0.0
    assert ep_planes[64].flatten()[square] == 0.0
    unified = (
        "feat_see_capture now expresses en passant, so the two evaluators may "
        "have been unified — read the ⚑⚑ block at the top of _fastq_see.h and "
        "docs/fastq_search.md before deleting this test"
    )
    assert not ep_planes[63].any(), unified
    assert not ep_planes[64].any(), unified
