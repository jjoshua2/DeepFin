/*
 * _fastq_see.h — static exchange evaluation for the FastQ verifier.
 *
 * A swap-off evaluator on one target square: play out the capture sequence with
 * each side always recapturing with its least valuable attacker, and fold the
 * result back with the option to stand pat at every ply. Used for BOTH move
 * ordering and the SEE gate of docs/fastq_design.md §3.4 — one computation, two
 * consumers, so an ordering change cannot silently disagree with the gate.
 *
 * ⚑⚑ THIS IS NOT THE REPO'S ONLY SEE, AND THE OTHER ONE MUST NOT BE REUSED HERE.
 * `feat_see_capture` in encoding/_features_impl.h is a working swap evaluator,
 * and reusing it would look like the obvious anti-duplication move. Three
 * reasons, each sufficient on its own:
 *
 *   1. It is compiled into _lc0_ext / _features_ext / _mcts_tree, NOT into
 *      _nnue_ext, which includes only _cboard_impl.h. Reaching it means pulling
 *      1,384 lines of feature-plane code into this extension.
 *   2. Its values are pawn units {1,3,3,5,9,1000}. FastQ's delta pruning
 *      compares a victim's value against an NNUE stand-pat in internal units,
 *      so the scale has to match the evaluator, not a feature plane.
 *   3. It is square-based, not move-based — it answers "what happens if the
 *      swap on this square starts" — and it handles neither en passant (whose
 *      victim does not stand on the capture square, so the plane loop never
 *      even visits it) nor promotion. §5 requires both.
 *
 * ⚑⚑ AN EARLIER REVISION OF THIS BLOCK CLAIMED A FOURTH, "DISQUALIFYING" REASON:
 * that feat_see_capture feeds the live net's input planes. THAT WAS WRONG, and
 * it is the "diff the file you measured against production" trap in miniature.
 * Its output reaches only the FEAT_EXTRA_V3_SEE planes (_features_impl.h:902-934),
 * and v3_see is a SEPARATE 65-plane family from the 63-plane v2_threats that
 * `configs/pbt2_small.yaml` selects — the graded-SEE planes are appended after
 * index 63 and exist only when v3_see is chosen. Production does not read them,
 * so changing feat_see_capture would NOT move the live input distribution.
 *
 * What is true instead, and it is a weaker claim: v3_see is a shipped encoder
 * with a CLOSED experimental verdict (the v3 feature family was tested and
 * rejected). Changing feat_see_capture would silently invalidate the
 * reproducibility of that verdict without touching production. Reasons 1-3 are
 * what actually decide this; reason 4 is a caution, not a disqualification.
 *
 * ⇒ two SEEs exist ON PURPOSE, at different scales, for different consumers.
 * tests/test_fastq_see.py pins this file's behaviour against a brute-force
 * oracle AND pins one case where the two must disagree, so unifying them breaks
 * a test rather than passing quietly. The reverse pointer lives above
 * feat_see_capture itself.
 *
 * ⚑ Pins are deliberately ignored (static SEE, Stockfish-style): an attacker
 * that is absolutely pinned to its own king is still counted. This is a known
 * approximation, and it is PINNED RATHER THAN HIDDEN — the oracle fixtures
 * include crafted pinned-attacker rows asserted as EXPECTED divergences, so the
 * approximation has a test that fails if its shape ever changes.
 */

#ifndef _CAE_FASTQ_SEE_H
#define _CAE_FASTQ_SEE_H

#include <stdint.h>

/* Internal-unit piece values, on the same scale as the NNUE evaluation the
 * search compares them against (roughly centipawns). The king's value only has
 * to exceed any reachable swap total; it is never actually won. */
static const int32_t CAE_SEE_VALUE[6] = {100, 320, 330, 500, 900, 20000};

/* A swap on one square cannot exceed 32 captures (each removes a piece and at
 * most 30 non-king pieces can ever be captured); 40 leaves slack for the
 * promotion bookkeeping without a bounds question. */
#define CAE_SEE_MAX_SWAP 40

static inline int32_t cae_see_max(int32_t a, int32_t b) { return a > b ? a : b; }

/* Square of `color`'s least valuable attacker of `target` under `occ`, or -1.
 *
 * ⚑ EVERY LOOKUP IS MASKED BY `occ`, WHICH IS WHAT MAKES X-RAYS WORK. The
 * sliding lookups are recomputed against the CURRENT occupancy each call, so a
 * rook behind a bishop that has just been removed from `occ` becomes an
 * attacker on the next ply with no special-case code. The piece bitboards
 * themselves are never mutated — `occ` alone tracks the swap.
 */
static int cae_see_least_valuable_attacker(
    const CBoard *b, int target, int color, uint64_t occ, int *out_pt)
{
    const uint64_t side = b->occ[color] & occ;

    /* PAWN_ATTACKS[c][sq] is "squares a pawn OF COLOUR c standing on sq
     * attacks", so the pawns attacking `target` are found by indexing the
     * OPPOSITE colour's table at `target`. */
    uint64_t m = PAWN_ATTACKS[1 - color][target] & b->bb[PAWN] & side;
    if (m) { *out_pt = PAWN; return lsb64(m); }

    m = KNIGHT_ATTACKS[target] & b->bb[KNIGHT] & side;
    if (m) { *out_pt = KNIGHT; return lsb64(m); }

    const uint64_t diag = bishop_attacks(target, occ);
    m = diag & b->bb[BISHOP] & side;
    if (m) { *out_pt = BISHOP; return lsb64(m); }

    const uint64_t orth = rook_attacks(target, occ);
    m = orth & b->bb[ROOK] & side;
    if (m) { *out_pt = ROOK; return lsb64(m); }

    m = (diag | orth) & b->bb[QUEEN] & side;
    if (m) { *out_pt = QUEEN; return lsb64(m); }

    m = KING_ATTACKS[target] & b->bb[KING] & side;
    if (m) { *out_pt = KING; return lsb64(m); }

    *out_pt = -1;
    return -1;
}

static inline int cae_see_is_promotion_rank(int sq) {
    const int r = sq_rank(sq);
    return r == 0 || r == 7;
}

/*
 * SEE of the capture (or promotion) `from_sq -> to_sq` for the side to move.
 *
 * `promotion` follows cboard_push's encoding (0 none, 2 N, 3 B, 4 R, 5 Q); it
 * is consulted only for the FIRST move, because every recapture inside the swap
 * promotes to a queen, which is the only choice that can matter to a material
 * count.
 *
 * Returns the net gain in CAE_SEE_VALUE units. A quiet (non-capture,
 * non-promotion) move scores 0 or worse and is never generated by FastQ; the
 * function still answers for one so the gate has no undefined input.
 */
static int32_t cae_see_capture(const CBoard *b, int from_sq, int to_sq, int promotion)
{
    const int us = b->turn;
    const int them = 1 - us;
    const int mover_pt = piece_type_at(b, from_sq);
    if (mover_pt < 0) return 0;

    uint64_t occ = b->occ[0] | b->occ[1];
    occ &= ~sq_bit(from_sq);

    /* What this move wins outright. */
    int32_t captured_value = 0;
    if (mover_pt == PAWN && b->ep_square >= 0 && to_sq == b->ep_square) {
        /* En passant: the victim stands beside the mover, not on `to_sq`, so it
         * has to leave the occupancy explicitly or the x-ray behind it stays
         * blocked for the rest of the swap. */
        const int victim_sq = make_sq(sq_file(to_sq), sq_rank(from_sq));
        occ &= ~sq_bit(victim_sq);
        captured_value = CAE_SEE_VALUE[PAWN];
    } else {
        const int victim_pt = piece_type_at(b, to_sq);
        if (victim_pt >= 0) captured_value = CAE_SEE_VALUE[victim_pt];
    }

    /* The mover's own promotion, if any: it both adds the upgrade to this ply's
     * gain and changes which piece the opponent is recapturing. */
    int on_square_pt = mover_pt;
    if (mover_pt == PAWN && cae_see_is_promotion_rank(to_sq)) {
        /* cboard_push's promotion codes are 2=N 3=B 4=R 5=Q; PAWN..KING is
         * 0..5. Map, defaulting to a queen when the caller passed none. */
        int promo_pt = QUEEN;
        switch (promotion) {
            case 2: promo_pt = KNIGHT; break;
            case 3: promo_pt = BISHOP; break;
            case 4: promo_pt = ROOK;   break;
            default: promo_pt = QUEEN; break;
        }
        captured_value += CAE_SEE_VALUE[promo_pt] - CAE_SEE_VALUE[PAWN];
        on_square_pt = promo_pt;
    }

    /* `won[k]` is the material the side to move at ply k takes off the board by
     * capturing: the piece standing on the square, plus that capture's own
     * promotion upgrade. Ply 0 is the caller's forced move.
     *
     * ⚑⚑ THIS IS AN EXPLICIT BACKWARD RECURRENCE, NOT THE CLASSIC IN-PLACE
     * FOLD, AND THE DIFFERENCE IS A BUG THIS FILE ALREADY HAD. The textbook
     * version accumulates `gain[d] = value - gain[d-1]` forward, breaks early on
     * `max(-gain[d-1], gain[d]) < 0`, and unwinds with `while (--d)` — whose
     * pre-decrement quietly DISCARDS the pruned ply. Writing the unwind as the
     * obvious `while (d > 0)` folds that discarded ply back in and silently
     * returns a different number: measured on a real corpus row, Bxe7 winning a
     * free queen scored 570 instead of 900, because the bishop recapture that
     * the king would answer was folded in as if it had happened. Stating the
     * recurrence forwards costs one small array and removes the trap. */
    int32_t won[CAE_SEE_MAX_SWAP];
    won[0] = captured_value;
    int plies = 1;

    int side = them;
    for (;;) {
        if (plies >= CAE_SEE_MAX_SWAP) break;

        int lva_pt = -1;
        const int lva_sq = cae_see_least_valuable_attacker(b, to_sq, side, occ, &lva_pt);
        if (lva_sq < 0) break;

        /* ⚑⚑ THERE IS DELIBERATELY NO "A KING MAY NOT CAPTURE INTO A DEFENDED
         * SQUARE" GUARD HERE, AND THAT WAS MEASURED RATHER THAN ASSUMED. One
         * was written, and it is the obvious thing to write: a king that
         * captures into a defended square gets "taken" on the next ply and the
         * swap returns a number no legal sequence can reach. But the optional-
         * recapture fold below already refuses it. At the king's ply the fold
         * computes `won[k] - best`, where `best` contains winning the king at
         * 20000 — so the term is hugely negative, `max(0, ...)` clamps it to
         * zero, the king declines, and every ply after it is discarded.
         *
         * The guard was removed after its mutant was run: deleting it changed
         * NOTHING across 6,422 corpus captures and every crafted fixture,
         * including the one named for it. A branch no test can fail is this
         * repo's signature defect wearing a safety helmet, and it cost an extra
         * attacker scan at every king ply. The fixture stays — it now pins that
         * the FOLD produces the right answer, which is the real mechanism. */
        int32_t value_won = CAE_SEE_VALUE[on_square_pt];
        int next_on_square = lva_pt;
        if (lva_pt == PAWN && cae_see_is_promotion_rank(to_sq)) {
            value_won += CAE_SEE_VALUE[QUEEN] - CAE_SEE_VALUE[PAWN];
            next_on_square = QUEEN;
        }
        won[plies++] = value_won;

        occ &= ~sq_bit(lva_sq);
        on_square_pt = next_on_square;
        side ^= 1;
    }

    /* Fold back. Every recapture is OPTIONAL, so a side that cannot profit
     * declines and the sequence ends there — that `max(0, ...)` is what makes a
     * losing continuation cost nothing rather than being forced. Ply 0 alone is
     * mandatory: the caller asked about that move specifically. */
    int32_t best = 0;
    for (int k = plies - 1; k >= 1; k--) {
        const int32_t take = won[k] - best;
        best = cae_see_max(0, take);
    }
    return won[0] - best;
}

#endif /* _CAE_FASTQ_SEE_H */
