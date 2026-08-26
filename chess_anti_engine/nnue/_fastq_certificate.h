/*
 * _fastq_certificate.h — FastQ's quiet certificate (docs/fastq_design.md §3.1).
 *
 *   quiet(node) := !in_check && no promotion available && no capture with SEE >= 0
 *
 * ⚑⚑ THIS FILE EXISTS SEPARATELY TO MAKE §4.2's BOUNDARY STRUCTURAL RATHER THAN
 * COMMENTED. §4.1 permits the certificate to be cached against a canonical DAG
 * node; §4.2 forbids caching anything the SEARCH computed, because a value that
 * depended on a window would let the first caller's alpha/beta decide the answer
 * for every later one. What separates the two is not intent, it is whether the
 * quantity is a function of the position alone — and the way to keep that
 * checkable is to compute the cacheable facts in a file that has no access to a
 * window, an alpha, a beta or a search context. Nothing here takes one, and
 * `cae_fastq_certificate` takes no argument but the board.
 *
 * The caching wrapper that reads and writes the DAG payload lives in
 * _fastq_search.h with the context it needs; only the pure predicate is here.
 *
 * Requires (via the includer): _cboard_impl.h for CBoard/POLICY_LUT,
 * _fastq_see.h for cae_see_capture, _nnue_dag_store.h for the CAE_DAG_CERT_* bits.
 */

#ifndef _CAE_FASTQ_CERTIFICATE_H
#define _CAE_FASTQ_CERTIFICATE_H

#include <stdint.h>

/* Moves considered at one node. cboard_legal_move_indices can emit up to
 * CBOARD_MAX_LEGAL_MOVES, and evasion nodes generate ALL of them. */
#define CAE_FASTQ_MAX_MOVES CBOARD_MAX_LEGAL_MOVES

/* ⚑⚑ POLICY_LUT's `promotion` FIELD IS SPECULATIVE AND MUST NOT BE TRUSTED BY
 * ITS NAME. init_policy_lut stamps PROMO_MAYBE_QUEEN (5) onto EVERY entry whose
 * destination is rank 0 or rank 7, whatever piece is moving, and cboard_push
 * resolves it at push time by checking whether a pawn was actually involved. So
 * a rook lifting to the back rank — or a king stepping along it — carries
 * promotion == 5.
 *
 * This bit me exactly as it bit the qsearch arm before it (see the ⚑⚑ on
 * cae_qsearch_is_tactical). Reading the field by its name made FastQ generate
 * every back-rank king and queen move as "tactical" and mark quiet positions as
 * having a promotion available: the §8.6 fixture whose only forcing moves are
 * quiet checks was searched instead of standing pat, returning 251 where the
 * static value is 26. The piece test is what makes the field mean what it is
 * being read to mean. */
static inline int cae_fastq_promotion_of(const CBoard *b, const PolicyMove *pm) {
    if (pm->promotion == 0) return 0;
    if (piece_type_at(b, pm->from_sq) != PAWN) return 0;
    return pm->promotion;
}

/* An en-passant capture. The file test distinguishes it from a pawn PUSH onto
 * the same square, mirroring cae_qsearch_is_tactical rather than inventing a
 * second spelling of the same rule. */
static inline int cae_fastq_is_en_passant(const CBoard *b, const PolicyMove *pm) {
    return b->ep_square >= 0 && pm->to_sq == b->ep_square
           && piece_type_at(b, pm->from_sq) == PAWN
           && sq_file(pm->from_sq) != sq_file(pm->to_sq);
}

static inline int cae_fastq_is_capture(const CBoard *b, const PolicyMove *pm) {
    return (b->occ[1 - b->turn] & sq_bit(pm->to_sq)) != 0
           || cae_fastq_is_en_passant(b, pm);
}

/* Compute the certificate bits for `board`. NO WINDOW ARGUMENT, and that is the
 * property §8 mutant 1 attacks: every term below is a function of the position
 * alone, which is what makes the result storable against a canonical node. */
static uint8_t cae_fastq_certificate(const CBoard *board)
{
    uint8_t bits = CAE_DAG_CERT_COMPUTED;
    if (cboard_in_check(board)) return (uint8_t)(bits | CAE_DAG_CERT_IN_CHECK);

    int moves[CAE_FASTQ_MAX_MOVES];
    const int n = cboard_legal_move_indices(board, moves, 0);
    for (int i = 0; i < n; i++) {
        const PolicyMove pm = POLICY_LUT[board->turn][moves[i]];
        const int promo = cae_fastq_promotion_of(board, &pm);
        if (promo != 0) {
            bits |= CAE_DAG_CERT_PROMOTION;
            continue;
        }
        if (!cae_fastq_is_capture(board, &pm)) continue;
        if (cae_see_capture(board, pm.from_sq, pm.to_sq, promo) >= 0) {
            bits |= CAE_DAG_CERT_GOOD_CAP;
            /* Both loud bits are set as soon as they are known, but a promotion
             * elsewhere in the list can still turn up, so the scan continues.
             * Bailing here would make the stored bits depend on move ORDER,
             * which is not part of the position. */
        }
    }
    return bits;
}

/* ⚑ A NODE WITH NO CERTIFICATE IS NOT QUIET. The COMPUTED bit is tested rather
 * than assumed because an all-zero payload — a freshly interned node, or a node
 * some other arm published — would otherwise read as "no loud bits set" and be
 * stood pat on. Absence of evidence is spelled out here so it cannot be read as
 * evidence of absence. */
static inline int cae_fastq_is_quiet(uint8_t bits)
{
    return (bits & CAE_DAG_CERT_COMPUTED) && !(bits & CAE_DAG_CERT_LOUD);
}

#endif /* _CAE_FASTQ_CERTIFICATE_H */
