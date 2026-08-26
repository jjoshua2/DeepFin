/*
 * _nnue_state.h — the incremental NNUE accumulator state, make-on-copy.
 *
 * Extracted from _arm_providers.h so that the two consumers which both need it
 * can each include it directly instead of one reaching into the other: the
 * quiescence arms (_arm_providers.h) and the canonical structural-position DAG
 * store (_nnue_dag_store.h). The code below is unchanged by that move.
 *
 * ⚑ WHY A STATE IS SAFE TO CACHE AGAINST A STRUCTURAL POSITION. cae_nnue_state_*
 * reads nothing but the board: cae_nnue_pos_from_cboard() fills a CaeNnuePos from
 * the bitboards, occupancy, side to move and in-check flag, and nothing else —
 * no halfmove clock, no repetition history, no castling rights, no en passant.
 * The accumulator is a sum over the active feature multiset with exact modular
 * int16/int32 add and subtract, so it does not depend on the ORDER the features
 * were applied in and therefore not on the path the position was reached by.
 * That is what lets _nnue_dag_store.h store one state per canonical position and
 * reuse it on a transposition, and it is why the DAG holds only history-free,
 * window-independent facts.
 *
 * ⚑ INCLUDED BY EXACTLY ONE TRANSLATION UNIT — _nnue_ext.c — for the same reason
 * _arm_providers.h is: it pulls in _nnue_provider.h, whose header-only statics
 * (kernel flag, weight cache) must exist in exactly one copy.
 */

#ifndef CAE_NNUE_STATE_H
#define CAE_NNUE_STATE_H

#include <string.h>

#include "_nnue_provider.h"

/* ================================================================
 * Incremental NNUE state for qsearch
 * ================================================================
 *
 * CBoard search is already make-on-copy: every child is memcpy(parent), then
 * cboard_push_index(child). Mirror that exact ownership rule for NNUE. A state
 * belongs to one board and a child state is derived from the parent state plus
 * the already-pushed child board. There is deliberately NO unmake API here.
 * Unmake only buys something if the search itself becomes one mutable DFS board;
 * with owned child copies it would add reversible bookkeeping and another way to
 * corrupt a sibling for no saved work.
 *
 * HalfKAv2_hm is cheap to update exactly from changed squares while the king
 * bucket is unchanged. FullThreats is more global: moving one blocker can alter
 * several sliding relations. We therefore cache the ACTIVE threat indices and
 * diff the old/new sorted sets. That still avoids touching the dozens of 1024-
 * wide weight rows whose features did not change. A king move changes the
 * orientation/bucket for an entire perspective, so that perspective falls back
 * to the existing exact full refresh. The cache is capped at Stockfish's own
 * active-threat ceiling (128); an unexpected wider legal position also falls
 * back rather than becoming a correctness assumption.
 */
#define CAE_NNUE_INC_THREAT_CACHE 128

typedef struct CaeNnueState {
    CaeNnuePos pos;
    CaeNnueAcc acc;
    uint32_t threat_idx[2][CAE_NNUE_INC_THREAT_CACHE];
    uint16_t threat_count[2];
    uint8_t threat_cache_valid;
} CaeNnueState;

static inline void cae_nnue_inc_sub_row_i16(int16_t *acc, const int16_t *row, uint32_t n) {
#if CAE_NNUE_HAVE_AVX2
    if (cae_nnue_simd_enabled) {
        for (uint32_t j = 0; j < n; j += 16) {
            __m256i a = _mm256_load_si256((const __m256i *)(acc + j));
            __m256i b = _mm256_loadu_si256((const __m256i *)(row + j));
            _mm256_store_si256((__m256i *)(acc + j), _mm256_sub_epi16(a, b));
        }
        return;
    }
#endif
    for (uint32_t j = 0; j < n; j++)
        acc[j] = (int16_t)((uint16_t)acc[j] - (uint16_t)row[j]);
}

static inline void cae_nnue_inc_sub_row_i8(int16_t *acc, const int8_t *row, uint32_t n) {
#if CAE_NNUE_HAVE_AVX2
    if (cae_nnue_simd_enabled) {
        for (uint32_t j = 0; j < n; j += 16) {
            __m256i a = _mm256_load_si256((const __m256i *)(acc + j));
            __m128i w8 = _mm_loadu_si128((const __m128i *)(row + j));
            _mm256_store_si256((__m256i *)(acc + j),
                               _mm256_sub_epi16(a, _mm256_cvtepi8_epi16(w8)));
        }
        return;
    }
#endif
    for (uint32_t j = 0; j < n; j++)
        acc[j] = (int16_t)((uint16_t)acc[j] - (uint16_t)(int16_t)row[j]);
}

static inline void cae_nnue_inc_sort_u32(uint32_t *v, int n) {
    /* Tiny lists (normally ~tens of entries): insertion sort beats pulling a
     * comparator/function-pointer qsort into every make. */
    for (int i = 1; i < n; i++) {
        uint32_t x = v[i];
        int j = i;
        while (j > 0 && v[j - 1] > x) {
            v[j] = v[j - 1];
            j--;
        }
        v[j] = x;
    }
}

static int cae_nnue_inc_build_threat_cache(
    const CaeNnueWeights *w,
    const CaeNnuePos *p,
    uint32_t out[2][CAE_NNUE_INC_THREAT_CACHE],
    uint16_t counts[2],
    uint8_t *valid)
{
    CaeThreatRel rel[CAE_NNUE_MAX_RELATIONS];
    int n_rel = cae_nnue_threat_relations(p, rel);
    if (n_rel < 0) return CAE_VALUE_ERR_BAD_POS;
    counts[0] = counts[1] = 0;
    *valid = 0;

    /* Cache size is an optimisation boundary, not a validity boundary. The
     * evaluator itself accepts the larger relation buffer; this path simply
     * declines to delta-update it. */
    if (n_rel > CAE_NNUE_INC_THREAT_CACHE) return CAE_VALUE_OK;

    for (int perspective = 0; perspective < 2; perspective++) {
        int ksq = p->king_sq[perspective];
        for (int i = 0; i < n_rel; i++) {
            uint32_t idx = cae_nnue_threat_index(
                perspective, rel[i].attacker, rel[i].from, rel[i].to,
                p->piece_on[rel[i].to], ksq);
            if (idx >= w->threat_dims) continue;
            if (counts[perspective] >= CAE_NNUE_INC_THREAT_CACHE) {
                counts[0] = counts[1] = 0;
                return CAE_VALUE_OK;
            }
            out[perspective][counts[perspective]++] = idx;
        }
        cae_nnue_inc_sort_u32(out[perspective], counts[perspective]);
    }
    *valid = 1;
    return CAE_VALUE_OK;
}

static inline void cae_nnue_inc_halfka_feature(
    const CaeNnueWeights *w, CaeNnueAcc *a, int perspective,
    int sq, int piece, int ksq, int add)
{
    uint32_t idx = cae_nnue_halfka_index(perspective, sq, piece, ksq);
    const int16_t *row = w->ft_weight + (size_t)idx * w->l1;
    if (add) cae_nnue_add_row_i16(a->acc[perspective], row, w->l1);
    else cae_nnue_inc_sub_row_i16(a->acc[perspective], row, w->l1);

    const int32_t *prow = w->ft_psqt + (size_t)idx * w->psqt_buckets;
    for (uint32_t k = 0; k < w->psqt_buckets; k++)
        a->psqt[perspective][k] += add ? prow[k] : -prow[k];
}

static inline void cae_nnue_inc_threat_feature(
    const CaeNnueWeights *w, CaeNnueAcc *a, int perspective, uint32_t idx, int add)
{
    const int8_t *row = w->threat_weight + (size_t)idx * w->l1;
    if (add) cae_nnue_add_row_i8(a->acc[perspective], row, w->l1);
    else cae_nnue_inc_sub_row_i8(a->acc[perspective], row, w->l1);

    const int32_t *prow = w->threat_psqt + (size_t)idx * w->psqt_buckets;
    for (uint32_t k = 0; k < w->psqt_buckets; k++)
        a->psqt[perspective][k] += add ? prow[k] : -prow[k];
}

static int cae_nnue_state_init(
    const CaeNnueWeights *w, const CBoard *board, CaeNnueState *out)
{
    CaeNnuePos pos;
    int rc = cae_nnue_pos_from_cboard(board, &pos);
    if (rc != CAE_VALUE_OK) return rc;

    rc = cae_nnue_refresh(w, &pos, &out->acc);
    if (rc != CAE_VALUE_OK) return rc;
    out->pos = pos;
    return cae_nnue_inc_build_threat_cache(
        w, &pos, out->threat_idx, out->threat_count, &out->threat_cache_valid);
}

static int cae_nnue_state_evaluate(
    const CaeNnueWeights *w, const CaeNnueState *state, int32_t *out_value)
{
    if (!w) return CAE_VALUE_ERR_NOT_LOADED;
    if (state->pos.in_check) return CAE_VALUE_ERR_IN_CHECK;
    int bucket = cae_nnue_bucket(&state->pos);
    int rc = cae_nnue_check_bucket(w, bucket);
    if (rc != CAE_VALUE_OK) return rc;

    uint8_t ft[CAE_NNUE_MAX_L1] __attribute__((aligned(32)));
    int32_t psqt = cae_nnue_transform(w, &state->acc, state->pos.side_to_move, bucket, ft);
    int32_t positional = cae_nnue_propagate(w, ft, bucket);
    *out_value = psqt / CAE_NNUE_OUTPUT_SCALE + positional / CAE_NNUE_OUTPUT_SCALE;
    return CAE_VALUE_OK;
}

static int cae_nnue_state_make(
    const CaeNnueWeights *w,
    const CaeNnueState *parent,
    const CBoard *child_board,
    CaeNnueState *out)
{
    CaeNnuePos next;
    int rc = cae_nnue_pos_from_cboard(child_board, &next);
    if (rc != CAE_VALUE_OK) return rc;

    uint32_t new_threat[2][CAE_NNUE_INC_THREAT_CACHE] = {{0}};
    uint16_t new_count[2] = {0, 0};
    uint8_t new_valid = 0;
    rc = cae_nnue_inc_build_threat_cache(w, &next, new_threat, new_count, &new_valid);
    if (rc != CAE_VALUE_OK) return rc;

    /* Copy is intentional. Search already gives each child its own CBoard; the
     * accumulator follows the same ownership model, so sibling state cannot be
     * corrupted and no unmake log is required. ~5 KiB/node is small beside the
     * weight-row traffic this removes and can be revisited only if profiling
     * identifies the copy itself as the next bottleneck. */
    *out = *parent;

    int refresh_perspective[2];
    for (int perspective = 0; perspective < 2; perspective++) {
        refresh_perspective[perspective] =
            !parent->threat_cache_valid || !new_valid
            || parent->pos.king_sq[perspective] != next.king_sq[perspective];
    }

    CaeNnueAcc fresh;
    if (refresh_perspective[0] || refresh_perspective[1]) {
        rc = cae_nnue_refresh(w, &next, &fresh);
        if (rc != CAE_VALUE_OK) return rc;
    }

    for (int perspective = 0; perspective < 2; perspective++) {
        if (refresh_perspective[perspective]) {
            memcpy(out->acc.acc[perspective], fresh.acc[perspective],
                   (size_t)w->l1 * sizeof(int16_t));
            memcpy(out->acc.psqt[perspective], fresh.psqt[perspective],
                   (size_t)w->psqt_buckets * sizeof(int32_t));
            continue;
        }

        int ksq = next.king_sq[perspective];
        for (int sq = 0; sq < 64; sq++) {
            int old_piece = parent->pos.piece_on[sq];
            int new_piece = next.piece_on[sq];
            if (old_piece == new_piece) continue;
            if (old_piece)
                cae_nnue_inc_halfka_feature(w, &out->acc, perspective,
                                            sq, old_piece, ksq, 0);
            if (new_piece)
                cae_nnue_inc_halfka_feature(w, &out->acc, perspective,
                                            sq, new_piece, ksq, 1);
        }

        /* Sorted multiset difference. Equal indices cancel; old-only rows are
         * subtracted and new-only rows are added. This handles sliding rays,
         * captures, en-passant and promotions without trying to predict which
         * threat relations a move is capable of changing. */
        int i = 0, j = 0;
        int old_n = parent->threat_count[perspective];
        int new_n = new_count[perspective];
        while (i < old_n || j < new_n) {
            if (i < old_n && j < new_n
                && parent->threat_idx[perspective][i] == new_threat[perspective][j]) {
                i++;
                j++;
            } else if (j >= new_n
                       || (i < old_n
                           && parent->threat_idx[perspective][i] < new_threat[perspective][j])) {
                cae_nnue_inc_threat_feature(
                    w, &out->acc, perspective, parent->threat_idx[perspective][i], 0);
                i++;
            } else {
                cae_nnue_inc_threat_feature(
                    w, &out->acc, perspective, new_threat[perspective][j], 1);
                j++;
            }
        }
    }

    out->pos = next;
    out->threat_cache_valid = new_valid;
    out->threat_count[0] = new_count[0];
    out->threat_count[1] = new_count[1];
    if (new_valid)
        memcpy(out->threat_idx, new_threat, sizeof(new_threat));
    return CAE_VALUE_OK;
}

#endif /* CAE_NNUE_STATE_H */
