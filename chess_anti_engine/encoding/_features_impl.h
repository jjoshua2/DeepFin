/*
 * _features_impl.h — Pure C implementation of the extra feature planes.
 *
 * Shared between _features_ext.c (Python binding) and _lc0_ext.c (CBoard
 * fused encode path).  All functions are static to avoid ODR issues.
 *
 * Usage: #include this header AFTER defining the standard bitboard macros
 * (popcount64, lsb64, sq_bit, sq_file, sq_rank, make_sq, FOR_EACH_BIT)
 * and after calling import_array() / init_tables_features().
 *
 * Versions exist (model.input_extra_features): v1 = 34 planes,
 * v2_threats = 63 planes (v1 + 29 threat planes appended after index 34),
 * v3_checks = 67 planes (v2_threats + 4 opponent-safe-check planes appended
 * after index 63), v3_xray = 69 planes (v2_threats + 6 per-slider-type x-ray
 * attack planes appended after index 63 — a SEPARATE v3 family from v3_checks,
 * NOT including the opponent-safe-check planes), v3_see = 65 planes
 * (v2_threats + 2 graded-SEE planes appended after index 63), v3_passers = 71
 * planes (v2_threats + 8 passed-pawn-quality planes, 4 per side, appended
 * after index 63). All v3 families are SEPARATE — they share the [63:] start
 * index and are selected by n_extra/version. Existing planes are never
 * reordered. Layout — the Python twin of this table lives in features.py's
 * module docstring; keep both in sync:
 *
 *   [0:10]  king-zone safety (v1)  per side: king zone + N/B/R/Q overlaps
 *   [10:16] pins (v1)              per side: pinned, pin rays, discovered
 *   [16:24] pawn structure (v1)    per side: passed/isolated/backward/connected
 *   [24:30] mobility (v1)          per type: normalized move counts at source
 *   [30:34] outposts (v1)          per side: outposts, space control
 *   [34]    attacked-by-us (v2)    union over piece types
 *   [35]    attacked-by-them (v2)
 *   [36:42] attack maps us (v2)    P, N, B, R, Q, K
 *   [42:48] attack maps them (v2)  P, N, B, R, Q, K
 *   [48]    attacker count us (v2)   /4, clamped to 1
 *   [49]    attacker count them (v2) /4, clamped to 1
 *   [50]    hanging us (v2)        attacked by them, undefended by us
 *   [51]    hanging them (v2)
 *   [52]    cheaper-attacked us (v2)   P→N/B/R/Q, minor→R/Q, R→Q
 *   [53]    cheaper-attacked them (v2)
 *   [54:58] safe checks (v2)       side to move: N, B, R, Q
 *   [58]    control margin (v2)    (us − them attackers)/4, clamp [-1,1]
 *   [59]    pawn tension us (v2)   our pawns that can capture an enemy pawn
 *   [60]    pawn tension them (v2)
 *   [61]    pawn storm vs us (v2)  enemy pawns near OUR king, 1 − rank_dist/7
 *   [62]    pawn storm vs them (v2)
 *   [63:67] opp safe checks (v3_checks)  opponent's N, B, R, Q safe checks vs
 *                                 OUR king (mirror of [54:58])
 *   [63:69] x-ray attacks (v3_xray)  per-slider-type x-ray attacks (squares
 *                                 seen THROUGH the first blocker on each ray):
 *                                 our B, R, Q then their B, R, Q. SEPARATE
 *                                 v3 family from v3_checks — same start index,
 *                                 selected by n_extra/version.
 *   [63:65] SEE (v3_see)          graded Static Exchange Evaluation, normalized
 *                                 clip(see/8, -1, 1): [63] our pieces' SEE when
 *                                 the OPPONENT initiates (<=0 = we lose), [64]
 *                                 their pieces' SEE when WE initiate (>=0 = we
 *                                 win). SEPARATE v3 family.
 *   [63:71] passers (v3_passers)  passed-pawn quality, 4 per side (us 63:67,
 *                                 them 67:71): rank-advancement, safe passer,
 *                                 blocked passer, promo-path enemy-controlled.
 *                                 SEPARATE v3 family.
 *
 * Attack maps / counts are direct attacks at current occupancy (no x-rays);
 * pawn attacks are capture squares only (no en passant). The v3_xray planes
 * are the only x-ray feature: per slider, direct ^ att(s, occ ^ (direct&occ)).
 */

#ifndef FEATURES_IMPL_H
#define FEATURES_IMPL_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>  /* abs() */

/* ================================================================
 * Precomputed tables
 * ================================================================ */

/* Knight/king/pawn attack tables + slider ray helpers are also defined
 * by _cboard_impl.h. When it's been included first (in _lc0_ext.c and
 * _mcts_tree.c) alias to those copies instead of carrying a second set —
 * eliminates ~5 KB of duplicate tables and two duplicate ray walks per .so.
 * When _features_ext.c includes this header alone, the FEAT_-prefixed
 * standalone versions below are compiled in. */
#ifdef _CBOARD_IMPL_H
#define FEAT_KNIGHT_ATTACKS  KNIGHT_ATTACKS
#define FEAT_KING_ATTACKS    KING_ATTACKS
#define FEAT_PAWN_ATTACKS    PAWN_ATTACKS
#define feat_rook_attacks    rook_attacks
#define feat_bishop_attacks  bishop_attacks
#else
static uint64_t FEAT_KNIGHT_ATTACKS[64];
static uint64_t FEAT_KING_ATTACKS[64];
static uint64_t FEAT_PAWN_ATTACKS[2][64];   /* [color][sq], color: 0=BLACK 1=WHITE */
#endif

static uint64_t FEAT_ADJACENT_FILE_MASKS[8];
static uint64_t FEAT_PASSED_PAWN_MASKS[2][64];
static uint64_t FEAT_CONNECTED_NEIGHBOR_MASKS[64];
static uint64_t FEAT_BACKWARD_SUPPORT_MASKS[2][64];
static uint64_t FEAT_PAWN_ATTACKERS_TO_SQ[2][64];
static uint64_t FEAT_PAWN_SINGLE_PUSH[2][64];
static uint64_t FEAT_PAWN_DOUBLE_PUSH[2][64];
static uint64_t FEAT_PAWN_CAPTURE_MASKS[2][64];

static uint64_t FEAT_BB_FILES[8];
static uint64_t FEAT_BB_RANKS[8];

static int feat_tables_initialized = 0;

static void init_tables_features(void) {
    if (feat_tables_initialized) return;

#ifdef _CBOARD_IMPL_H
    /* Shared attack tables are populated by cboard's init (idempotent). */
    init_attack_tables();
#endif

    for (int f = 0; f < 8; f++) {
        uint64_t mask = 0;
        for (int r = 0; r < 8; r++) mask |= ((uint64_t)1 << (r * 8 + f));
        FEAT_BB_FILES[f] = mask;
    }
    for (int r = 0; r < 8; r++) {
        uint64_t mask = 0;
        for (int f = 0; f < 8; f++) mask |= ((uint64_t)1 << (r * 8 + f));
        FEAT_BB_RANKS[r] = mask;
    }

    for (int f = 0; f < 8; f++) {
        uint64_t m = 0;
        if (f > 0) m |= FEAT_BB_FILES[f - 1];
        if (f < 7) m |= FEAT_BB_FILES[f + 1];
        FEAT_ADJACENT_FILE_MASKS[f] = m;
    }

#ifndef _CBOARD_IMPL_H
    /* Standalone mode: build KNIGHT/KING/PAWN attack tables ourselves.
     * In shared mode init_attack_tables() above already populated them. */
    static const int knight_deltas[8][2] = {
        {-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}
    };
    for (int sq = 0; sq < 64; sq++) {
        uint64_t a = 0;
        int f = (sq & 7), r = (sq >> 3);
        for (int i = 0; i < 8; i++) {
            int nf = f + knight_deltas[i][0], nr = r + knight_deltas[i][1];
            if (nf >= 0 && nf <= 7 && nr >= 0 && nr <= 7)
                a |= ((uint64_t)1 << (nr * 8 + nf));
        }
        FEAT_KNIGHT_ATTACKS[sq] = a;
    }

    static const int king_deltas[8][2] = {
        {-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}
    };
    for (int sq = 0; sq < 64; sq++) {
        uint64_t a = 0;
        int f = (sq & 7), r = (sq >> 3);
        for (int i = 0; i < 8; i++) {
            int nf = f + king_deltas[i][0], nr = r + king_deltas[i][1];
            if (nf >= 0 && nf <= 7 && nr >= 0 && nr <= 7)
                a |= ((uint64_t)1 << (nr * 8 + nf));
        }
        FEAT_KING_ATTACKS[sq] = a;
    }

    for (int sq = 0; sq < 64; sq++) {
        int f = (sq & 7), r = (sq >> 3);
        uint64_t w = 0, b = 0;
        if (r < 7) {
            if (f > 0) w |= ((uint64_t)1 << ((r + 1) * 8 + f - 1));
            if (f < 7) w |= ((uint64_t)1 << ((r + 1) * 8 + f + 1));
        }
        if (r > 0) {
            if (f > 0) b |= ((uint64_t)1 << ((r - 1) * 8 + f - 1));
            if (f < 7) b |= ((uint64_t)1 << ((r - 1) * 8 + f + 1));
        }
        FEAT_PAWN_ATTACKS[1][sq] = w;
        FEAT_PAWN_ATTACKS[0][sq] = b;
    }
#endif  /* !_CBOARD_IMPL_H */

    for (int sq = 0; sq < 64; sq++) {
        int f = (sq & 7), r = (sq >> 3);

        uint64_t conn = 0;
        for (int df = -1; df <= 1; df += 2) {
            int f2 = f + df;
            if (f2 < 0 || f2 > 7) continue;
            for (int dr = -1; dr <= 1; dr++) {
                int r2 = r + dr;
                if (r2 >= 0 && r2 <= 7)
                    conn |= ((uint64_t)1 << (r2 * 8 + f2));
            }
        }
        FEAT_CONNECTED_NEIGHBOR_MASKS[sq] = conn;

        for (int color = 0; color <= 1; color++) {
            int direction = color ? 1 : -1;

            uint64_t passed = 0;
            for (int ff = (f > 0 ? f - 1 : 0); ff <= (f < 7 ? f + 1 : 7); ff++) {
                int rr = r + direction;
                while (rr >= 0 && rr <= 7) {
                    passed |= ((uint64_t)1 << (rr * 8 + ff));
                    rr += direction;
                }
            }
            FEAT_PASSED_PAWN_MASKS[color][sq] = passed;

            uint64_t support = 0;
            for (int af = f - 1; af <= f + 1; af += 2) {
                if (af < 0 || af > 7) continue;
                if (color) {
                    for (int rr = r; rr <= 7; rr++)
                        support |= ((uint64_t)1 << (rr * 8 + af));
                } else {
                    for (int rr = 0; rr <= r; rr++)
                        support |= ((uint64_t)1 << (rr * 8 + af));
                }
            }
            FEAT_BACKWARD_SUPPORT_MASKS[color][sq] = support;

            uint64_t attackers = 0;
            int src_rank = r - direction;
            if (src_rank >= 0 && src_rank <= 7) {
                if (f > 0) attackers |= ((uint64_t)1 << (src_rank * 8 + f - 1));
                if (f < 7) attackers |= ((uint64_t)1 << (src_rank * 8 + f + 1));
            }
            FEAT_PAWN_ATTACKERS_TO_SQ[color][sq] = attackers;

            uint64_t single = 0, dbl = 0, captures = 0;
            int r1 = r + direction;
            if (r1 >= 0 && r1 <= 7) {
                single = ((uint64_t)1 << (r1 * 8 + f));
                int start_rank = color ? 1 : 6;
                int r2 = r + 2 * direction;
                if (r == start_rank && r2 >= 0 && r2 <= 7)
                    dbl = ((uint64_t)1 << (r2 * 8 + f));
                if (f > 0) captures |= ((uint64_t)1 << (r1 * 8 + f - 1));
                if (f < 7) captures |= ((uint64_t)1 << (r1 * 8 + f + 1));
            }
            FEAT_PAWN_SINGLE_PUSH[color][sq] = single;
            FEAT_PAWN_DOUBLE_PUSH[color][sq] = dbl;
            FEAT_PAWN_CAPTURE_MASKS[color][sq] = captures;
        }
    }

    feat_tables_initialized = 1;
}

/* ================================================================
 * Sliding piece attacks (standalone copy; shared mode aliases to
 * _cboard_impl.h's rook_attacks/bishop_attacks via #define above).
 * ================================================================ */

#ifndef _CBOARD_IMPL_H
static uint64_t feat_rook_attacks(int sq, uint64_t occ) {
    uint64_t attacks = 0;
    int f = (sq & 7), r = (sq >> 3);
    for (int rr = r + 1; rr <= 7; rr++) { int s = rr*8+f; attacks |= ((uint64_t)1<<s); if (occ & ((uint64_t)1<<s)) break; }
    for (int rr = r - 1; rr >= 0; rr--) { int s = rr*8+f; attacks |= ((uint64_t)1<<s); if (occ & ((uint64_t)1<<s)) break; }
    for (int ff = f + 1; ff <= 7; ff++) { int s = r*8+ff; attacks |= ((uint64_t)1<<s); if (occ & ((uint64_t)1<<s)) break; }
    for (int ff = f - 1; ff >= 0; ff--) { int s = r*8+ff; attacks |= ((uint64_t)1<<s); if (occ & ((uint64_t)1<<s)) break; }
    return attacks;
}

static uint64_t feat_bishop_attacks(int sq, uint64_t occ) {
    uint64_t attacks = 0;
    int f = (sq & 7), r = (sq >> 3);
    for (int d=1; f+d<=7&&r+d<=7; d++) { int s=(r+d)*8+f+d; attacks|=((uint64_t)1<<s); if(occ&((uint64_t)1<<s)) break; }
    for (int d=1; f+d<=7&&r-d>=0; d++) { int s=(r-d)*8+f+d; attacks|=((uint64_t)1<<s); if(occ&((uint64_t)1<<s)) break; }
    for (int d=1; f-d>=0&&r+d<=7; d++) { int s=(r+d)*8+f-d; attacks|=((uint64_t)1<<s); if(occ&((uint64_t)1<<s)) break; }
    for (int d=1; f-d>=0&&r-d>=0; d++) { int s=(r-d)*8+f-d; attacks|=((uint64_t)1<<s); if(occ&((uint64_t)1<<s)) break; }
    return attacks;
}
#endif  /* !_CBOARD_IMPL_H */

static uint64_t feat_piece_attacks(int sq, int piece_type, uint64_t occ) {
    switch (piece_type) {
        case 0: return 0;
        case 1: return FEAT_KNIGHT_ATTACKS[sq];
        case 2: return feat_bishop_attacks(sq, occ);
        case 3: return feat_rook_attacks(sq, occ);
        case 4: return feat_bishop_attacks(sq, occ) | feat_rook_attacks(sq, occ);
        case 5: return FEAT_KING_ATTACKS[sq];
        default: return 0;
    }
}

/* X-ray of one slider ray family (diagonal=bishop, else=rook): the squares
 * seen THROUGH the first blocker on each ray. direct ^ att(s, occ ^
 * (direct & occ)) — remove each ray's first blocker from occupancy, recompute
 * the attack, XOR with the direct attack. Disjoint from the direct attack. */
static uint64_t feat_slider_xray(int sq, uint64_t occ, int diagonal) {
    uint64_t direct = diagonal ? feat_bishop_attacks(sq, occ)
                               : feat_rook_attacks(sq, occ);
    uint64_t occ_xray = occ ^ (direct & occ);
    uint64_t through = diagonal ? feat_bishop_attacks(sq, occ_xray)
                                : feat_rook_attacks(sq, occ_xray);
    return direct ^ through;
}

/* ================================================================
 * Feature computation helpers
 * ================================================================ */

static void feat_bb_to_plane(float *plane, uint64_t bb, int turn_white) {
    /* Plane must be pre-zeroed. This only sets 1.0f bits. */
    if (bb == 0) return;
#if defined(__AVX2__)
    /* Use the AVX2 bitboard→plane path (writes all 64 floats, not just set bits).
     * This overwrites the pre-zeroed plane, which is fine — and avoids
     * branch-heavy scalar bit extraction. */
    if (turn_white) bitboard_to_plane_white(bb, plane);
    else            bitboard_to_plane_black(bb, plane);
#else
    int sq;
    for (uint64_t _bb = (bb); _bb; _bb &= _bb - 1) {
        sq = __builtin_ctzll(_bb);
        int f = (sq & 7), r = (sq >> 3);
        int row = turn_white ? r : (7 - r);
        plane[row * 8 + f] = 1.0f;
    }
#endif
}

static uint64_t feat_king_zone(int king_sq, int color) {
    if (king_sq < 0 || king_sq > 63) return 0;
    uint64_t zone = FEAT_KING_ATTACKS[king_sq] | ((uint64_t)1 << king_sq);
    int kf = (king_sq & 7), kr = (king_sq >> 3);
    int dr1 = color ? 1 : -1;
    int dr2 = color ? 2 : -2;
    for (int df = -1; df <= 1; df++) {
        int f = kf + df;
        if (f < 0 || f > 7) continue;
        int r1 = kr + dr1;
        if (r1 >= 0 && r1 <= 7) zone |= ((uint64_t)1 << (r1 * 8 + f));
        int r2 = kr + dr2;
        if (r2 >= 0 && r2 <= 7) zone |= ((uint64_t)1 << (r2 * 8 + f));
    }
    return zone;
}

static int feat_is_slider_aligned(int src, int dst, int piece_type) {
    int dx = (dst & 7) - (src & 7);
    int dy = (dst >> 3) - (src >> 3);
    if (piece_type == 2) return abs(dx) == abs(dy) && dx != 0;
    if (piece_type == 3) return (dx == 0) != (dy == 0);
    if (piece_type == 4) return ((dx == 0) != (dy == 0)) || (abs(dx) == abs(dy) && dx != 0);
    return 0;
}

static int feat_ray_step(int src, int dst) {
    int sf = (src & 7), sr = (src >> 3);
    int df = (dst & 7), dr = (dst >> 3);
    int dx = df - sf, dy = dr - sr;
    if (dx == 0 && dy != 0) return dy > 0 ? 8 : -8;
    if (dy == 0 && dx != 0) return dx > 0 ? 1 : -1;
    if (dx != 0 && abs(dx) == abs(dy)) {
        if (dx > 0 && dy > 0) return 9;
        if (dx > 0 && dy < 0) return -7;
        if (dx < 0 && dy > 0) return 7;
        return -9;
    }
    return 0;
}

static int feat_is_attacked_by(uint64_t pieces[6], int color_idx, uint64_t occ, int target_sq) {
    if (FEAT_KNIGHT_ATTACKS[target_sq] & pieces[1]) return 1;
    uint64_t diag = feat_bishop_attacks(target_sq, occ);
    if (diag & (pieces[2] | pieces[4])) return 1;
    uint64_t orth = feat_rook_attacks(target_sq, occ);
    if (orth & (pieces[3] | pieces[4])) return 1;
    if (FEAT_PAWN_ATTACKERS_TO_SQ[color_idx][target_sq] & pieces[0]) return 1;
    if (FEAT_KING_ATTACKS[target_sq] & pieces[5]) return 1;
    return 0;
}

static uint64_t feat_discovered_attack_mask(
    uint64_t own_pieces[6], uint64_t all_own, uint64_t occ,
    int opp_king_sq, int color
) {
    if (opp_king_sq < 0 || opp_king_sq > 63) return 0;

    /* Compute attacker masks once and reuse for both the "is king attacked"
     * test and the follow-up attacker enumeration — avoids the redundant
     * bishop/rook ray walks that the feat_is_attacked_by + recompute pattern
     * incurred on the in-check branch. */
    uint64_t sliders_dr = own_pieces[2] | own_pieces[4];  /* bishops + queens */
    uint64_t sliders_or = own_pieces[3] | own_pieces[4];  /* rooks + queens */

    uint64_t ka = FEAT_KNIGHT_ATTACKS[opp_king_sq] & own_pieces[1];
    uint64_t diag = feat_bishop_attacks(opp_king_sq, occ);
    uint64_t da = diag & sliders_dr;
    uint64_t orth = feat_rook_attacks(opp_king_sq, occ);
    uint64_t ra = orth & sliders_or;
    uint64_t pa = FEAT_PAWN_ATTACKERS_TO_SQ[color][opp_king_sq] & own_pieces[0];

    uint64_t attackers = ka | da | ra | pa;

    if (attackers) {
        int n_attackers = __builtin_popcountll(attackers);
        if (n_attackers >= 2) return all_own;

        int attacker_sq = __builtin_ctzll(attackers);
        uint64_t att_bit = (uint64_t)1 << attacker_sq;
        uint64_t occ_without = occ & ~att_bit;
        uint64_t diag2 = feat_bishop_attacks(opp_king_sq, occ_without);
        uint64_t orth2 = feat_rook_attacks(opp_king_sq, occ_without);
        if ((diag2 & sliders_dr & ~att_bit) || (orth2 & sliders_or & ~att_bit))
            return all_own;
        return all_own & ~attackers;
    }

    uint64_t has_sliders = own_pieces[2] | own_pieces[3] | own_pieces[4];
    if (!has_sliders) return 0;

    uint64_t discovered = 0;
    for (int pt = 2; pt <= 4; pt++) {
        int sq;
        for (uint64_t _bb = own_pieces[pt]; _bb; _bb &= _bb - 1) {
            sq = __builtin_ctzll(_bb);
            if (!feat_is_slider_aligned(sq, opp_king_sq, pt)) continue;
            int step = feat_ray_step(sq, opp_king_sq);
            if (!step) continue;
            int blocker_sq = -1, count = 0;
            int cur = sq + step;
            while (cur != opp_king_sq && cur >= 0 && cur < 64) {
                int cf = (cur & 7), pf = ((cur - step) & 7);
                if (abs(cf - pf) > 1) break;
                if (occ & ((uint64_t)1 << cur)) {
                    blocker_sq = cur; count++;
                    if (count > 1) break;
                }
                cur += step;
                if (cur < 0 || cur > 63) break;
            }
            if (count == 1 && blocker_sq >= 0 && (all_own & ((uint64_t)1 << blocker_sq)))
                discovered |= ((uint64_t)1 << blocker_sq);
        }
    }
    return discovered;
}

/* Sweep the four directions given by df[]/dr[] around king_sq looking for
 * pin patterns (king, own blocker, opposing slider along the ray). When
 * found, set the blocker in *pinned_out and extend the full through-ray
 * (both directions) into *pin_ray_out. Used once per slider family
 * (diagonal bishops/queens, orthogonal rooks/queens). */
static inline void feat_scan_pin_dirs(
    const int df[4], const int dr[4],
    uint64_t color_occ, uint64_t occ, int king_sq, uint64_t sliders,
    uint64_t *pinned_out, uint64_t *pin_ray_out
) {
    int kf = (king_sq & 7), kr = (king_sq >> 3);
    for (int d = 0; d < 4; d++) {
        int own_blocker = -1;
        uint64_t ray = 0;
        int found_pinner = 0;
        for (int dist = 1; dist <= 7; dist++) {
            int f = kf + df[d] * dist;
            int r = kr + dr[d] * dist;
            if (f < 0 || f > 7 || r < 0 || r > 7) break;
            int sq = r * 8 + f;
            ray |= ((uint64_t)1 << sq);
            if (found_pinner) continue;
            if (occ & ((uint64_t)1 << sq)) {
                if (color_occ & ((uint64_t)1 << sq)) {
                    if (own_blocker >= 0) break;
                    own_blocker = sq;
                } else {
                    if ((sliders & ((uint64_t)1 << sq)) && own_blocker >= 0) {
                        *pinned_out |= ((uint64_t)1 << own_blocker);
                        found_pinner = 1;
                    } else break;
                }
            }
        }
        if (found_pinner) {
            for (int dist = 1; dist <= 7; dist++) {
                int f = kf - df[d] * dist;
                int r = kr - dr[d] * dist;
                if (f < 0 || f > 7 || r < 0 || r > 7) break;
                ray |= ((uint64_t)1 << (r * 8 + f));
            }
            *pin_ray_out |= ray | ((uint64_t)1 << king_sq);
        }
    }
}

static void feat_compute_pins(
    uint64_t color_occ, uint64_t occ, int king_sq,
    uint64_t opp_bishops, uint64_t opp_rooks, uint64_t opp_queens,
    uint64_t *pinned_out, uint64_t *pin_ray_out
) {
    *pinned_out = 0;
    *pin_ray_out = 0;
    if (king_sq < 0 || king_sq > 63) return;

    static const int diag_df[4] = {1, 1, -1, -1};
    static const int diag_dr[4] = {1, -1, 1, -1};
    static const int orth_df[4] = {1, -1, 0, 0};
    static const int orth_dr[4] = {0, 0, 1, -1};

    feat_scan_pin_dirs(diag_df, diag_dr, color_occ, occ, king_sq,
                       opp_bishops | opp_queens, pinned_out, pin_ray_out);
    feat_scan_pin_dirs(orth_df, orth_dr, color_occ, occ, king_sq,
                       opp_rooks | opp_queens, pinned_out, pin_ray_out);
}

static uint64_t feat_passed_pawns(uint64_t own_pawns, uint64_t enemy_pawns, int color) {
    uint64_t passed = 0;
    int sq;
    for (uint64_t _bb = own_pawns; _bb; _bb &= _bb - 1) {
        sq = __builtin_ctzll(_bb);
        if (!(FEAT_PASSED_PAWN_MASKS[color][sq] & enemy_pawns))
            passed |= ((uint64_t)1 << sq);
    }
    return passed;
}

static uint64_t feat_isolated_pawns(uint64_t own_pawns) {
    uint64_t isolated = 0;
    int sq;
    for (uint64_t _bb = own_pawns; _bb; _bb &= _bb - 1) {
        sq = __builtin_ctzll(_bb);
        if (!(FEAT_ADJACENT_FILE_MASKS[sq & 7] & own_pawns))
            isolated |= ((uint64_t)1 << sq);
    }
    return isolated;
}

static uint64_t feat_backward_pawns(uint64_t own_pawns, uint64_t enemy_pawns, int color) {
    uint64_t backward = 0;
    int direction = color ? 1 : -1;
    int sq;
    for (uint64_t _bb = own_pawns; _bb; _bb &= _bb - 1) {
        sq = __builtin_ctzll(_bb);
        int f = (sq & 7), r = (sq >> 3);
        if (!(FEAT_ADJACENT_FILE_MASKS[f] & own_pawns)) continue;
        int r1 = r + direction;
        if (r1 < 0 || r1 > 7) continue;
        int front_sq = r1 * 8 + f;
        int opp_color = color ? 0 : 1;
        if (!(FEAT_PAWN_ATTACKERS_TO_SQ[opp_color][front_sq] & enemy_pawns)) continue;
        if (FEAT_BACKWARD_SUPPORT_MASKS[color][sq] & own_pawns) continue;
        backward |= ((uint64_t)1 << sq);
    }
    return backward;
}

static uint64_t feat_connected_pawns(uint64_t own_pawns) {
    uint64_t connected = 0;
    int sq;
    for (uint64_t _bb = own_pawns; _bb; _bb &= _bb - 1) {
        sq = __builtin_ctzll(_bb);
        if (FEAT_CONNECTED_NEIGHBOR_MASKS[sq] & own_pawns)
            connected |= ((uint64_t)1 << sq);
    }
    return connected;
}

/* ================================================================
 * v2 threat planes
 * ================================================================ */

#define FEAT_EXTRA_V1 34
#define FEAT_EXTRA_V2 63
#define FEAT_EXTRA_V3_CHECKS 67
#define FEAT_EXTRA_V3_XRAY 69
#define FEAT_EXTRA_V3_SEE 65
#define FEAT_EXTRA_V3_PASSERS 71

/* Attack maps + per-square attacker counts, harvested from the mobility
 * pass (index 0 = us, 1 = them; piece order P,N,B,R,Q,K).
 *
 * Counts are kept as carry-save bitboard counters (3 bits per square,
 * saturating at 7) so accumulating one attack map costs a handful of
 * bitwise ops instead of a per-set-bit loop. 7 is above the /4 clamp used
 * by the count planes, so those are exact. The control-margin plane can
 * underestimate when one side has >7 attackers on a square AND the other
 * has >=4 (a capped 7-vs-5 margin reads 0.5 instead of the true 1.0) —
 * an extreme pile-up we accept for the cheap accumulate. The Python
 * reference applies min(count, 7) for exact C parity either way. */
typedef struct {
    uint64_t by_type[2][6];
    uint64_t cnt[2][3];   /* per-square 3-bit saturating counter planes */
} FeatThreatAccum;

static inline void feat_accum_add(
    FeatThreatAccum *acc, int side, int pt, uint64_t attacks
) {
    acc->by_type[side][pt] |= attacks;
    uint64_t *c = acc->cnt[side];
    uint64_t carry0 = c[0] & attacks;
    c[0] ^= attacks;
    uint64_t carry1 = c[1] & carry0;
    c[1] ^= carry0;
    uint64_t overflow = c[2] & carry1;  /* 7 + 1 → clamp back to 7 */
    c[2] |= carry1;
    c[0] |= overflow;
    c[1] |= overflow;
}

static inline int feat_accum_count(const FeatThreatAccum *acc, int side, int sq) {
    return (int)((acc->cnt[side][0] >> sq) & 1)
         | ((int)((acc->cnt[side][1] >> sq) & 1) << 1)
         | ((int)((acc->cnt[side][2] >> sq) & 1) << 2);
}

/* Write a per-square float value plane with side-to-move orientation. */
static inline void feat_value_to_plane_sq(
    float *plane, int sq, float value, int turn_white
) {
    int f = (sq & 7), r = (sq >> 3);
    int row = turn_white ? r : (7 - r);
    plane[row * 8 + f] = value;
}

/* ================================================================
 * v3_see: graded Static Exchange Evaluation
 * ================================================================ */

/* SEE piece values in pawns, indexed by bitboard piece slot
 * (0=P,1=N,2=B,3=R,4=Q,5=K). King "large" so it is never voluntarily traded
 * into a losing recapture but can deliver the final capture. */
static const int FEAT_SEE_VALUE[6] = {1, 3, 3, 5, 9, 1000};

/* Square of ``side``'s least-valuable piece attacking ``target_sq`` at the
 * given occupancy, with the per-side piece bitboards in pieces[6]. Returns -1
 * if no attacker. Direct attackers only (no x-ray re-attacker discovery beyond
 * what removing pieces from ``occ`` between calls already reveals). */
static int feat_least_valuable_attacker(
    const uint64_t pieces[6], uint64_t occ, int color_idx, int target_sq
) {
    int best_sq = -1, best_val = 0;
    /* Pawns (cheapest), then knight, bishop, rook, queen, king. */
    uint64_t pawn_att = FEAT_PAWN_ATTACKERS_TO_SQ[color_idx][target_sq] & pieces[0];
    if (pawn_att) return __builtin_ctzll(pawn_att);

    uint64_t cand;
    cand = FEAT_KNIGHT_ATTACKS[target_sq] & pieces[1];
    if (cand) { best_sq = __builtin_ctzll(cand); best_val = FEAT_SEE_VALUE[1]; }

    uint64_t diag = feat_bishop_attacks(target_sq, occ);
    uint64_t orth = feat_rook_attacks(target_sq, occ);
    cand = diag & pieces[2];
    if (cand && (best_sq < 0 || FEAT_SEE_VALUE[2] < best_val)) {
        best_sq = __builtin_ctzll(cand); best_val = FEAT_SEE_VALUE[2];
    }
    cand = orth & pieces[3];
    if (cand && (best_sq < 0 || FEAT_SEE_VALUE[3] < best_val)) {
        best_sq = __builtin_ctzll(cand); best_val = FEAT_SEE_VALUE[3];
    }
    cand = (diag | orth) & pieces[4];
    if (cand && (best_sq < 0 || FEAT_SEE_VALUE[4] < best_val)) {
        best_sq = __builtin_ctzll(cand); best_val = FEAT_SEE_VALUE[4];
    }
    cand = FEAT_KING_ATTACKS[target_sq] & pieces[5];
    if (cand && (best_sq < 0 || FEAT_SEE_VALUE[5] < best_val)) {
        best_sq = __builtin_ctzll(cand);
    }
    return best_sq;
}

/* Find which piece-type slot occupies ``sq`` in pieces[6]; -1 if none. */
static int feat_piece_type_at(const uint64_t pieces[6], int sq) {
    uint64_t bit = (uint64_t)1 << sq;
    for (int pt = 0; pt < 6; pt++)
        if (pieces[pt] & bit) return pt;
    return -1;
}

/* Static exchange evaluation (pawns) for ``initiator`` capturing on ``square``.
 * Iterative least-valuable-attacker swap-off with negamax fold-back. Mutates
 * local copies of the piece bitboards / occupancy as pieces are consumed (so
 * sliders revealed once a closer same-line attacker is removed ARE counted),
 * but does not recompute x-ray batteries through a same-square removed enemy
 * attacker — see the Python twin's docstring. ``init_pieces``/``other_pieces``
 * are the initiator's and victim's per-side bitboards. */
static int feat_see_capture(
    uint64_t init_pieces[6], uint64_t other_pieces[6], uint64_t occ,
    int init_color_idx, int square
) {
    int victim_pt = feat_piece_type_at(other_pieces, square);
    if (victim_pt < 0) return 0;

    uint64_t side_pieces[2][6];
    for (int pt = 0; pt < 6; pt++) {
        side_pieces[0][pt] = init_pieces[pt];   /* side 0 = initiator */
        side_pieces[1][pt] = other_pieces[pt];  /* side 1 = victim's owner */
    }
    int side_color[2] = {init_color_idx, 1 - init_color_idx};

    int gain[40];
    int depth = 0;
    /* Classic swap-off (Chess Programming Wiki SEE). gain[0] = value of the
     * piece captured by the initiator; gain[d] = value[attacker_d] −
     * gain[d-1]. Remove the standing victim from occupancy so its owner's
     * sliders behind it can be revealed as the swap-off continues. */
    gain[0] = FEAT_SEE_VALUE[victim_pt];
    occ &= ~((uint64_t)1 << square);
    side_pieces[1][victim_pt] &= ~((uint64_t)1 << square);

    int side = 0;  /* initiator moves first */
    while (depth + 1 < 40) {
        int att_sq = feat_least_valuable_attacker(
            side_pieces[side], occ, side_color[side], square
        );
        if (att_sq < 0) break;
        int att_pt = feat_piece_type_at(side_pieces[side], att_sq);
        if (att_pt < 0) break;  /* defensive: should not happen */
        depth++;
        gain[depth] = FEAT_SEE_VALUE[att_pt] - gain[depth - 1];
        /* The attacker leaves att_sq (revealing rear sliders along its ray).
         * What stands on ``square`` does not affect attacker enumeration of
         * ``square``, so we only clear the attacker's origin. */
        side_pieces[side][att_pt] &= ~((uint64_t)1 << att_sq);
        occ &= ~((uint64_t)1 << att_sq);
        side = 1 - side;
    }

    /* Negamax fold-back (CPW: while (--d) gain[d-1] = -max(-gain[d-1],
     * gain[d])). With ``depth`` captures, fold i from depth-2 down to 0 — the
     * deepest capture stands; a lone capture (depth==1) is never folded, so an
     * undefended piece nets its full value. */
    for (int i = depth - 2; i >= 0; i--) {
        int a = -gain[i];
        int b = gain[i + 1];
        gain[i] = -(a > b ? a : b);
    }
    return gain[0];
}

/* Threat planes [34:63] of the extra block. ``out`` is the extra-block
 * base pointer (planes are written at out + 34*64 onward, pre-zeroed). */
static void compute_features_threats(
    uint64_t us_pieces[6], uint64_t them_pieces[6],
    uint64_t us_occ, uint64_t them_occ, uint64_t occupied,
    int king_sq_us, int king_sq_them,
    int turn_white, int n_extra, const FeatThreatAccum *acc,
    float * restrict out
) {
    uint64_t all_us = 0, all_them = 0;
    for (int pt = 0; pt < 6; pt++) {
        all_us |= acc->by_type[0][pt];
        all_them |= acc->by_type[1][pt];
    }

    int plane_idx = 34;

    /* [34:36] union attack maps, [36:48] per-type attack maps */
    feat_bb_to_plane(&out[plane_idx++ * 64], all_us, turn_white);
    feat_bb_to_plane(&out[plane_idx++ * 64], all_them, turn_white);
    for (int side = 0; side < 2; side++)
        for (int pt = 0; pt < 6; pt++)
            feat_bb_to_plane(&out[plane_idx++ * 64], acc->by_type[side][pt], turn_white);

    /* [48:50] attacker counts (/4, clamped to 1) */
    for (int side = 0; side < 2; side++) {
        float *plane = &out[plane_idx++ * 64];
        for (int sq = 0; sq < 64; sq++) {
            float v = (float)feat_accum_count(acc, side, sq) / 4.0f;
            if (v > 1.0f) v = 1.0f;
            if (v != 0.0f)
                feat_value_to_plane_sq(plane, sq, v, turn_white);
        }
    }

    /* [50:52] hanging: attacked by the enemy, undefended by the owner */
    feat_bb_to_plane(&out[plane_idx++ * 64], us_occ & all_them & ~all_us, turn_white);
    feat_bb_to_plane(&out[plane_idx++ * 64], them_occ & all_us & ~all_them, turn_white);

    /* [52:54] attacked by a strictly cheaper piece:
     * pawn → N/B/R/Q, minor → R/Q, rook → Q */
    for (int side = 0; side < 2; side++) {
        uint64_t *victims = side == 0 ? us_pieces : them_pieces;
        const uint64_t *att = acc->by_type[1 - side];
        uint64_t cheap =
            ((victims[1] | victims[2] | victims[3] | victims[4]) & att[0])
            | ((victims[3] | victims[4]) & (att[1] | att[2]))
            | (victims[4] & att[3]);
        feat_bb_to_plane(&out[plane_idx++ * 64], cheap, turn_white);
    }

    /* [54:58] safe checks for the side to move (Stockfish semantics):
     * squares an N/B/R/Q could check the enemy king from, not attacked
     * by the enemy and not occupied by us. */
    uint64_t check_n = 0, check_b = 0, check_r = 0, check_q = 0;
    if (king_sq_them >= 0 && king_sq_them < 64) {
        check_n = FEAT_KNIGHT_ATTACKS[king_sq_them];
        check_b = feat_bishop_attacks(king_sq_them, occupied);
        check_r = feat_rook_attacks(king_sq_them, occupied);
        check_q = check_b | check_r;
    }
    uint64_t safe = ~all_them & ~us_occ;
    feat_bb_to_plane(&out[plane_idx++ * 64], check_n & safe, turn_white);
    feat_bb_to_plane(&out[plane_idx++ * 64], check_b & safe, turn_white);
    feat_bb_to_plane(&out[plane_idx++ * 64], check_r & safe, turn_white);
    feat_bb_to_plane(&out[plane_idx++ * 64], check_q & safe, turn_white);

    /* [58] control margin: (us − them attackers)/4, clamped to [-1, 1] */
    {
        float *plane = &out[plane_idx++ * 64];
        for (int sq = 0; sq < 64; sq++) {
            float v = ((float)feat_accum_count(acc, 0, sq)
                       - (float)feat_accum_count(acc, 1, sq)) / 4.0f;
            if (v > 1.0f) v = 1.0f;
            if (v < -1.0f) v = -1.0f;
            if (v != 0.0f)
                feat_value_to_plane_sq(plane, sq, v, turn_white);
        }
    }

    /* [59:61] pawn tension: pawns that can capture an enemy pawn (no EP) */
    int us_color = turn_white ? 1 : 0;
    int them_color = 1 - us_color;
    for (int side = 0; side < 2; side++) {
        int c = side == 0 ? us_color : them_color;
        uint64_t own_pawns = (side == 0 ? us_pieces : them_pieces)[0];
        uint64_t enemy_pawns = (side == 0 ? them_pieces : us_pieces)[0];
        uint64_t tension = 0;
        for (uint64_t m = own_pawns; m; m &= m - 1) {
            int sq = __builtin_ctzll(m);
            if (FEAT_PAWN_CAPTURE_MASKS[c][sq] & enemy_pawns)
                tension |= ((uint64_t)1 << sq);
        }
        feat_bb_to_plane(&out[plane_idx++ * 64], tension, turn_white);
    }

    /* [61:63] pawn storm: enemy pawns on the defender-king file ±1,
     * scaled by rank closeness to the king (1 − rank_dist/7). */
    for (int side = 0; side < 2; side++) {
        float *plane = &out[plane_idx++ * 64];
        int ksq = side == 0 ? king_sq_us : king_sq_them;
        if (ksq < 0 || ksq > 63) continue;
        uint64_t enemy_pawns = (side == 0 ? them_pieces : us_pieces)[0];
        int kf = (ksq & 7), kr = (ksq >> 3);
        for (uint64_t m = enemy_pawns; m; m &= m - 1) {
            int sq = __builtin_ctzll(m);
            int f = (sq & 7), r = (sq >> 3);
            if (abs(f - kf) > 1) continue;
            float v = 1.0f - (float)abs(r - kr) / 7.0f;
            if (v != 0.0f)
                feat_value_to_plane_sq(plane, sq, v, turn_white);
        }
    }

    /* v3 families: both append at [63:...] but are SELECTED by n_extra/version.
     * v3_checks (67) writes opponent-safe-checks [63:67]; v3_xray (69) writes
     * per-slider x-ray attacks [63:69]. Exactly one runs (or neither for v2). */
    if (n_extra == FEAT_EXTRA_V3_CHECKS) {
        /* [63:67] v3_checks: opponent's safe checks against OUR king (mirror of
         * [54:58]). Squares an N/B/R/Q of THEIRS could check our king from
         * (attacks projected from our king square at current occupancy), not
         * attacked by us and not occupied by them. */
        uint64_t ocheck_n = 0, ocheck_b = 0, ocheck_r = 0, ocheck_q = 0;
        if (king_sq_us >= 0 && king_sq_us < 64) {
            ocheck_n = FEAT_KNIGHT_ATTACKS[king_sq_us];
            ocheck_b = feat_bishop_attacks(king_sq_us, occupied);
            ocheck_r = feat_rook_attacks(king_sq_us, occupied);
            ocheck_q = ocheck_b | ocheck_r;
        }
        uint64_t safe_opp = ~all_us & ~them_occ;
        feat_bb_to_plane(&out[plane_idx++ * 64], ocheck_n & safe_opp, turn_white);
        feat_bb_to_plane(&out[plane_idx++ * 64], ocheck_b & safe_opp, turn_white);
        feat_bb_to_plane(&out[plane_idx++ * 64], ocheck_r & safe_opp, turn_white);
        feat_bb_to_plane(&out[plane_idx++ * 64], ocheck_q & safe_opp, turn_white);
    } else if (n_extra == FEAT_EXTRA_V3_XRAY) {
        /* [63:69] v3_xray: per-slider-type x-ray attacks (squares seen THROUGH
         * the first blocker on each ray). Order: our B, R, Q then their B, R, Q.
         * Queen x-ray = bishop-xray(queen sq) | rook-xray(queen sq). */
        for (int side = 0; side < 2; side++) {
            uint64_t *cp = side == 0 ? us_pieces : them_pieces;
            uint64_t xr_b = 0, xr_r = 0, xr_q = 0;
            for (uint64_t m = cp[2]; m; m &= m - 1)  /* bishops */
                xr_b |= feat_slider_xray(__builtin_ctzll(m), occupied, 1);
            for (uint64_t m = cp[3]; m; m &= m - 1)  /* rooks */
                xr_r |= feat_slider_xray(__builtin_ctzll(m), occupied, 0);
            for (uint64_t m = cp[4]; m; m &= m - 1) {  /* queens */
                int sq = __builtin_ctzll(m);
                xr_q |= feat_slider_xray(sq, occupied, 1)
                      | feat_slider_xray(sq, occupied, 0);
            }
            feat_bb_to_plane(&out[plane_idx++ * 64], xr_b, turn_white);
            feat_bb_to_plane(&out[plane_idx++ * 64], xr_r, turn_white);
            feat_bb_to_plane(&out[plane_idx++ * 64], xr_q, turn_white);
        }
    } else if (n_extra == FEAT_EXTRA_V3_SEE) {
        /* [63:65] v3_see: graded SEE, normalized clip(see/8, -1, 1).
         * [63] our pieces' SEE when the OPPONENT initiates a capture (negated
         *      to our perspective; <= 0 = we lose material).
         * [64] their pieces' SEE when WE initiate (>= 0 = we win material). */
        float *plane_us = &out[plane_idx++ * 64];
        float *plane_them = &out[plane_idx++ * 64];
        /* [63] iterate our occupied squares with an enemy attacker. */
        for (uint64_t m = us_occ; m; m &= m - 1) {
            int sq = __builtin_ctzll(m);
            if (!(feat_is_attacked_by(them_pieces, them_color, occupied, sq)))
                continue;
            int see = feat_see_capture(them_pieces, us_pieces, occupied,
                                       them_color, sq);
            float v = (float)(-see) / 8.0f;
            if (v > 1.0f) v = 1.0f;
            if (v < -1.0f) v = -1.0f;
            if (v != 0.0f)
                feat_value_to_plane_sq(plane_us, sq, v, turn_white);
        }
        /* [64] iterate their occupied squares with one of our attackers. */
        for (uint64_t m = them_occ; m; m &= m - 1) {
            int sq = __builtin_ctzll(m);
            if (!(feat_is_attacked_by(us_pieces, us_color, occupied, sq)))
                continue;
            int see = feat_see_capture(us_pieces, them_pieces, occupied,
                                       us_color, sq);
            float v = (float)see / 8.0f;
            if (v > 1.0f) v = 1.0f;
            if (v < -1.0f) v = -1.0f;
            if (v != 0.0f)
                feat_value_to_plane_sq(plane_them, sq, v, turn_white);
        }
    } else if (n_extra == FEAT_EXTRA_V3_PASSERS) {
        /* [63:71] v3_passers: passed-pawn quality, 4 planes per side
         * (us 63:67, them 67:71): rank-advancement, safe passer, blocked
         * passer, promotion-path-controlled-by-enemy. */
        for (int side = 0; side < 2; side++) {
            uint64_t *own = side == 0 ? us_pieces : them_pieces;
            uint64_t *enemy_p = side == 0 ? them_pieces : us_pieces;
            uint64_t own_pawns = own[0];
            uint64_t enemy_pawns = enemy_p[0];
            int c = side == 0 ? us_color : them_color;
            int enemy_color = 1 - c;
            int direction = c ? 1 : -1;
            uint64_t passers = feat_passed_pawns(own_pawns, enemy_pawns, c);

            float *plane_rank = &out[(plane_idx + 0) * 64];
            uint64_t safe_pass = 0, blocked = 0, promo_ctrl = 0;
            for (uint64_t m = passers; m; m &= m - 1) {
                int sq = __builtin_ctzll(m);
                int f = (sq & 7), r = (sq >> 3);
                int rel_rank = c ? r : 7 - r;
                feat_value_to_plane_sq(plane_rank, sq,
                                       (float)rel_rank / 7.0f, turn_white);

                int stop_r = r + direction;
                if (stop_r >= 0 && stop_r <= 7) {
                    int stop_sq = stop_r * 8 + f;
                    uint64_t stop_bit = (uint64_t)1 << stop_sq;
                    if (occupied & stop_bit) {
                        blocked |= (uint64_t)1 << sq;
                    } else if (!feat_is_attacked_by(enemy_p, enemy_color,
                                                    occupied, stop_sq)) {
                        safe_pass |= (uint64_t)1 << sq;
                    }
                }

                int rr = r + direction;
                while (rr >= 0 && rr <= 7) {
                    int ahead_sq = rr * 8 + f;
                    if (feat_is_attacked_by(enemy_p, enemy_color,
                                            occupied, ahead_sq)) {
                        promo_ctrl |= (uint64_t)1 << sq;
                        break;
                    }
                    rr += direction;
                }
            }
            feat_bb_to_plane(&out[(plane_idx + 1) * 64], safe_pass, turn_white);
            feat_bb_to_plane(&out[(plane_idx + 2) * 64], blocked, turn_white);
            feat_bb_to_plane(&out[(plane_idx + 3) * 64], promo_ctrl, turn_white);
            plane_idx += 4;
        }
    }
}

/* ================================================================
 * Main: compute all extra feature planes into pre-allocated buffer
 *
 * out must point to n_extra*64 floats, pre-zeroed.
 * n_extra is FEAT_EXTRA_V1 (34), FEAT_EXTRA_V2 (63),
 * FEAT_EXTRA_V3_CHECKS (67), or FEAT_EXTRA_V3_XRAY (69).
 * ================================================================ */

static void compute_features_ext(
    uint64_t us_pieces[6], uint64_t them_pieces[6],
    uint64_t occupied,
    int king_sq_us, int king_sq_them,
    int turn_white, int ep_square,
    int n_extra,
    float * restrict out
) {
    init_tables_features();

    uint64_t us_occ = 0, them_occ = 0;
    for (int i = 0; i < 6; i++) {
        us_occ |= us_pieces[i];
        them_occ |= them_pieces[i];
    }

    int us_color = turn_white ? 1 : 0;
    int them_color = turn_white ? 0 : 1;

    int plane_idx = 0;
    int colors[2] = {us_color, them_color};
    int king_sqs[2] = {king_sq_us, king_sq_them};
    uint64_t *color_pieces[2] = {us_pieces, them_pieces};

    /* ---- King safety: 10 planes [0:10] ---- */
    for (int ci = 0; ci < 2; ci++) {
        uint64_t kz = feat_king_zone(king_sqs[ci], colors[ci]);
        feat_bb_to_plane(&out[plane_idx * 64], kz, turn_white);
        plane_idx++;

        int opp_ci = 1 - ci;
        uint64_t *opp_p = color_pieces[opp_ci];
        for (int pt = 1; pt <= 4; pt++) {
            uint64_t overlap = 0;
            int sq;
            for (uint64_t _bb = opp_p[pt]; _bb; _bb &= _bb - 1) {
                sq = __builtin_ctzll(_bb);
                overlap |= feat_piece_attacks(sq, pt, occupied) & kz;
            }
            feat_bb_to_plane(&out[plane_idx * 64], overlap, turn_white);
            plane_idx++;
        }
    }

    /* ---- Pins/x-rays/discovered: 6 planes [10:16] ---- */
    for (int ci = 0; ci < 2; ci++) {
        uint64_t co = ci == 0 ? us_occ : them_occ;
        int ksq = king_sqs[ci];
        int opp_ci = 1 - ci;
        uint64_t *opp_p = color_pieces[opp_ci];

        uint64_t pinned, pin_ray;
        feat_compute_pins(co, occupied, ksq,
                          opp_p[2], opp_p[3], opp_p[4],
                          &pinned, &pin_ray);

        uint64_t disc = feat_discovered_attack_mask(
            color_pieces[ci],
            ci == 0 ? us_occ : them_occ,
            occupied, king_sqs[opp_ci], colors[ci]
        );

        feat_bb_to_plane(&out[plane_idx * 64], pinned, turn_white);
        plane_idx++;
        feat_bb_to_plane(&out[plane_idx * 64], pin_ray, turn_white);
        plane_idx++;
        feat_bb_to_plane(&out[plane_idx * 64], disc, turn_white);
        plane_idx++;
    }

    /* ---- Pawn structure: 8 planes [16:24] ---- */
    for (int ci = 0; ci < 2; ci++) {
        uint64_t own_p = color_pieces[ci][0];
        uint64_t enemy_p = color_pieces[1 - ci][0];
        int c = colors[ci];

        feat_bb_to_plane(&out[plane_idx * 64], feat_passed_pawns(own_p, enemy_p, c), turn_white);
        plane_idx++;
        feat_bb_to_plane(&out[plane_idx * 64], feat_isolated_pawns(own_p), turn_white);
        plane_idx++;
        feat_bb_to_plane(&out[plane_idx * 64], feat_backward_pawns(own_p, enemy_p, c), turn_white);
        plane_idx++;
        feat_bb_to_plane(&out[plane_idx * 64], feat_connected_pawns(own_p), turn_white);
        plane_idx++;
    }

    /* ---- Mobility: 6 planes [24:30] ---- */
    static const float MOB_MAX[6] = {8.0f, 13.0f, 14.0f, 27.0f, 8.0f, 4.0f};
    static const int MOB_PT[6] = {1, 2, 3, 4, 5, 0};

    uint64_t ep_mask = (ep_square >= 0 && ep_square < 64) ? ((uint64_t)1 << ep_square) : 0;

    /* v2_threats: harvest the per-piece attack masks this pass already
     * computes instead of recomputing them for the threat planes. */
    FeatThreatAccum acc;
    int want_threats = (n_extra >= FEAT_EXTRA_V2);
    if (want_threats) memset(&acc, 0, sizeof(acc));

    for (int mi = 0; mi < 6; mi++) {
        float *plane = &out[plane_idx * 64];
        int pt = MOB_PT[mi];
        float max_m = MOB_MAX[mi];

        for (int c = 1; c >= 0; c--) {
            uint64_t *cp = (c == us_color) ? us_pieces : them_pieces;
            uint64_t own_occ = (c == us_color) ? us_occ : them_occ;
            uint64_t opp_occ = (c == us_color) ? them_occ : us_occ;
            int si = (c == us_color) ? 0 : 1;
            int sq;

            for (uint64_t _bb = cp[pt]; _bb; _bb &= _bb - 1) {
                sq = __builtin_ctzll(_bb);
                int mobility;
                if (pt == 0) {
                    mobility = 0;
                    uint64_t single = FEAT_PAWN_SINGLE_PUSH[c][sq];
                    if (single && !(occupied & single)) {
                        mobility++;
                        uint64_t dbl = FEAT_PAWN_DOUBLE_PUSH[c][sq];
                        if (dbl && !(occupied & dbl))
                            mobility++;
                    }
                    uint64_t cap = FEAT_PAWN_CAPTURE_MASKS[c][sq];
                    mobility += __builtin_popcountll(cap & opp_occ);
                    if (ep_mask && (cap & ep_mask))
                        mobility++;
                    if (want_threats)
                        feat_accum_add(&acc, si, 0, cap);
                } else {
                    uint64_t att = feat_piece_attacks(sq, pt, occupied);
                    mobility = __builtin_popcountll(att & ~own_occ);
                    if (want_threats)
                        feat_accum_add(&acc, si, pt, att);
                }

                int f = (sq & 7), r = (sq >> 3);
                int row = turn_white ? r : (7 - r);
                plane[row * 8 + f] = (float)mobility / max_m;
            }
        }
        plane_idx++;
    }

    /* ---- Outpost/space: 4 planes [30:34] ---- */
    for (int ci = 0; ci < 2; ci++) {
        uint64_t own_pawns = color_pieces[ci][0];
        uint64_t enemy_pawns = color_pieces[1 - ci][0];
        int c = colors[ci];
        int opp_c = colors[1 - ci];

        uint64_t own_att = 0, enemy_att = 0;
        int sq;
        for (uint64_t _bb = own_pawns; _bb; _bb &= _bb - 1) {
            sq = __builtin_ctzll(_bb);
            own_att |= FEAT_PAWN_ATTACKS[c][sq];
        }
        for (uint64_t _bb = enemy_pawns; _bb; _bb &= _bb - 1) {
            sq = __builtin_ctzll(_bb);
            enemy_att |= FEAT_PAWN_ATTACKS[opp_c][sq];
        }
        feat_bb_to_plane(&out[plane_idx * 64], own_att & ~enemy_att, turn_white);
        plane_idx++;

        int direction = c ? -1 : 1;
        uint64_t space = 0;
        for (uint64_t _bb = own_pawns; _bb; _bb &= _bb - 1) {
            sq = __builtin_ctzll(_bb);
            int f = (sq & 7), r = (sq >> 3);
            if (f < 2 || f > 5) continue;
            for (int dd = 1; dd <= 2; dd++) {
                int r2 = r + direction * dd;
                if (r2 >= 0 && r2 <= 7)
                    space |= ((uint64_t)1 << (r2 * 8 + f));
            }
        }
        feat_bb_to_plane(&out[plane_idx * 64], space, turn_white);
        plane_idx++;
    }

    /* ---- v2 threats: 29 planes [34:63] (+ v3 opp safe checks [63:67]) ---- */
    if (want_threats)
        compute_features_threats(us_pieces, them_pieces,
                                 us_occ, them_occ, occupied,
                                 king_sq_us, king_sq_them,
                                 turn_white, n_extra, &acc, out);
}

/* CBoard → extra feature planes. Only available when _cboard_impl.h was also
 * included beforehand (i.e., in _lc0_ext.c / _mcts_tree.c, not _features_ext.c). */
#ifdef _CBOARD_IMPL_H
static inline void cboard_compute_features_ext(
    const CBoard *b, float * restrict out, int n_extra
) {
    int us = b->turn, them = 1 - us;
    uint64_t us_pieces[6], them_pieces[6];
    for (int i = 0; i < 6; i++) {
        us_pieces[i] = b->bb[i] & b->occ[us];
        them_pieces[i] = b->bb[i] & b->occ[them];
    }
    uint64_t us_king = b->bb[KING] & b->occ[us];
    uint64_t them_king = b->bb[KING] & b->occ[them];
    int king_sq_us = us_king ? lsb64(us_king) : -1;
    int king_sq_them = them_king ? lsb64(them_king) : -1;
    uint64_t occupied = b->occ[0] | b->occ[1];
    int turn_white = (b->turn == WHITE_C) ? 1 : 0;
    compute_features_ext(us_pieces, them_pieces, occupied,
                         king_sq_us, king_sq_them, turn_white,
                         (int)b->ep_square, n_extra, out);
}

static inline void cboard_compute_features_34(const CBoard *b, float * restrict out) {
    cboard_compute_features_ext(b, out, FEAT_EXTRA_V1);
}
#endif  /* _CBOARD_IMPL_H */


/* ================================================================
 * Dynamic board relations (attention-bias input)
 *
 * FEAT_RELATION_COUNT binary 64x64 matrices, side-to-move oriented
 * exactly like the feature planes (rank flip when black to move),
 * row = from-square, col = to-square:
 *
 *   R0 attacks(from, to)          piece on `from` attacks square `to`
 *                                 (pawns: capture squares only, no EP)
 *   R1 defends(from, to)          R0 restricted to `to` occupied by a
 *                                 same-color piece
 *   R2 pinned_by(from, to)        piece on `from` is absolutely pinned to
 *                                 its own king by the enemy slider on `to`
 *   R3 shares_open_line(from, to) both squares occupied, aligned on a file
 *                                 or diagonal, all squares strictly between
 *                                 empty (symmetric; ranks intentionally
 *                                 excluded)
 *   R4 pawn_tension(from, to)     pawn on `from` can capture the enemy
 *                                 pawn on `to`
 *
 * The Python twin is features.relation_matrices(); keep both in sync
 * (tests/test_dynamic_relations.py enforces value parity).
 * ================================================================ */

#define FEAT_RELATION_COUNT 5

static inline void feat_rel_mark(
    uint8_t * restrict out, int k, int from_o, int to_o
) {
    out[(size_t)k * 4096 + (size_t)from_o * 64 + to_o] = 1;
}

/* out must point to FEAT_RELATION_COUNT*64*64 uint8; zeroed here. */
static void compute_relations(
    uint64_t us_pieces[6], uint64_t them_pieces[6],
    uint64_t occupied,
    int king_sq_us, int king_sq_them,
    int turn_white,
    uint8_t * restrict out
) {
    init_tables_features();
    memset(out, 0, (size_t)FEAT_RELATION_COUNT * 4096);

    uint64_t us_occ = 0, them_occ = 0;
    for (int i = 0; i < 6; i++) {
        us_occ |= us_pieces[i];
        them_occ |= them_pieces[i];
    }
    int us_color = turn_white ? 1 : 0;
    int flip = turn_white ? 0 : 56;  /* sq ^ 56 flips ranks */

    /* R0 attacks + R1 defends */
    for (int side = 0; side < 2; side++) {
        uint64_t *pieces = side == 0 ? us_pieces : them_pieces;
        uint64_t own_occ = side == 0 ? us_occ : them_occ;
        int c = side == 0 ? us_color : 1 - us_color;
        for (int pt = 0; pt < 6; pt++) {
            for (uint64_t m = pieces[pt]; m; m &= m - 1) {
                int sq = __builtin_ctzll(m);
                uint64_t att = (pt == 0)
                    ? FEAT_PAWN_CAPTURE_MASKS[c][sq]
                    : feat_piece_attacks(sq, pt, occupied);
                int from_o = sq ^ flip;
                for (uint64_t a = att; a; a &= a - 1) {
                    int to = __builtin_ctzll(a);
                    feat_rel_mark(out, 0, from_o, to ^ flip);
                }
                for (uint64_t a = att & own_occ; a; a &= a - 1) {
                    int to = __builtin_ctzll(a);
                    feat_rel_mark(out, 1, from_o, to ^ flip);
                }
            }
        }
    }

    /* R2 pinned_by: walk the 8 rays from each king; pattern is
     * king .. own-blocker .. matching enemy slider with nothing between. */
    static const int rel_df[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    static const int rel_dr[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    for (int side = 0; side < 2; side++) {
        int ksq = side == 0 ? king_sq_us : king_sq_them;
        if (ksq < 0 || ksq > 63) continue;
        uint64_t own_occ = side == 0 ? us_occ : them_occ;
        uint64_t *opp = side == 0 ? them_pieces : us_pieces;
        int kf = ksq & 7, kr = ksq >> 3;
        for (int d = 0; d < 8; d++) {
            uint64_t sliders = d < 4 ? (opp[3] | opp[4]) : (opp[2] | opp[4]);
            if (!sliders) continue;
            int blocker = -1;
            for (int dist = 1; dist <= 7; dist++) {
                int f = kf + rel_df[d] * dist;
                int r = kr + rel_dr[d] * dist;
                if (f < 0 || f > 7 || r < 0 || r > 7) break;
                int sq = r * 8 + f;
                uint64_t bit = (uint64_t)1 << sq;
                if (!(occupied & bit)) continue;
                if (own_occ & bit) {
                    if (blocker >= 0) break;  /* two own pieces: no pin */
                    blocker = sq;
                } else {
                    if (blocker >= 0 && (sliders & bit))
                        feat_rel_mark(out, 2, blocker ^ flip, sq ^ flip);
                    break;
                }
            }
        }
    }

    /* R3 shares_open_line: files + diagonals only, first occupied square in
     * each direction. Marking one direction per pair covers both because
     * every occupied square is iterated. */
    static const int line_df[6] = {0, 0, 1, 1, -1, -1};
    static const int line_dr[6] = {1, -1, 1, -1, 1, -1};
    for (uint64_t m = occupied; m; m &= m - 1) {
        int sq = __builtin_ctzll(m);
        int f0 = sq & 7, r0 = sq >> 3;
        int from_o = sq ^ flip;
        for (int d = 0; d < 6; d++) {
            for (int dist = 1; dist <= 7; dist++) {
                int f = f0 + line_df[d] * dist;
                int r = r0 + line_dr[d] * dist;
                if (f < 0 || f > 7 || r < 0 || r > 7) break;
                int to = r * 8 + f;
                if (occupied & ((uint64_t)1 << to)) {
                    feat_rel_mark(out, 3, from_o, to ^ flip);
                    break;
                }
            }
        }
    }

    /* R4 pawn_tension */
    for (int side = 0; side < 2; side++) {
        uint64_t own_pawns = (side == 0 ? us_pieces : them_pieces)[0];
        uint64_t enemy_pawns = (side == 0 ? them_pieces : us_pieces)[0];
        int c = side == 0 ? us_color : 1 - us_color;
        for (uint64_t m = own_pawns; m; m &= m - 1) {
            int sq = __builtin_ctzll(m);
            uint64_t caps = FEAT_PAWN_CAPTURE_MASKS[c][sq] & enemy_pawns;
            int from_o = sq ^ flip;
            for (uint64_t a = caps; a; a &= a - 1) {
                int to = __builtin_ctzll(a);
                feat_rel_mark(out, 4, from_o, to ^ flip);
            }
        }
    }
}

#ifdef _CBOARD_IMPL_H
static inline void cboard_compute_relations(const CBoard *b, uint8_t * restrict out) {
    int us = b->turn, them = 1 - us;
    uint64_t us_pieces[6], them_pieces[6];
    for (int i = 0; i < 6; i++) {
        us_pieces[i] = b->bb[i] & b->occ[us];
        them_pieces[i] = b->bb[i] & b->occ[them];
    }
    uint64_t us_king = b->bb[KING] & b->occ[us];
    uint64_t them_king = b->bb[KING] & b->occ[them];
    compute_relations(us_pieces, them_pieces, b->occ[0] | b->occ[1],
                      us_king ? lsb64(us_king) : -1,
                      them_king ? lsb64(them_king) : -1,
                      (b->turn == WHITE_C) ? 1 : 0, out);
}
#endif  /* _CBOARD_IMPL_H */

#endif /* FEATURES_IMPL_H */
