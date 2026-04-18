/*
 * _features_impl.h — Pure C implementation of 34 extra feature planes.
 *
 * Shared between _features_ext.c (Python binding) and _lc0_ext.c (CBoard
 * fused encode_146 path).  All functions are static to avoid ODR issues.
 *
 * Usage: #include this header AFTER defining the standard bitboard macros
 * (popcount64, lsb64, sq_bit, sq_file, sq_rank, make_sq, FOR_EACH_BIT)
 * and after calling import_array() / init_tables_features().
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

static void feat_compute_pins(
    uint64_t color_occ, uint64_t occ, int king_sq,
    uint64_t opp_bishops, uint64_t opp_rooks, uint64_t opp_queens,
    uint64_t *pinned_out, uint64_t *pin_ray_out
) {
    *pinned_out = 0;
    *pin_ray_out = 0;
    if (king_sq < 0 || king_sq > 63) return;

    uint64_t diag_sliders = opp_bishops | opp_queens;
    uint64_t orth_sliders = opp_rooks | opp_queens;

    static const int diag_df[4] = {1, 1, -1, -1};
    static const int diag_dr[4] = {1, -1, 1, -1};
    static const int orth_df[4] = {1, -1, 0, 0};
    static const int orth_dr[4] = {0, 0, 1, -1};

    for (int d = 0; d < 4; d++) {
        int kf = (king_sq & 7), kr = (king_sq >> 3);
        int own_blocker = -1;
        uint64_t ray = 0;
        int found_pinner = 0;
        for (int dist = 1; dist <= 7; dist++) {
            int f = kf + diag_df[d] * dist;
            int r = kr + diag_dr[d] * dist;
            if (f < 0 || f > 7 || r < 0 || r > 7) break;
            int sq = r * 8 + f;
            ray |= ((uint64_t)1 << sq);
            if (found_pinner) continue;
            if (occ & ((uint64_t)1 << sq)) {
                if (color_occ & ((uint64_t)1 << sq)) {
                    if (own_blocker >= 0) break;
                    own_blocker = sq;
                } else {
                    if ((diag_sliders & ((uint64_t)1 << sq)) && own_blocker >= 0) {
                        *pinned_out |= ((uint64_t)1 << own_blocker);
                        found_pinner = 1;
                    } else break;
                }
            }
        }
        if (found_pinner) {
            for (int dist = 1; dist <= 7; dist++) {
                int f = kf - diag_df[d] * dist;
                int r = kr - diag_dr[d] * dist;
                if (f < 0 || f > 7 || r < 0 || r > 7) break;
                ray |= ((uint64_t)1 << (r * 8 + f));
            }
            *pin_ray_out |= ray | ((uint64_t)1 << king_sq);
        }
    }

    for (int d = 0; d < 4; d++) {
        int kf = (king_sq & 7), kr = (king_sq >> 3);
        int own_blocker = -1;
        uint64_t ray = 0;
        int found_pinner = 0;
        for (int dist = 1; dist <= 7; dist++) {
            int f = kf + orth_df[d] * dist;
            int r = kr + orth_dr[d] * dist;
            if (f < 0 || f > 7 || r < 0 || r > 7) break;
            int sq = r * 8 + f;
            ray |= ((uint64_t)1 << sq);
            if (found_pinner) continue;
            if (occ & ((uint64_t)1 << sq)) {
                if (color_occ & ((uint64_t)1 << sq)) {
                    if (own_blocker >= 0) break;
                    own_blocker = sq;
                } else {
                    if ((orth_sliders & ((uint64_t)1 << sq)) && own_blocker >= 0) {
                        *pinned_out |= ((uint64_t)1 << own_blocker);
                        found_pinner = 1;
                    } else break;
                }
            }
        }
        if (found_pinner) {
            for (int dist = 1; dist <= 7; dist++) {
                int f = kf - orth_df[d] * dist;
                int r = kr - orth_dr[d] * dist;
                if (f < 0 || f > 7 || r < 0 || r > 7) break;
                ray |= ((uint64_t)1 << (r * 8 + f));
            }
            *pin_ray_out |= ray | ((uint64_t)1 << king_sq);
        }
    }
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
 * Main: compute all 34 feature planes into pre-allocated buffer
 *
 * out must point to 34*64 floats, pre-zeroed.
 * ================================================================ */

static void compute_features_34(
    uint64_t us_pieces[6], uint64_t them_pieces[6],
    uint64_t occupied,
    int king_sq_us, int king_sq_them,
    int turn_white, int ep_square,
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

    for (int mi = 0; mi < 6; mi++) {
        float *plane = &out[plane_idx * 64];
        int pt = MOB_PT[mi];
        float max_m = MOB_MAX[mi];

        for (int c = 1; c >= 0; c--) {
            uint64_t *cp = (c == us_color) ? us_pieces : them_pieces;
            uint64_t own_occ = (c == us_color) ? us_occ : them_occ;
            uint64_t opp_occ = (c == us_color) ? them_occ : us_occ;
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
                } else {
                    uint64_t att = feat_piece_attacks(sq, pt, occupied);
                    mobility = __builtin_popcountll(att & ~own_occ);
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
}

/* CBoard → 34 feature planes. Only available when _cboard_impl.h was also
 * included beforehand (i.e., in _lc0_ext.c / _mcts_tree.c, not _features_ext.c). */
#ifdef _CBOARD_IMPL_H
static inline void cboard_compute_features_34(const CBoard *b, float * restrict out) {
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
    compute_features_34(us_pieces, them_pieces, occupied,
                        king_sq_us, king_sq_them, turn_white,
                        (int)b->ep_square, out);
}
#endif  /* _CBOARD_IMPL_H */

#endif /* FEATURES_IMPL_H */
