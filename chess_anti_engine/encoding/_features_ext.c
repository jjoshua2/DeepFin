/*
 * _features_ext.c — C-accelerated computation of 34 extra feature planes.
 *
 * Replaces the pure-Python extra_feature_planes() / extra_feature_planes_fast()
 * with native bitboard operations. Eliminates per-piece Python loop overhead
 * and per-plane numpy allocation.
 *
 * Layout (34 planes, 8x8 float32):
 *   [0:10]  king safety
 *   [10:16] pins / x-rays / discovered attacks
 *   [16:24] pawn structure
 *   [24:30] mobility
 *   [30:34] outpost / space
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdint.h>
#include <string.h>

/* ================================================================
 * Bitboard utilities
 * ================================================================ */

static inline int popcount64(uint64_t x) { return __builtin_popcountll(x); }
static inline int lsb64(uint64_t x) { return __builtin_ctzll(x); }
static inline uint64_t sq_bit(int sq) { return 1ULL << sq; }
static inline int sq_file(int sq) { return sq & 7; }
static inline int sq_rank(int sq) { return sq >> 3; }
static inline int make_sq(int f, int r) { return r * 8 + f; }

#define FOR_EACH_BIT(bb, sq) \
    for (uint64_t _bb = (bb); _bb; _bb &= _bb - 1) \
        if (((sq) = lsb64(_bb)), 1)

/* ================================================================
 * Precomputed tables (initialized once)
 * ================================================================ */

static uint64_t KNIGHT_ATTACKS[64];
static uint64_t KING_ATTACKS[64];
static uint64_t PAWN_ATTACKS[2][64];   /* [color][sq], color: 0=BLACK 1=WHITE */

/* Pawn structure masks */
static uint64_t ADJACENT_FILE_MASKS[8];
static uint64_t PASSED_PAWN_MASKS[2][64];
static uint64_t CONNECTED_NEIGHBOR_MASKS[64];
static uint64_t BACKWARD_SUPPORT_MASKS[2][64];
static uint64_t PAWN_ATTACKERS_TO_SQ[2][64];
static uint64_t PAWN_SINGLE_PUSH[2][64];
static uint64_t PAWN_DOUBLE_PUSH[2][64];
static uint64_t PAWN_CAPTURE_MASKS[2][64];

static uint64_t BB_FILES[8];
static uint64_t BB_RANKS[8];

static int tables_initialized = 0;

static void init_tables(void) {
    if (tables_initialized) return;

    /* Files and ranks */
    for (int f = 0; f < 8; f++) {
        uint64_t mask = 0;
        for (int r = 0; r < 8; r++) mask |= sq_bit(make_sq(f, r));
        BB_FILES[f] = mask;
    }
    for (int r = 0; r < 8; r++) {
        uint64_t mask = 0;
        for (int f = 0; f < 8; f++) mask |= sq_bit(make_sq(f, r));
        BB_RANKS[r] = mask;
    }

    /* Adjacent file masks */
    for (int f = 0; f < 8; f++) {
        uint64_t m = 0;
        if (f > 0) m |= BB_FILES[f - 1];
        if (f < 7) m |= BB_FILES[f + 1];
        ADJACENT_FILE_MASKS[f] = m;
    }

    /* Knight attacks */
    static const int knight_deltas[8][2] = {
        {-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}
    };
    for (int sq = 0; sq < 64; sq++) {
        uint64_t a = 0;
        int f = sq_file(sq), r = sq_rank(sq);
        for (int i = 0; i < 8; i++) {
            int nf = f + knight_deltas[i][0], nr = r + knight_deltas[i][1];
            if (nf >= 0 && nf <= 7 && nr >= 0 && nr <= 7)
                a |= sq_bit(make_sq(nf, nr));
        }
        KNIGHT_ATTACKS[sq] = a;
    }

    /* King attacks */
    static const int king_deltas[8][2] = {
        {-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}
    };
    for (int sq = 0; sq < 64; sq++) {
        uint64_t a = 0;
        int f = sq_file(sq), r = sq_rank(sq);
        for (int i = 0; i < 8; i++) {
            int nf = f + king_deltas[i][0], nr = r + king_deltas[i][1];
            if (nf >= 0 && nf <= 7 && nr >= 0 && nr <= 7)
                a |= sq_bit(make_sq(nf, nr));
        }
        KING_ATTACKS[sq] = a;
    }

    /* Pawn attacks: WHITE=1 attacks up-diagonals, BLACK=0 attacks down-diagonals */
    for (int sq = 0; sq < 64; sq++) {
        int f = sq_file(sq), r = sq_rank(sq);
        uint64_t w = 0, b = 0;
        if (r < 7) {
            if (f > 0) w |= sq_bit(make_sq(f - 1, r + 1));
            if (f < 7) w |= sq_bit(make_sq(f + 1, r + 1));
        }
        if (r > 0) {
            if (f > 0) b |= sq_bit(make_sq(f - 1, r - 1));
            if (f < 7) b |= sq_bit(make_sq(f + 1, r - 1));
        }
        PAWN_ATTACKS[1][sq] = w;  /* WHITE */
        PAWN_ATTACKS[0][sq] = b;  /* BLACK */
    }

    /* Pawn structure tables */
    for (int sq = 0; sq < 64; sq++) {
        int f = sq_file(sq), r = sq_rank(sq);

        /* Connected neighbor mask */
        uint64_t conn = 0;
        for (int df = -1; df <= 1; df += 2) {
            int f2 = f + df;
            if (f2 < 0 || f2 > 7) continue;
            for (int dr = -1; dr <= 1; dr++) {
                int r2 = r + dr;
                if (r2 >= 0 && r2 <= 7)
                    conn |= sq_bit(make_sq(f2, r2));
            }
        }
        CONNECTED_NEIGHBOR_MASKS[sq] = conn;

        for (int color = 0; color <= 1; color++) {
            int direction = color ? 1 : -1;  /* WHITE=1 goes up, BLACK=0 goes down */

            /* Passed pawn mask: all squares ahead on same + adjacent files */
            uint64_t passed = 0;
            for (int ff = (f > 0 ? f - 1 : 0); ff <= (f < 7 ? f + 1 : 7); ff++) {
                int rr = r + direction;
                while (rr >= 0 && rr <= 7) {
                    passed |= sq_bit(make_sq(ff, rr));
                    rr += direction;
                }
            }
            PASSED_PAWN_MASKS[color][sq] = passed;

            /* Backward support: adjacent files, from current rank forward */
            uint64_t support = 0;
            for (int af = f - 1; af <= f + 1; af += 2) {
                if (af < 0 || af > 7) continue;
                if (color) {  /* WHITE */
                    for (int rr = r; rr <= 7; rr++)
                        support |= sq_bit(make_sq(af, rr));
                } else {      /* BLACK */
                    for (int rr = 0; rr <= r; rr++)
                        support |= sq_bit(make_sq(af, rr));
                }
            }
            BACKWARD_SUPPORT_MASKS[color][sq] = support;

            /* Pawn attackers to square (which pawns of `color` can attack `sq`) */
            uint64_t attackers = 0;
            int src_rank = r - direction;
            if (src_rank >= 0 && src_rank <= 7) {
                if (f > 0) attackers |= sq_bit(make_sq(f - 1, src_rank));
                if (f < 7) attackers |= sq_bit(make_sq(f + 1, src_rank));
            }
            PAWN_ATTACKERS_TO_SQ[color][sq] = attackers;

            /* Pawn push masks */
            uint64_t single = 0, dbl = 0, captures = 0;
            int r1 = r + direction;
            if (r1 >= 0 && r1 <= 7) {
                single = sq_bit(make_sq(f, r1));
                int start_rank = color ? 1 : 6;
                int r2 = r + 2 * direction;
                if (r == start_rank && r2 >= 0 && r2 <= 7)
                    dbl = sq_bit(make_sq(f, r2));
                if (f > 0) captures |= sq_bit(make_sq(f - 1, r1));
                if (f < 7) captures |= sq_bit(make_sq(f + 1, r1));
            }
            PAWN_SINGLE_PUSH[color][sq] = single;
            PAWN_DOUBLE_PUSH[color][sq] = dbl;
            PAWN_CAPTURE_MASKS[color][sq] = captures;
        }
    }

    tables_initialized = 1;
}

/* ================================================================
 * Sliding piece attack generation
 * ================================================================ */

static uint64_t rook_attacks(int sq, uint64_t occ) {
    uint64_t attacks = 0;
    int f = sq_file(sq), r = sq_rank(sq);
    for (int rr = r + 1; rr <= 7; rr++) { int s = make_sq(f, rr); attacks |= sq_bit(s); if (occ & sq_bit(s)) break; }
    for (int rr = r - 1; rr >= 0; rr--) { int s = make_sq(f, rr); attacks |= sq_bit(s); if (occ & sq_bit(s)) break; }
    for (int ff = f + 1; ff <= 7; ff++) { int s = make_sq(ff, r); attacks |= sq_bit(s); if (occ & sq_bit(s)) break; }
    for (int ff = f - 1; ff >= 0; ff--) { int s = make_sq(ff, r); attacks |= sq_bit(s); if (occ & sq_bit(s)) break; }
    return attacks;
}

static uint64_t bishop_attacks(int sq, uint64_t occ) {
    uint64_t attacks = 0;
    int f = sq_file(sq), r = sq_rank(sq);
    for (int d = 1; f+d <= 7 && r+d <= 7; d++) { int s = make_sq(f+d,r+d); attacks |= sq_bit(s); if (occ & sq_bit(s)) break; }
    for (int d = 1; f+d <= 7 && r-d >= 0; d++) { int s = make_sq(f+d,r-d); attacks |= sq_bit(s); if (occ & sq_bit(s)) break; }
    for (int d = 1; f-d >= 0 && r+d <= 7; d++) { int s = make_sq(f-d,r+d); attacks |= sq_bit(s); if (occ & sq_bit(s)) break; }
    for (int d = 1; f-d >= 0 && r-d >= 0; d++) { int s = make_sq(f-d,r-d); attacks |= sq_bit(s); if (occ & sq_bit(s)) break; }
    return attacks;
}

static uint64_t piece_attacks(int sq, int piece_type, uint64_t occ) {
    switch (piece_type) {
        case 0: return 0;  /* PAWN handled separately */
        case 1: return KNIGHT_ATTACKS[sq];
        case 2: return bishop_attacks(sq, occ);
        case 3: return rook_attacks(sq, occ);
        case 4: return bishop_attacks(sq, occ) | rook_attacks(sq, occ);  /* QUEEN */
        case 5: return KING_ATTACKS[sq];
        default: return 0;
    }
}

/* ================================================================
 * Feature computation helpers
 * ================================================================ */

/* Write bitboard to a plane with orientation */
static void bb_to_plane(float *plane, uint64_t bb, int turn_white) {
    /* plane is 8x8 row-major, pre-zeroed */
    int sq;
    FOR_EACH_BIT(bb, sq) {
        int f = sq_file(sq), r = sq_rank(sq);
        int row = turn_white ? r : (7 - r);
        plane[row * 8 + f] = 1.0f;
    }
}

/* King zone: king square + adjacent + 1-2 ranks forward */
static uint64_t king_zone(int king_sq, int color) {
    if (king_sq < 0 || king_sq > 63) return 0;
    uint64_t zone = KING_ATTACKS[king_sq] | sq_bit(king_sq);
    int kf = sq_file(king_sq), kr = sq_rank(king_sq);
    int dr1 = color ? 1 : -1;
    int dr2 = color ? 2 : -2;
    for (int df = -1; df <= 1; df++) {
        int f = kf + df;
        if (f < 0 || f > 7) continue;
        int r1 = kr + dr1;
        if (r1 >= 0 && r1 <= 7) zone |= sq_bit(make_sq(f, r1));
        int r2 = kr + dr2;
        if (r2 >= 0 && r2 <= 7) zone |= sq_bit(make_sq(f, r2));
    }
    return zone;
}

/* Ray step between two aligned squares, or 0 if not aligned */
static int ray_step(int src, int dst) {
    int sf = sq_file(src), sr = sq_rank(src);
    int df = sq_file(dst), dr = sq_rank(dst);
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

/* Check if a slider is aligned with a target square */
static int is_slider_aligned(int src, int dst, int piece_type) {
    int dx = sq_file(dst) - sq_file(src);
    int dy = sq_rank(dst) - sq_rank(src);
    if (piece_type == 2)  /* BISHOP */
        return abs(dx) == abs(dy) && dx != 0;
    if (piece_type == 3)  /* ROOK */
        return (dx == 0) != (dy == 0);
    if (piece_type == 4)  /* QUEEN */
        return ((dx == 0) != (dy == 0)) || (abs(dx) == abs(dy) && dx != 0);
    return 0;
}

/* Walk from src toward dst by step, counting blockers */
static int walk_ray_blockers(int src, int dst, int step, uint64_t occ, int *blocker_sq) {
    int count = 0;
    *blocker_sq = -1;
    int cur = src + step;
    while (cur != dst && cur >= 0 && cur < 64) {
        /* Bounds check: make sure we haven't wrapped around the board */
        int cf = sq_file(cur), pf = sq_file(cur - step);
        if (abs(cf - pf) > 1) break;  /* wrapped */
        if (occ & sq_bit(cur)) {
            *blocker_sq = cur;
            count++;
            if (count > 1) break;
        }
        cur += step;
        if (cur < 0 || cur > 63) break;
    }
    return count;
}

/* Check if color attacks target_sq (for discovered attack in-check path) */
static int is_attacked_by(uint64_t pieces[6], int color_idx, uint64_t occ, int target_sq) {
    /* pieces[0..5] = PAWN..KING for this color */
    int sq;
    /* Knights */
    if (KNIGHT_ATTACKS[target_sq] & pieces[1]) return 1;
    /* Bishops + Queens (diagonal) */
    uint64_t diag_attackers = pieces[2] | pieces[4];
    uint64_t diag = bishop_attacks(target_sq, occ);
    if (diag & diag_attackers) return 1;
    /* Rooks + Queens (orthogonal) */
    uint64_t orth_attackers = pieces[3] | pieces[4];
    uint64_t orth = rook_attacks(target_sq, occ);
    if (orth & orth_attackers) return 1;
    /* Pawns: target_sq attacked by a pawn of color means a pawn can capture to target_sq */
    /* Pawn of WHITE on (f-1,r-1) or (f+1,r-1) attacks target_sq */
    int opp_color = color_idx;  /* color attacking */
    if (PAWN_ATTACKERS_TO_SQ[opp_color][target_sq] & pieces[0]) return 1;
    /* King */
    if (KING_ATTACKS[target_sq] & pieces[5]) return 1;
    return 0;
}

/* Discovered attack mask */
static uint64_t discovered_attack_mask(
    uint64_t own_pieces[6], uint64_t all_own, uint64_t occ,
    int opp_king_sq, int color /* 0=BLACK, 1=WHITE */
) {
    if (opp_king_sq < 0 || opp_king_sq > 63) return 0;

    /* Check if we're already attacking the opponent king */
    if (is_attacked_by(own_pieces, color, occ, opp_king_sq)) {
        /* Count attackers */
        int n_attackers = 0;
        uint64_t attackers = 0;
        int sq;

        /* Knight attackers */
        uint64_t ka = KNIGHT_ATTACKS[opp_king_sq] & own_pieces[1];
        n_attackers += popcount64(ka);
        attackers |= ka;

        /* Bishop + Queen (diagonal) */
        uint64_t diag = bishop_attacks(opp_king_sq, occ);
        uint64_t da = diag & (own_pieces[2] | own_pieces[4]);
        n_attackers += popcount64(da);
        attackers |= da;

        /* Rook + Queen (orthogonal) */
        uint64_t orth = rook_attacks(opp_king_sq, occ);
        uint64_t ra = orth & (own_pieces[3] | own_pieces[4]);
        n_attackers += popcount64(ra);
        attackers |= ra;

        /* Pawn */
        uint64_t pa = PAWN_ATTACKERS_TO_SQ[color][opp_king_sq] & own_pieces[0];
        n_attackers += popcount64(pa);
        attackers |= pa;

        if (n_attackers >= 2) return all_own;

        /* Single attacker: check if removing it reveals a hidden slider */
        int attacker_sq = lsb64(attackers);
        uint64_t occ_without = occ & ~sq_bit(attacker_sq);
        /* Recheck diagonal and orthogonal attacks without the attacker */
        uint64_t diag2 = bishop_attacks(opp_king_sq, occ_without);
        uint64_t orth2 = rook_attacks(opp_king_sq, occ_without);
        if ((diag2 & (own_pieces[2] | own_pieces[4]) & ~sq_bit(attacker_sq)) ||
            (orth2 & (own_pieces[3] | own_pieces[4]) & ~sq_bit(attacker_sq)))
            return all_own;
        return all_own & ~attackers;
    }

    /* Not in check: find pieces on slider rays to opponent king */
    uint64_t has_sliders = own_pieces[2] | own_pieces[3] | own_pieces[4];
    if (!has_sliders) return 0;

    uint64_t discovered = 0;
    for (int pt = 2; pt <= 4; pt++) {
        int sq;
        FOR_EACH_BIT(own_pieces[pt], sq) {
            if (!is_slider_aligned(sq, opp_king_sq, pt)) continue;
            int step = ray_step(sq, opp_king_sq);
            if (!step) continue;
            int blocker_sq;
            int count = walk_ray_blockers(sq, opp_king_sq, step, occ, &blocker_sq);
            if (count == 1 && blocker_sq >= 0 && (all_own & sq_bit(blocker_sq)))
                discovered |= sq_bit(blocker_sq);
        }
    }
    return discovered;
}

/* Pin mask computation for one color */
static void compute_pins(
    uint64_t color_occ, uint64_t occ,
    int king_sq,
    uint64_t opp_bishops, uint64_t opp_rooks, uint64_t opp_queens,
    uint64_t *pinned_out, uint64_t *pin_ray_out
) {
    *pinned_out = 0;
    *pin_ray_out = 0;
    if (king_sq < 0 || king_sq > 63) return;

    /* Check each slider type's rays from the king */
    /* Diagonal sliders: bishops + queens */
    uint64_t diag_sliders = opp_bishops | opp_queens;
    /* Orthogonal sliders: rooks + queens */
    uint64_t orth_sliders = opp_rooks | opp_queens;

    /* For each direction, trace ray from king outward */
    static const int diag_df[4] = {1, 1, -1, -1};
    static const int diag_dr[4] = {1, -1, 1, -1};
    static const int orth_df[4] = {1, -1, 0, 0};
    static const int orth_dr[4] = {0, 0, 1, -1};

    /* Diagonal rays — python-chess pin_mask extends to board edge in BOTH
       directions from king, so we must do the same. */
    for (int d = 0; d < 4; d++) {
        int kf = sq_file(king_sq), kr = sq_rank(king_sq);
        int own_blocker = -1;
        uint64_t ray = 0;
        int found_pinner = 0;
        for (int dist = 1; dist <= 7; dist++) {
            int f = kf + diag_df[d] * dist;
            int r = kr + diag_dr[d] * dist;
            if (f < 0 || f > 7 || r < 0 || r > 7) break;
            int sq = make_sq(f, r);
            ray |= sq_bit(sq);
            if (found_pinner) continue;  /* extend ray to board edge */
            if (occ & sq_bit(sq)) {
                if (color_occ & sq_bit(sq)) {
                    if (own_blocker >= 0) break;
                    own_blocker = sq;
                } else {
                    if ((diag_sliders & sq_bit(sq)) && own_blocker >= 0) {
                        *pinned_out |= sq_bit(own_blocker);
                        found_pinner = 1;
                    } else {
                        break;
                    }
                }
            }
        }
        if (found_pinner) {
            /* Extend ray in opposite direction from king to board edge */
            for (int dist = 1; dist <= 7; dist++) {
                int f = kf - diag_df[d] * dist;
                int r = kr - diag_dr[d] * dist;
                if (f < 0 || f > 7 || r < 0 || r > 7) break;
                ray |= sq_bit(make_sq(f, r));
            }
            *pin_ray_out |= ray | sq_bit(king_sq);
        }
    }

    /* Orthogonal rays */
    for (int d = 0; d < 4; d++) {
        int kf = sq_file(king_sq), kr = sq_rank(king_sq);
        int own_blocker = -1;
        uint64_t ray = 0;
        int found_pinner = 0;
        for (int dist = 1; dist <= 7; dist++) {
            int f = kf + orth_df[d] * dist;
            int r = kr + orth_dr[d] * dist;
            if (f < 0 || f > 7 || r < 0 || r > 7) break;
            int sq = make_sq(f, r);
            ray |= sq_bit(sq);
            if (found_pinner) continue;
            if (occ & sq_bit(sq)) {
                if (color_occ & sq_bit(sq)) {
                    if (own_blocker >= 0) break;
                    own_blocker = sq;
                } else {
                    if ((orth_sliders & sq_bit(sq)) && own_blocker >= 0) {
                        *pinned_out |= sq_bit(own_blocker);
                        found_pinner = 1;
                    } else {
                        break;
                    }
                }
            }
        }
        if (found_pinner) {
            for (int dist = 1; dist <= 7; dist++) {
                int f = kf - orth_df[d] * dist;
                int r = kr - orth_dr[d] * dist;
                if (f < 0 || f > 7 || r < 0 || r > 7) break;
                ray |= sq_bit(make_sq(f, r));
            }
            *pin_ray_out |= ray | sq_bit(king_sq);
        }
    }
}

/* Pawn structure */
static uint64_t passed_pawns(uint64_t own_pawns, uint64_t enemy_pawns, int color) {
    uint64_t passed = 0;
    int sq;
    FOR_EACH_BIT(own_pawns, sq) {
        if (!(PASSED_PAWN_MASKS[color][sq] & enemy_pawns))
            passed |= sq_bit(sq);
    }
    return passed;
}

static uint64_t isolated_pawns(uint64_t own_pawns) {
    uint64_t isolated = 0;
    int sq;
    FOR_EACH_BIT(own_pawns, sq) {
        int f = sq_file(sq);
        if (!(ADJACENT_FILE_MASKS[f] & own_pawns))
            isolated |= sq_bit(sq);
    }
    return isolated;
}

static uint64_t backward_pawns(uint64_t own_pawns, uint64_t enemy_pawns, int color) {
    uint64_t backward = 0;
    int direction = color ? 1 : -1;
    int sq;
    FOR_EACH_BIT(own_pawns, sq) {
        int f = sq_file(sq), r = sq_rank(sq);
        if (!(ADJACENT_FILE_MASKS[f] & own_pawns)) continue;
        int r1 = r + direction;
        if (r1 < 0 || r1 > 7) continue;
        int front_sq = make_sq(f, r1);
        int opp_color = color ? 0 : 1;
        if (!(PAWN_ATTACKERS_TO_SQ[opp_color][front_sq] & enemy_pawns)) continue;
        if (BACKWARD_SUPPORT_MASKS[color][sq] & own_pawns) continue;
        backward |= sq_bit(sq);
    }
    return backward;
}

static uint64_t connected_pawns(uint64_t own_pawns) {
    uint64_t connected = 0;
    int sq;
    FOR_EACH_BIT(own_pawns, sq) {
        if (CONNECTED_NEIGHBOR_MASKS[sq] & own_pawns)
            connected |= sq_bit(sq);
    }
    return connected;
}

/* ================================================================
 * Main: compute all 34 feature planes
 * ================================================================ */

/*
 * Arguments from Python:
 *   pieces_us:   uint64[6] — PAWN..KING for side-to-move
 *   pieces_them: uint64[6] — PAWN..KING for opponent
 *   occupied:    uint64
 *   king_sq_us:  int
 *   king_sq_them: int
 *   turn_white:  int (1 if White to move)
 *   ep_square:   int (-1 if none)
 */
static PyObject* py_compute_features(PyObject *self, PyObject *args) {
    PyArrayObject *pieces_us_arr, *pieces_them_arr;
    uint64_t occupied;
    int king_sq_us, king_sq_them, turn_white, ep_square;

    if (!PyArg_ParseTuple(args, "O!O!Kiipi",
            &PyArray_Type, &pieces_us_arr,
            &PyArray_Type, &pieces_them_arr,
            &occupied,
            &king_sq_us, &king_sq_them,
            &turn_white, &ep_square))
        return NULL;

    init_tables();

    /* Extract piece bitboards */
    uint64_t *pus = (uint64_t *)PyArray_DATA(pieces_us_arr);
    uint64_t *pthem = (uint64_t *)PyArray_DATA(pieces_them_arr);

    uint64_t us_pieces[6], them_pieces[6];
    for (int i = 0; i < 6; i++) {
        us_pieces[i] = pus[i];
        them_pieces[i] = pthem[i];
    }

    uint64_t us_occ = 0, them_occ = 0;
    for (int i = 0; i < 6; i++) {
        us_occ |= us_pieces[i];
        them_occ |= them_pieces[i];
    }

    int us_color = turn_white ? 1 : 0;
    int them_color = turn_white ? 0 : 1;

    /* Allocate output (34, 8, 8) float32 */
    npy_intp dims[3] = {34, 8, 8};
    PyObject *out_arr = PyArray_ZEROS(3, dims, NPY_FLOAT32, 0);
    if (!out_arr) return NULL;
    float *out = (float *)PyArray_DATA((PyArrayObject *)out_arr);

    /* ---- King safety: 10 planes [0:10] ---- */
    int plane_idx = 0;
    int colors[2] = {us_color, them_color};
    int king_sqs[2] = {king_sq_us, king_sq_them};
    uint64_t *color_pieces[2] = {us_pieces, them_pieces};

    for (int ci = 0; ci < 2; ci++) {
        uint64_t kz = king_zone(king_sqs[ci], colors[ci]);
        bb_to_plane(&out[plane_idx * 64], kz, turn_white);
        plane_idx++;

        int opp_ci = 1 - ci;
        uint64_t *opp_p = color_pieces[opp_ci];
        /* N, B, R, Q attacks on king zone */
        for (int pt = 1; pt <= 4; pt++) {
            uint64_t overlap = 0;
            int sq;
            FOR_EACH_BIT(opp_p[pt], sq) {
                overlap |= piece_attacks(sq, pt, occupied) & kz;
            }
            bb_to_plane(&out[plane_idx * 64], overlap, turn_white);
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
        compute_pins(co, occupied, ksq,
                     opp_p[2], opp_p[3], opp_p[4],
                     &pinned, &pin_ray);

        uint64_t disc = discovered_attack_mask(
            color_pieces[ci],
            ci == 0 ? us_occ : them_occ,
            occupied,
            king_sqs[opp_ci],
            colors[ci]
        );

        bb_to_plane(&out[plane_idx * 64], pinned, turn_white);
        plane_idx++;
        bb_to_plane(&out[plane_idx * 64], pin_ray, turn_white);
        plane_idx++;
        bb_to_plane(&out[plane_idx * 64], disc, turn_white);
        plane_idx++;
    }

    /* ---- Pawn structure: 8 planes [16:24] ---- */
    for (int ci = 0; ci < 2; ci++) {
        uint64_t own_p = color_pieces[ci][0];
        uint64_t enemy_p = color_pieces[1 - ci][0];
        int c = colors[ci];

        bb_to_plane(&out[plane_idx * 64], passed_pawns(own_p, enemy_p, c), turn_white);
        plane_idx++;
        bb_to_plane(&out[plane_idx * 64], isolated_pawns(own_p), turn_white);
        plane_idx++;
        bb_to_plane(&out[plane_idx * 64], backward_pawns(own_p, enemy_p, c), turn_white);
        plane_idx++;
        bb_to_plane(&out[plane_idx * 64], connected_pawns(own_p), turn_white);
        plane_idx++;
    }

    /* ---- Mobility: 6 planes [24:30] ---- */
    static const float MOB_MAX[6] = {8.0f, 13.0f, 14.0f, 27.0f, 8.0f, 4.0f};
    /* Order: KNIGHT, BISHOP, ROOK, QUEEN, KING, PAWN */
    static const int MOB_PT[6] = {1, 2, 3, 4, 5, 0};

    uint64_t ep_mask = (ep_square >= 0 && ep_square < 64) ? sq_bit(ep_square) : 0;

    for (int mi = 0; mi < 6; mi++) {
        float *plane = &out[plane_idx * 64];
        int pt = MOB_PT[mi];
        float max_m = MOB_MAX[mi];

        for (int c = 1; c >= 0; c--) {  /* WHITE=1 first, then BLACK=0 */
            uint64_t *cp = (c == us_color) ? us_pieces : them_pieces;
            uint64_t own_occ = (c == us_color) ? us_occ : them_occ;
            uint64_t opp_occ = (c == us_color) ? them_occ : us_occ;
            int sq;

            FOR_EACH_BIT(cp[pt], sq) {
                int mobility;
                if (pt == 0) {  /* PAWN */
                    mobility = 0;
                    uint64_t single = PAWN_SINGLE_PUSH[c][sq];
                    if (single && !(occupied & single)) {
                        mobility++;
                        uint64_t dbl = PAWN_DOUBLE_PUSH[c][sq];
                        if (dbl && !(occupied & dbl))
                            mobility++;
                    }
                    uint64_t cap = PAWN_CAPTURE_MASKS[c][sq];
                    mobility += popcount64(cap & opp_occ);
                    if (ep_mask && (cap & ep_mask))
                        mobility++;
                } else {
                    uint64_t att = piece_attacks(sq, pt, occupied);
                    mobility = popcount64(att & ~own_occ);
                }

                int f = sq_file(sq), r = sq_rank(sq);
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

        /* Outpost: squares attacked by own pawns but not by enemy pawns */
        uint64_t own_att = 0, enemy_att = 0;
        int sq;
        FOR_EACH_BIT(own_pawns, sq) { own_att |= PAWN_ATTACKS[c][sq]; }
        FOR_EACH_BIT(enemy_pawns, sq) { enemy_att |= PAWN_ATTACKS[opp_c][sq]; }
        bb_to_plane(&out[plane_idx * 64], own_att & ~enemy_att, turn_white);
        plane_idx++;

        /* Space: squares behind own center-file pawns */
        int direction = c ? -1 : 1;  /* Behind = opposite of forward */
        uint64_t space = 0;
        FOR_EACH_BIT(own_pawns, sq) {
            int f = sq_file(sq), r = sq_rank(sq);
            if (f < 2 || f > 5) continue;  /* center files c-f only */
            for (int dd = 1; dd <= 2; dd++) {
                int r2 = r + direction * dd;
                if (r2 >= 0 && r2 <= 7)
                    space |= sq_bit(make_sq(f, r2));
            }
        }
        bb_to_plane(&out[plane_idx * 64], space, turn_white);
        plane_idx++;
    }

    return out_arr;
}

/* ================================================================
 * Module definition
 * ================================================================ */

static PyMethodDef methods[] = {
    {"compute_extra_features", py_compute_features, METH_VARARGS,
     "Compute 34 extra feature planes from bitboard data.\n\n"
     "Args: pieces_us(uint64[6]), pieces_them(uint64[6]), occupied(uint64),\n"
     "      king_sq_us(int), king_sq_them(int), turn_white(bool), ep_square(int)\n"
     "Returns: ndarray (34, 8, 8) float32"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_features_ext",
    "C-accelerated feature plane computation",
    -1,
    methods,
};

PyMODINIT_FUNC PyInit__features_ext(void) {
    PyObject *m = PyModule_Create(&moduledef);
    if (!m) return NULL;
    import_array();
    init_tables();
    return m;
}
