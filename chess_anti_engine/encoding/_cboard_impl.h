/*
 * _cboard_impl.h — Pure-C CBoard implementation (no Python/NumPy dependencies).
 *
 * All functions are static so each .so that includes this gets its own copy.
 * Call cboard_init_all() once at module init time to populate lookup tables.
 */

#ifndef _CBOARD_IMPL_H
#define _CBOARD_IMPL_H

#include <stdint.h>
#include <stdio.h>   /* snprintf in cboard_to_fen */
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ================================================================
 * Bitboard utilities
 * ================================================================ */

static inline int popcount64(uint64_t x) { return __builtin_popcountll(x); }
static inline int lsb64(uint64_t x)      { return __builtin_ctzll(x); }
static inline uint64_t sq_bit(int sq)     { return 1ULL << sq; }
static inline int sq_file(int sq)         { return sq & 7; }
static inline int sq_rank(int sq)         { return sq >> 3; }
static inline int make_sq(int f, int r)   { return r * 8 + f; }

/* Orient square for side to move (BLACK flips ranks) */
static inline int orient_sq(int sq, int is_white) {
    return is_white ? sq : (sq ^ 56);
}

/* Token-paste __COUNTER__ so every expansion gets a unique iterator name.
 * Without this, nesting FOR_EACH_BIT inside itself triggers -Wshadow. */
#define _FOR_EACH_BIT_IMPL(bb, sq, n) \
    for (uint64_t _bb##n = (bb); _bb##n; _bb##n &= _bb##n - 1) \
        if (((sq) = lsb64(_bb##n)), 1)
#define _FOR_EACH_BIT_CAT(bb, sq, n) _FOR_EACH_BIT_IMPL(bb, sq, n)
#define FOR_EACH_BIT(bb, sq) _FOR_EACH_BIT_CAT(bb, sq, __COUNTER__)

/* ================================================================
 * Attack tables (initialized once)
 * ================================================================ */

static uint64_t KNIGHT_ATTACKS[64];
static uint64_t KING_ATTACKS[64];
static uint64_t PAWN_ATTACKS[2][64];  /* [0=BLACK, 1=WHITE][sq] */

/* Ray attacks: for sliding pieces, precomputed first-blocker lookup is expensive
 * to set up (magic bitboards). Instead we use simple ray iteration which is
 * fast enough for ~30 moves per position. */

static int attack_tables_initialized = 0;

static void init_attack_tables(void) {
    if (attack_tables_initialized) return;

    static const int knight_df[] = {1, 2, 2, 1, -1, -2, -2, -1};
    static const int knight_dr[] = {2, 1, -1, -2, -2, -1, 1, 2};
    static const int king_df[]   = {-1, -1, -1, 0, 0, 1, 1, 1};
    static const int king_dr[]   = {-1, 0, 1, -1, 1, -1, 0, 1};

    for (int sq = 0; sq < 64; sq++) {
        int f = sq_file(sq), r = sq_rank(sq);
        uint64_t n = 0, k = 0;
        for (int i = 0; i < 8; i++) {
            int nf = f + knight_df[i], nr = r + knight_dr[i];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8)
                n |= sq_bit(make_sq(nf, nr));
            int kf = f + king_df[i], kr = r + king_dr[i];
            if (kf >= 0 && kf < 8 && kr >= 0 && kr < 8)
                k |= sq_bit(make_sq(kf, kr));
        }
        KNIGHT_ATTACKS[sq] = n;
        KING_ATTACKS[sq] = k;

        /* White pawn attacks (from white pawn at sq) */
        uint64_t wp = 0;
        if (r < 7) {
            if (f > 0) wp |= sq_bit(make_sq(f - 1, r + 1));
            if (f < 7) wp |= sq_bit(make_sq(f + 1, r + 1));
        }
        PAWN_ATTACKS[1][sq] = wp;

        /* Black pawn attacks */
        uint64_t bp = 0;
        if (r > 0) {
            if (f > 0) bp |= sq_bit(make_sq(f - 1, r - 1));
            if (f < 7) bp |= sq_bit(make_sq(f + 1, r - 1));
        }
        PAWN_ATTACKS[0][sq] = bp;
    }

    attack_tables_initialized = 1;
}

/* ================================================================
 * Zobrist hashing tables (for repetition detection)
 * ================================================================ */

/* 12 piece-color types x 64 squares, plus turn, castling */
static uint64_t ZOBRIST_PIECE[12][64];  /* [piece_color_idx][sq] */
static uint64_t ZOBRIST_TURN;
static uint64_t ZOBRIST_CASTLING[16];   /* indexed by 4-bit castling */
/* En-passant file. NOT part of CBoard.hash (see cboard_compute_hash) — used
 * only by cboard_transposition_key for search transposition tables, whose
 * entries must agree on the legal move set. Drawn AFTER the tables above so
 * every pre-existing Zobrist value stays bit-identical: CBoard.hash is
 * persisted in selfplay resume records (`selfplay/resume.py` pos_hash checks),
 * and perturbing it would invalidate in-flight resume state. */
static uint64_t ZOBRIST_EP[8];          /* indexed by file of ep_square */
static int zobrist_initialized = 0;

/* Simple xorshift64 PRNG for deterministic Zobrist values */
static uint64_t zobrist_rand_state = 0x3243F6A8885A308DULL;
static uint64_t zobrist_rand64(void) {
    uint64_t x = zobrist_rand_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    zobrist_rand_state = x;
    return x;
}

static void init_zobrist(void) {
    if (zobrist_initialized) return;
    zobrist_rand_state = 0x3243F6A8885A308DULL;
    for (int pc = 0; pc < 12; pc++)
        for (int sq = 0; sq < 64; sq++)
            ZOBRIST_PIECE[pc][sq] = zobrist_rand64();
    ZOBRIST_TURN = zobrist_rand64();
    for (int c = 0; c < 16; c++)
        ZOBRIST_CASTLING[c] = zobrist_rand64();
    /* Appended last on purpose — keeps every value above bit-identical. */
    for (int f = 0; f < 8; f++)
        ZOBRIST_EP[f] = zobrist_rand64();
    zobrist_initialized = 1;
}

/* Piece-color index: 0-5 = white PNBRQK, 6-11 = black PNBRQK */
static inline int piece_color_idx(int piece_type, int color) {
    return color * 6 + piece_type; /* color: 0=BLACK, 1=WHITE */
}

/* ================================================================
 * Sliding piece ray attacks
 * ================================================================ */

static const int RAY_DF[8] = {0, 1, 1, 1, 0, -1, -1, -1};
static const int RAY_DR[8] = {1, 1, 0, -1, -1, -1, 0, 1};

/* Get all squares attacked by a slider from sq in given directions, blocked by occupied */
static uint64_t slider_attacks(int sq, uint64_t occupied, int bishop_like) {
    uint64_t attacks = 0;
    /* bishop: dirs 1,3,5,7; rook: dirs 0,2,4,6 */
    int start = bishop_like ? 1 : 0;
    for (int d = start; d < 8; d += 2) {
        int f = sq_file(sq), r = sq_rank(sq);
        for (;;) {
            f += RAY_DF[d];
            r += RAY_DR[d];
            if (f < 0 || f > 7 || r < 0 || r > 7) break;
            int s = make_sq(f, r);
            attacks |= sq_bit(s);
            if (occupied & sq_bit(s)) break;
        }
    }
    return attacks;
}

static uint64_t bishop_attacks(int sq, uint64_t occ) { return slider_attacks(sq, occ, 1); }
static uint64_t rook_attacks(int sq, uint64_t occ)   { return slider_attacks(sq, occ, 0); }
static uint64_t queen_attacks(int sq, uint64_t occ)   {
    return bishop_attacks(sq, occ) | rook_attacks(sq, occ);
}

/* Check if square is attacked by given side */
static int is_attacked_by(int sq, uint64_t occ,
                           uint64_t pawns, uint64_t knights, uint64_t bishops,
                           uint64_t rooks, uint64_t queens, uint64_t kings,
                           int attacker_is_white) {
    /* Pawn attacks */
    if (PAWN_ATTACKS[1 - attacker_is_white][sq] & pawns) return 1;
    /* Knight attacks */
    if (KNIGHT_ATTACKS[sq] & knights) return 1;
    /* King attacks */
    if (KING_ATTACKS[sq] & kings) return 1;
    /* Bishop/queen diagonal */
    if (bishop_attacks(sq, occ) & (bishops | queens)) return 1;
    /* Rook/queen straight */
    if (rook_attacks(sq, occ) & (rooks | queens)) return 1;
    return 0;
}

/* ================================================================
 * Insertion sort for small int arrays
 * ================================================================ */

static void sort_int(int *arr, int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

/* ================================================================
 * Policy index computation (LC0 encoding)
 * ================================================================ */

/* Queen-like directions: same as QUEEN_DIRS in encode.py */
static const int Q_DF[8] = {0, 1, 1, 1, 0, -1, -1, -1};
static const int Q_DR[8] = {1, 1, 0, -1, -1, -1, 0, 1};

/* Knight deltas: same as KNIGHT_DELTAS in encode.py */
static const int KN_DF[8] = {1, 2, 2, 1, -1, -2, -2, -1};
static const int KN_DR[8] = {2, 1, -1, -2, -2, -1, 1, 2};

/* Precomputed: (df,dr) -> plane index. -1 = invalid. */
static int DELTA_TO_PLANE[15][15];  /* indexed by [df+7][dr+7] */
static int tables_ready = 0;

static void init_policy_tables(void) {
    if (tables_ready) return;
    memset(DELTA_TO_PLANE, -1, sizeof(DELTA_TO_PLANE));

    int plane = 0;
    for (int d = 0; d < 8; d++) {
        for (int dist = 1; dist <= 7; dist++) {
            int df = Q_DF[d] * dist;
            int dr = Q_DR[d] * dist;
            DELTA_TO_PLANE[df + 7][dr + 7] = plane;
            plane++;
        }
    }
    /* Knight planes 56..63 */
    for (int i = 0; i < 8; i++) {
        DELTA_TO_PLANE[KN_DF[i] + 7][KN_DR[i] + 7] = 56 + i;
    }
    tables_ready = 1;
}

/* Compute policy index for a move (oriented from/to) */
static inline int move_to_policy_index(int from_o, int to_o, int promotion) {
    int ff = sq_file(from_o), fr = sq_rank(from_o);
    int tf = sq_file(to_o), tr = sq_rank(to_o);
    int df = tf - ff, dr = tr - fr;

    /* Underpromotion */
    if (promotion > 0 && promotion != 5) {  /* 5 = QUEEN */
        /* promotion: 2=KNIGHT, 3=BISHOP, 4=ROOK */
        int piece_idx;
        switch (promotion) {
            case 2: piece_idx = 0; break;  /* KNIGHT */
            case 3: piece_idx = 1; break;  /* BISHOP */
            case 4: piece_idx = 2; break;  /* ROOK */
            default: piece_idx = 0; break;
        }
        int dir_idx;
        if (df == -1) dir_idx = 0;
        else if (df == 0) dir_idx = 1;
        else dir_idx = 2;
        return from_o * 73 + 64 + piece_idx * 3 + dir_idx;
    }

    if (df < -7 || df > 7 || dr < -7 || dr > 7) return -1;
    int plane = DELTA_TO_PLANE[df + 7][dr + 7];
    if (plane < 0) return -1;
    return from_o * 73 + plane;
}


/* ================================================================
 * Bitboard -> plane conversion
 * ================================================================ */

#include "_bitboard_planes_impl.h"


/* ================================================================
 * BoardState + legal move generation
 * ================================================================ */

typedef struct {
    uint64_t us_pawns, us_knights, us_bishops, us_rooks, us_queens, us_kings;
    uint64_t them_pawns, them_knights, them_bishops, them_rooks, them_queens, them_kings;
    uint64_t us_occ, them_occ, all_occ;
    int turn;  /* 1=WHITE, 0=BLACK */
    int us_castle_k, us_castle_q, them_castle_k, them_castle_q;
    int ep_square;  /* -1 or 0..63 */
    int king_sq;
} BoardState;

/* Max legal moves any caller's ``indices`` buffer must hold. A real chess
 * position has at most 218 legal moves, so this never truncates a reachable
 * position; it is the buffer size every caller uses (``int indices[256]``)
 * and the hard cap the generator must not write past. The cap matters only
 * for non-chess bitboards reaching the generator via ``from_raw`` (arbitrary
 * Python input), where an unbounded write would overflow the caller stack. */
#define CBOARD_MAX_LEGAL_MOVES 256

/* Generate all pseudo-legal moves and filter by legality.
 * Writes policy indices to `indices` (capacity CBOARD_MAX_LEGAL_MOVES),
 * returns count. */
static int generate_legal_move_indices(const BoardState *bs, int *indices) {
    int count = 0;
    int is_white = bs->turn;
    int king_sq = bs->king_sq;

    /* Helper: test if a move is legal by making it on copies of bitboards
     * and checking if own king is attacked.
     * This is the simplest approach -- check king safety after each move. */

    /* We need: after removing piece from `from` and placing at `to` (with capture),
     * is our king attacked by their pieces? */

#define ADD_MOVE(from_sq, to_sq, promo) do { \
    int _from_o = orient_sq(from_sq, is_white); \
    int _to_o = orient_sq(to_sq, is_white); \
    int _idx = move_to_policy_index(_from_o, _to_o, promo); \
    if (_idx >= 0 && count < CBOARD_MAX_LEGAL_MOVES) indices[count++] = _idx; \
} while(0)

    /* Test if king would be in check after moving piece from `from` to `to`.
     * `capture_sq` is the square where an enemy piece is removed (-1 if none).
     * If the moving piece is the king, king_sq changes to `to`. */
#define IS_LEGAL_MOVE(from_sq, to_sq, capture_sq, moving_king) do { \
    uint64_t new_all = bs->all_occ & ~sq_bit(from_sq); \
    uint64_t new_them_pawns = bs->them_pawns; \
    uint64_t new_them_knights = bs->them_knights; \
    uint64_t new_them_bishops = bs->them_bishops; \
    uint64_t new_them_rooks = bs->them_rooks; \
    uint64_t new_them_queens = bs->them_queens; \
    uint64_t new_them_kings = bs->them_kings; \
    if (capture_sq >= 0) { \
        uint64_t cap_bit = sq_bit(capture_sq); \
        new_all &= ~cap_bit; \
        new_them_pawns &= ~cap_bit; \
        new_them_knights &= ~cap_bit; \
        new_them_bishops &= ~cap_bit; \
        new_them_rooks &= ~cap_bit; \
        new_them_queens &= ~cap_bit; \
    } \
    new_all |= sq_bit(to_sq); \
    int check_sq = (moving_king) ? (to_sq) : king_sq; \
    _is_legal = !is_attacked_by(check_sq, new_all, \
        new_them_pawns, new_them_knights, new_them_bishops, \
        new_them_rooks, new_them_queens, new_them_kings, \
        1 - is_white); \
} while(0)

    int _is_legal;

    /* ---- Pawn moves ---- */
    {
        int fwd = is_white ? 8 : -8;
        int start_rank = is_white ? 1 : 6;
        int promo_rank = is_white ? 7 : 0;

        int sq;
        uint64_t pawns = bs->us_pawns;
        FOR_EACH_BIT(pawns, sq) {
            int r = sq_rank(sq), f = sq_file(sq);

            /* Single push */
            int push1 = sq + fwd;
            if (push1 >= 0 && push1 < 64 && !(bs->all_occ & sq_bit(push1))) {
                if (sq_rank(push1) == promo_rank) {
                    /* Promotion: queen (default), and underpromotions */
                    IS_LEGAL_MOVE(sq, push1, -1, 0);
                    if (_is_legal) {
                        ADD_MOVE(sq, push1, 5);  /* queen */
                        ADD_MOVE(sq, push1, 2);  /* knight */
                        ADD_MOVE(sq, push1, 3);  /* bishop */
                        ADD_MOVE(sq, push1, 4);  /* rook */
                    }
                } else {
                    IS_LEGAL_MOVE(sq, push1, -1, 0);
                    if (_is_legal) ADD_MOVE(sq, push1, 0);

                    /* Double push */
                    if (r == start_rank) {
                        int push2 = push1 + fwd;
                        if (!(bs->all_occ & sq_bit(push2))) {
                            IS_LEGAL_MOVE(sq, push2, -1, 0);
                            if (_is_legal) ADD_MOVE(sq, push2, 0);
                        }
                    }
                }
            }

            /* Captures */
            for (int df = -1; df <= 1; df += 2) {
                int cf = f + df;
                if (cf < 0 || cf > 7) continue;
                int cap_sq = make_sq(cf, r + (is_white ? 1 : -1));
                if (cap_sq < 0 || cap_sq >= 64) continue;

                if (bs->them_occ & sq_bit(cap_sq)) {
                    if (sq_rank(cap_sq) == promo_rank) {
                        IS_LEGAL_MOVE(sq, cap_sq, cap_sq, 0);
                        if (_is_legal) {
                            ADD_MOVE(sq, cap_sq, 5);
                            ADD_MOVE(sq, cap_sq, 2);
                            ADD_MOVE(sq, cap_sq, 3);
                            ADD_MOVE(sq, cap_sq, 4);
                        }
                    } else {
                        IS_LEGAL_MOVE(sq, cap_sq, cap_sq, 0);
                        if (_is_legal) ADD_MOVE(sq, cap_sq, 0);
                    }
                }
                /* En passant */
                else if (cap_sq == bs->ep_square) {
                    /* The captured pawn is on the same rank as the moving pawn */
                    int captured_pawn_sq = make_sq(cf, r);
                    /* Special legality check: remove both moving pawn and captured pawn */
                    uint64_t new_all = (bs->all_occ & ~sq_bit(sq) & ~sq_bit(captured_pawn_sq)) | sq_bit(cap_sq);
                    uint64_t new_them_pawns = bs->them_pawns & ~sq_bit(captured_pawn_sq);
                    _is_legal = !is_attacked_by(king_sq, new_all,
                        new_them_pawns, bs->them_knights, bs->them_bishops,
                        bs->them_rooks, bs->them_queens, bs->them_kings,
                        1 - is_white);
                    if (_is_legal) ADD_MOVE(sq, cap_sq, 0);
                }
            }
        }
    }

    /* ---- Knight moves ---- */
    {
        int sq;
        FOR_EACH_BIT(bs->us_knights, sq) {
            uint64_t targets = KNIGHT_ATTACKS[sq] & ~bs->us_occ;
            int to;
            FOR_EACH_BIT(targets, to) {
                int cap = (bs->them_occ & sq_bit(to)) ? to : -1;
                IS_LEGAL_MOVE(sq, to, cap, 0);
                if (_is_legal) ADD_MOVE(sq, to, 0);
            }
        }
    }

    /* ---- Bishop moves ---- */
    {
        int sq;
        FOR_EACH_BIT(bs->us_bishops, sq) {
            uint64_t targets = bishop_attacks(sq, bs->all_occ) & ~bs->us_occ;
            int to;
            FOR_EACH_BIT(targets, to) {
                int cap = (bs->them_occ & sq_bit(to)) ? to : -1;
                IS_LEGAL_MOVE(sq, to, cap, 0);
                if (_is_legal) ADD_MOVE(sq, to, 0);
            }
        }
    }

    /* ---- Rook moves ---- */
    {
        int sq;
        FOR_EACH_BIT(bs->us_rooks, sq) {
            uint64_t targets = rook_attacks(sq, bs->all_occ) & ~bs->us_occ;
            int to;
            FOR_EACH_BIT(targets, to) {
                int cap = (bs->them_occ & sq_bit(to)) ? to : -1;
                IS_LEGAL_MOVE(sq, to, cap, 0);
                if (_is_legal) ADD_MOVE(sq, to, 0);
            }
        }
    }

    /* ---- Queen moves ---- */
    {
        int sq;
        FOR_EACH_BIT(bs->us_queens, sq) {
            uint64_t targets = queen_attacks(sq, bs->all_occ) & ~bs->us_occ;
            int to;
            FOR_EACH_BIT(targets, to) {
                int cap = (bs->them_occ & sq_bit(to)) ? to : -1;
                IS_LEGAL_MOVE(sq, to, cap, 0);
                if (_is_legal) ADD_MOVE(sq, to, 0);
            }
        }
    }

    /* ---- King moves (non-castling) ---- */
    {
        uint64_t targets = KING_ATTACKS[king_sq] & ~bs->us_occ;
        int to;
        FOR_EACH_BIT(targets, to) {
            int cap = (bs->them_occ & sq_bit(to)) ? to : -1;
            IS_LEGAL_MOVE(king_sq, to, cap, 1);
            if (_is_legal) ADD_MOVE(king_sq, to, 0);
        }
    }

    /* ---- Castling ---- */
    {
        /* Kingside */
        if (bs->us_castle_k) {
            int r = is_white ? 0 : 7;
            int e = make_sq(4, r), f_sq = make_sq(5, r), g = make_sq(6, r);
            int rook_sq = make_sq(7, r);
            /* Squares between must be empty */
            if (king_sq == e && (bs->us_kings & sq_bit(e)) &&
                (bs->us_rooks & sq_bit(rook_sq)) &&
                !(bs->all_occ & (sq_bit(f_sq) | sq_bit(g)))) {
                /* King must not be in check, pass through check, or end in check */
                int ok = !is_attacked_by(e, bs->all_occ,
                    bs->them_pawns, bs->them_knights, bs->them_bishops,
                    bs->them_rooks, bs->them_queens, bs->them_kings, 1 - is_white);
                if (ok) ok = !is_attacked_by(f_sq, bs->all_occ,
                    bs->them_pawns, bs->them_knights, bs->them_bishops,
                    bs->them_rooks, bs->them_queens, bs->them_kings, 1 - is_white);
                if (ok) ok = !is_attacked_by(g, bs->all_occ,
                    bs->them_pawns, bs->them_knights, bs->them_bishops,
                    bs->them_rooks, bs->them_queens, bs->them_kings, 1 - is_white);
                if (ok) ADD_MOVE(e, g, 0);
            }
        }
        /* Queenside */
        if (bs->us_castle_q) {
            int r = is_white ? 0 : 7;
            int e = make_sq(4, r), d = make_sq(3, r), c = make_sq(2, r), b = make_sq(1, r);
            int rook_sq = make_sq(0, r);
            if (king_sq == e && (bs->us_kings & sq_bit(e)) &&
                (bs->us_rooks & sq_bit(rook_sq)) &&
                !(bs->all_occ & (sq_bit(d) | sq_bit(c) | sq_bit(b)))) {
                int ok = !is_attacked_by(e, bs->all_occ,
                    bs->them_pawns, bs->them_knights, bs->them_bishops,
                    bs->them_rooks, bs->them_queens, bs->them_kings, 1 - is_white);
                if (ok) ok = !is_attacked_by(d, bs->all_occ,
                    bs->them_pawns, bs->them_knights, bs->them_bishops,
                    bs->them_rooks, bs->them_queens, bs->them_kings, 1 - is_white);
                if (ok) ok = !is_attacked_by(c, bs->all_occ,
                    bs->them_pawns, bs->them_knights, bs->them_bishops,
                    bs->them_rooks, bs->them_queens, bs->them_kings, 1 - is_white);
                if (ok) ADD_MOVE(e, c, 0);
            }
        }
    }

#undef ADD_MOVE
#undef IS_LEGAL_MOVE

    return count;
}

/* Same as generate_legal_move_indices, but sorts the output in-place.
 * Policy-index order matters whenever callers compare across encodings. */
static inline int generate_legal_move_indices_sorted(const BoardState *bs, int *indices) {
    int count = generate_legal_move_indices(bs, indices);
    sort_int(indices, count);
    return count;
}


/* ================================================================
 * CBoard: lightweight C chess board for MCTS hot loop
 *
 * Replaces python-chess Board in the simulation loop.
 * copy() = memcpy ~72 bytes, push = pure C bitboard ops.
 * ================================================================ */

enum { PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5 };
enum { BLACK_C=0, WHITE_C=1 };

/* Tri-state flag in POLICY_LUT entries: init_policy_lut speculatively marks
 * pawn moves that *might* promote (based on the un-oriented target rank)
 * with promotion=PROMO_MAYBE_QUEEN. cboard_push then checks the real
 * destination rank and either applies a queen promotion or treats it as a
 * regular pawn move. Explicit underpromotion uses KNIGHT/BISHOP/ROOK (1..3). */
#define PROMO_MAYBE_QUEEN 5
/* Castling bits */
enum { WK_CASTLE=1, WQ_CASTLE=2, BK_CASTLE=4, BQ_CASTLE=8 };

#define CBOARD_HASH_STACK_MAX 128
#define CBOARD_HISTORY_MAX 7  /* previous positions (current is live) */

typedef struct {
    uint64_t bb[6];    /* piece bitboards: pawns, knights, bishops, rooks, queens, kings */
    uint64_t occ[2];   /* color occupancy: [0]=BLACK, [1]=WHITE */
    int8_t turn;       /* WHITE_C=1, BLACK_C=0 */
    uint8_t castling;  /* WK=1, WQ=2, BK=4, BQ=8 */
    int8_t ep_square;  /* -1 or 0..63 */
    uint8_t halfmove_clock;
    /* --- Zobrist hash for repetition detection --- */
    uint64_t hash;
    int16_t hash_stack_len;
    uint64_t hash_stack[CBOARD_HASH_STACK_MAX];
    /* --- History for encoding (7 previous positions) --- */
    uint64_t hist_bb[CBOARD_HISTORY_MAX][6];   /* piece bitboards */
    uint64_t hist_occ[CBOARD_HISTORY_MAX][2];  /* color occupancy */
    int8_t hist_turn[CBOARD_HISTORY_MAX];      /* side to move */
    uint8_t hist_castling[CBOARD_HISTORY_MAX]; /* castling rights */
    uint64_t hist_hash[CBOARD_HISTORY_MAX];    /* REPETITION KEY of each
                                                * snapshot (cboard_repetition_key,
                                                * i.e. ep-aware), recorded while
                                                * it was the live position. Lets
                                                * the current-repetition check
                                                * see the kept window even after
                                                * hash_stack saturation,
                                                * without recomputing hashes. */
    int8_t hist_ep[CBOARD_HISTORY_MAX];        /* ep_square of each snapshot.
                                                * The encoder's reconstruction
                                                * path re-derives a slot's key
                                                * from the stored bitboards, and
                                                * legal-ep is not recoverable
                                                * from them alone, so the ep
                                                * square must be kept too or the
                                                * reconstructed keys land on a
                                                * different ruler from
                                                * hash_stack's. */
    int8_t hist_was_rep[CBOARD_HISTORY_MAX];   /* was this snapshot a repetition
                                                * when it was the live position?
                                                * Recorded at push time (full
                                                * look-back valid then) so the
                                                * encoder need not reconstruct it
                                                * from the since-cleared
                                                * hash_stack. Used only when
                                                * g_history_rep_fix is enabled. */
    int8_t hist_len;                           /* 0..7 valid history entries */
    int8_t hist_head;                          /* circular buffer write index */
    uint16_t ply;                              /* total half-moves from game start */
} CBoard;

/* Reset the per-slot ep squares to the "no ep" sentinel: hist_ep is int8_t, so a
 * zeroed slot reads as ep on square 0 (a1), not as "no ep".
 *
 * ⚑ DEFENCE IN DEPTH, NOT A LIVE INVARIANT — say so rather than imply a duty no
 * test enforces. Established by mutation (review B7): making this a no-op is
 * INDISTINGUISHABLE — the mutant survives both the new tests and the whole
 * repetition/encode/lc0 subset, and for two independent reasons.
 *   1. Slots at or beyond hist_len are never read. cboard_fill_lc0_112_root walks
 *      back over min(hist_len, CBOARD_HISTORY_MAX) slots, every one of which
 *      cboard_push or from_board wrote. The tell is that hist_turn and
 *      hist_castling are left zeroed by the SAME memsets and nobody resets them.
 *   2. Even if such a slot were read, hist_ep == 0 (a1) cannot produce a
 *      candidate under bitboards_have_legal_ep's capturer-rank mask — a1 is
 *      attackable only from b2, which is on neither rank 4 nor rank 5 — so 0 is
 *      already indistinguishable from -1.
 * It is kept because reason 2 evaporates if that rank mask is ever relaxed, and
 * it costs 7 stores at construction. Do NOT write a test that "proves" the
 * callers are required: by construction there is no observation that separates
 * calling it from not, and a test asserting otherwise would be vacuous. */
static inline void cboard_reset_hist_ep(CBoard *b) {
    for (int i = 0; i < CBOARD_HISTORY_MAX; i++) b->hist_ep[i] = -1;
}

/* Normalize an untrusted ep-square to the CBoard invariant (-1 or 0..63).
 * Several paths shift by ep_square after only an `>= 0` check, so an
 * out-of-range value (e.g. 64..127 through from_raw's int8 cast) would be
 * undefined behaviour downstream; enforce the invariant at construction. */
static inline int8_t cboard_sanitize_ep(int sq) {
    return (int8_t)((sq >= 0 && sq < 64) ? sq : -1);
}

/* Reverse LUT: policy_index -> (from_sq, to_sq, promotion) in real coordinates.
 * Built once at init time. */
typedef struct { int8_t from_sq, to_sq, promotion; } PolicyMove;
static PolicyMove POLICY_LUT[2][4672];  /* [turn][index] */
static int policy_lut_ready = 0;

static void init_policy_lut(void) {
    if (policy_lut_ready) return;
    memset(POLICY_LUT, -1, sizeof(POLICY_LUT));

    for (int turn = 0; turn < 2; turn++) {
        for (int idx = 0; idx < 4672; idx++) {
            int from_o = idx / 73;
            int plane = idx % 73;
            int ff = sq_file(from_o), fr = sq_rank(from_o);
            int df, dr, promo = 0;

            if (plane >= 64) {
                /* Underpromotion */
                int rel = plane - 64;
                int piece_idx = rel / 3;
                int dir_idx = rel % 3;
                df = dir_idx - 1;
                dr = 1;
                switch (piece_idx) {
                    case 0: promo = 2; break; /* KNIGHT */
                    case 1: promo = 3; break; /* BISHOP */
                    case 2: promo = 4; break; /* ROOK */
                }
            } else if (plane >= 56) {
                /* Knight */
                int ki = plane - 56;
                df = KN_DF[ki];
                dr = KN_DR[ki];
            } else {
                /* Queen-like */
                int dir = plane / 7;
                int dist = plane % 7 + 1;
                df = Q_DF[dir] * dist;
                dr = Q_DR[dir] * dist;
            }

            int tf = ff + df, tr = fr + dr;
            if (tf < 0 || tf > 7 || tr < 0 || tr > 7) continue;

            int to_o = make_sq(tf, tr);
            int from_real = orient_sq(from_o, turn);
            int to_real = orient_sq(to_o, turn);

            /* Queen promotion: pawn reaching last rank */
            if (promo == 0) {
                int real_to_rank = sq_rank(to_real);
                if (real_to_rank == 0 || real_to_rank == 7)
                    promo = PROMO_MAYBE_QUEEN;  /* confirmed in cboard_push */
            }

            POLICY_LUT[turn][idx].from_sq = (int8_t)from_real;
            POLICY_LUT[turn][idx].to_sq = (int8_t)to_real;
            POLICY_LUT[turn][idx].promotion = (int8_t)promo;
        }
    }
    policy_lut_ready = 1;
}

/* Compute full Zobrist hash from CBoard state (for init, not incremental). */
static uint64_t cboard_compute_hash(const CBoard *b) {
    uint64_t h = 0;
    for (int color = 0; color < 2; color++) {
        for (int pt = 0; pt < 6; pt++) {
            uint64_t pieces = b->bb[pt] & b->occ[color];
            int sq;
            FOR_EACH_BIT(pieces, sq) {
                h ^= ZOBRIST_PIECE[piece_color_idx(pt, color)][sq];
            }
        }
    }
    if (b->turn == BLACK_C) h ^= ZOBRIST_TURN;
    h ^= ZOBRIST_CASTLING[b->castling & 0xF];
    /* EP excluded from this hash. It is the persisted POSITION-IDENTITY hash,
     * NOT the repetition key: selfplay resume records store it as pos_hash /
     * final_pos_hash and `selfplay/resume.py` re-derives and compares it, so
     * perturbing its value would make every in-flight record fail with
     * ResumeStateError("position_mismatch") — a reason NOT in
     * _PRESERVE_FILE_REASONS, so those games are DISCARDED, not deferred.
     *
     * ⚑ Do not use this hash to answer a question about equivalence:
     *   - "same legal move set"  -> cboard_transposition_key (pseudo-legal ep)
     *   - "same position for repetition" -> cboard_repetition_key (LEGAL ep)
     * Both are defined below as this hash XOR an ep term, so this value stays
     * bit-identical while they answer correctly. Keying a search transposition
     * table on this raw hash silently equates positions with different legal
     * moves (audit W1); keying REPETITION on it silently equates positions that
     * python-chess distinguishes (see cboard_repetition_key). */
    return h;
}

/* Is the ep right on this board actually exercisable — i.e. does a pawn of the
 * side to move stand on a square from which it attacks ep_square? A set
 * ep_square with no such pawn does not change the legal move set at all, which
 * is why both the FEN writer and the transposition key ignore it.
 *
 * Deliberately pseudo-legal: a pinned capturer answers 1 here and splits one
 * extra transposition entry. That costs a hit, never correctness. */
static inline int cboard_ep_capture_available(const CBoard *b) {
    if (b->ep_square < 0 || b->ep_square >= 64) return 0;
    uint64_t ep_bit = 1ULL << b->ep_square;
    uint64_t attackers;
    if (b->turn == WHITE_C) {
        /* White to move, so black just pushed; white pawns capture upward. */
        attackers = ((ep_bit >> 7) & 0xFEFEFEFEFEFEFEFEULL)
                  | ((ep_bit >> 9) & 0x7F7F7F7F7F7F7F7FULL);
    } else {
        attackers = ((ep_bit << 7) & 0x7F7F7F7F7F7F7F7FULL)
                  | ((ep_bit << 9) & 0xFEFEFEFEFEFEFEFEULL);
    }
    uint64_t our_pawns = b->bb[PAWN] & b->occ[b->turn];
    return (our_pawns & attackers) ? 1 : 0;
}

/* Key for search transposition tables: CBoard.hash plus the en-passant file
 * when an ep capture is actually available. Two boards sharing this key have
 * the same pieces, side to move, castling rights AND ep capture rights, so
 * they have the same legal move set.
 *
 * NOT sufficient for reusing a *value*: the halfmove clock and the repetition
 * history still do not enter, so draw-adjacent positions still share a key.
 * Consumers that copy a child action list must therefore also verify it
 * (see gss_prepare_batch). */
static inline uint64_t cboard_transposition_key(const CBoard *b) {
    uint64_t k = b->hash;
    if (cboard_ep_capture_available(b))
        k ^= ZOBRIST_EP[sq_file(b->ep_square)];
    return k;
}

/* Legality-EXACT en-passant availability: is there an ep capture that is
 * actually LEGAL (survives pin and check)?
 *
 * This is the predicate python-chess's repetition key uses. Verified against
 * python-chess 1.11.2:
 *
 *     def _transposition_key(self):
 *         return (..., self.ep_square if self.has_legal_en_passant() else None)
 *
 * and has_legal_en_passant() is `self.ep_square is not None and
 * any(self.generate_legal_ep())` — LEGAL, not pseudo-legal.
 *
 * ⚑ Deliberately NOT cboard_ep_capture_available: that one is pseudo-legal on
 * purpose (a pinned capturer answers 1). For a transposition table that is
 * harmless — it splits an entry, costing a hit, never correctness. For
 * REPETITION it is wrong in the other direction: it would split the key for a
 * position whose ep cannot legally be played, so a genuine repetition of that
 * position would go UNDETECTED (false negative — a real draw missed). We are
 * not trading one unsound answer for another.
 *
 * ⚑ The pseudo-legal half is a LITERAL transcription of python-chess's
 * generate_pseudo_legal_ep — target square empty, capturers taken from
 * BB_PAWN_ATTACKS[them][ep_square] AND restricted to BB_RANKS[4 if turn else 3]
 * — and deliberately nothing more. In particular it does NOT require the
 * captured pawn to be present, because python-chess does not either. Both
 * halves are load-bearing on caller-supplied (from_raw / from_board) ep
 * squares, which need not be consistent with the pieces:
 *
 *   - requiring the captured pawn made "4k3/8/8/4P3/8/8/8/4K3 w - d6 0 1"
 *     answer 0 where python-chess answers 1;
 *   - omitting the rank mask made "Nr1n4/3N4/6P1/k3P3/8/8/1PpnR1RK/8 w - c3"
 *     answer 1 (a b2 pawn "capturing" on c3) where python-chess answers 0.
 *
 * Both were real: each is a repetition-key divergence from the oracle this
 * whole key exists to mirror, and the first also made the C and Python paths
 * disagree with each other. test_c_matches_python_chess_on_inconsistent_ep_fens
 * pins the pair by name and
 * test_ep_predicate_oracle_sweep_over_inconsistent_ep_fields sweeps the class,
 * so this comment's "matches python-chess" cannot go stale silently.
 *
 * ⚑ It is NOT "matches python-chess exactly", and the old comment's version of
 * that sentence is what let the original defect live for months. THE CLAIM, AND
 * ITS TWO PRECONDITIONS — both narrower than an earlier revision of this comment
 * asserted, and each narrowed only after a reviewer executed the case it got
 * wrong:
 *
 *   we agree with python-chess on every position where (i) THE SIDE TO MOVE HAS
 *   EXACTLY ONE KING and (ii) the ep field is PAWN-CONSISTENT — captured square
 *   empty, or holding an enemy pawn.
 *
 * (i) is about the MOVER only, and that is measured, not assumed: a board where
 * the OPPONENT has no king agrees fine, because the king we look up is the one
 * whose exposure the capture could create.
 *
 * That domain contains everything reachable from a legal double push. Outside it
 * there are exactly two known divergence classes, and they run in OPPOSITE
 * directions, so neither may be summarised away:
 *
 *   - NO KING for the side to move → we return 0 where python-chess returns 1.
 *     `is_into_check` opens `king = self.king(self.turn); if king is None:
 *     return False`, i.e. an ep capture on a kingless board is LEGAL to it, so
 *     has_legal_en_passant() is true. We have no king to test for exposure and
 *     answer 0. ⚑ This is the key-MERGING direction — we DROP an ep term the
 *     oracle KEEPS, so "8/8/8/3pP3/8/8/8/8 w - d6 0 1" and the same board with
 *     no ep right share our repetition key while python-chess separates them.
 *     That is the invent-a-repetition direction this whole key exists to remove,
 *     and it is accepted ONLY because no searched board can be in that state.
 *     It is NOT accepted on the grounds of being rare.
 *
 *     ⚑ THAT PRECONDITION IS ENFORCED, NOT ASSUMED — and it is written this way
 *     because the previous revision of this paragraph asserted it instead, said
 *     "cannot arise from play_batch, selfplay, MCTS or UCI parsing — every one of
 *     those starts from a legal position", and was WRONG on the fourth of its
 *     four paths. chess.Board(fen) is a STRUCTURAL parse: it raises only on a
 *     malformed FEN string, so "4k3/8/8/8/8/8/8/8 w - - 0 1" parsed, passed
 *     _handle_position's only filter (`except ValueError`), and was searched;
 *     nothing under uci/ looked at status(). Enumerating paths is not checking
 *     them. The two boundaries that now make it true, by name:
 *       - selfplay/arena seeds — selfplay/opening.py::_fen_reject_reason rejects
 *         on the full board.is_valid(), so a seed list cannot carry one. This is
 *         why the TRAINING path never had the hole;
 *       - UCI — uci/engine.py::_unsearchable_king_reason, called from
 *         _handle_position, rejects onto the same "rejected FEN ...; using the
 *         start position" path a malformed FEN takes. It checks ONLY that each
 *         side has exactly one king that python-chess also sees as one —
 *         deliberately NOT status()==VALID, which would reject the
 *         weird-but-legal positions the EPD/puzzle/blind-spot drivers named on
 *         that fall-back path actually send.
 *     ⚑ The guard is symmetric (both colours) even though precondition (i) is
 *     about the MOVER: search pushes moves, so a root whose OPPONENT lacks a
 *     king produces children whose MOVER lacks one. Root-level agreement does
 *     not survive a ply.
 *     ⚑ The guarantee is at those ENTRY POINTS, not in this type. from_raw and
 *     from_board still build a kingless board on request, and eval/puzzles.py
 *     parses EPD FENs with no such check — an offline eval driver, off the
 *     training path, recorded as a follow-up rather than fixed here. A NEW path
 *     that turns an externally-supplied FEN into a searched CBoard needs the
 *     same guard, or this paragraph goes stale the way its predecessor did.
 *
 *     Note this also re-opens the C-vs-Python split: _check_repetitions asks
 *     python-chess and answers True on the same board.
 *   - NON-PAWN on the captured square → we return 1 where python-chess returns 0
 *     (~1 in 3,000 such positions). Also illegal by the rules of chess: the ep
 *     square asserts a pawn just double-pushed onto an occupied square. Here WE
 *     are the exact one — python-chess answers with pin_mask + _ep_skewered,
 *     which are sound only when a pawn is there — and the direction is
 *     key-SPLITTING, so it can only MISS a repetition, never invent one. Chasing
 *     it would mean transcribing those approximations into C, replacing a correct
 *     test with an incorrect one for positions that cannot occur.
 *
 * Two more ways to break precondition (i), both now quantified rather than
 * waved at, both MERGING, both excluded by the same UCI guard:
 *   - TWO kings for the mover: we take lsb64, python-chess's king() takes msb,
 *     and they pick different squares — "4k3/8/7K/r2pP3/8/8/8/K7 w - d6 0 1"
 *     (msb h6 is safe, lsb a1 is attacked down the a-file after the capture)
 *     reads python-chess 1, us 0;
 *   - the king square carries a '~' PROMOTED marker: Board.king() masks with
 *     `& ~promoted` and returns None while the raw king bitboard has one
 *     bit, so python-chess short-circuits exactly as in the kingless case —
 *     "4k3/8/8/K~2pP2r/8/8/8/8 w - d6 0 1" reads python-chess 1, us 0. This is
 *     why the guard tests Board.king(color) is None and not only a popcount.
 *
 * ⚑ The measurement that used to back this sentence COULD NOT SEE THE KINGLESS
 * CASE: its generator always placed both kings, so 720,000 samples reporting
 * zero merges was a gate that structurally could not fail — this repo's own
 * signature defect, one level down, inside the fix for it. The sweep now emits
 * kingless boards deliberately and classifies them, and
 * test_kingless_board_is_a_known_accepted_divergence pins the exact FEN, so
 * whichever side of the boundary a future change lands on, a test moves with it.
 *
 * Cost is paid only when ep_square is set (rare), hence the early-out. The
 * occupancy is simulated fully — capturer removed from its origin, captured
 * pawn removed from its square, capturer placed on ep_square — so the
 * horizontal "ep skewer" (both pawns leaving one rank to expose a rook/queen
 * on the king) is handled by construction rather than by a special case. */
static inline int bitboards_have_legal_ep(const uint64_t bb[6],
                                          const uint64_t occ[2],
                                          int turn, int ep_square) {
    if (ep_square < 0 || ep_square >= 64) return 0;   /* the common path */
    int us = turn, them = 1 - us;
    uint64_t us_kings = bb[KING] & occ[us];
    /* ⚑ DELIBERATE, DOCUMENTED DIVERGENCE — see the header comment's precondition
     * (i). python-chess's is_into_check() returns False when there is no king,
     * making the ep capture LEGAL to it; we have nothing to test for exposure and
     * answer 0. That is the key-MERGING direction, accepted only because the
     * entry points ENFORCE a king (selfplay/opening.py's is_valid(); uci/
     * engine.py::_unsearchable_king_reason) — not because it is rare, and not
     * on the say-so of an enumeration of callers. Moving this to `return 1` would
     * match the oracle and is a defensible alternative — but do it deliberately,
     * and update test_kingless_board_is_a_known_accepted_divergence, which asserts
     * today's answer BY NAME so the boundary cannot move silently. */
    if (!us_kings) return 0;

    uint64_t ep_bit = 1ULL << ep_square;
    /* The ep TARGET must be empty, mirroring python-chess's own guard in
     * generate_pseudo_legal_ep:
     *
     *     if BB_SQUARES[self.ep_square] & self.occupied:
     *         return
     *
     * Unreachable from a real double push, but from_raw / from_board accept a
     * caller-supplied ep_square: python-chess keeps Board.ep_square set on a
     * hand-written FEN whose target is occupied (its fen() prints "-" while the
     * attribute stays 43), so from_board reads it straight through. Without
     * this the simulated occupancy below would also be wrong — the occupying
     * piece stays in `all`, so the capturer would "land on" it while it still
     * blocks rays for the king-safety test. */
    if ((occ[0] | occ[1]) & ep_bit) return 0;
    uint64_t attackers;
    int captured_sq;
    if (us == WHITE_C) {
        /* White to move, so black just pushed; white pawns capture upward. */
        attackers = ((ep_bit >> 7) & 0xFEFEFEFEFEFEFEFEULL)
                  | ((ep_bit >> 9) & 0x7F7F7F7F7F7F7F7FULL);
        captured_sq = ep_square - 8;
    } else {
        attackers = ((ep_bit << 7) & 0x7F7F7F7F7F7F7F7FULL)
                  | ((ep_bit << 9) & 0xFEFEFEFEFEFEFEFEULL);
        captured_sq = ep_square + 8;
    }
    /* python-chess's capturer rank mask, BB_RANKS[4 if self.turn else 3]:
     * White captures en passant only from rank 5, Black only from rank 4.
     * `attackers` alone constrains the capturer to the rank below/above
     * ep_square, which is NOT the same thing when ep_square itself is
     * inconsistent — without this, ep_square c3 with White to move let a b2
     * pawn "capture" on c3. */
    uint64_t capturer_rank = (us == WHITE_C) ? 0x000000FF00000000ULL   /* rank 5 */
                                             : 0x00000000FF000000ULL;  /* rank 4 */
    uint64_t candidates = (bb[PAWN] & occ[us]) & attackers & capturer_rank;
    if (!candidates) return 0;
    if (captured_sq < 0 || captured_sq >= 64) return 0;

    /* NOT checked: that a pawn actually stands on captured_sq. python-chess's
     * generate_pseudo_legal_ep does not check it, so neither may we (see the
     * header comment). The simulated occupancy below tolerates an empty or
     * differently-occupied square by construction — `& ~cap_bit` is a no-op on
     * an empty one — and mirrors _ep_skewered, which clears last_double the
     * same way regardless of what is on it. */
    uint64_t cap_bit = 1ULL << captured_sq;

    int king_sq = lsb64(us_kings);
    uint64_t all = occ[0] | occ[1];
    /* Every enemy piece set is cleared of cap_bit, not just the pawns. The
     * captured square is removed from occ_after below, so leaving a piece there
     * in the ATTACKER sets would make this simulation disagree with its own
     * occupancy: a knight/rook/king on captured_sq would be gone as a blocker
     * yet still radiate attacks. Reachable positions always hold the
     * just-double-pushed PAWN there, so this is a no-op in play; it matters only
     * for caller-supplied ep squares, where it is the difference between
     * matching python-chess (which reaches such an ep through
     * _generate_evasions' "capture the checking piece en passant" branch) and
     * calling a legal ep illegal. */
    uint64_t them_after = occ[them] & ~cap_bit;
    uint64_t them_pawns_after = bb[PAWN]   & them_after;
    uint64_t them_knights     = bb[KNIGHT] & them_after;
    uint64_t them_bishops     = bb[BISHOP] & them_after;
    uint64_t them_rooks       = bb[ROOK]   & them_after;
    uint64_t them_queens      = bb[QUEEN]  & them_after;
    uint64_t them_kings       = bb[KING]   & them_after;

    int from_sq;
    FOR_EACH_BIT(candidates, from_sq) {
        /* The mover is a pawn, so king_sq is unaffected by the move itself. */
        uint64_t occ_after = (all & ~(1ULL << from_sq) & ~cap_bit) | ep_bit;
        if (!is_attacked_by(king_sq, occ_after,
                            them_pawns_after, them_knights, them_bishops,
                            them_rooks, them_queens, them_kings, them))
            return 1;
    }
    return 0;
}

static inline int cboard_has_legal_ep(const CBoard *b) {
    return bitboards_have_legal_ep(b->bb, b->occ, b->turn, b->ep_square);
}

/* Key for REPETITION detection: cboard_compute_hash plus the en-passant file
 * when an ep capture is actually LEGAL. This is the C mirror of python-chess's
 * Board._transposition_key(), which is what is_repetition() compares.
 *
 * FIDE and python-chess agree that an UNUSABLE ep right is irrelevant — that
 * much the pre-fix comment had right. Where it went wrong was concluding that
 * ep may therefore be dropped ENTIRELY: python-chess drops it only when
 * has_legal_en_passant() is false and KEEPS it otherwise. Dropping it
 * unconditionally makes a position with a legal ep compare EQUAL to the same
 * pieces/turn/castling without it — a false repetition. That is not a cosmetic
 * plane bug: cboard_search_terminal (mcts/_mcts_tree.c) answers a repetition
 * with SOLVED_DRAW, and tree_resolve_from_children lets one drawn child turn a
 * proven-LOST node into a proven-DRAWN one. SOLVED is terminal, so more visits
 * never correct it and the error propagates upward — unsound, and directional
 * (positions only ever look better than they are).
 *
 * XOR-on-top of b->hash rather than folded into it, so the persisted
 * position-identity hash keeps its value (see cboard_compute_hash). */
static inline uint64_t cboard_repetition_key(const CBoard *b) {
    uint64_t k = b->hash;
    if (cboard_has_legal_ep(b))
        k ^= ZOBRIST_EP[sq_file(b->ep_square)];
    return k;
}

/* Find which piece type is on a square (for 'us' side) */
static inline int piece_type_at(const CBoard *b, int sq) {
    uint64_t bit = sq_bit(sq);
    if (b->bb[PAWN]   & bit) return PAWN;
    if (b->bb[KNIGHT] & bit) return KNIGHT;
    if (b->bb[BISHOP] & bit) return BISHOP;
    if (b->bb[ROOK]   & bit) return ROOK;
    if (b->bb[QUEEN]  & bit) return QUEEN;
    if (b->bb[KING]   & bit) return KING;
    return -1;
}

/* Would a legal (from_sq,to_sq) move reset the halfmove clock — i.e. is it a
 * pawn move or a capture? Mirrors the is_irreversible test inside cboard_push so
 * the fifty-move claim check can avoid copying + pushing each candidate reply. */
static inline int cboard_move_is_zeroing(const CBoard *b, int from_sq, int to_sq) {
    if (piece_type_at(b, from_sq) == PAWN) return 1;        /* pawn move (incl. EP) */
    if (b->occ[1 - b->turn] & sq_bit(to_sq)) return 1;      /* capture */
    return 0;
}

/* Gated candidate flag (default off → byte-identical encoding). When enabled,
 * cboard_fill_lc0_112_root reads per-slot repetition flags recorded at push
 * time instead of reconstructing them from the hash_stack, which is cleared on
 * irreversible moves and so under-reports repetitions older than the kept
 * window. Set per-process via each module's ``set_history_rep_fix`` hook; every
 * .so including this header keeps its own copy. */
static int g_history_rep_fix = 0;

/* Defined further down; cboard_push records repetition status before mutating. */
static int cboard_is_repetition(const CBoard *b);

static void cboard_push(CBoard *b, int from_sq, int to_sq, int promotion) {
    if (from_sq < 0 || from_sq > 63 || to_sq < 0 || to_sq > 63) {
        /* Off-board square — e.g. the -1 sentinel from an unused POLICY_LUT
         * slot. sq_bit's shift on it is undefined behaviour, so bail before
         * touching either square. The Python boundary (PyCBoard_push_index)
         * validates and raises before reaching here; this is the C-level UB
         * floor for any other caller. */
        return;
    }
    int us = b->turn;
    int them = 1 - us;
    uint64_t from_bit = sq_bit(from_sq);
    uint64_t to_bit = sq_bit(to_sq);

    int moving_piece = piece_type_at(b, from_sq);
    if (moving_piece < 0) {
        /* No piece on the source square — a malformed push (e.g. an index
         * decoded against the wrong position). Indexing ZOBRIST_PIECE with
         * the -1 sentinel is undefined behaviour, so bail out instead. */
        return;
    }
    int is_capture = (b->occ[them] & to_bit) != 0;
    int is_ep = (moving_piece == PAWN && to_sq == b->ep_square && b->ep_square >= 0);
    int is_pawn_move = (moving_piece == PAWN);
    int is_irreversible = is_pawn_move || is_capture;

    /* The pre-move position's repetition key, computed ONCE. Both the history
     * slot and the hash_stack record this position, and the legality-exact ep
     * test behind it is not free when ep_square is set.
     *
     * ⚑ DECISION RECORDED (review B9), so it is explicit rather than absent: do
     * NOT cache the POST-move key on the board to save cboard_is_repetition's
     * recomputation at every cboard_search_terminal check. It would erase the
     * C-path cost, and it is exactly the write-here/read-there construct whose
     * failures this PR exists to fix — a stored key that silently stops matching
     * the rule that produced it is this defect, one layer up. The ep_square < 0
     * early-out already bounds the work to the plies carrying an ep square:
     * measured 4.6% on uniform play, 10.3% on a double-push-biased corpus, of
     * which only ~0.3% carry a LEGAL ep and reach the attack scans. */
    uint64_t rep_key = cboard_repetition_key(b);

    /* --- Save current position to history circular buffer --- */
    {
        int slot = b->hist_head;
        memcpy(b->hist_bb[slot], b->bb, 6 * sizeof(uint64_t));
        memcpy(b->hist_occ[slot], b->occ, 2 * sizeof(uint64_t));
        b->hist_turn[slot] = b->turn;
        b->hist_castling[slot] = b->castling;
        /* Record whether the position now entering history is a repetition.
         * The live hash_stack is still valid for this position (the current
         * hash is pushed below), so the look-back is complete here — unlike
         * the encoder's after-the-fact reconstruction. Gated: the flags are
         * only read when the fix is enabled, and the flag is applied before
         * boards are constructed (rep_fix.apply at batch start / model
         * build), so paying the hash_stack scan with the flag off — every
         * push of every MCTS tree replay — would be pure waste. The slot is
         * still zeroed so a later flag flip can't read a stale value. */
        int was_rep = 0;
        if (g_history_rep_fix) {
            was_rep = cboard_is_repetition(b);
            /* hash_stack saturation supplement: the stack stops appending
             * when full (reachable only by pushing past a claimable 50-move
             * draw, e.g. a GUI-replayed shuffle), so also check the kept
             * window's recorded hashes. Repeats farther back than the window
             * inside a saturated run stay undetected — bounded, and confined
             * to positions already drawn by rule. */
            for (int k = 0; !was_rep && k < b->hist_len; k++) {
                if (b->hist_hash[k] == rep_key) was_rep = 1;
            }
        }
        b->hist_was_rep[slot] = (int8_t)was_rep;
        /* Repetition key, not the identity hash: hist_hash is only ever
         * compared against other repetition keys (here, and in the encoder's
         * current-position supplement), so it must be on the same ruler. */
        b->hist_hash[slot] = rep_key;
        b->hist_ep[slot] = b->ep_square;
        b->hist_head = (slot + 1) % CBOARD_HISTORY_MAX;
        if (b->hist_len < CBOARD_HISTORY_MAX)
            b->hist_len++;
    }

    /* --- Push current hash onto hash_stack for repetition detection --- */
    if (is_irreversible) {
        /* Irreversible move: clear the hash stack (no repetition possible across these) */
        b->hash_stack_len = 0;
    } else {
        /* The ep-aware repetition key, matching python-chess's
         * _transposition_key(); b->hash alone would equate a legal-ep position
         * with the same pieces/turn/castling without the ep right. */
        if (b->hash_stack_len < CBOARD_HASH_STACK_MAX)
            b->hash_stack[b->hash_stack_len++] = rep_key;
    }

    /* --- Save old state for incremental hash update --- */
    uint8_t old_castling = b->castling;
    uint64_t h = b->hash;

    /* --- Halfmove clock --- */
    if (is_irreversible)
        b->halfmove_clock = 0;
    else
        b->halfmove_clock++;

    /* --- Find captured piece type for hash update --- */
    int captured_piece = -1;
    int capture_sq = -1;
    if (is_capture) {
        capture_sq = to_sq;
        for (int p = 0; p < 6; p++) {
            if (b->bb[p] & to_bit) { captured_piece = p; break; }
        }
    } else if (is_ep) {
        capture_sq = make_sq(sq_file(to_sq), sq_rank(from_sq));
        captured_piece = PAWN;
    }

    /* --- Hash: remove captured piece --- */
    if (captured_piece >= 0 && capture_sq >= 0) {
        h ^= ZOBRIST_PIECE[piece_color_idx(captured_piece, them)][capture_sq];
    }

    /* --- Remove captured piece from bitboards --- */
    if (is_capture) {
        b->occ[them] &= ~to_bit;
        for (int p = 0; p < 6; p++)
            b->bb[p] &= ~to_bit;
    }

    /* En passant capture: remove the captured pawn */
    if (is_ep) {
        uint64_t cap_bit = sq_bit(capture_sq);
        b->bb[PAWN] &= ~cap_bit;
        b->occ[them] &= ~cap_bit;
    }

    /* --- Hash: remove moving piece from source, add to dest --- */
    h ^= ZOBRIST_PIECE[piece_color_idx(moving_piece, us)][from_sq];

    /* Move the piece */
    b->bb[moving_piece] = (b->bb[moving_piece] & ~from_bit) | to_bit;
    b->occ[us] = (b->occ[us] & ~from_bit) | to_bit;

    /* Promotion: replace pawn with promoted piece */
    int final_piece = moving_piece;
    if (promotion > 0 && promotion != PROMO_MAYBE_QUEEN && is_pawn_move) {
        b->bb[PAWN] &= ~to_bit;
        b->bb[promotion - 1] |= to_bit;
        final_piece = promotion - 1;
    } else if (promotion == PROMO_MAYBE_QUEEN && is_pawn_move) {
        int to_rank = sq_rank(to_sq);
        if (to_rank == 0 || to_rank == 7) {
            b->bb[PAWN] &= ~to_bit;
            b->bb[QUEEN] |= to_bit;
            final_piece = QUEEN;
        }
    }

    /* --- Hash: add piece at destination (may be promoted piece) --- */
    h ^= ZOBRIST_PIECE[piece_color_idx(final_piece, us)][to_sq];

    /* Castling: move the rook */
    if (moving_piece == KING) {
        int diff = to_sq - from_sq;
        if (diff == 2) { /* Kingside */
            int rook_from = from_sq + 3;
            int rook_to = from_sq + 1;
            uint64_t rf = sq_bit(rook_from), rt = sq_bit(rook_to);
            b->bb[ROOK] = (b->bb[ROOK] & ~rf) | rt;
            b->occ[us] = (b->occ[us] & ~rf) | rt;
            h ^= ZOBRIST_PIECE[piece_color_idx(ROOK, us)][rook_from];
            h ^= ZOBRIST_PIECE[piece_color_idx(ROOK, us)][rook_to];
        } else if (diff == -2) { /* Queenside */
            int rook_from = from_sq - 4;
            int rook_to = from_sq - 1;
            uint64_t rf = sq_bit(rook_from), rt = sq_bit(rook_to);
            b->bb[ROOK] = (b->bb[ROOK] & ~rf) | rt;
            b->occ[us] = (b->occ[us] & ~rf) | rt;
            h ^= ZOBRIST_PIECE[piece_color_idx(ROOK, us)][rook_from];
            h ^= ZOBRIST_PIECE[piece_color_idx(ROOK, us)][rook_to];
        }
        if (us == WHITE_C) b->castling &= ~(WK_CASTLE | WQ_CASTLE);
        else b->castling &= ~(BK_CASTLE | BQ_CASTLE);
    }

    /* Update castling rights: rook moves or rook captured */
    if (from_sq == 0  || to_sq == 0)  b->castling &= ~WQ_CASTLE;
    if (from_sq == 7  || to_sq == 7)  b->castling &= ~WK_CASTLE;
    if (from_sq == 56 || to_sq == 56) b->castling &= ~BQ_CASTLE;
    if (from_sq == 63 || to_sq == 63) b->castling &= ~BK_CASTLE;

    /* --- Hash: update castling --- */
    h ^= ZOBRIST_CASTLING[old_castling & 0xF];
    h ^= ZOBRIST_CASTLING[b->castling & 0xF];

    /* EP square is stored for encoding and excluded from b->hash, which is the
     * persisted position-identity hash. The ep-dependent keys are layered on
     * top of it: cboard_transposition_key (pseudo-legal ep) and
     * cboard_repetition_key (LEGAL ep, matching python-chess). */
    b->ep_square = -1;
    if (is_pawn_move) {
        int diff = to_sq - from_sq;
        if (diff == 16 || diff == -16)
            b->ep_square = (int8_t)(from_sq + diff / 2);
    }
    /* --- Hash: flip turn (do this before EP legal check since turn affects it) --- */
    h ^= ZOBRIST_TURN;
    b->turn = (int8_t)them;

    b->hash = h;
    b->ply++;
}

/* Push by policy index -- decode index to move, then push */
static void cboard_push_index(CBoard *b, int policy_index) {
    PolicyMove pm = POLICY_LUT[b->turn][policy_index];
    cboard_push(b, pm.from_sq, pm.to_sq, pm.promotion);
}

/* Build BoardState from CBoard for legal move generation */
static void cboard_to_boardstate(const CBoard *b, BoardState *bs) {
    int us = b->turn, them = 1 - us;
    bs->us_pawns   = b->bb[PAWN]   & b->occ[us];
    bs->us_knights = b->bb[KNIGHT] & b->occ[us];
    bs->us_bishops = b->bb[BISHOP] & b->occ[us];
    bs->us_rooks   = b->bb[ROOK]   & b->occ[us];
    bs->us_queens  = b->bb[QUEEN]  & b->occ[us];
    bs->us_kings   = b->bb[KING]   & b->occ[us];
    bs->them_pawns   = b->bb[PAWN]   & b->occ[them];
    bs->them_knights = b->bb[KNIGHT] & b->occ[them];
    bs->them_bishops = b->bb[BISHOP] & b->occ[them];
    bs->them_rooks   = b->bb[ROOK]   & b->occ[them];
    bs->them_queens  = b->bb[QUEEN]  & b->occ[them];
    bs->them_kings   = b->bb[KING]   & b->occ[them];
    bs->us_occ = b->occ[us];
    bs->them_occ = b->occ[them];
    bs->all_occ = b->occ[0] | b->occ[1];
    bs->turn = us;
    bs->ep_square = b->ep_square;
    if (bs->us_kings) bs->king_sq = lsb64(bs->us_kings);
    else bs->king_sq = -1;

    /* Castling from CBoard perspective (us/them) */
    if (us == WHITE_C) {
        bs->us_castle_k   = (b->castling & WK_CASTLE) ? 1 : 0;
        bs->us_castle_q   = (b->castling & WQ_CASTLE) ? 1 : 0;
        bs->them_castle_k = (b->castling & BK_CASTLE) ? 1 : 0;
        bs->them_castle_q = (b->castling & BQ_CASTLE) ? 1 : 0;
    } else {
        bs->us_castle_k   = (b->castling & BK_CASTLE) ? 1 : 0;
        bs->us_castle_q   = (b->castling & BQ_CASTLE) ? 1 : 0;
        bs->them_castle_k = (b->castling & WK_CASTLE) ? 1 : 0;
        bs->them_castle_q = (b->castling & WQ_CASTLE) ? 1 : 0;
    }
}

static int cboard_has_legal_moves(const CBoard *b) {
    BoardState bs;
    cboard_to_boardstate(b, &bs);
    if (bs.king_sq < 0) return 0;
    int indices[256];
    return generate_legal_move_indices(&bs, indices) > 0;
}

/* Fill indices[] with legal policy indices for `b`. Returns count, or 0 if
 * the side to move has no king (malformed position). `sorted != 0` produces
 * ascending indices. The indices buffer must have room for 256 entries. */
static inline int cboard_legal_move_indices(const CBoard *b, int *indices, int sorted) {
    BoardState bs;
    cboard_to_boardstate(b, &bs);
    if (bs.king_sq < 0) return 0;
    return sorted ? generate_legal_move_indices_sorted(&bs, indices)
                  : generate_legal_move_indices(&bs, indices);
}

/* Check for insufficient material (KK, KNK, KBK, KBKB same-color) */
static int cboard_insufficient_material(const CBoard *b) {
    if (b->bb[PAWN] || b->bb[ROOK] || b->bb[QUEEN]) return 0;
    uint64_t all = b->occ[0] | b->occ[1];
    int total = popcount64(all);
    if (total <= 2) return 1;  /* K vs K */
    if (total == 3 && (b->bb[KNIGHT] || b->bb[BISHOP])) return 1;  /* KN/KB vs K */
    if (total == 4 && popcount64(b->bb[BISHOP]) == 2) {
        /* K+B vs K+B: draw only if same-color squares */
        uint64_t light = 0x55AA55AA55AA55AAULL;
        int on_light = popcount64(b->bb[BISHOP] & light);
        if (on_light == 0 || on_light == 2) return 1;
    }
    return 0;
}

/* Fast repetition: any prior occurrence (2-fold). Good for search pruning.
 * Keyed on cboard_repetition_key — b->hash XOR the ep file when an ep capture
 * is LEGAL — which is the C mirror of python-chess's Board._transposition_key(),
 * the key its is_repetition() compares. Using the raw b->hash here (as this did
 * before) reports a false repetition whenever a legal-ep position shares its
 * pieces/turn/castling with an earlier one. */
static int cboard_is_repetition(const CBoard *b) {
    uint64_t h = cboard_repetition_key(b);
    for (int i = 0; i < b->hash_stack_len; i++) {
        if (b->hash_stack[i] == h) return 1;
    }
    return 0;
}

/* Strict repetition: 3-fold (current + 2 prior). Matches FIDE/python-chess rules.
 * Same key as cboard_is_repetition. */
static int cboard_is_threefold_repetition(const CBoard *b) {
    uint64_t h = cboard_repetition_key(b);
    int count = 0;
    for (int i = 0; i < b->hash_stack_len; i++) {
        if (b->hash_stack[i] == h) {
            count++;
            if (count >= 2) return 1;  /* current + 2 prior = 3-fold */
        }
    }
    return 0;
}

/* Check if game is over: no legal moves, 50-move rule, 3-fold repetition, insufficient material.
 * Uses strict 3-fold (not 2-fold) to match FIDE/python-chess claim_draw rules. */
static int cboard_is_game_over(const CBoard *b) {
    if (b->halfmove_clock >= 100) return 1; /* 50-move rule (claim draw) */
    if (cboard_is_threefold_repetition(b)) return 1;  /* 3-fold repetition */
    if (cboard_insufficient_material(b)) return 1;
    return !cboard_has_legal_moves(b);
}

/* Is the side-to-move's king in check? */
static int cboard_in_check(const CBoard *b) {
    int us = b->turn, them = 1 - us;
    uint64_t us_kings = b->bb[KING] & b->occ[us];
    if (!us_kings) return 0;
    int king_sq = lsb64(us_kings);
    uint64_t all = b->occ[0] | b->occ[1];
    return is_attacked_by(king_sq, all,
        b->bb[PAWN] & b->occ[them], b->bb[KNIGHT] & b->occ[them],
        b->bb[BISHOP] & b->occ[them], b->bb[ROOK] & b->occ[them],
        b->bb[QUEEN] & b->occ[them], b->bb[KING] & b->occ[them],
        them);
}

static int cboard_is_checkmate(const CBoard *b) {
    return !cboard_has_legal_moves(b) && cboard_in_check(b);
}

static int cboard_is_stalemate(const CBoard *b) {
    return !cboard_has_legal_moves(b) && !cboard_in_check(b);
}

/* Terminal value from side-to-move perspective.
 * Safe to call on any position — returns 0.0 if game is not over. */
static float cboard_terminal_value(const CBoard *b) {
    if (!cboard_is_game_over(b)) return 0.0f;
    return cboard_is_checkmate(b) ? -1.0f : 0.0f;
}

/* Encode piece planes for current position only (no history).
 * Writes 12 planes into out[0..11][8][8]. */
static void cboard_encode_piece_planes(const CBoard *b, float *out) {
    int us = b->turn, them = 1 - us;
    int is_white = (us == WHITE_C);
    uint64_t bbs[12];
    bbs[0]  = b->bb[PAWN]   & b->occ[us];
    bbs[1]  = b->bb[KNIGHT] & b->occ[us];
    bbs[2]  = b->bb[BISHOP] & b->occ[us];
    bbs[3]  = b->bb[ROOK]   & b->occ[us];
    bbs[4]  = b->bb[QUEEN]  & b->occ[us];
    bbs[5]  = b->bb[KING]   & b->occ[us];
    bbs[6]  = b->bb[PAWN]   & b->occ[them];
    bbs[7]  = b->bb[KNIGHT] & b->occ[them];
    bbs[8]  = b->bb[BISHOP] & b->occ[them];
    bbs[9]  = b->bb[ROOK]   & b->occ[them];
    bbs[10] = b->bb[QUEEN]  & b->occ[them];
    bbs[11] = b->bb[KING]   & b->occ[them];
    for (int p = 0; p < 12; p++) {
        if (is_white) bitboard_to_plane_white(bbs[p], out + p * 64);
        else          bitboard_to_plane_black(bbs[p], out + p * 64);
    }
}

/* Encode piece planes for a history entry into out[0..11][8][8] */
static void cboard_encode_hist_planes(const uint64_t hist_bb[6],
                                       const uint64_t hist_occ[2],
                                       int hist_turn, float *out) {
    int us = hist_turn, them = 1 - us;
    int is_white = (us == WHITE_C);
    uint64_t bbs[12];
    bbs[0]  = hist_bb[PAWN]   & hist_occ[us];
    bbs[1]  = hist_bb[KNIGHT] & hist_occ[us];
    bbs[2]  = hist_bb[BISHOP] & hist_occ[us];
    bbs[3]  = hist_bb[ROOK]   & hist_occ[us];
    bbs[4]  = hist_bb[QUEEN]  & hist_occ[us];
    bbs[5]  = hist_bb[KING]   & hist_occ[us];
    bbs[6]  = hist_bb[PAWN]   & hist_occ[them];
    bbs[7]  = hist_bb[KNIGHT] & hist_occ[them];
    bbs[8]  = hist_bb[BISHOP] & hist_occ[them];
    bbs[9]  = hist_bb[ROOK]   & hist_occ[them];
    bbs[10] = hist_bb[QUEEN]  & hist_occ[them];
    bbs[11] = hist_bb[KING]   & hist_occ[them];
    for (int p = 0; p < 12; p++) {
        if (is_white) bitboard_to_plane_white(bbs[p], out + p * 64);
        else          bitboard_to_plane_black(bbs[p], out + p * 64);
    }
}

/* Encode history entry from the root side-to-move POV.
 * This matches LC0-style history: each slot keeps a stable us/them and board
 * orientation instead of reinterpreting the old position by its own turn. */
static void cboard_encode_hist_planes_root(const uint64_t hist_bb[6],
                                           const uint64_t hist_occ[2],
                                           int root_turn, float *out) {
    int us = root_turn, them = 1 - us;
    int is_white = (us == WHITE_C);
    uint64_t bbs[12];
    bbs[0]  = hist_bb[PAWN]   & hist_occ[us];
    bbs[1]  = hist_bb[KNIGHT] & hist_occ[us];
    bbs[2]  = hist_bb[BISHOP] & hist_occ[us];
    bbs[3]  = hist_bb[ROOK]   & hist_occ[us];
    bbs[4]  = hist_bb[QUEEN]  & hist_occ[us];
    bbs[5]  = hist_bb[KING]   & hist_occ[us];
    bbs[6]  = hist_bb[PAWN]   & hist_occ[them];
    bbs[7]  = hist_bb[KNIGHT] & hist_occ[them];
    bbs[8]  = hist_bb[BISHOP] & hist_occ[them];
    bbs[9]  = hist_bb[ROOK]   & hist_occ[them];
    bbs[10] = hist_bb[QUEEN]  & hist_occ[them];
    bbs[11] = hist_bb[KING]   & hist_occ[them];
    for (int p = 0; p < 12; p++) {
        if (is_white) bitboard_to_plane_white(bbs[p], out + p * 64);
        else          bitboard_to_plane_black(bbs[p], out + p * 64);
    }
}

/* Compute a REPETITION KEY from loose bitboards + meta — the free-standing
 * twin of cboard_repetition_key, for callers that hold a history snapshot
 * rather than a CBoard.
 *
 * ⚑ ep_square is REQUIRED and must be the snapshot's own. It mirrors
 * python-chess's Board._transposition_key(), which keeps ep_square exactly when
 * has_legal_en_passant() is true; a caller that cannot supply the snapshot's ep
 * square must pass -1 and accept that its keys will not match a CBoard's for
 * legal-ep positions. This used to omit ep unconditionally to "match"
 * _check_repetitions — which was itself wrong the same way, and has been fixed
 * alongside this (encoding/lc0.py::_check_repetitions). */
static uint64_t cboard_hist_hash(const uint64_t hist_bb[6],
                                  const uint64_t hist_occ[2],
                                  int hist_turn, uint8_t castling,
                                  int hist_ep_square) {
    uint64_t h = 0;
    for (int color = 0; color < 2; color++) {
        for (int pt = 0; pt < 6; pt++) {
            uint64_t pieces = hist_bb[pt] & hist_occ[color];
            int sq;
            FOR_EACH_BIT(pieces, sq) {
                h ^= ZOBRIST_PIECE[piece_color_idx(pt, color)][sq];
            }
        }
    }
    if (hist_turn == BLACK_C) h ^= ZOBRIST_TURN;
    h ^= ZOBRIST_CASTLING[castling & 0xF];
    if (bitboards_have_legal_ep(hist_bb, hist_occ, hist_turn, hist_ep_square))
        h ^= ZOBRIST_EP[sq_file(hist_ep_square)];
    return h;
}

static void cboard_fill_lc0_112(const CBoard *b, float * restrict out) {
    /* Plane 0-11: piece planes for current position */
    cboard_encode_piece_planes(b, out);

    /* Planes 12-95: history positions (7 previous positions x 12 planes each)
     * History is stored in a circular buffer; index 0 = most recent. */
    for (int hi = 0; hi < b->hist_len && hi < CBOARD_HISTORY_MAX; hi++) {
        int idx = (b->hist_head - 1 - hi + CBOARD_HISTORY_MAX) % CBOARD_HISTORY_MAX;
        float *dest = out + (hi + 1) * 12 * 64;
        cboard_encode_hist_planes(b->hist_bb[idx], b->hist_occ[idx],
                                  b->hist_turn[idx], dest);
    }

    /* Castling (us-K, us-Q, them-K, them-Q) */
    int us = b->turn;
    int us_k, us_q, them_k, them_q;
    if (us == WHITE_C) {
        us_k = b->castling & WK_CASTLE; us_q = b->castling & WQ_CASTLE;
        them_k = b->castling & BK_CASTLE; them_q = b->castling & BQ_CASTLE;
    } else {
        us_k = b->castling & BK_CASTLE; us_q = b->castling & BQ_CASTLE;
        them_k = b->castling & WK_CASTLE; them_q = b->castling & WQ_CASTLE;
    }
    if (us_k)   for (int i = 0; i < 64; i++) out[96*64 + i] = 1.0f;
    if (us_q)   for (int i = 0; i < 64; i++) out[97*64 + i] = 1.0f;
    if (them_k) for (int i = 0; i < 64; i++) out[98*64 + i] = 1.0f;
    if (them_q) for (int i = 0; i < 64; i++) out[99*64 + i] = 1.0f;

    /* EP file */
    if (b->ep_square >= 0) {
        int ep_file = sq_file(b->ep_square);
        for (int r = 0; r < 8; r++)
            out[100*64 + r*8 + ep_file] = 1.0f;
    }

    /* Color to move (always 1) */
    for (int i = 0; i < 64; i++) out[101*64 + i] = 1.0f;

    /* Rule50 */
    float r50 = (float)(b->halfmove_clock < 100 ? b->halfmove_clock : 100) / 100.0f;
    for (int i = 0; i < 64; i++) out[102*64 + i] = r50;

    /* Repetition planes 103-110: check each history position against all
     * positions before it (using hash_stack + history hashes).
     * Plane 103 + i is set to 1.0 if history position i is a repetition.
     * We approximate using the Zobrist hash of the current position and
     * the hash_stack. For simplicity, set plane 103+0 if current position
     * is a repetition, and leave others at 0 (history repetitions are rare
     * and minimally impact model accuracy). */
    if (cboard_is_repetition(b)) {
        for (int i = 0; i < 64; i++) out[103*64 + i] = 1.0f;
    }

    /* All-ones bias */
    for (int i = 0; i < 64; i++) out[111*64 + i] = 1.0f;
}

/* Set the repetition plane for history step hi (1-based from the current
 * position). Shared by both encoder paths so the plane math cannot drift. */
static inline void cboard_set_hist_rep_plane(float * restrict out, int hi) {
    int plane = (hi + 1) * 13 + 12;
    for (int i = 0; i < 64; i++) out[plane*64 + i] = 1.0f;
}

static void cboard_fill_lc0_112_root(const CBoard *b, float * restrict out) {
    /* Planes 0-103: 8 history slots of 12 piece planes + 1 repetition plane. */
    cboard_encode_piece_planes(b, out);

    for (int hi = 0; hi < b->hist_len && hi < CBOARD_HISTORY_MAX; hi++) {
        int idx = (b->hist_head - 1 - hi + CBOARD_HISTORY_MAX) % CBOARD_HISTORY_MAX;
        float *dest = out + ((hi + 1) * 13) * 64;
        cboard_encode_hist_planes_root(b->hist_bb[idx], b->hist_occ[idx],
                                       b->turn, dest);
    }

    int hist_n = b->hist_len < CBOARD_HISTORY_MAX ? b->hist_len : CBOARD_HISTORY_MAX;
    if (g_history_rep_fix) {
        /* Candidate path: per-slot repetition flags were recorded at push time
         * with the full look-back available, so they match Python
         * _check_repetitions() even across irreversible-move boundaries that
         * cleared the hash_stack. */
        for (int hi = hist_n - 1; hi >= 0; hi--) {
            int idx = (b->hist_head - 1 - hi + CBOARD_HISTORY_MAX) % CBOARD_HISTORY_MAX;
            if (b->hist_was_rep[idx])
                cboard_set_hist_rep_plane(out, hi);
        }
    } else {
        /* Default path: reconstruct from hash_stack + history hashes. Under-
         * reports slots whose repetition predates a cleared hash_stack window;
         * preserved as the byte-identical default until the candidate clears
         * its arena gate. */
        uint64_t seen[CBOARD_HASH_STACK_MAX + CBOARD_HISTORY_MAX + 1];
        int seen_n = 0;
        int seed_n = b->hash_stack_len - hist_n;
        if (seed_n < 0) seed_n = 0;
        for (int i = 0; i < seed_n && seen_n < (int)(sizeof(seen) / sizeof(seen[0])); i++) {
            seen[seen_n++] = b->hash_stack[i];
        }
        for (int hi = hist_n - 1; hi >= 0; hi--) {
            int idx = (b->hist_head - 1 - hi + CBOARD_HISTORY_MAX) % CBOARD_HISTORY_MAX;
            uint64_t h = cboard_hist_hash(
                b->hist_bb[idx], b->hist_occ[idx], b->hist_turn[idx],
                b->hist_castling[idx], b->hist_ep[idx]
            );
            int repeated = 0;
            for (int j = 0; j < seen_n; j++) {
                if (seen[j] == h) { repeated = 1; break; }
            }
            if (repeated)
                cboard_set_hist_rep_plane(out, hi);
            if (seen_n < (int)(sizeof(seen) / sizeof(seen[0]))) {
                seen[seen_n++] = h;
            }
        }
    }
    /* Current-position repetition (plane 12), identical for both paths.
     * The live hash_stack covers the reversible run; the kept window's
     * recorded hashes supplement it for runs that saturated the stack
     * (the stack stops appending when full), matching the pre-refactor
     * default path, which also checked the reconstructed window hashes. */
    int cur_rep = cboard_is_repetition(b);
    uint64_t cur_rep_key = cboard_repetition_key(b);
    for (int k = 0; !cur_rep && k < b->hist_len; k++) {
        if (b->hist_hash[k] == cur_rep_key) cur_rep = 1;
    }
    if (cur_rep) {
        for (int i = 0; i < 64; i++) out[12*64 + i] = 1.0f;
    }

    /* LC0 classical castling (us-Q, us-K, them-Q, them-K) at 104..107 */
    int us = b->turn;
    int us_k, us_q, them_k, them_q;
    if (us == WHITE_C) {
        us_k = b->castling & WK_CASTLE; us_q = b->castling & WQ_CASTLE;
        them_k = b->castling & BK_CASTLE; them_q = b->castling & BQ_CASTLE;
    } else {
        us_k = b->castling & BK_CASTLE; us_q = b->castling & BQ_CASTLE;
        them_k = b->castling & WK_CASTLE; them_q = b->castling & WQ_CASTLE;
    }
    if (us_q)   for (int i = 0; i < 64; i++) out[104*64 + i] = 1.0f;
    if (us_k)   for (int i = 0; i < 64; i++) out[105*64 + i] = 1.0f;
    if (them_q) for (int i = 0; i < 64; i++) out[106*64 + i] = 1.0f;
    if (them_k) for (int i = 0; i < 64; i++) out[107*64 + i] = 1.0f;

    /* Color/flipped flag: set when the root side-to-move is black. */
    if (b->turn == BLACK_C) {
        for (int i = 0; i < 64; i++) out[108*64 + i] = 1.0f;
    }

    float r50 = (float)b->halfmove_clock;
    for (int i = 0; i < 64; i++) out[109*64 + i] = r50;

    for (int i = 0; i < 64; i++) out[111*64 + i] = 1.0f;
}

static void cboard_apply_lc0_root_legacy_meta(const CBoard *b, float * restrict out) {
    float r50 = (float)b->halfmove_clock;
    if (r50 > 100.0f) r50 = 100.0f;
    r50 /= 100.0f;
    for (int i = 0; i < 64; i++) {
        out[109*64 + i] = r50;
        out[110*64 + i] = 0.0f;
    }
    if (b->ep_square >= 0 && b->ep_square < 64) {
        int file = b->ep_square & 7;
        for (int rank = 0; rank < 8; rank++) {
            out[110*64 + rank*8 + file] = 1.0f;
        }
    }
}

static void cboard_fill_lc0_112_root_legacy_meta(const CBoard *b, float * restrict out) {
    cboard_fill_lc0_112_root(b, out);
    cboard_apply_lc0_root_legacy_meta(b, out);
}

/* Generate FEN string from board state.
 * buf must be at least 100 bytes. Returns length written (excluding NUL). */
static int cboard_to_fen(const CBoard *b, char *buf, int buf_size) {
    const char *piece_chars = "PNBRQKpnbrqk";
    int pos = 0;
    for (int rank = 7; rank >= 0; rank--) {
        int empty = 0;
        for (int file = 0; file < 8; file++) {
            int sq = rank * 8 + file;
            uint64_t bit = 1ULL << sq;
            int found = 0;
            for (int color = 1; color >= 0; color--) {
                if (!(b->occ[color] & bit)) continue;
                for (int pt = 0; pt < 6; pt++) {
                    if (b->bb[pt] & bit) {
                        if (empty > 0) { buf[pos++] = '0' + empty; empty = 0; }
                        buf[pos++] = piece_chars[pt + (color == BLACK_C ? 6 : 0)];
                        found = 1;
                        break;
                    }
                }
                if (found) break;
            }
            if (!found) empty++;
        }
        if (empty > 0) buf[pos++] = '0' + empty;
        if (rank > 0) buf[pos++] = '/';
    }
    buf[pos++] = ' ';
    buf[pos++] = (b->turn == WHITE_C) ? 'w' : 'b';
    buf[pos++] = ' ';
    if (b->castling == 0) {
        buf[pos++] = '-';
    } else {
        if (b->castling & 1) buf[pos++] = 'K';
        if (b->castling & 2) buf[pos++] = 'Q';
        if (b->castling & 4) buf[pos++] = 'k';
        if (b->castling & 8) buf[pos++] = 'q';
    }
    buf[pos++] = ' ';
    /* Only emit EP if a pawn of the side to move can actually capture there —
     * the same test the transposition key uses. */
    if (cboard_ep_capture_available(b)) {
        buf[pos++] = 'a' + sq_file(b->ep_square);
        buf[pos++] = '1' + sq_rank(b->ep_square);
    } else {
        buf[pos++] = '-';
    }
    int fullmove = b->ply / 2 + 1;
    pos += snprintf(buf + pos, buf_size - pos, " %d %d", (int)b->halfmove_clock, fullmove);
    buf[pos] = '\0';
    return pos;
}

/* Return game result string: "1-0", "0-1", "1/2-1/2", or "*".
 * buf must be at least 8 bytes. */
static void cboard_result(const CBoard *b, char *buf) {
    if (!cboard_is_game_over(b)) { strcpy(buf, "*"); return; }
    if (cboard_is_checkmate(b)) {
        strcpy(buf, (b->turn == WHITE_C) ? "0-1" : "1-0");
        return;
    }
    strcpy(buf, "1/2-1/2");
}

/* ================================================================
 * Master init: call once at module load time
 * ================================================================ */

static void cboard_init_all(void) {
    init_attack_tables();
    init_policy_tables();
    init_policy_lut();
    init_zobrist();
    init_byte_to_float_lut();
}

#endif /* _CBOARD_IMPL_H */
