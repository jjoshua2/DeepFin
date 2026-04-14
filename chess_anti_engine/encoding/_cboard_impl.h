/*
 * _cboard_impl.h — Pure-C CBoard implementation (no Python/NumPy dependencies).
 *
 * All functions are static so each .so that includes this gets its own copy.
 * Call cboard_init_all() once at module init time to populate lookup tables.
 */

#ifndef _CBOARD_IMPL_H
#define _CBOARD_IMPL_H

#include <stdint.h>
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

#define FOR_EACH_BIT(bb, sq) \
    for (uint64_t _bb = (bb); _bb; _bb &= _bb - 1) \
        if (((sq) = lsb64(_bb)), 1)

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

/* 12 piece-color types x 64 squares, plus turn, castling, ep file */
static uint64_t ZOBRIST_PIECE[12][64];  /* [piece_color_idx][sq] */
static uint64_t ZOBRIST_TURN;
static uint64_t ZOBRIST_CASTLING[16];   /* indexed by 4-bit castling */
static uint64_t ZOBRIST_EP[8];          /* indexed by ep file 0..7 */
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
    int step = bishop_like ? 2 : 2;
    for (int d = start; d < 8; d += step) {
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
 * Byte-to-float lookup table for fast bitboard -> plane conversion.
 * 256 entries x 8 floats = 8 KB, fits in L1 cache.
 * ================================================================ */

static float BYTE_TO_8FLOATS[256][8];

static void init_byte_to_float_lut(void) {
    for (int b = 0; b < 256; b++)
        for (int i = 0; i < 8; i++)
            BYTE_TO_8FLOATS[b][i] = (float)((b >> i) & 1);
}

/* ================================================================
 * Bitboard -> plane conversion
 * ================================================================ */

/* Compile-time endianness detection. The LUT path reinterprets uint64_t
 * memory bytes directly and is only correct on little-endian hosts. */
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define BB_PLANE_USE_LUT 1
#elif defined(_WIN32) || defined(__x86_64__) || defined(__i386__) || \
      defined(__aarch64__) || defined(__arm__)
/* All common desktop/server targets are LE. */
#define BB_PLANE_USE_LUT 1
#else
#define BB_PLANE_USE_LUT 0
#endif

#ifdef __AVX2__
#include <immintrin.h>
#define BB_PLANE_USE_AVX2 1
#else
#define BB_PLANE_USE_AVX2 0
#endif

#if BB_PLANE_USE_AVX2
/* AVX2: expand one byte to 8 floats (0.0f or 1.0f) using SIMD.
 * Processes the entire 64-float plane in 8 iterations of pure SIMD ops. */
static const int32_t _avx2_bit_masks[8] __attribute__((aligned(32))) =
    {1, 2, 4, 8, 16, 32, 64, 128};

static inline void _avx2_byte_to_8floats(uint8_t byte, float *out) {
    __m256i masks = _mm256_load_si256((const __m256i *)_avx2_bit_masks);
    __m256i val = _mm256_set1_epi32(byte);
    __m256i and_r = _mm256_and_si256(val, masks);
    /* non-zero → all-ones mask */
    __m256i cmp = _mm256_cmpeq_epi32(and_r, _mm256_setzero_si256());
    __m256 result = _mm256_andnot_ps(_mm256_castsi256_ps(cmp),
                                      _mm256_set1_ps(1.0f));
    _mm256_storeu_ps(out, result);
}

static void bitboard_to_plane_white(uint64_t bb, float *out) {
    if (bb == 0) {
        _mm256_storeu_ps(out,      _mm256_setzero_ps());
        _mm256_storeu_ps(out + 8,  _mm256_setzero_ps());
        _mm256_storeu_ps(out + 16, _mm256_setzero_ps());
        _mm256_storeu_ps(out + 24, _mm256_setzero_ps());
        _mm256_storeu_ps(out + 32, _mm256_setzero_ps());
        _mm256_storeu_ps(out + 40, _mm256_setzero_ps());
        _mm256_storeu_ps(out + 48, _mm256_setzero_ps());
        _mm256_storeu_ps(out + 56, _mm256_setzero_ps());
        return;
    }
    const uint8_t *bytes = (const uint8_t *)&bb;
    for (int r = 0; r < 8; r++)
        _avx2_byte_to_8floats(bytes[r], out + r * 8);
}

static void bitboard_to_plane_black(uint64_t bb, float *out) {
    if (bb == 0) {
        _mm256_storeu_ps(out,      _mm256_setzero_ps());
        _mm256_storeu_ps(out + 8,  _mm256_setzero_ps());
        _mm256_storeu_ps(out + 16, _mm256_setzero_ps());
        _mm256_storeu_ps(out + 24, _mm256_setzero_ps());
        _mm256_storeu_ps(out + 32, _mm256_setzero_ps());
        _mm256_storeu_ps(out + 40, _mm256_setzero_ps());
        _mm256_storeu_ps(out + 48, _mm256_setzero_ps());
        _mm256_storeu_ps(out + 56, _mm256_setzero_ps());
        return;
    }
    const uint8_t *bytes = (const uint8_t *)&bb;
    for (int r = 0; r < 8; r++)
        _avx2_byte_to_8floats(bytes[7 - r], out + r * 8);
}

#else /* non-AVX2 path */

static void bitboard_to_plane_white(uint64_t bb, float *out) {
    if (bb == 0) {
        memset(out, 0, 64 * sizeof(float));
        return;
    }
#if BB_PLANE_USE_LUT
    const uint8_t *bytes = (const uint8_t *)&bb;
    for (int r = 0; r < 8; r++)
        memcpy(out + r * 8, BYTE_TO_8FLOATS[bytes[r]], 8 * sizeof(float));
#else
    for (int r = 0; r < 8; r++)
        for (int f = 0; f < 8; f++)
            out[r * 8 + f] = (float)((bb >> (r * 8 + f)) & 1);
#endif
}

static void bitboard_to_plane_black(uint64_t bb, float *out) {
    if (bb == 0) {
        memset(out, 0, 64 * sizeof(float));
        return;
    }
#if BB_PLANE_USE_LUT
    const uint8_t *bytes = (const uint8_t *)&bb;
    for (int r = 0; r < 8; r++)
        memcpy(out + r * 8, BYTE_TO_8FLOATS[bytes[7 - r]], 8 * sizeof(float));
#else
    for (int r = 0; r < 8; r++)
        for (int f = 0; f < 8; f++)
            out[r * 8 + f] = (float)((bb >> ((7 - r) * 8 + f)) & 1);
#endif
}

#endif /* BB_PLANE_USE_AVX2 */


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

/* Generate all pseudo-legal moves and filter by legality.
 * Writes policy indices to `indices`, returns count. */
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
    if (_idx >= 0) indices[count++] = _idx; \
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
            /* Squares between must be empty */
            if (!(bs->all_occ & (sq_bit(f_sq) | sq_bit(g)))) {
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
            if (!(bs->all_occ & (sq_bit(d) | sq_bit(c) | sq_bit(b)))) {
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


/* ================================================================
 * CBoard: lightweight C chess board for MCTS hot loop
 *
 * Replaces python-chess Board in the simulation loop.
 * copy() = memcpy ~72 bytes, push = pure C bitboard ops.
 * ================================================================ */

enum { PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5 };
enum { BLACK_C=0, WHITE_C=1 };
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
    int8_t hist_len;                           /* 0..7 valid history entries */
    int8_t hist_head;                          /* circular buffer write index */
    uint16_t ply;                              /* total half-moves from game start */
} CBoard;

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
                    promo = 5; /* mark as potential queen promo -- confirmed in push */
            }

            POLICY_LUT[turn][idx].from_sq = (int8_t)from_real;
            POLICY_LUT[turn][idx].to_sq = (int8_t)to_real;
            POLICY_LUT[turn][idx].promotion = (int8_t)promo;
        }
    }
    policy_lut_ready = 1;
}

/* Compute full Zobrist hash from CBoard state (for init, not incremental) */
/* Check if EP capture is actually legal (enemy pawn adjacent to double-pushed pawn) */
static int cboard_has_legal_ep(const CBoard *b) {
    if (b->ep_square < 0) return 0;
    /* The captured pawn is on the same rank as the side-to-move's pawns that could capture.
     * EP square is where the capturing pawn lands.
     * For white EP (ep on rank 5): captured pawn is on rank 4, capturing pawns on rank 4.
     * For black EP (ep on rank 2): captured pawn is on rank 3, capturing pawns on rank 3. */
    int ep = b->ep_square;
    int capture_rank_sq = (b->turn == WHITE_C) ? (ep - 8) : (ep + 8);
    /* Enemy pawns that could capture are on capture_rank_sq ± 1 file */
    /* Actually: the capturing side is b->turn. Their pawns must be adjacent to the
     * double-pushed pawn (which is at capture_rank_sq on the same file as ep). */
    int ep_file = sq_file(ep);
    /* Side-to-move pawns on the same rank as the captured pawn, adjacent file */
    uint64_t our_pawns = b->bb[PAWN] & b->occ[b->turn];
    uint64_t adjacent = 0;
    if (ep_file > 0) adjacent |= sq_bit(capture_rank_sq - 1);
    if (ep_file < 7) adjacent |= sq_bit(capture_rank_sq + 1);
    return (our_pawns & adjacent) ? 1 : 0;
}

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
    /* EP excluded from hash — improves transposition/cache hit rate.
     * Repetition detection already strips EP separately. */
    return h;
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

static void cboard_push(CBoard *b, int from_sq, int to_sq, int promotion) {
    int us = b->turn;
    int them = 1 - us;
    uint64_t from_bit = sq_bit(from_sq);
    uint64_t to_bit = sq_bit(to_sq);

    int moving_piece = piece_type_at(b, from_sq);
    int is_capture = (b->occ[them] & to_bit) != 0;
    int is_ep = (moving_piece == PAWN && to_sq == b->ep_square && b->ep_square >= 0);
    int is_pawn_move = (moving_piece == PAWN);
    int is_irreversible = is_pawn_move || is_capture;

    /* --- Save current position to history circular buffer --- */
    {
        int slot = b->hist_head;
        memcpy(b->hist_bb[slot], b->bb, 6 * sizeof(uint64_t));
        memcpy(b->hist_occ[slot], b->occ, 2 * sizeof(uint64_t));
        b->hist_turn[slot] = b->turn;
        b->hist_head = (slot + 1) % CBOARD_HISTORY_MAX;
        if (b->hist_len < CBOARD_HISTORY_MAX)
            b->hist_len++;
    }

    /* --- Push current hash onto hash_stack for repetition detection --- */
    if (is_irreversible) {
        /* Irreversible move: clear the hash stack (no repetition possible across these) */
        b->hash_stack_len = 0;
    } else {
        /* Hash already excludes EP, so use it directly for repetition detection. */
        uint64_t rep_hash = b->hash;
        if (b->hash_stack_len < CBOARD_HASH_STACK_MAX)
            b->hash_stack[b->hash_stack_len++] = rep_hash;
    }

    /* --- Save old state for incremental hash update --- */
    uint8_t old_castling = b->castling;
    int8_t old_ep = b->ep_square;
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
    if (promotion > 0 && promotion != 5 && is_pawn_move) {
        b->bb[PAWN] &= ~to_bit;
        b->bb[promotion - 1] |= to_bit;
        final_piece = promotion - 1;
    } else if (promotion == 5 && is_pawn_move) {
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

    /* EP excluded from hash for better transposition detection */

    /* Update EP square (always store for encoding, but hash only if legal) */
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
 * Hash already excludes EP, matching python-chess is_repetition() semantics. */
static int cboard_is_repetition(const CBoard *b) {
    uint64_t h = b->hash;
    for (int i = 0; i < b->hash_stack_len; i++) {
        if (b->hash_stack[i] == h) return 1;
    }
    return 0;
}

/* Strict repetition: 3-fold (current + 2 prior). Matches FIDE/python-chess rules. */
static int cboard_is_threefold_repetition(const CBoard *b) {
    uint64_t h = b->hash;
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

/* Compute a position key for repetition detection (from bitboards + meta).
 * Uses Zobrist hash -- matches cboard_compute_hash semantics. */
static uint64_t cboard_hist_hash(const uint64_t hist_bb[6],
                                  const uint64_t hist_occ[2],
                                  int hist_turn, uint8_t castling,
                                  int8_t ep_square) {
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
    if (ep_square >= 0) h ^= ZOBRIST_EP[sq_file(ep_square)];
    return h;
}

static void cboard_fill_lc0_112(const CBoard *b, float *out) {
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
    if (b->ep_square >= 0) {
        /* Only emit EP if an enemy pawn can actually capture there */
        int ep = b->ep_square;
        uint64_t ep_bit = 1ULL << ep;
        /* Pawns that could capture on ep_square: for white pawns attacking up,
         * black pawns attacking down */
        uint64_t attackers;
        if (b->turn == WHITE_C) {
            /* White to move, so black just pushed. White pawns capture ep. */
            attackers = ((ep_bit >> 7) & 0xFEFEFEFEFEFEFEFEULL)
                      | ((ep_bit >> 9) & 0x7F7F7F7F7F7F7F7FULL);
        } else {
            /* Black to move, white just pushed. Black pawns capture ep. */
            attackers = ((ep_bit << 7) & 0x7F7F7F7F7F7F7F7FULL)
                      | ((ep_bit << 9) & 0xFEFEFEFEFEFEFEFEULL);
        }
        /* "them" is the side that just moved; "turn" side has pawns that capture */
        uint64_t our_pawns = b->bb[PAWN] & b->occ[b->turn];
        if (our_pawns & attackers) {
            buf[pos++] = 'a' + sq_file(ep);
            buf[pos++] = '1' + sq_rank(ep);
        } else {
            buf[pos++] = '-';
        }
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
