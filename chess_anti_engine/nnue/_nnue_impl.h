/*
 * _nnue_impl.h — native big-net Stockfish-NNUE evaluator, header-only.
 *
 * NO STOCKFISH CODE IS IMPORTED, LINKED OR COPIED. The Stockfish sources were
 * read as a FORMAT AND ALGORITHM SPECIFICATION (nnue_architecture.h,
 * nnue_feature_transformer.h, features/full_threats.cpp,
 * features/half_ka_v2_hm.cpp, the layers/ headers); this is our own implementation of
 * the integer semantics they describe. The oracle that it is right is Stockfish
 * itself: scripts/nnue_parity.py requires EXACT integer equality against the
 * engine's own "(Big net) NNUE evaluation ... internal units" line. Internal
 * parity between our C and our numpy reference cannot find a rule that is wrong
 * in both, so it is used only to localise a failure the engine already found.
 *
 * Scope: the BIG net only, always. Stockfish's small net is a speed
 * optimisation for material-imbalanced positions, not an accuracy win, and
 * supporting only one architecture means one format, one code path, and no
 * net-selection rule to keep in parity. Documented consequence: our labels
 * differ from Stockfish's `eval` exactly where Stockfish would have picked the
 * small net.
 *
 * ⚑⚑ THE EVALUATOR REFUSES IN-CHECK POSITIONS, UNCONDITIONALLY AND AT THE SEAM.
 * The NNUE evaluation is undefined in check: Stockfish asserts !pos.checkers()
 * before evaluating and its `eval` command prints "Final evaluation: none (in
 * check)" with no network lines at all. cae_nnue_evaluate() returns
 * CAE_VALUE_ERR_IN_CHECK and writes NOTHING to the out-parameter, so there is no
 * sentinel a caller could mistake for an evaluation. THE CONTRACT ON CALLERS IS:
 * resolve check nodes RECURSIVELY (minimax backup over evasions, which may
 * themselves give check; repetition and 50-move terminals inside the resolver;
 * mate when there are no evasions) until a non-check position or a terminal is
 * reached, and only then call eval. This refusal is the enforcement backstop for
 * that invariant, not a substitute for it.
 *
 * Units: psqt/16 + positional/16, side-to-move POV — Stockfish's internal
 * units. We deliberately do NOT reproduce Eval::evaluate()'s post-processing
 * (optimism blending, complexity damping, material scaling, rule50 damping):
 * matching a number four scalings downstream would hide an eval defect.
 *
 * Weights come from an mmap-able pack built by scripts/nnue_pack.py. Both the
 * .nnue and the .pack are RUNTIME ARTIFACTS and are never committed; the loader
 * takes a path.
 *
 * Requires _cboard_impl.h (bitboards, attack tables, CBoard) to be includable.
 */

#ifndef CAE_NNUE_IMPL_H
#define CAE_NNUE_IMPL_H

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "_cboard_impl.h"
/* The eval seam owns the status contract this evaluator returns. */
#include "../mcts/_value_provider.h"

#if defined(__AVX2__)
#include <immintrin.h>
#define CAE_NNUE_HAVE_AVX2 1
#else
#define CAE_NNUE_HAVE_AVX2 0
#endif

/* ⚑ The AVX2 kernels are selected at RUNTIME, not only at compile time, so both
 * paths live in one binary and the parity gate can be run against each of them.
 * A SIMD path that is never compared against the engine is a SIMD path nobody
 * has checked; making the scalar fallback unreachable in a -march=native build
 * would have made that check impossible without a second build. The flag is set
 * once at load time and read-only thereafter, and the branch is per weight ROW
 * (once per 1024 elements), so it costs nothing measurable. */
static int cae_nnue_simd_enabled = CAE_NNUE_HAVE_AVX2;

static inline int cae_nnue_simd_available(void) { return CAE_NNUE_HAVE_AVX2; }
static inline int cae_nnue_simd_active(void) { return cae_nnue_simd_enabled; }

/* Returns 0 on success, -1 if SIMD was requested but is not compiled in. */
static inline int cae_nnue_set_simd(int enabled) {
    if (enabled && !CAE_NNUE_HAVE_AVX2) return -1;
    cae_nnue_simd_enabled = enabled ? 1 : 0;
    return 0;
}

/* ================================================================
 * Constants — read off the Stockfish sources, not guessed
 * ================================================================ */

#define CAE_NNUE_PACK_MAGIC      "CAENNUE1"
#define CAE_NNUE_PACK_VERSION    1u
#define CAE_NNUE_FILE_VERSION    0x7AF32F20u   /* the ONLY accepted .nnue version */
#define CAE_NNUE_HEADER_BYTES    256u
#define CAE_NNUE_HALFKA_DIMS     22528u
#define CAE_NNUE_THREAT_DIMS     60720u
#define CAE_NNUE_PSQT_BUCKETS    8u
#define CAE_NNUE_LAYER_STACKS    8u
#define CAE_NNUE_OUTPUT_SCALE    16
#define CAE_NNUE_WEIGHT_SCALE_BITS 6

/* Generous ceilings so the hot path never bounds-checks. Stockfish caps its own
 * active-threat list at 128 and scoping measured a maximum of 63; 1024 is above
 * any reachable count (fewer than 31 non-king attackers, each seeing at most 31
 * occupied squares is a loose bound this cannot approach in a legal position). */
#define CAE_NNUE_MAX_RELATIONS   1024
#define CAE_NNUE_MAX_L1          1024u
#define CAE_NNUE_MAX_FC0_OUT     64u
#define CAE_NNUE_MAX_FC1_IN      128u
#define CAE_NNUE_MAX_FC1_OUT     64u

/* Stockfish colour/piece encodings. ⚑ These are NOT CBoard's: CBoard uses
 * WHITE_C=1/BLACK_C=0 and piece types PAWN=0..KING=5, while the feature sets are
 * defined in terms of WHITE=0/BLACK=1 and PAWN=1..KING=6 with Piece = type +
 * 8*colour. cae_nnue_pos_from_cboard() is the single place the two meet. */
#define CAE_SF_WHITE 0
#define CAE_SF_BLACK 1
#define CAE_SF_PAWN   1
#define CAE_SF_KNIGHT 2
#define CAE_SF_BISHOP 3
#define CAE_SF_ROOK   4
#define CAE_SF_QUEEN  5
#define CAE_SF_KING   6

/* ================================================================
 * Pack header — mirrors scripts/nnue_pack.py byte for byte
 * ================================================================ */

typedef struct {
    char     magic[8];
    uint32_t pack_version;
    uint32_t nnue_version;
    uint32_t net_hash;
    uint32_t ft_hash;
    uint32_t l1, l2, l3;
    uint32_t psqt_buckets;
    uint32_t layer_stacks;
    uint32_t halfka_dims;
    uint32_t threat_dims;
    uint32_t use_threats;
    uint32_t fc0_outputs;
    uint32_t fc0_padded_in;
    uint32_t fc1_outputs;
    uint32_t fc1_padded_in;
    uint32_t fc2_padded_in;
    uint32_t reserved0;
    uint64_t total_size;
    uint64_t off[11];
    uint8_t  sha256[32];
} CaeNnuePackHeader;

_Static_assert(sizeof(CaeNnuePackHeader) == 208, "pack header layout drifted");

typedef struct CaeNnueWeights {
    void   *map;
    size_t  map_size;

    uint32_t l1, l2, l3;
    uint32_t halfka_dims, threat_dims;
    uint32_t layer_stacks, psqt_buckets;
    uint32_t fc0_outputs, fc0_padded_in;
    uint32_t fc1_outputs, fc1_padded_in, fc2_padded_in;
    uint32_t net_hash, ft_hash;

    const int16_t *ft_bias;        /* [l1] */
    const int16_t *ft_weight;      /* [halfka_dims][l1] */
    const int32_t *ft_psqt;        /* [halfka_dims][psqt_buckets] */
    const int8_t  *threat_weight;  /* [threat_dims][l1] */
    const int32_t *threat_psqt;    /* [threat_dims][psqt_buckets] */
    const int32_t *fc0_bias;       /* [stacks][fc0_outputs] */
    const int8_t  *fc0_weight;     /* [stacks][fc0_outputs][fc0_padded_in] */
    const int32_t *fc1_bias;       /* [stacks][fc1_outputs] */
    const int8_t  *fc1_weight;     /* [stacks][fc1_outputs][fc1_padded_in] */
    const int32_t *fc2_bias;       /* [stacks][1] */
    const int8_t  *fc2_weight;     /* [stacks][1][fc2_padded_in] */

    char sha256_hex[65];
    char path[4096];
    int  refcount;                 /* guarded by cae_nnue_cache_lock */
} CaeNnueWeights;

/* ================================================================
 * Feature-index lookup tables (built once, read-only thereafter)
 * ================================================================ */

/* PseudoAttacks: ⚑ index 0 and 1 hold WHITE and BLACK PAWN attacks, exactly as
 * in Stockfish, where the colour enum overlaps the piece-type enum. Indices
 * 2..6 are KNIGHT..KING. */
static uint64_t NN_PSEUDO[7][64];
static uint64_t NN_PAWN_PUSH_OR_ATK[2][64];

static uint16_t NN_TH_OFFSETS[16][64];      /* offsets[piece][from] */
static uint32_t NN_TH_LUT1[16][16][2];      /* [attacker][attacked][from_o < to_o] */
static uint8_t  NN_TH_LUT2[16][64][64];     /* [piece][from][to] */
static uint32_t NN_HALFKA_KING_BUCKET[64];
static uint8_t  NN_HALFKA_ORIENT[64];
static uint8_t  NN_THREATS_ORIENT[64];

/* HalfKAv2_hm::PieceSquareIndex[perspective][piece]. */
static const uint16_t NN_PIECE_SQUARE_INDEX[2][16] = {
    {0, 0, 2 * 64, 4 * 64, 6 * 64, 8 * 64, 10 * 64, 0,
     0, 1 * 64, 3 * 64, 5 * 64, 7 * 64, 9 * 64, 10 * 64, 0},
    {0, 1 * 64, 3 * 64, 5 * 64, 7 * 64, 9 * 64, 10 * 64, 0,
     0, 0, 2 * 64, 4 * 64, 6 * 64, 8 * 64, 10 * 64, 0}
};

/* FullThreats::numValidTargets, indexed by Piece. */
static const int NN_NUM_VALID_TARGETS[16] = {
    0, 6, 10, 8, 8, 10, 0, 0, 0, 6, 10, 8, 8, 10, 0, 0
};

/* FullThreats::map, indexed [attacker_type - 1][attacked_type - 1]. */
static const int NN_THREAT_MAP[6][6] = {
    { 0,  1, -1,  2, -1, -1},
    { 0,  1,  2,  3,  4, -1},
    { 0,  1,  2,  3, -1, -1},
    { 0,  1,  2,  3, -1, -1},
    { 0,  1,  2,  3,  4, -1},
    {-1, -1, -1, -1, -1, -1}
};

/* HalfKAv2_hm::KingBuckets, before the PS_NB multiply, a1..h8. */
static const uint8_t NN_KING_BUCKET_ID[64] = {
    28, 29, 30, 31, 31, 30, 29, 28,
    24, 25, 26, 27, 27, 26, 25, 24,
    20, 21, 22, 23, 23, 22, 21, 20,
    16, 17, 18, 19, 19, 18, 17, 16,
    12, 13, 14, 15, 15, 14, 13, 12,
     8,  9, 10, 11, 11, 10,  9,  8,
     4,  5,  6,  7,  7,  6,  5,  4,
     0,  1,  2,  3,  3,  2,  1,  0
};

static const int NN_ALL_PIECES[12] = {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14};

static uint64_t nn_ray_attacks(int sq, uint64_t occ, int diagonal) {
    static const int DF[2][4] = {{1, 1, -1, -1}, {1, -1, 0, 0}};
    static const int DR[2][4] = {{1, -1, 1, -1}, {0, 0, 1, -1}};
    int f = sq_file(sq), r = sq_rank(sq);
    uint64_t bb = 0;
    for (int d = 0; d < 4; d++) {
        int nf = f + DF[diagonal ? 0 : 1][d];
        int nr = r + DR[diagonal ? 0 : 1][d];
        while (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
            int t = make_sq(nf, nr);
            bb |= sq_bit(t);
            if (occ & sq_bit(t)) break;
            nf += DF[diagonal ? 0 : 1][d];
            nr += DR[diagonal ? 0 : 1][d];
        }
    }
    return bb;
}

/* attacks_bb(pt, s, occupied) for the non-pawn types. */
static inline uint64_t nn_attacks_from(int piece_type, int sq, uint64_t occ) {
    switch (piece_type) {
        case CAE_SF_KNIGHT: return KNIGHT_ATTACKS[sq];
        case CAE_SF_KING:   return KING_ATTACKS[sq];
        case CAE_SF_BISHOP: return nn_ray_attacks(sq, occ, 1);
        case CAE_SF_ROOK:   return nn_ray_attacks(sq, occ, 0);
        case CAE_SF_QUEEN:  return nn_ray_attacks(sq, occ, 1) | nn_ray_attacks(sq, occ, 0);
        default:            return 0;
    }
}

static uint64_t nn_pawn_attacks_sf(int colour, int sq) {
    int f = sq_file(sq), r = sq_rank(sq);
    int dr = (colour == CAE_SF_WHITE) ? 1 : -1;
    uint64_t bb = 0;
    for (int df = -1; df <= 1; df += 2) {
        int nf = f + df, nr = r + dr;
        if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) bb |= sq_bit(make_sq(nf, nr));
    }
    return bb;
}

static uint64_t nn_pawn_push_sf(int colour, int sq) {
    int nr = sq_rank(sq) + ((colour == CAE_SF_WHITE) ? 1 : -1);
    return (nr >= 0 && nr < 8) ? sq_bit(make_sq(sq_file(sq), nr)) : 0ULL;
}

/* The empty-board attack set the FullThreats offsets are enumerated from. */
static inline uint64_t nn_threat_attack_set(int piece, int sq) {
    int pt = piece & 7;
    if (pt == CAE_SF_PAWN) return NN_PAWN_PUSH_OR_ATK[piece >> 3][sq];
    return NN_PSEUDO[pt][sq];
}

static pthread_once_t cae_nnue_tables_once = PTHREAD_ONCE_INIT;

static void cae_nnue_build_tables(void) {
    for (int sq = 0; sq < 64; sq++) {
        NN_PSEUDO[CAE_SF_WHITE][sq] = nn_pawn_attacks_sf(CAE_SF_WHITE, sq);
        NN_PSEUDO[CAE_SF_BLACK][sq] = nn_pawn_attacks_sf(CAE_SF_BLACK, sq);
        NN_PSEUDO[CAE_SF_KNIGHT][sq] = KNIGHT_ATTACKS[sq];
        NN_PSEUDO[CAE_SF_KING][sq]   = KING_ATTACKS[sq];
        NN_PSEUDO[CAE_SF_BISHOP][sq] = nn_ray_attacks(sq, 0, 1);
        NN_PSEUDO[CAE_SF_ROOK][sq]   = nn_ray_attacks(sq, 0, 0);
        NN_PSEUDO[CAE_SF_QUEEN][sq]  = NN_PSEUDO[CAE_SF_BISHOP][sq] | NN_PSEUDO[CAE_SF_ROOK][sq];
        for (int c = 0; c < 2; c++)
            NN_PAWN_PUSH_OR_ATK[c][sq] = nn_pawn_push_sf(c, sq) | nn_pawn_attacks_sf(c, sq);

        /* HalfKAv2_hm mirrors so the king sits on files e..h; FullThreats
         * mirrors the OPPOSITE way, onto files a..d. They are genuinely
         * different tables — do not "simplify" one into the other. */
        NN_HALFKA_ORIENT[sq]  = (uint8_t)((sq_file(sq) < 4) ? 7 : 0);
        NN_THREATS_ORIENT[sq] = (uint8_t)((sq_file(sq) < 4) ? 0 : 7);
        NN_HALFKA_KING_BUCKET[sq] = (uint32_t)NN_KING_BUCKET_ID[sq] * 11u * 64u;
    }

    /* init_threat_offsets(): a running enumeration of every (piece, from, to)
     * pseudo-attack pair, in the piece order Stockfish walks AllPieces. */
    int piece_span[16] = {0};
    int piece_base[16] = {0};
    uint32_t cumulative_offset = 0;
    for (int i = 0; i < 12; i++) {
        int piece = NN_ALL_PIECES[i];
        int pt = piece & 7;
        int cumulative_piece_offset = 0;
        for (int from = 0; from < 64; from++) {
            NN_TH_OFFSETS[piece][from] = (uint16_t)cumulative_piece_offset;
            if (pt != CAE_SF_PAWN)
                cumulative_piece_offset += popcount64(NN_PSEUDO[pt][from]);
            else if (from >= 8 && from <= 55)
                cumulative_piece_offset += popcount64(NN_PAWN_PUSH_OR_ATK[piece >> 3][from]);
        }
        piece_span[piece] = cumulative_piece_offset;
        piece_base[piece] = (int)cumulative_offset;
        cumulative_offset += (uint32_t)(NN_NUM_VALID_TARGETS[piece] * cumulative_piece_offset);
    }

    for (int a = 0; a < 16; a++)
        for (int d = 0; d < 16; d++)
            NN_TH_LUT1[a][d][0] = NN_TH_LUT1[a][d][1] = CAE_NNUE_THREAT_DIMS;

    for (int i = 0; i < 12; i++) {
        int attacker = NN_ALL_PIECES[i];
        for (int j = 0; j < 12; j++) {
            int attacked = NN_ALL_PIECES[j];
            int enemy = ((attacker ^ attacked) == 8);
            int at = attacker & 7, ad = attacked & 7;
            int mapped = NN_THREAT_MAP[at - 1][ad - 1];
            int semi_excluded = (at == ad) && (enemy || at != CAE_SF_PAWN);
            uint32_t feature = (uint32_t)(piece_base[attacker]
                + ((attacked >> 3) * (NN_NUM_VALID_TARGETS[attacker] / 2) + mapped)
                  * piece_span[attacker]);
            NN_TH_LUT1[attacker][attacked][0] =
                (mapped < 0) ? CAE_NNUE_THREAT_DIMS : feature;
            NN_TH_LUT1[attacker][attacked][1] =
                (mapped < 0 || semi_excluded) ? CAE_NNUE_THREAT_DIMS : feature;
        }
    }

    for (int i = 0; i < 12; i++) {
        int piece = NN_ALL_PIECES[i];
        for (int from = 0; from < 64; from++) {
            uint64_t attacks = nn_threat_attack_set(piece, from);
            for (int to = 0; to < 64; to++) {
                uint64_t below = (to == 0) ? 0ULL : ((1ULL << to) - 1ULL);
                NN_TH_LUT2[piece][from][to] = (uint8_t)popcount64(below & attacks);
            }
        }
    }
}

static inline void cae_nnue_init_tables(void) {
    cboard_init_all();
    pthread_once(&cae_nnue_tables_once, cae_nnue_build_tables);
}

/* The total FullThreats feature count implied by the tables we just built.
 * Stockfish hardcodes 60720; recomputing it and checking is a cheap proof that
 * our attack geometry agrees with the one the weights were trained against. */
static uint32_t cae_nnue_computed_threat_dims(void) {
    uint32_t total = 0;
    for (int i = 0; i < 12; i++) {
        int piece = NN_ALL_PIECES[i];
        int span = 0;
        int pt = piece & 7;
        for (int from = 0; from < 64; from++) {
            if (pt != CAE_SF_PAWN) span += popcount64(NN_PSEUDO[pt][from]);
            else if (from >= 8 && from <= 55)
                span += popcount64(NN_PAWN_PUSH_OR_ATK[piece >> 3][from]);
        }
        total += (uint32_t)(NN_NUM_VALID_TARGETS[piece] * span);
    }
    return total;
}

/* ================================================================
 * Position view
 * ================================================================ */

typedef struct {
    uint64_t pieces[2][7];   /* [SF colour][SF piece type 1..6] */
    uint64_t occupied;
    uint8_t  piece_on[64];   /* SF Piece code: 0 empty, 1..6 white, 9..14 black */
    uint8_t  king_sq[2];
    uint8_t  side_to_move;   /* CAE_SF_WHITE / CAE_SF_BLACK */
    uint8_t  piece_count;
    uint8_t  in_check;
} CaeNnuePos;

static int cae_nnue_pos_from_cboard(const CBoard *b, CaeNnuePos *p) {
    memset(p, 0, sizeof(*p));
    for (int pt = CAE_SF_PAWN; pt <= CAE_SF_KING; pt++) {
        p->pieces[CAE_SF_WHITE][pt] = b->bb[pt - 1] & b->occ[WHITE_C];
        p->pieces[CAE_SF_BLACK][pt] = b->bb[pt - 1] & b->occ[BLACK_C];
    }
    p->occupied = b->occ[WHITE_C] | b->occ[BLACK_C];
    p->piece_count = (uint8_t)popcount64(p->occupied);
    if (p->piece_count < 2 || p->piece_count > 32)
        return CAE_VALUE_ERR_BAD_POS;
    if (!p->pieces[CAE_SF_WHITE][CAE_SF_KING] || !p->pieces[CAE_SF_BLACK][CAE_SF_KING])
        return CAE_VALUE_ERR_BAD_POS;
    p->king_sq[CAE_SF_WHITE] = (uint8_t)lsb64(p->pieces[CAE_SF_WHITE][CAE_SF_KING]);
    p->king_sq[CAE_SF_BLACK] = (uint8_t)lsb64(p->pieces[CAE_SF_BLACK][CAE_SF_KING]);
    p->side_to_move = (b->turn == WHITE_C) ? CAE_SF_WHITE : CAE_SF_BLACK;

    int sq;
    FOR_EACH_BIT(p->occupied, sq) {
        int pt = piece_type_at(b, sq);                  /* CBoard 0..5, -1 if none */
        if (pt < 0) return CAE_VALUE_ERR_BAD_POS;       /* occupancy disagrees with bb[] */
        int colour = (b->occ[WHITE_C] & sq_bit(sq)) ? CAE_SF_WHITE : CAE_SF_BLACK;
        p->piece_on[sq] = (uint8_t)((pt + 1) + 8 * colour);   /* -> SF Piece code */
    }
    p->in_check = (uint8_t)(cboard_in_check(b) ? 1 : 0);
    return CAE_VALUE_OK;
}

static inline int cae_nnue_bucket(const CaeNnuePos *p) {
    return ((int)p->piece_count - 1) / 4;
}

/* ⚑ The bucket indexes an array, so it gets VALIDATED, not clamped. A legal
 * position has 2..32 pieces and therefore a bucket in 0..7, so this can only
 * fire on a malformed position — but the failure mode it replaces is an
 * out-of-bounds read of the weight mapping that returns a plausible number,
 * which is this repo's signature defect wearing a different hat. Clamping
 * instead would keep that number and only hide where it came from. */
static inline int cae_nnue_check_bucket(const CaeNnueWeights *w, int bucket) {
    return (bucket >= 0 && bucket < (int)w->layer_stacks) ? CAE_VALUE_OK : CAE_VALUE_ERR_BAD_POS;
}

/* ================================================================
 * Feature indices
 * ================================================================ */

static inline uint32_t cae_nnue_halfka_index(int perspective, int sq, int piece, int ksq) {
    int flip = 56 * perspective;
    return (uint32_t)(sq ^ NN_HALFKA_ORIENT[ksq] ^ flip)
         + NN_PIECE_SQUARE_INDEX[perspective][piece]
         + NN_HALFKA_KING_BUCKET[ksq ^ flip];
}

static inline uint32_t cae_nnue_threat_index(
    int perspective, int attacker, int from, int to, int attacked, int ksq)
{
    int orientation = NN_THREATS_ORIENT[ksq] ^ (56 * perspective);
    unsigned from_o = (unsigned)(from ^ orientation);
    unsigned to_o   = (unsigned)(to ^ orientation);
    int swap = 8 * perspective;
    unsigned attacker_o = (unsigned)(attacker ^ swap);
    unsigned attacked_o = (unsigned)(attacked ^ swap);
    return NN_TH_LUT1[attacker_o][attacked_o][from_o < to_o]
         + NN_TH_OFFSETS[attacker_o][from_o]
         + NN_TH_LUT2[attacker_o][from_o][to_o];
}

/* One (attacker, from, to) threat relation. ⚑ The relation SET is
 * perspective-independent — only the index each relation maps to depends on the
 * perspective — so movegen runs once for both sides. Stockfish walks the two
 * colours in perspective-dependent order, but the accumulator sums them, and
 * modular int16 addition does not care about order. */
typedef struct {
    uint8_t attacker;
    uint8_t from;
    uint8_t to;
} CaeThreatRel;

static int cae_nnue_threat_relations(const CaeNnuePos *p, CaeThreatRel *out) {
    uint64_t occupied = p->occupied;
    uint64_t pawns = p->pieces[CAE_SF_WHITE][CAE_SF_PAWN] | p->pieces[CAE_SF_BLACK][CAE_SF_PAWN];
    int n = 0;

    for (int c = CAE_SF_WHITE; c <= CAE_SF_BLACK; c++) {
        for (int pt = CAE_SF_PAWN; pt < CAE_SF_KING; pt++) {
            int attacker = pt + 8 * c;
            uint64_t bb = p->pieces[c][pt];
            if (!bb) continue;

            if (pt == CAE_SF_PAWN) {
                int right = (c == CAE_SF_WHITE) ? 9 : -9;
                int left  = (c == CAE_SF_WHITE) ? 7 : -7;
                uint64_t attacks_left, attacks_right;
                if (c == CAE_SF_WHITE) {
                    attacks_left  = (bb & ~0x8080808080808080ULL) << 9;
                    attacks_right = (bb & ~0x0101010101010101ULL) << 7;
                } else {
                    attacks_left  = (bb & ~0x0101010101010101ULL) >> 9;
                    attacks_right = (bb & ~0x8080808080808080ULL) >> 7;
                }
                attacks_left  &= occupied;
                attacks_right &= occupied;
                int to;
                FOR_EACH_BIT(attacks_left, to) {
                    out[n].attacker = (uint8_t)attacker;
                    out[n].from = (uint8_t)(to - right);
                    out[n].to = (uint8_t)to;
                    n++;
                }
                FOR_EACH_BIT(attacks_right, to) {
                    out[n].attacker = (uint8_t)attacker;
                    out[n].from = (uint8_t)(to - left);
                    out[n].to = (uint8_t)to;
                    n++;
                }
                /* Pawns blocked by a pawn of either colour directly in front. */
                uint64_t shifted = (c == CAE_SF_WHITE) ? (pawns >> 8) : (pawns << 8);
                uint64_t pushers = shifted & p->pieces[c][CAE_SF_PAWN];
                int from;
                FOR_EACH_BIT(pushers, from) {
                    out[n].attacker = (uint8_t)attacker;
                    out[n].from = (uint8_t)from;
                    out[n].to = (uint8_t)(from + ((c == CAE_SF_WHITE) ? 8 : -8));
                    n++;
                }
            } else {
                int from;
                FOR_EACH_BIT(bb, from) {
                    uint64_t attacks = nn_attacks_from(pt, from, occupied) & occupied;
                    int to;
                    FOR_EACH_BIT(attacks, to) {
                        out[n].attacker = (uint8_t)attacker;
                        out[n].from = (uint8_t)from;
                        out[n].to = (uint8_t)to;
                        n++;
                    }
                }
            }
        }
    }
    return n;
}

/* ================================================================
 * Accumulator
 * ================================================================ */

typedef struct {
    int16_t acc[2][CAE_NNUE_MAX_L1] __attribute__((aligned(32)));
    int32_t psqt[2][CAE_NNUE_PSQT_BUCKETS];
} CaeNnueAcc;

/* Add one int16 weight row into the int16 accumulator, WRAPPING on overflow the
 * way Stockfish's int16 accumulators (and its vec_add_16) do. Unsigned
 * arithmetic makes the wrap defined instead of UB. */
static inline void cae_nnue_add_row_i16(int16_t *acc, const int16_t *row, uint32_t n) {
#if CAE_NNUE_HAVE_AVX2
    if (cae_nnue_simd_enabled) {
        for (uint32_t j = 0; j < n; j += 16) {
            __m256i a = _mm256_load_si256((const __m256i *)(acc + j));
            __m256i b = _mm256_loadu_si256((const __m256i *)(row + j));
            _mm256_store_si256((__m256i *)(acc + j), _mm256_add_epi16(a, b));
        }
        return;
    }
#endif
    for (uint32_t j = 0; j < n; j++)
        acc[j] = (int16_t)((uint16_t)acc[j] + (uint16_t)row[j]);
}

/* Same, widening an int8 threat weight row to int16 first. */
static inline void cae_nnue_add_row_i8(int16_t *acc, const int8_t *row, uint32_t n) {
#if CAE_NNUE_HAVE_AVX2
    if (cae_nnue_simd_enabled) {
        for (uint32_t j = 0; j < n; j += 16) {
            __m256i a = _mm256_load_si256((const __m256i *)(acc + j));
            __m128i w = _mm_loadu_si128((const __m128i *)(row + j));
            _mm256_store_si256((__m256i *)(acc + j),
                               _mm256_add_epi16(a, _mm256_cvtepi8_epi16(w)));
        }
        return;
    }
#endif
    for (uint32_t j = 0; j < n; j++)
        acc[j] = (int16_t)((uint16_t)acc[j] + (uint16_t)(int16_t)row[j]);
}

/* Full refresh. ⚑ Stockfish keeps the HalfKA and the threat contributions in two
 * separate int16 accumulators and adds them in int16 at transform() time. We sum
 * them into one accumulator instead: int16 addition is modular, so associativity
 * makes the two forms bit-identical, and one pass halves the memory traffic. */
static void cae_nnue_refresh(const CaeNnueWeights *w, const CaeNnuePos *p, CaeNnueAcc *a) {
    CaeThreatRel rel[CAE_NNUE_MAX_RELATIONS];
    int n_rel = cae_nnue_threat_relations(p, rel);
    const uint32_t l1 = w->l1;
    const uint32_t nb = w->psqt_buckets;

    for (int perspective = 0; perspective < 2; perspective++) {
        int16_t *acc = a->acc[perspective];
        int32_t *psqt = a->psqt[perspective];
        memcpy(acc, w->ft_bias, (size_t)l1 * sizeof(int16_t));
        memset(psqt, 0, sizeof(int32_t) * nb);

        int ksq = p->king_sq[perspective];
        int flip = 56 * perspective;
        int orient = NN_HALFKA_ORIENT[ksq] ^ flip;
        uint32_t king_bucket = NN_HALFKA_KING_BUCKET[ksq ^ flip];

        int sq;
        FOR_EACH_BIT(p->occupied, sq) {
            uint32_t idx = (uint32_t)(sq ^ orient)
                         + NN_PIECE_SQUARE_INDEX[perspective][p->piece_on[sq]]
                         + king_bucket;
            cae_nnue_add_row_i16(acc, w->ft_weight + (size_t)idx * l1, l1);
            const int32_t *prow = w->ft_psqt + (size_t)idx * nb;
            for (uint32_t k = 0; k < nb; k++) psqt[k] += prow[k];
        }

        /* ⚑ One index formula, one call site. An inlined second copy here would
         * mean the unit test that checks active_features() and the parity gate
         * that checks the evaluation were policing DIFFERENT code, so an
         * off-by-one could survive whichever one was not run. */
        for (int i = 0; i < n_rel; i++) {
            uint32_t idx = cae_nnue_threat_index(perspective, rel[i].attacker, rel[i].from,
                                                 rel[i].to, p->piece_on[rel[i].to], ksq);
            if (idx >= w->threat_dims) continue;   /* excluded relation */
            cae_nnue_add_row_i8(acc, w->threat_weight + (size_t)idx * l1, l1);
            const int32_t *prow = w->threat_psqt + (size_t)idx * nb;
            for (uint32_t k = 0; k < nb; k++) psqt[k] += prow[k];
        }
    }
}

/* ================================================================
 * Feature transformer and layer stack
 * ================================================================ */

static int32_t cae_nnue_transform(
    const CaeNnueWeights *w, const CaeNnueAcc *a, int stm, int bucket, uint8_t *out)
{
    const int p0 = stm, p1 = stm ^ 1;
    int32_t psqt = a->psqt[p0][bucket] - a->psqt[p1][bucket];
    psqt = psqt / 2;   /* C division truncates toward zero, as Stockfish's does */

    const uint32_t half = w->l1 / 2;
    for (int p = 0; p < 2; p++) {
        const int16_t *side = a->acc[p ? p1 : p0];
        uint8_t *dst = out + half * (uint32_t)p;
#if CAE_NNUE_HAVE_AVX2
        if (cae_nnue_simd_enabled) {
        const __m256i zero = _mm256_setzero_si256();
        const __m256i c255 = _mm256_set1_epi16(255);
        for (uint32_t j = 0; j < half; j += 32) {
            /* Two 16-wide halves feed one 32-byte packus result, mirroring
             * Stockfish's shift-left-7 + mulhi trick, which nets the /512. */
            __m256i a0 = _mm256_load_si256((const __m256i *)(side + j));
            __m256i a1 = _mm256_load_si256((const __m256i *)(side + j + 16));
            __m256i b0 = _mm256_load_si256((const __m256i *)(side + half + j));
            __m256i b1 = _mm256_load_si256((const __m256i *)(side + half + j + 16));
            __m256i s0 = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(a0, c255), zero), 7);
            __m256i s1 = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(a1, c255), zero), 7);
            __m256i t0 = _mm256_min_epi16(b0, c255);
            __m256i t1 = _mm256_min_epi16(b1, c255);
            __m256i pa = _mm256_mulhi_epi16(s0, t0);
            __m256i pb = _mm256_mulhi_epi16(s1, t1);
            /* packus interleaves the two 128-bit lanes; permute4x64 undoes it. */
            __m256i packed = _mm256_packus_epi16(pa, pb);
            packed = _mm256_permute4x64_epi64(packed, 0xD8);
            _mm256_storeu_si256((__m256i *)(dst + j), packed);
        }
        continue;
        }
#endif
        for (uint32_t j = 0; j < half; j++) {
            int32_t s0 = side[j];
            int32_t s1 = side[j + half];
            s0 = s0 < 0 ? 0 : (s0 > 255 ? 255 : s0);
            s1 = s1 < 0 ? 0 : (s1 > 255 ? 255 : s1);
            dst[j] = (uint8_t)((unsigned)(s0 * s1) / 512u);
        }
    }
    return psqt;
}

static int32_t cae_nnue_propagate(const CaeNnueWeights *w, const uint8_t *ft, int bucket) {
    const uint32_t n = w->l2;                 /* FC_0_OUTPUTS */
    int32_t fc0[CAE_NNUE_MAX_FC0_OUT];

    const int32_t *b0 = w->fc0_bias + (size_t)bucket * w->fc0_outputs;
    const int8_t  *W0 = w->fc0_weight
                      + (size_t)bucket * w->fc0_outputs * w->fc0_padded_in;
    for (uint32_t o = 0; o < w->fc0_outputs; o++) {
        const int8_t *row = W0 + (size_t)o * w->fc0_padded_in;
        int32_t s = b0[o];
        for (uint32_t j = 0; j < w->l1; j++) s += (int32_t)row[j] * (int32_t)ft[j];
        fc0[o] = s;
    }

    /* SqrClippedReLU over outputs 0..n-1, then ClippedReLU of the same outputs
     * appended; the pair is fc_1's 2n-wide input, zero-padded to fc1_padded_in. */
    uint8_t fc1_in[CAE_NNUE_MAX_FC1_IN];
    memset(fc1_in, 0, w->fc1_padded_in);
    for (uint32_t i = 0; i < n; i++) {
        int64_t sq = (int64_t)fc0[i] * (int64_t)fc0[i];
        int64_t v = sq >> (2 * CAE_NNUE_WEIGHT_SCALE_BITS + 7);
        fc1_in[i] = (uint8_t)(v > 127 ? 127 : v);
        int32_t c = fc0[i] >> CAE_NNUE_WEIGHT_SCALE_BITS;
        fc1_in[n + i] = (uint8_t)(c < 0 ? 0 : (c > 127 ? 127 : c));
    }

    int32_t ac1[CAE_NNUE_MAX_FC1_OUT];
    const int32_t *b1 = w->fc1_bias + (size_t)bucket * w->fc1_outputs;
    const int8_t  *W1 = w->fc1_weight
                      + (size_t)bucket * w->fc1_outputs * w->fc1_padded_in;
    for (uint32_t o = 0; o < w->fc1_outputs; o++) {
        const int8_t *row = W1 + (size_t)o * w->fc1_padded_in;
        int32_t s = b1[o];
        for (uint32_t j = 0; j < 2 * n; j++) s += (int32_t)row[j] * (int32_t)fc1_in[j];
        int32_t c = s >> CAE_NNUE_WEIGHT_SCALE_BITS;
        ac1[o] = c < 0 ? 0 : (c > 127 ? 127 : c);
    }

    const int8_t *W2 = w->fc2_weight + (size_t)bucket * w->fc2_padded_in;
    int32_t fc2 = w->fc2_bias[bucket];
    for (uint32_t j = 0; j < w->fc1_outputs; j++) fc2 += (int32_t)W2[j] * ac1[j];

    /* fc0[n] is a forward-skip term on a different scale: 1.0 there is
     * 127 << WeightScaleBits, but the output wants 1.0 == 600 * OutputScale.
     * ⚑ Stockfish does this multiply in int32, so we do too — computing it in
     * int64 would DIVERGE from the engine on an overflowing activation instead
     * of matching it. Unsigned arithmetic makes the wrap defined. */
    int32_t scaled = (int32_t)((uint32_t)fc0[n] * (uint32_t)(600 * CAE_NNUE_OUTPUT_SCALE));
    int32_t fwd = scaled / (127 * (1 << CAE_NNUE_WEIGHT_SCALE_BITS));
    return fc2 + fwd;
}

/* ================================================================
 * Public evaluation entry points
 * ================================================================ */

/* Full per-bucket split, for the parity harness's three-layer localisation.
 * psqt_out/positional_out receive w->psqt_buckets entries each, already divided
 * by OutputScale, so psqt_out[b] + positional_out[b] is the value Stockfish's
 * trace table shows for bucket b. */
static int cae_nnue_trace(
    const CaeNnueWeights *w, const CaeNnuePos *p, int32_t *psqt_out, int32_t *positional_out)
{
    if (!w) return CAE_VALUE_ERR_NOT_LOADED;
    if (p->in_check) return CAE_VALUE_ERR_IN_CHECK;

    CaeNnueAcc acc;
    uint8_t ft[CAE_NNUE_MAX_L1] __attribute__((aligned(32)));
    cae_nnue_refresh(w, p, &acc);
    for (uint32_t b = 0; b < w->psqt_buckets; b++) {
        int32_t psqt = cae_nnue_transform(w, &acc, p->side_to_move, (int)b, ft);
        int32_t positional = cae_nnue_propagate(w, ft, (int)b);
        psqt_out[b] = psqt / CAE_NNUE_OUTPUT_SCALE;
        positional_out[b] = positional / CAE_NNUE_OUTPUT_SCALE;
    }
    return CAE_VALUE_OK;
}

/* The evaluator. Returns CAE_VALUE_OK and writes *out, or a negative status and
 * writes NOTHING — in particular an in-check position yields no number at all. */
static int cae_nnue_evaluate(const CaeNnueWeights *w, const CaeNnuePos *p, int32_t *out) {
    if (!w) return CAE_VALUE_ERR_NOT_LOADED;
    if (p->in_check) return CAE_VALUE_ERR_IN_CHECK;

    int bucket = cae_nnue_bucket(p);
    int rc = cae_nnue_check_bucket(w, bucket);
    if (rc != CAE_VALUE_OK) return rc;

    CaeNnueAcc acc;
    uint8_t ft[CAE_NNUE_MAX_L1] __attribute__((aligned(32)));
    cae_nnue_refresh(w, p, &acc);
    int32_t psqt = cae_nnue_transform(w, &acc, p->side_to_move, bucket, ft);
    int32_t positional = cae_nnue_propagate(w, ft, bucket);
    *out = psqt / CAE_NNUE_OUTPUT_SCALE + positional / CAE_NNUE_OUTPUT_SCALE;
    return CAE_VALUE_OK;
}

static int cae_nnue_evaluate_cboard(const CaeNnueWeights *w, const CBoard *b, int32_t *out) {
    CaeNnuePos pos;
    int rc = cae_nnue_pos_from_cboard(b, &pos);
    if (rc != CAE_VALUE_OK) return rc;
    return cae_nnue_evaluate(w, &pos, out);
}

/* ================================================================
 * Weight loading — mmap, read-only, one mapping per path per process
 * ================================================================ */

#define CAE_NNUE_CACHE_SLOTS 4
static pthread_mutex_t cae_nnue_cache_lock = PTHREAD_MUTEX_INITIALIZER;
static CaeNnueWeights *cae_nnue_cache[CAE_NNUE_CACHE_SLOTS];

static void cae_nnue_err(char *err, size_t errlen, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));

static void cae_nnue_err(char *err, size_t errlen, const char *fmt, ...) {
    if (!err || errlen == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

static int cae_nnue_bind(CaeNnueWeights *w, char *err, size_t errlen) {
    const CaeNnuePackHeader *h = (const CaeNnuePackHeader *)w->map;

    if (memcmp(h->magic, CAE_NNUE_PACK_MAGIC, 8) != 0) {
        cae_nnue_err(err, errlen, "not an NNUE weight pack (bad magic)");
        return -1;
    }
    if (h->pack_version != CAE_NNUE_PACK_VERSION) {
        cae_nnue_err(err, errlen, "pack version %u, expected %u",
                     h->pack_version, CAE_NNUE_PACK_VERSION);
        return -1;
    }
    /* FATAL on any .nnue version but the one we implement. A foreign layout
     * would produce plausible numbers that are wrong everywhere. */
    if (h->nnue_version != CAE_NNUE_FILE_VERSION) {
        cae_nnue_err(err, errlen, ".nnue version 0x%08X, this build only accepts 0x%08X",
                     h->nnue_version, CAE_NNUE_FILE_VERSION);
        return -1;
    }
    if (!h->use_threats || h->threat_dims != CAE_NNUE_THREAT_DIMS
        || h->halfka_dims != CAE_NNUE_HALFKA_DIMS) {
        cae_nnue_err(err, errlen,
                     "only the big (threat) architecture is supported "
                     "(use_threats=%u halfka=%u threats=%u)",
                     h->use_threats, h->halfka_dims, h->threat_dims);
        return -1;
    }
    if (h->l1 > CAE_NNUE_MAX_L1 || h->fc0_outputs > CAE_NNUE_MAX_FC0_OUT
        || h->fc1_padded_in > CAE_NNUE_MAX_FC1_IN || h->fc1_outputs > CAE_NNUE_MAX_FC1_OUT
        || h->psqt_buckets != CAE_NNUE_PSQT_BUCKETS
        || h->layer_stacks != CAE_NNUE_LAYER_STACKS || (h->l1 % 32u) != 0u) {
        cae_nnue_err(err, errlen, "pack dimensions outside the supported range");
        return -1;
    }
    if (h->total_size != w->map_size) {
        cae_nnue_err(err, errlen, "pack says %llu bytes, file is %llu",
                     (unsigned long long)h->total_size, (unsigned long long)w->map_size);
        return -1;
    }

    const size_t sizes[11] = {
        (size_t)h->l1 * 2,
        (size_t)h->halfka_dims * h->l1 * 2,
        (size_t)h->halfka_dims * h->psqt_buckets * 4,
        (size_t)h->threat_dims * h->l1,
        (size_t)h->threat_dims * h->psqt_buckets * 4,
        (size_t)h->layer_stacks * h->fc0_outputs * 4,
        (size_t)h->layer_stacks * h->fc0_outputs * h->fc0_padded_in,
        (size_t)h->layer_stacks * h->fc1_outputs * 4,
        (size_t)h->layer_stacks * h->fc1_outputs * h->fc1_padded_in,
        (size_t)h->layer_stacks * 4,
        (size_t)h->layer_stacks * h->fc2_padded_in,
    };
    for (int i = 0; i < 11; i++) {
        if (h->off[i] < CAE_NNUE_HEADER_BYTES || h->off[i] + sizes[i] > w->map_size) {
            cae_nnue_err(err, errlen, "pack tensor %d runs outside the file", i);
            return -1;
        }
    }

    const uint8_t *base = (const uint8_t *)w->map;
    w->l1 = h->l1; w->l2 = h->l2; w->l3 = h->l3;
    w->halfka_dims = h->halfka_dims;
    w->threat_dims = h->threat_dims;
    w->layer_stacks = h->layer_stacks;
    w->psqt_buckets = h->psqt_buckets;
    w->fc0_outputs = h->fc0_outputs;
    w->fc0_padded_in = h->fc0_padded_in;
    w->fc1_outputs = h->fc1_outputs;
    w->fc1_padded_in = h->fc1_padded_in;
    w->fc2_padded_in = h->fc2_padded_in;
    w->net_hash = h->net_hash;
    w->ft_hash = h->ft_hash;

    w->ft_bias       = (const int16_t *)(base + h->off[0]);
    w->ft_weight     = (const int16_t *)(base + h->off[1]);
    w->ft_psqt       = (const int32_t *)(base + h->off[2]);
    w->threat_weight = (const int8_t  *)(base + h->off[3]);
    w->threat_psqt   = (const int32_t *)(base + h->off[4]);
    w->fc0_bias      = (const int32_t *)(base + h->off[5]);
    w->fc0_weight    = (const int8_t  *)(base + h->off[6]);
    w->fc1_bias      = (const int32_t *)(base + h->off[7]);
    w->fc1_weight    = (const int8_t  *)(base + h->off[8]);
    w->fc2_bias      = (const int32_t *)(base + h->off[9]);
    w->fc2_weight    = (const int8_t  *)(base + h->off[10]);

    for (int i = 0; i < 32; i++)
        snprintf(w->sha256_hex + i * 2, 3, "%02x", h->sha256[i]);
    w->sha256_hex[64] = '\0';

    if (cae_nnue_computed_threat_dims() != h->threat_dims) {
        cae_nnue_err(err, errlen,
                     "our attack geometry implies %u threat features, the pack has %u",
                     cae_nnue_computed_threat_dims(), h->threat_dims);
        return -1;
    }
    return 0;
}

/* Load (or share) a weight pack. The mapping is READ-ONLY and PRIVATE, so every
 * process on the box that maps the same file shares its physical pages through
 * the page cache; within a process the cache below hands out one mapping per
 * path, refcounted. */
static CaeNnueWeights *cae_nnue_load(const char *path, char *err, size_t errlen) {
    cae_nnue_init_tables();

    pthread_mutex_lock(&cae_nnue_cache_lock);
    for (int i = 0; i < CAE_NNUE_CACHE_SLOTS; i++) {
        if (cae_nnue_cache[i] && strcmp(cae_nnue_cache[i]->path, path) == 0) {
            cae_nnue_cache[i]->refcount++;
            pthread_mutex_unlock(&cae_nnue_cache_lock);
            return cae_nnue_cache[i];
        }
    }
    pthread_mutex_unlock(&cae_nnue_cache_lock);

    if (strlen(path) >= sizeof(((CaeNnueWeights *)0)->path)) {
        cae_nnue_err(err, errlen, "weight path too long");
        return NULL;
    }

    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        cae_nnue_err(err, errlen, "cannot open weight pack: %s", strerror(errno));
        return NULL;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < (off_t)CAE_NNUE_HEADER_BYTES) {
        cae_nnue_err(err, errlen, "weight pack is too small to hold a header");
        close(fd);
        return NULL;
    }
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) {
        cae_nnue_err(err, errlen, "mmap failed: %s", strerror(errno));
        return NULL;
    }

    CaeNnueWeights *w = (CaeNnueWeights *)calloc(1, sizeof(CaeNnueWeights));
    if (!w) {
        munmap(map, (size_t)st.st_size);
        cae_nnue_err(err, errlen, "out of memory");
        return NULL;
    }
    w->map = map;
    w->map_size = (size_t)st.st_size;
    snprintf(w->path, sizeof(w->path), "%s", path);
    if (cae_nnue_bind(w, err, errlen) != 0) {
        munmap(map, w->map_size);
        free(w);
        return NULL;
    }
    w->refcount = 1;

    pthread_mutex_lock(&cae_nnue_cache_lock);
    for (int i = 0; i < CAE_NNUE_CACHE_SLOTS; i++) {
        if (!cae_nnue_cache[i]) { cae_nnue_cache[i] = w; break; }
    }
    pthread_mutex_unlock(&cae_nnue_cache_lock);
    return w;
}

static void cae_nnue_release(CaeNnueWeights *w) {
    if (!w) return;
    pthread_mutex_lock(&cae_nnue_cache_lock);
    if (--w->refcount > 0) {
        pthread_mutex_unlock(&cae_nnue_cache_lock);
        return;
    }
    for (int i = 0; i < CAE_NNUE_CACHE_SLOTS; i++)
        if (cae_nnue_cache[i] == w) cae_nnue_cache[i] = NULL;
    pthread_mutex_unlock(&cae_nnue_cache_lock);
    munmap(w->map, w->map_size);
    free(w);
}

#endif /* CAE_NNUE_IMPL_H */
