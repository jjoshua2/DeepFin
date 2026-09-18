/* Experimental native neural-input boundary. No Python or Bend runtime types.
 * Reuses production CBoard history/feature/encoding semantics. Single-threaded
 * context: each call restores its selected repetition mode before replay. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "chess_anti_engine/encoding/_cboard_impl.h"
#include "chess_anti_engine/encoding/_features_impl.h"

#define MAX_GAME 128
#define MAX_PATH 32
#define MAX_PLANES 179
#define FULL 4672
#define COMPACT 1858

typedef struct {
    CBoard root;
    int mode, extra, rep_fix;
    int32_t mapping[FULL];
} NeuralRoot;

static int fail(char *error, uint32_t size, const char *message) {
    if (error && size) snprintf(error, size, "%s", message);
    return 0;
}
static uint32_t move_key(const CBoard *b, int index) {
    PolicyMove m = POLICY_LUT[b->turn][index];
    int pawn = !!(b->bb[PAWN] & sq_bit(m.from_sq));
    unsigned promo = pawn && (m.to_sq / 8 == 0 || m.to_sq / 8 == 7) ?
        (m.promotion == PROMO_MAYBE_QUEEN ? 4u : (unsigned)(m.promotion - 1)) : 0;
    unsigned flag = pawn && m.to_sq == b->ep_square ? 1u :
        ((b->bb[KING] & sq_bit(m.from_sq)) && abs(m.to_sq - m.from_sq) == 2 ? 2u : 0u);
    return (uint32_t)m.from_sq | (uint32_t)m.to_sq << 6 | promo << 12 | flag << 15;
}
static int push_key(CBoard *b, uint32_t key) {
    int indices[CBOARD_MAX_LEGAL_MOVES];
    int count = cboard_legal_move_indices(b, indices, 1);
    for (int i = 0; i < count; ++i) if (move_key(b, indices[i]) == key) {
        cboard_push_index(b, indices[i]);
        return 1;
    }
    return 0;
}
static int same_board(const CBoard *b, const uint64_t *bits, const uint32_t *meta) {
    return memcmp(b->bb, bits, 6 * sizeof(uint64_t)) == 0 &&
        b->occ[WHITE_C] == bits[6] && b->occ[BLACK_C] == bits[7] &&
        b->turn == meta[0] && b->castling == meta[1] &&
        (b->ep_square < 0 ? 64u : (uint32_t)b->ep_square) == meta[2];
}
static int board_ok(const CBoard *b) {
    uint64_t pieces = 0, occ = b->occ[0] | b->occ[1];
    if (b->occ[0] & b->occ[1]) return 0;
    for (int i = 0; i < 6; ++i) {
        if (pieces & b->bb[i]) return 0;
        pieces |= b->bb[i];
    }
    if (pieces != occ || popcount64(occ) > 32 || (b->bb[PAWN] & UINT64_C(0xff000000000000ff))) return 0;
    if (popcount64(b->bb[KING] & b->occ[0]) != 1 || popcount64(b->bb[KING] & b->occ[1]) != 1) return 0;
    const int corners[] = {7, 0, 63, 56};
    for (int i = 0; i < 4; ++i) if (b->castling & (1 << i)) {
        uint64_t own = b->occ[i < 2 ? 1 : 0];
        if (!(b->bb[ROOK] & own & sq_bit(corners[i])) || !(b->bb[KING] & own & sq_bit(i < 2 ? 4 : 60))) return 0;
    }
    if (b->ep_square >= 0) {
        int ep = b->ep_square;
        if (ep / 8 != (b->turn ? 5 : 2) || (occ & sq_bit(ep))) return 0;
        if (!(b->bb[PAWN] & b->occ[1-b->turn] & sq_bit(ep ^ 8))) return 0;
        if (occ & sq_bit(b->turn ? ep + 8 : ep - 8)) return 0;
    }
    CBoard other = *b; other.turn = 1-b->turn;
    return !cboard_in_check(&other);
}

/* bits: six piece masks, white, black. meta: turn, rights, raw ep (64=none),
 * halfmove, absolute ply. game[] is chronological packed moves from this seed.
 * mapping[] comes from DeepFin's canonical FULL_TO_COMPACT_POLICY, not a copied
 * table. Its complete identity is checked against the Python source by tests. */
void *df_root_new(const uint64_t *bits, const uint32_t *meta, const uint32_t *game,
    uint32_t count, uint32_t mode, uint32_t extra, uint32_t rep_fix,
    const int32_t *mapping, char *error, uint32_t error_size) {
    if (!bits || !meta || !mapping || (count && !game) || count > MAX_GAME || mode > 2 ||
        (extra != 34 && extra != 63 && extra != 67) || rep_fix > 1 || meta[0] > 1 ||
        meta[1] > 15 || meta[2] > 64 || meta[3] > 255 || meta[4] > 65535 - MAX_GAME - MAX_PATH) {
        fail(error, error_size, "invalid native root configuration"); return NULL;
    }
    int next = 0;
    for (int i = 0; i < FULL; ++i) if (mapping[i] != -1 && mapping[i] != next++) {
        fail(error, error_size, "invalid compact policy map"); return NULL;
    }
    if (next != COMPACT) { fail(error, error_size, "invalid compact policy map size"); return NULL; }
    NeuralRoot *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) { fail(error, error_size, "root allocation failed"); return NULL; }
    cboard_init_all(); g_history_rep_fix = (int)rep_fix;
    ctx->mode = (int)mode; ctx->extra = (int)extra; ctx->rep_fix = (int)rep_fix;
    memcpy(ctx->mapping, mapping, sizeof(ctx->mapping));
    CBoard *b = &ctx->root;
    memcpy(b->bb, bits, 6*sizeof(uint64_t)); b->occ[1] = bits[6]; b->occ[0] = bits[7];
    b->turn = (int8_t)meta[0]; b->castling = (uint8_t)meta[1]; b->ep_square = meta[2] == 64 ? -1 : (int8_t)meta[2];
    b->halfmove_clock = (uint8_t)meta[3]; b->ply = (uint16_t)meta[4];
    cboard_reset_hist_ep(b); b->hash = cboard_compute_hash(b);
    if (!board_ok(b)) { free(ctx); fail(error, error_size, "invalid seed board"); return NULL; }
    for (uint32_t i = 0; i < count; ++i) if (!push_key(b, game[i])) {
        free(ctx); fail(error, error_size, "illegal game-history move"); return NULL;
    }
    return ctx;
}
void df_root_free(void *handle) { free(handle); }

/* Transactional output: no caller buffer is touched unless the full path,
 * leaf identity and bijective legal action set validate. Root never mutates. */
int df_prepare(void *handle, const uint32_t *path, uint32_t depth,
    const uint64_t *bits, const uint32_t *meta, const uint32_t *keys, uint32_t count,
    float *planes, uint32_t plane_capacity, uint32_t *full, uint32_t *compact,
    char *error, uint32_t error_size) {
    NeuralRoot *ctx = handle;
    if (!ctx || !bits || !meta || !planes || !full || !compact || (depth && !path) ||
        (count && !keys) || depth > MAX_PATH || count > 256 ||
        plane_capacity < (uint32_t)(112+ctx->extra)*64) return fail(error, error_size, "invalid native request buffers");
    g_history_rep_fix = ctx->rep_fix;
    CBoard board = ctx->root;
    for (uint32_t i = 0; i < depth; ++i) if (!push_key(&board, path[i]))
        return fail(error, error_size, "illegal search-path move");
    if (!same_board(&board, bits, meta)) return fail(error, error_size, "leaf board/path mismatch");
    int legal[CBOARD_MAX_LEGAL_MOVES];
    int n = cboard_legal_move_indices(&board, legal, 1);
    if (n != (int)count) return fail(error, error_size, "incomplete legal action set");
    uint32_t indices[256], compact_indices[256];
    unsigned char seen[256] = {0};
    for (uint32_t i = 0; i < count; ++i) {
        int found = -1;
        for (int j = 0; j < n; ++j) if (move_key(&board, legal[j]) == keys[i]) { found = j; break; }
        if (found < 0 || seen[found]) return fail(error, error_size, "illegal or duplicate packed action");
        seen[found] = 1; indices[i] = (uint32_t)legal[found];
        int32_t mapped = ctx->mapping[indices[i]];
        if (mapped < 0 || mapped >= COMPACT) return fail(error, error_size, "unmapped legal policy action");
        compact_indices[i] = (uint32_t)mapped;
    }
    float output[MAX_PLANES * 64] = {0};
    if (ctx->mode == 2) cboard_fill_lc0_112_root_legacy_meta(&board, output);
    else if (ctx->mode == 1) cboard_fill_lc0_112_root(&board, output);
    else cboard_fill_lc0_112(&board, output);
    cboard_compute_features_ext(&board, output + 112*64, ctx->extra);
    memcpy(planes, output, (size_t)(112+ctx->extra)*64*sizeof(float));
    memcpy(full, indices, count*sizeof(uint32_t)); memcpy(compact, compact_indices, count*sizeof(uint32_t));
    return 1;
}
