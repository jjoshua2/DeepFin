/*
 * Coverage-guided libFuzzer harness for the pure-C CBoard implementation.
 *
 * Build/run via scripts/fuzz/run_fuzz.sh. Two modes selected by the first
 * input byte:
 *
 *   even -> LEGAL WALK: start from the standard position and consume the
 *           remaining bytes as candidate policy indices; push each one that
 *           the C movegen itself reports legal. Exercises movegen, push,
 *           repetition/terminal logic, and FEN emission along realistic
 *           game trajectories.
 *
 *   odd  -> RAW BOARD: interpret the next 52 bytes as arbitrary bitboards +
 *           state, exactly what ``CBoard.from_raw`` accepts from Python.
 *           The C layer must be memory-safe (no ASAN/UBSAN findings) on any
 *           such input even when it is not a reachable chess position.
 *
 * Crashes/UB reproduce with: ./cboard_fuzz <crash-file>
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "_cboard_impl.h"

static int initialized = 0;

static void init_once(void) {
    if (!initialized) {
        cboard_init_all();
        initialized = 1;
    }
}

static void startpos(CBoard *b) {
    memset(b, 0, sizeof(CBoard));
    b->bb[PAWN]   = 0x00FF00000000FF00ULL;
    b->bb[KNIGHT] = 0x4200000000000042ULL;
    b->bb[BISHOP] = 0x2400000000000024ULL;
    b->bb[ROOK]   = 0x8100000000000081ULL;
    b->bb[QUEEN]  = 0x0800000000000008ULL;
    b->bb[KING]   = 0x1000000000000010ULL;
    b->occ[WHITE_C] = 0x000000000000FFFFULL;
    b->occ[BLACK_C] = 0xFFFF000000000000ULL;
    b->turn = WHITE_C;
    b->castling = 0xF;
    b->ep_square = -1;
    b->hash = cboard_compute_hash(b);
}

static void exercise_queries(CBoard *b) {
    char fen[128];
    cboard_to_fen(b, fen, sizeof(fen));
    (void)cboard_is_game_over(b);
    (void)cboard_terminal_value(b);
    (void)cboard_compute_hash(b);
}

static void legal_walk(const uint8_t *data, size_t size) {
    CBoard b;
    startpos(&b);
    int indices[256];
    for (size_t i = 0; i + 1 < size; i += 2) {
        int n = cboard_legal_move_indices(&b, indices, 0);
        if (n <= 0)
            break;
        int candidate = ((data[i] << 8) | data[i + 1]) % 4672;
        int legal = 0;
        for (int k = 0; k < n; k++) {
            if (indices[k] == candidate) { legal = 1; break; }
        }
        /* Bias toward progress: an illegal candidate selects a legal move
         * deterministically so the fuzzer reaches deep positions. */
        if (!legal)
            candidate = indices[candidate % n];
        cboard_push_index(&b, candidate);
        if ((i & 0xF) == 0)
            exercise_queries(&b);
    }
    exercise_queries(&b);
}

static void raw_board(const uint8_t *data, size_t size) {
    if (size < 52)
        return;
    CBoard b;
    memset(&b, 0, sizeof(CBoard));
    uint64_t v[6];
    memcpy(v, data, 48);
    for (int p = 0; p < 6; p++)
        b.bb[p] = v[p];
    /* occ derived from data so white/black overlap arbitrarily, as from_raw
     * allows. */
    b.occ[WHITE_C] = v[0] ^ v[2] ^ v[4];
    b.occ[BLACK_C] = v[1] ^ v[3] ^ v[5];
    b.turn = (data[48] & 1) ? WHITE_C : BLACK_C;
    b.castling = (uint8_t)(data[49] & 0xF);
    int ep = (int)data[50];
    b.ep_square = (int8_t)(ep < 64 ? ep : -1);
    b.halfmove_clock = data[51];
    b.hash = cboard_compute_hash(&b);

    int indices[256];
    int n = cboard_legal_move_indices(&b, indices, 1);
    for (int k = 0; k < n && k < 8; k++)
        ; /* movegen on garbage state is the target; results are unused */
    exercise_queries(&b);
    /* Push a handful of self-reported-legal moves on the garbage board. */
    for (size_t i = 52; i + 1 < size && i < 72; i += 2) {
        n = cboard_legal_move_indices(&b, indices, 0);
        if (n <= 0)
            break;
        cboard_push_index(&b, indices[((data[i] << 8) | data[i + 1]) % n]);
    }
    exercise_queries(&b);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 2)
        return 0;
    init_once();
    if (data[0] & 1)
        raw_board(data + 1, size - 1);
    else
        legal_walk(data + 1, size - 1);
    return 0;
}
