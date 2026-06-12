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
 *   odd  -> RAW BOARD: interpret the next 68 bytes as arbitrary bitboards +
 *           state, exactly what ``CBoard.from_raw`` accepts from Python
 *           (six piece bitboards plus independent occ_white/occ_black).
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
    if (size < 68)
        return;
    CBoard b;
    memset(&b, 0, sizeof(CBoard));
    uint64_t v[8];
    memcpy(v, data, 64);
    for (int p = 0; p < 6; p++)
        b.bb[p] = v[p];
    /* occ_white/occ_black are independent from_raw arguments, so fuzz them
     * independently too: occupied squares with no piece bit, overlapping
     * colors, piece bits outside both occupancies — all reachable states. */
    b.occ[WHITE_C] = v[6];
    b.occ[BLACK_C] = v[7];
    b.turn = (data[64] & 1) ? WHITE_C : BLACK_C;
    b.castling = (uint8_t)(data[65] & 0xF);
    /* Feed the signed byte through the same normalization from_raw applies,
     * so out-of-range inputs (64..127, negatives) exercise the boundary. */
    b.ep_square = cboard_sanitize_ep((int8_t)data[66]);
    b.halfmove_clock = data[67];
    b.hash = cboard_compute_hash(&b);

    int indices[256];
    /* Movegen on garbage state is the target; the results are unused. */
    int n = cboard_legal_move_indices(&b, indices, 1);
    (void)n;
    exercise_queries(&b);
    /* Push a handful of self-reported-legal moves on the garbage board. */
    for (size_t i = 68; i + 1 < size && i < 88; i += 2) {
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
