/* Test/build boundary only. Candidate move generation and move application
 * never call CBoard. A separately built oracle DOES use CBoard, deliberately. */
#include "support.h"
#include <errno.h>
#include <inttypes.h>
#define DEEPFIN_FAST_SLIDERS 1
#define init_attack_tables init_attack_tables_reference
#define slider_attacks slider_attacks_reference
#define bishop_attacks bishop_attacks_reference
#define rook_attacks rook_attacks_reference
#define queen_attacks queen_attacks_reference
#define is_attacked_by is_attacked_by_reference
#include "chess_anti_engine/encoding/_cboard_impl.h"

static int input_ok(const LegalInput *in) {
    if (in->turn > 1 || in->rights > 15 || in->ep > 64 || in->mode > 2) return 0;
    if (in->depth > (in->mode == 2 ? 256u : 5u)) return 0;
    uint64_t occupied = in->bb[6] | in->bb[7], pieces = 0;
    if (in->bb[6] & in->bb[7]) return 0;
    for (unsigned p = 0; p < 6; p++) {
        if (pieces & in->bb[p]) return 0;
        pieces |= in->bb[p];
    }
    if (pieces != occupied || popcount64(occupied) > 32) return 0;
    if (popcount64(in->bb[5] & in->bb[6]) != 1 || popcount64(in->bb[5] & in->bb[7]) != 1) return 0;
    if (in->bb[0] & UINT64_C(0xff000000000000ff)) return 0;
    const unsigned homes[4] = {7, 0, 63, 56};
    for (unsigned r = 0; r < 4; r++) if (in->rights & (1u << r)) {
        uint64_t color = in->bb[r < 2 ? 6 : 7];
        if (!(in->bb[3] & color & sq_bit((int)homes[r]))) return 0;
        if (!(in->bb[5] & color & sq_bit(r < 2 ? 4 : 60))) return 0;
    }
    if (in->ep != 64) {
        if (in->ep / 8 != (in->turn ? 5u : 2u) || (occupied & sq_bit((int)in->ep))) return 0;
        if (!(in->bb[0] & in->bb[in->turn ? 7 : 6] & sq_bit((int)(in->ep ^ 8)))) return 0;
        unsigned origin = in->turn ? in->ep + 8 : in->ep - 8;
        if (occupied & sq_bit((int)origin)) return 0;
    }
    return 1;
}

int legal_read_input(LegalInput *in) {
    char line[512];
    if (!fgets(line, sizeof(line), stdin)) return 0;
    if (!strchr(line, '\n') && !feof(stdin)) return -1;
    uint64_t fields[14];
    char *token = strtok(line, " \t\r\n");
    for (unsigned i = 0; i < 14; i++) {
        if (!token || !*token || strlen(token) > 16 || strspn(token, "0123456789abcdefABCDEF") != strlen(token)) return -1;
        errno = 0;
        fields[i] = strtoull(token, NULL, 16);
        if (errno) return -1;
        token = strtok(NULL, " \t\r\n");
    }
    if (token) return -1;
    for (unsigned i = 0; i < 3; i++) if (fields[i] > UINT32_MAX || fields[11+i] > UINT32_MAX) return -1;
    in->depth = (uint32_t)fields[0]; in->mode = (uint32_t)fields[1]; in->seed = (uint32_t)fields[2];
    memcpy(in->bb, fields + 3, sizeof(in->bb));
    in->turn = (uint32_t)fields[11]; in->rights = (uint32_t)fields[12]; in->ep = (uint32_t)fields[13];
    return input_ok(in) ? 1 : -1;
}

void legal_make_tables(uint64_t *out) {
    cboard_init_all();
    memset(out, 0, LEGAL_TABLE_WORDS * sizeof(*out));
    uint32_t at = LEGAL_TABLE_START;
    for (int bishop = 0; bishop < 2; bishop++) for (int sq = 0; sq < 64; sq++) {
        uint64_t mask = bishop ? DEEPFIN_BISHOP_MASKS[sq] : DEEPFIN_ROOK_MASKS[sq];
        unsigned entries = 1u << popcount64(mask), key = (unsigned)(64 * bishop + sq);
        if (at + entries > LEGAL_TABLE_WORDS) abort();
        out[key] = mask; out[128 + key] = at;
        uint64_t subset = 0;
        for (unsigned i = 0; i < entries; i++) {
            out[at++] = slider_attacks_reference(sq, subset, bishop);
            subset = (subset - mask) & mask;
        }
        if (subset != 0) abort();
    }
    if (at != LEGAL_TABLE_START + DEEPFIN_ROOK_TABLE_SIZE + DEEPFIN_BISHOP_TABLE_SIZE) abort();
    for (int sq = 0; sq < 64; sq++) {
        out[256 + sq] = KNIGHT_ATTACKS[sq]; out[320 + sq] = KING_ATTACKS[sq];
        out[384 + sq] = PAWN_ATTACKS[1][sq]; out[448 + sq] = PAWN_ATTACKS[0][sq];
    }
}

#ifdef LEGAL_ORACLE
#ifdef LEGAL_BENCH
#include "bench_clock.h"
#endif
static CBoard from_input(const LegalInput *in) {
    CBoard b = {0};
    memcpy(b.bb, in->bb, 6 * sizeof(uint64_t));
    b.occ[1] = in->bb[6]; b.occ[0] = in->bb[7];
    b.turn = (int8_t)in->turn; b.castling = (uint8_t)in->rights;
    b.ep_square = in->ep == 64 ? -1 : (int8_t)in->ep;
    cboard_reset_hist_ep(&b);
    b.hash = cboard_compute_hash(&b);
    return b;
}
static uint64_t reference_perft(const CBoard *b, unsigned depth) {
    if (!depth) return 1;
    int moves[CBOARD_MAX_LEGAL_MOVES];
    int count = cboard_legal_move_indices(b, moves, 1);
    if (depth == 1) return (uint64_t)count;
    uint64_t total = 0;
    for (int i = 0; i < count; i++) {
        CBoard child = *b;
        cboard_push_index(&child, moves[i]);
        total += reference_perft(&child, depth - 1);
    }
    return total;
}
static void print_word(uint64_t x) {
    printf("%" PRIu32 " %" PRIu32, (uint32_t)(x >> 32), (uint32_t)x);
}
static void print_board(const CBoard *b) {
    for (int i = 0; i < 6; i++) { print_word(b->bb[i]); putchar(' '); }
    print_word(b->occ[1]); putchar(' '); print_word(b->occ[0]);
    printf(" %d %u %u\n", b->turn, b->castling, b->ep_square < 0 ? 64u : (unsigned)b->ep_square);
}
int main(void) {
    cboard_init_all();
    LegalInput in;
    if (legal_read_input(&in) != 1) { fputs("invalid oracle input\n", stderr); return 2; }
    CBoard b = from_input(&in);
    uint64_t us = b.occ[b.turn];
    if (is_attacked_by(lsb64(b.bb[KING] & b.occ[1-b.turn]), b.occ[0] | b.occ[1],
        b.bb[PAWN] & us, b.bb[KNIGHT] & us, b.bb[BISHOP] & us,
        b.bb[ROOK] & us, b.bb[QUEEN] & us, b.bb[KING] & us, b.turn)) {
        fputs("invalid input: nonmoving king is attacked\n", stderr); return 2;
    }
#ifdef LEGAL_BENCH
    /* Same CBoard legal generation/copy/push traversal as the parity oracle.
     * Native-target compilation selects the real PEXT/magic backend above. */
    volatile uint64_t warmup = reference_perft(&b, in.depth);
    printf("warmup %" PRIu32 " %" PRIu32 "\n", (uint32_t)(warmup >> 32), (uint32_t)warmup);
    fflush(stdout);
    /* Tell the optimizer the second input must be read again. */
    __asm__ __volatile__("" : "+m"(b) : : "memory");
    perft_clock_start();
    uint64_t nodes = reference_perft(&b, in.depth);
    perft_clock_finish(DEEPFIN_SLIDER_BACKEND_NAME, in.depth, nodes);
    return 0;
#endif
    if (in.mode == 0 && in.depth == 0) { puts("nodes 0 1"); return 0; }
    int moves[CBOARD_MAX_LEGAL_MOVES];
    int count = cboard_legal_move_indices(&b, moves, 1);
    if (in.mode == 2) {
        puts(count ? "playing" : (cboard_in_check(&b) ? "checkmate" : "stalemate"));
        return 0;
    }
    uint64_t total = 0;
    for (int i = 0; i < count; i++) {
        PolicyMove m = POLICY_LUT[b.turn][moves[i]];
        int pawn = !!(b.bb[PAWN] & sq_bit(m.from_sq));
        unsigned promotion = pawn && (m.to_sq / 8 == 0 || m.to_sq / 8 == 7) ?
            (m.promotion == PROMO_MAYBE_QUEEN ? 4u : (unsigned)(m.promotion - 1)) : 0u;
        unsigned flag = pawn && m.to_sq == b.ep_square ? 1u :
            ((b.bb[KING] & sq_bit(m.from_sq)) && abs(m.to_sq - m.from_sq) == 2 ? 2u : 0u);
        CBoard child = b; cboard_push_index(&child, moves[i]);
        printf("%s %u %u %u %u ", in.mode == 0 ? "divide" : "move", (unsigned)m.from_sq, (unsigned)m.to_sq, promotion, flag);
        if (in.mode == 0) {
            uint64_t n = reference_perft(&child, in.depth - 1); total += n;
            print_word(n); putchar('\n');
        } else print_board(&child);
    }
    if (in.mode == 0) { fputs("nodes ", stdout); print_word(total); putchar('\n'); }
    else puts("end");
    return 0;
}
#endif
