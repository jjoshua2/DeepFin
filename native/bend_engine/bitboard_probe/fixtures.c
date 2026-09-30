/* Test-only fixture producer: reuse DeepFin's actual masks/magics/attack
 * tables and its independently retained ray walker. Compile this TU without
 * BMI2 so both metadata families are present. Bend's generated TU can still
 * be compiled for BMI2/native. No copied magic constants, no Python runtime. */
#define DEEPFIN_FAST_SLIDERS 1
#define init_attack_tables init_attack_tables_reference
#define slider_attacks slider_attacks_reference
#define bishop_attacks bishop_attacks_reference
#define rook_attacks rook_attacks_reference
#define queen_attacks queen_attacks_reference
#define is_attacked_by is_attacked_by_reference
#include "chess_anti_engine/encoding/_cboard_impl.h"

#if DEEPFIN_SLIDER_USE_PEXT
#error "fixtures.c needs the magic metadata: compile without BMI2"
#endif

enum { HEADER = 8, RANDOM_CASES = 64, MAX_WORDS = 32768 };

static uint64_t next_occupancy(uint64_t x) {
    x ^= x << 13;
    x ^= x >> 7;
    return x ^ (x << 17);
}

static void store_case(uint64_t *out, uint32_t at, int sq, int bishop,
                       uint64_t occupied) {
    uint64_t expected = slider_attacks_reference(sq, occupied, bishop);
    if (slider_attacks(sq, occupied, bishop) != expected) {
        fprintf(stderr, "CBoard oracle disagreement: kind=%d square=%d\n", bishop, sq);
        abort();
    }
    out[at] = occupied;
    out[at + 1] = expected;
}

/* Header: mask, magic, shift, entries, cases, square, bishop, format marker.
 * Next come PEXT-order and magic-order tables, each entries words long.
 * Finally (occupied, expected_attack) pairs. All counts are bounded here. */
uint32_t deepfin_u64_fixture(uint32_t bishop, uint32_t sq, uint64_t *out,
                            uint32_t capacity) {
    if (bishop > 1 || sq >= 64 || capacity < MAX_WORDS || out == NULL) {
        fprintf(stderr, "invalid U64 slider fixture request\n");
        abort();
    }
    cboard_init_all();
    uint64_t mask = bishop ? DEEPFIN_BISHOP_MASKS[sq] : DEEPFIN_ROOK_MASKS[sq];
    uint64_t magic = bishop ? DEEPFIN_BISHOP_MAGICS[sq] : DEEPFIN_ROOK_MAGICS[sq];
    uint32_t shift = bishop ? DEEPFIN_BISHOP_SHIFTS[sq] : DEEPFIN_ROOK_SHIFTS[sq];
    uint32_t offset = bishop ? DEEPFIN_BISHOP_OFFSETS[sq] : DEEPFIN_ROOK_OFFSETS[sq];
    const uint64_t *attacks = bishop ? DEEPFIN_BISHOP_ATTACKS : DEEPFIN_ROOK_ATTACKS;
    uint32_t entries = 1u << (64 - shift);
    uint32_t cases = 2 * entries + RANDOM_CASES;
    uint32_t start = HEADER + 2 * entries;
    uint32_t used = start + 2 * cases;
    if (used > capacity) abort();
    out[0] = mask; out[1] = magic; out[2] = shift; out[3] = entries;
    out[4] = cases; out[5] = sq; out[6] = bishop; out[7] = 0x55363431u;
    memcpy(out + HEADER + entries, attacks + offset, entries * sizeof(uint64_t));

    /* Carry-rippler order is PDEP(index, mask) order, but neither PEXT nor
     * PDEP is used to produce it. This avoids a circular intrinsic oracle. */
    uint64_t occupied = 0;
    for (uint32_t i = 0; i < entries; i++) {
        out[HEADER + i] = slider_attacks_reference((int)sq, occupied, (int)bishop);
        store_case(out, start + 4 * i, (int)sq, (int)bishop, occupied);
        store_case(out, start + 4 * i + 2, (int)sq, (int)bishop, occupied | ~mask);
        occupied = (occupied - mask) & mask;
    }
    if (occupied != 0) abort();
    uint64_t state = UINT64_C(0x9e3779b97f4a7c15) ^ ((uint64_t)bishop << 32) ^ sq;
    for (uint32_t i = 0; i < RANDOM_CASES; i++) {
        state = next_occupancy(state);
        uint64_t sample = state;
        if (i % 3 == 1) sample &= sample >> 7;
        if (i % 3 == 2) sample |= sample << 11;
        store_case(out, start + 4 * entries + 2 * i, (int)sq, (int)bishop, sample);
    }
    /* Test that the checker actually rejects incorrect table contents. */
    const char *corrupt = getenv("BEND_U64_CORRUPT");
    if (bishop == 0 && sq == 0 && corrupt && strcmp(corrupt, "1") == 0)
        out[HEADER] ^= UINT64_C(1) << 63;
    return used;
}
