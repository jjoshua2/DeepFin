/*
 * Table-backed sliding attacks for CBoard.
 *
 * Native BMI2 builds use PEXT indexing. Portable builds use generated magic
 * multipliers with the same relevant occupancy masks and packed table sizes.
 * The build preserves the legacy ray walker as *_reference; it is used to
 * populate and exhaustively verify these tables at module initialization.
 */
#ifndef DEEPFIN_SLIDER_ATTACKS_IMPL_H
#define DEEPFIN_SLIDER_ATTACKS_IMPL_H

#ifdef DEEPFIN_FAST_SLIDERS

#undef init_attack_tables
#undef slider_attacks
#undef bishop_attacks
#undef rook_attacks
#undef queen_attacks
#undef is_attacked_by

#if defined(__BMI2__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define DEEPFIN_SLIDER_USE_PEXT 1
#else
#define DEEPFIN_SLIDER_USE_PEXT 0
#endif

#define DEEPFIN_ROOK_TABLE_SIZE 102400
#define DEEPFIN_BISHOP_TABLE_SIZE 5248

static uint64_t DEEPFIN_ROOK_MASKS[64];
static uint64_t DEEPFIN_BISHOP_MASKS[64];
static uint32_t DEEPFIN_ROOK_OFFSETS[64];
static uint32_t DEEPFIN_BISHOP_OFFSETS[64];
static uint8_t DEEPFIN_ROOK_SHIFTS[64];
static uint8_t DEEPFIN_BISHOP_SHIFTS[64];
static uint64_t DEEPFIN_ROOK_ATTACKS[DEEPFIN_ROOK_TABLE_SIZE];
static uint64_t DEEPFIN_BISHOP_ATTACKS[DEEPFIN_BISHOP_TABLE_SIZE];
static int deepfin_slider_tables_initialized = 0;

#if !DEEPFIN_SLIDER_USE_PEXT
static const uint64_t DEEPFIN_ROOK_MAGICS[64] = {
    0x0080018215204000ULL, 0x1140100040002000ULL, 0x020010408188A200ULL, 0x0100082100100004ULL,
    0x0A00100200208804ULL, 0x0100080400010002ULL, 0x2200440200108801ULL, 0xC080022656800100ULL,
    0x0144800840002080ULL, 0x0480804000802008ULL, 0x0211004104200010ULL, 0x0011804800821000ULL,
    0x0011000800041300ULL, 0x0102000402001008ULL, 0x40040001105A4824ULL, 0x4042800441000480ULL,
    0xC00420800388C000ULL, 0x2C3000400C201040ULL, 0x800A020014802440ULL, 0x1800818008005000ULL,
    0x8009010005100800ULL, 0x0000808002000400ULL, 0xC8000400100108C2ULL, 0x0003020004006081ULL,
    0x4040042180088640ULL, 0x0C00400080200080ULL, 0x4000200080801000ULL, 0x0000080080100084ULL,
    0x0008000404004020ULL, 0x1081004900140022ULL, 0x0040480C00412A10ULL, 0x001194A20000C304ULL,
    0x8200401020800080ULL, 0x2400401000402000ULL, 0x1000200080801000ULL, 0x0000804801801000ULL,
    0x3188810401802800ULL, 0x0084008004800200ULL, 0x0840800200800100ULL, 0x10800C9102000464ULL,
    0x2244248040008004ULL, 0x401000402000C000ULL, 0x1018200010008080ULL, 0x4000100008008080ULL,
    0x0140080011010004ULL, 0x8002002904160010ULL, 0x004E020001008080ULL, 0x090100208A470002ULL,
    0x8404842200450200ULL, 0x4081004000802100ULL, 0x4081044020001100ULL, 0x0200200810050100ULL,
    0x1408800400080080ULL, 0xA000040080020080ULL, 0x810401188A100C00ULL, 0x0000250414804200ULL,
    0x1000802200104102ULL, 0x4025001082400227ULL, 0x2100408220100A02ULL, 0x4002882104100101ULL,
    0xA002001020040802ULL, 0xC002001004010802ULL, 0x0820501118061084ULL, 0x00000D0284104022ULL,
};

static const uint64_t DEEPFIN_BISHOP_MAGICS[64] = {
    0x0002046448004100ULL, 0x2104040444002002ULL, 0x1084010421001040ULL, 0x0404050210808290ULL,
    0x44040420002C0000ULL, 0xA008443004088010ULL, 0x0000808820120000ULL, 0x8082008044022002ULL,
    0x0120094230020200ULL, 0x008A040800841080ULL, 0x4A00A101340083C2ULL, 0x2000440408801000ULL,
    0x0004085840210400ULL, 0x0100051012100000ULL, 0x00020B1808120928ULL, 0x618200808088200DULL,
    0x0822001020B10101ULL, 0x3002089002020420ULL, 0x04A1089000420042ULL, 0x0004084804121000ULL,
    0x0204902404200C20ULL, 0x0522000090442002ULL, 0x2002000048048400ULL, 0x004040008200B004ULL,
    0x0020084010428800ULL, 0x4181207044240410ULL, 0x0811CB0108120401ULL, 0x0024010010200880ULL,
    0x2020840000802002ULL, 0x0801084002005000ULL, 0x40010A0009009000ULL, 0x0000828018220820ULL,
    0x000442A000400480ULL, 0x0042014401204811ULL, 0x4002402800100043ULL, 0x3200C04800028200ULL,
    0x0020020400028082ULL, 0x8C01100080810040ULL, 0x0002140100004810ULL, 0x0011040828A08208ULL,
    0x0044100804020800ULL, 0x2005011022C01009ULL, 0x0844420250002100ULL, 0x0004020202000428ULL,
    0x8108440408200402ULL, 0x1890101018205042ULL, 0x08602802004800A1ULL, 0x001022020020A040ULL,
    0x400200D008081C08ULL, 0x4002004402082409ULL, 0x1426021A01040400ULL, 0x0800020420881808ULL,
    0x1004004050410114ULL, 0x2003410801010400ULL, 0x102048B001004281ULL, 0x0202080214204084ULL,
    0x3104240108080200ULL, 0x0040508044022000ULL, 0x00202810A4221800ULL, 0x480003410120A800ULL,
    0x8040008011420200ULL, 0x0011810803080200ULL, 0x0000204210010900ULL, 0x100808C108020010ULL,
};
#endif

static uint64_t deepfin_slider_relevant_mask(int sq, int bishop_like) {
    uint64_t mask = 0ULL;
    int start = bishop_like ? 1 : 0;
    for (int d = start; d < 8; d += 2) {
        int f = sq_file(sq), r = sq_rank(sq);
        int ray[7], n = 0;
        for (;;) {
            f += RAY_DF[d];
            r += RAY_DR[d];
            if (f < 0 || f > 7 || r < 0 || r > 7) break;
            ray[n++] = make_sq(f, r);
        }
        /* An edge blocker cannot change attacks beyond the edge square. */
        for (int i = 0; i + 1 < n; i++)
            mask |= sq_bit(ray[i]);
    }
    return mask;
}

static inline uint32_t deepfin_slider_index(
    uint64_t occupied, uint64_t mask, uint64_t magic, uint8_t shift
) {
#if DEEPFIN_SLIDER_USE_PEXT
    (void)magic;
    (void)shift;
    return (uint32_t)_pext_u64(occupied, mask);
#else
    return (uint32_t)(((occupied & mask) * magic) >> shift);
#endif
}

static void deepfin_slider_init_one(
    int sq, int bishop_like, uint64_t mask, uint32_t offset, uint8_t shift,
    uint64_t magic, uint64_t *table, uint32_t table_size
) {
    uint64_t subset = 0ULL;
    do {
        uint32_t idx = deepfin_slider_index(subset, mask, magic, shift);
        if (offset + idx >= table_size)
            abort();
        uint64_t attack = slider_attacks_reference(sq, subset, bishop_like);
#if !DEEPFIN_SLIDER_USE_PEXT
        uint64_t prior = table[offset + idx];
        if (prior != 0ULL && prior != attack)
            abort();
#endif
        table[offset + idx] = attack;
        subset = (subset - mask) & mask;
    } while (subset != 0ULL);
}

static inline uint64_t deepfin_bishop_lookup(int sq, uint64_t occupied) {
    uint64_t mask = DEEPFIN_BISHOP_MASKS[sq];
#if DEEPFIN_SLIDER_USE_PEXT
    uint32_t idx = (uint32_t)_pext_u64(occupied, mask);
#else
    uint32_t idx = (uint32_t)(((occupied & mask) * DEEPFIN_BISHOP_MAGICS[sq])
                              >> DEEPFIN_BISHOP_SHIFTS[sq]);
#endif
    return DEEPFIN_BISHOP_ATTACKS[DEEPFIN_BISHOP_OFFSETS[sq] + idx];
}

static inline uint64_t deepfin_rook_lookup(int sq, uint64_t occupied) {
    uint64_t mask = DEEPFIN_ROOK_MASKS[sq];
#if DEEPFIN_SLIDER_USE_PEXT
    uint32_t idx = (uint32_t)_pext_u64(occupied, mask);
#else
    uint32_t idx = (uint32_t)(((occupied & mask) * DEEPFIN_ROOK_MAGICS[sq])
                              >> DEEPFIN_ROOK_SHIFTS[sq]);
#endif
    return DEEPFIN_ROOK_ATTACKS[DEEPFIN_ROOK_OFFSETS[sq] + idx];
}

static inline uint64_t slider_attacks(int sq, uint64_t occupied, int bishop_like) {
    return bishop_like ? deepfin_bishop_lookup(sq, occupied)
                       : deepfin_rook_lookup(sq, occupied);
}

static inline uint64_t bishop_attacks(int sq, uint64_t occ) {
    return deepfin_bishop_lookup(sq, occ);
}

static inline uint64_t rook_attacks(int sq, uint64_t occ) {
    return deepfin_rook_lookup(sq, occ);
}

static inline uint64_t queen_attacks(int sq, uint64_t occ) {
    return deepfin_bishop_lookup(sq, occ) | deepfin_rook_lookup(sq, occ);
}

static inline int is_attacked_by(int sq, uint64_t occ,
                                 uint64_t pawns, uint64_t knights, uint64_t bishops,
                                 uint64_t rooks, uint64_t queens, uint64_t kings,
                                 int attacker_is_white) {
    if (PAWN_ATTACKS[1 - attacker_is_white][sq] & pawns) return 1;
    if (KNIGHT_ATTACKS[sq] & knights) return 1;
    if (KING_ATTACKS[sq] & kings) return 1;
    if (bishop_attacks(sq, occ) & (bishops | queens)) return 1;
    if (rook_attacks(sq, occ) & (rooks | queens)) return 1;
    return 0;
}

static void deepfin_slider_verify_one(int sq, int bishop_like, uint64_t mask) {
    uint64_t subset = 0ULL;
    do {
        uint64_t expected = slider_attacks_reference(sq, subset, bishop_like);
        uint64_t actual = slider_attacks(sq, subset, bishop_like);
        if (expected != actual)
            abort();
        subset = (subset - mask) & mask;
    } while (subset != 0ULL);
}

static void init_slider_attack_tables(void) {
    if (deepfin_slider_tables_initialized) return;

    uint32_t rook_offset = 0, bishop_offset = 0;
    for (int sq = 0; sq < 64; sq++) {
        uint64_t rook_mask = deepfin_slider_relevant_mask(sq, 0);
        uint64_t bishop_mask = deepfin_slider_relevant_mask(sq, 1);
        int rook_bits = popcount64(rook_mask);
        int bishop_bits = popcount64(bishop_mask);

        DEEPFIN_ROOK_MASKS[sq] = rook_mask;
        DEEPFIN_BISHOP_MASKS[sq] = bishop_mask;
        DEEPFIN_ROOK_OFFSETS[sq] = rook_offset;
        DEEPFIN_BISHOP_OFFSETS[sq] = bishop_offset;
        DEEPFIN_ROOK_SHIFTS[sq] = (uint8_t)(64 - rook_bits);
        DEEPFIN_BISHOP_SHIFTS[sq] = (uint8_t)(64 - bishop_bits);

#if DEEPFIN_SLIDER_USE_PEXT
        uint64_t rook_magic = 0ULL, bishop_magic = 0ULL;
#else
        uint64_t rook_magic = DEEPFIN_ROOK_MAGICS[sq];
        uint64_t bishop_magic = DEEPFIN_BISHOP_MAGICS[sq];
#endif
        deepfin_slider_init_one(
            sq, 0, rook_mask, rook_offset, DEEPFIN_ROOK_SHIFTS[sq],
            rook_magic, DEEPFIN_ROOK_ATTACKS, DEEPFIN_ROOK_TABLE_SIZE);
        deepfin_slider_init_one(
            sq, 1, bishop_mask, bishop_offset, DEEPFIN_BISHOP_SHIFTS[sq],
            bishop_magic, DEEPFIN_BISHOP_ATTACKS, DEEPFIN_BISHOP_TABLE_SIZE);

        rook_offset += (uint32_t)(1U << rook_bits);
        bishop_offset += (uint32_t)(1U << bishop_bits);
    }

    if (rook_offset != DEEPFIN_ROOK_TABLE_SIZE ||
        bishop_offset != DEEPFIN_BISHOP_TABLE_SIZE)
        abort();

    /* Exhaustive equivalence over every relevant blocker subset. Edge and
     * off-ray occupancy bits cannot affect a slider attack, so this covers
     * every board occupancy while retaining the old ray walker as oracle. */
    deepfin_slider_tables_initialized = 1;
    for (int sq = 0; sq < 64; sq++) {
        deepfin_slider_verify_one(sq, 0, DEEPFIN_ROOK_MASKS[sq]);
        deepfin_slider_verify_one(sq, 1, DEEPFIN_BISHOP_MASKS[sq]);
    }
}

static void init_attack_tables(void) {
    init_attack_tables_reference();
    init_slider_attack_tables();
}

#endif /* DEEPFIN_FAST_SLIDERS */
#endif /* DEEPFIN_SLIDER_ATTACKS_IMPL_H */
