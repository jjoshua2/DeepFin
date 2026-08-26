/*
 * Table-backed sliding attacks for CBoard.
 *
 * Two backends, chosen at BUILD time, never per lookup:
 *   "pext"  — BMI2 PEXT indexing. Selected on x86-64 BMI2 targets EXCEPT the
 *             CPU families where PEXT is microcoded; see the gate below.
 *   "magic" — generated magic multipliers. Everything else, including every
 *             portable build (so CI, and the distributed worker wheels).
 * Both use the same relevant-occupancy masks and the same packed table sizes,
 * so they are interchangeable and differ only in how the index is formed. Each
 * module publishes which one it got as SLIDER_BACKEND ("pext" / "magic", or
 * "rays" for a module built without fast sliders at all), because "which
 * backend does this .so run" has to be answerable from the binary rather than
 * from the build system's intent.
 *
 * The build preserves the legacy ray walker as *_reference; it is used to
 * populate and verify these tables at module initialization.
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

#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>

/* ⚑ THE BACKEND GATE. Compile-time only — there is deliberately no per-lookup
 * runtime branch in the hot path.
 *
 * __x86_64__ (not __i386__): GCC exposes _pext_u64 only on x86-64, so a 32-bit
 * BMI2 target that selected PEXT would fail to compile.
 *
 * The __znver1__/__znver2__/__bdver4__ exclusion is the interesting part.
 * `PEXT r64,r64,r64` is MICROCODED on AMD Zen 1, Zen 2 and Excavator — roughly
 * 18 cycles, against ~3 on Zen 3+ and Intel Haswell+ — yet those parts define
 * __BMI2__ under -march=native exactly like the fast ones. Gating on __BMI2__
 * alone would therefore hand a volunteer worker on a Ryzen 3000 the SLOWEST of
 * our three implementations while it followed the documented production build
 * recipe, and nothing would say so. Stockfish declines to imply USE_PEXT from
 * -march=native for this same reason. Excluding the three slow families keeps
 * PEXT on where it is genuinely fast — including this project's Zen 3
 * production host, which must land on "pext" — and drops those hosts to magic,
 * which is faster there. (bdver1..3 need no exclusion: they predate BMI2, so
 * __BMI2__ is not defined and they fall through to magic anyway. znver4/znver5
 * inherit Zen 3's fast PEXT and are deliberately NOT excluded.)
 *
 * Margin, measured on the Zen 3 host over the same slider-primitive sample:
 * PEXT 5.87ms, magic 6.96ms, legacy ray walker 21.04ms. So PEXT is worth ~19%
 * over magic and both are worth ~3x over the walker — the exclusion protects a
 * large downside while conceding a modest upside, which is the right trade for
 * a build that ships to machines we do not own.
 *
 * ⚑ RESIDUAL GAP, stated rather than papered over: the exclusion recognizes a
 * CPU family only when the build NAMES one, which -march=native does and which
 * is how both the production recipe and any worker following it build. A build
 * that asks for a feature LEVEL instead (-march=x86-64-v3, or bare -mbmi2)
 * defines __BMI2__ with no __znverN__, so a Zen 2 host built that way would
 * still select PEXT. Read SLIDER_BACKEND off the built module to find out what
 * a given binary actually chose. */
#if defined(__BMI2__) && defined(__x86_64__) \
    && !defined(__znver1__) && !defined(__znver2__) && !defined(__bdver4__)
#  include <immintrin.h>
#  define DEEPFIN_SLIDER_USE_PEXT 1
#  define DEEPFIN_SLIDER_BACKEND_NAME "pext"
#else
#  define DEEPFIN_SLIDER_USE_PEXT 0
#  define DEEPFIN_SLIDER_BACKEND_NAME "magic"
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

/* A distributed selfplay worker that hits one of these dies inside a module
 * import with no output at all, which is indistinguishable from a segfault at
 * the other end of the pipeline. Say what mismatched, on stderr, before
 * aborting. */
#define DEEPFIN_SLIDER_FATAL(...)                                            \
    do {                                                                     \
        fflush(stdout);                                                      \
        fprintf(stderr, "deepfin sliders [%s] FATAL: ",                      \
                DEEPFIN_SLIDER_BACKEND_NAME);                                \
        fprintf(stderr, __VA_ARGS__);                                        \
        fputc('\n', stderr);                                                 \
        fflush(stderr);                                                      \
        abort();                                                             \
    } while (0)

#define DEEPFIN_SLIDER_KIND(bishop_like) ((bishop_like) ? "bishop" : "rook")

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
            DEEPFIN_SLIDER_FATAL(
                "%s sq=%d index out of range: offset=%" PRIu32 " idx=%" PRIu32
                " (table_size=%" PRIu32 ") mask=0x%016" PRIx64
                " subset=0x%016" PRIx64,
                DEEPFIN_SLIDER_KIND(bishop_like), sq, offset, idx, table_size,
                mask, subset);
        uint64_t attack = slider_attacks_reference(sq, subset, bishop_like);
#if !DEEPFIN_SLIDER_USE_PEXT
        uint64_t prior = table[offset + idx];
        if (prior != 0ULL && prior != attack)
            DEEPFIN_SLIDER_FATAL(
                "%s sq=%d DESTRUCTIVE magic collision at slot %" PRIu32
                ": magic=0x%016" PRIx64 " shift=%u mask=0x%016" PRIx64
                " subset=0x%016" PRIx64 " stored=0x%016" PRIx64
                " incoming=0x%016" PRIx64,
                DEEPFIN_SLIDER_KIND(bishop_like), sq, offset + idx, magic,
                (unsigned)shift, mask, subset, prior, attack);
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

/* ⚑⚑ THIS CHECK IS CIRCULAR WITH RESPECT TO THE MASK, AND ON ITS OWN IT IS NOT
 * THE SAFETY STORY.
 *
 * It Carry-Ripples over the SAME mask the table was filled from, comparing
 * reference(sq, s) against a slot that was WRITTEN as reference(sq, s). It
 * therefore cannot construct an occupancy with a bit outside the mask, and so
 * it cannot see a wrong mask at all: a popcount-preserving edit to
 * deepfin_slider_relevant_mask (drop a real blocker from a ray, keep the edge
 * square instead) leaves the table-size assertion satisfied and passes this
 * loop cleanly, while producing thousands of wrong attack sets on real
 * full-board occupancies. The magic backend aborts on such an edit only
 * incidentally — its multipliers were generated against the CORRECT masks, so
 * a changed mask makes them collide destructively — which means the arm that
 * survives the bad edit is PEXT, the one with no collision check by
 * construction.
 *
 * The mask itself is pinned by deepfin_slider_verify_mask below, and the
 * arm-against-arm differential over occupancies WITH off-mask bits lives in
 * deepfin_slider_selftest, driven from pytest rather than paid at every
 * process start. */
static void deepfin_slider_verify_one(int sq, int bishop_like, uint64_t mask) {
    uint64_t subset = 0ULL;
    do {
        uint64_t expected = slider_attacks_reference(sq, subset, bishop_like);
        uint64_t actual = slider_attacks(sq, subset, bishop_like);
        if (expected != actual)
            DEEPFIN_SLIDER_FATAL(
                "%s sq=%d table disagrees with the ray walker: "
                "mask=0x%016" PRIx64 " subset=0x%016" PRIx64
                " expected=0x%016" PRIx64 " table=0x%016" PRIx64,
                DEEPFIN_SLIDER_KIND(bishop_like), sq, mask, subset, expected,
                actual);
        subset = (subset - mask) & mask;
    } while (subset != 0ULL);
}

/* The mask is complete iff occupying every square in it blocks each ray exactly
 * where a FULL board would. Non-circular, because the right-hand side uses an
 * occupancy the Carry-Rippler above can never reach: ~0 sets the off-mask bits
 * the mask claims are irrelevant. Cost is 128 ray walks, once. */
static void deepfin_slider_verify_mask(int sq, int bishop_like, uint64_t mask) {
    uint64_t masked = slider_attacks_reference(sq, mask, bishop_like);
    uint64_t full = slider_attacks_reference(sq, ~0ULL, bishop_like);
    if (masked != full)
        DEEPFIN_SLIDER_FATAL(
            "%s sq=%d relevant mask is INCOMPLETE: mask=0x%016" PRIx64
            " attacks(mask)=0x%016" PRIx64 " attacks(full)=0x%016" PRIx64
            " differ at 0x%016" PRIx64
            " — a blocker square is missing from the mask, so table lookups "
            "will read the wrong slot for real occupancies",
            DEEPFIN_SLIDER_KIND(bishop_like), sq, mask, masked, full,
            masked ^ full);
}

/* Arm-against-arm differential over occupancies with OFF-MASK bits set, which
 * is the class deepfin_slider_verify_one structurally cannot reach. Returns the
 * number of mismatches; 0 is the only acceptable answer. Pure C so every
 * extension that compiles these tables can publish it and be checked as the
 * binary it actually ships. xorshift64* keeps the stream reproducible from the
 * seed without depending on any host RNG. */
static long deepfin_slider_selftest(uint64_t seed, long samples) {
    uint64_t state = seed ? seed : 0x9E3779B97F4A7C15ULL;
    long mismatches = 0;
    for (long i = 0; i < samples; i++) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        uint64_t occupied = state * 0x2545F4914F6CDD1DULL;
        /* Vary the density too: a uniform 50%-full board never exercises the
         * long-ray cases that a sparse endgame occupancy produces. */
        if ((i & 3) == 1) occupied &= (occupied >> 7) | (occupied << 7);
        if ((i & 3) == 2) occupied |= (occupied >> 11) | (occupied << 11);
        for (int sq = 0; sq < 64; sq++) {
            for (int bishop_like = 0; bishop_like < 2; bishop_like++) {
                uint64_t expected =
                    slider_attacks_reference(sq, occupied, bishop_like);
                uint64_t actual = slider_attacks(sq, occupied, bishop_like);
                if (expected != actual) mismatches++;
            }
        }
    }
    return mismatches;
}

static void deepfin_slider_build_tables(void) {
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
        DEEPFIN_SLIDER_FATAL(
            "packed table size mismatch: rook %" PRIu32 " (expected %d), "
            "bishop %" PRIu32 " (expected %d)",
            rook_offset, DEEPFIN_ROOK_TABLE_SIZE, bishop_offset,
            DEEPFIN_BISHOP_TABLE_SIZE);

    /* The mask check runs FIRST: every claim the loop below makes is
     * conditional on the mask being the right one, and it is the only one of
     * the two that can see a mask defect. */
    for (int sq = 0; sq < 64; sq++) {
        deepfin_slider_verify_mask(sq, 0, DEEPFIN_ROOK_MASKS[sq]);
        deepfin_slider_verify_mask(sq, 1, DEEPFIN_BISHOP_MASKS[sq]);
    }

    /* Exhaustive equivalence over every relevant blocker subset — see the
     * circularity warning on deepfin_slider_verify_one for what this does and
     * does not establish. The lookups it calls do not consult
     * deepfin_slider_tables_initialized, so the flag is set only once every
     * check has passed: no caller ever observes it true over an unverified
     * table. */
    for (int sq = 0; sq < 64; sq++) {
        deepfin_slider_verify_one(sq, 0, DEEPFIN_ROOK_MASKS[sq]);
        deepfin_slider_verify_one(sq, 1, DEEPFIN_BISHOP_MASKS[sq]);
    }
    deepfin_slider_tables_initialized = 1;
}

/* ⚑ NOT "single-threaded by precondition" — MADE safe, following the
 * pthread_once precedent at _nnue_impl.h:316/395. The precondition was already
 * untrue on paper: _features_impl.h reaches init_tables_features() ->
 * init_attack_tables() from inside threaded compute paths, and today those
 * short-circuit only because PyInit happened to run first under the GIL. That
 * is an ordering accident, not an invariant, and the failure it guards against
 * is a second thread reading a half-filled 102,400-entry table — silent wrong
 * attack sets, not a crash. pthread_once makes latecomers block until the
 * tables are built AND verified. */
static pthread_once_t deepfin_slider_tables_once = PTHREAD_ONCE_INIT;

static void init_slider_attack_tables(void) {
    pthread_once(&deepfin_slider_tables_once, deepfin_slider_build_tables);
}

static void init_attack_tables(void) {
    init_attack_tables_reference();
    init_slider_attack_tables();
}

/* ================================================================
 * Python surface — defined once here and pasted into each extension's method
 * table by DEEPFIN_SLIDER_PY_METHODS, following the CAE_NNUE_DAG_METHODS
 * precedent in nnue/_nnue_dag_api.h.
 *
 * ⚑ It has to be PER-MODULE, not one shared helper module. These are
 * header-only statics, so every .so carries its OWN tables, its OWN ray walker
 * and its OWN backend selection — which is exactly how _nnue_ext came to ship
 * ray-walking sliders while _lc0_ext and _mcts_tree were table-backed, with no
 * test able to see the difference. Asking one module proves nothing about
 * another; each binary answers for itself.
 *
 * Guarded on Py_PYTHON_H so scripts/fuzz/cboard_libfuzzer.c, which includes
 * _cboard_impl.h with no Python at all, still compiles.
 * ================================================================ */
#ifdef Py_PYTHON_H

PyDoc_STRVAR(deepfin_slider_selftest_doc,
"slider_selftest(seed=..., samples=...) -> int\n\n"
"Differential-test THIS module's slider tables against THIS module's legacy ray\n"
"walker over pseudo-random FULL-BOARD occupancies, and return the mismatch\n"
"count. Zero is the only acceptable answer.\n\n"
"Full-board is the point. The exhaustive check at module init walks the\n"
"Carry-Rippler over each square's relevant mask, so every occupancy it can\n"
"build has zero bits outside that mask -- it is circular with respect to the\n"
"mask and cannot detect a wrong one. These occupancies set off-mask bits, so a\n"
"mask that dropped a real blocker shows up here as thousands of mismatches.\n\n"
"Each sample tests all 64 squares x {rook, bishop}, i.e. 128 comparisons.");

static PyObject *deepfin_slider_selftest_py(PyObject *Py_UNUSED(self),
                                            PyObject *args) {
    unsigned long long seed = 0x243F6A8885A308D3ULL;
    long samples = 2000;
    if (!PyArg_ParseTuple(args, "|Kl", &seed, &samples))
        return NULL;
    if (samples < 0) {
        PyErr_SetString(PyExc_ValueError, "samples must be >= 0");
        return NULL;
    }
    init_attack_tables();
    long mismatches;
    Py_BEGIN_ALLOW_THREADS
    mismatches = deepfin_slider_selftest((uint64_t)seed, samples);
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(mismatches);
}

#define DEEPFIN_SLIDER_PY_METHODS                                            \
    {"slider_selftest", deepfin_slider_selftest_py, METH_VARARGS,            \
     deepfin_slider_selftest_doc},

#endif /* Py_PYTHON_H */

#endif /* DEEPFIN_FAST_SLIDERS */
#endif /* DEEPFIN_SLIDER_ATTACKS_IMPL_H */
