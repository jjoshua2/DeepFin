/* FFI is confined to fixture loading, never one call per attack lookup.
 * Pinned Bend revision: Array<U64> is a BUF with two u32 cells per element.
 * Raw 64-bit values MUST NOT be stored in tagged Term cells. */
#include <stdint.h>
extern uint32_t deepfin_u64_fixture(uint32_t bishop, uint32_t sq,
                                  uint64_t *out, uint32_t capacity);

Term fixture_load_run(Env e, Term *f, IoWork *w) {
    (void)w;
    uint64_t *words = io_mem(calloc(32768, sizeof(uint64_t)));
    uint32_t count = deepfin_u64_fixture((uint32_t)f[0], (uint32_t)f[1], words, 32768);
    if (count == 0 || count > 32768) err_fail("invalid slider fixture size");
    u32 depth = 0;
    while ((1u << depth) < count) depth++;
    Term zero[2] = {0, 0};
    Term a = blk_new(e, false, depth, 1, 2, zero);
    for (u32 i = 0; i < count; i++) {
        blk_write(e.mem, false, term_loc(a), 2 * i, (u32)words[i]);
        blk_write(e.mem, false, term_loc(a), 2 * i + 1, (u32)(words[i] >> 32));
    }
    /* Fixture's concrete node layout is pinned with the compiler. */
    if (cid_arity(CID_FIXTURE) != 8) err_fail("Fixture ABI changed");
    Loc loc = heap_alloc(e, 3);
    e.mem[loc] = (u32)words[0];
    e.mem[loc + 1] = (u32)(words[0] >> 32);
    e.mem[loc + 2] = (u32)words[1];
    e.mem[loc + 3] = (u32)(words[1] >> 32);
    e.mem[loc + 4] = words[2];
    e.mem[loc + 5] = words[3];
    e.mem[loc + 6] = words[4];
    e.mem[loc + 7] = a;
    free(words);
    return term_ctr(CID_FIXTURE, loc);
}

static void __attribute__((constructor)) fixture_effects_use(void) {
    io_eff(CID_FIXTURE_LOAD, fixture_load_run, 0);
}
