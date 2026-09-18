/* One input/table transfer per invocation; no foreign call in movegen/perft.
 * The private ABI is pinned by bitboard_probe/toolchain.json. */
#include "support.h"

Term job_load_run(Env e, Term *f, IoWork *w) {
    (void)f; (void)w;
    LegalInput in;
    if (legal_read_input(&in) != 1) {
        fputs("invalid legal probe input\n", stderr); exit(2);
    }
    if (cid_arity(CID_JOB) != 23) err_fail("Job ABI changed");
    uint64_t *words = io_mem(calloc(LEGAL_TABLE_WORDS, sizeof(uint64_t)));
    legal_make_tables(words);
    Term zero[2] = {0, 0};
    Term a = blk_new(e, false, 17, 1, 2, zero);
    if (err_seen(e.mem)) err_fail("attack-table allocation failed");
    for (u32 i = 0; i < LEGAL_TABLE_WORDS; i++) {
        blk_write(e.mem, false, term_loc(a), 2*i, (u32)words[i]);
        blk_write(e.mem, false, term_loc(a), 2*i+1, (u32)(words[i] >> 32));
    }
    free(words);
    Loc loc = heap_alloc(e, cls_fit(23));
    if (err_seen(e.mem)) err_fail("input allocation failed");
    for (u32 i = 0; i < 8; i++) {
        e.mem[loc+2*i] = (u32)in.bb[i];
        e.mem[loc+2*i+1] = (u32)(in.bb[i] >> 32);
    }
    e.mem[loc+16] = in.turn; e.mem[loc+17] = in.rights; e.mem[loc+18] = in.ep;
    e.mem[loc+19] = a; e.mem[loc+20] = in.depth; e.mem[loc+21] = in.mode;
    e.mem[loc+22] = in.seed ? in.seed : 0x6d2b79f5u;
    return term_ctr(CID_JOB, loc);
}
static void __attribute__((constructor)) legal_effects_use(void) {
    io_eff(CID_JOB_LOAD, job_load_run, 0);
}
