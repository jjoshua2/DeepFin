/* Scalar timing effects only; no move generation or per-node foreign calls. */
#include "bench_clock.h"
Term bench_start_run(Env e, Term *f, IoWork *w) {
    (void)e; (void)f; (void)w;
    perft_clock_start();
    return term_pak(CID_UNIT, 0);
}
Term bench_finish_run(Env e, Term *f, IoWork *w) {
    (void)w;
    /* Four arguments plus the IO continuation occupy five fields. */
    if (cid_arity(CID_BENCH_FINISH) != 5) err_fail("benchmark timer ABI changed");
    perft_clock_finish("bend-pext", (unsigned)f[1],
        ((uint64_t)(uint32_t)f[2] << 32) | (uint32_t)f[3]);
    /* Keep the last table owner alive until AFTER the measurement. */
    term_drop(e, f[0]);
    return term_pak(CID_UNIT, 0);
}
Term bench_warmup_run(Env e, Term *f, IoWork *w) {
    (void)e; (void)w;
    printf("warmup %" PRIu32 " %" PRIu32 "\n", (uint32_t)f[0], (uint32_t)f[1]);
    fflush(stdout);
    return term_pak(CID_UNIT, 0);
}
static void __attribute__((constructor)) bench_effects_use(void) {
    io_eff(CID_BENCH_WARMUP, bench_warmup_run, 0);
    io_eff(CID_BENCH_START, bench_start_run, 0);
    io_eff(CID_BENCH_FINISH, bench_finish_run, 0);
}
