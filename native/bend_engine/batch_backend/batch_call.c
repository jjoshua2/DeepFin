/* Bounded packed-array transport, not a scheduler or chess implementation.
 * This module is built only with the explicit backend package binding. */
#include <stdint.h>
#include <stdlib.h>
#include "model_contract.h"
extern uint32_t deepfin_model_open_batch(void);
extern int deepfin_model_run_batch(const float *, uint32_t, float *, uint32_t);

static uint32_t batch_log2(void) {
    uint32_t n = DEEPFIN_MODEL_BATCH, bits = 0;
    while (n > 1) { n >>= 1; ++bits; }
    return bits;
}
/* The pinned compiler emits this shared import even when only one effect is
 * reachable. Register only its real generated IDs; never synthesize missing IDs. */
#ifdef CID_BATCH_OPEN
static Term batch_open_run(Env e, Term *f, IoWork *w) {
    (void)f; (void)w;
    const uint32_t profile = deepfin_model_open_batch();
    if (!profile) exit(2);
    Loc result = heap_alloc(e, cls_fit(5));
    if (err_seen(e.mem)) exit(2);
    e.mem[result] = profile;
    e.mem[result + 1] = DEEPFIN_MODEL_BATCH;
    e.mem[result + 2] = DEEPFIN_MODEL_CHANNELS;
    e.mem[result + 3] = 14 + batch_log2();
    e.mem[result + 4] = 11 + batch_log2();
    return term_ctr(CID_CAPABILITIES, result);
}
#endif
#ifdef CID_BATCH_RUN
static Term batch_run_run(Env e, Term *f, IoWork *w) {
    (void)w;
    const uint32_t rows = f[0];
    const Term input = f[1], output = f[2];
    if (!rows || rows > DEEPFIN_MODEL_BATCH
        || term_tag(input) != TAG_BUF || blk_cls(input) != 14 + batch_log2()
        || term_tag(output) != TAG_BUF || blk_cls(output) != 11 + batch_log2()
        || term_loc(input) == term_loc(output)) {
        fputs("native batch buffer/row contract failed\n", stderr); exit(2);
    }
    const float *x = (const float *)(const void *)blk_ptr(e.mem, term_loc(input), 0);
    float *y = (float *)(void *)blk_ptr(e.mem, term_loc(output), 0);
    if (deepfin_model_run_batch(x, rows, y, rows * 1861) != 0) exit(2);
    Loc result = heap_alloc(e, cls_fit(5));
    if (err_seen(e.mem)) exit(2);
    e.mem[result] = io_seal(e, input, CID_RESULT);
    e.mem[result + 1] = io_seal(e, output, CID_RESULT);
    e.mem[result + 2] = rows;
    e.mem[result + 3] = DEEPFIN_MODEL_BATCH;
    e.mem[result + 4] = rows * 1861;
    return term_ctr(CID_RESULT, result);
}
#endif
static void __attribute__((constructor)) batch_effects_use(void) {
#ifdef CID_BATCH_OPEN
    io_eff(CID_BATCH_OPEN, batch_open_run, 0);
#endif
#ifdef CID_BATCH_RUN
    io_eff(CID_BATCH_RUN, batch_run_run, 0);
#endif
}
