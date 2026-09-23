/* Packed-array transport only; no root or search semantics. CPU opt-in build. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "model_contract.h"
#ifdef DEEPFIN_ASYNC_BATCH
extern uint32_t deepfin_async_batch_submit(const float*, uint32_t);
extern uint32_t deepfin_async_batch_poll(uint32_t, float*, uint32_t);
extern void deepfin_async_batch_shutdown(void);
#endif
static void async_batch_bad(void) {
    fputs("async batch buffer/row/configuration contract failed\n", stderr); exit(2);
}
static uint32_t async_batch_log2(void) {
    uint32_t n = DEEPFIN_MODEL_BATCH, bits = 0;
    while (n > 1) { n >>= 1; ++bits; }
    return bits;
}
static Term batchasync_enabled_run(Env e, Term *f, IoWork *w) {
    (void)e; (void)f; (void)w;
    const char *v = getenv("DEEPFIN_COHORT_ASYNC");
    if (!v || !*v || !strcmp(v,"0")) return term_pak(CID_FALSE,0);
    if (strcmp(v,"1")) async_batch_bad();
#ifdef DEEPFIN_ASYNC_BATCH
    return term_pak(CID_TRUE,0);
#else
    async_batch_bad(); return 0;
#endif
}
static Term batchasync_submit_run(Env e, Term *f, IoWork *w) {
    (void)w;
#ifdef DEEPFIN_ASYNC_BATCH
    const uint32_t rows = f[0]; const Term input = f[1];
    if (!rows || rows > DEEPFIN_MODEL_BATCH || term_tag(input) != TAG_BUF
        || blk_cls(input) != 14 + async_batch_log2()) async_batch_bad();
    const float *x = (const float*)(const void*)blk_ptr(e.mem,term_loc(input),0);
    const uint32_t token = deepfin_async_batch_submit(x,rows);
    if (!token) async_batch_bad();
    Loc r = heap_alloc(e,cls_fit(2)); if (err_seen(e.mem)) async_batch_bad();
    e.mem[r] = io_seal(e,input,CID_BATCHSUBMITTED); e.mem[r+1] = token;
    return term_ctr(CID_BATCHSUBMITTED,r);
#else
    (void)e; (void)f; async_batch_bad(); return 0;
#endif
}
static Term batchasync_poll_run(Env e, Term *f, IoWork *w) {
    (void)w;
#ifdef DEEPFIN_ASYNC_BATCH
    const uint32_t token = f[0], rows = f[1]; const Term output = f[2];
    if (!token || !rows || rows > DEEPFIN_MODEL_BATCH || term_tag(output) != TAG_BUF
        || blk_cls(output) != 11 + async_batch_log2()) async_batch_bad();
    float *y = (float*)(void*)blk_ptr(e.mem,term_loc(output),0);
    const uint32_t status = deepfin_async_batch_poll(token,y,rows);
    Loc r = heap_alloc(e,cls_fit(2)); if (err_seen(e.mem)) async_batch_bad();
    e.mem[r] = io_seal(e,output,CID_BATCHPOLLED); e.mem[r+1] = status;
    return term_ctr(CID_BATCHPOLLED,r);
#else
    (void)e; (void)f; async_batch_bad(); return 0;
#endif
}
static Term batchasync_shutdown_run(Env e, Term *f, IoWork *w) {
    (void)e; (void)f; (void)w;
#ifdef DEEPFIN_ASYNC_BATCH
    deepfin_async_batch_shutdown();
#endif
    return term_pak(CID_UNIT,0);
}
static void __attribute__((constructor)) async_batch_effects(void) {
    io_eff(CID_BATCHASYNC_ENABLED,batchasync_enabled_run,0);
    io_eff(CID_BATCHASYNC_SUBMIT,batchasync_submit_run,0);
    io_eff(CID_BATCHASYNC_POLL,batchasync_poll_run,0);
    io_eff(CID_BATCHASYNC_SHUTDOWN,batchasync_shutdown_run,0);
}
