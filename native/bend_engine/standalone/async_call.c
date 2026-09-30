/* Narrow packed-array ownership boundary. Never expose a Bend heap pointer to
 * the worker after submit returns: its private input is copied synchronously. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef DEEPFIN_BEND_NATIVE_MODEL
extern uint32_t deepfin_async_submit(uint32_t, uint32_t, uint32_t, const float*, uint32_t);
extern uint32_t deepfin_async_poll(uint32_t, uint32_t, uint32_t, uint32_t, float*, uint32_t);
extern uint32_t deepfin_async_cancel(uint32_t, uint32_t, uint32_t, uint32_t);
extern void deepfin_async_shutdown(void);
#endif
static int async_enabled;
static void async_bad(void) {
    fputs("native async identity/buffer/state contract failed\n", stderr);
    exit(2);
}
static Term async_enabled_run(Env e, Term* f, IoWork* w) {
    (void)e; (void)f; (void)w;
    const char* v = getenv("DEEPFIN_BEND_ASYNC");
    if (!v || !*v || !strcmp(v, "0")) return term_pak(CID_FALSE, 0);
    if (strcmp(v, "1")) async_bad();
#ifdef DEEPFIN_BEND_NATIVE_MODEL
    async_enabled = 1;
    return term_pak(CID_TRUE, 0);
#else
    async_bad(); return 0;
#endif
}
static Term async_submit_run(Env e, Term* f, IoWork* w) {
    (void)w;
#ifdef DEEPFIN_BEND_NATIVE_MODEL
    const Term input = f[4];
    if (term_tag(input) != TAG_BUF || blk_cls(input) != 14) async_bad();
    const float* x = (const float*)(const void*)blk_ptr(e.mem, term_loc(input), 0);
    const uint32_t token = deepfin_async_submit(f[0], f[1], f[2], x, f[3]);
    if (!token) async_bad();  /* Bend's single-slot state must prevent double admission */
    Loc r = heap_alloc(e, cls_fit(2));
    if (err_seen(e.mem)) async_bad();
    e.mem[r] = io_seal(e, input, CID_ASYNCSUBMITTED);
    e.mem[r + 1] = token;
    return term_ctr(CID_ASYNCSUBMITTED, r);
#else
    (void)e; (void)f; async_bad(); return 0;
#endif
}
static Term async_poll_run(Env e, Term* f, IoWork* w) {
    (void)w;
#ifdef DEEPFIN_BEND_NATIVE_MODEL
    const Term output = f[4];
    if (term_tag(output) != TAG_BUF || blk_cls(output) != 11) async_bad();
    float* y = (float*)(void*)blk_ptr(e.mem, term_loc(output), 0);
    const uint32_t status = deepfin_async_poll(f[0], f[1], f[2], f[3], y, 1861);
    Loc r = heap_alloc(e, cls_fit(2));
    if (err_seen(e.mem)) async_bad();
    e.mem[r] = io_seal(e, output, CID_ASYNCPOLLED);
    e.mem[r + 1] = status;
    return term_ctr(CID_ASYNCPOLLED, r);
#else
    (void)e; (void)f; async_bad(); return 0;
#endif
}
static Term async_cancel_run(Env e, Term* f, IoWork* w) {
    (void)e; (void)w;
#ifdef DEEPFIN_BEND_NATIVE_MODEL
    if (!deepfin_async_cancel(f[0], f[1], f[2], f[3])) async_bad();
#else
    (void)f; async_bad();
#endif
    return term_pak(CID_UNIT, 0);
}
static Term async_drain_run(Env e, Term* f, IoWork* w) {
    (void)e; (void)w;
#ifdef DEEPFIN_BEND_NATIVE_MODEL
    return deepfin_async_poll(f[0], f[1], f[2], f[3], NULL, 0);
#else
    (void)f; async_bad(); return 0;
#endif
}
static Term async_shutdown_run(Env e, Term* f, IoWork* w) {
    (void)e; (void)f; (void)w;
#ifdef DEEPFIN_BEND_NATIVE_MODEL
    if (async_enabled) deepfin_async_shutdown();
#endif
    return term_pak(CID_UNIT, 0);
}
static void __attribute__((constructor)) async_effects_use(void) {
    io_eff(CID_ASYNC_ENABLED, async_enabled_run, 0);
    io_eff(CID_ASYNC_SUBMIT, async_submit_run, 0);
    io_eff(CID_ASYNC_POLL, async_poll_run, 0);
    io_eff(CID_ASYNC_CANCEL, async_cancel_run, 0);
    io_eff(CID_ASYNC_DRAIN, async_drain_run, 0);
    io_eff(CID_ASYNC_SHUTDOWN, async_shutdown_run, 0);
}
