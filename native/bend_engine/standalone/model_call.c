/* Packed F32 transport for the checked Bend compiler's scalar Array layout.
 * Linear buffers return to Bend only after synchronous model execution finishes.
 * No chess, encoding, policy mapping, masking, probabilities or scheduling here. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef DEEPFIN_BEND_NATIVE_MODEL
extern uint32_t deepfin_model_open(void);
extern int deepfin_model_run(const float *, uint32_t, float *, uint32_t);
#endif

static Term native_open_run(Env e, Term *f, IoWork *w) {
    (void)e; (void)f; (void)w;
#ifdef DEEPFIN_BEND_NATIVE_MODEL
    uint32_t profile = deepfin_model_open();
    if (profile == 0) exit(2);
    return profile;
#else
    return 0;
#endif
}
static Term native_diagnostics_run(Env e, Term *f, IoWork *w) {
    (void)e; (void)f; (void)w;
    const char *value = getenv("DEEPFIN_BEND_NATIVE_DIAGNOSTICS");
    if (!value || !*value || strcmp(value, "0") == 0) return term_pak(CID_FALSE, 0);
    if (strcmp(value, "1") == 0) return term_pak(CID_TRUE, 0);
    fputs("DEEPFIN_BEND_NATIVE_DIAGNOSTICS must be 0 or 1\n", stderr);
    exit(2);
}
static Term native_run_run(Env e, Term *f, IoWork *w) {
    (void)w;
#ifdef DEEPFIN_BEND_NATIVE_MODEL
    const uint32_t count = f[0];
    const Term input = f[1], output = f[2];
    /* Count is the logical shape, never the padded power-of-two capacity.
     * These are owned scalar buffers, not generic/refcounted constructor arrays.
     * Reject aliasing even though well-typed linear Bend cannot produce it. */
    if ((count != 9344 && count != 11200)
        || term_tag(input) != TAG_BUF || blk_cls(input) != 14
        || term_tag(output) != TAG_BUF || blk_cls(output) != 11
        || term_loc(input) == term_loc(output)) {
        fputs("native input/output buffer contract failed\n", stderr);
        exit(2);
    }
    /* The pinned runtime stores scalar words densely. memcpy in the bridge
     * reads/writes these bytes without dereferencing an incompatible C type. */
    const float *x = (const float *)(const void *)blk_ptr(e.mem, term_loc(input), 0);
    float *y = (float *)(void *)blk_ptr(e.mem, term_loc(output), 0);
    if (deepfin_model_run(x, count, y, 1861) != 0) exit(2);
    Loc result = heap_alloc(e, cls_fit(3));
    if (err_seen(e.mem)) exit(2);
    e.mem[result] = io_seal(e, input, CID_BUFFERS);
    e.mem[result + 1] = io_seal(e, output, CID_BUFFERS);
    e.mem[result + 2] = 1861;
    return term_ctr(CID_BUFFERS, result);
#else
    (void)e; (void)f;
    fputs("native model support is not linked; no material fallback\n", stderr);
    exit(2);
#endif
}
static void __attribute__((constructor)) model_effects_use(void) {
    io_eff(CID_NATIVE_OPEN, native_open_run, 0);
    io_eff(CID_NATIVE_DIAGNOSTICS, native_diagnostics_run, 0);
    io_eff(CID_NATIVE_RUN, native_run_run, 0);
}
