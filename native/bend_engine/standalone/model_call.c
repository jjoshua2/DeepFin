/* Generic finite-length float transport for a synchronous model effect only.
 * No chess, encoding, legal masking, softmax, scheduling or fallback decisions.
 * Material builds link a disabled capability, not a hidden model substitute. */
#include <stdint.h>
#include <stdlib.h>
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
static Term native_run_run(Env e, Term *f, IoWork *w) {
    (void)w;
#ifdef DEEPFIN_BEND_NATIVE_MODEL
    float input[11200], output[1861];
    uint32_t n = 0;
    Term xs = f[0];
    while (term_aux(xs) == CID_CON) {
        if (n == 11200) { fputs("native input exceeds bound\n", stderr); exit(2); }
        Term fields[2];
        spare_free(e, cls_fit(2), ctr_take(e, xs, 2, fields));
        input[n++] = f32_unbox(fields[0]);
        xs = fields[1];
    }
    if (term_aux(xs) != CID_NIL || deepfin_model_run(input, n, output, 1861) != 0) exit(2);
    Term result = term_pak(CID_NIL, 0);
    for (uint32_t i = 1861; i > 0; --i) result = io_node(e, CID_CON, f32_rewrap(output[i - 1]), result);
    return result;
#else
    (void)e; (void)f;
    fputs("native model support is not linked; no material fallback\n", stderr);
    exit(2);
#endif
}
static void __attribute__((constructor)) model_effects_use(void) {
    io_eff(CID_NATIVE_OPEN, native_open_run, 0);
    io_eff(CID_NATIVE_RUN, native_run_run, 0);
}
