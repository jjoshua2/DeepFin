/* Typed ABI marshalling only. Model loading/shape checks are in model.cpp;
 * all chess, feature construction, legal gathering and softmax stay in Bend.
 * One synchronous caller. No Python, shell, subprocess or native chess policy. */
#include <stdint.h>
#include <string.h>
extern uint32_t deepfin_model_open(const char *path);
extern uint32_t deepfin_model_input_count(void);
extern void deepfin_model_spec(uint32_t words[4]);
extern uint32_t deepfin_model_run(uint32_t sequence, const float *in, uint32_t n, float *out);
extern void deepfin_model_close(void);

static Term native_open_run(Env e, Term *f, IoWork *w) {
    (void)w;
    uint64_t n = 0;
    char *path = io_cstr(e, f[0], &n);
    uint32_t code = io_nul(path, n) ? 2 : deepfin_model_open(path);
    free(path);
    return (Term)code;
}
static Term native_spec_run(Env e, Term *f, IoWork *w) {
    (void)f; (void)w;
    uint32_t words[4]; deepfin_model_spec(words);
    Term xs = term_pak(CID_NIL, 0);
    for (unsigned i = 4; i > 0; --i) xs = io_node(e, CID_CON, (Term)words[i-1], xs);
    return xs;
}
static Term native_forward_run(Env e, Term *f, IoWork *w) {
    (void)w;
    const uint32_t expected = deepfin_model_input_count();
    if (expected != 146*64 && expected != 175*64) err_fail("native model input contract");
    float *input = io_mem(malloc(expected * sizeof(float)));
    uint32_t count = 0, invalid = 0;
    Term xs = f[1];
    while (term_aux(xs) == CID_CON) {
        Term fields[2];
        spare_free(e, cls_fit(2), ctr_take(e, xs, 2, fields));
        if (count < expected) {
            uint32_t bits = (uint32_t)fields[0];
            memcpy(&input[count++], &bits, sizeof(bits));
        } else invalid = 1;
        xs = fields[1];
    }
    float output[1861];
    uint32_t n = (!invalid && count == expected)
        ? deepfin_model_run((uint32_t)f[0], input, count, output) : 0;
    free(input);
    xs = term_pak(CID_NIL, 0);
    if (n != 0 && n != 1861) err_fail("native model output contract");
    for (uint32_t i = n; i > 0; --i) {
        uint32_t bits; memcpy(&bits, &output[i-1], sizeof(bits));
        xs = io_node(e, CID_CON, (Term)bits, xs);
    }
    return xs;
}
static Term native_close_run(Env e, Term *f, IoWork *w) {
    (void)e; (void)f; (void)w;
    deepfin_model_close();
    return term_pak(CID_UNIT, 0);
}
static void __attribute__((constructor)) native_model_effects(void) {
    io_eff(CID_NATIVE_OPEN, native_open_run, 0);
    io_eff(CID_NATIVE_SPEC, native_spec_run, 0);
    io_eff(CID_NATIVE_FORWARD, native_forward_run, 0);
    io_eff(CID_NATIVE_CLOSE, native_close_run, 0);
}
