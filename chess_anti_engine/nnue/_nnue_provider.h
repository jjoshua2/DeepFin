/*
 * _nnue_provider.h — the NNUE evaluator dressed as a CaeValueProvider.
 *
 * This is the first provider on the eval seam. It is deliberately thin: all it
 * does is map init/eval/destroy onto cae_nnue_load / cae_nnue_evaluate_cboard /
 * cae_nnue_release, and hand the in-check refusal straight through. A second
 * provider (a leaf qsearch on the same NNUE eval) composes by holding an inner
 * {provider, ctx} pair and calling cae_value_eval() on it.
 *
 * ⚑ THIS HEADER IS INCLUDED BY EXACTLY ONE TRANSLATION UNIT — _nnue_ext.c, the
 * module that PUBLISHES the provider. Everything below is header-only statics,
 * so a second includer would get a second copy of the evaluator's kernel flag
 * and weight cache and quietly diverge from the one the publishing module
 * configures. Consumers take the PyCapsule instead; see the capsule section of
 * ../mcts/_value_provider.h for why that is not a stylistic preference.
 */

#ifndef CAE_NNUE_PROVIDER_H
#define CAE_NNUE_PROVIDER_H

#include "_nnue_impl.h"   /* pulls in ../mcts/_value_provider.h */

static void *cae_nnue_provider_init(const char *weights_path, char *err, size_t errlen) {
    return (void *)cae_nnue_load(weights_path, err, errlen);
}

static int cae_nnue_provider_eval(void *ctx, const CBoard *board, int32_t *out_value) {
    return cae_nnue_evaluate_cboard((const CaeNnueWeights *)ctx, board, out_value);
}

static void *cae_nnue_provider_retain(void *ctx) {
    return (void *)cae_nnue_retain((CaeNnueWeights *)ctx);
}

static void cae_nnue_provider_destroy(void *ctx) {
    cae_nnue_release((CaeNnueWeights *)ctx);
}

/* Reports the kernel THIS copy of the evaluator will run, so a consumer holding
 * the vtable can observe it without owning the flag. */
static const char *cae_nnue_provider_kernel_name(void) {
    return cae_nnue_simd_active() ? "avx2" : "scalar";
}

static const CaeValueProvider CAE_NNUE_PROVIDER = {
    "nnue",
    cae_nnue_provider_init,
    cae_nnue_provider_eval,
    cae_nnue_provider_retain,
    cae_nnue_provider_destroy,
    cae_nnue_provider_kernel_name,
};

/* The registry the tree looks a provider up in by name. PR 1 ships exactly one
 * entry; the point of the table is that the next one is an append, not a
 * rewrite of the call site. */
static const CaeValueProvider *const CAE_VALUE_PROVIDERS[] = {
    &CAE_NNUE_PROVIDER,
    NULL
};

static const CaeValueProvider *cae_value_provider_by_name(const char *name) {
    for (int i = 0; CAE_VALUE_PROVIDERS[i]; i++)
        if (strcmp(CAE_VALUE_PROVIDERS[i]->name, name) == 0)
            return CAE_VALUE_PROVIDERS[i];
    return NULL;
}

#endif /* CAE_NNUE_PROVIDER_H */
