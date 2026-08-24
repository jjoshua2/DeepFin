/*
 * _nnue_provider.h — the NNUE evaluator dressed as a CaeValueProvider.
 *
 * This is the first provider on the eval seam. It is deliberately thin: all it
 * does is map init/eval/destroy onto cae_nnue_load / cae_nnue_evaluate_cboard /
 * cae_nnue_release, and hand the in-check refusal straight through. A second
 * provider (a leaf qsearch on the same NNUE eval) composes by holding an inner
 * {provider, ctx} pair and calling cae_value_eval() on it.
 *
 * Header-only and static, like the rest of the C here: each extension that
 * includes it gets its own copy, and the weight mapping is shared through the
 * page cache rather than through a symbol.
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

static void cae_nnue_provider_destroy(void *ctx) {
    cae_nnue_release((CaeNnueWeights *)ctx);
}

static const CaeValueProvider CAE_NNUE_PROVIDER = {
    "nnue",
    cae_nnue_provider_init,
    cae_nnue_provider_eval,
    cae_nnue_provider_destroy,
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
