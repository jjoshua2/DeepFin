/*
 * _value_provider.h — the eval-plugin seam in the C Gumbel tree.
 *
 * The tree holds a POINTER to a provider, never a hard-wired call. That is the
 * whole point: the same seam has to carry, later, a leaf qsearch, per-node mate
 * extensions, and composed evaluators — a provider that wants an inner
 * evaluator keeps the inner {vtable, ctx} pair in its own ctx and calls
 * cae_value_eval() on it, so composition is recursion through this same
 * interface rather than a second mechanism.
 *
 * ⚑⚑ THE CONTRACT ON eval():
 *
 *   1. It returns a STATUS, and writes the value through an out-parameter. It
 *      does NOT return the value directly. This is not ceremony: the NNUE
 *      evaluation is undefined in check and must be refusable, and a refusal
 *      encoded as some in-band int32 is a sentinel a caller can read as an
 *      evaluation. On any status other than CAE_VALUE_OK the out-parameter is
 *      untouched.
 *
 *   2. CALLERS MUST RESOLVE CHECK NODES RECURSIVELY BEFORE CALLING eval. A
 *      position in check is never statically evaluated: its evasions are
 *      searched, and because an evasion can itself give check the resolution is
 *      recursive (minimax backup, repetition and 50-move terminals handled
 *      inside the resolver, mate when there are no evasions), continuing until
 *      a non-check position or a terminal is reached. The provider's
 *      CAE_VALUE_ERR_IN_CHECK refusal is the ENFORCEMENT BACKSTOP for that
 *      invariant, not a substitute for it — a caller that leans on the refusal
 *      is a caller with a hole in its search.
 *
 * init() takes a weights path and returns an opaque ctx (NULL on failure, with
 * a message written into err). destroy() releases it. Weights are mapped
 * read-only and shared, so several trees in one process pointing at the same
 * path cost one mapping.
 */

#ifndef CAE_VALUE_PROVIDER_H
#define CAE_VALUE_PROVIDER_H

#include <stddef.h>
#include <stdint.h>

#include "_cboard_impl.h"

/* This header OWNS the status contract; providers include it for the codes they
 * must return. The dependency runs evaluator -> seam, which is the right way
 * round: a provider implements a contract the seam defines, and nothing here
 * depends on the tree's own types (only on CBoard and stdint), so including it
 * from an evaluator costs nothing. */
typedef enum {
    CAE_VALUE_OK             = 0,
    CAE_VALUE_ERR_IN_CHECK   = -1,
    CAE_VALUE_ERR_NOT_LOADED = -2,
    CAE_VALUE_ERR_BAD_POS    = -3
} CaeValueStatus;

typedef struct CaeValueProvider {
    /* Stable identifier, e.g. "nnue". Reported back from the CONSUMER's stored
     * pointer, so "which provider is live" is read off the thing that does the
     * work rather than off the argument someone passed in. */
    const char *name;

    /* Build a context from a weights path. NULL on failure; err gets why. */
    void *(*init)(const char *weights_path, char *err, size_t errlen);

    /* Evaluate. Returns CAE_VALUE_OK and writes *out_value, or a negative
     * CaeValueStatus and writes nothing. Must be reentrant: several search
     * threads share one ctx. */
    int (*eval)(void *ctx, const CBoard *board, int32_t *out_value);

    /* Release a context from init(). Safe on NULL. */
    void (*destroy)(void *ctx);
} CaeValueProvider;

/* Dispatch helper — the one call site a composing provider uses on its inner
 * evaluator, and the one the tree uses on its own. */
static inline int cae_value_eval(const CaeValueProvider *vp, void *ctx,
                                 const CBoard *board, int32_t *out_value) {
    if (!vp || !vp->eval) return CAE_VALUE_ERR_NOT_LOADED;
    return vp->eval(ctx, board, out_value);
}

static inline const char *cae_value_status_name(int status) {
    switch (status) {
        case CAE_VALUE_OK:             return "ok";
        case CAE_VALUE_ERR_IN_CHECK:   return "in-check (resolve check nodes recursively)";
        case CAE_VALUE_ERR_NOT_LOADED: return "no value provider loaded";
        case CAE_VALUE_ERR_BAD_POS:    return "malformed position";
        default:                       return "unknown value-provider status";
    }
}

#endif /* CAE_VALUE_PROVIDER_H */
