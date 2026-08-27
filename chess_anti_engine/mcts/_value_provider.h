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
 * a message written into err). retain()/destroy() are a REFCOUNT PAIR, not an
 * alloc/free pair — see the contract on retain(). Weights are mapped read-only
 * and shared, so several trees in one process pointing at the same path cost
 * one mapping.
 *
 * ⚑⚑ HOW A PROVIDER REACHES THE TREE: THROUGH A CAPSULE, NEVER AN #include.
 *
 * Every function in an evaluator implemented as header-only statics is DUPLICATED
 * into each extension that includes it, and so is every static variable it owns.
 * A tree that got its provider by including the evaluator's header would hold a
 * SECOND copy of that evaluator's kernel-selection flag and weight cache: the
 * eval() the tree calls would ignore a set_simd() made through the evaluator's own
 * module, and the same weight file would be mapped twice. Both are silent — the
 * tree keeps returning plausible evaluations from the copy nobody configured.
 *
 * So the provider is published by the module that DEFINES it, as a PyCapsule named
 * CAE_VALUE_PROVIDER_CAPSULE_NAME wrapping a CaeValueProviderExport, and consumers
 * import that capsule instead of the header. One copy of the code, one copy of its
 * state, and "which provider is installed" stays answerable from the pointer the
 * tree actually holds. A future provider publishes the same capsule shape from its
 * own module; the tree needs no edit to accept it.
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

    /* ⚑ Take an additional reference to a ctx, returning it. REQUIRED, and it
     * pairs with destroy(): destroy DROPS one reference and frees only at zero.
     * A caller that is about to release the GIL and evaluate holds a reference
     * across the call, so another thread swapping or clearing the provider
     * cannot unmap the weights out from under an evaluation in flight. A
     * provider without this cannot be installed — the alternative is a
     * use-after-free that reads as a wrong evaluation. */
    void *(*retain)(void *ctx);

    /* Drop one reference from init()/retain(). Safe on NULL. */
    void (*destroy)(void *ctx);

    /* Which compute kernel this provider will actually use, e.g. "avx2" or
     * "scalar". Read by the CONSUMER off the vtable it holds, so a caller can
     * observe the state of the code that will really run rather than the state
     * of its own copy. NULL when the provider has no kernel choice to report. */
    const char *(*kernel_name)(void);

    /* ⚑⚑ NONZERO DECLARES THAT eval() IS *NOT* REENTRANT — the one carve-out
     * from the contract on eval() above, and the reason it lives in the VTABLE
     * rather than in any consumer's private list.
     *
     * A provider sets this when its eval mutates state shared across calls that
     * a concurrent call would corrupt: the position-DAG arm's probe -> evaluate
     * -> publish -> link path is not atomic, and its payload grow() frees an
     * accumulator array another thread may be reading. Such a provider must be
     * driven with the GIL held (or otherwise serialized).
     *
     * ⚑ WHY HERE AND NOT IN A CONSUMER-SIDE PREDICATE. The first version of
     * this guard was a function in the publishing module that listed the
     * affected vtables by pointer, and only that module's own batch loops
     * consulted it. MCTSTree never did — so the tree's exclusion rested on the
     * provider merely not being REACHABLE (absent from the name table, no
     * capsule exported), not on the stated rule. Exporting a capsule
     * symmetrically with the other arms, which the name table's own comment
     * invites, would have handed tree threads the non-atomic path with the GIL
     * released while the guard named in the docs never ran. That is a value
     * accepted and then silently ignored — this codebase's signature defect,
     * inside the fix for it. In the vtable, every consumer sees it, and
     * MCTSTree refuses such a provider AT INSTALL regardless of how it was
     * named or reached. */
    int requires_gil;
} CaeValueProvider;

/* Does driving this provider require holding the GIL / serializing calls?
 * Read off the vtable the consumer actually holds, never off a name. */
static inline int cae_value_requires_gil(const CaeValueProvider *vp) {
    return vp && vp->requires_gil;
}

/* Dispatch helper — the one call site a composing provider uses on its inner
 * evaluator, and the one the tree uses on its own. */
static inline int cae_value_eval(const CaeValueProvider *vp, void *ctx,
                                 const CBoard *board, int32_t *out_value) {
    if (!vp || !vp->eval) return CAE_VALUE_ERR_NOT_LOADED;
    return vp->eval(ctx, board, out_value);
}

/* Returns the retained ctx, or NULL if this provider cannot be held safely. */
static inline void *cae_value_retain(const CaeValueProvider *vp, void *ctx) {
    if (!vp || !vp->retain) return NULL;
    return vp->retain(ctx);
}

static inline void cae_value_destroy(const CaeValueProvider *vp, void *ctx) {
    if (vp && vp->destroy) vp->destroy(ctx);
}

/* ================================================================
 * The cross-extension publication shape
 * ================================================================ */

#define CAE_VALUE_PROVIDER_CAPSULE_NAME "cae.value_provider.v1"
/* ⚑ BUMPED TO 2 WHEN CaeValueProvider GAINED requires_gil. struct_size below
 * covers only CaeValueProviderExport's own layout, so a change to the VTABLE
 * struct is invisible to it: a stale consumer would keep reading the fields it
 * knows, never see the new one, and silently skip the guard that field exists
 * to arm. The version is what makes that combination refuse to load instead.
 * (The capsule NAME is an identifier, not the version, and stays as it is.) */
#define CAE_VALUE_PROVIDER_ABI 2u

/* PyObject, forward-declared: this header stays usable without Python.h, and
 * `struct _object *` is exactly what PyObject * is. */
struct _object;

typedef struct CaeValueProviderExport {
    uint32_t abi_version;               /* CAE_VALUE_PROVIDER_ABI */
    uint32_t struct_size;               /* sizeof(CaeValueProviderExport) */

    /* The vtable, owned by the publishing module and valid for its lifetime. */
    const CaeValueProvider *provider;

    /* The typed exception the publishing module raises for
     * CAE_VALUE_ERR_IN_CHECK, so a consumer can raise the SAME class rather
     * than a stringly-typed stand-in. The resolver that has to catch this is in
     * a different module again, and matching on message text is not a contract.
     * Borrowed; the publishing module holds the reference. */
    struct _object *in_check_error;
} CaeValueProviderExport;

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
