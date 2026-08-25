/*
 * _arm_providers.h — the two race arms, as providers on the eval seam.
 *
 * PR 1 shipped one provider: "nnue", the raw evaluator, which REFUSES a position
 * in check. That refusal is a contract, not a limitation, and it stays exactly
 * as it is — it is the enforcement backstop for everything below.
 *
 * This header adds the two arms the gate races, both of which are safe to call
 * on any legal position:
 *
 *   "nnue-static"   recursive check resolution + a static NNUE leaf
 *   "nnue-qsearch"  the SAME check resolution + tactical quiescence beyond it
 *
 * ⚑⚑ ARM FAIRNESS IS STRUCTURAL, NOT A PROMISE. Check resolution is mandatory
 * correctness work: without it either arm would ask the evaluator for a number
 * it does not define. So it must not show up as a cost of the qsearch arm, and
 * the way to guarantee that is to make it ONE component both arms call —
 * cae_resolve_eval() from ../mcts/_check_resolver.h. The two arms differ only in
 * the leaf hook they hand it. The qsearch arm cannot bypass it: its hook is
 * reachable only from inside the resolver, and every child it generates that is
 * in check goes back through cae_resolve_node() instead of to the evaluator.
 * Because both arms drive the same resolver, `nodes / resolved_leaves` and
 * `calls_in_check / calls` are DEFINED identically in both, which is what makes
 * them comparable. ⚑ Comparable is not equal: the qsearch arm's resolver figure
 * is larger because quiescence walks into check nodes of its own, and resolving
 * those is quiescence's cost, not a surcharge on the shared component. The
 * shared part — resolving the check the CALLER asked about — is byte-identical
 * work in both arms.
 *
 * ⚑ INCLUDED BY EXACTLY ONE TRANSLATION UNIT — _nnue_ext.c, the module that
 * PUBLISHES these providers. It pulls in _nnue_provider.h, whose header-only
 * statics (kernel flag, weight cache) must exist in exactly one copy. Consumers
 * take the published capsules; see ../mcts/_value_provider.h for why.
 */

#ifndef CAE_ARM_PROVIDERS_H
#define CAE_ARM_PROVIDERS_H

#include <stdlib.h>

#include "../mcts/_check_resolver.h"
#include "_nnue_provider.h"

/* Quiescence configuration.
 *
 * ⚑ THE PLY BUDGET IS MEASURED, NOT CHOSEN. This quiescence has no SEE or delta
 * pruning, so its cost is close to exponential in the budget while its VALUE
 * saturates early. Sweep over 600 stratified positions with the real big net,
 * scoring each budget against the deepest one available (mean |v - v@8|, in
 * internal units; /16 for cp):
 *
 *     max_ply  qnodes/eval   mean|v-v@8|   positions differing (of 600)
 *        0            1.0        985.7             201     <- pure stand-pat
 *        1           10.2        201.5             101
 *        2           12.0        182.3              62
 *        3           58.7         14.0              43
 *        4           72.4          6.5              20     <- the knee
 *        6          345.2          4.2              10
 *        8         1796.5          0.0               0
 *
 * From 4 to 8 is 25x the work to close a mean 6.5 internal units (~0.4 cp), on
 * 20 positions in 600. From 0 to 4 is 72x the work to close 985. So 4 it is —
 * and 8, which an earlier revision of this line had, would have made the arm
 * ~1100x slower than the static arm for a difference the corpus cannot see.
 * ⚑ The tail is NOT closed: max |v - v@8| is 2425 units at both 3 and 4 plies,
 * so a handful of positions really do need depth. That is a reason to sweep the
 * knob in the race, which is why it is a knob.
 *
 * Both numbers are counted against at run time (qply_cutoffs, qmax_ply_seen) so
 * "the budget was enough" stays an observation, not a belief. */
#define CAE_QSEARCH_DEFAULT_MAX_PLY     4
#define CAE_QSEARCH_DEFAULT_CHECK_PLIES 1

/* Wider than any score the resolver can produce (mate base is 100000), so an
 * initial window of +-this prunes nothing. */
#define CAE_QSEARCH_INF                 (CAE_RESOLVER_MATE_BASE * 2)

/* ================================================================
 * Configuration
 * ================================================================
 *
 * ⚑ MUTABLE STATICS, AND THAT IS SAFE *HERE* AND NOT IN _check_resolver.h. This
 * header is included by exactly one translation unit — the module that publishes
 * the providers — so there is one copy of these, and the tree runs the providers
 * it gets from that module's capsule. Set them through _nnue_ext.set_arm_config()
 * and the tree obeys, for the same reason set_simd() works. _check_resolver.h may
 * be included by several extensions and therefore owns no state at all.
 *
 * ⚑⚑ THEY ARE READ AT init(), NOT AT eval(). A context built before a change
 * keeps the old values for its whole life — this repo's signature defect is a
 * value accepted and then silently ignored, and a setter that appeared to retune
 * a running provider would be exactly that. So the numbers a context actually
 * used are reported back out of THAT CONTEXT (see CaeArmCtx and the stats dict),
 * never restated from these globals. */
/* ⚑ GUARDED BY A MUTEX, not by the GIL. init() runs from several call sites —
 * arm_open, arm_eval, and MCTSTree.set_value_provider — and "do all of them hold
 * the GIL, today and after the next edit" is a premise about code elsewhere, not
 * a property of this file. A torn read here would build a context with one
 * knob's old value and another's new one, and nothing downstream could tell. The
 * lock makes the triple a unit; the cost is one uncontended lock per context. */
static pthread_mutex_t g_arm_config_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_arm_resolver_max_depth   = CAE_RESOLVER_DEFAULT_MAX_DEPTH;
static int g_arm_qsearch_max_ply      = CAE_QSEARCH_DEFAULT_MAX_PLY;
static int g_arm_qsearch_check_plies  = CAE_QSEARCH_DEFAULT_CHECK_PLIES;

typedef struct CaeArmConfig {
    int resolver_max_depth;
    int qsearch_max_ply;
    int qsearch_check_plies;
} CaeArmConfig;

static void cae_arm_get_config(CaeArmConfig *out) {
    pthread_mutex_lock(&g_arm_config_lock);
    out->resolver_max_depth = g_arm_resolver_max_depth;
    out->qsearch_max_ply = g_arm_qsearch_max_ply;
    out->qsearch_check_plies = g_arm_qsearch_check_plies;
    pthread_mutex_unlock(&g_arm_config_lock);
}

/* Returns 0, or -1 with a reason written into err. Validates BEFORE taking the
 * lock so a rejected call cannot leave the triple half-written. */
static int cae_arm_set_config(int resolver_max_depth, int qsearch_max_ply,
                              int qsearch_check_plies, char *err, size_t errlen) {
    if (resolver_max_depth < 1 || resolver_max_depth > CAE_RESOLVER_MAX_PLIES) {
        cae_nnue_err(err, errlen, "resolver_max_depth must be in [1, %d], got %d",
                     CAE_RESOLVER_MAX_PLIES, resolver_max_depth);
        return -1;
    }
    /* 0 is meaningful and allowed: quiescence collapses to a stand-pat, which
     * makes the qsearch arm's leaf identical to the static arm's. That is the
     * negative control for the whole arm, so it must be reachable.
     *
     * The upper bound is resolver_max_depth because depth >= qply always (both
     * increment together from a depth that is already >= 0), so a quiescence
     * budget above the recursion cap could never be reached — a knob that is
     * accepted and then cannot take effect, which is the thing this file exists
     * to stop shipping. */
    if (qsearch_max_ply < 0 || qsearch_max_ply > resolver_max_depth) {
        cae_nnue_err(err, errlen, "qsearch_max_ply must be in [0, resolver_max_depth=%d], got %d",
                     resolver_max_depth, qsearch_max_ply);
        return -1;
    }
    if (qsearch_check_plies < 0 || qsearch_check_plies > qsearch_max_ply) {
        cae_nnue_err(err, errlen, "qsearch_check_plies must be in [0, qsearch_max_ply=%d], got %d",
                     qsearch_max_ply, qsearch_check_plies);
        return -1;
    }
    pthread_mutex_lock(&g_arm_config_lock);
    g_arm_resolver_max_depth = resolver_max_depth;
    g_arm_qsearch_max_ply = qsearch_max_ply;
    g_arm_qsearch_check_plies = qsearch_check_plies;
    pthread_mutex_unlock(&g_arm_config_lock);
    return 0;
}

/* ================================================================
 * Arm statistics
 * ================================================================ */

typedef struct CaeArmStats {
    /* The shared resolver's counters. Directly comparable between arms. */
    CaeResolverStats resolver;

    /* Quiescence-only counters, kept SEPARATE on purpose: folding them into the
     * resolver's would make the qsearch arm's expansion factor incomparable to
     * the static arm's, and that comparison is the readout. */
    uint64_t qnodes;           /* qsearch nodes entered (one stand-pat each) */
    uint64_t qterminal_draw;   /* draws found inside quiescence */
    uint64_t qply_cutoffs;     /* stood pat because a ply budget ran out */
    /* ⚑ Plies of QUIESCENCE, not depth below the resolution root. The two are
     * different numbers and reporting the wrong one was a real bug here. */
    uint32_t qmax_ply_seen;
} CaeArmStats;

static inline void cae_arm_stats_reset(CaeArmStats *s) { memset(s, 0, sizeof(*s)); }

typedef struct CaeArmCtx {
    CaeNnueWeights *weights;
    int resolver_max_depth;
    int qsearch_max_ply;
    /* How many qsearch plies also try non-capturing CHECKS. Testing a quiet move
     * for check costs a push, so the budget is bounded rather than unlimited. */
    int qsearch_check_plies;

    /* retain()/destroy() are a refcount PAIR per the seam contract, and the
     * count lives HERE rather than being inferred from the weights': the weight
     * cache is shared between contexts pointing at the same file, so its count
     * cannot say when THIS ctx is done with. */
    int refcount;

    /* Accumulated across every eval() through this ctx.
     * ⚑ Merged under atomics ONCE per top-level eval from a per-call block on
     * the stack, never incremented in place: eval() must be reentrant across
     * search threads, and an atomic in the resolver's inner loop would tax the
     * hot path to maintain a counter. */
    CaeArmStats totals;
} CaeArmCtx;

/* Raise *slot to `v` if v is larger. Relaxed: the counter is diagnostic and
 * orders nothing. */
static inline void cae_arm_atomic_max_u32(uint32_t *slot, uint32_t v) {
    uint32_t seen = __atomic_load_n(slot, __ATOMIC_RELAXED);
    while (v > seen
           && !__atomic_compare_exchange_n(slot, &seen, v, 1,
                                           __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
        /* `seen` is reloaded by the failed compare-exchange. */
    }
}

static void cae_arm_merge_stats(CaeArmStats *dst, const CaeArmStats *src) {
    __atomic_fetch_add(&dst->resolver.calls, src->resolver.calls, __ATOMIC_RELAXED);
    __atomic_fetch_add(&dst->resolver.calls_in_check, src->resolver.calls_in_check,
                       __ATOMIC_RELAXED);
    __atomic_fetch_add(&dst->resolver.nodes, src->resolver.nodes, __ATOMIC_RELAXED);
    __atomic_fetch_add(&dst->resolver.resolved_leaves, src->resolver.resolved_leaves,
                       __ATOMIC_RELAXED);
    __atomic_fetch_add(&dst->resolver.terminal_mate, src->resolver.terminal_mate,
                       __ATOMIC_RELAXED);
    __atomic_fetch_add(&dst->resolver.terminal_draw, src->resolver.terminal_draw,
                       __ATOMIC_RELAXED);
    __atomic_fetch_add(&dst->resolver.depth_cutoffs, src->resolver.depth_cutoffs,
                       __ATOMIC_RELAXED);
    __atomic_fetch_add(&dst->qnodes, src->qnodes, __ATOMIC_RELAXED);
    __atomic_fetch_add(&dst->qterminal_draw, src->qterminal_draw, __ATOMIC_RELAXED);
    __atomic_fetch_add(&dst->qply_cutoffs, src->qply_cutoffs, __ATOMIC_RELAXED);

    cae_arm_atomic_max_u32(&dst->resolver.max_depth_seen, src->resolver.max_depth_seen);
    cae_arm_atomic_max_u32(&dst->qmax_ply_seen, src->qmax_ply_seen);
}

/* ================================================================
 * Arm A — static NNUE leaf
 * ================================================================ */

/* The leaf hook for the static arm. `ply` is unused: a static evaluation does
 * not depend on how deep it was reached. */
static int cae_arm_static_leaf(void *leaf_ctx, const CBoard *board, int ply,
                               int32_t *out_value) {
    (void)ply;
    const CaeArmCtx *ctx = (const CaeArmCtx *)leaf_ctx;
    int32_t v = 0;
    /* ⚑ Through the evaluator's normal entry point, refusal and all. The
     * resolver calls this only at a non-check node, so a refusal here would mean
     * the resolver had a hole — and it surfaces as a hard error rather than as a
     * plausible number. */
    int status = cae_nnue_evaluate_cboard(ctx->weights, board, &v);
    if (status != CAE_VALUE_OK) return status;
    /* Clamped AT THE HOOK, where "this is a raw evaluation and not a mate score"
     * is known. The resolver passes mate-band values through untouched (it has
     * to — the qsearch hook can legitimately return one), so a raw evaluation
     * has to be kept out of that band here. */
    *out_value = cae_resolver_clamp(v);
    return CAE_VALUE_OK;
}

/* ================================================================
 * Arm B — tactical quiescence beyond the mandatory resolution
 * ================================================================ */

typedef struct CaeQsearchCtx {
    const CaeArmCtx *arm;
    const CaeResolverCtx *resolver;   /* the SAME resolver, re-entered for checks */
    CaeArmStats *stats;               /* the per-call block, not the accumulator */

    /* ⚑⚑ THE QUIESCENCE PLY IN FORCE WHEN QUIESCENCE HANDED OFF TO THE RESOLVER.
     *
     * The resolver's leaf hook has to start quiescence at SOME ply, and neither
     * obvious answer is right:
     *   - the resolver's `depth` (the original bug) makes an in-check root start
     *     quiescence already over budget;
     *   - a literal 0 REFUNDS the budget: quiescence plays a checking move, the
     *     resolver resolves it, and its leaves restart quiescence with a full
     *     allowance — measured, and it does not finish. A knob that cannot bound
     *     the work it names is not a budget.
     * So the count is CARRIED across the excursion. Check resolution is mandatory
     * and free; quiescence moves are what the budget buys.
     *
     * Mutable, saved and restored around each excursion. Safe because this struct
     * is a per-call stack object and the recursion is depth-first — there is one
     * live path at a time, and no other thread can see it. */
    int handoff_qply;
} CaeQsearchCtx;

/* Does this legal move change material — capture, en passant, or promotion?
 *
 * ⚑ NOT cboard_move_is_zeroing(): that answers "pawn move or capture", which is
 * a different question — a quiet pawn push is zeroing and is not tactical.
 * Reusing it here would silently widen quiescence to every pawn move.
 *
 * ⚑⚑ AND `promotion > 0` IS NOT "THIS MOVE PROMOTES". POLICY_LUT's promotion
 * field is a SPECULATIVE flag: init_policy_lut stamps PROMO_MAYBE_QUEEN (5) onto
 * every entry whose destination is rank 0 or rank 7, whatever piece is moving,
 * and cboard_push resolves it by checking at push time whether a pawn was
 * actually involved. So a rook lifting to the back rank carries promotion == 5.
 * Trusting the field by its name made quiescence treat every back-rank rook and
 * queen move as tactical — a quiet-move flood that the check-ply budget could
 * not govern because those moves were never classified as quiet. The piece test
 * is what makes the field mean what it is being read to mean. */
static inline int cae_qsearch_is_tactical(const CBoard *b, int from_sq, int to_sq,
                                          int promotion) {
    if (b->occ[1 - b->turn] & sq_bit(to_sq)) return 1;   /* capture */
    if (piece_type_at(b, from_sq) != PAWN) return 0;     /* see ⚑⚑ above */
    if (promotion > 0) return 1;                         /* promotion */
    if (to_sq == b->ep_square && sq_file(from_sq) != sq_file(to_sq))
        return 1;                                        /* en passant */
    return 0;
}

/* ⚑⚑ TWO COUNTERS, AND CONFLATING THEM IS A REAL BUG THIS CODE ONCE HAD.
 *
 *   `depth` — plies below the RESOLUTION ROOT, through both directions of the
 *             mutual recursion. It bounds the whole thing (resolver_max_depth)
 *             and it is what mate scores are measured in, so it must keep
 *             increasing monotonically no matter which side of the recursion
 *             we are on.
 *   `qply`  — plies of QUIESCENCE, reset to 0 every time the resolver hands a
 *             node to the leaf hook. Both quiescence budgets are counted in it.
 *
 * The first version passed the resolver's `depth` as quiescence's ply. The
 * resolver only calls its leaf hook at a NON-CHECK node, so an in-check root
 * enters quiescence at depth >= 1 — and `try_checks = (ply < check_plies)` with
 * the default check_plies=1 was therefore FALSE on exactly the positions the
 * check budget exists for. A check chain 4 deep entered quiescence at ply 4 and
 * stood pat without looking at a single capture. Both knobs were accepted,
 * consumed, and silently meant something else: this repo's signature defect,
 * inside the fix for the last one.
 *
 * TERMINATION still holds, and on `depth` rather than on `qply`: every edge in
 * either direction increments depth, both sides test it against
 * resolver_max_depth, so no cycle can restart a budget. `qply` only ever makes
 * quiescence stop SOONER. */

/* Known to be neither terminal nor in check. */
static int cae_qsearch_node(CaeQsearchCtx *q, const CBoard *b, int qply, int depth,
                            int32_t alpha, int32_t beta, int32_t *out_value);

/* Dispatch a child. `depth` is the CHILD's depth below the resolution root.
 *
 * ⚑ THE ORDER MATCHES cae_resolve_node'S, AND FOR THE SAME REASON: in check
 * goes first. cboard_is_game_over() tests the fifty-move clock and the "no legal
 * moves" condition without asking whether the side to move is in check, so
 * testing drawn-ness first would score a CHECKMATE as a draw. The resolver
 * scores mate; only once it has declined does "no moves" mean stalemate. */
static int cae_qsearch_child(CaeQsearchCtx *q, const CBoard *b, int qply, int depth,
                             int32_t alpha, int32_t beta, int32_t *out_value) {
    if (cboard_in_check(b)) {
        /* ⚑ BACK THROUGH THE SHARED RESOLVER. The evaluator is undefined in
         * check, so a checking move's child is RESOLVED, never evaluated — the
         * same code, and the same cost, that the static arm pays. The resolver
         * takes no window: it is exact minimax, and a bound would change what
         * its result means to whoever stores it.
         *
         * `depth`, not `qply`: the resolver measures mate distance from the
         * resolution root, and it owns the cap that makes all of this finite.
         *
         * ⚑ The quiescence budget is CARRIED into the excursion, so quiescence
         * resumed at the resolver's leaves picks up where it left off instead of
         * being handed a fresh allowance. Save/restore, because this frame
         * resumes its own sibling loop afterwards. */
        int saved = q->handoff_qply;
        q->handoff_qply = qply;
        int rc = cae_resolve_node(q->resolver, b, depth, out_value);
        q->handoff_qply = saved;
        return rc;
    }
    if (cae_resolver_is_drawn(b)) {
        q->stats->qterminal_draw++;
        *out_value = 0;
        return CAE_VALUE_OK;
    }
    return cae_qsearch_node(q, b, qply, depth, alpha, beta, out_value);
}

/* The leaf hook for the qsearch arm: the resolver hands it a node it has already
 * established is neither in check nor terminal, and quiescence takes over from
 * there. A full window, because the resolver above it is exact minimax and no
 * bound from it can be carried across.
 *
 * ⚑ QUIESCENCE RESUMES AT handoff_qply, NOT AT THE RESOLVER'S DEPTH. "How many
 * plies of quiescence have I done" is what both quiescence budgets ask, and the
 * resolver's depth is not an answer to it — it counts forced evasions too. At the
 * resolution root handoff_qply is 0; inside an excursion quiescence itself
 * started, it is the ply quiescence had reached. See CaeQsearchCtx. */
static int cae_arm_qsearch_leaf(void *leaf_ctx, const CBoard *board, int ply,
                                int32_t *out_value) {
    CaeQsearchCtx *q = (CaeQsearchCtx *)leaf_ctx;
    return cae_qsearch_node(q, board, q->handoff_qply, ply,
                            -CAE_QSEARCH_INF, CAE_QSEARCH_INF, out_value);
}

/* Stand-pat quiescence over captures, promotions, and — for the first
 * qsearch_check_plies plies — checking moves. Fail-soft negamax. */
static int cae_qsearch_node(CaeQsearchCtx *q, const CBoard *b, int qply, int depth,
                            int32_t alpha, int32_t beta, int32_t *out_value) {
    const CaeArmCtx *arm = q->arm;
    q->stats->qnodes++;
    if ((uint32_t)qply > q->stats->qmax_ply_seen) q->stats->qmax_ply_seen = (uint32_t)qply;

    int32_t stand_pat = 0;
    int status = cae_nnue_evaluate_cboard(arm->weights, b, &stand_pat);
    if (status != CAE_VALUE_OK) return status;
    stand_pat = cae_resolver_clamp(stand_pat);

    if (stand_pat >= beta) { *out_value = stand_pat; return CAE_VALUE_OK; }
    if (stand_pat > alpha) alpha = stand_pat;

    /* The quiescence budget is in qply; the recursion cap that makes the whole
     * mutual recursion finite is in depth. See the ⚑⚑ block above for why these
     * cannot be the same number. */
    if (qply >= arm->qsearch_max_ply || depth >= arm->resolver_max_depth) {
        q->stats->qply_cutoffs++;
        /* ⚑ stand_pat, NOT alpha. No move was searched here, so stand_pat is
         * this node's value; alpha at this point is max(stand_pat, whatever the
         * caller already had), and returning that would report a sibling's score
         * as this node's. Fail-soft means "return the best value you actually
         * found", and the best value found here is the one evaluation made. */
        *out_value = stand_pat;
        return CAE_VALUE_OK;
    }

    int moves[CBOARD_MAX_LEGAL_MOVES];
    int n = cboard_legal_move_indices(b, moves, 0);
    int try_checks = (qply < arm->qsearch_check_plies);
    int32_t best = stand_pat;

    for (int i = 0; i < n; i++) {
        PolicyMove pm = POLICY_LUT[b->turn][moves[i]];
        if (pm.from_sq < 0 || pm.to_sq < 0) continue;
        int tactical = cae_qsearch_is_tactical(b, pm.from_sq, pm.to_sq, pm.promotion);
        if (!tactical && !try_checks) continue;

        CBoard child;
        memcpy(&child, b, sizeof(CBoard));
        cboard_push_index(&child, moves[i]);

        /* A quiet move earns a search only if it gives check, and testing that
         * needs the push we just made. */
        int gives_check = cboard_in_check(&child);
        if (!tactical && !gives_check) continue;

        int32_t child_value = 0;
        int rc = cae_qsearch_child(q, &child, qply + 1, depth + 1, -beta, -alpha,
                                   &child_value);
        if (rc != CAE_VALUE_OK) return rc;
        child_value = -child_value;   /* a child's value is child-STM POV */

        if (child_value > best) best = child_value;
        if (child_value > alpha) alpha = child_value;
        if (alpha >= beta) break;     /* fail-soft cutoff */
    }

    *out_value = best;
    return CAE_VALUE_OK;
}

/* ================================================================
 * The provider vtables
 * ================================================================ */

static void *cae_arm_init(const char *weights_path, char *err, size_t errlen) {
    CaeNnueWeights *w = cae_nnue_load(weights_path, err, errlen);
    if (!w) return NULL;
    CaeArmCtx *ctx = (CaeArmCtx *)calloc(1, sizeof(CaeArmCtx));
    if (!ctx) {
        cae_nnue_release(w);
        cae_nnue_err(err, errlen, "out of memory");
        return NULL;
    }
    ctx->weights = w;
    /* Snapshotted here, once, as a UNIT under the lock. See the ⚑⚑ on the
     * globals: the ctx's own copy is what every eval through it uses and what
     * its stats report. */
    CaeArmConfig cfg;
    cae_arm_get_config(&cfg);
    ctx->resolver_max_depth = cfg.resolver_max_depth;
    ctx->qsearch_max_ply = cfg.qsearch_max_ply;
    ctx->qsearch_check_plies = cfg.qsearch_check_plies;
    ctx->refcount = 1;
    cae_arm_stats_reset(&ctx->totals);
    return ctx;
}

static void *cae_arm_retain(void *ctx_void) {
    CaeArmCtx *ctx = (CaeArmCtx *)ctx_void;
    if (!ctx) return NULL;
    __atomic_fetch_add(&ctx->refcount, 1, __ATOMIC_RELAXED);
    return ctx;
}

static void cae_arm_destroy(void *ctx_void) {
    CaeArmCtx *ctx = (CaeArmCtx *)ctx_void;
    if (!ctx) return;
    /* acq_rel so every write made through a reference happens-before the free. */
    if (__atomic_fetch_sub(&ctx->refcount, 1, __ATOMIC_ACQ_REL) != 1) return;
    cae_nnue_release(ctx->weights);
    free(ctx);
}

static const char *cae_arm_kernel_name(void) {
    return cae_nnue_simd_active() ? "avx2" : "scalar";
}

static int cae_arm_static_eval(void *ctx_void, const CBoard *board, int32_t *out_value) {
    CaeArmCtx *ctx = (CaeArmCtx *)ctx_void;
    CaeArmStats local;
    cae_arm_stats_reset(&local);

    CaeResolverCtx rc;
    rc.leaf_eval = cae_arm_static_leaf;
    rc.leaf_ctx = ctx;
    rc.max_depth = ctx->resolver_max_depth;
    rc.stats = &local.resolver;

    int status = cae_resolve_eval(&rc, board, out_value);
    cae_arm_merge_stats(&ctx->totals, &local);
    return status;
}

static int cae_arm_qsearch_eval(void *ctx_void, const CBoard *board, int32_t *out_value) {
    CaeArmCtx *ctx = (CaeArmCtx *)ctx_void;
    CaeArmStats local;
    cae_arm_stats_reset(&local);

    CaeResolverCtx rc;
    CaeQsearchCtx q;
    q.arm = ctx;
    q.resolver = &rc;
    q.stats = &local;
    q.handoff_qply = 0;   /* the resolution root's own leaves start quiescence */
    rc.leaf_eval = cae_arm_qsearch_leaf;
    rc.leaf_ctx = &q;
    rc.max_depth = ctx->resolver_max_depth;
    rc.stats = &local.resolver;

    int status = cae_resolve_eval(&rc, board, out_value);
    cae_arm_merge_stats(&ctx->totals, &local);
    return status;
}

static const CaeValueProvider CAE_ARM_STATIC_PROVIDER = {
    "nnue-static",
    cae_arm_init,
    cae_arm_static_eval,
    cae_arm_retain,
    cae_arm_destroy,
    cae_arm_kernel_name,
};

static const CaeValueProvider CAE_ARM_QSEARCH_PROVIDER = {
    "nnue-qsearch",
    cae_arm_init,
    cae_arm_qsearch_eval,
    cae_arm_retain,
    cae_arm_destroy,
    cae_arm_kernel_name,
};

/* ================================================================
 * The registry
 * ================================================================
 *
 * One table, listing every provider this module publishes. It lives here rather
 * than in _nnue_provider.h so that "which providers exist" has a single answer:
 * a second array that also called itself the registry is exactly the shape of
 * defect where a value is accepted and then silently ignored. */
static const CaeValueProvider *const CAE_VALUE_PROVIDERS[] = {
    &CAE_NNUE_PROVIDER,
    &CAE_ARM_STATIC_PROVIDER,
    &CAE_ARM_QSEARCH_PROVIDER,
    NULL
};

static const CaeValueProvider *cae_value_provider_by_name(const char *name) {
    for (int i = 0; CAE_VALUE_PROVIDERS[i]; i++)
        if (strcmp(CAE_VALUE_PROVIDERS[i]->name, name) == 0)
            return CAE_VALUE_PROVIDERS[i];
    return NULL;
}

/* Is this provider one of the resolver-backed arms — i.e. does its ctx carry a
 * CaeArmStats a caller may read? Answered from the VTABLE POINTER, not from the
 * name string, so nothing can talk itself into casting the raw evaluator's ctx
 * to a CaeArmCtx by passing a lookalike name. */
static inline int cae_provider_is_arm(const CaeValueProvider *vp) {
    return vp == &CAE_ARM_STATIC_PROVIDER || vp == &CAE_ARM_QSEARCH_PROVIDER;
}

#endif /* CAE_ARM_PROVIDERS_H */
