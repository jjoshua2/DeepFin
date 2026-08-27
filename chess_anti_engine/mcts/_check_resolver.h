/*
 * _check_resolver.h — recursive check resolution, tree-side search infrastructure.
 *
 * ⚑⚑ THE INVARIANT: THE EVALUATOR IS NEVER INVOKED ON A NODE WHOSE SIDE TO MOVE
 * IS IN CHECK.
 *
 * The NNUE evaluation is undefined in check — Stockfish's own `eval` refuses
 * such a position outright, and its search never evaluates one. An earlier
 * design said "extend one ply out of check"; that is not sufficient, because an
 * evasion can itself give check. So resolution is RECURSIVE: a check node
 * generates its evasions, recurses on each child, and backs the results up by
 * MINIMAX (forced resolution — never an average, because the side to move has
 * no choice about being in check and every evasion is a move it must survive).
 * Recursion continues until a non-check position or a terminal.
 *
 * WHY IT TERMINATES. Every recursive step is a move, so it either resets the
 * halfmove clock (a capture or a pawn move, of which there can only be finitely
 * many) or repeats a position within the current reversible run. A perpetual
 * check therefore hits the repetition terminal, which is what makes an
 * unbounded-looking recursion finite. The depth cap below is a backstop for the
 * pathological remainder and is COUNTED, not assumed away.
 *
 * WHAT THIS IS NOT. It is not part of the evaluator, and the evaluator's
 * unconditional in-check refusal stays exactly as it is. That refusal is the
 * ENFORCEMENT BACKSTOP for this invariant: if the resolver ever has a hole, the
 * symptom is a loud CAE_VALUE_ERR_IN_CHECK, never a silently wrong evaluation.
 * A resolver that "handles" the refusal by returning a number would convert the
 * backstop into exactly the defect it exists to prevent.
 *
 * ⚑ ARM FAIRNESS. Check resolution is MANDATORY SHARED infrastructure for both
 * race arms: the static arm is (NNUE leaf + this), the qsearch arm is (NNUE leaf
 * + tactical quiescence + this). It is one component, parameterised by what
 * happens at a resolved non-check node — see CaeLeafEvalFn — so that required
 * correctness work can never be scored as a qsearch advantage. The qsearch arm
 * reaches its own check nodes through this same function.
 *
 * ⚑ NO MUTABLE STATICS LIVE HERE. All counters are caller-owned (CaeResolverStats
 * passed in), so this header may safely be included by more than one extension
 * without the two copies drifting — unlike an evaluator, which must be reached
 * through its capsule. See _value_provider.h for that rule.
 */

#ifndef CAE_CHECK_RESOLVER_H
#define CAE_CHECK_RESOLVER_H

#include <stdint.h>
#include <string.h>

#include "_search_terminal.h"
#include "_value_provider.h"

/* ================================================================
 * Score scale — integer end to end
 * ================================================================
 *
 * Resolver scores are in the evaluator's own units (NNUE internal units,
 * psqt/16 + positional/16, side-to-move POV), so a resolved value is a drop-in
 * replacement for the static evaluation it stands in for. No floats appear in
 * any scoring path here.
 *
 * ⚑⚑ THE MATE BAND MUST NOT OVERLAP THE EVALUATION BAND, and this repo has
 * already paid for getting that wrong once. Audit N1: the codebase carried two
 * mate->cp formulas, one of which folded mates into +-1500..2480 while real cp
 * scores reach +-32000 — so on 1.34% of live rows a mate scored BELOW a plain
 * evaluation and the two mappings named different best moves. The fix was to put
 * the mate band entirely above the evaluation clamp, and
 * tests/test_mate_score_single_home.py pins that inequality. The same property
 * is required here, in this scale, and is pinned the same way.
 *
 *   CAE_RESOLVER_EVAL_CLAMP  32000   every leaf value is clamped into +-this
 *   mate band floor          74400 = 100000 - 256*100
 *
 * The clamp is measured, not guessed: over the 49,619-position stratified parity
 * pool the largest |evaluation| is 14,295, so the clamp is more than 2x above
 * anything observed and never binds in practice. Its job is to make the band
 * separation TOTAL rather than empirical.
 *
 * ⚑ THIS IS A SEARCH-INTERNAL SCALE AND IT IS NOT CENTIPAWNS. It shares numeric
 * constants with stockfish/wdl.py's cp mapping by deliberate analogy, NOT by
 * shared definition — the units differ. A resolver score must never be handed to
 * the training-target pipeline as a cp value; anything that needs to become a
 * target goes through mate_to_effective_cp, which remains the single home for
 * the mate->cp mapping. Adding a second home is the defect this comment exists
 * to prevent someone from re-introducing.
 */
#define CAE_RESOLVER_EVAL_CLAMP      32000
#define CAE_RESOLVER_MATE_BASE      100000
#define CAE_RESOLVER_MATE_PLY_STEP     100
#define CAE_RESOLVER_MAX_PLIES         256

/* Default recursion cap. Each frame holds a CBoard copy plus a move buffer
 * (~2.6 KB), so this bounds the resolver's stack at ~85 KB. Check chains are
 * short in practice — a perpetual terminates at the two-fold repetition — and
 * cutoffs are counted so "it never fires" is an observation rather than a
 * belief. */
#define CAE_RESOLVER_DEFAULT_MAX_DEPTH 32

/* ================================================================
 * Instrumentation
 * ================================================================ */

typedef struct CaeResolverStats {
    /* Top-level cae_resolve_eval() calls, and how many of them arrived in
     * check. Their ratio is the IN-CHECK LEAF FRACTION the two-arm readout
     * reports. */
    uint64_t calls;
    uint64_t calls_in_check;

    /* Every node the resolver visited, and every non-check node at which it
     * called the leaf evaluator. nodes / resolved_leaves is the EXPANSION
     * FACTOR — how much work one resolved evaluation costs. */
    uint64_t nodes;
    uint64_t resolved_leaves;

    /* Terminals, split so a readout can say which rule ended a line. */
    uint64_t terminal_mate;
    uint64_t terminal_draw;

    /* ⚑ Must stay 0. A cutoff means a line was neither resolved nor terminal
     * and was scored as a draw — defensible as a backstop, indefensible as a
     * silent one, so it is counted and reported. */
    uint64_t depth_cutoffs;

    /* Deepest recursion actually reached, for sizing the cap against reality. */
    uint32_t max_depth_seen;
} CaeResolverStats;

static inline void cae_resolver_stats_reset(CaeResolverStats *s) {
    memset(s, 0, sizeof(*s));
}

/* ================================================================
 * The leaf hook — what makes one resolver serve both arms
 * ================================================================ */

/* Called ONLY at a node that is not in check and not terminal. Returns a
 * CaeValueStatus and writes the value through the out-parameter, exactly like
 * the provider seam it wraps.
 *
 * The static arm passes a hook that calls the NNUE provider. The qsearch arm
 * passes a hook that runs tactical quiescence and calls the NNUE provider at ITS
 * quiet leaves — and re-enters this resolver for any check node it meets. Both
 * therefore pay for check resolution and neither can skip it.
 *
 * ⚑ `ply` is the depth below the resolution root, and it is ONE budget shared by
 * both directions of that mutual recursion. Without it the qsearch hook could
 * not know how deep it already was, and a qsearch->resolver->qsearch chain would
 * each restart its own counter — a recursion that terminates only by luck. */
typedef int (*CaeLeafEvalFn)(void *leaf_ctx, const CBoard *board, int ply,
                             int32_t *out_value);

typedef struct CaeResolverCtx {
    CaeLeafEvalFn leaf_eval;
    void *leaf_ctx;
    int max_depth;
    CaeResolverStats *stats;   /* may be NULL */
} CaeResolverCtx;

static inline int32_t cae_resolver_clamp(int32_t v) {
    if (v > CAE_RESOLVER_EVAL_CLAMP) return CAE_RESOLVER_EVAL_CLAMP;
    if (v < -CAE_RESOLVER_EVAL_CLAMP) return -CAE_RESOLVER_EVAL_CLAMP;
    return v;
}

/* The score for "the side to move is mated, `depth` plies below the root".
 * Negative because it is from the mated side's own point of view; deeper mates
 * are less bad, which is what makes a shorter mate preferred once negated. */
static inline int32_t cae_resolver_mated_score(int depth) {
    int d = depth < 0 ? 0 : (depth > CAE_RESOLVER_MAX_PLIES ? CAE_RESOLVER_MAX_PLIES : depth);
    return -(CAE_RESOLVER_MATE_BASE - d * CAE_RESOLVER_MATE_PLY_STEP);
}

static inline int cae_resolver_is_mate_score(int32_t v) {
    int32_t floor_ = CAE_RESOLVER_MATE_BASE - CAE_RESOLVER_MAX_PLIES * CAE_RESOLVER_MATE_PLY_STEP;
    return v >= floor_ || v <= -floor_;
}

/* ================================================================
 * The resolver
 * ================================================================ */

static int cae_resolve_node(const CaeResolverCtx *rc, const CBoard *b, int depth,
                            int32_t *out_value);

/* Terminal decision, taken from the TREE'S OWN source of truth.
 *
 * ⚑ cboard_search_terminal() is what the MCTS search already uses to decide
 * that a node is over — game-over (50-move, three-fold, insufficient material,
 * no legal moves) plus LC0-style two-fold-as-draw. Re-deriving the rules here
 * would give the resolver and the tree two answers to one question, and the
 * two-fold rule in particular is what makes perpetual check terminate promptly.
 * Only its BOOLEAN is used: its Q is a double, and nothing in this file's
 * scoring is allowed to be. */
static inline int cae_resolver_is_drawn(const CBoard *b) {
    double q;
    int8_t solved;
    return cboard_search_terminal(b, &q, &solved);
}

/* Resolve one node. Returns CAE_VALUE_OK and writes *out_value, or a negative
 * CaeValueStatus (propagated from the leaf hook) and writes nothing. */
static int cae_resolve_node(const CaeResolverCtx *rc, const CBoard *b, int depth,
                            int32_t *out_value) {
    CaeResolverStats *st = rc->stats;
    if (st) {
        st->nodes++;
        if ((uint32_t)depth > st->max_depth_seen) st->max_depth_seen = (uint32_t)depth;
    }

    if (!cboard_in_check(b)) {
        /* Not in check: a terminal here is a draw (stalemate, repetition,
         * 50-move, insufficient material) — checkmate is impossible without
         * check, so no mate score can arise on this branch. */
        if (cae_resolver_is_drawn(b)) {
            if (st) st->terminal_draw++;
            *out_value = 0;
            return CAE_VALUE_OK;
        }
        if (st) st->resolved_leaves++;
        int32_t v = 0;
        int rc_status = rc->leaf_eval(rc->leaf_ctx, b, depth, &v);
        if (rc_status != CAE_VALUE_OK) return rc_status;
        /* ⚑ A MATE-BAND VALUE PASSES THROUGH; ONLY AN EVALUATION IS CLAMPED.
         * The qsearch hook can legitimately return a mate it found below this
         * node, and clamping that to +-32000 would silently demote a forced mate
         * to an ordinary good position — the exact class of defect the band
         * separation exists to prevent. A hook returning a RAW evaluation is
         * required to clamp it itself, where "this is not a mate score" is
         * known; cae_arm_static_leaf does. */
        *out_value = cae_resolver_is_mate_score(v) ? v : cae_resolver_clamp(v);
        return CAE_VALUE_OK;
    }

    /* In check. The move generator is legality-filtered, so in a check position
     * it emits exactly the evasions. */
    int moves[CBOARD_MAX_LEGAL_MOVES];
    int n = cboard_legal_move_indices(b, moves, 0);

    /* ⚑ MATE IS TESTED BEFORE THE DRAW RULES, because checkmate ends the game
     * outright and outranks a 50-move or repetition claim. cboard_is_game_over
     * happens to test the clock first; here the mated side has no move at all,
     * so deferring to that ordering would score a mate as a draw. */
    if (n == 0) {
        if (st) st->terminal_mate++;
        *out_value = cae_resolver_mated_score(depth);
        return CAE_VALUE_OK;
    }
    if (cae_resolver_is_drawn(b)) {
        if (st) st->terminal_draw++;
        *out_value = 0;
        return CAE_VALUE_OK;
    }
    if (depth >= rc->max_depth) {
        if (st) st->depth_cutoffs++;
        *out_value = 0;
        return CAE_VALUE_OK;
    }

    /* MINIMAX over the evasions, in negamax form: a child's value is from the
     * child's side-to-move POV, so it is negated on the way up. The side to
     * move picks the best evasion available to it. */
    int32_t best = INT32_MIN;
    for (int i = 0; i < n; i++) {
        CBoard child;
        memcpy(&child, b, sizeof(CBoard));
        cboard_push_index(&child, moves[i]);

        int32_t child_value = 0;
        int status = cae_resolve_node(rc, &child, depth + 1, &child_value);
        if (status != CAE_VALUE_OK) return status;

        int32_t from_our_side = -child_value;
        if (from_our_side > best) best = from_our_side;
    }
    *out_value = best;
    return CAE_VALUE_OK;
}

/* The entry point. Returns a value for `board` that is safe to use in place of
 * a static evaluation, having resolved any check first. */
static inline int cae_resolve_eval(const CaeResolverCtx *rc, const CBoard *board,
                                   int32_t *out_value) {
    if (!rc || !rc->leaf_eval) return CAE_VALUE_ERR_NOT_LOADED;
    if (rc->stats) {
        rc->stats->calls++;
        if (cboard_in_check(board)) rc->stats->calls_in_check++;
    }
    return cae_resolve_node(rc, board, 0, out_value);
}

#endif /* CAE_CHECK_RESOLVER_H */
