/*
 * _fastq_search.h — FastQ-4+, a bounded tactical verifier on the position DAG.
 *
 * Implements docs/fastq_design.md. One narrow question per leaf: "is the static
 * NNUE value tactically unstable here, and if so what is the corrected value?"
 * — in a handful of evaluations rather than the qsearch arm's ~72 average.
 *
 * ⚑⚑ THIS IS A DIFFERENT SEARCH, NOT A FOURTH CaeQsearchSubstrate, AND THE
 * DISTINCTION IS LOAD-BEARING. `CaeQsearchSubstrate` is documented in
 * _arm_providers.h as "where a node's stand-pat NUMBER comes from. Nothing else
 * about the search varies with it", and tests/test_nnue_incremental.py asserts
 * each qsearch wrapper selects a distinct value of it. FastQ differs in move
 * policy, pruning, recursion shape and budget — everything EXCEPT where the
 * number comes from, which is the same DAG the qsearch-dag arm uses. Adding
 * CAE_QSUB_FASTQ would have made that enum mean two incompatible things and
 * forced every qsearch branch to ask which search it was in. So FastQ owns its
 * own eval entry point and the substrate enum is left alone at three.
 *
 * ⚑ EVASIONS ARE OWNED HERE (§3.2). The generic cae_resolve_node would resolve
 * checks for us, but it has its own depth limit and its own idea of when to stop
 * — and a node budget that some nodes escape is not a budget. The scoring
 * CONSTANTS and the terminal helper are reused from the resolver so a mate score
 * means the same thing in both.
 */

#ifndef _CAE_FASTQ_SEARCH_H
#define _CAE_FASTQ_SEARCH_H

#include <stdint.h>
#include <string.h>

/* Defaults from the §6 knob table. Every one is snapshotted into the context at
 * init() and read from there by the search — never from these globals — so
 * "what did this call actually run with" is answerable from the context alone. */
#define CAE_FASTQ_DEFAULT_MAX_QPLY        4
#define CAE_FASTQ_DEFAULT_NODE_CAP        32
#define CAE_FASTQ_DEFAULT_DELTA_MARGIN    200
#define CAE_FASTQ_DEFAULT_RECAPTURE_EXEMPT 1

/* Deepest DFS path FastQ can hold. max_qply is bounded well below this; the
 * array is what the cycle guard scans, so it is sized for the worst legal
 * configuration rather than the default. */
#define CAE_FASTQ_MAX_PATH 64

/* The initial window bound. Spelled from CAE_RESOLVER_MATE_BASE rather than
 * borrowed from the qsearch arm's CAE_QSEARCH_INF, so this header does not
 * depend on where it sits in _arm_providers.h's include order — but it is deliberately the SAME value,
 * because FastQ and the qsearch arm score mates through the same
 * cae_resolver_mated_score and the §8 harness compares their outputs directly. */
#define CAE_FASTQ_INF (CAE_RESOLVER_MATE_BASE * 2)

typedef struct CaeFastqConfig {
    int max_qply;
    int node_cap;
    int delta_margin;
    int see_recapture_exempt;
} CaeFastqConfig;

typedef struct CaeFastqStats {
    uint64_t calls;
    uint64_t nodes;                  /* recursion nodes entered */
    uint64_t evasion_nodes;          /* of which, in check */
    uint64_t nodes_created;          /* new canonical nodes interned */
    uint64_t nodes_created_in_check; /* of which had no static value */
    uint64_t nnue_evals;             /* §7 identity, see cae_fastq_stats_ok */
    uint64_t hits_within_call;
    uint64_t hits_cross_call;
    uint64_t quiet_certificates;     /* certificates computed from scratch */
    uint64_t quiet_certificate_hits; /* certificates read back from a node */
    uint64_t quiet_returns;          /* nodes that stood pat on the certificate */
    uint64_t see_prunes;
    uint64_t delta_prunes;
    uint64_t recapture_exemptions;   /* SEE-negative captures kept anyway */
    /* ⚑ SPLIT, BECAUSE ONE "beta_cutoffs" MEANT TWO DIFFERENT EVENTS. A
     * stand-pat cutoff is a node that never generated a move; a move cutoff is a
     * node that searched at least one. Summing them under one name made the
     * counter unable to answer the only question it gets asked — how often did
     * the move loop actually pay off — and a counter that cannot answer its own
     * question is the same defect as one that reads zero. */
    uint64_t stand_pat_cutoffs;
    uint64_t move_cutoffs;
    uint64_t budget_trips;
    uint64_t path_ceilings;
    uint64_t cycle_draws;
    uint64_t terminal_mate;
    uint64_t terminal_draw;
    uint32_t max_ply_seen;
} CaeFastqStats;

typedef struct CaeFastqCtx {
    CaeNnueDagHandle *store;
    CaeFastqStats *stats;
    CaeFastqConfig cfg;

    int32_t dag_watermark;   /* node_count when this call began, for the split */
    uint64_t nodes_used;     /* against cfg.node_cap */

    int32_t path[CAE_FASTQ_MAX_PATH];
    int path_len;

    /* The square the parent captured on, or -1. §3.4's recapture exemption is
     * about THIS square only, so it is per-visit state rather than a node fact. */
    int recapture_square;
} CaeFastqCtx;

static inline void cae_fastq_stats_reset(CaeFastqStats *s) { memset(s, 0, sizeof(*s)); }

/* ================================================================
 * The quiet certificate, node-cached (§3.1/§4.1)
 *
 * The PREDICATE itself is in _fastq_certificate.h, which by construction cannot
 * see a window; only the caching lives here, where the context does.
 * ================================================================ */

/* Certificate for a node, computing and storing it on first request.
 *
 * ⚑ LAZY, AND THAT IS NOT A WEAKENING OF "STORED ONCE PER NODE" (§4.1). The
 * certificate costs a full legal-move generation plus a SEE per capture, and a
 * node that beta-cuts before it is asked never needs one. Computed at most once
 * per node either way; the laziness only decides whether it is computed at all.
 */
static uint8_t cae_fastq_node_certificate(
    CaeFastqCtx *q, int32_t node_id, const CBoard *board)
{
    uint8_t bits = q->store->quiet_bits[node_id];
    if (bits & CAE_DAG_CERT_COMPUTED) {
        q->stats->quiet_certificate_hits++;
        return bits;
    }
    bits = cae_fastq_certificate(board);
    q->store->quiet_bits[node_id] = bits;
    q->stats->quiet_certificates++;
    return bits;
}

/* ================================================================
 * Move policy and ordering (§3.2, §5)
 * ================================================================ */

typedef struct CaeFastqMove {
    int action;      /* policy index */
    int to_sq;
    int32_t see;
    int32_t victim;   /* for delta pruning, and the MVV half of the tiebreak */
    int32_t attacker; /* the LVA half */
} CaeFastqMove;

/* §5's ordering: SEE descending, MVV-LVA as the tiebreak WITHIN an equal-SEE
 * group. Returns >0 when `a` should come first. */
static inline int cae_fastq_move_before(const CaeFastqMove *a, const CaeFastqMove *b)
{
    if (a->see != b->see) return a->see > b->see;
    if (a->victim != b->victim) return a->victim > b->victim;
    return a->attacker < b->attacker;
}

static inline int32_t cae_fastq_victim_value(const CBoard *b, const PolicyMove *pm)
{
    int32_t v = 0;
    const int victim_pt = piece_type_at(b, pm->to_sq);
    if (victim_pt >= 0) v = CAE_SEE_VALUE[victim_pt];
    else if (cae_fastq_is_en_passant(b, pm)) v = CAE_SEE_VALUE[PAWN];
    const int promo = cae_fastq_promotion_of(b, pm);
    if (promo != 0) {
        int promo_pt = QUEEN;
        switch (promo) {
            case 2: promo_pt = KNIGHT; break;
            case 3: promo_pt = BISHOP; break;
            case 4: promo_pt = ROOK;   break;
            default: promo_pt = QUEEN; break;
        }
        v += CAE_SEE_VALUE[promo_pt] - CAE_SEE_VALUE[PAWN];
    }
    return v;
}

/* Captures and promotions only — NEVER a quiet check (§3.2). Returns the count,
 * sorted best-SEE-first; ties keep generation order, which is deterministic. */
static int cae_fastq_tactical_moves(
    const CBoard *b, const int *moves, int n, CaeFastqMove *out)
{
    int count = 0;
    for (int i = 0; i < n; i++) {
        const PolicyMove pm = POLICY_LUT[b->turn][moves[i]];
        const int promo = cae_fastq_promotion_of(b, &pm);
        if (promo == 0 && !cae_fastq_is_capture(b, &pm)) continue;
        out[count].action = moves[i];
        out[count].to_sq = pm.to_sq;
        out[count].see = cae_see_capture(b, pm.from_sq, pm.to_sq, promo);
        out[count].victim = cae_fastq_victim_value(b, &pm);
        {
            const int att_pt = piece_type_at(b, pm.from_sq);
            out[count].attacker = att_pt >= 0 ? CAE_SEE_VALUE[att_pt] : 0;
        }
        count++;
    }
    /* Insertion sort, descending by SEE. n is tiny (tactical moves at a real
     * node are single digits) and a stable sort keeps the order reproducible. */
    for (int i = 1; i < count; i++) {
        const CaeFastqMove key = out[i];
        int j = i - 1;
        while (j >= 0 && cae_fastq_move_before(&key, &out[j])) {
            out[j + 1] = out[j];
            j--;
        }
        out[j + 1] = key;
    }
    return count;
}

/* ================================================================
 * The recursion (§3.3)
 * ================================================================ */

/* §3.4's node budget: a TRIPWIRE, not a tuned knob.
 *
 * ⚑⚑ CHECKED BEFORE THE CHILD IS INTERNED, WHICH IS THE ONLY PLACE IT BOUNDS
 * ANYTHING. The first version of this sat at the top of cae_fastq_node, matching
 * the shape of §3.3's pseudocode — and it did not bound the cost it exists to
 * bound. Interning happens in the PARENT, before the recursive call, and
 * interning is what performs the NNUE evaluation; a node turned away on entry
 * had already been created and evaluated. Measured on the 467-row corpus at
 * node_cap 32: a single call reached 65 NNUE evaluations. The budget was
 * accepted, counted, reported — and silently ignored by the cost it names,
 * which is this repo's signature defect exactly.
 *
 * It counts EVERY expansion, evasions included: a budget some nodes escape is
 * not a budget, which is what §8 mutant 5 exists to prove. The root is free — it
 * is the position being asked about, not work the search chose to do.
 *
 * On exhaustion the caller BREAKS its move loop and returns its current bound,
 * which for a non-check node is at least its stand-pat. */
static inline int cae_fastq_budget_spent(CaeFastqCtx *q) {
    if (q->cfg.node_cap <= 0) return 0;
    if (q->nodes_used < (uint64_t)q->cfg.node_cap) return 0;
    q->stats->budget_trips++;
    return 1;
}

static int cae_fastq_node(
    CaeFastqCtx *q, const CBoard *b, int32_t node_id, int ply,
    int32_t alpha, int32_t beta, int32_t *out_value);

/* Intern `child_board` under `parent_id`, classifying the probe as a creation, a
 * within-call hit or a cross-call hit. The watermark split is exact because node
 * ids are dense, monotonic and never recycled without a reset. */
static int cae_fastq_intern_child(
    CaeFastqCtx *q, int32_t parent_id, int action, const CBoard *child_board,
    int32_t *out_node)
{
    int created = 0;
    const int status = cae_nnue_dag_intern_child(
        q->store, parent_id, action, child_board, out_node, &created);
    if (status != CAE_VALUE_OK) return status;
    if (created) {
        q->stats->nodes_created++;
        if (cboard_in_check(child_board)) q->stats->nodes_created_in_check++;
        else q->stats->nnue_evals++;
    } else if (*out_node < q->dag_watermark) {
        q->stats->hits_cross_call++;
    } else {
        q->stats->hits_within_call++;
    }
    return CAE_VALUE_OK;
}

/* One child: push the move, intern, recurse negamax. */
static int cae_fastq_child_value(
    CaeFastqCtx *q, const CBoard *b, int32_t node_id, int action, int to_sq,
    int ply, int32_t alpha, int32_t beta, int32_t *out_value)
{
    CBoard child;
    memcpy(&child, b, sizeof(CBoard));
    cboard_push_index(&child, action);

    int32_t child_id = CAE_DAG_NO_NODE;
    const int status = cae_fastq_intern_child(q, node_id, action, &child, &child_id);
    if (status != CAE_VALUE_OK) return status;

    const int saved_square = q->recapture_square;
    q->recapture_square = to_sq;
    int32_t child_value = 0;
    const int rc = cae_fastq_node(q, &child, child_id, ply + 1, -beta, -alpha, &child_value);
    q->recapture_square = saved_square;
    if (rc != CAE_VALUE_OK) return rc;

    *out_value = -child_value;
    return CAE_VALUE_OK;
}

static int cae_fastq_node(
    CaeFastqCtx *q, const CBoard *b, int32_t node_id, int ply,
    int32_t alpha, int32_t beta, int32_t *out_value)
{
    /* §4.3 cycle guard. Structural identity admits back-edges, so a node already
     * on the DFS path is a repetition within this search and adjudicates as a
     * draw. The path is a small array because it is at most max_qply deep. */
    for (int i = 0; i < q->path_len; i++) {
        if (q->path[i] == node_id) {
            q->stats->cycle_draws++;
            *out_value = 0;
            return CAE_VALUE_OK;
        }
    }

    q->stats->nodes++;
    if ((uint32_t)ply > q->stats->max_ply_seen) q->stats->max_ply_seen = (uint32_t)ply;

    const int in_check = cboard_in_check(b);
    if (in_check) q->stats->evasion_nodes++;

    /* ⚑⚑ THE PATH ARRAY IS THE HARD RECURSION CEILING, AND IT IS COUNTED. The
     * qply limit does not bound evasion recursion (§3.2 resolves checks exactly,
     * however deep the forcing sequence runs), so with the budget disabled a
     * check storm is bounded only by this. Firing it is observable rather than
     * silent, because an invisible clamp is indistinguishable from a search that
     * simply chose to stop. */
    if (q->path_len >= CAE_FASTQ_MAX_PATH) {
        q->stats->path_ceilings++;
        /* Same fallback rule as the budget trip below, for the same reason: in
         * check there is no stand-pat, and `alpha` here can be -CAE_FASTQ_INF,
         * which the parent would negate into a supermate. See the ⚑⚑ block on
         * the evasion loop's `if (!searched)`. */
        *out_value = (!in_check && q->store->value_valid[node_id])
                         ? cae_resolver_clamp(q->store->values[node_id])
                         : cae_resolver_clamp(beta);
        return CAE_VALUE_OK;
    }
    q->path[q->path_len++] = node_id;

    int status = CAE_VALUE_OK;
    int32_t best;

    int moves[CAE_FASTQ_MAX_MOVES];
    const int n_moves = cboard_legal_move_indices(b, moves, 0);

    if (in_check) {
        /* §3.2: ALL legal evasions, no stand-pat. Owned here rather than handed
         * to cae_resolve_node so that every node passes the budget above. */
        if (n_moves == 0) {
            q->stats->terminal_mate++;
            *out_value = cae_resolver_mated_score(ply);
            q->path_len--;
            return CAE_VALUE_OK;
        }
        best = -CAE_FASTQ_INF;
        int searched = 0;
        for (int i = 0; i < n_moves; i++) {
            if (cae_fastq_budget_spent(q)) break;
            q->nodes_used++;
            const PolicyMove pm = POLICY_LUT[b->turn][moves[i]];
            int32_t value = 0;
            status = cae_fastq_child_value(
                q, b, node_id, moves[i], pm.to_sq, ply, alpha, beta, &value);
            if (status != CAE_VALUE_OK) goto done;
            searched = 1;
            if (value > best) best = value;
            if (best > alpha) alpha = best;
            if (alpha >= beta) { q->stats->move_cutoffs++; break; }
        }
        /* ⚑⚑ THE BUDGET CAN TRIP ON ITERATION 0, AND -CAE_FASTQ_INF IS NOT A
         * VALUE. An in-check node has no stand-pat to fall back on — the NNUE
         * evaluation is undefined in check — so `best` is seeded at -INF and, if
         * the very first evasion is refused by the budget, that seed is what
         * leaves this function. -200000 negates to +200000 at the parent, which
         * is TWICE CAE_RESOLVER_MATE_BASE: a "mate" score better than mate in 0,
         * from a node the search declined to look at. The §8 harness classifies
         * anything past the eval clamp as a mate, so it would have been reported
         * as FastQ finding a forced win.
         *
         * ⚑⚑ "RETURN alpha, LIKE THE PATH-CEILING BRANCH ABOVE" REPRODUCES THE
         * BUG, WHICH IS WHY THIS RETURNS SOMETHING ELSE. Trace the root call:
         * cae_arm_fastq_eval enters with beta = +CAE_FASTQ_INF, the root passes
         * `beta` down unchanged, and cae_fastq_child_value negates it — so a
         * first-generation child's alpha IS -CAE_FASTQ_INF. Returning alpha
         * there returns -200000 and the parent negates it to +200000: the exact
         * supermate this block exists to prevent. The path-ceiling branch had the
         * same defect and is fixed with it.
         *
         * TWO THINGS ARE NEEDED, and only together:
         *
         *   1. `beta`, not `alpha`. Returning alpha claims a fail-LOW, which the
         *      parent reads as a fail-HIGH on the move that reached here — a
         *      budget trip would make a checking move look GOOD. Returning beta
         *      claims a fail-high here, which the parent reads as "no
         *      improvement", so an unsearched move cannot become the best move.
         *      Neither is sound — nothing was searched — but only one of them can
         *      promote a move the search declined to look at.
         *
         *   2. The clamp. beta is -alpha_parent, and an in-check parent's alpha
         *      starts at -CAE_FASTQ_INF too, so beta can itself be ±200000 and
         *      the escape just moves up one level. Clamping is what actually
         *      enforces the invariant: A NODE THE SEARCH DECLINED TO LOOK AT
         *      NEVER EMITS A MATE-MAGNITUDE SCORE. Real mates are unaffected —
         *      they come from the n_moves == 0 branch above, which is untouched.
         *
         * The residual distortion is bounded by the eval clamp and COUNTED:
         * budget_trips is what turns it from a silent wrong number into §3.4's
         * "nonzero trip rate outside crafted fixtures is a finding". */
        if (!searched) best = cae_resolver_clamp(beta);
        *out_value = best;
        goto done;
    }

    /* A drawn position adjudicates before its static value is consulted. */
    if (cae_resolver_is_drawn(b)) {
        q->stats->terminal_draw++;
        *out_value = 0;
        goto done;
    }

    if (!q->store->value_valid[node_id]) {
        /* Not in check, so the store must hold a static value; if it does not,
         * the node was published by something that disagrees with this search
         * about what a node is. Fail loudly rather than inventing a number. */
        status = CAE_VALUE_ERR_BAD_POS;
        goto done;
    }
    /* ⚑ CLAMPED ON READ, THE SAME WAY cae_qsearch_node CLAMPS ITS STAND-PAT.
     * The DAG stores the RAW NNUE value — that is the store's contract and the
     * qsearch-dag arm depends on it — so the clamp belongs at every reader, and
     * a reader that skips it is a reader whose search values can leave the
     * evaluation range. Measured: with a high-magnitude synthetic pack, 147 of
     * 444 non-check corpus rows evaluate past CAE_RESOLVER_EVAL_CLAMP and reach
     * 89044, which is inside the mate band the §8 harness classifies on. Not
     * reachable with the production net (max |v| = 4546), which is exactly why
     * it needs a test rather than an argument. */
    const int32_t stand_pat = cae_resolver_clamp(q->store->values[node_id]);

    if (stand_pat >= beta) {
        q->stats->stand_pat_cutoffs++;
        *out_value = stand_pat;
        goto done;
    }
    if (alpha < stand_pat) alpha = stand_pat;
    best = stand_pat;

    if (ply >= q->cfg.max_qply) {
        *out_value = stand_pat;
        goto done;
    }

    /* The certificate. A quiet node has nothing to search by construction. */
    if (cae_fastq_is_quiet(cae_fastq_node_certificate(q, node_id, b))) {
        q->stats->quiet_returns++;
        *out_value = stand_pat;
        goto done;
    }

    {
        CaeFastqMove tactical[CAE_FASTQ_MAX_MOVES];
        const int n_tactical = cae_fastq_tactical_moves(b, moves, n_moves, tactical);
        for (int i = 0; i < n_tactical; i++) {
            const CaeFastqMove *mv = &tactical[i];

            /* §3.4 SEE gate, with the recapture-square exemption so a forced
             * recapture is never blinded. */
            if (mv->see < 0) {
                if (q->cfg.see_recapture_exempt && mv->to_sq == q->recapture_square) {
                    q->stats->recapture_exemptions++;
                } else {
                    q->stats->see_prunes++;
                    continue;
                }
            }

            /* §3.4 delta pruning. ⚑ PER VISIT, NEVER STORED: it reads `alpha`,
             * so folding it into the certificate would let the first caller's
             * window decide the answer for every later one. */
            if (stand_pat + mv->victim + q->cfg.delta_margin <= alpha) {
                q->stats->delta_prunes++;
                continue;
            }

            /* After the prunes: a move that was never searched cost nothing,
             * so charging it to the budget would make the cap depend on how
             * many moves happened to be pruned. */
            if (cae_fastq_budget_spent(q)) break;
            q->nodes_used++;

            int32_t value = 0;
            status = cae_fastq_child_value(
                q, b, node_id, mv->action, mv->to_sq, ply, alpha, beta, &value);
            if (status != CAE_VALUE_OK) goto done;
            if (value > best) best = value;
            if (best > alpha) alpha = best;
            if (alpha >= beta) { q->stats->move_cutoffs++; break; }
        }
        *out_value = best;
    }

done:
    q->path_len--;
    return status;
}

/* §7's counter identity, as a function so the C and the tests agree on it.
 *
 * ⚑ THE SPEC SAYS "NNUE evaluations must equal nodes created". That is true only
 * where every created node has a static value, and an in-check node deliberately
 * has none — cae_nnue_dag_intern_position stores value_valid = 0 there because
 * the NNUE evaluation is undefined in check. So the exact identity carries the
 * in-check term, and stating it that way is what lets it be ASSERTED rather than
 * approximately believed. */
static inline int cae_fastq_stats_ok(const CaeFastqStats *s)
{
    return s->nnue_evals + s->nodes_created_in_check == s->nodes_created;
}

#endif /* _CAE_FASTQ_SEARCH_H */
