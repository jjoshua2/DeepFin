/*
 * _nnue_dag_store.h — the NNUE payload store over CaePositionDag, without Python.
 *
 * One structural chess position gets one CaeNnueState and, when it is not in
 * check, one static NNUE value. Both are computed exactly once and reused by
 * every later parent that reaches the same canonical position.
 *
 * This layer was split out of _nnue_dag_api.h so that the two consumers can
 * share ONE implementation of the payload arrays and — more importantly — ONE
 * implementation of the accounting:
 *
 *   _nnue_dag_api.h    the Python probe surface (dag_open / dag_intern_*)
 *   _arm_providers.h   the "nnue-qsearch-dag" arm, which uses the same store as
 *                      its evaluation substrate
 *
 * ⚑ THE ACCOUNTING IS WHY THE SPLIT IS WORTH IT, NOT THE ARRAYS. The headline
 * invariant of this store is the exact identity
 *
 *     state_inits + state_makes == node_count
 *
 * and it is only an invariant while every publication path increments it. A
 * second consumer that grew its own copy of "probe, evaluate, publish" would
 * have its own copy of that bookkeeping to get wrong, and the identity would
 * quietly become a property of one caller instead of a property of the store.
 * So both callers go through cae_nnue_dag_intern_* below, which are the only
 * functions that publish.
 *
 * ⚑⚑ SINGLE-THREADED CONSTRUCTION, ENFORCED BY THE CALLER. Nothing here locks.
 * A probe/evaluate/publish sequence that interleaves with another thread's
 * publishes both duplicates nodes (breaking the identity above — MEASURED: 21
 * accounted constructions against a node_count of 87 at 6 threads) and reads
 * &h->states[parent_id] across a cae_nnue_dag_grow_payload() that free()s that
 * very array. The Python surface holds the GIL across each call; the arm holds
 * it across a whole batch. See the ⚑⚑ block in _nnue_dag_api.h.
 */

#ifndef CAE_NNUE_DAG_STORE_H
#define CAE_NNUE_DAG_STORE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../mcts/_position_dag.h"
#include "_nnue_state.h"

typedef struct CaeNnueDagHandle {
    CaePositionDag dag;
    CaeNnueWeights *weights;

    /* CaeNnueAcc is 32-byte aligned and the AVX2 kernels use aligned loads.
     * Do not replace this storage with plain realloc(): C only promises malloc
     * alignment for fundamental types, not this explicit 32-byte alignment. */
    CaeNnueState *states;
    int32_t *values;
    uint8_t *value_valid;
    int32_t payload_cap;

    uint64_t state_inits;
    uint64_t state_makes;
    uint64_t nnue_evals;
    uint64_t node_reuses;
} CaeNnueDagHandle;

/* Release everything the handle OWNS, but not the handle itself: the two owners
 * allocate it with different allocators (PyMem_Calloc for the capsule, calloc
 * for the arm context) and each frees its own. */
static void cae_nnue_dag_store_release(CaeNnueDagHandle *h) {
    if (!h) return;
    cae_position_dag_free(&h->dag);
    if (h->weights) cae_nnue_release(h->weights);
    free(h->states);
    free(h->values);
    free(h->value_valid);
    h->weights = NULL;
    h->states = NULL;
    h->values = NULL;
    h->value_valid = NULL;
    h->payload_cap = 0;
}

static int cae_nnue_dag_grow_payload(CaeNnueDagHandle *h, int32_t need) {
    if (need <= h->payload_cap) return 0;
    int32_t new_cap = h->payload_cap > 0 ? h->payload_cap : CAE_DAG_MIN_NODE_CAP;
    while (new_cap < need) {
        if (new_cap > INT32_MAX / 2) return -1;
        new_cap *= 2;
    }

    CaeNnueState *new_states = NULL;
    size_t state_bytes = (size_t)new_cap * sizeof(*new_states);
    if (posix_memalign((void **)&new_states, 32, state_bytes) != 0) return -1;
    int32_t *new_values = (int32_t *)malloc((size_t)new_cap * sizeof(*new_values));
    uint8_t *new_valid = (uint8_t *)calloc((size_t)new_cap, sizeof(*new_valid));
    if (!new_values || !new_valid) {
        free(new_states);
        free(new_values);
        free(new_valid);
        return -1;
    }
    if (h->payload_cap > 0) {
        memcpy(new_states, h->states, (size_t)h->payload_cap * sizeof(*new_states));
        memcpy(new_values, h->values, (size_t)h->payload_cap * sizeof(*new_values));
        memcpy(new_valid, h->value_valid, (size_t)h->payload_cap * sizeof(*new_valid));
    }
    free(h->states);
    free(h->values);
    free(h->value_valid);
    h->states = new_states;
    h->values = new_values;
    h->value_valid = new_valid;
    h->payload_cap = new_cap;
    return 0;
}

/* Build the store's graph and payload arrays. Returns 0, or -1 having left
 * nothing allocated. */
static int cae_nnue_dag_store_init(
    CaeNnueDagHandle *h, CaeNnueWeights *weights, int32_t initial_nodes)
{
    memset(h, 0, sizeof(*h));
    h->weights = cae_nnue_retain(weights);
    if (!h->weights) return -1;
    if (cae_position_dag_init(&h->dag, initial_nodes) != 0
        || cae_nnue_dag_grow_payload(h, initial_nodes) != 0) {
        cae_nnue_dag_store_release(h);
        return -1;
    }
    return 0;
}

static void cae_nnue_dag_store_reset(CaeNnueDagHandle *h) {
    cae_position_dag_reset(&h->dag);
    if (h->payload_cap > 0)
        memset(h->value_valid, 0, (size_t)h->payload_cap * sizeof(*h->value_valid));
    h->state_inits = h->state_makes = h->nnue_evals = h->node_reuses = 0;
}

static int64_t cae_nnue_dag_payload_bytes(const CaeNnueDagHandle *h) {
    return (int64_t)h->payload_cap * (int64_t)(
        sizeof(CaeNnueState) + sizeof(int32_t) + sizeof(uint8_t));
}

/* Finish publishing a NEW structural node only after its NNUE payload is valid.
 * The state/value are computed before this helper is called; an evaluation
 * failure therefore cannot leave a half-initialised canonical node behind. */
static int32_t cae_nnue_dag_publish_new(
    CaeNnueDagHandle *h,
    const CaeDagPosition *position,
    const CaeNnueState *state,
    int32_t value,
    int value_valid)
{
    if (cae_nnue_dag_grow_payload(h, h->dag.node_count + 1) != 0)
        return CAE_DAG_NO_NODE;
    int32_t node_id = cae_position_dag_insert_position(&h->dag, position);
    if (node_id == CAE_DAG_NO_NODE) return CAE_DAG_NO_NODE;
    h->states[node_id] = *state;
    h->values[node_id] = value;
    h->value_valid[node_id] = (uint8_t)(value_valid ? 1 : 0);
    return node_id;
}

/* ================================================================
 * The two publication paths
 * ================================================================
 *
 * Both return a CaeValueStatus. CAE_VALUE_OK writes *out_node and *out_created;
 * anything else writes neither and has published nothing.
 *
 * ⚑ NEITHER OF THESE VALIDATES ITS INPUT, and that is the layering rather than
 * an omission. "Is this action legal at this parent and does it really produce
 * that child" is a question about an UNTRUSTED caller, so it belongs to the
 * Python surface, which asks it (cae_position_dag_edge_matches_board) before
 * calling in here. The search path has just generated and pushed the move
 * itself, so re-deriving the parent's legal-move list per node would cost a full
 * move generation to re-establish something it already knows.
 */

/* CAE_DAG_NO_NODE-returning allocation failures need a status of their own; the
 * seam has no out-of-memory code, and BAD_POS would name the wrong cause. Every
 * caller turns this into its own memory error. */
#define CAE_NNUE_DAG_ERR_NO_MEMORY (-100)
/* Single-threaded construction cannot produce a parent whose action already maps
 * to a different child, so this is the signature of concurrent use. */
#define CAE_NNUE_DAG_ERR_LINK      (-101)

/* Intern a position with NO parent edge, refreshing its state from the board.
 * This is what a root — or a search node the caller reached without a DAG parent
 * to derive from — costs on a miss; on a hit it costs a probe and nothing else. */
static int cae_nnue_dag_intern_position(
    CaeNnueDagHandle *h, const CBoard *board, int32_t *out_node, int *out_created)
{
    CaeDagPosition position;
    cae_dag_position_from_cboard(board, &position);
    int32_t existing = cae_position_dag_find_position(&h->dag, &position);
    if (existing != CAE_DAG_NO_NODE) {
        h->node_reuses++;
        *out_node = existing;
        *out_created = 0;
        return CAE_VALUE_OK;
    }

    CaeNnueState state;
    int status = cae_nnue_state_init(h->weights, board, &state);
    if (status != CAE_VALUE_OK) return status;

    /* An in-check structural position is a valid node and still owns an
     * accumulator, but has no static value: the NNUE evaluation is undefined
     * there, and inventing a sentinel would be a number that looks like one. */
    int value_valid = !state.pos.in_check;
    int32_t value = 0;
    if (value_valid) {
        status = cae_nnue_state_evaluate(h->weights, &state, &value);
        if (status != CAE_VALUE_OK) return status;
    }

    int32_t node_id = cae_nnue_dag_publish_new(h, &position, &state, value, value_valid);
    if (node_id == CAE_DAG_NO_NODE) return CAE_NNUE_DAG_ERR_NO_MEMORY;
    h->state_inits++;
    if (value_valid) h->nnue_evals++;
    *out_node = node_id;
    *out_created = 1;
    return CAE_VALUE_OK;
}

/* Intern parent_id --action--> child_board, deriving the child's state from the
 * parent's incrementally. A transposition hit adds the edge and performs no
 * make() and no evaluate(). */
static int cae_nnue_dag_intern_child(
    CaeNnueDagHandle *h, int32_t parent_id, int action, const CBoard *child_board,
    int32_t *out_node, int *out_created)
{
    CaeDagPosition child_pos;
    cae_dag_position_from_cboard(child_board, &child_pos);

    int32_t existing = cae_position_dag_find_position(&h->dag, &child_pos);
    if (existing != CAE_DAG_NO_NODE) {
        int link_rc = cae_position_dag_link(&h->dag, parent_id, action, existing);
        if (link_rc < 0)
            return link_rc == -1 ? CAE_NNUE_DAG_ERR_NO_MEMORY : CAE_NNUE_DAG_ERR_LINK;
        h->node_reuses++;
        *out_node = existing;
        *out_created = 0;
        return CAE_VALUE_OK;
    }

    /* ⚑ &h->states[parent_id] IS READ BEFORE ANY PUBLISH, and it has to be:
     * cae_nnue_dag_publish_new() can grow the payload arrays, which free()s the
     * very array this pointer indexes. Reading it into `state` here is what
     * keeps the parent's accumulator alive across the growth. A caller that
     * holds a CaeNnueState* across a child's publish has the same bug — which
     * is why the arm carries node IDS down its recursion, never pointers. */
    CaeNnueState state;
    int status = cae_nnue_state_make(
        h->weights, &h->states[parent_id], child_board, &state);
    if (status != CAE_VALUE_OK) return status;

    int value_valid = !state.pos.in_check;
    int32_t value = 0;
    if (value_valid) {
        status = cae_nnue_state_evaluate(h->weights, &state, &value);
        if (status != CAE_VALUE_OK) return status;
    }

    /* ⚑ RESERVE THE EDGE BEFORE PUBLISHING THE NODE. The link's only remaining
     * failure is the edge-array growth allocation, and taking it here — while
     * the DAG is still untouched — is what keeps every failure path clean.
     * Publishing first and discovering the allocation failure afterwards left a
     * node in the canonical table that no edge reached: a retry found it and
     * reported reuse, its NNUE work was never accounted, and
     * state_inits + state_makes == node_count was broken permanently rather
     * than transiently. Reserve-first means the caller either gets a complete
     * node+edge or an untouched DAG, so the identity holds on EVERY path,
     * MemoryError included.
     *
     * ⚑ IT LIVES HERE RATHER THAN IN THE PYTHON SURFACE, and that is the whole
     * reason this function exists. The fix landed on _nnue_dag_api.h while the
     * search consumer was being written; leaving it there would have given the
     * Python path a clean OOM and the C search the old orphan-node behaviour,
     * from one store, with the identity that is supposed to detect it holding
     * for one caller and not the other. */
    if (cae_position_dag_reserve_edge(&h->dag) != 0)
        return CAE_NNUE_DAG_ERR_NO_MEMORY;

    int32_t node_id = cae_nnue_dag_publish_new(h, &child_pos, &state, value, value_valid);
    if (node_id == CAE_DAG_NO_NODE) return CAE_NNUE_DAG_ERR_NO_MEMORY;
    int link_rc = cae_position_dag_link(&h->dag, parent_id, action, node_id);
    if (link_rc != 1) {
        /* Unreachable by construction now: the edge is reserved, parent, child
         * and action are all in range, and no edge for this action exists (the
         * caller checked, or the probe above found no node), so link() can
         * return neither -1 nor 0 nor -2. -2 in particular would mean two
         * constructions interleaved, which the retained GIL prevents. Kept as a
         * loud failure rather than an assert because it is the shape a future
         * concurrent consumer would hit first.
         *
         * ⚑ AND IT DELIBERATELY DOES NOT ROLL THE NODE BACK. Review suggested
         * un-publishing here so the identity would survive. Declined, and the
         * reasoning is the point: single-threaded this branch cannot be taken,
         * so any rollback code would be untestable and unexercised — and the
         * ONLY way to get here is the concurrent use this store forbids. In
         * that case a loud RuntimeError plus a broken
         * state_inits + state_makes == node_count is exactly the alarm the
         * identity exists to be. Quietly repairing the graph would convert the
         * one observable symptom of a real misuse into silence. */
        return link_rc == -1 ? CAE_NNUE_DAG_ERR_NO_MEMORY : CAE_NNUE_DAG_ERR_LINK;
    }
    h->state_makes++;
    if (value_valid) h->nnue_evals++;
    *out_node = node_id;
    *out_created = 1;
    return CAE_VALUE_OK;
}

#endif /* CAE_NNUE_DAG_STORE_H */
