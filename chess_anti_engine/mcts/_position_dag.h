/*
 * _position_dag.h — canonical structural chess-position DAG.
 *
 * This is deliberately NOT an MCTS tree and carries no visits, Q, priors,
 * virtual loss, parent pointer, or search-specific solved value.  One structural
 * chess position is one node and any number of parent edges may point at it.
 * Search algorithms layer their own path/edge state on top.
 *
 * Node identity is the current position state that determines legal moves:
 * pieces, side to move, castling rights, and an exercisable en-passant right.
 * Halfmove/repetition/history state is intentionally NOT part of a node.  Those
 * are path properties: they can change draw/terminal semantics and DeepFin's
 * history-sensitive neural input without changing the structural graph.  A
 * future Gumbel consumer must therefore keep that context in its search overlay;
 * it must not mistake a structural-node hit for permission to reuse a
 * history-sensitive value.
 *
 * The current cboard_transposition_key() is the fast hash for exactly this
 * structural identity.  It is only a hash: every hit is checked against the
 * canonical position fields, and the table uses open addressing so a 64-bit
 * collision cannot merge two positions or evict the canonical node.
 */

#ifndef CAE_POSITION_DAG_H
#define CAE_POSITION_DAG_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../encoding/_cboard_impl.h"

#define CAE_DAG_NO_NODE (-1)
#define CAE_DAG_MIN_NODE_CAP 16
#define CAE_DAG_MIN_EDGE_CAP 32
#define CAE_DAG_MIN_HT_CAP 64

typedef struct CaeDagPosition {
    uint64_t key;
    uint64_t bb[6];
    uint64_t occ[2];
    int8_t turn;
    uint8_t castling;
    int8_t ep_square;  /* canonical: -1 when no pseudo-legal EP capture exists */
} CaeDagPosition;

typedef struct CaePositionDag {
    CaeDagPosition *positions;
    int32_t *first_edge;
    int32_t *out_degree;
    uint8_t *expanded;
    int32_t node_count;
    int32_t node_cap;

    int32_t *edge_action;
    int32_t *edge_child;
    int32_t *edge_next;
    int32_t edge_count;
    int32_t edge_cap;

    /* Open-addressed table: -1 = empty.  No deletion/tombstones are needed;
     * reset() clears the whole table and rerooting keeps nodes alive. */
    int32_t *hash_table;
    int32_t ht_cap;
    int32_t ht_mask;

    int32_t root_id;

    /* Operation counters are part of the abstraction: a consumer claiming DAG
     * reuse has to be able to show that canonical hits actually occurred. */
    uint64_t probes;
    uint64_t hits;
    uint64_t inserts;
    uint64_t collision_steps;
    uint64_t edge_reuses;
} CaePositionDag;

static inline int32_t cae_dag_next_pow2(int32_t x) {
    int32_t v = 1;
    while (v < x && v > 0 && v <= (INT32_MAX / 2)) v <<= 1;
    return v > 0 ? v : x;
}

static inline void cae_dag_position_from_cboard(const CBoard *b, CaeDagPosition *out) {
    out->key = cboard_transposition_key(b);
    memcpy(out->bb, b->bb, sizeof(out->bb));
    memcpy(out->occ, b->occ, sizeof(out->occ));
    out->turn = b->turn;
    out->castling = b->castling;
    out->ep_square = cboard_ep_capture_available(b) ? b->ep_square : -1;
}

static inline int cae_dag_position_equal(
    const CaeDagPosition *a, const CaeDagPosition *b)
{
    return a->key == b->key
        && a->turn == b->turn
        && a->castling == b->castling
        && a->ep_square == b->ep_square
        && memcmp(a->bb, b->bb, sizeof(a->bb)) == 0
        && memcmp(a->occ, b->occ, sizeof(a->occ)) == 0;
}

static inline void cae_dag_position_to_cboard(const CaeDagPosition *p, CBoard *out) {
    /* A structural node intentionally has no history/draw context.  This shallow
     * CBoard exists only for validating an action's structural transition. */
    memset(out, 0, sizeof(*out));
    memcpy(out->bb, p->bb, sizeof(p->bb));
    memcpy(out->occ, p->occ, sizeof(p->occ));
    out->turn = p->turn;
    out->castling = p->castling;
    out->ep_square = p->ep_square;
    out->hash = cboard_compute_hash(out);
}

static void cae_position_dag_free(CaePositionDag *d) {
    if (!d) return;
    free(d->positions);
    free(d->first_edge);
    free(d->out_degree);
    free(d->expanded);
    free(d->edge_action);
    free(d->edge_child);
    free(d->edge_next);
    free(d->hash_table);
    memset(d, 0, sizeof(*d));
    d->root_id = CAE_DAG_NO_NODE;
}

static int cae_position_dag_init(CaePositionDag *d, int32_t initial_nodes) {
    memset(d, 0, sizeof(*d));
    d->root_id = CAE_DAG_NO_NODE;
    if (initial_nodes < CAE_DAG_MIN_NODE_CAP) initial_nodes = CAE_DAG_MIN_NODE_CAP;
    d->node_cap = initial_nodes;
    d->edge_cap = initial_nodes * 2;
    if (d->edge_cap < CAE_DAG_MIN_EDGE_CAP) d->edge_cap = CAE_DAG_MIN_EDGE_CAP;

    int32_t want_ht = initial_nodes * 2;
    if (want_ht < CAE_DAG_MIN_HT_CAP) want_ht = CAE_DAG_MIN_HT_CAP;
    d->ht_cap = cae_dag_next_pow2(want_ht);
    d->ht_mask = d->ht_cap - 1;

    d->positions = (CaeDagPosition *)calloc((size_t)d->node_cap, sizeof(*d->positions));
    d->first_edge = (int32_t *)malloc((size_t)d->node_cap * sizeof(*d->first_edge));
    d->out_degree = (int32_t *)calloc((size_t)d->node_cap, sizeof(*d->out_degree));
    d->expanded = (uint8_t *)calloc((size_t)d->node_cap, sizeof(*d->expanded));
    d->edge_action = (int32_t *)malloc((size_t)d->edge_cap * sizeof(*d->edge_action));
    d->edge_child = (int32_t *)malloc((size_t)d->edge_cap * sizeof(*d->edge_child));
    d->edge_next = (int32_t *)malloc((size_t)d->edge_cap * sizeof(*d->edge_next));
    d->hash_table = (int32_t *)malloc((size_t)d->ht_cap * sizeof(*d->hash_table));
    if (!d->positions || !d->first_edge || !d->out_degree || !d->expanded
        || !d->edge_action || !d->edge_child || !d->edge_next || !d->hash_table) {
        cae_position_dag_free(d);
        return -1;
    }
    for (int32_t i = 0; i < d->node_cap; i++) d->first_edge[i] = CAE_DAG_NO_NODE;
    for (int32_t i = 0; i < d->ht_cap; i++) d->hash_table[i] = CAE_DAG_NO_NODE;
    return 0;
}

static int cae_position_dag_grow_nodes(CaePositionDag *d) {
    int32_t old_cap = d->node_cap;
    int32_t new_cap = old_cap * 2;
    if (new_cap <= old_cap) return -1;

#define CAE_DAG_GROW(field, type) do { \
    type *_p = (type *)realloc(d->field, (size_t)new_cap * sizeof(type)); \
    if (!_p) return -1; \
    d->field = _p; \
} while (0)
    CAE_DAG_GROW(positions, CaeDagPosition);
    CAE_DAG_GROW(first_edge, int32_t);
    CAE_DAG_GROW(out_degree, int32_t);
    CAE_DAG_GROW(expanded, uint8_t);
#undef CAE_DAG_GROW

    memset(d->positions + old_cap, 0,
           (size_t)(new_cap - old_cap) * sizeof(*d->positions));
    memset(d->out_degree + old_cap, 0,
           (size_t)(new_cap - old_cap) * sizeof(*d->out_degree));
    memset(d->expanded + old_cap, 0,
           (size_t)(new_cap - old_cap) * sizeof(*d->expanded));
    for (int32_t i = old_cap; i < new_cap; i++) d->first_edge[i] = CAE_DAG_NO_NODE;
    d->node_cap = new_cap;
    return 0;
}

static int cae_position_dag_grow_edges(CaePositionDag *d) {
    int32_t new_cap = d->edge_cap * 2;
    if (new_cap <= d->edge_cap) return -1;
#define CAE_DAG_GROW_EDGE(field) do { \
    int32_t *_p = (int32_t *)realloc(d->field, (size_t)new_cap * sizeof(int32_t)); \
    if (!_p) return -1; \
    d->field = _p; \
} while (0)
    CAE_DAG_GROW_EDGE(edge_action);
    CAE_DAG_GROW_EDGE(edge_child);
    CAE_DAG_GROW_EDGE(edge_next);
#undef CAE_DAG_GROW_EDGE
    d->edge_cap = new_cap;
    return 0;
}

static int cae_position_dag_rehash(CaePositionDag *d, int32_t new_cap) {
    new_cap = cae_dag_next_pow2(new_cap);
    if (new_cap < CAE_DAG_MIN_HT_CAP) new_cap = CAE_DAG_MIN_HT_CAP;
    int32_t *new_ht = (int32_t *)malloc((size_t)new_cap * sizeof(*new_ht));
    if (!new_ht) return -1;
    for (int32_t i = 0; i < new_cap; i++) new_ht[i] = CAE_DAG_NO_NODE;
    int32_t mask = new_cap - 1;
    for (int32_t nid = 0; nid < d->node_count; nid++) {
        uint64_t key = d->positions[nid].key;
        int32_t slot = (int32_t)(key & (uint64_t)mask);
        while (new_ht[slot] != CAE_DAG_NO_NODE)
            slot = (slot + 1) & mask;
        new_ht[slot] = nid;
    }
    free(d->hash_table);
    d->hash_table = new_ht;
    d->ht_cap = new_cap;
    d->ht_mask = mask;
    return 0;
}

static int cae_position_dag_ensure_hash_room(CaePositionDag *d) {
    /* Keep load below 70%.  Open addressing is the canonical store here, not a
     * best-effort cache, so silently overwriting a collision is forbidden. */
    if ((int64_t)(d->node_count + 1) * 10 < (int64_t)d->ht_cap * 7) return 0;
    return cae_position_dag_rehash(d, d->ht_cap * 2);
}

static int32_t cae_position_dag_find_position(
    CaePositionDag *d, const CaeDagPosition *p)
{
    d->probes++;
    int32_t slot = (int32_t)(p->key & (uint64_t)d->ht_mask);
    for (int32_t step = 0; step < d->ht_cap; step++) {
        int32_t nid = d->hash_table[slot];
        if (nid == CAE_DAG_NO_NODE) return CAE_DAG_NO_NODE;
        if (nid >= 0 && nid < d->node_count
            && cae_dag_position_equal(&d->positions[nid], p)) {
            d->hits++;
            return nid;
        }
        d->collision_steps++;
        slot = (slot + 1) & d->ht_mask;
    }
    return CAE_DAG_NO_NODE;
}

static int32_t cae_position_dag_find_board(CaePositionDag *d, const CBoard *b) {
    CaeDagPosition p;
    cae_dag_position_from_cboard(b, &p);
    return cae_position_dag_find_position(d, &p);
}

/* Insert a position the caller has already established is absent.  Keeping this
 * separate from find() lets evaluator users compute an expensive payload before
 * publication: an evaluation failure cannot leave a half-initialised canonical
 * node in the graph. */
static int32_t cae_position_dag_insert_position(
    CaePositionDag *d, const CaeDagPosition *p)
{
    if (cae_position_dag_ensure_hash_room(d) != 0) return CAE_DAG_NO_NODE;
    if (d->node_count >= d->node_cap && cae_position_dag_grow_nodes(d) != 0)
        return CAE_DAG_NO_NODE;

    int32_t nid = d->node_count++;
    d->positions[nid] = *p;
    d->first_edge[nid] = CAE_DAG_NO_NODE;
    d->out_degree[nid] = 0;
    d->expanded[nid] = 0;

    int32_t slot = (int32_t)(p->key & (uint64_t)d->ht_mask);
    while (d->hash_table[slot] != CAE_DAG_NO_NODE)
        slot = (slot + 1) & d->ht_mask;
    d->hash_table[slot] = nid;
    d->inserts++;
    return nid;
}

static int32_t cae_position_dag_intern_board(
    CaePositionDag *d, const CBoard *b, int *created)
{
    CaeDagPosition p;
    cae_dag_position_from_cboard(b, &p);
    int32_t existing = cae_position_dag_find_position(d, &p);
    if (existing != CAE_DAG_NO_NODE) {
        if (created) *created = 0;
        return existing;
    }
    int32_t nid = cae_position_dag_insert_position(d, &p);
    if (nid != CAE_DAG_NO_NODE && created) *created = 1;
    return nid;
}

static int32_t cae_position_dag_child_for_action(
    const CaePositionDag *d, int32_t parent, int32_t action)
{
    if (parent < 0 || parent >= d->node_count) return CAE_DAG_NO_NODE;
    for (int32_t e = d->first_edge[parent]; e != CAE_DAG_NO_NODE; e = d->edge_next[e]) {
        if (d->edge_action[e] == action) return d->edge_child[e];
    }
    return CAE_DAG_NO_NODE;
}

/* Return 1 when inserted, 0 when the exact edge already exists, -2 when the
 * parent already maps this action to a DIFFERENT child, and -1 on allocation or
 * range failure.  The -2 case is a hard caller bug: silently keeping either
 * child would make an accepted action mean something other than what was asked. */
static int cae_position_dag_link(
    CaePositionDag *d, int32_t parent, int32_t action, int32_t child)
{
    if (parent < 0 || parent >= d->node_count || child < 0 || child >= d->node_count)
        return -1;
    if (action < 0 || action >= 4672) return -1;
    for (int32_t e = d->first_edge[parent]; e != CAE_DAG_NO_NODE; e = d->edge_next[e]) {
        if (d->edge_action[e] != action) continue;
        if (d->edge_child[e] == child) {
            d->edge_reuses++;
            return 0;
        }
        return -2;
    }
    if (d->edge_count >= d->edge_cap && cae_position_dag_grow_edges(d) != 0) return -1;
    int32_t e = d->edge_count++;
    d->edge_action[e] = action;
    d->edge_child[e] = child;
    d->edge_next[e] = d->first_edge[parent];
    d->first_edge[parent] = e;
    d->out_degree[parent]++;
    return 1;
}

/* Validate that `action` is legal at the canonical parent AND transforms it
 * into the supplied child STRUCTURE.  The explicit legal-membership check is
 * load-bearing: cboard_push_index() is a defensive no-op for malformed LUT
 * entries, so "push then compare" alone could accidentally accept an illegal
 * no-op edge if a caller also supplied the unchanged position as its child. */
static int cae_position_dag_edge_matches_board(
    const CaePositionDag *d, int32_t parent, int32_t action, const CBoard *child_board)
{
    if (parent < 0 || parent >= d->node_count || action < 0 || action >= 4672) return 0;
    CBoard pushed;
    cae_dag_position_to_cboard(&d->positions[parent], &pushed);

    int legal[CBOARD_MAX_LEGAL_MOVES];
    int n_legal = cboard_legal_move_indices(&pushed, legal, 0);
    int action_is_legal = 0;
    for (int i = 0; i < n_legal; i++) {
        if (legal[i] == action) {
            action_is_legal = 1;
            break;
        }
    }
    if (!action_is_legal) return 0;

    cboard_push_index(&pushed, action);
    CaeDagPosition got, want;
    cae_dag_position_from_cboard(&pushed, &got);
    cae_dag_position_from_cboard(child_board, &want);
    return cae_dag_position_equal(&got, &want);
}

static int cae_position_dag_set_root(CaePositionDag *d, int32_t node_id) {
    if (node_id < 0 || node_id >= d->node_count) return -1;
    d->root_id = node_id;
    return 0;
}

static int cae_position_dag_mark_expanded(CaePositionDag *d, int32_t node_id) {
    if (node_id < 0 || node_id >= d->node_count) return -1;
    d->expanded[node_id] = 1;
    return 0;
}

static void cae_position_dag_reset(CaePositionDag *d) {
    d->node_count = 0;
    d->edge_count = 0;
    d->root_id = CAE_DAG_NO_NODE;
    d->probes = d->hits = d->inserts = d->collision_steps = d->edge_reuses = 0;
    for (int32_t i = 0; i < d->ht_cap; i++) d->hash_table[i] = CAE_DAG_NO_NODE;
}

static int64_t cae_position_dag_memory_bytes(const CaePositionDag *d) {
    int64_t bytes = 0;
    bytes += (int64_t)d->node_cap * (int64_t)(
        sizeof(CaeDagPosition) + sizeof(int32_t) + sizeof(int32_t) + sizeof(uint8_t));
    bytes += (int64_t)d->edge_cap * (int64_t)(3 * sizeof(int32_t));
    bytes += (int64_t)d->ht_cap * (int64_t)sizeof(int32_t);
    return bytes;
}

#endif /* CAE_POSITION_DAG_H */
