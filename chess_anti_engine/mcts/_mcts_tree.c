/*
 * _mcts_tree.c — Array-based MCTS tree with C-accelerated select/expand/backprop.
 *
 * Replaces Python Node objects + dict-based children with contiguous arrays.
 * The hot MCTS inner loop (select → expand → backprop) runs entirely in C.
 * Python only handles leaf board creation, encoding, and GPU evaluation.
 *
 * Tree layout:
 *   Per-node arrays: N (int32), W (double), prior (double), expanded (int8),
 *                    parent (int32), action_from_parent (int32),
 *                    num_children (int32), children_offset (int32)
 *   Children pool:   child_action (int32), child_node (int32)
 *                    stored contiguously, indexed by [children_offset + i]
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Pure-C CBoard implementation (bitboard utilities, attack tables, move gen, CBoard) */
#include "../encoding/_cboard_impl.h"
/* Feature planes for fused encode_146 */
#include "../encoding/_features_impl.h"

/* PyCBoard layout — must match _lc0_ext.c's typedef exactly. */
typedef struct { PyObject_HEAD CBoard board; } PyCBoard;

/* WDL logits → Q value: softmax([w,d,l]) then (p_w - p_l). */
static inline double wdl_logits_to_q(double w, double d, double l) {
    double mx = fmax(fmax(w, d), l);
    double ew = exp(w - mx), ed = exp(d - mx), el = exp(l - mx);
    double ws = ew + ed + el;
    return (ws > 0.0) ? ((ew - el) / ws) : 0.0;
}

/* PyArray_FROMANY wrappers: shorten the repeated C-contiguous-array coercion
 * boilerplate at the Python-binding boundaries. Each returns a new reference
 * (or NULL on failure) that the caller must DECREF. */
#define FROMANY_1D(obj, dtype) \
    ((PyArrayObject *)PyArray_FROMANY((obj), (dtype), 1, 1, NPY_ARRAY_C_CONTIGUOUS))
#define FROMANY_1D_RW(obj, dtype) \
    ((PyArrayObject *)PyArray_FROMANY((obj), (dtype), 1, 1, \
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE))
#define FROMANY_2D(obj, dtype) \
    ((PyArrayObject *)PyArray_FROMANY((obj), (dtype), 2, 2, NPY_ARRAY_C_CONTIGUOUS))
#define FROMANY_4D_RW(obj, dtype) \
    ((PyArrayObject *)PyArray_FROMANY((obj), (dtype), 4, 4, \
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE))

/* Cached PyCBoard type pointer — resolved lazily on first use so we don't
 * need a compile-time dependency on the _lc0_ext module's type object. */
static PyTypeObject *_cached_cboard_type = NULL;

/* Validate and extract CBoard pointers from a Python list.
 * Returns 0 on success, -1 on error (Python exception set).
 * mode=0: copy boards into out_boards (caller provides CBoard[n])
 * mode=1: store pointers into out_ptrs (caller provides const CBoard*[n]) */
static int extract_cboards(PyObject *list, int32_t n,
                           CBoard *out_boards, const CBoard **out_ptrs) {
    if (n <= 0) return 0;
    PyTypeObject *cb_type = _cached_cboard_type;
    if (!cb_type) {
        cb_type = Py_TYPE(PyList_GET_ITEM(list, 0));
        if (strstr(cb_type->tp_name, "CBoard") == NULL) {
            PyErr_SetString(PyExc_TypeError, "list elements must be CBoard objects");
            return -1;
        }
        _cached_cboard_type = cb_type;
    }
    for (int32_t i = 0; i < n; i++) {
        PyObject *item = PyList_GET_ITEM(list, i);
        if (Py_TYPE(item) != cb_type) {
            PyErr_SetString(PyExc_TypeError, "all list elements must be CBoard objects");
            return -1;
        }
        PyCBoard *pcb = (PyCBoard *)item;
        if (out_boards) out_boards[i] = pcb->board;
        if (out_ptrs) out_ptrs[i] = &pcb->board;
    }
    return 0;
}

/* ================================================================
 * NN evaluation cache (position hash → expanded node data)
 * ================================================================ */

#define NNCACHE_MAX_LEGAL  256  /* max legal moves in chess is 218 */
#define MCTS_MAX_PATH      512  /* max tree-traversal depth per query */

typedef struct {
    uint64_t key;           /* zobrist hash */
    int32_t  legal[NNCACHE_MAX_LEGAL];
    double   priors[NNCACHE_MAX_LEGAL]; /* softmaxed */
    int32_t  n_legal;
    double   q_value;       /* from WDL logits */
    uint32_t generation;    /* matches NNCacheData.generation when valid */
} NNCacheEntry;

typedef struct {
    NNCacheEntry *entries;
    int32_t cap;            /* power of 2 */
    int32_t mask;           /* cap - 1 */
    int32_t count;
    uint32_t generation;    /* bumped on clear — avoids memset */
} NNCacheData;

static int nncache_init(NNCacheData *c, int32_t cap) {
    c->cap = cap;
    c->mask = cap - 1;
    c->count = 0;
    c->generation = 1;
    c->entries = (NNCacheEntry *)calloc(cap, sizeof(NNCacheEntry));
    return c->entries ? 0 : -1;
}

static void nncache_free(NNCacheData *c) {
    free(c->entries);
    c->entries = NULL;
    c->count = 0;
}

/* Probe cache. Returns pointer to entry if hit, NULL if miss. */
static NNCacheEntry *nncache_probe(NNCacheData *c, uint64_t hash) {
    int32_t slot = (int32_t)(hash & (uint64_t)c->mask);
    NNCacheEntry *e = &c->entries[slot];
    if (e->generation == c->generation && e->key == hash)
        return e;
    return NULL;
}

/* Insert into cache (direct-mapped, keeps existing same-generation entry).
 * Returns entry pointer to fill, or NULL if slot is occupied by a
 * same-generation entry with a different key (preserve older entry). */
static NNCacheEntry *nncache_insert(NNCacheData *c, uint64_t hash) {
    int32_t slot = (int32_t)(hash & (uint64_t)c->mask);
    NNCacheEntry *e = &c->entries[slot];
    if (e->generation == c->generation && e->key != hash)
        return NULL;  /* keep existing entry (older = more useful) */
    if (e->generation != c->generation) c->count++;
    e->key = hash;
    e->generation = c->generation;
    return e;
}

/* ================================================================
 * Tree data structure
 * ================================================================ */

#define INITIAL_NODE_CAP   4096
#define INITIAL_CHILD_CAP  65536

typedef struct {
    /* Per-node arrays */
    int32_t *N;                /* visit count */
    double  *W;                /* total value */
    double  *prior;            /* prior probability */
    int8_t  *expanded;         /* 1 if expanded */
    int32_t *parent;           /* parent node id (-1 for root) */
    int32_t *action_from_parent; /* action index that led here from parent */
    int32_t *num_children;     /* number of children */
    int32_t *children_offset;  /* offset into children pool */

    int32_t node_count;
    int32_t node_cap;

    /* Children pool (contiguous) */
    int32_t *child_action;     /* action index */
    int32_t *child_node;       /* node id */

    int32_t child_count;
    int32_t child_cap;

    /* Optional CBoard cache for prepare_gumbel_leaves (lazily allocated) */
    CBoard  *cb_cache;      /* one CBoard per node (indexed by node_id) */
    int8_t  *cb_valid;      /* 1 if cb_cache[node_id] is populated */
    int32_t  cb_cache_cap;  /* allocated capacity */

    /* Per-node Zobrist hash (for DAG transposition detection) */
    uint64_t *node_hash;    /* hash per node_id */

    /* Hash table: zobrist_hash → node_id (direct-mapped, -1 = empty) */
    int32_t  *hash_table;
    int32_t   ht_cap;       /* power of 2 */
    int32_t   ht_mask;      /* ht_cap - 1 */
} TreeData;


static int tree_init(TreeData *t) {
    t->node_cap = INITIAL_NODE_CAP;
    t->child_cap = INITIAL_CHILD_CAP;
    t->node_count = 0;
    t->child_count = 0;

    t->N = (int32_t *)calloc(t->node_cap, sizeof(int32_t));
    t->W = (double *)calloc(t->node_cap, sizeof(double));
    t->prior = (double *)calloc(t->node_cap, sizeof(double));
    t->expanded = (int8_t *)calloc(t->node_cap, sizeof(int8_t));
    t->parent = (int32_t *)malloc(t->node_cap * sizeof(int32_t));
    t->action_from_parent = (int32_t *)malloc(t->node_cap * sizeof(int32_t));
    t->num_children = (int32_t *)calloc(t->node_cap, sizeof(int32_t));
    t->children_offset = (int32_t *)calloc(t->node_cap, sizeof(int32_t));

    t->child_action = (int32_t *)malloc(t->child_cap * sizeof(int32_t));
    t->child_node = (int32_t *)malloc(t->child_cap * sizeof(int32_t));

    t->cb_cache = NULL;
    t->cb_valid = NULL;
    t->cb_cache_cap = 0;

    /* Per-node hash + hash table for DAG transposition detection */
    t->node_hash = (uint64_t *)calloc(t->node_cap, sizeof(uint64_t));
    t->ht_cap = 1 << 17;  /* 131072 */
    t->ht_mask = t->ht_cap - 1;
    t->hash_table = (int32_t *)malloc(t->ht_cap * sizeof(int32_t));

    if (!t->N || !t->W || !t->prior || !t->expanded || !t->parent ||
        !t->action_from_parent || !t->num_children || !t->children_offset ||
        !t->child_action || !t->child_node || !t->node_hash || !t->hash_table) {
        return -1;
    }
    memset(t->parent, -1, t->node_cap * sizeof(int32_t));
    memset(t->action_from_parent, -1, t->node_cap * sizeof(int32_t));
    memset(t->hash_table, -1, t->ht_cap * sizeof(int32_t));
    return 0;
}


static void tree_free(TreeData *t) {
    free(t->N);
    free(t->W);
    free(t->prior);
    free(t->expanded);
    free(t->parent);
    free(t->action_from_parent);
    free(t->num_children);
    free(t->children_offset);
    free(t->child_action);
    free(t->child_node);
    free(t->cb_cache);
    free(t->cb_valid);
    free(t->node_hash);
    free(t->hash_table);
    memset(t, 0, sizeof(TreeData));
}


static int tree_grow_nodes(TreeData *t) {
    int32_t new_cap = t->node_cap * 2;

    /* Each realloc is committed to the struct only on success so that an
     * OOM mid-growth leaves t->* pointing at the still-valid old buffers
     * (realloc keeps the original allocation live when it returns NULL). */
    #define GROW(field, type) do { \
        type *_p = (type *)realloc(t->field, new_cap * sizeof(type)); \
        if (!_p) return -1; \
        t->field = _p; \
    } while (0)

    GROW(N, int32_t);
    GROW(W, double);
    GROW(prior, double);
    GROW(expanded, int8_t);
    GROW(parent, int32_t);
    GROW(action_from_parent, int32_t);
    GROW(num_children, int32_t);
    GROW(children_offset, int32_t);
    GROW(node_hash, uint64_t);

    #undef GROW

    /* Zero-init new region */
    int32_t old_cap = t->node_cap;
    memset(t->N + old_cap, 0, (new_cap - old_cap) * sizeof(int32_t));
    memset(t->W + old_cap, 0, (new_cap - old_cap) * sizeof(double));
    memset(t->prior + old_cap, 0, (new_cap - old_cap) * sizeof(double));
    memset(t->expanded + old_cap, 0, (new_cap - old_cap) * sizeof(int8_t));
    memset(t->parent + old_cap, -1, (new_cap - old_cap) * sizeof(int32_t));
    memset(t->action_from_parent + old_cap, -1, (new_cap - old_cap) * sizeof(int32_t));
    memset(t->num_children + old_cap, 0, (new_cap - old_cap) * sizeof(int32_t));
    memset(t->children_offset + old_cap, 0, (new_cap - old_cap) * sizeof(int32_t));
    memset(t->node_hash + old_cap, 0, (new_cap - old_cap) * sizeof(uint64_t));

    t->node_cap = new_cap;
    return 0;
}


/* Hash table: probe for existing node with given Zobrist hash. Returns node_id or -1. */
static int32_t tree_ht_probe(const TreeData *t, uint64_t hash) {
    int32_t slot = (int32_t)(hash & (uint64_t)t->ht_mask);
    int32_t nid = t->hash_table[slot];
    if (nid >= 0 && t->node_hash[nid] == hash)
        return nid;
    return -1;
}

/* Hash table: register node_id for given hash (direct-mapped, overwrites on collision). */
static void tree_ht_insert(TreeData *t, uint64_t hash, int32_t node_id) {
    int32_t slot = (int32_t)(hash & (uint64_t)t->ht_mask);
    /* Don't overwrite an existing exact-hash entry (keep canonical node) */
    int32_t existing = t->hash_table[slot];
    if (existing >= 0 && existing < t->node_count && t->node_hash[existing] == hash)
        return;
    t->hash_table[slot] = node_id;
    t->node_hash[node_id] = hash;
}


static int tree_grow_children(TreeData *t, int32_t need) {
    int32_t new_cap = t->child_cap;
    while (new_cap < t->child_count + need)
        new_cap *= 2;
    if (new_cap == t->child_cap) return 0;

    /* Commit each realloc only on success — leaves t->* valid on OOM. */
    int32_t *new_action = (int32_t *)realloc(t->child_action, new_cap * sizeof(int32_t));
    if (!new_action) return -1;
    t->child_action = new_action;
    int32_t *new_node = (int32_t *)realloc(t->child_node, new_cap * sizeof(int32_t));
    if (!new_node) return -1;
    t->child_node = new_node;

    t->child_cap = new_cap;
    return 0;
}


/* Ensure CBoard cache can hold at least `need` entries. All-or-nothing: on
 * any failure t->cb_cache/cb_valid/cb_cache_cap remain mutually consistent. */
static int tree_ensure_cb_cache(TreeData *t, int32_t need) {
    if (need <= t->cb_cache_cap) return 0;
    int32_t new_cap = t->cb_cache_cap ? t->cb_cache_cap : 256;
    while (new_cap < need) new_cap *= 2;
    CBoard *new_cb = (CBoard *)realloc(t->cb_cache, new_cap * sizeof(CBoard));
    if (!new_cb) return -1;
    t->cb_cache = new_cb;
    int8_t *new_valid = (int8_t *)realloc(t->cb_valid, new_cap * sizeof(int8_t));
    if (!new_valid) return -1;
    t->cb_valid = new_valid;
    memset(t->cb_valid + t->cb_cache_cap, 0, (new_cap - t->cb_cache_cap) * sizeof(int8_t));
    t->cb_cache_cap = new_cap;
    return 0;
}

/* Add a new node. Returns node id. */
static int32_t tree_add_node(TreeData *t, int32_t parent_id, int32_t action, double prior_val) {
    if (t->node_count >= t->node_cap) {
        if (tree_grow_nodes(t) < 0) return -1;
    }
    int32_t id = t->node_count++;
    t->N[id] = 0;
    t->W[id] = 0.0;
    t->prior[id] = prior_val;
    t->expanded[id] = 0;
    t->parent[id] = parent_id;
    t->action_from_parent[id] = action;
    t->num_children[id] = 0;
    t->children_offset[id] = 0;
    return id;
}


/* Expand a node: add children for each (action, prior) pair.
 * All-or-nothing: if any tree_add_node fails, the node is left unexpanded
 * and any partially-allocated child nodes are rolled back. */
static int tree_expand(TreeData *t, int32_t node_id,
                       const int32_t *actions, const double *priors, int32_t n_children) {
    if (n_children <= 0) {
        t->expanded[node_id] = 1;
        return 0;
    }

    if (tree_grow_children(t, n_children) < 0) return -1;

    int32_t offset = t->child_count;
    int32_t saved_node_count = t->node_count;

    for (int32_t i = 0; i < n_children; i++) {
        int32_t child_id = tree_add_node(t, node_id, actions[i], priors[i]);
        if (child_id < 0) {
            /* Roll back: children added so far revert; node stays unexpanded. */
            t->node_count = saved_node_count;
            return -1;
        }
        t->child_action[offset + i] = actions[i];
        t->child_node[offset + i] = child_id;
    }

    /* Commit only after all children exist. */
    t->children_offset[node_id] = offset;
    t->num_children[node_id] = n_children;
    t->expanded[node_id] = 1;
    t->child_count += n_children;
    return 0;
}


/* ================================================================
 * PUCT select child (hottest path)
 * ================================================================ */

/* Select best child of `node_id` using PUCT formula with FPU.
 * Returns index into children pool (children_offset[node_id] + best_slot). */
static int32_t tree_select_child(const TreeData *t, int32_t node_id,
                                  double c_puct, double fpu_reduction) {
    int32_t n_ch = t->num_children[node_id];
    int32_t off = t->children_offset[node_id];

    double parent_N = (double)t->N[node_id];
    double parent_Q = (parent_N > 0) ? (t->W[node_id] / parent_N) : 0.0;
    double c_sqrt_n = c_puct * sqrt(parent_N > 1.0 ? parent_N : 1.0);

    /* Single pass: track best visited score, best unvisited prior, and
     * visited_policy for FPU. All unvisited children share fpu_value, so
     * argmax among them collapses to argmax by prior — compare the two
     * winners after the loop. Child W is in the child's side-to-move frame;
     * negate for the parent-frame score. */
    double visited_policy = 0.0;
    int32_t best_visited_slot = -1;
    double best_visited_score = -1e30;
    int32_t best_unvisited_slot = -1;
    double best_unvisited_prior = -1.0;

    for (int32_t i = 0; i < n_ch; i++) {
        int32_t cid = t->child_node[off + i];
        int32_t n = t->N[cid];
        double prior = t->prior[cid];
        if (n > 0) {
            visited_policy += prior;
            double score = -t->W[cid] / (double)n + c_sqrt_n * prior / (1.0 + (double)n);
            if (score > best_visited_score) {
                best_visited_score = score;
                best_visited_slot = i;
            }
        } else if (prior > best_unvisited_prior) {
            best_unvisited_prior = prior;
            best_unvisited_slot = i;
        }
    }

    if (best_unvisited_slot < 0) return best_visited_slot >= 0 ? best_visited_slot : 0;
    if (best_visited_slot  < 0) return best_unvisited_slot;

    double fpu_value = parent_Q - fpu_reduction * sqrt(visited_policy);
    double best_unvisited_score = fpu_value + c_sqrt_n * best_unvisited_prior;
    return (best_unvisited_score > best_visited_score) ? best_unvisited_slot : best_visited_slot;
}


/* Select leaf from root, following PUCT. Writes path of node ids into `path`.
 * Returns path length. path[0] = root, path[len-1] = leaf. */
static int32_t tree_select_leaf(const TreeData *t, int32_t root_id,
                                 double c_puct, double fpu_at_root, double fpu_reduction,
                                 int32_t *path, int32_t max_path) {
    int32_t node = root_id;
    int32_t depth = 0;
    path[depth++] = node;

    double fpu = fpu_at_root;
    while (t->expanded[node] && t->num_children[node] > 0) {
        if (depth >= max_path) break;
        int32_t slot = tree_select_child(t, node, c_puct, fpu);
        int32_t off = t->children_offset[node];
        node = t->child_node[off + slot];
        path[depth++] = node;
        fpu = fpu_reduction;
    }
    return depth;
}


/* Backprop value up the path. Value is from leaf's side-to-move perspective.
 * Alternates sign at each level. */
static void tree_backprop(TreeData *t, const int32_t *path, int32_t path_len, double value) {
    double v = value;
    for (int32_t i = path_len - 1; i >= 0; i--) {
        int32_t nid = path[i];
        t->N[nid] += 1;
        t->W[nid] += v;
        v = -v;
    }
}


/* Get action path from root to node (for replaying moves on a chess.Board).
 * Writes actions into `out` in order from root→node. Returns path length. */
static int32_t tree_action_path(const TreeData *t, int32_t node_id,
                                 int32_t *out, int32_t max_len) {
    /* First, trace back to root to find path length */
    int32_t len = 0;
    int32_t cur = node_id;
    while (t->parent[cur] >= 0) {
        len++;
        cur = t->parent[cur];
    }
    if (len > max_len) return -1;

    /* Write in reverse */
    cur = node_id;
    for (int32_t i = len - 1; i >= 0; i--) {
        out[i] = t->action_from_parent[cur];
        cur = t->parent[cur];
    }
    return len;
}


/* ================================================================
 * Python wrapper: MCTSTree
 * ================================================================ */

/* Stored state for prepare_and_store / finish_stored workflow.
 * Keeps intermediate data in C memory to avoid expensive Python object creation. */
typedef struct {
    int32_t n_leaves;
    int32_t n_terminals;
    int32_t cap;          /* allocated capacity for per-leaf arrays */

    int32_t *leaf_ids;
    uint64_t *hashes;
    CBoard  *leaf_cboards;   /* saved CBoards for deferred parallel encoding */

    /* Legal moves per leaf: flat buffer + per-leaf offset/count */
    int32_t *legal_flat;
    int32_t *legal_offset;
    int32_t *legal_count;
    int32_t  legal_flat_used;
    int32_t  legal_flat_cap;

    /* Node paths per leaf: flat buffer + per-leaf offset/count */
    int32_t *path_flat;
    int32_t *path_offset;
    int32_t *path_count;
    int32_t  path_flat_used;
    int32_t  path_flat_cap;

    /* Terminal backprop info */
    int32_t *term_path_flat;
    int32_t *term_path_offset;
    int32_t *term_path_count;
    double  *term_values;
    int32_t  term_path_flat_used;
    int32_t  term_path_flat_cap;
    int32_t  term_cap;
} StoredPrepState;

/* Scoring / selection tunables shared across the improved-policy descent. */
typedef struct {
    double c_scale;
    double c_visit;
    double c_puct;
    double fpu_reduction;
    int full_tree;
} GumbelSelectParams;

/* Forward declarations for functions used by GumbelSimState helpers */
static int stored_ensure_cap(StoredPrepState *s, int32_t leaf_cap, int32_t term_cap);
static int stored_ensure_legal_flat(StoredPrepState *s, int32_t need);
static int stored_ensure_path_flat(StoredPrepState *s, int32_t need);
static int stored_ensure_term_path_flat(StoredPrepState *s, int32_t need);
static inline int32_t stored_append_leaf(StoredPrepState *s, TreeData *t,
                                         const CBoard *cb, int32_t leaf_id,
                                         const int32_t *path_buf, int32_t path_len,
                                         int cache_ok);
static int32_t tree_gumbel_collect_leaf(const TreeData *t, int32_t root_id,
    int32_t forced_action, const GumbelSelectParams *sel,
    int32_t *path_buf, int32_t path_cap);
static void cboard_encode_146_into(const CBoard *b, float * restrict out);
static void softmax_inplace(double *arr, int n);

/* Store a CBoard into the tree's per-node cache slot, growing if needed. */
static inline void tree_cb_cache_put(TreeData *t, int32_t leaf_id, const CBoard *cb) {
    if (leaf_id >= t->cb_cache_cap) tree_ensure_cb_cache(t, leaf_id + 1);
    if (leaf_id < t->cb_cache_cap) {
        t->cb_cache[leaf_id] = *cb;
        t->cb_valid[leaf_id] = 1;
    }
}

/* Append a terminal leaf (path + value) to StoredPrepState. */
static inline void stored_append_terminal(StoredPrepState *s,
                                          const int32_t *path_buf, int32_t path_len,
                                          double value) {
    int32_t ti = s->n_terminals;
    stored_ensure_term_path_flat(s, s->term_path_flat_used + path_len);
    s->term_path_offset[ti] = s->term_path_flat_used;
    s->term_path_count[ti] = path_len;
    memcpy(s->term_path_flat + s->term_path_flat_used, path_buf, path_len * sizeof(int32_t));
    s->term_path_flat_used += path_len;
    s->term_values[ti] = value;
    s->n_terminals++;
}

/* ---- Gumbel simulation state machine ------------------------------------ */
#define GSS_POLICY_SIZE 4672
/* Maximum candidates per root per halving round (Gumbel sim). Defined by the
 * score_buf[64] in gss_score_and_halve; raise both if the Gumbel sampler
 * ever needs wider candidate sets. */
#define GSS_MAX_CANDS 64
/* Target leaves per GPU eval batch. Production GPU pipelines run best at
 * 1024-2048 per flush; CPU/TinyNet paths would prefer 256-512. */
#define GSS_GPU_BATCH 1024

typedef struct {
    /* Configuration */
    int32_t n_boards;
    GumbelSelectParams sel;

    /* Per-board state (owned by this struct) */
    int32_t *root_ids;           /* [n_boards] */
    int32_t *budget_remaining;   /* [n_boards] */
    int32_t *visits_per_action;  /* [n_boards] */
    CBoard  *root_cboards;       /* [n_boards] */
    double  *root_qs;            /* [n_boards] */
    double  *root_priors;        /* [n_boards * POLICY_SIZE] */
    double  *gumbels;            /* [n_boards * POLICY_SIZE] */

    /* Per-board candidate lists: flat array + offset/count */
    int32_t *cands_flat;
    int32_t *cands_offset;       /* [n_boards] */
    int32_t *cands_count;        /* [n_boards] */
    int32_t  cands_flat_cap;

    /* Active board indices */
    int32_t *active;             /* [n_boards] */
    int32_t  n_active;

    /* Iteration state */
    int32_t max_reps;
    int32_t current_rep;

    /* Encoding buffer (borrowed pointer — owned by Python) */
    float *enc_data;
    int32_t enc_capacity;
    PyObject *enc_arr_ref;       /* Strong ref to keep enc buffer alive */

    /* Query arrays for prepare_and_store (owned) */
    int32_t *q_board_idx;
    int32_t *q_root_ids;
    int32_t *q_forced;
    int32_t  q_cap;

    /* NNCache pointer (borrowed, valid for lifetime of sim) */
    NNCacheData *nncache;
    PyObject *nncache_ref;       /* Strong ref to keep NNCache alive */

    /* State machine phase */
    int phase;    /* 0=not started, 1=needs_eval, 2=done */
    int allocated;
} GumbelSimState;

static void gss_free(GumbelSimState *g) {
    if (!g->allocated) return;
    free(g->root_ids); free(g->budget_remaining); free(g->visits_per_action);
    free(g->root_cboards); free(g->root_qs); free(g->root_priors);
    free(g->gumbels);
    free(g->cands_flat); free(g->cands_offset); free(g->cands_count);
    free(g->active);
    free(g->q_board_idx); free(g->q_root_ids); free(g->q_forced);
    Py_XDECREF(g->enc_arr_ref);
    Py_XDECREF(g->nncache_ref);
    memset(g, 0, sizeof(*g));
}

/* Compute active boards and visits_per_action for current halving round.
 * Returns max_reps (0 if nothing to do). */
static int32_t gss_begin_round(GumbelSimState *g) {
    g->n_active = 0;
    g->max_reps = 0;
    for (int32_t i = 0; i < g->n_boards; i++) {
        if (g->budget_remaining[i] <= 0 || g->cands_count[i] == 0)
            continue;
        g->active[g->n_active++] = i;
        int32_t rem_count = g->cands_count[i];
        if (rem_count <= 1) {
            g->visits_per_action[i] = g->budget_remaining[i];
        } else {
            int32_t rounds_left = 0;
            int32_t tmp = rem_count;
            while (tmp > 1) { rounds_left++; tmp = (tmp + 1) / 2; }
            int32_t vpa = g->budget_remaining[i] / (rem_count * rounds_left);
            if (vpa < 1) vpa = 1;
            g->visits_per_action[i] = vpa;
        }
        if (g->visits_per_action[i] > g->max_reps)
            g->max_reps = g->visits_per_action[i];
    }
    g->current_rep = 0;
    return g->max_reps;
}

/* Score candidates and halve: keep top half for each active board.
 * Pure C, no Python. */
static void gss_score_and_halve(GumbelSimState *g, TreeData *t) {
    /* Reusable action->child-slot map (sentinel -1). Allocated once, dirty
     * entries are cleared after each board so subsequent boards start clean. */
    int16_t action_to_slot[GSS_POLICY_SIZE];
    memset(action_to_slot, 0xFF, sizeof(action_to_slot));

    for (int32_t ai = 0; ai < g->n_active; ai++) {
        int32_t bi = g->active[ai];
        int32_t n_cands = g->cands_count[bi];

        /* Always deduct budget, even for 1 candidate */
        g->budget_remaining[bi] -= g->visits_per_action[bi] * n_cands;
        if (g->budget_remaining[bi] < 0) g->budget_remaining[bi] = 0;

        if (n_cands <= 1) continue;
        /* Clamp here rather than assert — an upstream bug producing >64 cands
         * would otherwise corrupt scores_buf below. The clamp preserves the
         * top GSS_MAX_CANDS without crashing, and is a no-op in practice. */
        if (n_cands > GSS_MAX_CANDS) n_cands = GSS_MAX_CANDS;

        int32_t rid = g->root_ids[bi];
        double root_Q = (t->N[rid] > 0) ? (t->W[rid] / (double)t->N[rid]) : 0.0;

        /* Single pass over children: max_visit + populate action_to_slot map */
        int32_t n_ch = t->num_children[rid];
        int32_t off = t->children_offset[rid];
        int32_t max_visit = 0;
        for (int32_t j = 0; j < n_ch; j++) {
            int32_t n = t->N[t->child_node[off + j]];
            if (n > max_visit) max_visit = n;
            action_to_slot[t->child_action[off + j]] = (int16_t)j;
        }
        double sigma = g->sel.c_scale * (g->sel.c_visit + (double)max_visit);

        int32_t coff = g->cands_offset[bi];
        double scores_buf[GSS_MAX_CANDS];
        double *scores = scores_buf;
        for (int32_t ci = 0; ci < n_cands; ci++) {
            int32_t action = g->cands_flat[coff + ci];
            double log_prior = log(g->root_priors[bi * GSS_POLICY_SIZE + action] > 1e-12
                                   ? g->root_priors[bi * GSS_POLICY_SIZE + action] : 1e-12);
            double q_hat = root_Q;
            int16_t slot = action_to_slot[action];
            if (slot >= 0) {
                int32_t cid = t->child_node[off + slot];
                if (t->N[cid] > 0) q_hat = -t->W[cid] / (double)t->N[cid];
            }
            scores[ci] = g->gumbels[bi * GSS_POLICY_SIZE + action] + log_prior + sigma * q_hat;
        }

        /* Simple selection sort to find top half (n_cands is small, <=12) */
        int32_t keep = (n_cands + 1) / 2;
        if (keep < 1) keep = 1;
        for (int32_t k = 0; k < keep; k++) {
            int32_t best = k;
            for (int32_t j = k + 1; j < n_cands; j++) {
                if (scores[j] > scores[best]) best = j;
            }
            if (best != k) {
                double tmp_s = scores[k]; scores[k] = scores[best]; scores[best] = tmp_s;
                int32_t tmp_c = g->cands_flat[coff + k];
                g->cands_flat[coff + k] = g->cands_flat[coff + best];
                g->cands_flat[coff + best] = tmp_c;
            }
        }
        g->cands_count[bi] = keep;

        /* Clear the dirty slots we wrote so the map stays -1 for next board. */
        for (int32_t j = 0; j < n_ch; j++) {
            action_to_slot[t->child_action[off + j]] = -1;
        }
    }
}

/* Build query arrays for one batch of reps. Returns number of queries built.
 * Advances current_rep. Fills q_board_idx, q_root_ids, q_forced. Breaks as
 * soon as any queries are built (caller decides when to accumulate more). */
static int32_t gss_build_queries(GumbelSimState *g) {
    int32_t n = 0;
    while (g->current_rep < g->max_reps) {
        int32_t rep_n = 0;
        for (int32_t ai = 0; ai < g->n_active; ai++) {
            int32_t bi = g->active[ai];
            if (g->current_rep >= g->visits_per_action[bi]) continue;
            int32_t rid = g->root_ids[bi];
            if (rid < 0) continue;
            int32_t coff = g->cands_offset[bi];
            int32_t ccount = g->cands_count[bi];
            /* Ensure query arrays have space — commit all three buffers
             * atomically so q_cap and the three pointers never drift. */
            if (n + rep_n + ccount > g->q_cap) {
                int32_t new_cap = (n + rep_n + ccount) * 2;
                if (new_cap < 4096) new_cap = 4096;
                int32_t *nb = realloc(g->q_board_idx, new_cap * sizeof(int32_t));
                int32_t *nr = nb ? realloc(g->q_root_ids, new_cap * sizeof(int32_t)) : NULL;
                int32_t *nf = nr ? realloc(g->q_forced,   new_cap * sizeof(int32_t)) : NULL;
                if (!nb || !nr || !nf) {
                    /* Commit whatever succeeded (prevents leaking the grown buffers)
                     * but leave q_cap at its old value so next call retries growth. */
                    if (nb) g->q_board_idx = nb;
                    if (nr) g->q_root_ids  = nr;
                    return n;
                }
                g->q_board_idx = nb;
                g->q_root_ids  = nr;
                g->q_forced    = nf;
                g->q_cap = new_cap;
            }
            for (int32_t ci = 0; ci < ccount; ci++) {
                g->q_board_idx[n + rep_n] = bi;
                g->q_root_ids[n + rep_n] = rid;
                g->q_forced[n + rep_n] = g->cands_flat[coff + ci];
                rep_n++;
            }
        }
        n += rep_n;
        g->current_rep++;
        if (n > 0 || g->current_rep >= g->max_reps)
            break;
    }
    return n;
}

/* Run one batch of tree queries (tree walk + replay + encode) using the
 * gumbel sim state's query arrays.  Pure C, no GIL needed.
 * Appends to StoredPrepState, writes encoded positions to enc_data.
 * Returns the number of new leaves that need GPU eval. */
static int32_t gss_prepare_batch(
    TreeData *t, StoredPrepState *s, GumbelSimState *g,
    int32_t n_queries, float *enc_data)
{
    int cache_ok = (t->cb_cache != NULL);
    int32_t path_buf[MCTS_MAX_PATH];
    int32_t old_n_leaves = s->n_leaves;

    for (int32_t qi = 0; qi < n_queries; qi++) {
        int32_t bi = g->q_board_idx[qi];
        int32_t rid = g->q_root_ids[qi];
        int32_t forced = g->q_forced[qi];

        int32_t path_len = tree_gumbel_collect_leaf(
            t, rid, forced, &g->sel, path_buf, MCTS_MAX_PATH);

        int32_t leaf_id = path_buf[path_len - 1];

        if (path_len <= 1) {
            stored_append_terminal(s, path_buf, path_len, g->root_qs[bi]);
            continue;
        }

        /* CBoard replay */
        CBoard cb;
        int32_t replay_actions[MCTS_MAX_PATH];
        int32_t n_replay = 0;
        int32_t found_cached = 0;

        if (cache_ok) {
            int32_t nid = leaf_id;
            while (nid >= 0 && nid != rid && n_replay < MCTS_MAX_PATH) {
                if (nid < t->cb_cache_cap && t->cb_valid[nid]) {
                    cb = t->cb_cache[nid];
                    found_cached = 1;
                    break;
                }
                int32_t act = t->action_from_parent[nid];
                if (act >= 0) replay_actions[n_replay++] = act;
                nid = t->parent[nid];
            }
            if (!found_cached && nid == rid &&
                nid < t->cb_cache_cap && t->cb_valid[nid]) {
                cb = t->cb_cache[nid];
                found_cached = 1;
            }
        }
        if (!found_cached) {
            cb = g->root_cboards[bi];
        }
        for (int32_t ri = n_replay - 1; ri >= 0; ri--)
            cboard_push_index(&cb, replay_actions[ri]);

        if (cboard_is_game_over(&cb)) {
            stored_append_terminal(s, path_buf, path_len, (double)cboard_terminal_value(&cb));
            continue;
        }

        /* Transposition / NNCache check */
        if (!t->expanded[leaf_id]) {
            int32_t existing = tree_ht_probe(t, cb.hash);
            if (existing >= 0 && existing != leaf_id && t->expanded[existing]) {
                int32_t n_ch = t->num_children[existing];
                if (n_ch > 0) {
                    int32_t ex_off = t->children_offset[existing];
                    int32_t legal_buf[256];
                    double prior_buf[256];
                    int32_t n_copy = (n_ch <= 256) ? n_ch : 256;
                    for (int32_t c = 0; c < n_copy; c++) {
                        legal_buf[c] = t->child_action[ex_off + c];
                        prior_buf[c] = t->prior[t->child_node[ex_off + c]];
                    }
                    tree_expand(t, leaf_id, legal_buf, prior_buf, n_copy);
                } else {
                    t->expanded[leaf_id] = 1;
                }
                double q = (t->N[existing] > 0) ? (t->W[existing] / (double)t->N[existing]) : 0.0;
                tree_backprop(t, path_buf, path_len, q);
                tree_ht_insert(t, cb.hash, leaf_id);
                if (cache_ok) tree_cb_cache_put(t, leaf_id, &cb);
                continue;
            }
            if (g->nncache) {
                NNCacheEntry *ce = nncache_probe(g->nncache, cb.hash);
                if (ce) {
                    tree_expand(t, leaf_id, ce->legal, ce->priors, ce->n_legal);
                    tree_backprop(t, path_buf, path_len, ce->q_value);
                    tree_ht_insert(t, cb.hash, leaf_id);
                    if (cache_ok) tree_cb_cache_put(t, leaf_id, &cb);
                    continue;
                }
            }
        }

        if (s->n_leaves >= g->enc_capacity) {
            /* Buffer full — fall back to root Q (not 0.0 which biases toward draws). */
            stored_append_terminal(s, path_buf, path_len, g->root_qs[bi]);
            continue;
        }

        /* Save leaf for deferred encoding */
        stored_append_leaf(s, t, &cb, leaf_id, path_buf, path_len, cache_ok);
    }

    /* Backprop terminals */
    for (int32_t ti = 0; ti < s->n_terminals; ti++) {
        tree_backprop(t, s->term_path_flat + s->term_path_offset[ti],
                      s->term_path_count[ti], s->term_values[ti]);
    }
    /* Reset terminal count since we've backpropped them */
    s->n_terminals = 0;
    s->term_path_flat_used = 0;

    /* Parallel encoding */
    {
        int32_t n_to_encode = s->n_leaves - old_n_leaves;
        int32_t base = old_n_leaves;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(n_to_encode > 64)
#endif
        for (int32_t i = 0; i < n_to_encode; i++) {
            int32_t li = base + i;
            cboard_encode_146_into(&s->leaf_cboards[li], enc_data + li * 146 * 64);
        }
    }

    return s->n_leaves - old_n_leaves;
}

/* Run expand + backprop for all stored leaves using GPU eval results.
 * Pure C, no GIL needed. */
static void gss_finish_batch(
    TreeData *t, StoredPrepState *s, NNCacheData *nncache,
    const float *pol_data, const float *wdl_data)
{
    for (int32_t li = 0; li < s->n_leaves; li++) {
        int32_t nid = s->leaf_ids[li];
        const int32_t *legal = s->legal_flat + s->legal_offset[li];
        int32_t n_legal = s->legal_count[li];
        const int32_t *path = s->path_flat + s->path_offset[li];
        int32_t path_len = s->path_count[li];
        const float *wdl = wdl_data + li * 3;
        double q = wdl_logits_to_q((double)wdl[0], (double)wdl[1], (double)wdl[2]);

        if (!t->expanded[nid] && n_legal > 0) {
            const float *logits = pol_data + li * 4672;
            double priors_stack[256];
            double *priors = (n_legal <= 256) ? priors_stack
                                               : (double *)malloc(n_legal * sizeof(double));
            if (priors) {
                for (int32_t j = 0; j < n_legal; j++)
                    priors[j] = (double)logits[legal[j]];
                softmax_inplace(priors, n_legal);
                tree_expand(t, nid, legal, priors, n_legal);
                if (nncache && n_legal <= NNCACHE_MAX_LEGAL) {
                    NNCacheEntry *ce = nncache_insert(nncache, s->hashes[li]);
                    if (ce) {
                        ce->n_legal = n_legal;
                        memcpy(ce->legal, legal, n_legal * sizeof(int32_t));
                        memcpy(ce->priors, priors, n_legal * sizeof(double));
                        ce->q_value = q;
                    }
                }
                if (priors != priors_stack) free(priors);
            }
        }

        tree_backprop(t, path, path_len, q);
        tree_ht_insert(t, s->hashes[li], nid);
    }
}

/* Core simulation loop: build queries → prepare → return for GPU eval,
 * or do scoring/halving and continue to next round.
 * Returns number of leaves needing eval (>0), or 0 if simulation is done. */
static int32_t gss_step(TreeData *t, StoredPrepState *s, GumbelSimState *g, float *enc_data)
{
    /* Accumulate leaves across multiple reps before returning for GPU eval:
     * reduces Python↔C round trips at the cost of slightly staler tree state
     * on later leaves in the batch. See GSS_GPU_BATCH definition for tuning
     * guidance. */
    int32_t target_batch = GSS_GPU_BATCH;

    for (;;) {
        /* Build queries for one rep at a time */
        int32_t n_queries = gss_build_queries(g);

        if (n_queries > 0) {
            /* Ensure stored capacity */
            stored_ensure_cap(s, s->n_leaves + n_queries, s->n_terminals + n_queries);
            stored_ensure_legal_flat(s, s->legal_flat_used + n_queries * 40);
            stored_ensure_path_flat(s, s->path_flat_used + n_queries * 8);
            stored_ensure_term_path_flat(s, s->term_path_flat_used + n_queries * 8);

            /* Seed CBoard cache with roots for any new board indices */
            for (int32_t qi = 0; qi < n_queries; qi++) {
                int32_t rid = g->q_root_ids[qi];
                int32_t bi = g->q_board_idx[qi];
                if (rid >= 0 && rid < t->cb_cache_cap) {
                    if (!t->cb_valid[rid]) {
                        t->cb_cache[rid] = g->root_cboards[bi];
                        t->cb_valid[rid] = 1;
                    }
                }
            }

            gss_prepare_batch(t, s, g, n_queries, enc_data);

            /* Accumulate more leaves before returning, unless:
             * - We've hit the target batch size
             * - Buffer is getting full
             * - No more reps to build */
            if (s->n_leaves >= target_batch ||
                s->n_leaves + n_queries * 2 > g->enc_capacity ||
                g->current_rep >= g->max_reps) {
                if (s->n_leaves > 0) {
                    return s->n_leaves;
                }
            }
            continue;
        }

        /* Reps exhausted for this round — flush any accumulated leaves first */
        if (s->n_leaves > 0) {
            return s->n_leaves;
        }

        /* Do scoring and halving */
        gss_score_and_halve(g, t);

        /* Start next halving round */
        if (gss_begin_round(g) == 0) {
            g->phase = 2;
            return 0;
        }
    }
}

typedef struct {
    PyObject_HEAD
    TreeData tree;
    StoredPrepState stored;
    GumbelSimState gsim;
} MCTSTreeObject;


static void stored_free(StoredPrepState *s) {
    free(s->leaf_ids); free(s->hashes); free(s->leaf_cboards);
    free(s->legal_flat); free(s->legal_offset); free(s->legal_count);
    free(s->path_flat); free(s->path_offset); free(s->path_count);
    free(s->term_path_flat); free(s->term_path_offset); free(s->term_path_count);
    free(s->term_values);
    memset(s, 0, sizeof(*s));
}

static int stored_ensure_cap(StoredPrepState *s, int32_t leaf_cap, int32_t term_cap) {
    /* Commit each realloc only on success so OOM mid-growth leaves s->*
     * pointing at the previous (still-valid) buffers. */
    #define SGROW(field, type, cap) do { \
        type *_p = (type *)realloc(s->field, (cap) * sizeof(type)); \
        if (!_p) return -1; \
        s->field = _p; \
    } while (0)

    if (leaf_cap > s->cap) {
        SGROW(leaf_ids,     int32_t,  leaf_cap);
        SGROW(hashes,       uint64_t, leaf_cap);
        SGROW(leaf_cboards, CBoard,   leaf_cap);
        SGROW(legal_offset, int32_t,  leaf_cap);
        SGROW(legal_count,  int32_t,  leaf_cap);
        SGROW(path_offset,  int32_t,  leaf_cap);
        SGROW(path_count,   int32_t,  leaf_cap);
        s->cap = leaf_cap;
    }
    if (term_cap > s->term_cap) {
        SGROW(term_path_offset, int32_t, term_cap);
        SGROW(term_path_count,  int32_t, term_cap);
        SGROW(term_values,      double,  term_cap);
        s->term_cap = term_cap;
    }
    #undef SGROW
    return 0;
}

/* Grow a flat int32 buffer to `needed` with amortized 2x expansion.
 * Returns 0 on success, -1 on OOM. */
static int flat_buffer_grow(int32_t **buf, int32_t *cap, int32_t needed) {
    if (needed <= *cap) return 0;
    int32_t new_cap = (needed > *cap * 2) ? needed : *cap * 2;
    int32_t *p = (int32_t *)realloc(*buf, new_cap * sizeof(int32_t));
    if (!p) return -1;
    *buf = p;
    *cap = new_cap;
    return 0;
}

static int stored_ensure_legal_flat(StoredPrepState *s, int32_t needed) {
    return flat_buffer_grow(&s->legal_flat, &s->legal_flat_cap, needed);
}

static int stored_ensure_path_flat(StoredPrepState *s, int32_t needed) {
    return flat_buffer_grow(&s->path_flat, &s->path_flat_cap, needed);
}

static int stored_ensure_term_path_flat(StoredPrepState *s, int32_t needed) {
    return flat_buffer_grow(&s->term_path_flat, &s->term_path_flat_cap, needed);
}

/* Append one leaf to the stored prep state: record its CBoard, legal moves,
 * and search path into flat buffers, optionally cache the CBoard on the tree,
 * and bump n_leaves. Returns the leaf's slot index (pre-increment). */
static inline int32_t stored_append_leaf(StoredPrepState *s, TreeData *t,
                                         const CBoard *cb, int32_t leaf_id,
                                         const int32_t *path_buf, int32_t path_len,
                                         int cache_ok) {
    int32_t li = s->n_leaves;
    s->leaf_cboards[li] = *cb;
    s->leaf_ids[li] = leaf_id;
    s->hashes[li] = cb->hash;

    int indices[256];
    int count = cboard_legal_move_indices(cb, indices, /*sorted=*/0);
    stored_ensure_legal_flat(s, s->legal_flat_used + count);
    s->legal_offset[li] = s->legal_flat_used;
    s->legal_count[li] = count;
    for (int32_t j = 0; j < count; j++)
        s->legal_flat[s->legal_flat_used + j] = indices[j];
    s->legal_flat_used += count;

    stored_ensure_path_flat(s, s->path_flat_used + path_len);
    s->path_offset[li] = s->path_flat_used;
    s->path_count[li] = path_len;
    memcpy(s->path_flat + s->path_flat_used, path_buf, path_len * sizeof(int32_t));
    s->path_flat_used += path_len;

    if (cache_ok) tree_cb_cache_put(t, leaf_id, cb);

    s->n_leaves++;
    return li;
}

static void MCTSTree_dealloc(MCTSTreeObject *self) {
    gss_free(&self->gsim);
    stored_free(&self->stored);
    tree_free(&self->tree);
    Py_TYPE(self)->tp_free((PyObject *)self);
}


static int MCTSTree_init(MCTSTreeObject *self, PyObject *args, PyObject *kwds) {
    (void)args;
    (void)kwds;
    memset(&self->stored, 0, sizeof(self->stored));
    if (tree_init(&self->tree) < 0) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate MCTS tree");
        return -1;
    }
    return 0;
}


/* ================================================================
 * Python wrapper: NNCache
 * ================================================================ */

typedef struct {
    PyObject_HEAD
    NNCacheData cache;
} PyNNCacheObject;

static void PyNNCache_dealloc(PyNNCacheObject *self) {
    nncache_free(&self->cache);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int PyNNCache_init(PyNNCacheObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"capacity", NULL};
    int cap = 1 << 17;  /* 131072 default */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &cap))
        return -1;
    /* Round up to power of 2 */
    int p = 1;
    while (p < cap) p <<= 1;
    if (nncache_init(&self->cache, p) < 0) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate NNCache");
        return -1;
    }
    return 0;
}

static PyObject *PyNNCache_count(PyNNCacheObject *self, PyObject *Py_UNUSED(ignored)) {
    return PyLong_FromLong(self->cache.count);
}

static PyObject *PyNNCache_clear(PyNNCacheObject *self, PyObject *Py_UNUSED(ignored)) {
    self->cache.generation++;
    self->cache.count = 0;
    Py_RETURN_NONE;
}

static PyObject *PyNNCache_probe(PyNNCacheObject *self, PyObject *args) {
    unsigned long long hash;
    if (!PyArg_ParseTuple(args, "K", &hash))
        return NULL;
    NNCacheEntry *e = nncache_probe(&self->cache, (uint64_t)hash);
    if (!e)
        Py_RETURN_NONE;
    /* Return (legal_indices, priors, q_value) */
    npy_intp dims[1] = {e->n_legal};
    PyObject *legal_arr = PyArray_SimpleNew(1, dims, NPY_INT32);
    PyObject *prior_arr = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    memcpy(PyArray_DATA((PyArrayObject *)legal_arr), e->legal, e->n_legal * sizeof(int32_t));
    memcpy(PyArray_DATA((PyArrayObject *)prior_arr), e->priors, e->n_legal * sizeof(double));
    return Py_BuildValue("(NNd)", legal_arr, prior_arr, e->q_value);
}

static PyObject *PyNNCache_keys(PyNNCacheObject *self, PyObject *Py_UNUSED(ignored)) {
    NNCacheData *c = &self->cache;
    PyObject *list = PyList_New(0);
    if (!list) return NULL;
    for (int32_t i = 0; i < c->cap; i++) {
        NNCacheEntry *e = &c->entries[i];
        if (e->generation == c->generation) {
            PyObject *val = PyLong_FromUnsignedLongLong(e->key);
            PyList_Append(list, val);
            Py_DECREF(val);
        }
    }
    return list;
}

static PyMethodDef PyNNCache_methods[] = {
    {"count", (PyCFunction)PyNNCache_count, METH_NOARGS, "count() -> int"},
    {"clear", (PyCFunction)PyNNCache_clear, METH_NOARGS, "clear() -> None"},
    {"probe", (PyCFunction)PyNNCache_probe, METH_VARARGS, "probe(hash) -> (q, n_legal) or None"},
    {"keys", (PyCFunction)PyNNCache_keys, METH_NOARGS, "keys() -> list of cached hashes"},
    {NULL}
};

static PyTypeObject PyNNCacheType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_mcts_tree.NNCache",
    .tp_doc = "C-level NN evaluation cache keyed by Zobrist hash.",
    .tp_basicsize = sizeof(PyNNCacheObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)PyNNCache_init,
    .tp_dealloc = (destructor)PyNNCache_dealloc,
    .tp_methods = PyNNCache_methods,
};


/* add_root(N, W) -> int (root node id) */
static PyObject *MCTSTree_add_root(MCTSTreeObject *self, PyObject *args) {
    int N;
    double W;
    if (!PyArg_ParseTuple(args, "id", &N, &W))
        return NULL;

    int32_t id = tree_add_node(&self->tree, -1, -1, 1.0);
    if (id < 0) {
        PyErr_SetString(PyExc_MemoryError, "Failed to add root node");
        return NULL;
    }
    self->tree.N[id] = N;
    self->tree.W[id] = W;
    return PyLong_FromLong(id);
}


/* expand(node_id, actions_array, priors_array) -> None
 * actions: int32 numpy array, priors: float64 numpy array */
static PyObject *MCTSTree_expand(MCTSTreeObject *self, PyObject *args) {
    int node_id;
    PyObject *actions_obj, *priors_obj;
    if (!PyArg_ParseTuple(args, "iOO", &node_id, &actions_obj, &priors_obj))
        return NULL;

    PyArrayObject *actions_arr = FROMANY_1D(actions_obj, NPY_INT32);
    PyArrayObject *priors_arr = FROMANY_1D(priors_obj, NPY_FLOAT64);

    if (!actions_arr || !priors_arr) {
        Py_XDECREF(actions_arr);
        Py_XDECREF(priors_arr);
        return NULL;
    }

    int32_t n = (int32_t)PyArray_SIZE(actions_arr);
    if (n != (int32_t)PyArray_SIZE(priors_arr)) {
        PyErr_SetString(PyExc_ValueError, "actions and priors must have same length");
        Py_DECREF(actions_arr);
        Py_DECREF(priors_arr);
        return NULL;
    }

    const int32_t *act = (const int32_t *)PyArray_DATA(actions_arr);
    const double *pri = (const double *)PyArray_DATA(priors_arr);

    if (tree_expand(&self->tree, (int32_t)node_id, act, pri, n) < 0) {
        PyErr_SetString(PyExc_MemoryError, "Failed to expand node");
        Py_DECREF(actions_arr);
        Py_DECREF(priors_arr);
        return NULL;
    }

    Py_DECREF(actions_arr);
    Py_DECREF(priors_arr);
    Py_RETURN_NONE;
}


/*
 * select_leaves(root_ids, c_puct, fpu_at_root, fpu_reduction)
 *   -> list of (leaf_node_id, [action_path from root to leaf])
 *
 * One leaf per root. Caller uses action_path to build the chess.Board.
 */
static PyObject *MCTSTree_select_leaves(MCTSTreeObject *self, PyObject *args) {
    PyObject *root_ids_obj;
    double c_puct, fpu_at_root, fpu_reduction;
    if (!PyArg_ParseTuple(args, "Oddd", &root_ids_obj, &c_puct, &fpu_at_root, &fpu_reduction))
        return NULL;

    PyArrayObject *root_ids_arr = FROMANY_1D(root_ids_obj, NPY_INT32);
    if (!root_ids_arr) return NULL;

    int32_t n_roots = (int32_t)PyArray_SIZE(root_ids_arr);
    const int32_t *root_ids = (const int32_t *)PyArray_DATA(root_ids_arr);

    int32_t path_buf[MCTS_MAX_PATH];

    PyObject *result = PyList_New(n_roots);
    if (!result) {
        Py_DECREF(root_ids_arr);
        return NULL;
    }

    for (int32_t i = 0; i < n_roots; i++) {
        int32_t root_id = root_ids[i];
        int32_t path_len = tree_select_leaf(&self->tree, root_id,
                                             c_puct, fpu_at_root, fpu_reduction,
                                             path_buf, MCTS_MAX_PATH);

        int32_t leaf_id = path_buf[path_len - 1];

        /* Build action path from root to leaf */
        int32_t action_buf[MCTS_MAX_PATH];
        int32_t action_len = tree_action_path(&self->tree, leaf_id, action_buf, MCTS_MAX_PATH);
        if (action_len < 0) action_len = 0;

        /* Build Python tuple: (leaf_id, path_as_int32_array, node_path_as_int32_array) */
        npy_intp dims[1] = {action_len};
        PyObject *action_path = PyArray_SimpleNew(1, dims, NPY_INT32);
        if (action_path) {
            memcpy(PyArray_DATA((PyArrayObject *)action_path),
                   action_buf, action_len * sizeof(int32_t));
        }

        npy_intp pdims[1] = {path_len};
        PyObject *node_path = PyArray_SimpleNew(1, pdims, NPY_INT32);
        if (node_path) {
            memcpy(PyArray_DATA((PyArrayObject *)node_path),
                   path_buf, path_len * sizeof(int32_t));
        }

        PyObject *is_exp = self->tree.expanded[leaf_id] ? Py_True : Py_False;
        Py_INCREF(is_exp);
        PyObject *leaf_id_obj = PyLong_FromLong(leaf_id);
        PyObject *tup = PyTuple_Pack(4,
            leaf_id_obj,
            action_path ? action_path : Py_None,
            node_path ? node_path : Py_None,
            is_exp);
        Py_XDECREF(leaf_id_obj);
        Py_XDECREF(action_path);
        Py_XDECREF(node_path);
        Py_DECREF(is_exp);

        PyList_SET_ITEM(result, i, tup);
    }

    Py_DECREF(root_ids_arr);
    return result;
}


/* backprop(node_path, value) -> None
 * node_path: int32 numpy array of node ids from root to leaf.
 * value: float, from leaf's perspective. */
static PyObject *MCTSTree_backprop(MCTSTreeObject *self, PyObject *args) {
    PyObject *path_obj;
    double value;
    if (!PyArg_ParseTuple(args, "Od", &path_obj, &value))
        return NULL;

    PyArrayObject *path_arr = FROMANY_1D(path_obj, NPY_INT32);
    if (!path_arr) return NULL;

    int32_t path_len = (int32_t)PyArray_SIZE(path_arr);
    const int32_t *path = (const int32_t *)PyArray_DATA(path_arr);

    tree_backprop(&self->tree, path, path_len, value);

    Py_DECREF(path_arr);
    Py_RETURN_NONE;
}


/* backprop_many(list_of_node_paths, list_of_values) -> None
 * Batched backprop: avoids per-call Python overhead. */
static PyObject *MCTSTree_backprop_many(MCTSTreeObject *self, PyObject *args) {
    PyObject *paths_list, *values_list;
    if (!PyArg_ParseTuple(args, "OO", &paths_list, &values_list))
        return NULL;

    Py_ssize_t n = PyList_Size(paths_list);
    if (n != PyList_Size(values_list)) {
        PyErr_SetString(PyExc_ValueError, "paths and values must have same length");
        return NULL;
    }

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *path_obj = PyList_GET_ITEM(paths_list, i);
        double value = PyFloat_AsDouble(PyList_GET_ITEM(values_list, i));
        if (value == -1.0 && PyErr_Occurred()) return NULL;

        PyArrayObject *path_arr = FROMANY_1D(path_obj, NPY_INT32);
        if (!path_arr) return NULL;

        int32_t path_len = (int32_t)PyArray_SIZE(path_arr);
        const int32_t *path = (const int32_t *)PyArray_DATA(path_arr);

        tree_backprop(&self->tree, path, path_len, value);
        Py_DECREF(path_arr);
    }

    Py_RETURN_NONE;
}


/* ================================================================
 * Softmax helper
 * ================================================================ */

static void softmax_inplace(double *arr, int n) {
    if (n <= 0) return;
    double max_val = arr[0];
    for (int i = 1; i < n; i++)
        if (arr[i] > max_val) max_val = arr[i];
    /* Fall back to uniform if inputs are non-finite (e.g. all -inf). */
    if (!isfinite(max_val)) {
        double u = 1.0 / (double)n;
        for (int i = 0; i < n; i++) arr[i] = u;
        return;
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        arr[i] = exp(arr[i] - max_val);
        sum += arr[i];
    }
    if (sum > 0.0 && isfinite(sum)) {
        for (int i = 0; i < n; i++)
            arr[i] /= sum;
    } else {
        double u = 1.0 / (double)n;
        for (int i = 0; i < n; i++) arr[i] = u;
    }
}

/* ================================================================
 * Gumbel improved-policy selection
 * ================================================================ */

/* Select best child using Gumbel improved-policy.
 * Returns slot index (offset from children_offset). */
static int32_t tree_gumbel_select_child(const TreeData *t, int32_t node_id,
                                         double c_scale, double c_visit) {
    int32_t n_ch = t->num_children[node_id];
    int32_t off = t->children_offset[node_id];
    if (n_ch <= 0) return 0;

    double parent_N = (double)t->N[node_id];
    double parent_Q = (parent_N > 0) ? (t->W[node_id] / parent_N) : 0.0;

    /* Two arrays — log_prior and completed_Q — precomputed in the same pass
     * that computes max_visit + total_visits. Avoids re-reading child arrays. */
    double *log_priors = (double *)alloca(n_ch * sizeof(double));
    double *cqs        = (double *)alloca(n_ch * sizeof(double));
    int32_t max_visit = 0;
    int32_t total_visits = 0;
    for (int32_t i = 0; i < n_ch; i++) {
        int32_t cid = t->child_node[off + i];
        int32_t n = t->N[cid];
        if (n > max_visit) max_visit = n;
        total_visits += n;
        log_priors[i] = log(t->prior[cid] > 1e-12 ? t->prior[cid] : 1e-12);
        cqs[i] = (n > 0) ? (-t->W[cid] / (double)n) : parent_Q;
    }
    double sigma = c_scale * (c_visit + (double)max_visit);
    double inv_total = 1.0 / (1.0 + (double)total_visits);

    /* Fuse score-compute with softmax-max tracking. */
    double *scores = (double *)alloca(n_ch * sizeof(double));
    double max_score = -INFINITY;
    for (int32_t i = 0; i < n_ch; i++) {
        double s = log_priors[i] + sigma * cqs[i];
        scores[i] = s;
        if (s > max_score) max_score = s;
    }

    /* exp + sum, with fallback to uniform if all inputs are non-finite. */
    double sum = 0.0;
    if (isfinite(max_score)) {
        for (int32_t i = 0; i < n_ch; i++) {
            scores[i] = exp(scores[i] - max_score);
            sum += scores[i];
        }
    }
    int uniform_fallback = !(sum > 0.0 && isfinite(sum));
    double inv_sum = uniform_fallback ? (1.0 / (double)n_ch) : (1.0 / sum);

    /* Fused normalize + argmax(prob - N/(1+total_N)). */
    int32_t best_slot = 0;
    double best_val = -1e30;
    for (int32_t i = 0; i < n_ch; i++) {
        double p = uniform_fallback ? inv_sum : (scores[i] * inv_sum);
        double target = p - (double)t->N[t->child_node[off + i]] * inv_total;
        if (target > best_val) {
            best_val = target;
            best_slot = i;
        }
    }
    return best_slot;
}


/* Traverse: follow forced_action from root, then improved-policy below.
 * Returns path length. path[0]=root, path[len-1]=leaf. */
static int32_t tree_gumbel_collect_leaf(const TreeData *t, int32_t root_id,
                                         int32_t forced_action,
                                         const GumbelSelectParams *sel,
                                         int32_t *path, int32_t max_path) {
    int32_t depth = 0;
    path[depth++] = root_id;

    int32_t n_ch = t->num_children[root_id];
    int32_t off = t->children_offset[root_id];
    int32_t child_id = -1;
    for (int32_t i = 0; i < n_ch; i++) {
        if (t->child_action[off + i] == forced_action) {
            child_id = t->child_node[off + i];
            break;
        }
    }
    if (child_id < 0) return depth;

    path[depth++] = child_id;
    int32_t node = child_id;

    while (t->expanded[node] && t->num_children[node] > 0) {
        if (depth >= max_path) break;

        int32_t n_ch2 = t->num_children[node];
        int32_t off2 = t->children_offset[node];
        /* Fused scan: find any visited child; if none, track best-prior slot.
         * Early exit as soon as a visited child is found — prior tracking
         * only matters on the frontier path. */
        int32_t any_visited = 0;
        int32_t best_prior_slot = 0;
        double best_pri = -1.0;
        for (int32_t i = 0; i < n_ch2; i++) {
            int32_t cid = t->child_node[off2 + i];
            if (t->N[cid] > 0) { any_visited = 1; break; }
            if (t->prior[cid] > best_pri) {
                best_pri = t->prior[cid];
                best_prior_slot = i;
            }
        }

        int32_t slot;
        if (!any_visited) {
            /* Frontier: all children unvisited — highest prior wins */
            slot = best_prior_slot;
        } else if (sel->full_tree) {
            slot = tree_gumbel_select_child(t, node, sel->c_scale, sel->c_visit);
        } else {
            slot = tree_select_child(t, node, sel->c_puct, sel->fpu_reduction);
        }
        node = t->child_node[t->children_offset[node] + slot];
        path[depth++] = node;
    }

    return depth;
}



/* get_children_visits(node_id) -> (actions: int32[], visits: int32[]) */
static PyObject *MCTSTree_get_children_visits(MCTSTreeObject *self, PyObject *args) {
    int node_id;
    if (!PyArg_ParseTuple(args, "i", &node_id))
        return NULL;

    TreeData *t = &self->tree;
    int32_t n_ch = t->num_children[node_id];
    int32_t off = t->children_offset[node_id];

    npy_intp dims[1] = {n_ch};
    PyObject *actions = PyArray_SimpleNew(1, dims, NPY_INT32);
    PyObject *visits = PyArray_SimpleNew(1, dims, NPY_INT32);
    if (!actions || !visits) {
        Py_XDECREF(actions);
        Py_XDECREF(visits);
        return NULL;
    }

    int32_t *a_data = (int32_t *)PyArray_DATA((PyArrayObject *)actions);
    int32_t *v_data = (int32_t *)PyArray_DATA((PyArrayObject *)visits);

    for (int32_t i = 0; i < n_ch; i++) {
        a_data[i] = t->child_action[off + i];
        v_data[i] = t->N[t->child_node[off + i]];
    }

    PyObject *result = PyTuple_Pack(2, actions, visits);
    Py_DECREF(actions);
    Py_DECREF(visits);
    return result;
}


/* get_children_q(node_id, default_q) -> (actions: int32[], visits: int32[], q: float64[])
 * q[i] = -child_W[i]/child_N[i] (parent-perspective) for visited children,
 * or default_q for unvisited. */
static PyObject *MCTSTree_get_children_q(MCTSTreeObject *self, PyObject *args) {
    int node_id;
    double default_q;
    if (!PyArg_ParseTuple(args, "id", &node_id, &default_q))
        return NULL;

    TreeData *t = &self->tree;
    int32_t n_ch = t->num_children[node_id];
    int32_t off = t->children_offset[node_id];

    npy_intp dims[1] = {n_ch};
    PyObject *actions = PyArray_SimpleNew(1, dims, NPY_INT32);
    PyObject *visits = PyArray_SimpleNew(1, dims, NPY_INT32);
    PyObject *q_out = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (!actions || !visits || !q_out) {
        Py_XDECREF(actions); Py_XDECREF(visits); Py_XDECREF(q_out);
        return NULL;
    }

    int32_t *a_data = (int32_t *)PyArray_DATA((PyArrayObject *)actions);
    int32_t *v_data = (int32_t *)PyArray_DATA((PyArrayObject *)visits);
    double *q_data = (double *)PyArray_DATA((PyArrayObject *)q_out);

    for (int32_t i = 0; i < n_ch; i++) {
        int32_t cid = t->child_node[off + i];
        a_data[i] = t->child_action[off + i];
        v_data[i] = t->N[cid];
        if (t->N[cid] > 0) {
            q_data[i] = -t->W[cid] / (double)t->N[cid];
        } else {
            q_data[i] = default_q;
        }
    }

    PyObject *result = PyTuple_Pack(3, actions, visits, q_out);
    Py_DECREF(actions);
    Py_DECREF(visits);
    Py_DECREF(q_out);
    return result;
}


/* node_q(node_id) -> float */
static PyObject *MCTSTree_node_q(MCTSTreeObject *self, PyObject *args) {
    int node_id;
    if (!PyArg_ParseTuple(args, "i", &node_id))
        return NULL;

    TreeData *t = &self->tree;
    double q = (t->N[node_id] > 0) ? (t->W[node_id] / (double)t->N[node_id]) : 0.0;
    return PyFloat_FromDouble(q);
}


/* is_expanded(node_id) -> bool */
static PyObject *MCTSTree_is_expanded(MCTSTreeObject *self, PyObject *args) {
    int node_id;
    if (!PyArg_ParseTuple(args, "i", &node_id))
        return NULL;
    if (node_id < 0 || node_id >= self->tree.node_count)
        Py_RETURN_FALSE;
    return PyBool_FromLong(self->tree.expanded[node_id]);
}


/* node_count() -> int */
static PyObject *MCTSTree_node_count(MCTSTreeObject *self, PyObject *Py_UNUSED(args)) {
    return PyLong_FromLong(self->tree.node_count);
}


/* reset() -> None  (clear tree for reuse) */
static PyObject *MCTSTree_reset(MCTSTreeObject *self, PyObject *Py_UNUSED(args)) {
    self->tree.node_count = 0;
    self->tree.child_count = 0;
    /* Invalidate CBoard cache */
    if (self->tree.cb_valid && self->tree.cb_cache_cap > 0)
        memset(self->tree.cb_valid, 0, self->tree.cb_cache_cap * sizeof(int8_t));
    /* Clear transposition hash table to prevent stale lookups */
    if (self->tree.hash_table && self->tree.ht_cap > 0)
        memset(self->tree.hash_table, -1, self->tree.ht_cap * sizeof(int32_t));
    Py_RETURN_NONE;
}


/*
 * expand_from_logits(node_id, legal_indices_int32, logits_float32_4672)
 *   -> None
 *
 * Takes the full 4672-logit policy vector, extracts values at legal indices,
 * computes softmax, and expands the node — all in C.
 * Eliminates per-leaf Python _softmax_legal + astype + expand overhead.
 */
static PyObject *MCTSTree_expand_from_logits(MCTSTreeObject *self, PyObject *args) {
    int node_id;
    PyObject *legal_obj, *logits_obj;
    if (!PyArg_ParseTuple(args, "iOO", &node_id, &legal_obj, &logits_obj))
        return NULL;

    PyArrayObject *legal_arr = FROMANY_1D(legal_obj, NPY_INT32);
    PyArrayObject *logits_arr = FROMANY_1D(logits_obj, NPY_FLOAT32);

    if (!legal_arr || !logits_arr) {
        Py_XDECREF(legal_arr);
        Py_XDECREF(logits_arr);
        return NULL;
    }

    int32_t n_legal = (int32_t)PyArray_SIZE(legal_arr);
    const int32_t *legal = (const int32_t *)PyArray_DATA(legal_arr);
    const float *logits = (const float *)PyArray_DATA(logits_arr);

    if (n_legal <= 0) {
        self->tree.expanded[node_id] = 1;
        Py_DECREF(legal_arr);
        Py_DECREF(logits_arr);
        Py_RETURN_NONE;
    }

    /* Extract legal logits and compute softmax in float64 */
    double priors_stack[256];
    double *priors = (n_legal <= 256) ? priors_stack : (double *)malloc(n_legal * sizeof(double));
    if (!priors) {
        Py_DECREF(legal_arr);
        Py_DECREF(logits_arr);
        return PyErr_NoMemory();
    }

    for (int32_t i = 0; i < n_legal; i++)
        priors[i] = (double)logits[legal[i]];

    softmax_inplace(priors, n_legal);

    if (tree_expand(&self->tree, (int32_t)node_id, legal, priors, n_legal) < 0) {
        if (priors != priors_stack) free(priors);
        Py_DECREF(legal_arr);
        Py_DECREF(logits_arr);
        return PyErr_NoMemory();
    }

    if (priors != priors_stack) free(priors);
    Py_DECREF(legal_arr);
    Py_DECREF(logits_arr);
    Py_RETURN_NONE;
}


/*
 * batch_wdl_to_q(wdl_logits_float32_Bx3) -> float64 array of Q values
 *
 * Vectorized WDL logits → Q ∈ [-1,1] for a batch of positions.
 * Eliminates 2048 individual _value_scalar_from_wdl_logits Python calls.
 */
static PyObject *MCTSTree_batch_wdl_to_q(MCTSTreeObject *self, PyObject *args) {
    (void)self;  /* static method, doesn't use self */
    PyObject *wdl_obj;
    if (!PyArg_ParseTuple(args, "O", &wdl_obj))
        return NULL;

    PyArrayObject *wdl_arr = FROMANY_2D(wdl_obj, NPY_FLOAT32);
    if (!wdl_arr) return NULL;

    npy_intp batch = PyArray_DIM(wdl_arr, 0);
    if (PyArray_DIM(wdl_arr, 1) != 3) {
        PyErr_SetString(PyExc_ValueError, "wdl must have shape (B, 3)");
        Py_DECREF(wdl_arr);
        return NULL;
    }

    npy_intp out_dims[1] = {batch};
    PyObject *out = PyArray_SimpleNew(1, out_dims, NPY_FLOAT64);
    if (!out) {
        Py_DECREF(wdl_arr);
        return NULL;
    }

    const float *wdl = (const float *)PyArray_DATA(wdl_arr);
    double *q = (double *)PyArray_DATA((PyArrayObject *)out);

    for (npy_intp i = 0; i < batch; i++) {
        q[i] = wdl_logits_to_q((double)wdl[i * 3 + 0],
                               (double)wdl[i * 3 + 1],
                               (double)wdl[i * 3 + 2]);
    }

    Py_DECREF(wdl_arr);
    return out;
}


/*
 * Encode a CBoard into 146 float32 planes (112 LC0 + 34 features).
 * Writes into a pre-allocated buffer — no Python/numpy allocation.
 */
static void cboard_encode_146_into(const CBoard *b, float * restrict out) {
    /* Targeted zeroing instead of full 146-plane memset.
     * Planes 0-11: overwritten by bitboard_to_plane (handles zero BBs internally)
     * Planes 12..12+hist*12-1: overwritten by history encoder
     * Planes 12+hist*12..95: unused history — need zeroing
     * Planes 96-102,111: overwritten by fill_lc0_112
     * Planes 103-110: repetition planes — need zeroing (only 103 may be set)
     * Planes 112-145: feature planes — need zeroing (feat_bb_to_plane adds 1.0f) */
    int hist = b->hist_len < CBOARD_HISTORY_MAX ? b->hist_len : CBOARD_HISTORY_MAX;
    int first_unused_hist = (1 + hist) * 12;  /* plane index of first unused history */
    if (first_unused_hist < 96) {
        memset(out + first_unused_hist * 64, 0,
               (96 - first_unused_hist) * 64 * sizeof(float));
    }
    /* Repetition planes 103-110 (8 planes) */
    memset(out + 103 * 64, 0, 8 * 64 * sizeof(float));
    /* Feature planes 112-145 (34 planes) */
    memset(out + 112 * 64, 0, 34 * 64 * sizeof(float));

    cboard_fill_lc0_112(b, out);
    cboard_compute_features_34(b, out + 112 * 64);
}




/*
 * start_gumbel_sims(root_cboards, root_ids, remaining_per_board,
 *                   gumbels_per_board, root_priors, budget_remaining,
 *                   root_qs, c_scale, c_visit, c_puct, fpu_reduction,
 *                   full_tree, enc_buf[, nn_cache])
 *
 * Initializes the gumbel simulation state machine and runs until
 * the first batch of positions needs GPU eval.
 * Returns n_leaves (int) or None if simulation completed immediately.
 */
static PyObject *MCTSTree_start_gumbel_sims(MCTSTreeObject *self, PyObject *args) {
    PyObject *root_cbs_list, *root_ids_obj, *remaining_list, *gumbels_list;
    PyObject *priors_list, *budget_obj, *root_qs_obj, *enc_buf_obj;
    double c_scale, c_visit, c_puct, fpu_reduction;
    int full_tree;
    PyObject *nn_cache_obj = NULL;

    if (!PyArg_ParseTuple(args, "OOOOOOOddddpO|O",
                          &root_cbs_list, &root_ids_obj, &remaining_list,
                          &gumbels_list, &priors_list, &budget_obj, &root_qs_obj,
                          &c_scale, &c_visit, &c_puct, &fpu_reduction, &full_tree,
                          &enc_buf_obj, &nn_cache_obj))
        return NULL;

    int32_t n_boards = (int32_t)PyList_Size(root_cbs_list);

    /* Parse numpy arrays */
    PyArrayObject *root_ids_arr = FROMANY_1D(root_ids_obj, NPY_INT32);
    PyArrayObject *budget_arr = FROMANY_1D(budget_obj, NPY_INT32);
    PyArrayObject *root_qs_arr = FROMANY_1D(root_qs_obj, NPY_FLOAT64);
    PyArrayObject *enc_arr = FROMANY_4D_RW(enc_buf_obj, NPY_FLOAT32);

    if (!root_ids_arr || !budget_arr || !root_qs_arr || !enc_arr) {
        Py_XDECREF(root_ids_arr); Py_XDECREF(budget_arr);
        Py_XDECREF(root_qs_arr); Py_XDECREF(enc_arr);
        return NULL;
    }

    /* Free previous sim state */
    GumbelSimState *g = &self->gsim;
    gss_free(g);

    /* Initialize */
    g->n_boards = n_boards;
    g->sel.c_scale = c_scale;
    g->sel.c_visit = c_visit;
    g->sel.c_puct = c_puct;
    g->sel.fpu_reduction = fpu_reduction;
    g->sel.full_tree = full_tree;
    g->allocated = 1;

    g->root_ids = (int32_t *)malloc(n_boards * sizeof(int32_t));
    g->budget_remaining = (int32_t *)malloc(n_boards * sizeof(int32_t));
    g->root_qs = (double *)malloc(n_boards * sizeof(double));
    g->root_cboards = (CBoard *)malloc(n_boards * sizeof(CBoard));
    g->root_priors = (double *)calloc((size_t)n_boards * GSS_POLICY_SIZE, sizeof(double));
    g->gumbels = (double *)calloc((size_t)n_boards * GSS_POLICY_SIZE, sizeof(double));
    g->cands_offset = (int32_t *)malloc(n_boards * sizeof(int32_t));
    g->cands_count = (int32_t *)malloc(n_boards * sizeof(int32_t));
    g->visits_per_action = (int32_t *)calloc(n_boards, sizeof(int32_t));
    g->active = (int32_t *)malloc(n_boards * sizeof(int32_t));
    g->q_board_idx = NULL;
    g->q_root_ids = NULL;
    g->q_forced = NULL;
    g->q_cap = 0;

    if (!g->root_ids || !g->budget_remaining || !g->root_qs || !g->root_cboards ||
        !g->root_priors || !g->gumbels || !g->cands_offset || !g->cands_count ||
        !g->visits_per_action || !g->active) {
        gss_free(g);
        Py_DECREF(root_ids_arr); Py_DECREF(budget_arr);
        Py_DECREF(root_qs_arr); Py_DECREF(enc_arr);
        return PyErr_NoMemory();
    }

    memcpy(g->root_ids, PyArray_DATA(root_ids_arr), n_boards * sizeof(int32_t));
    memcpy(g->budget_remaining, PyArray_DATA(budget_arr), n_boards * sizeof(int32_t));
    memcpy(g->root_qs, PyArray_DATA(root_qs_arr), n_boards * sizeof(double));
    for (int32_t i = 0; i < n_boards; i++) {
        PyCBoard *pycb = (PyCBoard *)PyList_GET_ITEM(root_cbs_list, i);
        g->root_cboards[i] = pycb->board;
    }

    for (int32_t i = 0; i < n_boards; i++) {
        PyArrayObject *pri = FROMANY_1D(PyList_GET_ITEM(priors_list, i), NPY_FLOAT64);
        if (!pri) {
            gss_free(g);
            Py_DECREF(root_ids_arr); Py_DECREF(budget_arr);
            Py_DECREF(root_qs_arr); Py_DECREF(enc_arr);
            return NULL;
        }
        if (PyArray_DIM(pri, 0) < GSS_POLICY_SIZE) {
            Py_DECREF(pri);
            gss_free(g);
            Py_DECREF(root_ids_arr); Py_DECREF(budget_arr);
            Py_DECREF(root_qs_arr); Py_DECREF(enc_arr);
            PyErr_SetString(PyExc_ValueError, "priors array must have at least GSS_POLICY_SIZE elements");
            return NULL;
        }
        memcpy(g->root_priors + i * GSS_POLICY_SIZE, PyArray_DATA(pri),
               GSS_POLICY_SIZE * sizeof(double));
        Py_DECREF(pri);
    }

    for (int32_t i = 0; i < n_boards; i++) {
        PyObject *item = PyList_GET_ITEM(gumbels_list, i);
        if (item == Py_None) continue;
        PyArrayObject *garr = FROMANY_1D(item, NPY_FLOAT64);
        if (!garr) {
            gss_free(g);
            Py_DECREF(root_ids_arr); Py_DECREF(budget_arr);
            Py_DECREF(root_qs_arr); Py_DECREF(enc_arr);
            return NULL;
        }
        if (PyArray_DIM(garr, 0) < GSS_POLICY_SIZE) {
            Py_DECREF(garr);
            gss_free(g);
            Py_DECREF(root_ids_arr); Py_DECREF(budget_arr);
            Py_DECREF(root_qs_arr); Py_DECREF(enc_arr);
            PyErr_SetString(PyExc_ValueError, "gumbels array must have at least GSS_POLICY_SIZE elements");
            return NULL;
        }
        memcpy(g->gumbels + i * GSS_POLICY_SIZE, PyArray_DATA(garr),
               GSS_POLICY_SIZE * sizeof(double));
        Py_DECREF(garr);
    }

    int32_t total_cands = 0;
    for (int32_t i = 0; i < n_boards; i++) {
        PyObject *item = PyList_GET_ITEM(remaining_list, i);
        if (item == Py_None) {
            g->cands_offset[i] = total_cands;
            g->cands_count[i] = 0;
        } else {
            Py_ssize_t len = PyList_Size(item);
            g->cands_offset[i] = total_cands;
            g->cands_count[i] = (int32_t)len;
            total_cands += (int32_t)len;
        }
    }
    g->cands_flat_cap = total_cands > 0 ? total_cands : 1;
    g->cands_flat = (int32_t *)malloc(g->cands_flat_cap * sizeof(int32_t));
    if (!g->cands_flat) {
        gss_free(g);
        Py_DECREF(root_ids_arr); Py_DECREF(budget_arr);
        Py_DECREF(root_qs_arr); Py_DECREF(enc_arr);
        return PyErr_NoMemory();
    }
    for (int32_t i = 0; i < n_boards; i++) {
        PyObject *item = PyList_GET_ITEM(remaining_list, i);
        if (item == Py_None) continue;
        int32_t off = g->cands_offset[i];
        for (Py_ssize_t j = 0; j < PyList_Size(item); j++) {
            long v = PyLong_AsLong(PyList_GET_ITEM(item, j));
            if (v == -1 && PyErr_Occurred()) {
                gss_free(g);
                Py_DECREF(root_ids_arr); Py_DECREF(budget_arr);
                Py_DECREF(root_qs_arr); Py_DECREF(enc_arr);
                return NULL;
            }
            g->cands_flat[off + j] = (int32_t)v;
        }
    }

    /* Encoding buffer (strong ref to keep alive) */
    g->enc_data = (float *)PyArray_DATA(enc_arr);
    g->enc_capacity = (int32_t)PyArray_DIM(enc_arr, 0);
    Py_INCREF(enc_arr);
    g->enc_arr_ref = (PyObject *)enc_arr;

    /* NNCache (strong ref to keep alive) */
    g->nncache = NULL;
    g->nncache_ref = NULL;
    if (nn_cache_obj != NULL && nn_cache_obj != Py_None &&
        Py_TYPE(nn_cache_obj) == &PyNNCacheType) {
        g->nncache = &((PyNNCacheObject *)nn_cache_obj)->cache;
        Py_INCREF(nn_cache_obj);
        g->nncache_ref = nn_cache_obj;
    }

    /* Seed CBoard cache */
    tree_ensure_cb_cache(&self->tree, self->tree.node_count);
    for (int32_t i = 0; i < n_boards; i++) {
        int32_t rid = g->root_ids[i];
        if (rid >= 0 && rid < self->tree.cb_cache_cap) {
            self->tree.cb_cache[rid] = g->root_cboards[i];
            self->tree.cb_valid[rid] = 1;
        }
    }

    Py_DECREF(root_ids_arr);
    Py_DECREF(budget_arr);
    Py_DECREF(root_qs_arr);
    Py_DECREF(enc_arr);  /* PyArray_FROMANY ref; strong ref held in g->enc_arr_ref */

    /* Reset stored state */
    StoredPrepState *s = &self->stored;
    s->n_leaves = 0;
    s->n_terminals = 0;
    s->legal_flat_used = 0;
    s->path_flat_used = 0;
    s->term_path_flat_used = 0;

    /* Begin first halving round */
    g->phase = 1;
    if (gss_begin_round(g) == 0) {
        g->phase = 2;
        Py_RETURN_NONE;
    }

    /* Run simulation until GPU eval needed */
    int32_t n_leaves_start;
    Py_BEGIN_ALLOW_THREADS
    n_leaves_start = gss_step(&self->tree, s, g, g->enc_data);
    Py_END_ALLOW_THREADS

    if (n_leaves_start == 0) {
        g->phase = 2;
        Py_RETURN_NONE;
    }

    return PyLong_FromLong(n_leaves_start);
}


/*
 * continue_gumbel_sims(pol_logits, wdl_logits)
 *
 * Feed GPU eval results back and continue simulation.
 * Returns n_leaves (int) for next GPU eval, or None if done.
 */
static PyObject *MCTSTree_continue_gumbel_sims(MCTSTreeObject *self, PyObject *args) {
    PyObject *pol_obj, *wdl_obj;

    if (!PyArg_ParseTuple(args, "OO", &pol_obj, &wdl_obj))
        return NULL;

    PyArrayObject *pol_arr = FROMANY_2D(pol_obj, NPY_FLOAT32);
    PyArrayObject *wdl_arr = FROMANY_2D(wdl_obj, NPY_FLOAT32);

    if (!pol_arr || !wdl_arr) {
        Py_XDECREF(pol_arr); Py_XDECREF(wdl_arr);
        return NULL;
    }

    GumbelSimState *g = &self->gsim;
    StoredPrepState *s = &self->stored;

    if (g->phase != 1) {
        PyErr_SetString(PyExc_RuntimeError, "continue_gumbel_sims called but simulation not in needs_eval state");
        Py_DECREF(pol_arr); Py_DECREF(wdl_arr);
        return NULL;
    }

    const float *pol_data = (const float *)PyArray_DATA(pol_arr);
    const float *wdl_data = (const float *)PyArray_DATA(wdl_arr);

    /* Finish batch (expand + backprop) and continue simulation */
    Py_BEGIN_ALLOW_THREADS

    gss_finish_batch(&self->tree, s, g->nncache, pol_data, wdl_data);

    /* Reset stored state for next batch */
    s->n_leaves = 0;
    s->n_terminals = 0;
    s->legal_flat_used = 0;
    s->path_flat_used = 0;
    s->term_path_flat_used = 0;

    Py_END_ALLOW_THREADS

    Py_DECREF(pol_arr);
    Py_DECREF(wdl_arr);

    /* Continue simulation */
    int32_t n_leaves_cont;
    Py_BEGIN_ALLOW_THREADS
    n_leaves_cont = gss_step(&self->tree, s, g, g->enc_data);
    Py_END_ALLOW_THREADS

    if (n_leaves_cont == 0) {
        g->phase = 2;
        Py_RETURN_NONE;
    }

    return PyLong_FromLong(n_leaves_cont);
}


/*
 * get_gumbel_remaining() -> list[list[int]]
 *
 * After simulation completes, get the remaining candidates per board.
 */
static PyObject *MCTSTree_get_gumbel_remaining(MCTSTreeObject *self, PyObject *Py_UNUSED(args)) {
    GumbelSimState *g = &self->gsim;
    if (!g->allocated) {
        PyErr_SetString(PyExc_RuntimeError, "No gumbel simulation state");
        return NULL;
    }
    PyObject *result = PyList_New(g->n_boards);
    for (int32_t i = 0; i < g->n_boards; i++) {
        int32_t off = g->cands_offset[i];
        int32_t cnt = g->cands_count[i];
        PyObject *lst = PyList_New(cnt);
        for (int32_t j = 0; j < cnt; j++) {
            PyList_SET_ITEM(lst, j, PyLong_FromLong(g->cands_flat[off + j]));
        }
        PyList_SET_ITEM(result, i, lst);
    }
    return result;
}


/* find_child(node_id, action) -> child_node_id or -1 */
static PyObject *MCTSTree_find_child(MCTSTreeObject *self, PyObject *args) {
    int32_t node_id, action;
    if (!PyArg_ParseTuple(args, "ii", &node_id, &action))
        return NULL;
    TreeData *t = &self->tree;
    if (node_id < 0 || node_id >= t->node_count)
        return PyLong_FromLong(-1);
    int32_t n_ch = t->num_children[node_id];
    int32_t off = t->children_offset[node_id];
    for (int32_t i = 0; i < n_ch; i++) {
        if (t->child_action[off + i] == action)
            return PyLong_FromLong(t->child_node[off + i]);
    }
    return PyLong_FromLong(-1);
}

static PyMethodDef MCTSTree_methods[] = {
    {"find_child", (PyCFunction)MCTSTree_find_child, METH_VARARGS,
     "find_child(node_id, action) -> child_node_id or -1"},
    {"add_root", (PyCFunction)MCTSTree_add_root, METH_VARARGS,
     "add_root(N, W) -> int node_id"},
    {"expand", (PyCFunction)MCTSTree_expand, METH_VARARGS,
     "expand(node_id, actions_int32, priors_float64) -> None"},
    {"expand_from_logits", (PyCFunction)MCTSTree_expand_from_logits, METH_VARARGS,
     "expand_from_logits(node_id, legal_int32, logits_float32) -> None"},
    {"batch_wdl_to_q", (PyCFunction)MCTSTree_batch_wdl_to_q, METH_VARARGS,
     "batch_wdl_to_q(wdl_float32_Bx3) -> float64 Q values"},
    {"select_leaves", (PyCFunction)MCTSTree_select_leaves, METH_VARARGS,
     "select_leaves(root_ids_int32, c_puct, fpu_root, fpu_tree) -> [(leaf_id, action_path, node_path)]"},
    {"backprop", (PyCFunction)MCTSTree_backprop, METH_VARARGS,
     "backprop(node_path_int32, value) -> None"},
    {"backprop_many", (PyCFunction)MCTSTree_backprop_many, METH_VARARGS,
     "backprop_many([node_paths], [values]) -> None"},
    {"start_gumbel_sims", (PyCFunction)MCTSTree_start_gumbel_sims, METH_VARARGS,
     "start_gumbel_sims(...) -> n_leaves or None. Start gumbel simulation state machine."},
    {"continue_gumbel_sims", (PyCFunction)MCTSTree_continue_gumbel_sims, METH_VARARGS,
     "continue_gumbel_sims(pol, wdl) -> n_leaves or None. Feed GPU results, continue sim."},
    {"get_gumbel_remaining", (PyCFunction)MCTSTree_get_gumbel_remaining, METH_NOARGS,
     "get_gumbel_remaining() -> list[list[int]]. Get remaining candidates after sim."},
    {"get_children_visits", (PyCFunction)MCTSTree_get_children_visits, METH_VARARGS,
     "get_children_visits(node_id) -> (actions_int32, visits_int32)"},
    {"get_children_q", (PyCFunction)MCTSTree_get_children_q, METH_VARARGS,
     "get_children_q(node_id, default_q) -> (actions_int32, visits_int32, q_float64)"},
    {"node_q", (PyCFunction)MCTSTree_node_q, METH_VARARGS,
     "node_q(node_id) -> float"},
    {"is_expanded", (PyCFunction)MCTSTree_is_expanded, METH_VARARGS,
     "is_expanded(node_id) -> bool"},
    {"node_count", (PyCFunction)MCTSTree_node_count, METH_NOARGS,
     "node_count() -> int"},
    {"reset", (PyCFunction)MCTSTree_reset, METH_NOARGS,
     "reset() -> None"},
    {NULL}
};


static PyTypeObject MCTSTreeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_mcts_tree.MCTSTree",
    .tp_doc = "Array-based MCTS tree for fast select/expand/backprop.",
    .tp_basicsize = sizeof(MCTSTreeObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)MCTSTree_init,
    .tp_dealloc = (destructor)MCTSTree_dealloc,
    .tp_methods = MCTSTree_methods,
};


/* ================================================================
 * Module definition
 * ================================================================ */

/*
 * batch_process_ply(cboards_list, pol_logits, wdl_logits, actions, values,
 *                   mcts_probs,
 *                   df_enabled, df_q_weight, df_pol_scale, df_min, df_slope)
 *   -> (sample_x, sample_probs, sample_wdl_net, sample_wdl_search,
 *       sample_priority, sample_keep_prob, sample_legal_mask,
 *       sample_ply, sample_pov, game_over)
 *
 * Fused per-ply processing: legal mask, masked softmax, KL divergence,
 * Q-surprise, search WDL, push move, game-over check, position encoding.
 * Runs with GIL released for thread parallelism.
 */
static PyObject *py_batch_process_ply(PyObject *self, PyObject *args) {
    PyObject *cboards_list;
    PyObject *pol_obj, *wdl_obj, *actions_obj, *values_obj;
    PyObject *mcts_probs_obj;
    int df_enabled;
    double df_q_weight, df_pol_scale, df_min, df_slope;

    if (!PyArg_ParseTuple(args, "OOOOOOidddd",
                          &cboards_list, &pol_obj, &wdl_obj, &actions_obj,
                          &values_obj, &mcts_probs_obj,
                          &df_enabled, &df_q_weight, &df_pol_scale, &df_min, &df_slope))
        return NULL;

    PyArrayObject *pol_arr = FROMANY_2D(pol_obj, NPY_FLOAT32);
    PyArrayObject *wdl_arr = FROMANY_2D(wdl_obj, NPY_FLOAT32);
    PyArrayObject *act_arr = FROMANY_1D(actions_obj, NPY_INT32);
    PyArrayObject *val_arr = FROMANY_1D(values_obj, NPY_FLOAT64);
    PyArrayObject *probs_arr = FROMANY_2D(mcts_probs_obj, NPY_FLOAT32);

    if (!pol_arr || !wdl_arr || !act_arr || !val_arr || !probs_arr) {
        Py_XDECREF(pol_arr); Py_XDECREF(wdl_arr); Py_XDECREF(act_arr);
        Py_XDECREF(val_arr); Py_XDECREF(probs_arr);
        return NULL;
    }

    /* Heap pointer declared first so the common fail: label can free()
     * unconditionally (free(NULL) is a no-op) even if we goto fail
     * before allocation. */
    CBoard *boards = NULL;

    int32_t n = (int32_t)PyArray_DIM(pol_arr, 0);
    if (PyList_Size(cboards_list) < n) {
        PyErr_SetString(PyExc_ValueError, "cboards_list too short");
        goto fail;
    }
    /* Validate companion arrays. pol_arr is the source of truth for n;
     * all others (and their trailing dims for 2D) must match. */
    if (PyArray_DIM(pol_arr, 1) != 4672) {
        PyErr_SetString(PyExc_ValueError, "pol must have shape (n, 4672)");
        goto fail;
    }
    if (PyArray_DIM(wdl_arr, 0) < n || PyArray_DIM(wdl_arr, 1) != 3) {
        PyErr_SetString(PyExc_ValueError, "wdl must have shape (>=n, 3)");
        goto fail;
    }
    if (PyArray_DIM(act_arr, 0) < n) {
        PyErr_SetString(PyExc_ValueError, "actions too short");
        goto fail;
    }
    if (PyArray_DIM(val_arr, 0) < n) {
        PyErr_SetString(PyExc_ValueError, "values too short");
        goto fail;
    }
    if (PyArray_DIM(probs_arr, 0) < n || PyArray_DIM(probs_arr, 1) != 4672) {
        PyErr_SetString(PyExc_ValueError, "mcts_probs must have shape (>=n, 4672)");
        goto fail;
    }

    /* Extract data pointers */
    const float *pol_data = (const float *)PyArray_DATA(pol_arr);
    const float *wdl_data = (const float *)PyArray_DATA(wdl_arr);
    const int32_t *actions = (const int32_t *)PyArray_DATA(act_arr);
    const double *values = (const double *)PyArray_DATA(val_arr);
    const float *mcts_probs = (const float *)PyArray_DATA(probs_arr);

    boards = (CBoard *)malloc(n * sizeof(CBoard));
    if (!boards) { PyErr_NoMemory(); goto fail; }
    if (extract_cboards(cboards_list, n, boards, NULL) < 0) goto fail;

    /* Allocate output arrays */
    npy_intp dims_x[4] = {n, 146, 8, 8};
    npy_intp dims_p[2] = {n, 4672};
    npy_intp dims_w[2] = {n, 3};
    npy_intp dims_n[1] = {n};
    npy_intp dims_m[2] = {n, 4672};

    PyArrayObject *out_x = (PyArrayObject *)PyArray_ZEROS(4, dims_x, NPY_FLOAT32, 0);
    PyArrayObject *out_probs = (PyArrayObject *)PyArray_SimpleNew(2, dims_p, NPY_FLOAT32);
    PyArrayObject *out_wdl_net = (PyArrayObject *)PyArray_SimpleNew(2, dims_w, NPY_FLOAT32);
    PyArrayObject *out_wdl_search = (PyArrayObject *)PyArray_SimpleNew(2, dims_w, NPY_FLOAT32);
    PyArrayObject *out_priority = (PyArrayObject *)PyArray_SimpleNew(1, dims_n, NPY_FLOAT32);
    PyArrayObject *out_keep = (PyArrayObject *)PyArray_SimpleNew(1, dims_n, NPY_FLOAT32);
    PyArrayObject *out_mask = (PyArrayObject *)PyArray_ZEROS(2, dims_m, NPY_UINT8, 0);
    PyArrayObject *out_ply = (PyArrayObject *)PyArray_SimpleNew(1, dims_n, NPY_INT32);
    PyArrayObject *out_pov = (PyArrayObject *)PyArray_SimpleNew(1, dims_n, NPY_INT32);
    PyArrayObject *out_over = (PyArrayObject *)PyArray_SimpleNew(1, dims_n, NPY_INT32);

    if (!out_x || !out_probs || !out_wdl_net || !out_wdl_search ||
        !out_priority || !out_keep || !out_mask || !out_ply || !out_pov || !out_over) {
        Py_XDECREF(out_x); Py_XDECREF(out_probs); Py_XDECREF(out_wdl_net);
        Py_XDECREF(out_wdl_search); Py_XDECREF(out_priority); Py_XDECREF(out_keep);
        Py_XDECREF(out_mask); Py_XDECREF(out_ply); Py_XDECREF(out_pov); Py_XDECREF(out_over);
        goto fail;
    }

    float *x_data = (float *)PyArray_DATA(out_x);
    float *probs_out = (float *)PyArray_DATA(out_probs);
    float *wdl_net_out = (float *)PyArray_DATA(out_wdl_net);
    float *wdl_search_out = (float *)PyArray_DATA(out_wdl_search);
    float *priority_out = (float *)PyArray_DATA(out_priority);
    float *keep_out = (float *)PyArray_DATA(out_keep);
    uint8_t *mask_out = (uint8_t *)PyArray_DATA(out_mask);
    int32_t *ply_out = (int32_t *)PyArray_DATA(out_ply);
    int32_t *pov_out = (int32_t *)PyArray_DATA(out_pov);
    int32_t *over_out = (int32_t *)PyArray_DATA(out_over);

    /* ── Release GIL for the main computation loop ── */
    Py_BEGIN_ALLOW_THREADS

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(n >= 32)
#endif
    for (int32_t i = 0; i < n; i++) {
        CBoard *cb = &boards[i];
        const float *pol = pol_data + i * 4672;
        const float *wdl = wdl_data + i * 3;
        const float *mprobs = mcts_probs + i * 4672;
        int32_t action = actions[i];
        double value = values[i];

        /* Ply index and POV */
        ply_out[i] = (int32_t)cb->hist_len;
        pov_out[i] = (int32_t)cb->turn;

        /* Legal mask */
        int legal_indices[256];
        int n_legal = cboard_legal_move_indices(cb, legal_indices, /*sorted=*/0);

        uint8_t *mask = mask_out + i * 4672;
        for (int j = 0; j < n_legal; j++)
            mask[legal_indices[j]] = 1;

        /* Masked softmax + KL. Operating on compact legal_logits instead of
         * the full 4672 drops ~3 full-width passes (~14k ops) to ~3 passes
         * of n_legal (~100 ops) — ~130× fewer iterations for typical ~35
         * legal moves. Illegal indices stay zero since the KL accumulator
         * would skip them anyway. */
        float legal_logits[256];
        float max_val = -1e30f;
        for (int j = 0; j < n_legal; j++) {
            float v = pol[legal_indices[j]];
            legal_logits[j] = v;
            if (v > max_val) max_val = v;
        }
        float sum = 0.0f;
        for (int j = 0; j < n_legal; j++) {
            legal_logits[j] = expf(legal_logits[j] - max_val);
            sum += legal_logits[j];
        }
        int uniform = !(sum > 0.0f) && n_legal > 0;
        float inv_sum = uniform ? (1.0f / (float)n_legal) : (sum > 0.0f ? 1.0f / sum : 0.0f);

        float kl = 0.0f;
        for (int j = 0; j < n_legal; j++) {
            float p = uniform ? inv_sum : (legal_logits[j] * inv_sum);
            float mp = mprobs[legal_indices[j]];
            if (p > 1e-12f && mp > 1e-12f)
                kl += p * (logf(p) - logf(mp));
        }

        /* Q-surprise */
        float wdl_w = wdl[0], wdl_d = wdl[1], wdl_l = wdl[2];
        /* Softmax on WDL logits */
        float wdl_max = wdl_w; if (wdl_d > wdl_max) wdl_max = wdl_d; if (wdl_l > wdl_max) wdl_max = wdl_l;
        float ew = expf(wdl_w - wdl_max), ed = expf(wdl_d - wdl_max), el = expf(wdl_l - wdl_max);
        float ws = ew + ed + el;
        float wdl_net[3];
        if (ws > 0.0f) { wdl_net[0] = ew/ws; wdl_net[1] = ed/ws; wdl_net[2] = el/ws; }
        else { wdl_net[0] = 0.0f; wdl_net[1] = 1.0f; wdl_net[2] = 0.0f; }

        float orig_q = wdl_net[0] - wdl_net[2];
        float best_q = (float)value;
        float q_surprise = fabsf(best_q - orig_q);

        /* Difficulty / keep_prob */
        float difficulty = q_surprise * (float)df_q_weight + kl * (float)df_pol_scale;
        if (!isfinite(difficulty)) difficulty = 1.0f;
        float keep_prob = 1.0f;
        if (df_enabled)
            keep_prob = fmaxf((float)df_min, fminf(1.0f, difficulty * (float)df_slope));

        priority_out[i] = difficulty;
        keep_out[i] = keep_prob;

        /* Store net WDL */
        wdl_net_out[i * 3 + 0] = wdl_net[0];
        wdl_net_out[i * 3 + 1] = wdl_net[1];
        wdl_net_out[i * 3 + 2] = wdl_net[2];

        /* Search WDL estimate from MCTS value */
        float d_raw = wdl_net[1];
        float rem = fmaxf(0.0f, 1.0f - d_raw);
        float q_clamped = fmaxf(-rem, fminf(rem, best_q));
        float w_search = 0.5f * (rem + q_clamped);
        wdl_search_out[i * 3 + 0] = w_search;
        wdl_search_out[i * 3 + 1] = d_raw;
        wdl_search_out[i * 3 + 2] = rem - w_search;
        if (!isfinite(wdl_search_out[i*3]) || !isfinite(wdl_search_out[i*3+1]) || !isfinite(wdl_search_out[i*3+2])) {
            wdl_search_out[i * 3 + 0] = 0.0f;
            wdl_search_out[i * 3 + 1] = 1.0f;
            wdl_search_out[i * 3 + 2] = 0.0f;
        }

        /* Copy MCTS probs to output */
        memcpy(probs_out + i * 4672, mprobs, 4672 * sizeof(float));

        /* Encode position BEFORE pushing move */
        cboard_encode_146_into(cb, x_data + i * 146 * 64);

        /* Push move */
        cboard_push_index(cb, action);

        /* Game over check */
        over_out[i] = cboard_is_game_over(cb) ? 1 : 0;
    }  /* omp parallel for */

    Py_END_ALLOW_THREADS

    /* Write back CBoard states to Python objects */
    for (int32_t i = 0; i < n; i++) {
        PyCBoard *pcb = (PyCBoard *)PyList_GET_ITEM(cboards_list, i);
        pcb->board = boards[i];
    }
    free(boards);

    Py_DECREF(pol_arr); Py_DECREF(wdl_arr); Py_DECREF(act_arr);
    Py_DECREF(val_arr); Py_DECREF(probs_arr);

    return Py_BuildValue("(NNNNNNNNNN)",
        out_x, out_probs, out_wdl_net, out_wdl_search,
        out_priority, out_keep, out_mask, out_ply, out_pov, out_over);

fail:
    free(boards);   /* NULL-safe */
    Py_XDECREF(pol_arr); Py_XDECREF(wdl_arr); Py_XDECREF(act_arr);
    Py_XDECREF(val_arr); Py_XDECREF(probs_arr);
    return NULL;
}

/* batch_encode_146(cboards_list, out_array)
 * Encode N CBoards into a pre-allocated (N, 146, 8, 8) float32 array.
 * GIL released during encoding for thread parallelism. */
static PyObject *py_batch_encode_146(PyObject *self, PyObject *args) {
    PyObject *cboards_list;
    PyObject *out_obj;

    if (!PyArg_ParseTuple(args, "OO", &cboards_list, &out_obj))
        return NULL;

    if (!PyList_Check(cboards_list)) {
        PyErr_SetString(PyExc_TypeError, "cboards_list must be a list");
        return NULL;
    }

    PyArrayObject *out_arr = FROMANY_4D_RW(out_obj, NPY_FLOAT32);
    if (!out_arr) return NULL;

    int32_t n = (int32_t)PyList_Size(cboards_list);
    if (n <= 0) { Py_DECREF(out_arr); Py_RETURN_NONE; }

    /* Verify output array is large enough */
    if (PyArray_DIM(out_arr, 0) < n || PyArray_DIM(out_arr, 1) != 146 ||
        PyArray_DIM(out_arr, 2) != 8 || PyArray_DIM(out_arr, 3) != 8) {
        Py_DECREF(out_arr);
        PyErr_SetString(PyExc_ValueError,
            "out_array must have shape (>=N, 146, 8, 8)");
        return NULL;
    }

    /* Pre-extract CBoard structs (validated) */
    CBoard *boards = (CBoard *)malloc(n * sizeof(CBoard));
    if (!boards) { Py_DECREF(out_arr); return PyErr_NoMemory(); }
    if (extract_cboards(cboards_list, n, boards, NULL) < 0) {
        free(boards); Py_DECREF(out_arr); return NULL;
    }

    float *out_data = (float *)PyArray_DATA(out_arr);

    Py_BEGIN_ALLOW_THREADS
    /* Zero entire buffer first — cboard_encode_146_into uses targeted zeroing
     * that assumes some planes are pre-zeroed (current piece planes 0-11). */
    memset(out_data, 0, (size_t)n * 146 * 64 * sizeof(float));
    for (int32_t i = 0; i < n; i++) {
        cboard_encode_146_into(&boards[i], out_data + i * 146 * 64);
    }
    Py_END_ALLOW_THREADS

    free(boards);
    Py_DECREF(out_arr);
    Py_RETURN_NONE;
}

/* ================================================================
 * classify_games — classify active games into net / opp groups.
 *
 * Replaces 5 Python list comprehensions + game-over loop in the
 * main selfplay step.  Runs with GIL released.
 *
 * Python signature:
 *   classify_games(cboards: list[CBoard],
 *                  network_color: ndarray[int8],   # 1=WHITE, 0=BLACK
 *                  done: ndarray[int8],            # mutable
 *                  finalized: ndarray[int8],       # read-only
 *                  selfplay_game: ndarray[int8],   # read-only
 *                  max_plies: int)
 *   -> (net_idxs, selfplay_opp_idxs, curriculum_opp_idxs)  int32 arrays
 * ================================================================ */
static PyObject *py_classify_games(PyObject *self, PyObject *args) {
    PyObject *cboards_list;
    PyObject *net_color_obj, *done_obj, *final_obj, *sp_obj;
    int max_plies;

    if (!PyArg_ParseTuple(args, "OOOOOi",
            &cboards_list, &net_color_obj, &done_obj, &final_obj,
            &sp_obj, &max_plies))
        return NULL;

    if (!PyList_Check(cboards_list)) {
        PyErr_SetString(PyExc_TypeError, "cboards must be a list");
        return NULL;
    }

    int32_t n = (int32_t)PyList_Size(cboards_list);
    if (n <= 0)
        return Py_BuildValue("(NNN)",
            PyArray_EMPTY(1, (npy_intp[]){0}, NPY_INT32, 0),
            PyArray_EMPTY(1, (npy_intp[]){0}, NPY_INT32, 0),
            PyArray_EMPTY(1, (npy_intp[]){0}, NPY_INT32, 0));

    PyArrayObject *net_color_arr = FROMANY_1D(net_color_obj, NPY_INT8);
    PyArrayObject *done_arr = FROMANY_1D_RW(done_obj, NPY_INT8);
    PyArrayObject *final_arr = FROMANY_1D(final_obj, NPY_INT8);
    PyArrayObject *sp_arr = FROMANY_1D(sp_obj, NPY_INT8);

    if (!net_color_arr || !done_arr || !final_arr || !sp_arr) {
        Py_XDECREF(net_color_arr); Py_XDECREF(done_arr);
        Py_XDECREF(final_arr); Py_XDECREF(sp_arr);
        return NULL;
    }

    /* Validate companion array lengths match n */
    if (PyArray_DIM(net_color_arr, 0) < n || PyArray_DIM(done_arr, 0) < n ||
        PyArray_DIM(final_arr, 0) < n || PyArray_DIM(sp_arr, 0) < n) {
        Py_DECREF(net_color_arr); Py_DECREF(done_arr);
        Py_DECREF(final_arr); Py_DECREF(sp_arr);
        PyErr_SetString(PyExc_ValueError, "arrays must be at least as long as cboards list");
        return NULL;
    }

    /* Extract CBoard pointers (GIL held here).
     * We store pointers, not copies — safe because the list holds
     * references to all PyCBoard objects, preventing GC. */
    const CBoard **boards = (const CBoard **)malloc(n * sizeof(CBoard *));
    if (!boards) {
        Py_DECREF(net_color_arr); Py_DECREF(done_arr);
        Py_DECREF(final_arr); Py_DECREF(sp_arr);
        return PyErr_NoMemory();
    }
    if (extract_cboards(cboards_list, n, NULL, boards) < 0) {
        free(boards);
        Py_DECREF(net_color_arr); Py_DECREF(done_arr);
        Py_DECREF(final_arr); Py_DECREF(sp_arr);
        return NULL;
    }

    int8_t *net_color = (int8_t *)PyArray_DATA(net_color_arr);
    int8_t *done_data = (int8_t *)PyArray_DATA(done_arr);
    int8_t *final_data = (int8_t *)PyArray_DATA(final_arr);
    int8_t *sp_data = (int8_t *)PyArray_DATA(sp_arr);

    /* Output buffers (worst case: all games in one group) */
    int32_t *net_buf = (int32_t *)malloc(n * sizeof(int32_t));
    int32_t *sp_opp_buf = (int32_t *)malloc(n * sizeof(int32_t));
    int32_t *cur_opp_buf = (int32_t *)malloc(n * sizeof(int32_t));
    if (!net_buf || !sp_opp_buf || !cur_opp_buf) {
        free(boards); free(net_buf); free(sp_opp_buf); free(cur_opp_buf);
        Py_DECREF(net_color_arr); Py_DECREF(done_arr);
        Py_DECREF(final_arr); Py_DECREF(sp_arr);
        return PyErr_NoMemory();
    }

    int32_t n_net = 0, n_sp_opp = 0, n_cur_opp = 0;

    Py_BEGIN_ALLOW_THREADS

    /* 1) Mark games that hit max_plies or are over */
    for (int32_t i = 0; i < n; i++) {
        if (final_data[i]) continue;
        if (!done_data[i]) {
            if (cboard_is_game_over(boards[i]) || boards[i]->ply >= (uint16_t)max_plies)
                done_data[i] = 1;
        }
    }

    /* 2) Classify active (not finalized, not done) games */
    for (int32_t i = 0; i < n; i++) {
        if (final_data[i] || done_data[i]) continue;
        if (boards[i]->turn == net_color[i]) {
            net_buf[n_net++] = i;
        } else {
            if (sp_data[i])
                sp_opp_buf[n_sp_opp++] = i;
            else
                cur_opp_buf[n_cur_opp++] = i;
        }
    }

    Py_END_ALLOW_THREADS

    free(boards);
    Py_DECREF(net_color_arr);
    /* done_arr was modified in-place — keep it alive via caller's reference */
    Py_DECREF(done_arr);
    Py_DECREF(final_arr);
    Py_DECREF(sp_arr);

    /* Build output arrays by copying from temp buffers */
    npy_intp dims_net[1] = {n_net};
    npy_intp dims_sp[1]  = {n_sp_opp};
    npy_intp dims_cur[1] = {n_cur_opp};

    PyArrayObject *net_out = (PyArrayObject *)PyArray_SimpleNew(1, dims_net, NPY_INT32);
    PyArrayObject *sp_out  = (PyArrayObject *)PyArray_SimpleNew(1, dims_sp, NPY_INT32);
    PyArrayObject *cur_out = (PyArrayObject *)PyArray_SimpleNew(1, dims_cur, NPY_INT32);
    if (!net_out || !sp_out || !cur_out) {
        free(net_buf); free(sp_opp_buf); free(cur_opp_buf);
        Py_XDECREF(net_out); Py_XDECREF(sp_out); Py_XDECREF(cur_out);
        return NULL;
    }
    if (n_net > 0) memcpy(PyArray_DATA(net_out), net_buf, n_net * sizeof(int32_t));
    if (n_sp_opp > 0) memcpy(PyArray_DATA(sp_out), sp_opp_buf, n_sp_opp * sizeof(int32_t));
    if (n_cur_opp > 0) memcpy(PyArray_DATA(cur_out), cur_opp_buf, n_cur_opp * sizeof(int32_t));

    free(net_buf); free(sp_opp_buf); free(cur_opp_buf);

    return Py_BuildValue("(NNN)", net_out, sp_out, cur_out);
}


/* ================================================================
 * temperature_resample — apply per-game temperature to MCTS
 * improved policies and resample actions.
 *
 * Runs with GIL released.
 *
 * Python signature:
 *   temperature_resample(probs: ndarray[float32, (N,4672)],
 *                        temps: ndarray[float64, (N,)],
 *                        actions: ndarray[int32, (N,)],  # mutable
 *                        rand_vals: ndarray[float64, (N,)])
 *   -> None
 * ================================================================ */
static PyObject *py_temperature_resample(PyObject *self, PyObject *args) {
    PyObject *probs_obj, *temps_obj, *actions_obj, *rand_obj;

    if (!PyArg_ParseTuple(args, "OOOO",
            &probs_obj, &temps_obj, &actions_obj, &rand_obj))
        return NULL;

    PyArrayObject *probs_arr = FROMANY_2D(probs_obj, NPY_FLOAT32);
    PyArrayObject *temps_arr = FROMANY_1D(temps_obj, NPY_FLOAT64);
    PyArrayObject *actions_arr = FROMANY_1D_RW(actions_obj, NPY_INT32);
    PyArrayObject *rand_arr = FROMANY_1D(rand_obj, NPY_FLOAT64);

    if (!probs_arr || !temps_arr || !actions_arr || !rand_arr) {
        Py_XDECREF(probs_arr); Py_XDECREF(temps_arr);
        Py_XDECREF(actions_arr); Py_XDECREF(rand_arr);
        return NULL;
    }

    int32_t n = (int32_t)PyArray_DIM(probs_arr, 0);
    int32_t policy_size = (int32_t)PyArray_DIM(probs_arr, 1);

    /* Validate companion arrays match n */
    if (PyArray_DIM(temps_arr, 0) < n || PyArray_DIM(actions_arr, 0) < n ||
        PyArray_DIM(rand_arr, 0) < n) {
        Py_DECREF(probs_arr); Py_DECREF(temps_arr);
        Py_DECREF(actions_arr); Py_DECREF(rand_arr);
        PyErr_SetString(PyExc_ValueError, "temps, actions, rand_vals must be at least as long as probs");
        return NULL;
    }

    float *probs_data    = (float *)PyArray_DATA(probs_arr);
    double *temps_data   = (double *)PyArray_DATA(temps_arr);
    int32_t *actions_data = (int32_t *)PyArray_DATA(actions_arr);
    double *rand_data    = (double *)PyArray_DATA(rand_arr);

    Py_BEGIN_ALLOW_THREADS

    /* Compact legal-move buffers — max ~218 legal moves in chess */
    int32_t legal_idx[256];
    double  legal_pw[256];

    for (int32_t i = 0; i < n; i++) {
        double t = temps_data[i];
        if (t == 1.0) continue;

        float *p = probs_data + i * policy_size;

        /* Collect legal moves (p[j] > 0) into compact arrays */
        int32_t n_legal = 0;
        for (int32_t j = 0; j < policy_size; j++) {
            if (p[j] > 0.0f) {
                legal_idx[n_legal] = j;
                legal_pw[n_legal]  = (double)p[j];
                n_legal++;
                if (n_legal >= 256) break;
            }
        }
        if (n_legal == 0) continue;

        if (t <= 0.0) {
            /* Greedy: pick legal move with highest prob */
            int32_t best = 0;
            for (int32_t k = 1; k < n_legal; k++) {
                if (legal_pw[k] > legal_pw[best]) best = k;
            }
            actions_data[i] = legal_idx[best];
            continue;
        }

        /* Apply temperature: pw[k] = pow(p[k], 1/t) */
        double inv_t = 1.0 / t;
        double sum = 0.0;
        for (int32_t k = 0; k < n_legal; k++) {
            legal_pw[k] = pow(legal_pw[k], inv_t);
            sum += legal_pw[k];
        }
        if (sum <= 0.0) continue;

        /* Sample: cumulative search over compact legal moves */
        double threshold = rand_data[i] * sum;
        double cumsum = 0.0;
        int32_t chosen = legal_idx[n_legal - 1]; /* fallback: last legal */
        for (int32_t k = 0; k < n_legal; k++) {
            cumsum += legal_pw[k];
            if (cumsum >= threshold) {
                chosen = legal_idx[k];
                break;
            }
        }
        actions_data[i] = chosen;
    }

    Py_END_ALLOW_THREADS

    Py_DECREF(probs_arr);
    Py_DECREF(temps_arr);
    Py_DECREF(actions_arr);
    Py_DECREF(rand_arr);

    Py_RETURN_NONE;
}


static PyMethodDef module_methods[] = {
    {"batch_process_ply", py_batch_process_ply, METH_VARARGS,
     "batch_process_ply(cboards, pol, wdl, actions, values, probs, "
     "df_enabled, df_q_w, df_pol_s, df_min, df_slope) -> tuple of arrays"},
    {"batch_encode_146", py_batch_encode_146, METH_VARARGS,
     "batch_encode_146(cboards_list, out_array) -> None. "
     "Encode CBoards into pre-allocated (N,146,8,8) float32 array. GIL released."},
    {"classify_games", py_classify_games, METH_VARARGS,
     "classify_games(cboards, net_color, done, finalized, selfplay, max_plies) "
     "-> (net_idxs, selfplay_opp_idxs, curriculum_opp_idxs). GIL released."},
    {"temperature_resample", py_temperature_resample, METH_VARARGS,
     "temperature_resample(probs, temps, actions, rand_vals) -> None. "
     "Apply per-game temperature and resample actions in-place. GIL released."},
    {NULL}
};

static struct PyModuleDef mcts_tree_module = {
    PyModuleDef_HEAD_INIT,
    "_mcts_tree",
    "C-accelerated MCTS tree operations",
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit__mcts_tree(void) {
    import_array();
    cboard_init_all();

    if (PyType_Ready(&MCTSTreeType) < 0)
        return NULL;
    if (PyType_Ready(&PyNNCacheType) < 0)
        return NULL;

    PyObject *m = PyModule_Create(&mcts_tree_module);
    if (!m) return NULL;

    Py_INCREF(&MCTSTreeType);
    if (PyModule_AddObject(m, "MCTSTree", (PyObject *)&MCTSTreeType) < 0) {
        Py_DECREF(&MCTSTreeType);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&PyNNCacheType);
    if (PyModule_AddObject(m, "NNCache", (PyObject *)&PyNNCacheType) < 0) {
        Py_DECREF(&PyNNCacheType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
