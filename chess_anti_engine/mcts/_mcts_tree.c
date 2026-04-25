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
#include <pthread.h>

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
    /* Lifetime counters — not reset on clear(), so stats span restarts of
     * the gumbel sim but are reset-on-process-start. Used to decide whether
     * NNCache earns its 131k×entry memory footprint. */
    uint64_t stat_hits;
    uint64_t stat_misses;
    uint64_t stat_inserts;
    uint64_t stat_insert_collisions;
    /* Thread-safety (phase 4): sharded locks keyed by hash & (NCACHE_SHARDS-1).
     * Walker pool probe/insert holds the shard lock. Uncontended x86 cost
     * is ~10ns; sharding keeps contention sub-linear in walker count. */
    pthread_mutex_t shard_locks[16];
} NNCacheData;
#define NCACHE_SHARDS 16

static int nncache_init(NNCacheData *c, int32_t cap) {
    c->cap = cap;
    c->mask = cap - 1;
    c->count = 0;
    c->generation = 1;
    c->stat_hits = 0;
    c->stat_misses = 0;
    c->stat_inserts = 0;
    c->stat_insert_collisions = 0;
    c->entries = (NNCacheEntry *)calloc(cap, sizeof(NNCacheEntry));
    if (!c->entries) return -1;
    for (int i = 0; i < NCACHE_SHARDS; i++) {
        if (pthread_mutex_init(&c->shard_locks[i], NULL) != 0) {
            for (int j = 0; j < i; j++) pthread_mutex_destroy(&c->shard_locks[j]);
            free(c->entries);
            c->entries = NULL;
            return -1;
        }
    }
    return 0;
}

static void nncache_free(NNCacheData *c) {
    if (c->entries) {
        for (int i = 0; i < NCACHE_SHARDS; i++) pthread_mutex_destroy(&c->shard_locks[i]);
    }
    free(c->entries);
    c->entries = NULL;
    c->count = 0;
}

static inline pthread_mutex_t *nncache_shard(NNCacheData *c, uint64_t hash) {
    return &c->shard_locks[hash & (NCACHE_SHARDS - 1)];
}

/* Probe cache (thread-safe). Returns locked pointer to entry on hit, NULL on
 * miss. On hit the caller OWNS the shard lock and MUST call nncache_release
 * after reading entry fields. Keeps priors/legal arrays stable against a
 * concurrent insert racing in on the same shard.
 *
 * Serial callers still get correct behavior — lock/unlock is ~10ns. */
static NNCacheEntry *nncache_probe(NNCacheData *c, uint64_t hash) {
    int32_t slot = (int32_t)(hash & (uint64_t)c->mask);
    pthread_mutex_t *lk = nncache_shard(c, hash);
    pthread_mutex_lock(lk);
    NNCacheEntry *e = &c->entries[slot];
    if (e->generation == c->generation && e->key == hash) {
        c->stat_hits++;
        return e;  /* caller owns the shard lock */
    }
    c->stat_misses++;
    pthread_mutex_unlock(lk);
    return NULL;
}

/* Release the shard lock returned by a successful nncache_probe hit. */
static inline void nncache_release(NNCacheData *c, uint64_t hash) {
    pthread_mutex_unlock(nncache_shard(c, hash));
}

/* Insert into cache (thread-safe). Caller must call nncache_release on
 * the same hash after writing entry fields. Returns NULL (with lock
 * already released) if slot collides with another live entry. */
static NNCacheEntry *nncache_insert(NNCacheData *c, uint64_t hash) {
    int32_t slot = (int32_t)(hash & (uint64_t)c->mask);
    pthread_mutex_t *lk = nncache_shard(c, hash);
    pthread_mutex_lock(lk);
    NNCacheEntry *e = &c->entries[slot];
    if (e->generation == c->generation && e->key != hash) {
        c->stat_insert_collisions++;
        pthread_mutex_unlock(lk);
        return NULL;  /* keep existing entry (older = more useful) */
    }
    if (e->generation != c->generation) c->count++;
    c->stat_inserts++;
    e->key = hash;
    e->generation = c->generation;
    return e;  /* caller owns the shard lock */
}

/* ================================================================
 * Tree data structure
 * ================================================================ */

#define INITIAL_NODE_CAP   4096
#define INITIAL_CHILD_CAP  65536

/* Solved-node status, from the node's side-to-move perspective.
 * Stored in TreeData.solved[node_id] as int8_t.
 *   UNKNOWN: not proven (default for any new node).
 *   WIN:     STM is provably winning (some legal move leads to opponent LOSS).
 *   LOSS:    STM is provably losing (every legal move leaves opponent WINning,
 *            or the leaf itself is checkmate against STM).
 *   DRAW:    STM provably draws (no winning child; at least one drawing child;
 *            no losing child — or terminal stalemate / 50-move / repetition /
 *            insufficient material).
 * Backup is min/max-style and respects the side-flip across plies. */
#define SOLVED_UNKNOWN  0
#define SOLVED_WIN      1
#define SOLVED_LOSS    -1
#define SOLVED_DRAW     2

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
    int32_t *virtual_loss;     /* vloss count per node (walker-pool scaffolding, phase 2+3). */
    int8_t  *solved;           /* SOLVED_UNKNOWN/WIN/LOSS/DRAW from this node's STM perspective.
                                * 0 = unknown (default). Once set non-zero, value is proven and
                                * MCTS selection short-circuits. See SOLVED_* constants below. */
    int8_t  *has_solved_child; /* 1 iff at least one direct child has solved != UNKNOWN.
                                * Lets the selection hot path skip the solved-aware pre-pass
                                * entirely on the common case where no propagation has reached
                                * this node yet. Set inside tree_mark_solved_and_propagate
                                * whenever a child's solved status changes. */

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

    /* Structure-mutation lock (phase 4): held during tree_expand (which can
     * trigger tree_grow_nodes/tree_grow_children realloc), tree_ht_insert,
     * and other writes that could race with concurrent walkers.
     *
     * Readers (descent, backprop) do NOT hold this lock on the hot path:
     *   - N[] is atomically incremented in tree_backprop
     *   - expanded[] uses acquire/release atomics so "expanded==1" implies
     *     children[] / children_offset[] / num_children[] are fully visible
     *   - W[] is read racily (torn double is theoretically possible on
     *     non-x86; on x86_64 aligned 8-byte reads are atomic); stale Q is
     *     tolerated, matching the lc0 MCTS concurrency model.
     *
     * Realloc (tree_grow_nodes) is the only scenario where readers can
     * segfault on a stale pointer. Callers that expect multi-walker use
     * MUST call MCTSTree.reserve(max_nodes) upfront so no realloc fires
     * during concurrent descent. */
    pthread_mutex_t tree_lock;
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
    t->virtual_loss = (int32_t *)calloc(t->node_cap, sizeof(int32_t));
    t->solved = (int8_t *)calloc(t->node_cap, sizeof(int8_t));
    t->has_solved_child = (int8_t *)calloc(t->node_cap, sizeof(int8_t));

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
        !t->virtual_loss || !t->solved || !t->has_solved_child ||
        !t->child_action || !t->child_node || !t->node_hash || !t->hash_table) {
        return -1;
    }
    memset(t->parent, -1, t->node_cap * sizeof(int32_t));
    memset(t->action_from_parent, -1, t->node_cap * sizeof(int32_t));
    memset(t->hash_table, -1, t->ht_cap * sizeof(int32_t));
    if (pthread_mutex_init(&t->tree_lock, NULL) != 0) return -1;
    return 0;
}


static void tree_free(TreeData *t) {
    /* Destroy the mutex only if the struct was actually initialized —
     * callers that hit OOM mid-tree_init bail before the mutex is set up,
     * and zeroing pthread_mutex_t + destroying is undefined on glibc. */
    if (t->N) pthread_mutex_destroy(&t->tree_lock);
    free(t->N);
    free(t->W);
    free(t->prior);
    free(t->expanded);
    free(t->parent);
    free(t->action_from_parent);
    free(t->num_children);
    free(t->children_offset);
    free(t->virtual_loss);
    free(t->solved);
    free(t->has_solved_child);
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
    GROW(virtual_loss, int32_t);
    GROW(solved, int8_t);
    GROW(has_solved_child, int8_t);
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
    memset(t->virtual_loss + old_cap, 0, (new_cap - old_cap) * sizeof(int32_t));
    memset(t->solved + old_cap, 0, (new_cap - old_cap) * sizeof(int8_t));
    memset(t->has_solved_child + old_cap, 0, (new_cap - old_cap) * sizeof(int8_t));
    memset(t->node_hash + old_cap, 0, (new_cap - old_cap) * sizeof(uint64_t));

    t->node_cap = new_cap;
    return 0;
}


/* Hash table: probe for existing node with given Zobrist hash. Returns node_id or -1.
 *
 * Thread-safety: acquire-load on hash_table[slot] pairs with release-store
 * in tree_ht_insert so that if we see a nid, node_hash[nid] is also visible. */
static int32_t tree_ht_probe(const TreeData *t, uint64_t hash) {
    int32_t slot = (int32_t)(hash & (uint64_t)t->ht_mask);
    int32_t nid = __atomic_load_n(&t->hash_table[slot], __ATOMIC_ACQUIRE);
    if (nid >= 0 && t->node_hash[nid] == hash)
        return nid;
    return -1;
}

/* Hash table: register node_id for given hash. Must be called under
 * tree_lock (guaranteed by tree_ht_insert wrapper below or by direct
 * callers that already hold the lock). Uses release-store on the slot
 * so probers that do an acquire-load see a coherent (slot, node_hash) pair. */
static void tree_ht_insert_locked(TreeData *t, uint64_t hash, int32_t node_id) {
    int32_t slot = (int32_t)(hash & (uint64_t)t->ht_mask);
    int32_t existing = t->hash_table[slot];
    if (existing >= 0 && existing < t->node_count && t->node_hash[existing] == hash)
        return;
    /* Write node_hash BEFORE publishing into the slot — probers that read
     * nid via acquire-load and then check node_hash[nid] must see the
     * correct hash, not a stale one. */
    t->node_hash[node_id] = hash;
    __atomic_store_n(&t->hash_table[slot], node_id, __ATOMIC_RELEASE);
}

static void tree_ht_insert(TreeData *t, uint64_t hash, int32_t node_id) {
    pthread_mutex_lock(&t->tree_lock);
    tree_ht_insert_locked(t, hash, node_id);
    pthread_mutex_unlock(&t->tree_lock);
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
    t->solved[id] = SOLVED_UNKNOWN;
    t->has_solved_child[id] = 0;
    return id;
}


/* Expand a node: add children for each (action, prior) pair.
 * All-or-nothing: if any tree_add_node fails, the node is left unexpanded
 * and any partially-allocated child nodes are rolled back.
 *
 * Thread-safety: holds tree_lock for the critical section (alloc + realloc
 * + children writes). Publishes expanded[node_id] = 1 with release semantics
 * so acquire-loading readers see a fully-populated (children_offset,
 * num_children, child_action, child_node) view. */
static int tree_expand(TreeData *t, int32_t node_id,
                       const int32_t *actions, const double *priors, int32_t n_children) {
    pthread_mutex_lock(&t->tree_lock);
    /* Re-check under the lock: another walker may have already expanded
     * this node between our descent and acquiring the lock. Idempotent. */
    if (__atomic_load_n(&t->expanded[node_id], __ATOMIC_RELAXED)) {
        pthread_mutex_unlock(&t->tree_lock);
        return 0;
    }
    if (n_children <= 0) {
        __atomic_store_n(&t->expanded[node_id], 1, __ATOMIC_RELEASE);
        pthread_mutex_unlock(&t->tree_lock);
        return 0;
    }

    if (tree_grow_children(t, n_children) < 0) {
        pthread_mutex_unlock(&t->tree_lock);
        return -1;
    }

    int32_t offset = t->child_count;
    int32_t saved_node_count = t->node_count;

    for (int32_t i = 0; i < n_children; i++) {
        int32_t child_id = tree_add_node(t, node_id, actions[i], priors[i]);
        if (child_id < 0) {
            t->node_count = saved_node_count;
            pthread_mutex_unlock(&t->tree_lock);
            return -1;
        }
        t->child_action[offset + i] = actions[i];
        t->child_node[offset + i] = child_id;
    }

    t->children_offset[node_id] = offset;
    t->num_children[node_id] = n_children;
    t->child_count += n_children;
    /* Release publishes the whole children payload to acquire-loaders. */
    __atomic_store_n(&t->expanded[node_id], 1, __ATOMIC_RELEASE);
    pthread_mutex_unlock(&t->tree_lock);
    return 0;
}


/* ================================================================
 * PUCT select child (hottest path)
 * ================================================================ */

/* Select best child of `node_id` using PUCT formula with FPU.
 * Returns index into children pool (children_offset[node_id] + best_slot).
 *
 * When vloss_weight > 0, in-flight walkers are treated as visits with a
 * losing Q from the child's own side-to-move frame: effective N = N + vl
 * and effective W = W + vl (so -W/N swings toward +loss). This matches
 * what tree_gumbel_select_child does for Gumbel descent. */
static int32_t tree_select_child(const TreeData *t, int32_t node_id,
                                  double c_puct, double fpu_reduction,
                                  int vloss_weight) {
    int32_t n_ch = t->num_children[node_id];
    int32_t off = t->children_offset[node_id];

    /* Solved-aware fast paths (only entered when at least one child has been
     * proven). The common case — no propagated proofs in this subtree —
     * skips the pre-pass + the per-child solved checks in the main loop.
     *
     *   - any child with solved==SOLVED_LOSS (from child's STM) means parent
     *     has a move to a losing-for-opponent position → take it. Pick the
     *     highest-prior such child so the soft policy target stays sensible.
     *   - children with solved==SOLVED_WIN are losing for parent; exclude
     *     them from candidate set. If *every* child is SOLVED_WIN, parent is
     *     also already LOSS — fall through and pick any (we still need a
     *     visit to keep N stats coherent on the path).
     * SOLVED_DRAW children stay in the candidate pool with their normal
     * scores; the W/N average converges to 0 anyway. */
    int32_t any_unsolved_or_draw = 1;  /* default for the no-solved-child fast path */
    if (t->has_solved_child[node_id]) {
        int32_t winning_slot = -1;
        double  winning_prior = -1.0;
        any_unsolved_or_draw = 0;
        for (int32_t i = 0; i < n_ch; i++) {
            int8_t cs = t->solved[t->child_node[off + i]];
            if (cs == SOLVED_LOSS) {
                double pr = t->prior[t->child_node[off + i]];
                if (pr > winning_prior) { winning_prior = pr; winning_slot = i; }
            } else if (cs != SOLVED_WIN) {
                any_unsolved_or_draw = 1;
            }
        }
        if (winning_slot >= 0) return winning_slot;
    }

    int32_t parent_vl = (vloss_weight > 0) ? t->virtual_loss[node_id] : 0;
    double parent_N = (double)(t->N[node_id] + vloss_weight * parent_vl);
    double parent_W = t->W[node_id] + (double)(vloss_weight * parent_vl);
    double parent_Q = (parent_N > 0) ? (parent_W / parent_N) : 0.0;
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

    /* Hoist the solved-aware-loop branch so the common case (no solved
     * children) skips both per-child solved[] loads. */
    int32_t check_solved = t->has_solved_child[node_id];
    for (int32_t i = 0; i < n_ch; i++) {
        int32_t cid = t->child_node[off + i];
        int8_t cs = check_solved ? t->solved[cid] : SOLVED_UNKNOWN;
        /* Skip SOLVED_WIN children unless every child is solved-losing for
         * parent (in which case any_unsolved_or_draw is 0 and we let them
         * back in — we still need to visit something). */
        if (any_unsolved_or_draw && cs == SOLVED_WIN) continue;
        int32_t vl = (vloss_weight > 0) ? t->virtual_loss[cid] : 0;
        int32_t n = t->N[cid] + vloss_weight * vl;
        double w = t->W[cid] + (double)(vloss_weight * vl);
        double prior = t->prior[cid];
        if (n > 0) {
            visited_policy += prior;
            /* For SOLVED_DRAW children, override Q to exact 0 — accumulated
             * W might disagree slightly from one or two NN-eval visits before
             * the leaf was proven, but the proven value is what selection
             * should use. */
            double q_parent = (cs == SOLVED_DRAW) ? 0.0 : (-w / (double)n);
            double score = q_parent + c_sqrt_n * prior / (1.0 + (double)n);
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
 * Returns path length. path[0] = root, path[len-1] = leaf.
 *
 * vloss_weight=0 keeps descent bit-identical to the pre-walker code path. */
static int32_t tree_select_leaf(const TreeData *t, int32_t root_id,
                                 double c_puct, double fpu_at_root, double fpu_reduction,
                                 int vloss_weight,
                                 int32_t *path, int32_t max_path) {
    int32_t node = root_id;
    int32_t depth = 0;
    path[depth++] = node;

    double fpu = fpu_at_root;
    /* Acquire-load pairs with release-store in tree_expand: when we see
     * expanded==1, the full children_offset/num_children/child_node view
     * is visible. */
    while (__atomic_load_n(&t->expanded[node], __ATOMIC_ACQUIRE) && t->num_children[node] > 0) {
        if (depth >= max_path) break;
        int32_t slot = tree_select_child(t, node, c_puct, fpu, vloss_weight);
        int32_t off = t->children_offset[node];
        node = t->child_node[off + slot];
        path[depth++] = node;
        fpu = fpu_reduction;
    }
    return depth;
}


/* Flip a solved status across one ply (parent <-> child STM swap).
 *   parent sees a child WIN  → parent considers it LOSS-for-self
 *   parent sees a child LOSS → parent considers it WIN-for-self
 *   DRAW and UNKNOWN are unchanged. */
static inline int8_t solved_flip(int8_t s) {
    if (s == SOLVED_WIN) return SOLVED_LOSS;
    if (s == SOLVED_LOSS) return SOLVED_WIN;
    return s;  /* DRAW or UNKNOWN */
}


/* Solved status for a known-terminal CBoard. Caller must have already
 * confirmed cboard_is_game_over(b). Companion to cboard_terminal_value:
 * checkmate ⇒ STM lost; everything else terminal (stalemate / 50-move /
 * repetition / insufficient material) ⇒ DRAW. */
static inline int8_t cboard_terminal_solved_status(const CBoard *b) {
    return cboard_is_checkmate(b) ? SOLVED_LOSS : SOLVED_DRAW;
}


/* Resolve `node`'s solved status from its expanded children, returning the
 * new status (SOLVED_UNKNOWN if not yet provable). Caller must hold tree_lock
 * — children_offset/num_children/solved[] are read across multiple slots and
 * we don't want concurrent expand/mark to race with this scan. */
static int8_t tree_resolve_from_children(const TreeData *t, int32_t node) {
    if (!__atomic_load_n(&t->expanded[node], __ATOMIC_ACQUIRE)) return SOLVED_UNKNOWN;
    int32_t nc = t->num_children[node];
    if (nc <= 0) return SOLVED_UNKNOWN;
    int32_t off = t->children_offset[node];
    int has_draw = 0;
    int all_solved = 1;
    for (int32_t j = 0; j < nc; j++) {
        int32_t cid = t->child_node[off + j];
        int8_t cs = t->solved[cid];
        int8_t cs_parent = solved_flip(cs);  /* what this child means for parent */
        if (cs_parent == SOLVED_WIN) return SOLVED_WIN;  /* one good move suffices */
        if (cs_parent == SOLVED_DRAW) has_draw = 1;
        else if (cs_parent != SOLVED_LOSS) all_solved = 0;  /* an unknown child blocks resolution */
    }
    if (!all_solved) return SOLVED_UNKNOWN;
    return has_draw ? SOLVED_DRAW : SOLVED_LOSS;
}


/* Forward decl: defined below, after tree_resolve_from_children. */
static void tree_mark_solved_and_propagate(TreeData *t,
                                           const int32_t *path,
                                           int32_t path_len,
                                           int8_t leaf_status);


/* Lock-free lookup of a child by action. Returns child node id, or -1 if
 * not present. Caller must have observed expanded[node_id]==1 with acquire
 * semantics so children_offset/num_children/child_action[] are coherent. */
static int32_t tree_find_child_action(const TreeData *t, int32_t node_id, int32_t action) {
    int32_t n_ch = t->num_children[node_id];
    int32_t off = t->children_offset[node_id];
    for (int32_t i = 0; i < n_ch; i++) {
        if (t->child_action[off + i] == action)
            return t->child_node[off + i];
    }
    return -1;
}


/* Forced-move chain collapse. When the current leaf has exactly one legal
 * move, expand it as a single-child node (prior=1.0), descend, and repeat
 * until branching, terminal, or depth_cap. Saves NN evals on perpetual-check
 * sequences, forced recaptures, etc. — positions where the policy would
 * collapse to one-hot anyway.
 *
 * In/out parameters (all updated when chain collapses):
 *   *leaf_id  — current leaf node id, advanced to chain end
 *   *cb       — board state, advanced to chain end (cboard_push_index'd)
 *   path_buf  — descent path, with new node ids appended
 *   *path_len — length of path_buf
 *
 * Returns:
 *    1 — chain ended at a terminal; leaf was marked solved + propagated.
 *        Caller must NOT queue this leaf for NN eval; should treat it like
 *        any other terminal-leaf detection (backprop the terminal value).
 *    0 — chain did not terminate. leaf_id/cb may have advanced 0+ plies;
 *        caller should proceed with NN eval / transposition check at the
 *        possibly-new leaf.
 *
 * Thread-safety: tree_expand internally takes tree_lock for each forced ply.
 * That's the only shared-state mutation. cb is caller-local. */
static int try_forced_collapse(TreeData *t,
                                int32_t *leaf_id, CBoard *cb,
                                int32_t *path_buf, int32_t *path_len,
                                int depth_cap) {
    int legal_buf[256];
    for (int d = 0; d < depth_cap; d++) {
        int n = cboard_legal_move_indices(cb, legal_buf, /*sorted=*/0);
        if (n != 1) return 0;
        int32_t action = (int32_t)legal_buf[0];

        int32_t cur = *leaf_id;
        if (!__atomic_load_n(&t->expanded[cur], __ATOMIC_ACQUIRE)) {
            double prior_one = 1.0;
            if (tree_expand(t, cur, &action, &prior_one, 1) < 0) return 0;
        }
        int32_t child_id = tree_find_child_action(t, cur, action);
        if (child_id < 0) return 0;  /* expand-then-not-found = race; bail */

        cboard_push_index(cb, action);
        if (*path_len >= MCTS_MAX_PATH) return 0;  /* path overflow — bail */
        path_buf[(*path_len)++] = child_id;
        *leaf_id = child_id;

        if (cboard_is_game_over(cb)) {
            tree_mark_solved_and_propagate(t, path_buf, *path_len,
                                            cboard_terminal_solved_status(cb));
            return 1;
        }
    }
    return 0;
}


/* Mark a leaf (or freshly evaluated node) with a solved status, then walk
 * upward and resolve as many ancestors as possible. Stops at the first
 * ancestor that doesn't yet resolve.
 *
 * Holds tree_lock for the scan + writes — keeps the child-array view
 * coherent against concurrent expansion (phase 4 walker pool, phase 2
 * brute-force worker). The hot backprop path is short (a few writes per
 * ancestor); contention with NN-eval-driven walkers is bounded by tree
 * branching factor.
 *
 * `path` is the descent path from root → leaf (same array passed to
 * tree_backprop). leaf_status is the status to write at path[path_len-1]. */
static void tree_mark_solved_and_propagate(TreeData *t,
                                           const int32_t *path,
                                           int32_t path_len,
                                           int8_t leaf_status) {
    if (path_len <= 0 || leaf_status == SOLVED_UNKNOWN) return;
    pthread_mutex_lock(&t->tree_lock);

    int32_t leaf = path[path_len - 1];
    /* If already solved, nothing to do — solved status is monotonic
     * (UNKNOWN → terminal once and stays). */
    if (t->solved[leaf] != SOLVED_UNKNOWN) {
        pthread_mutex_unlock(&t->tree_lock);
        return;
    }
    t->solved[leaf] = leaf_status;
    /* Notify the leaf's parent that it now has a solved child — lets the
     * selection hot path skip the solved-aware pre-pass when this flag is 0. */
    if (path_len >= 2) t->has_solved_child[path[path_len - 2]] = 1;

    /* Walk up. At each ancestor, try to resolve from its children. Stop on
     * the first one that's still UNKNOWN — anything above it can't resolve
     * either. */
    for (int32_t i = path_len - 2; i >= 0; i--) {
        int32_t nid = path[i];
        if (t->solved[nid] != SOLVED_UNKNOWN) break;  /* already resolved earlier */
        int8_t resolved = tree_resolve_from_children(t, nid);
        if (resolved == SOLVED_UNKNOWN) break;
        t->solved[nid] = resolved;
        /* Newly-resolved ancestor → its own parent now has a solved child. */
        if (i >= 1) t->has_solved_child[path[i - 1]] = 1;
    }

    pthread_mutex_unlock(&t->tree_lock);
}


/* Backprop value up the path. Value is from leaf's side-to-move perspective.
 * Alternates sign at each level.
 *
 * Thread-safety: N is atomically incremented (RELAXED is fine — we don't
 * need to order backprop against any particular external event). W is
 * non-atomic: on x86_64 aligned 8-byte writes are atomic at the ISA level,
 * and concurrent walkers reading Q can tolerate one-cycle stale W — this
 * is the standard lc0/KataGo concurrency model. Torn-double on non-x86
 * would produce slightly nonsense Q for one walker's select, not a crash. */
static void tree_backprop(TreeData *t, const int32_t *path, int32_t path_len, double value) {
    double v = value;
    for (int32_t i = path_len - 1; i >= 0; i--) {
        int32_t nid = path[i];
        __atomic_fetch_add(&t->N[nid], 1, __ATOMIC_RELAXED);
        t->W[nid] += v;
        v = -v;
    }
}


/* Apply one unit of virtual loss to every non-root node on `path`.
 * Walkers call this right after descending to a leaf and before requesting
 * GPU eval; the increment keeps other walkers off the same subtree until
 * the real eval result arrives. Root is skipped — all walkers share it.
 *
 * Uses __atomic_fetch_add so concurrent walkers on different paths don't
 * lose updates when they touch a shared interior node. */
static void tree_apply_vloss_path(TreeData *t, const int32_t *path, int32_t path_len) {
    for (int32_t i = 1; i < path_len; i++) {
        __atomic_fetch_add(&t->virtual_loss[path[i]], 1, __ATOMIC_RELAXED);
    }
}


/* Remove one unit of virtual loss from every non-root node on `path`. Paired
 * with tree_apply_vloss_path — called just before the real backprop, so the
 * net visit count on the path ends at (+1) not (0). Floors at zero so a
 * miscounted decrement can't flip the field negative. */
static void tree_remove_vloss_path(TreeData *t, const int32_t *path, int32_t path_len) {
    for (int32_t i = 1; i < path_len; i++) {
        int32_t prev = __atomic_fetch_sub(&t->virtual_loss[path[i]], 1, __ATOMIC_RELAXED);
        if (prev <= 0) {
            __atomic_store_n(&t->virtual_loss[path[i]], 0, __ATOMIC_RELAXED);
        }
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
    int8_t  *term_solved;        /* SOLVED_* status for each terminal; SOLVED_UNKNOWN
                                  * means "no proven status, just a fallback value (e.g.
                                  * root-Q for buffer-full / depth-1 root cases)". */
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
    /* Virtual-loss weight: when >0, in-flight walkers along a node's path
     * are treated as visits with a penalty (N += vl, W += vl per unit). Keeps
     * concurrent walkers from collapsing onto the same PUCT-best leaf. At 0
     * descent is bit-identical to the pre-vloss implementation. */
    int vloss_weight;
} GumbelSelectParams;

/* Forward declarations for functions used by GumbelSimState helpers */
static int stored_ensure_cap(StoredPrepState *s, int32_t leaf_cap, int32_t term_cap);
static int stored_ensure_legal_flat(StoredPrepState *s, int32_t need);
static int stored_ensure_path_flat(StoredPrepState *s, int32_t need);
static int stored_ensure_term_path_flat(StoredPrepState *s, int32_t need);
/* Returns the appended leaf index on success, -1 on OOM (state left unchanged). */
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

/* Append a terminal leaf (path + value) to StoredPrepState.
 * Returns 0 on success, -1 on OOM (state left unchanged). The ensure_*
 * fast path (needed <= cap) is branchless and compiles away; the -1
 * branch only fires when realloc actually runs and fails, so this adds
 * no measurable hot-path cost. */
static inline int stored_append_terminal(StoredPrepState *s,
                                         const int32_t *path_buf, int32_t path_len,
                                         double value, int8_t solved_status) {
    if (stored_ensure_term_path_flat(s, s->term_path_flat_used + path_len) < 0) return -1;
    int32_t ti = s->n_terminals;
    s->term_path_offset[ti] = s->term_path_flat_used;
    s->term_path_count[ti] = path_len;
    memcpy(s->term_path_flat + s->term_path_flat_used, path_buf, path_len * sizeof(int32_t));
    s->term_path_flat_used += path_len;
    s->term_values[ti] = value;
    s->term_solved[ti] = solved_status;
    s->n_terminals++;
    return 0;
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

    /* Minibatch-size target: gss_step accumulates up to this many leaves
     * before returning for GPU eval. Large = better GPU util, slightly
     * staler tree state on later leaves + higher stop-latency. Small =
     * the reverse. Set by start_gumbel_sims; falls back to GSS_GPU_BATCH. */
    int32_t target_batch;

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
static int32_t gss_begin_round(GumbelSimState *g, const TreeData *t) {
    g->n_active = 0;
    g->max_reps = 0;
    for (int32_t i = 0; i < g->n_boards; i++) {
        /* Early-exit when the root is already solved: the result of search is
         * known, every additional sim is wasted GPU work. Zero the budget so
         * it stays out of subsequent rounds too. */
        int32_t rid = g->root_ids[i];
        if (rid >= 0 && rid < t->node_count && t->solved[rid] != SOLVED_UNKNOWN) {
            g->budget_remaining[i] = 0;
        }
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
 * Returns the number of new leaves that need GPU eval, or -1 on OOM
 * (StoredPrepState left in its pre-call state — caller should abort). */
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
            /* Root-only "descent" — no real leaf was reached, just bias backprop
             * with the root's known Q. Not a proven status. */
            if (stored_append_terminal(s, path_buf, path_len, g->root_qs[bi], SOLVED_UNKNOWN) < 0) return -1;
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
            if (stored_append_terminal(s, path_buf, path_len,
                                       (double)cboard_terminal_value(&cb),
                                       cboard_terminal_solved_status(&cb)) < 0) return -1;
            continue;
        }

        /* Forced-move chain collapse: when this leaf has only one legal move
         * (forced recapture, perpetual-check, etc.), expand the chain into the
         * tree at prior=1.0 per ply and advance to the first branching position
         * before NN eval. Saves an NN eval per collapsed ply on every future
         * sim through this subtree. */
        if (try_forced_collapse(t, &leaf_id, &cb, path_buf, &path_len, 16)) {
            /* Chain ended at a terminal; collapse already marked solved.
             * Backprop the terminal value the same way regular terminals do. */
            if (stored_append_terminal(s, path_buf, path_len,
                                       (double)cboard_terminal_value(&cb),
                                       SOLVED_UNKNOWN) < 0) return -1;
            continue;
        }

        /* Transposition / NNCache check */
        if (!__atomic_load_n(&t->expanded[leaf_id], __ATOMIC_ACQUIRE)) {
            int32_t existing = tree_ht_probe(t, cb.hash);
            if (existing >= 0 && existing != leaf_id &&
                __atomic_load_n(&t->expanded[existing], __ATOMIC_ACQUIRE)) {
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
                    __atomic_store_n(&t->expanded[leaf_id], 1, __ATOMIC_RELEASE);
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
                    /* Copy fields under the shard lock, then release before
                     * calling tree_expand (which takes its own lock). */
                    int32_t legal_copy[NNCACHE_MAX_LEGAL];
                    double  prior_copy[NNCACHE_MAX_LEGAL];
                    int32_t n_legal_copy = ce->n_legal;
                    double  q_copy = ce->q_value;
                    memcpy(legal_copy, ce->legal, n_legal_copy * sizeof(int32_t));
                    memcpy(prior_copy, ce->priors, n_legal_copy * sizeof(double));
                    nncache_release(g->nncache, cb.hash);
                    tree_expand(t, leaf_id, legal_copy, prior_copy, n_legal_copy);
                    tree_backprop(t, path_buf, path_len, q_copy);
                    tree_ht_insert(t, cb.hash, leaf_id);
                    if (cache_ok) tree_cb_cache_put(t, leaf_id, &cb);
                    continue;
                }
            }
        }

        if (s->n_leaves >= g->enc_capacity) {
            /* Buffer full — fall back to root Q (not 0.0 which biases toward draws).
             * Not a proven terminal, so SOLVED_UNKNOWN. */
            if (stored_append_terminal(s, path_buf, path_len, g->root_qs[bi], SOLVED_UNKNOWN) < 0) return -1;
            continue;
        }

        /* Save leaf for deferred encoding */
        if (stored_append_leaf(s, t, &cb, leaf_id, path_buf, path_len, cache_ok) < 0) return -1;
    }

    /* Backprop terminals + propagate proven statuses upward. */
    for (int32_t ti = 0; ti < s->n_terminals; ti++) {
        const int32_t *tpath = s->term_path_flat + s->term_path_offset[ti];
        int32_t tlen = s->term_path_count[ti];
        tree_backprop(t, tpath, tlen, s->term_values[ti]);
        if (s->term_solved[ti] != SOLVED_UNKNOWN) {
            tree_mark_solved_and_propagate(t, tpath, tlen, s->term_solved[ti]);
        }
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

        if (!__atomic_load_n(&t->expanded[nid], __ATOMIC_ACQUIRE) && n_legal > 0) {
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
                        nncache_release(nncache, s->hashes[li]);
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
     * on later leaves in the batch. Driven by g->target_batch (set by
     * start_gumbel_sims); fall back to GSS_GPU_BATCH when unset. */
    int32_t target_batch = (g->target_batch > 0) ? g->target_batch : GSS_GPU_BATCH;

    for (;;) {
        /* Build queries for one rep at a time */
        int32_t n_queries = gss_build_queries(g);

        if (n_queries > 0) {
            /* Ensure stored capacity. A failed realloc stops the whole step
             * so the caller sees OOM as -1 instead of corrupted writes. */
            if (stored_ensure_cap(s, s->n_leaves + n_queries, s->n_terminals + n_queries) < 0) return -1;
            if (stored_ensure_legal_flat(s, s->legal_flat_used + n_queries * 40) < 0) return -1;
            if (stored_ensure_path_flat(s, s->path_flat_used + n_queries * 8) < 0) return -1;
            if (stored_ensure_term_path_flat(s, s->term_path_flat_used + n_queries * 8) < 0) return -1;

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

            if (gss_prepare_batch(t, s, g, n_queries, enc_data) < 0) return -1;

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
        if (gss_begin_round(g, t) == 0) {
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
    free(s->term_values); free(s->term_solved);
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
        SGROW(term_solved,      int8_t,  term_cap);
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
    /* Compute legals into stack buffer first so we can grow capacity before
     * committing anything to s->*. On OOM, return -1 with s unchanged. */
    int indices[256];
    int count = cboard_legal_move_indices(cb, indices, /*sorted=*/0);

    if (stored_ensure_legal_flat(s, s->legal_flat_used + count) < 0) return -1;
    if (stored_ensure_path_flat(s, s->path_flat_used + path_len) < 0) return -1;

    int32_t li = s->n_leaves;
    s->leaf_cboards[li] = *cb;
    s->leaf_ids[li] = leaf_id;
    s->hashes[li] = cb->hash;

    s->legal_offset[li] = s->legal_flat_used;
    s->legal_count[li] = count;
    for (int32_t j = 0; j < count; j++)
        s->legal_flat[s->legal_flat_used + j] = indices[j];
    s->legal_flat_used += count;

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
    /* Copy fields into stack buffers while the shard lock is held, then
     * release before doing Python allocation (which can arbitrarily long-tail
     * on GC and would block other walkers on this shard). */
    int32_t n_legal = e->n_legal;
    int32_t legal_buf[NNCACHE_MAX_LEGAL];
    double  prior_buf[NNCACHE_MAX_LEGAL];
    double  q_copy = e->q_value;
    memcpy(legal_buf, e->legal, n_legal * sizeof(int32_t));
    memcpy(prior_buf, e->priors, n_legal * sizeof(double));
    nncache_release(&self->cache, (uint64_t)hash);

    npy_intp dims[1] = {n_legal};
    PyObject *legal_arr = PyArray_SimpleNew(1, dims, NPY_INT32);
    PyObject *prior_arr = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (!legal_arr || !prior_arr) {
        Py_XDECREF(legal_arr); Py_XDECREF(prior_arr);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject *)legal_arr), legal_buf, n_legal * sizeof(int32_t));
    memcpy(PyArray_DATA((PyArrayObject *)prior_arr), prior_buf, n_legal * sizeof(double));
    return Py_BuildValue("(NNd)", legal_arr, prior_arr, q_copy);
}

static PyObject *PyNNCache_stats(PyNNCacheObject *self, PyObject *Py_UNUSED(ignored)) {
    NNCacheData *c = &self->cache;
    return Py_BuildValue(
        "{s:K,s:K,s:K,s:K,s:i,s:i}",
        "hits", (unsigned long long)c->stat_hits,
        "misses", (unsigned long long)c->stat_misses,
        "inserts", (unsigned long long)c->stat_inserts,
        "insert_collisions", (unsigned long long)c->stat_insert_collisions,
        "count", c->count,
        "cap", c->cap
    );
}

static PyObject *PyNNCache_reset_stats(PyNNCacheObject *self, PyObject *Py_UNUSED(ignored)) {
    NNCacheData *c = &self->cache;
    c->stat_hits = 0;
    c->stat_misses = 0;
    c->stat_inserts = 0;
    c->stat_insert_collisions = 0;
    Py_RETURN_NONE;
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
    {"stats", (PyCFunction)PyNNCache_stats, METH_NOARGS, "stats() -> {hits, misses, inserts, insert_collisions, count, cap}"},
    {"reset_stats", (PyCFunction)PyNNCache_reset_stats, METH_NOARGS, "reset_stats() -> None"},
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
                                             0,
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


/* ================================================================
 * Walker-pool reentrant API (Phase 5)
 *
 * Each walker thread drives ONE simulation at a time:
 *     leaf, path, legal, terminal_q = tree.walker_descend_puct(...)
 *     if terminal_q is None:
 *         pol, wdl = evaluator.evaluate_encoded(enc_buf)
 *         tree.walker_integrate_leaf(path, legal, pol, wdl, vloss_weight)
 *     else:
 *         tree.backprop(path, terminal_q)
 *
 * Safe for concurrent callers against the same tree provided:
 *   - tree.reserve(cap) called upfront so tree_grow_nodes cannot realloc
 *   - All concurrent callers pass the same vloss_weight (semantic mismatch
 *     between walkers is a design bug, not a crash; checked at your level)
 * ================================================================ */

/* walker_descend_puct(root_id, root_cboard, c_puct, fpu_root, fpu_reduction,
 *                     vloss_weight, enc_out)
 *   -> (leaf_id, node_path: int32[], legal: int32[], terminal_q: float | None)
 *
 * Descends from root using PUCT with virtual loss, replays moves on a fresh
 * CBoard, and either:
 *   - returns terminal_q set to the leaf's terminal value (no vloss applied,
 *     no encoding performed — caller just backprops) OR
 *   - applies vloss along path[1:], encodes the leaf into enc_out[0], and
 *     returns terminal_q = None plus the legal-move indices at the leaf.
 *
 * enc_out must be a writable float32 array of shape (>=1, 146, 8, 8). Only
 * enc_out[0] is written. */
static PyObject *MCTSTree_walker_descend_puct(MCTSTreeObject *self, PyObject *args) {
    int root_id;
    PyObject *root_cb_obj, *enc_obj;
    double c_puct, fpu_root, fpu_reduction;
    int vloss_weight;
    if (!PyArg_ParseTuple(args, "iOdddiO",
                          &root_id, &root_cb_obj,
                          &c_puct, &fpu_root, &fpu_reduction,
                          &vloss_weight, &enc_obj))
        return NULL;

    TreeData *t = &self->tree;
    if (root_id < 0 || root_id >= t->node_count) {
        PyErr_SetString(PyExc_ValueError, "root_id out of range");
        return NULL;
    }

    /* Type-check the CBoard without adding a reference (we only read). */
    PyTypeObject *cb_type = _cached_cboard_type;
    if (!cb_type) {
        cb_type = Py_TYPE(root_cb_obj);
        if (strstr(cb_type->tp_name, "CBoard") == NULL) {
            PyErr_SetString(PyExc_TypeError, "root_cboard must be a CBoard");
            return NULL;
        }
        _cached_cboard_type = cb_type;
    } else if (Py_TYPE(root_cb_obj) != cb_type) {
        PyErr_SetString(PyExc_TypeError, "root_cboard must be a CBoard");
        return NULL;
    }
    const CBoard *root_cb = &((PyCBoard *)root_cb_obj)->board;

    PyArrayObject *enc_arr = FROMANY_4D_RW(enc_obj, NPY_FLOAT32);
    if (!enc_arr) return NULL;
    if (PyArray_DIM(enc_arr, 0) < 1 ||
        PyArray_DIM(enc_arr, 1) != 146 ||
        PyArray_DIM(enc_arr, 2) != 8 ||
        PyArray_DIM(enc_arr, 3) != 8) {
        Py_DECREF(enc_arr);
        PyErr_SetString(PyExc_ValueError, "enc_out must be shape (>=1, 146, 8, 8) float32");
        return NULL;
    }
    float *enc_data = (float *)PyArray_DATA(enc_arr);

    /* Descend. */
    int32_t path_buf[MCTS_MAX_PATH];
    int32_t path_len = tree_select_leaf(t, root_id, c_puct, fpu_root, fpu_reduction,
                                        vloss_weight, path_buf, MCTS_MAX_PATH);
    int32_t leaf_id = path_buf[path_len - 1];

    /* Replay moves root → leaf by walking parent pointers and reversing. */
    int32_t actions[MCTS_MAX_PATH];
    int32_t n_actions = 0;
    {
        int32_t nid = leaf_id;
        while (nid != root_id && t->parent[nid] >= 0 && n_actions < MCTS_MAX_PATH) {
            int32_t act = t->action_from_parent[nid];
            if (act >= 0) actions[n_actions++] = act;
            nid = t->parent[nid];
        }
    }
    CBoard cb = *root_cb;
    for (int32_t i = n_actions - 1; i >= 0; i--) {
        cboard_push_index(&cb, actions[i]);
    }

    /* Build path output array (shared by terminal + non-terminal branches). */
    npy_intp pdims[1] = {path_len};
    PyObject *node_path = PyArray_SimpleNew(1, pdims, NPY_INT32);
    if (!node_path) { Py_DECREF(enc_arr); return NULL; }
    memcpy(PyArray_DATA((PyArrayObject *)node_path), path_buf,
           path_len * sizeof(int32_t));

    if (cboard_is_game_over(&cb)) {
        /* Terminal: no vloss (caller backprops immediately), no encoding,
         * empty legal array. Also mark the leaf solved and propagate up. */
        Py_DECREF(enc_arr);
        npy_intp ldims[1] = {0};
        PyObject *legal_arr = PyArray_SimpleNew(1, ldims, NPY_INT32);
        if (!legal_arr) { Py_DECREF(node_path); return NULL; }
        double term_q = (double)cboard_terminal_value(&cb);
        tree_mark_solved_and_propagate(t, path_buf, path_len,
                                       cboard_terminal_solved_status(&cb));
        return Py_BuildValue("(iNNd)", leaf_id, node_path, legal_arr, term_q);
    }

    /* Forced-move chain collapse — same idea as in gss_prepare_batch. Try to
     * walk forced replies before paying for an NN eval. May rebuild path_buf
     * + advance leaf_id/cb in place. The path output array was already built
     * above; if we collapse, rebuild it. */
    if (try_forced_collapse(t, &leaf_id, &cb, path_buf, &path_len, 16)) {
        /* Chain hit terminal: same shape as the cboard_is_game_over branch. */
        Py_DECREF(enc_arr);
        Py_DECREF(node_path);
        npy_intp pdims2[1] = {path_len};
        node_path = PyArray_SimpleNew(1, pdims2, NPY_INT32);
        if (!node_path) return NULL;
        memcpy(PyArray_DATA((PyArrayObject *)node_path), path_buf,
               path_len * sizeof(int32_t));
        npy_intp ldims[1] = {0};
        PyObject *legal_arr = PyArray_SimpleNew(1, ldims, NPY_INT32);
        if (!legal_arr) { Py_DECREF(node_path); return NULL; }
        double term_q = (double)cboard_terminal_value(&cb);
        return Py_BuildValue("(iNNd)", leaf_id, node_path, legal_arr, term_q);
    }
    /* Non-terminal collapse may still have advanced the leaf — rebuild the
     * node_path array if so. */
    if (path_len != (int32_t)PyArray_DIM((PyArrayObject *)node_path, 0)) {
        Py_DECREF(node_path);
        npy_intp pdims2[1] = {path_len};
        node_path = PyArray_SimpleNew(1, pdims2, NPY_INT32);
        if (!node_path) { Py_DECREF(enc_arr); return NULL; }
        memcpy(PyArray_DATA((PyArrayObject *)node_path), path_buf,
               path_len * sizeof(int32_t));
    }

    /* Non-terminal: apply vloss along path[1:] before exposing the leaf to
     * other walkers — ensures the next walker that races in picks a
     * different child. */
    if (vloss_weight > 0) {
        tree_apply_vloss_path(t, path_buf, path_len);
    }

    /* Encode into enc_out[0]. */
    cboard_encode_146_into(&cb, enc_data);
    Py_DECREF(enc_arr);

    /* Legal move indices at the leaf (caller needs them for softmax). */
    int legal_tmp[256];
    int n_legal_i = cboard_legal_move_indices(&cb, legal_tmp, 0);
    if (n_legal_i < 0) n_legal_i = 0;
    npy_intp ldims[1] = {n_legal_i};
    PyObject *legal_arr = PyArray_SimpleNew(1, ldims, NPY_INT32);
    if (!legal_arr) {
        if (vloss_weight > 0) tree_remove_vloss_path(t, path_buf, path_len);
        Py_DECREF(node_path);
        return NULL;
    }
    int32_t *legal_out = (int32_t *)PyArray_DATA((PyArrayObject *)legal_arr);
    for (int i = 0; i < n_legal_i; i++) legal_out[i] = legal_tmp[i];

    return Py_BuildValue("(iNNO)", leaf_id, node_path, legal_arr, Py_None);
}


/* walker_integrate_leaf(node_path, legal, pol_logits, wdl_logits, vloss_weight)
 *   -> None
 *
 * Expands the leaf (if not already expanded), removes virtual loss along
 * path[1:], and backprops the WDL-derived Q up the path. Pairs with
 * walker_descend_puct. Safe for concurrent callers. */
static PyObject *MCTSTree_walker_integrate_leaf(MCTSTreeObject *self, PyObject *args) {
    PyObject *path_obj, *legal_obj, *pol_obj, *wdl_obj;
    int vloss_weight;
    if (!PyArg_ParseTuple(args, "OOOOi",
                          &path_obj, &legal_obj, &pol_obj, &wdl_obj, &vloss_weight))
        return NULL;

    PyArrayObject *path_arr = FROMANY_1D(path_obj, NPY_INT32);
    PyArrayObject *legal_arr = FROMANY_1D(legal_obj, NPY_INT32);
    PyArrayObject *pol_arr = FROMANY_1D(pol_obj, NPY_FLOAT32);
    PyArrayObject *wdl_arr = FROMANY_1D(wdl_obj, NPY_FLOAT32);
    if (!path_arr || !legal_arr || !pol_arr || !wdl_arr) {
        Py_XDECREF(path_arr); Py_XDECREF(legal_arr);
        Py_XDECREF(pol_arr); Py_XDECREF(wdl_arr);
        return NULL;
    }
    if (PyArray_DIM(pol_arr, 0) < 4672 || PyArray_DIM(wdl_arr, 0) < 3) {
        Py_DECREF(path_arr); Py_DECREF(legal_arr);
        Py_DECREF(pol_arr); Py_DECREF(wdl_arr);
        PyErr_SetString(PyExc_ValueError, "pol must be >=4672, wdl must be >=3");
        return NULL;
    }

    TreeData *t = &self->tree;
    int32_t path_len = (int32_t)PyArray_DIM(path_arr, 0);
    const int32_t *path = (const int32_t *)PyArray_DATA(path_arr);
    int32_t leaf_id = path[path_len - 1];
    int32_t n_legal = (int32_t)PyArray_DIM(legal_arr, 0);
    const int32_t *legal = (const int32_t *)PyArray_DATA(legal_arr);
    const float *pol = (const float *)PyArray_DATA(pol_arr);
    const float *wdl = (const float *)PyArray_DATA(wdl_arr);
    double q = wdl_logits_to_q((double)wdl[0], (double)wdl[1], (double)wdl[2]);

    /* Expand if not already. tree_expand is idempotent under the tree_lock
     * so a racing walker's expansion is absorbed here as a no-op. */
    if (n_legal > 0 &&
        !__atomic_load_n(&t->expanded[leaf_id], __ATOMIC_ACQUIRE)) {
        double priors_stack[256];
        double *priors = (n_legal <= 256) ? priors_stack
                                          : (double *)malloc(n_legal * sizeof(double));
        if (!priors) {
            Py_DECREF(path_arr); Py_DECREF(legal_arr);
            Py_DECREF(pol_arr); Py_DECREF(wdl_arr);
            return PyErr_NoMemory();
        }
        for (int32_t j = 0; j < n_legal; j++) priors[j] = (double)pol[legal[j]];
        softmax_inplace(priors, n_legal);
        tree_expand(t, leaf_id, legal, priors, n_legal);
        if (priors != priors_stack) free(priors);
    }

    /* Remove vloss BEFORE backprop so the vloss unwind and the real +1
     * visit can coalesce: every node on path ends at N += 1, vl unchanged. */
    if (vloss_weight > 0) {
        tree_remove_vloss_path(t, path, path_len);
    }
    tree_backprop(t, path, path_len, q);

    Py_DECREF(path_arr); Py_DECREF(legal_arr);
    Py_DECREF(pol_arr); Py_DECREF(wdl_arr);
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
 * Returns slot index (offset from children_offset).
 * When vloss_weight > 0, child virtual_loss counts like visits (with a
 * penalty on Q) so concurrent walkers diverge. */
static int32_t tree_gumbel_select_child(const TreeData *t, int32_t node_id,
                                         double c_scale, double c_visit,
                                         int vloss_weight) {
    int32_t n_ch = t->num_children[node_id];
    int32_t off = t->children_offset[node_id];
    if (n_ch <= 0) return 0;

    /* Solved-aware fast paths (mirrors tree_select_child). Gated on
     * has_solved_child[node_id] so the no-proofs-yet common case skips the
     * pre-pass. When 0, all children are unknown ⇒ any_unsolved_or_draw=1,
     * winning_slot stays -1, no exclusions in the main scoring pass.
     *   - any SOLVED_LOSS child (winning move for parent) → pick highest-prior one.
     *   - exclude SOLVED_WIN children (losing for parent) unless every child is
     *     SOLVED_WIN, in which case we fall through (parent already lost). */
    int32_t any_unsolved_or_draw = 1;
    int32_t check_solved = t->has_solved_child[node_id];
    if (check_solved) {
        int32_t winning_slot = -1;
        double  winning_prior = -1.0;
        any_unsolved_or_draw = 0;
        for (int32_t i = 0; i < n_ch; i++) {
            int8_t cs = t->solved[t->child_node[off + i]];
            if (cs == SOLVED_LOSS) {
                double pr = t->prior[t->child_node[off + i]];
                if (pr > winning_prior) { winning_prior = pr; winning_slot = i; }
            } else if (cs != SOLVED_WIN) {
                any_unsolved_or_draw = 1;
            }
        }
        if (winning_slot >= 0) return winning_slot;
    }

    /* Parent's effective N also includes its own in-flight visits so
     * completed_Q for unvisited children reflects the penalized parent. */
    int32_t parent_vl = (vloss_weight > 0) ? t->virtual_loss[node_id] : 0;
    double parent_N = (double)(t->N[node_id] + vloss_weight * parent_vl);
    double parent_W = t->W[node_id] + (double)(vloss_weight * parent_vl);
    double parent_Q = (parent_N > 0) ? (parent_W / parent_N) : 0.0;

    /* Two arrays — log_prior and completed_Q — precomputed in the same pass
     * that computes max_visit + total_visits. Avoids re-reading child arrays. */
    double *log_priors = (double *)alloca(n_ch * sizeof(double));
    double *cqs        = (double *)alloca(n_ch * sizeof(double));
    int32_t *eff_ns    = (int32_t *)alloca(n_ch * sizeof(int32_t));
    int8_t  *exclude   = (int8_t *)alloca(n_ch * sizeof(int8_t));
    int32_t max_visit = 0;
    int32_t total_visits = 0;
    for (int32_t i = 0; i < n_ch; i++) {
        int32_t cid = t->child_node[off + i];
        int8_t  cs  = check_solved ? t->solved[cid] : SOLVED_UNKNOWN;
        /* Exclude SOLVED_WIN children when alternatives exist. Mark in
         * `exclude` and skip them in scoring + argmax below. */
        exclude[i] = (any_unsolved_or_draw && cs == SOLVED_WIN) ? 1 : 0;
        int32_t vl = (vloss_weight > 0) ? t->virtual_loss[cid] : 0;
        int32_t eff_n = t->N[cid] + vloss_weight * vl;
        eff_ns[i] = eff_n;
        if (!exclude[i] && eff_n > max_visit) max_visit = eff_n;
        if (!exclude[i]) total_visits += eff_n;
        log_priors[i] = log(t->prior[cid] > 1e-12 ? t->prior[cid] : 1e-12);
        /* Virtual loss adds vloss_weight per in-flight walker to both N and
         * W from the child's parent perspective; child-perspective Q inverts
         * the sign on W (parent and child alternate sides).
         *
         * For SOLVED_DRAW children the proven Q from parent's frame is 0
         * (drawing line); use that instead of the noisy averaged W/N. */
        if (cs == SOLVED_DRAW) {
            cqs[i] = 0.0;
        } else {
            cqs[i] = (eff_n > 0)
                ? (-(t->W[cid] + (double)(vloss_weight * vl)) / (double)eff_n)
                : parent_Q;
        }
    }
    double sigma = c_scale * (c_visit + (double)max_visit);
    double inv_total = 1.0 / (1.0 + (double)total_visits);

    /* Fuse score-compute with softmax-max tracking. Excluded slots get
     * -infinity so they're invisible to the softmax. */
    double *scores = (double *)alloca(n_ch * sizeof(double));
    double max_score = -INFINITY;
    for (int32_t i = 0; i < n_ch; i++) {
        if (exclude[i]) { scores[i] = -INFINITY; continue; }
        double s = log_priors[i] + sigma * cqs[i];
        scores[i] = s;
        if (s > max_score) max_score = s;
    }

    /* exp + sum, with fallback to uniform if all inputs are non-finite. */
    double sum = 0.0;
    if (isfinite(max_score)) {
        for (int32_t i = 0; i < n_ch; i++) {
            if (exclude[i]) { scores[i] = 0.0; continue; }
            scores[i] = exp(scores[i] - max_score);
            sum += scores[i];
        }
    }
    int uniform_fallback = !(sum > 0.0 && isfinite(sum));
    /* Uniform fallback only over the non-excluded slot count. */
    int32_t n_active = 0;
    for (int32_t i = 0; i < n_ch; i++) if (!exclude[i]) n_active++;
    double inv_sum = uniform_fallback
        ? (n_active > 0 ? (1.0 / (double)n_active) : (1.0 / (double)n_ch))
        : (1.0 / sum);

    /* Fused normalize + argmax(prob - N/(1+total_N)). */
    int32_t best_slot = -1;
    double best_val = -1e30;
    for (int32_t i = 0; i < n_ch; i++) {
        if (exclude[i]) continue;
        double p = uniform_fallback ? inv_sum : (scores[i] * inv_sum);
        double target = p - (double)eff_ns[i] * inv_total;
        if (target > best_val) {
            best_val = target;
            best_slot = i;
        }
    }
    /* If everything was excluded (all children solved-WIN against parent),
     * fall back to slot 0 — keeps N/W bookkeeping consistent. */
    return best_slot >= 0 ? best_slot : 0;
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

    /* Acquire-load pairs with release-store in tree_expand. */
    while (__atomic_load_n(&t->expanded[node], __ATOMIC_ACQUIRE) && t->num_children[node] > 0) {
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
            /* In-flight walkers count as visits for divergence: if vloss>0
             * on an unvisited child, PUCT still has enough signal to pick a
             * different child for this walker. */
            int32_t eff_n = t->N[cid];
            if (sel->vloss_weight > 0) eff_n += t->virtual_loss[cid];
            if (eff_n > 0) { any_visited = 1; break; }
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
            slot = tree_gumbel_select_child(t, node, sel->c_scale, sel->c_visit,
                                            sel->vloss_weight);
        } else {
            slot = tree_select_child(t, node, sel->c_puct, sel->fpu_reduction,
                                     sel->vloss_weight);
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


/* mark_solved_path(node_path, status) -> None
 *
 * Direct entry point for marking a node solved + propagating ancestors.
 * Used by tests and (forthcoming) by the phase-2 brute-force terminal
 * extender that runs a few plies past frontier leaves to discover proven
 * lines. status is SOLVED_WIN (+1), SOLVED_LOSS (-1), or SOLVED_DRAW (+2)
 * from the leaf's STM perspective; SOLVED_UNKNOWN is silently ignored. */
static PyObject *MCTSTree_mark_solved_path(MCTSTreeObject *self, PyObject *args) {
    PyObject *path_obj;
    int status;
    if (!PyArg_ParseTuple(args, "Oi", &path_obj, &status)) return NULL;
    PyArrayObject *p = FROMANY_1D(path_obj, NPY_INT32);
    if (!p) return NULL;
    npy_intp n = PyArray_DIM(p, 0);
    tree_mark_solved_and_propagate(&self->tree,
        (const int32_t *)PyArray_DATA(p), (int32_t)n, (int8_t)status);
    Py_DECREF(p);
    Py_RETURN_NONE;
}


/* get_solved_status(node_id) -> int (SOLVED_UNKNOWN/WIN/LOSS/DRAW). */
static PyObject *MCTSTree_get_solved_status(MCTSTreeObject *self, PyObject *args) {
    int node_id;
    if (!PyArg_ParseTuple(args, "i", &node_id)) return NULL;
    if (node_id < 0 || node_id >= self->tree.node_count)
        return PyLong_FromLong(SOLVED_UNKNOWN);
    return PyLong_FromLong((long)self->tree.solved[node_id]);
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


/* memory_bytes() -> int — rough total bytes allocated for the tree's own
 * buffers. Sums the realloc'd node arrays (sized to node_cap), the children
 * pool (child_cap), the lazy CBoard cache (cb_cache_cap, a big per-entry
 * cost because CBoard contains history stacks), per-node hashes, and the
 * transposition hash table. Enough for a "halt before OOM" safety check. */
static PyObject *MCTSTree_memory_bytes(MCTSTreeObject *self, PyObject *Py_UNUSED(args)) {
    const TreeData *t = &self->tree;
    int64_t bytes = 0;
    /* Per-node arrays sized to node_cap. */
    bytes += (int64_t)t->node_cap * (
        sizeof(int32_t)   /* N */
      + sizeof(double)    /* W */
      + sizeof(double)    /* prior */
      + sizeof(int8_t)    /* expanded */
      + sizeof(int32_t)   /* parent */
      + sizeof(int32_t)   /* action_from_parent */
      + sizeof(int32_t)   /* num_children */
      + sizeof(int32_t)   /* children_offset */
      + sizeof(int32_t)   /* virtual_loss */
      + sizeof(uint64_t)  /* node_hash */
    );
    /* Children pool. */
    bytes += (int64_t)t->child_cap * (sizeof(int32_t) + sizeof(int32_t));
    /* CBoard cache (only if lazily allocated). */
    if (t->cb_cache) {
        bytes += (int64_t)t->cb_cache_cap * (sizeof(CBoard) + sizeof(int8_t));
    }
    /* Transposition hash table. */
    bytes += (int64_t)t->ht_cap * sizeof(int32_t);
    return PyLong_FromLongLong(bytes);
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


/* reserve(node_cap, child_cap=0) -> None.
 *
 * Pre-grows the tree's per-node arrays (and child arrays if child_cap>0)
 * so that no tree_grow_nodes / tree_grow_children realloc fires during
 * subsequent concurrent descent. Required before running multiple walkers
 * against one tree — otherwise a reader can dereference a stale pointer
 * after a mid-descent realloc. Safe to call at any time.
 *
 * Idempotent: shrinking not supported (calls with smaller caps are no-ops). */
static PyObject *MCTSTree_reserve(MCTSTreeObject *self, PyObject *args) {
    int node_cap, child_cap = 0;
    if (!PyArg_ParseTuple(args, "i|i", &node_cap, &child_cap)) return NULL;
    TreeData *t = &self->tree;
    pthread_mutex_lock(&t->tree_lock);
    while (t->node_cap < node_cap) {
        if (tree_grow_nodes(t) < 0) {
            pthread_mutex_unlock(&t->tree_lock);
            PyErr_NoMemory();
            return NULL;
        }
    }
    if (child_cap > t->child_count) {
        int32_t need = child_cap - t->child_count;
        if (tree_grow_children(t, need) < 0) {
            pthread_mutex_unlock(&t->tree_lock);
            PyErr_NoMemory();
            return NULL;
        }
    }
    pthread_mutex_unlock(&t->tree_lock);
    Py_RETURN_NONE;
}


/* get_virtual_loss(node_id) -> int. Accessor for tests of the walker-pool
 * vloss scaffolding (phase 2+3). Returns 0 for out-of-range ids rather than
 * raising — matches is_expanded() / node_q() behavior on bad ids. */
static PyObject *MCTSTree_get_virtual_loss(MCTSTreeObject *self, PyObject *args) {
    int node_id;
    if (!PyArg_ParseTuple(args, "i", &node_id))
        return NULL;
    if (node_id < 0 || node_id >= self->tree.node_count)
        return PyLong_FromLong(0);
    return PyLong_FromLong(self->tree.virtual_loss[node_id]);
}


/* Shared parser for apply/remove_vloss_path(path_int32).
 * On success returns a borrowed pointer to path data + length; on failure
 * sets a Python error and returns NULL. Caller owns the returned
 * PyArrayObject reference in *out_arr — must Py_DECREF it. */
static int32_t *MCTSTree_parse_vloss_path_args(
    PyObject *args, int32_t node_count, int32_t *out_len, PyArrayObject **out_arr
) {
    PyObject *path_obj;
    if (!PyArg_ParseTuple(args, "O", &path_obj)) return NULL;
    PyArrayObject *path_arr = FROMANY_1D(path_obj, NPY_INT32);
    if (!path_arr) return NULL;
    int32_t path_len = (int32_t)PyArray_DIM(path_arr, 0);
    int32_t *path_data = (int32_t *)PyArray_DATA(path_arr);
    for (int32_t i = 0; i < path_len; i++) {
        if (path_data[i] < 0 || path_data[i] >= node_count) {
            Py_DECREF(path_arr);
            PyErr_SetString(PyExc_ValueError, "path contains out-of-range node id");
            return NULL;
        }
    }
    *out_len = path_len;
    *out_arr = path_arr;
    return path_data;
}


/* apply_vloss_path(path_int32) -> None.
 * Path[0] is treated as the root and skipped (walker-pool design: all
 * walkers share the root, so vloss there doesn't steer them apart). */
static PyObject *MCTSTree_apply_vloss_path(MCTSTreeObject *self, PyObject *args) {
    int32_t path_len;
    PyArrayObject *path_arr;
    int32_t *path_data = MCTSTree_parse_vloss_path_args(
        args, self->tree.node_count, &path_len, &path_arr);
    if (!path_data) return NULL;
    tree_apply_vloss_path(&self->tree, path_data, path_len);
    Py_DECREF(path_arr);
    Py_RETURN_NONE;
}


/* remove_vloss_path(path_int32) -> None. Pairs with apply_vloss_path. */
static PyObject *MCTSTree_remove_vloss_path(MCTSTreeObject *self, PyObject *args) {
    int32_t path_len;
    PyArrayObject *path_arr;
    int32_t *path_data = MCTSTree_parse_vloss_path_args(
        args, self->tree.node_count, &path_len, &path_arr);
    if (!path_data) return NULL;
    tree_remove_vloss_path(&self->tree, path_data, path_len);
    Py_DECREF(path_arr);
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
        __atomic_store_n(&self->tree.expanded[node_id], 1, __ATOMIC_RELEASE);
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
 *                   full_tree, enc_buf[, nn_cache, vloss_weight])
 *
 * Initializes the gumbel simulation state machine and runs until
 * the first batch of positions needs GPU eval.
 * Returns n_leaves (int) or None if simulation completed immediately.
 *
 * vloss_weight (default 0): penalty applied to in-flight child paths during
 * PUCT descent. Only matters when walker threads are running concurrent
 * descends; 0 keeps behavior bit-identical to the pre-vloss code path.
 */
static PyObject *MCTSTree_start_gumbel_sims(MCTSTreeObject *self, PyObject *args) {
    PyObject *root_cbs_list, *root_ids_obj, *remaining_list, *gumbels_list;
    PyObject *priors_list, *budget_obj, *root_qs_obj, *enc_buf_obj;
    double c_scale, c_visit, c_puct, fpu_reduction;
    int full_tree;
    PyObject *nn_cache_obj = NULL;
    int vloss_weight = 0;
    int target_batch = 0;  /* 0 => use GSS_GPU_BATCH default in gss_step */

    if (!PyArg_ParseTuple(args, "OOOOOOOddddpO|Oii",
                          &root_cbs_list, &root_ids_obj, &remaining_list,
                          &gumbels_list, &priors_list, &budget_obj, &root_qs_obj,
                          &c_scale, &c_visit, &c_puct, &fpu_reduction, &full_tree,
                          &enc_buf_obj, &nn_cache_obj, &vloss_weight, &target_batch))
        return NULL;

    /* Validate list args before any indexing — PyList_GET_ITEM has no
     * bounds check, and a mismatched companion list would silently read
     * past the last element into arbitrary PyObject pointers. */
    if (!PyList_Check(root_cbs_list) || !PyList_Check(remaining_list) ||
        !PyList_Check(gumbels_list) || !PyList_Check(priors_list)) {
        PyErr_SetString(PyExc_TypeError,
            "root_cbs / remaining / gumbels / priors must be lists");
        return NULL;
    }
    int32_t n_boards = (int32_t)PyList_Size(root_cbs_list);
    if (n_boards <= 0) {
        PyErr_SetString(PyExc_ValueError, "root_cbs_list must be non-empty");
        return NULL;
    }
    if (PyList_Size(remaining_list) != n_boards ||
        PyList_Size(gumbels_list)   != n_boards ||
        PyList_Size(priors_list)    != n_boards) {
        PyErr_SetString(PyExc_ValueError,
            "remaining/gumbels/priors lists must match root_cbs_list length");
        return NULL;
    }

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

    if (PyArray_DIM(root_ids_arr, 0) < n_boards ||
        PyArray_DIM(budget_arr, 0)   < n_boards ||
        PyArray_DIM(root_qs_arr, 0)  < n_boards) {
        Py_DECREF(root_ids_arr); Py_DECREF(budget_arr);
        Py_DECREF(root_qs_arr); Py_DECREF(enc_arr);
        PyErr_SetString(PyExc_ValueError,
            "root_ids/budget/root_qs must have at least n_boards elements");
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
    g->sel.vloss_weight = (vloss_weight > 0) ? vloss_weight : 0;
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

    /* Validate root_id is -1 (inactive sentinel, used by dual-tree PBT) or
     * a real in-range node, before any blind indexing of tree arrays. */
    int32_t node_count = self->tree.node_count;
    for (int32_t i = 0; i < n_boards; i++) {
        int32_t rid = g->root_ids[i];
        if (rid != -1 && (rid < 0 || rid >= node_count)) {
            gss_free(g);
            Py_DECREF(root_ids_arr); Py_DECREF(budget_arr);
            Py_DECREF(root_qs_arr); Py_DECREF(enc_arr);
            PyErr_Format(PyExc_ValueError, "root_id[%d]=%d out of range [-1, %d)",
                         i, rid, node_count);
            return NULL;
        }
    }

    /* Validate root_cbs_list items — extract_cboards caches the PyCBoard
     * type pointer and rejects anything else. Do it before dereferencing
     * raw pcb->board through PyList_GET_ITEM. */
    if (extract_cboards(root_cbs_list, n_boards, g->root_cboards, NULL) < 0) {
        gss_free(g);
        Py_DECREF(root_ids_arr); Py_DECREF(budget_arr);
        Py_DECREF(root_qs_arr); Py_DECREF(enc_arr);
        return NULL;
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
    /* Validate candidate actions at ingest so the hot path (gss_score_and_halve)
     * can index root_priors[bi * 4672 + action] / gumbels[...] / action_to_slot
     * without per-use range checks. Cost is paid once here, not per halving. */
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
            if (v < 0 || v >= GSS_POLICY_SIZE) {
                gss_free(g);
                Py_DECREF(root_ids_arr); Py_DECREF(budget_arr);
                Py_DECREF(root_qs_arr); Py_DECREF(enc_arr);
                PyErr_Format(PyExc_ValueError,
                    "remaining_list[%d][%zd]=%ld out of range [0, %d)",
                    i, (Py_ssize_t)j, v, GSS_POLICY_SIZE);
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

    /* Minibatch target (0 => fall back to GSS_GPU_BATCH in gss_step) */
    g->target_batch = (int32_t)target_batch;

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
    if (gss_begin_round(g, &self->tree) == 0) {
        g->phase = 2;
        Py_RETURN_NONE;
    }

    /* Run simulation until GPU eval needed */
    int32_t n_leaves_start;
    Py_BEGIN_ALLOW_THREADS
    n_leaves_start = gss_step(&self->tree, s, g, g->enc_data);
    Py_END_ALLOW_THREADS

    if (n_leaves_start < 0) return PyErr_NoMemory();
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

    if (n_leaves_cont < 0) return PyErr_NoMemory();
    if (n_leaves_cont == 0) {
        g->phase = 2;
        Py_RETURN_NONE;
    }

    return PyLong_FromLong(n_leaves_cont);
}


/*
 * get_pending_tb_leaves(max_pieces) -> (indices: np.int32, list[CBoard])
 *
 * Between start/continue_gumbel_sims returning n_leaves > 0 and the caller
 * calling continue_gumbel_sims with NN outputs, StoredPrepState holds the
 * leaf CBoards corresponding to rows [0..n_leaves-1] of the encoded-planes
 * buffer. This method returns the Syzygy-eligible subset (popcount(occ) ≤
 * max_pieces AND castling == 0), paired with their original indices.
 *
 * Filtering in C skips the PyCBoard alloc + Python iteration for ineligible
 * leaves — during opening/middlegame almost every leaf fails the popcount
 * check, so the per-batch cost drops to a single popcount loop.
 */
static PyObject *MCTSTree_get_pending_tb_leaves(MCTSTreeObject *self, PyObject *args) {
    int max_pieces;
    if (!PyArg_ParseTuple(args, "i", &max_pieces))
        return NULL;
    StoredPrepState *s = &self->stored;
    GumbelSimState *g = &self->gsim;
    if (g->phase != 1) {
        PyErr_SetString(PyExc_RuntimeError,
            "get_pending_tb_leaves: no batch pending (phase != 1)");
        return NULL;
    }
    PyTypeObject *cb_type = _cached_cboard_type;
    if (!cb_type) {
        PyErr_SetString(PyExc_RuntimeError,
            "CBoard type not cached; call start_gumbel_sims first");
        return NULL;
    }
    int32_t n = s->n_leaves;

    /* Stack buffer sized for max MCTS batch; n_leaves ≤ GSS_GPU_BATCH which
     * is O(few hundred), so a fixed-size alloca-style array is fine. */
    int32_t elig_idx[4096];
    int32_t n_elig = 0;
    int32_t n_scan = (n < 4096) ? n : 4096;
    for (int32_t i = 0; i < n_scan; i++) {
        const CBoard *cb = &s->leaf_cboards[i];
        if (cb->castling != 0) continue;
        uint64_t occ = cb->occ[0] | cb->occ[1];
        if (popcount64(occ) > max_pieces) continue;
        elig_idx[n_elig++] = i;
    }

    npy_intp dims[1] = { n_elig };
    PyArrayObject *idx_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT32);
    if (!idx_arr) return NULL;
    if (n_elig > 0) {
        memcpy(PyArray_DATA(idx_arr), elig_idx, n_elig * sizeof(int32_t));
    }
    PyObject *lst = PyList_New(n_elig);
    if (!lst) { Py_DECREF(idx_arr); return NULL; }
    for (int32_t j = 0; j < n_elig; j++) {
        PyCBoard *cp = (PyCBoard *)cb_type->tp_alloc(cb_type, 0);
        if (!cp) { Py_DECREF(idx_arr); Py_DECREF(lst); return NULL; }
        cp->board = s->leaf_cboards[elig_idx[j]];
        PyList_SET_ITEM(lst, j, (PyObject *)cp);
    }
    PyObject *tup = PyTuple_Pack(2, (PyObject *)idx_arr, lst);
    Py_DECREF(idx_arr);
    Py_DECREF(lst);
    return tup;
}


/*
 * mark_tb_solved(indices: np.int32, statuses: np.int8) -> int (count marked)
 *
 * For each i in indices, look up the corresponding leaf in the current
 * pending batch (must be in phase 1, paired with get_pending_tb_leaves) and
 * mark its tree node solved with statuses[i] (one of SOLVED_WIN, SOLVED_LOSS,
 * SOLVED_DRAW from STM perspective). Then propagate the solved status up
 * each leaf's path. UNKNOWN entries are skipped silently.
 *
 * The two arrays must have the same length and indices[k] must reference a
 * leaf row in [0, n_leaves). The leaf's path is taken from
 * StoredPrepState.path_flat — same source the eventual backprop uses.
 */
static PyObject *MCTSTree_mark_tb_solved(MCTSTreeObject *self, PyObject *args) {
    PyObject *indices_obj, *statuses_obj;
    if (!PyArg_ParseTuple(args, "OO", &indices_obj, &statuses_obj)) return NULL;
    StoredPrepState *s = &self->stored;
    GumbelSimState *g = &self->gsim;
    if (g->phase != 1) {
        PyErr_SetString(PyExc_RuntimeError,
            "mark_tb_solved: no batch pending (phase != 1)");
        return NULL;
    }
    PyArrayObject *idx_arr = FROMANY_1D(indices_obj, NPY_INT32);
    if (!idx_arr) return NULL;
    PyArrayObject *st_arr = FROMANY_1D(statuses_obj, NPY_INT8);
    if (!st_arr) { Py_DECREF(idx_arr); return NULL; }
    npy_intp n_idx = PyArray_DIM(idx_arr, 0);
    if (n_idx != PyArray_DIM(st_arr, 0)) {
        Py_DECREF(idx_arr); Py_DECREF(st_arr);
        PyErr_SetString(PyExc_ValueError,
            "mark_tb_solved: indices and statuses length mismatch");
        return NULL;
    }
    const int32_t *indices = (const int32_t *)PyArray_DATA(idx_arr);
    const int8_t  *statuses = (const int8_t *)PyArray_DATA(st_arr);
    int32_t marked = 0;
    for (npy_intp k = 0; k < n_idx; k++) {
        int32_t i = indices[k];
        int8_t status = statuses[k];
        if (status == SOLVED_UNKNOWN) continue;
        if (i < 0 || i >= s->n_leaves) continue;
        const int32_t *path = s->path_flat + s->path_offset[i];
        int32_t plen = s->path_count[i];
        if (plen <= 0) continue;
        tree_mark_solved_and_propagate(&self->tree, path, plen, status);
        marked++;
    }
    Py_DECREF(idx_arr);
    Py_DECREF(st_arr);
    return PyLong_FromLong(marked);
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
    {"get_pending_tb_leaves", (PyCFunction)MCTSTree_get_pending_tb_leaves, METH_VARARGS,
     "get_pending_tb_leaves(max_pieces) -> (np.int32 indices, list[CBoard]). Syzygy-eligible subset of the current pending batch."},
    {"mark_tb_solved", (PyCFunction)MCTSTree_mark_tb_solved, METH_VARARGS,
     "mark_tb_solved(indices: np.int32, statuses: np.int8) -> int. Mark each indexed leaf with its TB-derived solved status (1=WIN, -1=LOSS, 2=DRAW; 0=skip) and propagate upward."},
    {"get_gumbel_remaining", (PyCFunction)MCTSTree_get_gumbel_remaining, METH_NOARGS,
     "get_gumbel_remaining() -> list[list[int]]. Get remaining candidates after sim."},
    {"get_children_visits", (PyCFunction)MCTSTree_get_children_visits, METH_VARARGS,
     "get_children_visits(node_id) -> (actions_int32, visits_int32)"},
    {"get_children_q", (PyCFunction)MCTSTree_get_children_q, METH_VARARGS,
     "get_children_q(node_id, default_q) -> (actions_int32, visits_int32, q_float64)"},
    {"node_q", (PyCFunction)MCTSTree_node_q, METH_VARARGS,
     "node_q(node_id) -> float"},
    {"get_solved_status", (PyCFunction)MCTSTree_get_solved_status, METH_VARARGS,
     "get_solved_status(node_id) -> int (0=unknown, 1=win, -1=loss, 2=draw, STM perspective)"},
    {"mark_solved_path", (PyCFunction)MCTSTree_mark_solved_path, METH_VARARGS,
     "mark_solved_path(node_path: NDArray[np.int32], status: int) -> None. Mark leaf solved with status (1/-1/2 = WIN/LOSS/DRAW from STM, 0=skip) and propagate upward."},
    {"is_expanded", (PyCFunction)MCTSTree_is_expanded, METH_VARARGS,
     "is_expanded(node_id) -> bool"},
    {"memory_bytes", (PyCFunction)MCTSTree_memory_bytes, METH_NOARGS,
     "memory_bytes() -> int. Approximate total bytes allocated for tree buffers."},
    {"node_count", (PyCFunction)MCTSTree_node_count, METH_NOARGS,
     "node_count() -> int"},
    {"reset", (PyCFunction)MCTSTree_reset, METH_NOARGS,
     "reset() -> None"},
    {"reserve", (PyCFunction)MCTSTree_reserve, METH_VARARGS,
     "reserve(node_cap, child_cap=0) -> None (pre-grow for concurrent use)"},
    {"get_virtual_loss", (PyCFunction)MCTSTree_get_virtual_loss, METH_VARARGS,
     "get_virtual_loss(node_id) -> int"},
    {"apply_vloss_path", (PyCFunction)MCTSTree_apply_vloss_path, METH_VARARGS,
     "apply_vloss_path(path: int32[]) -> None (skips path[0] = root)"},
    {"remove_vloss_path", (PyCFunction)MCTSTree_remove_vloss_path, METH_VARARGS,
     "remove_vloss_path(path: int32[]) -> None (floors at 0; pair with apply_vloss_path)"},
    {"walker_descend_puct", (PyCFunction)MCTSTree_walker_descend_puct, METH_VARARGS,
     "walker_descend_puct(root_id, root_cboard, c_puct, fpu_root, fpu_reduction, vloss_weight, enc_out) -> (leaf_id, node_path, legal, terminal_q_or_None)"},
    {"walker_integrate_leaf", (PyCFunction)MCTSTree_walker_integrate_leaf, METH_VARARGS,
     "walker_integrate_leaf(node_path, legal, pol_logits, wdl_logits, vloss_weight) -> None"},
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
