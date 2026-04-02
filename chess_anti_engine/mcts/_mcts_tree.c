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

/* Pure-C CBoard implementation (bitboard utilities, attack tables, move gen, CBoard) */
#include "../encoding/_cboard_impl.h"
/* Feature planes for fused encode_146 */
#include "../encoding/_features_impl.h"

/* PyCBoard layout — must match _lc0_ext.c's typedef exactly. */
typedef struct { PyObject_HEAD CBoard board; } PyCBoard;

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

    if (!t->N || !t->W || !t->prior || !t->expanded || !t->parent ||
        !t->action_from_parent || !t->num_children || !t->children_offset ||
        !t->child_action || !t->child_node) {
        return -1;
    }
    memset(t->parent, -1, t->node_cap * sizeof(int32_t));
    memset(t->action_from_parent, -1, t->node_cap * sizeof(int32_t));
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
    memset(t, 0, sizeof(TreeData));
}


static int tree_grow_nodes(TreeData *t) {
    int32_t new_cap = t->node_cap * 2;

    t->N = (int32_t *)realloc(t->N, new_cap * sizeof(int32_t));
    t->W = (double *)realloc(t->W, new_cap * sizeof(double));
    t->prior = (double *)realloc(t->prior, new_cap * sizeof(double));
    t->expanded = (int8_t *)realloc(t->expanded, new_cap * sizeof(int8_t));
    t->parent = (int32_t *)realloc(t->parent, new_cap * sizeof(int32_t));
    t->action_from_parent = (int32_t *)realloc(t->action_from_parent, new_cap * sizeof(int32_t));
    t->num_children = (int32_t *)realloc(t->num_children, new_cap * sizeof(int32_t));
    t->children_offset = (int32_t *)realloc(t->children_offset, new_cap * sizeof(int32_t));

    if (!t->N || !t->W || !t->prior || !t->expanded || !t->parent ||
        !t->action_from_parent || !t->num_children || !t->children_offset) {
        return -1;
    }

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

    t->node_cap = new_cap;
    return 0;
}


static int tree_grow_children(TreeData *t, int32_t need) {
    int32_t new_cap = t->child_cap;
    while (new_cap < t->child_count + need)
        new_cap *= 2;
    if (new_cap == t->child_cap) return 0;

    t->child_action = (int32_t *)realloc(t->child_action, new_cap * sizeof(int32_t));
    t->child_node = (int32_t *)realloc(t->child_node, new_cap * sizeof(int32_t));
    if (!t->child_action || !t->child_node) return -1;

    t->child_cap = new_cap;
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


/* Expand a node: add children for each (action, prior) pair. */
static int tree_expand(TreeData *t, int32_t node_id,
                       const int32_t *actions, const double *priors, int32_t n_children) {
    if (n_children <= 0) {
        t->expanded[node_id] = 1;
        return 0;
    }

    if (tree_grow_children(t, n_children) < 0) return -1;

    int32_t offset = t->child_count;
    t->children_offset[node_id] = offset;
    t->num_children[node_id] = n_children;
    t->expanded[node_id] = 1;

    for (int32_t i = 0; i < n_children; i++) {
        int32_t child_id = tree_add_node(t, node_id, actions[i], priors[i]);
        if (child_id < 0) return -1;
        t->child_action[offset + i] = actions[i];
        t->child_node[offset + i] = child_id;
    }
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

    /* FPU: compute visited policy mass */
    double visited_policy = 0.0;
    for (int32_t i = 0; i < n_ch; i++) {
        int32_t cid = t->child_node[off + i];
        if (t->N[cid] > 0)
            visited_policy += t->prior[cid];
    }
    double fpu_value = parent_Q - fpu_reduction * sqrt(visited_policy);

    int32_t best_slot = 0;
    double best_score = -1e30;

    for (int32_t i = 0; i < n_ch; i++) {
        int32_t cid = t->child_node[off + i];
        int32_t n = t->N[cid];
        /* Child W/Q is stored from the child's side-to-move perspective.
         * Negate visited children so all scores are compared in the parent frame. */
        double q = (n > 0) ? (-t->W[cid] / (double)n) : fpu_value;
        double score = q + c_sqrt_n * t->prior[cid] / (1.0 + (double)n);
        if (score > best_score) {
            best_score = score;
            best_slot = i;
        }
    }
    return best_slot;
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

typedef struct {
    PyObject_HEAD
    TreeData tree;
} MCTSTreeObject;


static void MCTSTree_dealloc(MCTSTreeObject *self) {
    tree_free(&self->tree);
    Py_TYPE(self)->tp_free((PyObject *)self);
}


static int MCTSTree_init(MCTSTreeObject *self, PyObject *args, PyObject *kwds) {
    (void)args;
    (void)kwds;
    if (tree_init(&self->tree) < 0) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate MCTS tree");
        return -1;
    }
    return 0;
}


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

    PyArrayObject *actions_arr = (PyArrayObject *)PyArray_FROMANY(
        actions_obj, NPY_INT32, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *priors_arr = (PyArrayObject *)PyArray_FROMANY(
        priors_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_C_CONTIGUOUS);

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

    PyArrayObject *root_ids_arr = (PyArrayObject *)PyArray_FROMANY(
        root_ids_obj, NPY_INT32, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
    if (!root_ids_arr) return NULL;

    int32_t n_roots = (int32_t)PyArray_SIZE(root_ids_arr);
    const int32_t *root_ids = (const int32_t *)PyArray_DATA(root_ids_arr);

    /* Max path depth 512 should be plenty for any chess game tree */
    int32_t path_buf[512];

    PyObject *result = PyList_New(n_roots);
    if (!result) {
        Py_DECREF(root_ids_arr);
        return NULL;
    }

    for (int32_t i = 0; i < n_roots; i++) {
        int32_t root_id = root_ids[i];
        int32_t path_len = tree_select_leaf(&self->tree, root_id,
                                             c_puct, fpu_at_root, fpu_reduction,
                                             path_buf, 512);

        int32_t leaf_id = path_buf[path_len - 1];

        /* Build action path from root to leaf */
        int32_t action_buf[512];
        int32_t action_len = tree_action_path(&self->tree, leaf_id, action_buf, 512);
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

    PyArrayObject *path_arr = (PyArrayObject *)PyArray_FROMANY(
        path_obj, NPY_INT32, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
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

        PyArrayObject *path_arr = (PyArrayObject *)PyArray_FROMANY(
            path_obj, NPY_INT32, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
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

    /* Find max visit count for sigma */
    int32_t max_visit = 0;
    int32_t total_visits = 0;
    for (int32_t i = 0; i < n_ch; i++) {
        int32_t cid = t->child_node[off + i];
        int32_t n = t->N[cid];
        if (n > max_visit) max_visit = n;
        total_visits += n;
    }
    double sigma = c_scale * (c_visit + (double)max_visit);
    double inv_total = 1.0 / (1.0 + (double)total_visits);

    /* Compute logits + sigma * completed_Q, then softmax */
    double *scores = (double *)alloca(n_ch * sizeof(double));
    for (int32_t i = 0; i < n_ch; i++) {
        int32_t cid = t->child_node[off + i];
        int32_t n = t->N[cid];
        double log_prior = log(t->prior[cid] > 1e-12 ? t->prior[cid] : 1e-12);
        double cq = (n > 0) ? (-t->W[cid] / (double)n) : parent_Q;
        scores[i] = log_prior + sigma * cq;
    }

    /* Softmax */
    softmax_inplace(scores, n_ch);

    /* Select: argmax(prob - N/(1+total_N)) */
    int32_t best_slot = 0;
    double best_val = -1e30;
    for (int32_t i = 0; i < n_ch; i++) {
        int32_t cid = t->child_node[off + i];
        double target = scores[i] - (double)t->N[cid] * inv_total;
        if (target > best_val) {
            best_val = target;
            best_slot = i;
        }
    }
    return best_slot;
}


/* Traverse: follow forced_action from root, then improved-policy below.
 * Returns path length. path[0]=root, path[len-1]=leaf.
 * If leaf is terminal (no children, not expanded with 0 visits), sets *is_leaf=1. */
static int32_t tree_gumbel_collect_leaf(const TreeData *t, int32_t root_id,
                                         int32_t forced_action,
                                         double c_scale, double c_visit,
                                         double c_puct, double fpu_reduction,
                                         int full_tree,
                                         int32_t *path, int32_t max_path) {
    int32_t depth = 0;
    path[depth++] = root_id;

    /* Step 1: follow forced_action from root */
    int32_t n_ch = t->num_children[root_id];
    int32_t off = t->children_offset[root_id];
    int32_t child_id = -1;
    for (int32_t i = 0; i < n_ch; i++) {
        if (t->child_action[off + i] == forced_action) {
            child_id = t->child_node[off + i];
            break;
        }
    }
    if (child_id < 0) return depth; /* forced action not found */

    path[depth++] = child_id;
    int32_t node = child_id;

    /* Step 2: traverse using improved-policy (or PUCT) below */
    while (t->expanded[node] && t->num_children[node] > 0) {
        if (depth >= max_path) break;

        /* Check if any children have been visited */
        int32_t n_ch2 = t->num_children[node];
        int32_t off2 = t->children_offset[node];
        int32_t any_visited = 0;
        for (int32_t i = 0; i < n_ch2; i++) {
            if (t->N[t->child_node[off2 + i]] > 0) { any_visited = 1; break; }
        }

        int32_t slot;
        if (!any_visited) {
            /* Frontier: all children unvisited — select highest prior */
            slot = 0;
            double best_pri = -1.0;
            for (int32_t i = 0; i < n_ch2; i++) {
                int32_t cid = t->child_node[off2 + i];
                if (t->prior[cid] > best_pri) {
                    best_pri = t->prior[cid];
                    slot = i;
                }
            }
        } else if (full_tree) {
            slot = tree_gumbel_select_child(t, node, c_scale, c_visit);
        } else {
            slot = tree_select_child(t, node, c_puct, fpu_reduction);
        }
        node = t->child_node[t->children_offset[node] + slot];
        path[depth++] = node;
    }

    return depth;
}


/*
 * gumbel_collect_leaves(root_ids, forced_actions, c_scale, c_visit,
 *                       c_puct, fpu_reduction, full_tree)
 *   -> (leaf_ids: int32[], node_paths: list[int32[]], action_paths: list[int32[]])
 *
 * Batch collect one leaf per (root, forced_action) pair using Gumbel traversal.
 * Returns leaf node IDs, node paths (for backprop), and action paths (for board replay).
 */
static PyObject *MCTSTree_gumbel_collect_leaves(MCTSTreeObject *self, PyObject *args) {
    PyObject *root_ids_obj, *forced_actions_obj;
    double c_scale, c_visit, c_puct, fpu_reduction;
    int full_tree;
    if (!PyArg_ParseTuple(args, "OOddddp",
                          &root_ids_obj, &forced_actions_obj,
                          &c_scale, &c_visit, &c_puct, &fpu_reduction, &full_tree))
        return NULL;

    PyArrayObject *root_ids_arr = (PyArrayObject *)PyArray_FROMANY(
        root_ids_obj, NPY_INT32, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *forced_arr = (PyArrayObject *)PyArray_FROMANY(
        forced_actions_obj, NPY_INT32, 1, 1, NPY_ARRAY_C_CONTIGUOUS);

    if (!root_ids_arr || !forced_arr) {
        Py_XDECREF(root_ids_arr);
        Py_XDECREF(forced_arr);
        return NULL;
    }

    int32_t n = (int32_t)PyArray_SIZE(root_ids_arr);
    const int32_t *root_ids = (const int32_t *)PyArray_DATA(root_ids_arr);
    const int32_t *forced_actions = (const int32_t *)PyArray_DATA(forced_arr);

    int32_t path_buf[512];

    /* Output arrays */
    npy_intp dims[1] = {n};
    PyObject *leaf_ids = PyArray_SimpleNew(1, dims, NPY_INT32);
    PyObject *node_paths_list = PyList_New(n);
    PyObject *action_paths_list = PyList_New(n);

    if (!leaf_ids || !node_paths_list || !action_paths_list) {
        Py_XDECREF(leaf_ids);
        Py_XDECREF(node_paths_list);
        Py_XDECREF(action_paths_list);
        Py_DECREF(root_ids_arr);
        Py_DECREF(forced_arr);
        return NULL;
    }

    int32_t *leaf_data = (int32_t *)PyArray_DATA((PyArrayObject *)leaf_ids);

    for (int32_t i = 0; i < n; i++) {
        int32_t path_len = tree_gumbel_collect_leaf(
            &self->tree, root_ids[i], forced_actions[i],
            c_scale, c_visit, c_puct, fpu_reduction, full_tree,
            path_buf, 512);

        int32_t leaf_id = path_buf[path_len - 1];
        leaf_data[i] = leaf_id;

        /* Node path */
        npy_intp pdims[1] = {path_len};
        PyObject *np_arr = PyArray_SimpleNew(1, pdims, NPY_INT32);
        if (np_arr) {
            memcpy(PyArray_DATA((PyArrayObject *)np_arr),
                   path_buf, path_len * sizeof(int32_t));
        }
        if (!np_arr) { Py_INCREF(Py_None); np_arr = Py_None; }
        PyList_SET_ITEM(node_paths_list, i, np_arr);

        /* Action path (from root to leaf) */
        int32_t action_buf[512];
        int32_t action_len = tree_action_path(&self->tree, leaf_id, action_buf, 512);
        if (action_len < 0) action_len = 0;
        npy_intp adims[1] = {action_len};
        PyObject *ap_arr = PyArray_SimpleNew(1, adims, NPY_INT32);
        if (ap_arr) {
            memcpy(PyArray_DATA((PyArrayObject *)ap_arr),
                   action_buf, action_len * sizeof(int32_t));
        }
        if (!ap_arr) { Py_INCREF(Py_None); ap_arr = Py_None; }
        PyList_SET_ITEM(action_paths_list, i, ap_arr);
    }

    Py_DECREF(root_ids_arr);
    Py_DECREF(forced_arr);
    PyObject *result = PyTuple_Pack(3, leaf_ids, node_paths_list, action_paths_list);
    Py_DECREF(leaf_ids);
    Py_DECREF(node_paths_list);
    Py_DECREF(action_paths_list);
    return result;
}


/*
 * gumbel_score_candidates(root_id, candidate_actions, gumbel_values,
 *                         root_priors_full, c_scale, c_visit)
 *   -> float64 array of scores per candidate
 *
 * For sequential halving: compute sigma_scale * completed_Q + log_prior + gumbel.
 */
static PyObject *MCTSTree_gumbel_score_candidates(MCTSTreeObject *self, PyObject *args) {
    int root_id;
    PyObject *cands_obj, *gumbels_obj, *priors_obj;
    double c_scale, c_visit;
    if (!PyArg_ParseTuple(args, "iOOOdd",
                          &root_id, &cands_obj, &gumbels_obj, &priors_obj,
                          &c_scale, &c_visit))
        return NULL;

    PyArrayObject *cands_arr = (PyArrayObject *)PyArray_FROMANY(
        cands_obj, NPY_INT32, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *gumbels_arr = (PyArrayObject *)PyArray_FROMANY(
        gumbels_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *priors_arr = (PyArrayObject *)PyArray_FROMANY(
        priors_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_C_CONTIGUOUS);

    if (!cands_arr || !gumbels_arr || !priors_arr) {
        Py_XDECREF(cands_arr); Py_XDECREF(gumbels_arr); Py_XDECREF(priors_arr);
        return NULL;
    }

    int32_t n_cands = (int32_t)PyArray_SIZE(cands_arr);
    const int32_t *cands = (const int32_t *)PyArray_DATA(cands_arr);
    const double *gumbels = (const double *)PyArray_DATA(gumbels_arr);
    const double *priors_full = (const double *)PyArray_DATA(priors_arr);

    TreeData *t = &self->tree;
    double root_Q = (t->N[root_id] > 0) ? (t->W[root_id] / (double)t->N[root_id]) : 0.0;

    /* Find max visit among root's children */
    int32_t n_ch = t->num_children[root_id];
    int32_t off = t->children_offset[root_id];
    int32_t max_visit = 0;
    for (int32_t i = 0; i < n_ch; i++) {
        int32_t n = t->N[t->child_node[off + i]];
        if (n > max_visit) max_visit = n;
    }
    double sigma = c_scale * (c_visit + (double)max_visit);

    /* Build a fast lookup: action → child node id */
    /* Since n_ch is small (~20), linear scan is fine */

    npy_intp out_dims[1] = {n_cands};
    PyObject *out = PyArray_SimpleNew(1, out_dims, NPY_FLOAT64);
    if (!out) {
        Py_DECREF(cands_arr); Py_DECREF(gumbels_arr); Py_DECREF(priors_arr);
        return NULL;
    }
    double *out_data = (double *)PyArray_DATA((PyArrayObject *)out);

    for (int32_t ci = 0; ci < n_cands; ci++) {
        int32_t action = cands[ci];
        double log_prior = log(priors_full[action] > 1e-12 ? priors_full[action] : 1e-12);

        /* Find child for this action */
        double q_hat = root_Q;
        for (int32_t j = 0; j < n_ch; j++) {
            if (t->child_action[off + j] == action) {
                int32_t cid = t->child_node[off + j];
                int32_t n = t->N[cid];
                if (n > 0) q_hat = -t->W[cid] / (double)n;
                break;
            }
        }
        out_data[ci] = gumbels[ci] + log_prior + sigma * q_hat;
    }

    Py_DECREF(cands_arr); Py_DECREF(gumbels_arr); Py_DECREF(priors_arr);
    return out;
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

    PyArrayObject *legal_arr = (PyArrayObject *)PyArray_FROMANY(
        legal_obj, NPY_INT32, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *logits_arr = (PyArrayObject *)PyArray_FROMANY(
        logits_obj, NPY_FLOAT32, 1, 1, NPY_ARRAY_C_CONTIGUOUS);

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

    PyArrayObject *wdl_arr = (PyArrayObject *)PyArray_FROMANY(
        wdl_obj, NPY_FLOAT32, 2, 2, NPY_ARRAY_C_CONTIGUOUS);
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
        double w = (double)wdl[i * 3 + 0];
        double d = (double)wdl[i * 3 + 1];
        double l = (double)wdl[i * 3 + 2];
        double mx = w;
        if (d > mx) mx = d;
        if (l > mx) mx = l;
        double ew = exp(w - mx);
        double ed = exp(d - mx);
        double el = exp(l - mx);
        double s = ew + ed + el;
        q[i] = (s > 0.0) ? ((ew - el) / s) : 0.0;
    }

    Py_DECREF(wdl_arr);
    return out;
}


/*
 * Encode a CBoard into 146 float32 planes (112 LC0 + 34 features).
 * Writes into a pre-allocated buffer — no Python/numpy allocation.
 */
static void cboard_encode_146_into(const CBoard *b, float *out) {
    memset(out, 0, 146 * 64 * sizeof(float));
    cboard_fill_lc0_112(b, out);
    int us = b->turn, them = 1 - us;
    uint64_t us_pieces[6], them_pieces[6];
    for (int i = 0; i < 6; i++) {
        us_pieces[i] = b->bb[i] & b->occ[us];
        them_pieces[i] = b->bb[i] & b->occ[them];
    }
    uint64_t us_king = b->bb[KING] & b->occ[us];
    uint64_t them_king = b->bb[KING] & b->occ[them];
    int king_sq_us = us_king ? lsb64(us_king) : -1;
    int king_sq_them = them_king ? lsb64(them_king) : -1;
    uint64_t occupied = b->occ[0] | b->occ[1];
    int turn_white = (b->turn == WHITE_C) ? 1 : 0;
    compute_features_34(us_pieces, them_pieces, occupied,
                        king_sq_us, king_sq_them, turn_white,
                        (int)b->ep_square, out + 112 * 64);
}


/*
 * prepare_gumbel_leaves(root_cboards, board_indices, root_ids, forced_actions,
 *                       c_scale, c_visit, c_puct, fpu_reduction, full_tree,
 *                       enc_buf, root_qs)
 *
 * Fused C implementation of the _prepare_rep hot loop:
 * 1. gumbel_collect_leaves (tree traversal)
 * 2. CBoard replay from root via action paths
 * 3. Terminal detection + immediate backprop
 * 4. Encode non-terminal positions into enc_buf
 * 5. Generate legal moves for non-terminals
 *
 * Returns: (n_leaves, need_eval_int32, legal_list, leaf_ids_int32,
 *           node_paths_list, leaf_cboard_list)
 *   or None if no queries need GPU eval.
 */
static PyObject *MCTSTree_prepare_gumbel_leaves(MCTSTreeObject *self, PyObject *args) {
    PyObject *root_cbs_list, *enc_buf_obj, *root_qs_obj;
    PyObject *board_idx_obj, *root_ids_obj, *forced_obj;
    double c_scale, c_visit, c_puct, fpu_reduction;
    int full_tree;

    if (!PyArg_ParseTuple(args, "OOOOddddpOO",
                          &root_cbs_list, &board_idx_obj, &root_ids_obj, &forced_obj,
                          &c_scale, &c_visit, &c_puct, &fpu_reduction, &full_tree,
                          &enc_buf_obj, &root_qs_obj))
        return NULL;

    /* Parse numpy arrays */
    PyArrayObject *board_idx_arr = (PyArrayObject *)PyArray_FROMANY(
        board_idx_obj, NPY_INT32, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *root_ids_arr = (PyArrayObject *)PyArray_FROMANY(
        root_ids_obj, NPY_INT32, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *forced_arr = (PyArrayObject *)PyArray_FROMANY(
        forced_obj, NPY_INT32, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *enc_arr = (PyArrayObject *)PyArray_FROMANY(
        enc_buf_obj, NPY_FLOAT32, 4, 4, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE);
    PyArrayObject *root_qs_arr = (PyArrayObject *)PyArray_FROMANY(
        root_qs_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_C_CONTIGUOUS);

    if (!board_idx_arr || !root_ids_arr || !forced_arr || !enc_arr || !root_qs_arr) {
        Py_XDECREF(board_idx_arr); Py_XDECREF(root_ids_arr);
        Py_XDECREF(forced_arr); Py_XDECREF(enc_arr); Py_XDECREF(root_qs_arr);
        return NULL;
    }

    int32_t n_queries = (int32_t)PyArray_SIZE(root_ids_arr);
    const int32_t *board_indices = (const int32_t *)PyArray_DATA(board_idx_arr);
    const int32_t *root_ids = (const int32_t *)PyArray_DATA(root_ids_arr);
    const int32_t *forced_actions = (const int32_t *)PyArray_DATA(forced_arr);
    float *enc_data = (float *)PyArray_DATA(enc_arr);
    const double *root_qs = (const double *)PyArray_DATA(root_qs_arr);

    /* Step 1: tree traversal (reuse existing C function) */
    int32_t path_buf[512];
    int32_t action_buf[512];

    /* Collect all leaves, build CBoards, encode */
    int32_t n_leaves = 0;
    int32_t *need_eval = (int32_t *)malloc(n_queries * sizeof(int32_t));
    CBoard *leaf_boards = (CBoard *)malloc(n_queries * sizeof(CBoard));
    int32_t *leaf_ids_data = (int32_t *)malloc(n_queries * sizeof(int32_t));

    /* Per-query storage for node paths (for backprop) */
    int32_t **all_node_paths = (int32_t **)malloc(n_queries * sizeof(int32_t *));
    int32_t *all_path_lens = (int32_t *)malloc(n_queries * sizeof(int32_t));

    if (!need_eval || !leaf_boards || !leaf_ids_data || !all_node_paths || !all_path_lens) {
        free(need_eval); free(leaf_boards); free(leaf_ids_data);
        free(all_node_paths); free(all_path_lens);
        Py_DECREF(board_idx_arr); Py_DECREF(root_ids_arr);
        Py_DECREF(forced_arr); Py_DECREF(enc_arr); Py_DECREF(root_qs_arr);
        return PyErr_NoMemory();
    }

    /* Terminal backprop lists */
    int32_t n_terminals = 0;
    int32_t **term_paths = (int32_t **)malloc(n_queries * sizeof(int32_t *));
    int32_t *term_path_lens = (int32_t *)malloc(n_queries * sizeof(int32_t));
    double *term_values = (double *)malloc(n_queries * sizeof(double));

    for (int32_t qi = 0; qi < n_queries; qi++) {
        /* Tree traversal for this query */
        int32_t path_len = tree_gumbel_collect_leaf(
            &self->tree, root_ids[qi], forced_actions[qi],
            c_scale, c_visit, c_puct, fpu_reduction, full_tree,
            path_buf, 512);

        /* Save node path */
        all_path_lens[qi] = path_len;
        all_node_paths[qi] = (int32_t *)malloc(path_len * sizeof(int32_t));
        memcpy(all_node_paths[qi], path_buf, path_len * sizeof(int32_t));

        int32_t leaf_id = path_buf[path_len - 1];
        leaf_ids_data[qi] = leaf_id;

        /* Single-node path = root itself, use root Q */
        if (path_len <= 1) {
            term_paths[n_terminals] = all_node_paths[qi];
            term_path_lens[n_terminals] = path_len;
            term_values[n_terminals] = root_qs[board_indices[qi]];
            n_terminals++;
            continue;
        }

        /* Get action path for board replay */
        int32_t action_len = tree_action_path(&self->tree, leaf_id, action_buf, 512);
        if (action_len < 0) action_len = 0;

        /* Replay CBoard from root */
        int32_t bi = board_indices[qi];
        PyCBoard *root_pycb = (PyCBoard *)PyList_GET_ITEM(root_cbs_list, bi);
        CBoard cb = root_pycb->board;  /* struct copy */
        for (int32_t ai = 0; ai < action_len; ai++)
            cboard_push_index(&cb, action_buf[ai]);

        if (cboard_is_game_over(&cb)) {
            term_paths[n_terminals] = all_node_paths[qi];
            term_path_lens[n_terminals] = path_len;
            term_values[n_terminals] = (double)cboard_terminal_value(&cb);
            n_terminals++;
        } else {
            /* Encode into pre-allocated buffer */
            cboard_encode_146_into(&cb, enc_data + n_leaves * 146 * 64);
            leaf_boards[n_leaves] = cb;
            need_eval[n_leaves] = qi;
            n_leaves++;
        }
    }

    /* Backprop terminals */
    for (int32_t ti = 0; ti < n_terminals; ti++) {
        tree_backprop(&self->tree, term_paths[ti], term_path_lens[ti], term_values[ti]);
    }

    /* Clean up terminal-only storage */
    free(term_paths);
    free(term_path_lens);
    free(term_values);

    if (n_leaves == 0) {
        /* Free everything, return None */
        for (int32_t qi = 0; qi < n_queries; qi++) free(all_node_paths[qi]);
        free(all_node_paths); free(all_path_lens);
        free(need_eval); free(leaf_boards); free(leaf_ids_data);
        Py_DECREF(board_idx_arr); Py_DECREF(root_ids_arr);
        Py_DECREF(forced_arr); Py_DECREF(enc_arr); Py_DECREF(root_qs_arr);
        Py_RETURN_NONE;
    }

    /* Build Python return values */

    /* need_eval array */
    npy_intp ne_dims[1] = {n_leaves};
    PyObject *ne_arr = PyArray_SimpleNew(1, ne_dims, NPY_INT32);
    memcpy(PyArray_DATA((PyArrayObject *)ne_arr), need_eval, n_leaves * sizeof(int32_t));

    /* leaf_ids array */
    PyObject *lid_arr = PyArray_SimpleNew(1, ne_dims, NPY_INT32);
    {
        int32_t *lid_data = (int32_t *)PyArray_DATA((PyArrayObject *)lid_arr);
        for (int32_t li = 0; li < n_leaves; li++)
            lid_data[li] = leaf_ids_data[need_eval[li]];
    }

    /* legal moves list */
    PyObject *legal_list = PyList_New(n_leaves);
    for (int32_t li = 0; li < n_leaves; li++) {
        BoardState bs;
        cboard_to_boardstate(&leaf_boards[li], &bs);
        int indices[256];
        int count = 0;
        if (bs.king_sq >= 0) {
            count = generate_legal_move_indices(&bs, indices);
            sort_int(indices, count);
        }
        npy_intp dims[1] = {count};
        PyObject *larr = PyArray_SimpleNew(1, dims, NPY_INT32);
        if (count > 0)
            memcpy(PyArray_DATA((PyArrayObject *)larr), indices, count * sizeof(int));
        PyList_SET_ITEM(legal_list, li, larr);
    }

    /* node_paths list (only for need_eval queries) */
    PyObject *paths_list = PyList_New(n_leaves);
    for (int32_t li = 0; li < n_leaves; li++) {
        int32_t qi = need_eval[li];
        npy_intp pdims[1] = {all_path_lens[qi]};
        PyObject *parr = PyArray_SimpleNew(1, pdims, NPY_INT32);
        memcpy(PyArray_DATA((PyArrayObject *)parr), all_node_paths[qi],
               all_path_lens[qi] * sizeof(int32_t));
        PyList_SET_ITEM(paths_list, li, parr);
    }

    /* leaf CBoard list (as PyCBoard objects) */
    PyObject *cboard_type = (PyObject *)Py_TYPE(PyList_GET_ITEM(root_cbs_list, 0));
    PyObject *cb_list = PyList_New(n_leaves);
    for (int32_t li = 0; li < n_leaves; li++) {
        PyCBoard *pycb = PyObject_New(PyCBoard, (PyTypeObject *)cboard_type);
        if (pycb) pycb->board = leaf_boards[li];
        PyList_SET_ITEM(cb_list, li, (PyObject *)pycb);
    }

    /* Cleanup */
    for (int32_t qi = 0; qi < n_queries; qi++) free(all_node_paths[qi]);
    free(all_node_paths); free(all_path_lens);
    free(need_eval); free(leaf_boards); free(leaf_ids_data);
    Py_DECREF(board_idx_arr); Py_DECREF(root_ids_arr);
    Py_DECREF(forced_arr); Py_DECREF(enc_arr); Py_DECREF(root_qs_arr);

    /* Return (n_leaves, need_eval, legal_list, leaf_ids, node_paths, leaf_cboards) */
    PyObject *result = Py_BuildValue("(iNNNNN)",
        (int)n_leaves, ne_arr, legal_list, lid_arr, paths_list, cb_list);
    return result;
}


/*
 * finish_gumbel_rep(leaf_ids_int32, legal_list, pol_logits_float32,
 *                   wdl_logits_float32, node_paths_list) -> None
 *
 * Fused expand + backprop after GPU eval:
 * 1. batch_wdl_to_q (WDL logits → Q values)
 * 2. expand_from_logits for each non-expanded leaf
 * 3. backprop_many with Q values along node paths
 */
static PyObject *MCTSTree_finish_gumbel_rep(MCTSTreeObject *self, PyObject *args) {
    PyObject *leaf_ids_obj, *legal_list, *pol_obj, *wdl_obj, *paths_list;
    if (!PyArg_ParseTuple(args, "OOOOO",
                          &leaf_ids_obj, &legal_list, &pol_obj, &wdl_obj, &paths_list))
        return NULL;

    PyArrayObject *leaf_ids_arr = (PyArrayObject *)PyArray_FROMANY(
        leaf_ids_obj, NPY_INT32, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *pol_arr = (PyArrayObject *)PyArray_FROMANY(
        pol_obj, NPY_FLOAT32, 2, 2, NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *wdl_arr = (PyArrayObject *)PyArray_FROMANY(
        wdl_obj, NPY_FLOAT32, 2, 2, NPY_ARRAY_C_CONTIGUOUS);

    if (!leaf_ids_arr || !pol_arr || !wdl_arr) {
        Py_XDECREF(leaf_ids_arr); Py_XDECREF(pol_arr); Py_XDECREF(wdl_arr);
        return NULL;
    }

    int32_t n_leaves = (int32_t)PyArray_SIZE(leaf_ids_arr);
    const int32_t *leaf_ids = (const int32_t *)PyArray_DATA(leaf_ids_arr);
    const float *pol_data = (const float *)PyArray_DATA(pol_arr);
    const float *wdl_data = (const float *)PyArray_DATA(wdl_arr);

    TreeData *t = &self->tree;

    for (int32_t li = 0; li < n_leaves; li++) {
        int32_t nid = leaf_ids[li];

        /* Expand if not already expanded */
        if (!t->expanded[nid]) {
            PyObject *legal_arr_obj = PyList_GET_ITEM(legal_list, li);
            PyArrayObject *legal_arr = (PyArrayObject *)PyArray_FROMANY(
                legal_arr_obj, NPY_INT32, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
            if (legal_arr) {
                int32_t n_legal = (int32_t)PyArray_SIZE(legal_arr);
                if (n_legal > 0) {
                    const int32_t *legal = (const int32_t *)PyArray_DATA(legal_arr);
                    const float *logits = pol_data + li * 4672;

                    /* Softmax over legal moves */
                    double priors_stack[256];
                    double *priors = (n_legal <= 256) ? priors_stack
                                                      : (double *)malloc(n_legal * sizeof(double));
                    if (priors) {
                        for (int32_t j = 0; j < n_legal; j++)
                            priors[j] = (double)logits[legal[j]];
                        softmax_inplace(priors, n_legal);
                        tree_expand(t, nid, legal, priors, n_legal);
                        if (priors != priors_stack) free(priors);
                    }
                }
                Py_DECREF(legal_arr);
            }
        }

        /* Compute Q from WDL logits */
        double w = (double)wdl_data[li * 3 + 0];
        double d = (double)wdl_data[li * 3 + 1];
        double l = (double)wdl_data[li * 3 + 2];
        double mx = w; if (d > mx) mx = d; if (l > mx) mx = l;
        double ew = exp(w - mx), ed = exp(d - mx), el = exp(l - mx);
        double s = ew + ed + el;
        double q_val = (s > 0.0) ? ((ew - el) / s) : 0.0;

        /* Backprop along node path */
        PyObject *path_obj = PyList_GET_ITEM(paths_list, li);
        PyArrayObject *path_arr = (PyArrayObject *)PyArray_FROMANY(
            path_obj, NPY_INT32, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
        if (path_arr) {
            int32_t path_len = (int32_t)PyArray_SIZE(path_arr);
            const int32_t *path = (const int32_t *)PyArray_DATA(path_arr);
            tree_backprop(t, path, path_len, q_val);
            Py_DECREF(path_arr);
        }
    }

    Py_DECREF(leaf_ids_arr); Py_DECREF(pol_arr); Py_DECREF(wdl_arr);
    Py_RETURN_NONE;
}


static PyMethodDef MCTSTree_methods[] = {
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
    {"gumbel_collect_leaves", (PyCFunction)MCTSTree_gumbel_collect_leaves, METH_VARARGS,
     "gumbel_collect_leaves(root_ids, forced_actions, c_scale, c_visit, c_puct, fpu_reduction, full_tree)"},
    {"prepare_gumbel_leaves", (PyCFunction)MCTSTree_prepare_gumbel_leaves, METH_VARARGS,
     "prepare_gumbel_leaves(root_cbs, board_idx, root_ids, forced, c_scale, c_visit, c_puct, fpu, full_tree, enc_buf, root_qs)"},
    {"finish_gumbel_rep", (PyCFunction)MCTSTree_finish_gumbel_rep, METH_VARARGS,
     "finish_gumbel_rep(leaf_ids, legal_list, pol_logits, wdl_logits, node_paths)"},
    {"gumbel_score_candidates", (PyCFunction)MCTSTree_gumbel_score_candidates, METH_VARARGS,
     "gumbel_score_candidates(root_id, cands, gumbels, priors, c_scale, c_visit)"},
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

static PyMethodDef module_methods[] = {
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

    PyObject *m = PyModule_Create(&mcts_tree_module);
    if (!m) return NULL;

    Py_INCREF(&MCTSTreeType);
    if (PyModule_AddObject(m, "MCTSTree", (PyObject *)&MCTSTreeType) < 0) {
        Py_DECREF(&MCTSTreeType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
