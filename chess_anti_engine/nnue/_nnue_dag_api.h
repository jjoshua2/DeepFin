/*
 * _nnue_dag_api.h — Python probe surface for the NNUE-payload position DAG.
 *
 * Included by _nnue_ext.c AFTER its evaluator and Python helpers are defined.
 * The graph storage lives in ../mcts/_position_dag.h and has no NNUE or MCTS
 * semantics; the payload store and every publication path live in
 * _nnue_dag_store.h, which _arm_providers.h also uses. This file adds nothing
 * to that store but argument parsing, validation of an UNTRUSTED caller's
 * (parent, action, child) triple, and the dict/tuple shapes. It proves the
 * intended first consumer:
 *
 *   structural position discovered once
 *       -> one CaeNnueState
 *       -> at most one static NNUE evaluation
 *       -> any number of incoming parent edges reuse both
 *
 * An in-check structural position is still a valid DAG node and still owns an
 * accumulator state, but has no static value: Stockfish NNUE is undefined there.
 * The tactical solver can resolve its evasions without inventing a sentinel
 * evaluation.
 *
 * ⚑ THE SPLIT EXISTS SO A C CONSUMER CAN SHARE THE ACCOUNTING, not just the
 * arrays. A search driving the same store from C must increment state_inits,
 * state_makes and nnue_evals the way this file does, or the store's headline
 * identity quietly becomes a property of one caller. So publication happens in
 * exactly two places — cae_nnue_dag_intern_position and
 * cae_nnue_dag_intern_child — and both callers go through them.
 */

#ifndef CAE_NNUE_DAG_API_H
#define CAE_NNUE_DAG_API_H

#include "_nnue_dag_store.h"

#define CAE_NNUE_DAG_CAPSULE "cae.nnue.position_dag"

static void cae_nnue_dag_free(CaeNnueDagHandle *h) {
    if (!h) return;
    cae_nnue_dag_store_release(h);
    PyMem_Free(h);
}

static void cae_nnue_dag_capsule_destructor(PyObject *capsule) {
    CaeNnueDagHandle *h = (CaeNnueDagHandle *)PyCapsule_GetPointer(
        capsule, CAE_NNUE_DAG_CAPSULE);
    if (!h) {
        /* Destructors must not leak a stale capsule error into interpreter
         * shutdown; a bad public call is handled by the accessor below. */
        PyErr_Clear();
        return;
    }
    cae_nnue_dag_free(h);
}

static CaeNnueDagHandle *cae_nnue_dag_from_capsule(PyObject *capsule) {
    CaeNnueDagHandle *h = (CaeNnueDagHandle *)PyCapsule_GetPointer(
        capsule, CAE_NNUE_DAG_CAPSULE);
    if (!h)
        PyErr_SetString(PyExc_TypeError, "expected a DAG handle from dag_open()");
    return h;
}

/* Turn a store status into the exception it names. Never called with
 * CAE_VALUE_OK; the store's own two codes are not seam statuses, so
 * raise_status() would mislabel them. */
static void cae_nnue_dag_raise(int status) {
    if (status == CAE_NNUE_DAG_ERR_NO_MEMORY) {
        PyErr_NoMemory();
        return;
    }
    if (status == CAE_NNUE_DAG_ERR_LINK) {
        PyErr_SetString(PyExc_RuntimeError,
                        "new DAG node could not be linked to its parent");
        return;
    }
    raise_status(status);
}

static PyObject *cae_nnue_dag_value_object(const CaeNnueDagHandle *h, int32_t node_id) {
    /* Bound against the PAYLOAD array, not against dag.node_count: the two are
     * kept in step only by cae_nnue_dag_publish_new(), and the graph layer can
     * legitimately create a node without a payload obligation
     * (cae_position_dag_intern_board() does exactly that). A caller-supplied id
     * must not be able to read past the payload arrays because a different
     * writer grew the graph. */
    if (node_id < 0 || node_id >= h->payload_cap) {
        PyErr_Format(PyExc_IndexError,
                     "DAG node %d has no NNUE payload slot (payload_capacity %d)",
                     node_id, h->payload_cap);
        return NULL;
    }
    if (!h->value_valid[node_id]) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    return PyLong_FromLong((long)h->values[node_id]);
}

static PyObject *cae_nnue_dag_result(
    const CaeNnueDagHandle *h, int32_t node_id, int created)
{
    PyObject *value = cae_nnue_dag_value_object(h, node_id);
    if (!value) return NULL;
    PyObject *out = Py_BuildValue("(iNO)", node_id, value, created ? Py_True : Py_False);
    return out;
}

PyDoc_STRVAR(dag_open_doc,
"dag_open(weights_handle[, initial_nodes]) -> handle\n\n"
"Create a reusable canonical structural-position DAG over the SAME mapped NNUE\n"
"weights returned by load(). The DAG retains that mapping; it does not load or\n"
"compile a second evaluator. Nodes carry incremental CaeNnueState and a static\n"
"value when the position is not in check.");

static PyObject *py_dag_open(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *weights_capsule;
    int initial_nodes = 256;
    if (!PyArg_ParseTuple(args, "O|i", &weights_capsule, &initial_nodes)) return NULL;
    if (initial_nodes < 1 || initial_nodes > INT32_MAX / 4) {
        PyErr_SetString(PyExc_ValueError, "initial_nodes must be 1..INT32_MAX/4");
        return NULL;
    }
    CaeNnueWeights *w = weights_from_capsule(weights_capsule);
    if (!w) return NULL;

    CaeNnueDagHandle *h = (CaeNnueDagHandle *)PyMem_Calloc(1, sizeof(*h));
    if (!h) return PyErr_NoMemory();
    if (cae_nnue_dag_store_init(h, w, initial_nodes) != 0) {
        cae_nnue_dag_free(h);
        return PyErr_NoMemory();
    }

    PyObject *capsule = PyCapsule_New(h, CAE_NNUE_DAG_CAPSULE,
                                      cae_nnue_dag_capsule_destructor);
    if (!capsule) {
        cae_nnue_dag_free(h);
        return NULL;
    }
    return capsule;
}

/* ⚑⚑ THE CONSTRUCTION CALLS BELOW DELIBERATELY KEEP THE GIL.
 *
 * cae_nnue_state_init/state_make/state_evaluate are µs-scale, so there is
 * nothing to win by releasing it — and releasing it is what makes the
 * documented single-threaded constraint unenforceable rather than merely
 * documented. With a release window between "probe found nothing" and
 * "publish", two Python threads interning the same position both miss, both
 * publish, and the second link fails: MEASURED at 6 threads on the pre-fix
 * build, node_count 87 for 21 distinct structural positions. Worse, the make
 * reads &h->states[parent_id] inside that window while another thread's
 * publish can run cae_nnue_dag_grow_payload(), which free()s that very array —
 * a use-after-free reading an accumulator that has been handed back to malloc.
 *
 * Holding the GIL makes each of these functions atomic against other Python
 * threads, so probe -> evaluate -> publish -> link cannot interleave.
 * A future concurrent consumer must NOT simply reinstate Py_BEGIN_ALLOW_THREADS
 * here: it has to add real synchronization first — a single-owner check or a
 * lock spanning probe/publish/link, plus payload storage that a concurrent grow
 * cannot free under a reader (stable chunks or RCU-style retirement). The rule
 * binds every consumer of the store, not just this file. */

PyDoc_STRVAR(dag_intern_root_doc,
"dag_intern_root(handle, cboard) -> (node_id, value_or_none, created)\n\n"
"Intern a structural root and make it the DAG's current root. Halfmove clock and\n"
"history are deliberately not node identity; path-sensitive terminal/history\n"
"semantics belong to the search overlay. A new node refreshes NNUE once; an\n"
"existing node performs no NNUE work.");

static PyObject *py_dag_intern_root(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *capsule, *board_obj;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &board_obj)) return NULL;
    CaeNnueDagHandle *h = cae_nnue_dag_from_capsule(capsule);
    if (!h) return NULL;
    const CBoard *board = unwrap_cboard(board_obj);
    if (!board) return NULL;

    /* GIL held on purpose — see the block comment above. */
    int32_t node_id = CAE_DAG_NO_NODE;
    int created = 0;
    int status = cae_nnue_dag_intern_position(h, board, &node_id, &created);
    if (status != CAE_VALUE_OK) { cae_nnue_dag_raise(status); return NULL; }

    h->dag.root_id = node_id;
    return cae_nnue_dag_result(h, node_id, created);
}

PyDoc_STRVAR(dag_intern_child_doc,
"dag_intern_child(handle, parent_id, action, child_cboard)\n"
"    -> (node_id, value_or_none, created)\n\n"
"Validate the action->child structural transition, then add a true DAG edge. A\n"
"new child derives incremental NNUE state from the parent and evaluates at most\n"
"once. If another parent already reached that structural position, the existing\n"
"node/state/value is reused without make() or evaluate().");

static PyObject *py_dag_intern_child(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *capsule, *board_obj;
    int parent_id, action;
    if (!PyArg_ParseTuple(args, "OiiO", &capsule, &parent_id, &action, &board_obj))
        return NULL;
    CaeNnueDagHandle *h = cae_nnue_dag_from_capsule(capsule);
    if (!h) return NULL;
    if (parent_id < 0 || parent_id >= h->dag.node_count) {
        PyErr_SetString(PyExc_ValueError, "parent_id out of range");
        return NULL;
    }
    const CBoard *child_board = unwrap_cboard(board_obj);
    if (!child_board) return NULL;

    /* ⚑ VALIDATION LIVES HERE, NOT IN THE STORE. This entry point takes a
     * (parent, action, child) triple straight from Python, so "is that action
     * legal at the parent and does it really produce that child" has to be
     * asked before anything is published. The C search path skips this because
     * it generated and pushed the move itself. */
    if (!cae_position_dag_edge_matches_board(&h->dag, parent_id, action, child_board)) {
        PyErr_Format(PyExc_ValueError,
                     "action %d does not produce the supplied child from DAG node %d",
                     action, parent_id);
        return NULL;
    }

    CaeDagPosition child_pos;
    cae_dag_position_from_cboard(child_board, &child_pos);

    /* An exact repeat of a (parent, action) request never probes the canonical
     * table at all, which is why `node_reuses` is not the transposition signal
     * and `hits` is. */
    int32_t already_for_action = cae_position_dag_child_for_action(
        &h->dag, parent_id, action);
    if (already_for_action != CAE_DAG_NO_NODE) {
        if (!cae_dag_position_equal(&h->dag.positions[already_for_action], &child_pos)) {
            PyErr_Format(PyExc_RuntimeError,
                         "DAG action %d at node %d already points at a different position",
                         action, parent_id);
            return NULL;
        }
        h->node_reuses++;
        h->dag.edge_reuses++;
        return cae_nnue_dag_result(h, already_for_action, 0);
    }

    /* GIL held on purpose — see the block comment above. */
    int32_t node_id = CAE_DAG_NO_NODE;
    int created = 0;
    int status = cae_nnue_dag_intern_child(
        h, parent_id, action, child_board, &node_id, &created);
    if (status != CAE_VALUE_OK) { cae_nnue_dag_raise(status); return NULL; }
    return cae_nnue_dag_result(h, node_id, created);
}

PyDoc_STRVAR(dag_lookup_doc,
"dag_lookup(handle, cboard) -> int | None\n\n"
"Return the canonical structural node id, if already present. This is a graph\n"
"lookup only and performs no NNUE work.");

static PyObject *py_dag_lookup(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *capsule, *board_obj;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &board_obj)) return NULL;
    CaeNnueDagHandle *h = cae_nnue_dag_from_capsule(capsule);
    if (!h) return NULL;
    const CBoard *board = unwrap_cboard(board_obj);
    if (!board) return NULL;
    int32_t node_id = cae_position_dag_find_board(&h->dag, board);
    if (node_id == CAE_DAG_NO_NODE) Py_RETURN_NONE;
    return PyLong_FromLong((long)node_id);
}

static PyObject *py_dag_value(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *capsule;
    int node_id;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &node_id)) return NULL;
    CaeNnueDagHandle *h = cae_nnue_dag_from_capsule(capsule);
    if (!h) return NULL;
    if (node_id < 0 || node_id >= h->dag.node_count) {
        PyErr_SetString(PyExc_ValueError, "node_id out of range");
        return NULL;
    }
    return cae_nnue_dag_value_object(h, node_id);
}

static PyObject *py_dag_children(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *capsule;
    int node_id;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &node_id)) return NULL;
    CaeNnueDagHandle *h = cae_nnue_dag_from_capsule(capsule);
    if (!h) return NULL;
    if (node_id < 0 || node_id >= h->dag.node_count) {
        PyErr_SetString(PyExc_ValueError, "node_id out of range");
        return NULL;
    }
    PyObject *out = PyList_New(h->dag.out_degree[node_id]);
    if (!out) return NULL;
    Py_ssize_t i = 0;
    for (int32_t e = h->dag.first_edge[node_id]; e != CAE_DAG_NO_NODE; e = h->dag.edge_next[e]) {
        PyObject *pair = Py_BuildValue("(ii)", h->dag.edge_action[e], h->dag.edge_child[e]);
        if (!pair) { Py_DECREF(out); return NULL; }
        PyList_SET_ITEM(out, i++, pair);
    }
    return out;
}

static PyObject *py_dag_set_root(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *capsule;
    int node_id;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &node_id)) return NULL;
    CaeNnueDagHandle *h = cae_nnue_dag_from_capsule(capsule);
    if (!h) return NULL;
    if (cae_position_dag_set_root(&h->dag, node_id) != 0) {
        PyErr_SetString(PyExc_ValueError, "node_id out of range");
        return NULL;
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(dag_stats_doc,
"dag_stats(handle) -> dict\n\n"
"Structural graph and NNUE-work counters.\n\n"
"⚑ The headline invariant is the exact identity\n\n"
"    state_inits + state_makes == node_count\n\n"
"every canonical node was published by exactly one accounted NNUE state\n"
"construction, and no node exists without one. It is FALSIFIABLE, and it has\n"
"fired: at 6 threads on a build that released the GIL around state_make,\n"
"state_inits + state_makes read 21 against a node_count of 87, because\n"
"duplicate publications produced nodes whose work was never accounted (their\n"
"link failed).\n\n"
"⚑ Do NOT read nnue_evals <= node_count as the invariant. It holds by\n"
"construction on every path — a node is published at most once per evaluation —\n"
"and duplicating nodes makes it MORE satisfied, so it is exactly blind to the\n"
"failure it looks like it is watching.\n\n"
"Counters that do not mean what their names suggest:\n"
"  hits            canonical-table probe hits: THE transposition signal. A new\n"
"                  parent reaching an already-interned structural position\n"
"                  increments it (it also counts a re-request of a position\n"
"                  already interned, so read it against inserts/probes).\n"
"  node_reuses     NOT the transposition signal: it additionally counts a\n"
"                  repeated identical (parent, action) request, which never\n"
"                  probes the table at all.\n"
"  edge_reuses     ONLY that caller redundancy — an exact duplicate\n"
"                  (parent, action, child) edge request. Never a transposition.\n"
"  collision_steps linear-probe DISPLACEMENT: occupied slots stepped over,\n"
"                  whatever their key. It is not a count of 64-bit key\n"
"                  collisions, which open addressing makes far rarer than this\n"
"                  number.\n"
"  probes          find_position() calls, including pure dag_lookup() reads.\n\n"
"A true transposition raises hits and node_reuses and leaves state_makes and\n"
"nnue_evals unchanged for that request.");

/* Shared with arm_dag_stats(): one store, one stats shape, so a DAG driven from
 * C reports through the same builder as one driven from Python. */
static PyObject *cae_nnue_dag_stats_dict(const CaeNnueDagHandle *h) {
    int64_t dag_bytes = cae_position_dag_memory_bytes(&h->dag);
    int64_t payload_bytes = cae_nnue_dag_payload_bytes(h);
    return Py_BuildValue(
        "{s:i,s:i,s:i,s:i,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:L,s:L,s:L}",
        "root_id", h->dag.root_id,
        "node_count", h->dag.node_count,
        "edge_count", h->dag.edge_count,
        "payload_capacity", h->payload_cap,
        "probes", (unsigned long long)h->dag.probes,
        "hits", (unsigned long long)h->dag.hits,
        "inserts", (unsigned long long)h->dag.inserts,
        "collision_steps", (unsigned long long)h->dag.collision_steps,
        "edge_reuses", (unsigned long long)h->dag.edge_reuses,
        "state_inits", (unsigned long long)h->state_inits,
        "state_makes", (unsigned long long)h->state_makes,
        "nnue_evals", (unsigned long long)h->nnue_evals,
        "node_reuses", (unsigned long long)h->node_reuses,
        "dag_memory_bytes", (long long)dag_bytes,
        "nnue_payload_bytes", (long long)payload_bytes,
        "memory_bytes", (long long)(dag_bytes + payload_bytes));
}

static PyObject *py_dag_stats(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    CaeNnueDagHandle *h = cae_nnue_dag_from_capsule(capsule);
    if (!h) return NULL;
    return cae_nnue_dag_stats_dict(h);
}

static PyObject *py_dag_reset(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    CaeNnueDagHandle *h = cae_nnue_dag_from_capsule(capsule);
    if (!h) return NULL;
    cae_nnue_dag_store_reset(h);
    Py_RETURN_NONE;
}

#define CAE_NNUE_DAG_METHODS \
    {"dag_open", py_dag_open, METH_VARARGS, dag_open_doc}, \
    {"dag_intern_root", py_dag_intern_root, METH_VARARGS, dag_intern_root_doc}, \
    {"dag_intern_child", py_dag_intern_child, METH_VARARGS, dag_intern_child_doc}, \
    {"dag_lookup", py_dag_lookup, METH_VARARGS, dag_lookup_doc}, \
    {"dag_value", py_dag_value, METH_VARARGS, NULL}, \
    {"dag_children", py_dag_children, METH_VARARGS, NULL}, \
    {"dag_set_root", py_dag_set_root, METH_VARARGS, NULL}, \
    {"dag_stats", py_dag_stats, METH_VARARGS, dag_stats_doc}, \
    {"dag_reset", py_dag_reset, METH_VARARGS, NULL},

#endif /* CAE_NNUE_DAG_API_H */
