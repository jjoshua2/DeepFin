/*
 * _nnue_ext.c — Python surface for the native NNUE evaluator.
 *
 * This module exists so the parity gate, the unit tests and the throughput
 * benchmark can drive the evaluator directly, without going through the MCTS
 * tree — and it is also the module that OWNS the evaluator: it is the only
 * translation unit that includes _nnue_impl.h, so there is exactly one copy of
 * the kernel-selection flag and one weight cache in the process.
 *
 * The tree gets the SAME code by importing the value-provider capsule this
 * module publishes (see ../mcts/_value_provider.h), not by including the
 * evaluator's header. That is what makes set_simd() below actually govern what
 * the tree runs, and what keeps one weight file to one mapping.
 *
 * Positions arrive as CBoard objects built by
 * chess_anti_engine.encoding._lc0_ext.CBoard.from_board(python_chess_board),
 * which is the production adapter — so the parity gate exercises the same
 * position path the tree will.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../encoding/_cboard_impl.h"
#include "_nnue_impl.h"
/* Pulls in _nnue_provider.h and ../mcts/_check_resolver.h, and owns the provider
 * registry — the raw evaluator plus the two resolver-backed race arms. */
#include "_arm_providers.h"

/* Mirrors the layout of _lc0_ext.c's PyCBoard, the same way _mcts_tree.c does. */
typedef struct {
    PyObject_HEAD
    CBoard board;
} PyCBoardMirror;

static PyObject *NnueInCheckError = NULL;

/* ⚑ THE TYPE IS RESOLVED, NOT GUESSED FROM ITS NAME. An earlier version of this
 * compared tp_name's trailing component to "CBoard", which is not a type check:
 * any class anywhere called CBoard passed it, and the cast below then read a
 * small object's storage as a much larger PyCBoardMirror — out of bounds, after
 * the GIL is released, instead of the TypeError that was promised. So we import
 * the real type once and compare identity, and check tp_basicsize as well
 * because the cast's correctness rests on the layout and not just the identity.
 * The tree does the same, through its own copy of this helper. */
static PyTypeObject *cached_cboard_type = NULL;

static int ensure_cboard_type(void) {
    if (cached_cboard_type) return 0;
    PyObject *mod = PyImport_ImportModule("chess_anti_engine.encoding._lc0_ext");
    if (!mod) return -1;
    PyObject *type_obj = PyObject_GetAttrString(mod, "CBoard");
    Py_DECREF(mod);
    if (!type_obj) return -1;
    if (!PyType_Check(type_obj)
        || ((PyTypeObject *)type_obj)->tp_basicsize != (Py_ssize_t)sizeof(PyCBoardMirror)) {
        Py_DECREF(type_obj);
        PyErr_SetString(PyExc_ImportError,
                        "CBoard extension ABI mismatch; rebuild C extensions");
        return -1;
    }
    cached_cboard_type = (PyTypeObject *)type_obj;   /* module-lifetime reference */
    return 0;
}

static const CBoard *unwrap_cboard(PyObject *obj) {
    if (ensure_cboard_type() != 0) return NULL;
    if (Py_TYPE(obj) != cached_cboard_type) {
        PyErr_Format(PyExc_TypeError,
                     "expected a CBoard from chess_anti_engine.encoding._lc0_ext, got %s",
                     Py_TYPE(obj)->tp_name ? Py_TYPE(obj)->tp_name : "?");
        return NULL;
    }
    return &((PyCBoardMirror *)obj)->board;
}

static void weights_capsule_destructor(PyObject *capsule) {
    CaeNnueWeights *w = (CaeNnueWeights *)PyCapsule_GetPointer(capsule, "cae.nnue.weights");
    if (w) cae_nnue_release(w);
}

static CaeNnueWeights *weights_from_capsule(PyObject *capsule) {
    CaeNnueWeights *w = (CaeNnueWeights *)PyCapsule_GetPointer(capsule, "cae.nnue.weights");
    if (!w) PyErr_SetString(PyExc_TypeError, "expected an NNUE weights handle from load()");
    return w;
}

PyDoc_STRVAR(load_doc,
"load(pack_path) -> handle\n\n"
"mmap a weight pack built by scripts/nnue_pack.py, read-only. Repeated loads of\n"
"the same path in one process share one mapping. FATAL on any .nnue version but\n"
"0x7AF32F20 and on anything but the big (threat) architecture.");

static PyObject *py_load(PyObject *Py_UNUSED(self), PyObject *args) {
    const char *path;
    if (!PyArg_ParseTuple(args, "s", &path)) return NULL;

    char err[512] = {0};
    CaeNnueWeights *w;
    Py_BEGIN_ALLOW_THREADS
    w = cae_nnue_load(path, err, sizeof(err));
    Py_END_ALLOW_THREADS
    if (!w) {
        PyErr_Format(PyExc_ValueError, "NNUE weight load failed: %s", err[0] ? err : "unknown");
        return NULL;
    }
    PyObject *capsule = PyCapsule_New(w, "cae.nnue.weights", weights_capsule_destructor);
    if (!capsule) { cae_nnue_release(w); return NULL; }
    return capsule;
}

PyDoc_STRVAR(info_doc,
"info(handle) -> dict\n\n"
"Architecture and provenance read off the LOADED weights themselves, not off the\n"
"path that was passed in.");

static PyObject *py_info(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    CaeNnueWeights *w = weights_from_capsule(capsule);
    if (!w) return NULL;
    return Py_BuildValue(
        "{s:I,s:I,s:I,s:I,s:I,s:I,s:I,s:I,s:I,s:s,s:s,s:i}",
        "l1", w->l1, "l2", w->l2, "l3", w->l3,
        "halfka_dims", w->halfka_dims, "threat_dims", w->threat_dims,
        "layer_stacks", w->layer_stacks, "psqt_buckets", w->psqt_buckets,
        "net_hash", w->net_hash, "ft_hash", w->ft_hash,
        "source_sha256", w->sha256_hex, "path", w->path,
        "avx2", CAE_NNUE_HAVE_AVX2);
}

PyDoc_STRVAR(source_sha256_doc,
"source_sha256(handle) -> str\n\n"
"SHA-256 of the .nnue the pack was built from. Stockfish names its nets\n"
"nn-<first 12 hex>.nnue, so this is what proves the gate is measuring the same\n"
"network the engine is running.");

static PyObject *py_source_sha256(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    CaeNnueWeights *w = weights_from_capsule(capsule);
    if (!w) return NULL;
    return PyUnicode_FromString(w->sha256_hex);
}

static int raise_status(int status) {
    if (status == CAE_VALUE_ERR_IN_CHECK) {
        PyErr_SetString(NnueInCheckError,
                        "NNUE evaluation is undefined in check; resolve check nodes "
                        "recursively (search the evasions) instead of evaluating");
    } else {
        PyErr_Format(PyExc_ValueError, "NNUE evaluation failed: %s",
                     cae_value_status_name(status));
    }
    return -1;
}

PyDoc_STRVAR(evaluate_doc,
"evaluate(handle, cboard) -> int\n\n"
"Internal units (psqt/16 + positional/16), side-to-move POV — the same number\n"
"Stockfish's `eval` prints as '(Big net) NNUE evaluation ... internal units'.\n"
"Raises InCheckError for a position in check: the network is undefined there and\n"
"no sentinel is returned that a caller could read as an evaluation.");

static PyObject *py_evaluate(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *capsule, *board_obj;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &board_obj)) return NULL;
    CaeNnueWeights *w = weights_from_capsule(capsule);
    if (!w) return NULL;
    const CBoard *board = unwrap_cboard(board_obj);
    if (!board) return NULL;

    int32_t value = 0;
    int status;
    Py_BEGIN_ALLOW_THREADS
    status = cae_nnue_evaluate_cboard(w, board, &value);
    Py_END_ALLOW_THREADS
    if (status != CAE_VALUE_OK) { raise_status(status); return NULL; }
    return PyLong_FromLong((long)value);
}

PyDoc_STRVAR(trace_doc,
"trace(handle, cboard) -> (bucket, psqt_tuple, positional_tuple)\n\n"
"Every layer stack's psqt and positional contribution, already divided by\n"
"OutputScale. This is the localisation instrument: Stockfish's exact line gives\n"
"only the total, so a per-bucket split is what says WHICH half diverged.");

static PyObject *py_trace(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *capsule, *board_obj;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &board_obj)) return NULL;
    CaeNnueWeights *w = weights_from_capsule(capsule);
    if (!w) return NULL;
    const CBoard *board = unwrap_cboard(board_obj);
    if (!board) return NULL;

    CaeNnuePos pos;
    int status = cae_nnue_pos_from_cboard(board, &pos);
    if (status != CAE_VALUE_OK) { raise_status(status); return NULL; }

    int32_t psqt[CAE_NNUE_PSQT_BUCKETS], positional[CAE_NNUE_PSQT_BUCKETS];
    Py_BEGIN_ALLOW_THREADS
    status = cae_nnue_trace(w, &pos, psqt, positional);
    Py_END_ALLOW_THREADS
    if (status != CAE_VALUE_OK) { raise_status(status); return NULL; }

    PyObject *tp = PyTuple_New(w->psqt_buckets), *tq = PyTuple_New(w->psqt_buckets);
    if (!tp || !tq) { Py_XDECREF(tp); Py_XDECREF(tq); return NULL; }
    for (uint32_t b = 0; b < w->psqt_buckets; b++) {
        PyTuple_SET_ITEM(tp, b, PyLong_FromLong((long)psqt[b]));
        PyTuple_SET_ITEM(tq, b, PyLong_FromLong((long)positional[b]));
    }
    return Py_BuildValue("(iNN)", cae_nnue_bucket(&pos), tp, tq);
}

PyDoc_STRVAR(active_features_doc,
"active_features(cboard, perspective) -> (halfka_indices, threat_indices)\n\n"
"The active feature indices for one perspective (0 = white, 1 = black), sorted.\n"
"Needs no weights: this is the index computation on its own, so a unit test can\n"
"assert hand-checkable features on a fixed position.");

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static PyObject *py_active_features(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *board_obj;
    int perspective;
    if (!PyArg_ParseTuple(args, "Oi", &board_obj, &perspective)) return NULL;
    if (perspective != 0 && perspective != 1) {
        PyErr_SetString(PyExc_ValueError, "perspective must be 0 (white) or 1 (black)");
        return NULL;
    }
    const CBoard *board = unwrap_cboard(board_obj);
    if (!board) return NULL;

    cae_nnue_init_tables();
    CaeNnuePos pos;
    int status = cae_nnue_pos_from_cboard(board, &pos);
    if (status != CAE_VALUE_OK) { raise_status(status); return NULL; }

    uint32_t halfka[64];
    int n_halfka = 0;
    int ksq = pos.king_sq[perspective];
    int sq;
    FOR_EACH_BIT(pos.occupied, sq)
        halfka[n_halfka++] = cae_nnue_halfka_index(perspective, sq, pos.piece_on[sq], ksq);

    CaeThreatRel rel[CAE_NNUE_MAX_RELATIONS];
    int n_rel = cae_nnue_threat_relations(&pos, rel);
    uint32_t threats[CAE_NNUE_MAX_RELATIONS];
    int n_threats = 0;
    for (int i = 0; i < n_rel; i++) {
        uint32_t idx = cae_nnue_threat_index(perspective, rel[i].attacker, rel[i].from,
                                             rel[i].to, pos.piece_on[rel[i].to], ksq);
        if (idx < CAE_NNUE_THREAT_DIMS) threats[n_threats++] = idx;
    }
    qsort(halfka, (size_t)n_halfka, sizeof(uint32_t), cmp_u32);
    qsort(threats, (size_t)n_threats, sizeof(uint32_t), cmp_u32);

    PyObject *ta = PyTuple_New(n_halfka), *tb = PyTuple_New(n_threats);
    if (!ta || !tb) { Py_XDECREF(ta); Py_XDECREF(tb); return NULL; }
    for (int i = 0; i < n_halfka; i++)
        PyTuple_SET_ITEM(ta, i, PyLong_FromUnsignedLong(halfka[i]));
    for (int i = 0; i < n_threats; i++)
        PyTuple_SET_ITEM(tb, i, PyLong_FromUnsignedLong(threats[i]));
    return Py_BuildValue("(NN)", ta, tb);
}

PyDoc_STRVAR(benchmark_doc,
"benchmark(handle, cboards, repeats, threads) -> (evals, seconds, checksum)\n\n"
"Time evals/s on REAL positions with feature-index computation included — the\n"
"scoping projection measured the accumulator gather alone, so this is the number\n"
"that closes that caveat. In-check boards in the list are skipped and do not\n"
"count. The checksum exists so an optimiser cannot delete the work.");

static PyObject *py_benchmark(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *capsule, *seq;
    int repeats = 1, threads = 1;
    if (!PyArg_ParseTuple(args, "OOii", &capsule, &seq, &repeats, &threads)) return NULL;
    CaeNnueWeights *w = weights_from_capsule(capsule);
    if (!w) return NULL;
    if (repeats < 1 || threads < 1) {
        PyErr_SetString(PyExc_ValueError, "repeats and threads must be >= 1");
        return NULL;
    }

    PyObject *fast = PySequence_Fast(seq, "cboards must be a sequence");
    if (!fast) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    CBoard *boards = (CBoard *)malloc((size_t)(n > 0 ? n : 1) * sizeof(CBoard));
    if (!boards) { Py_DECREF(fast); return PyErr_NoMemory(); }
    Py_ssize_t n_boards = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        const CBoard *b = unwrap_cboard(PySequence_Fast_GET_ITEM(fast, i));
        if (!b) { free(boards); Py_DECREF(fast); return NULL; }
        if (cboard_in_check(b)) continue;   /* the evaluator refuses these */
        boards[n_boards++] = *b;
    }
    Py_DECREF(fast);
    if (n_boards == 0) {
        free(boards);
        PyErr_SetString(PyExc_ValueError, "no evaluable (not-in-check) positions supplied");
        return NULL;
    }

    double t0 = 0.0, t1 = 0.0;
    long long done = 0;
    long long checksum = 0;

    Py_BEGIN_ALLOW_THREADS
#ifdef _OPENMP
    t0 = omp_get_wtime();
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(+ : done, checksum)
    for (long long it = 0; it < (long long)repeats * (long long)n_boards; it++) {
        int32_t v = 0;
        if (cae_nnue_evaluate_cboard(w, &boards[it % n_boards], &v) == CAE_VALUE_OK) {
            done++;
            checksum += v;
        }
    }
    t1 = omp_get_wtime();
#else
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    for (long long it = 0; it < (long long)repeats * (long long)n_boards; it++) {
        int32_t v = 0;
        if (cae_nnue_evaluate_cboard(w, &boards[it % n_boards], &v) == CAE_VALUE_OK) {
            done++;
            checksum += v;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    t0 = (double)ts0.tv_sec + 1e-9 * (double)ts0.tv_nsec;
    t1 = (double)ts1.tv_sec + 1e-9 * (double)ts1.tv_nsec;
#endif
    Py_END_ALLOW_THREADS

    free(boards);
    return Py_BuildValue("(LdL)", done, t1 - t0, checksum);
}

PyDoc_STRVAR(provider_eval_doc,
"provider_eval(name, weights_path, cboard) -> int\n\n"
"Evaluate THROUGH the value-provider vtable rather than by calling the evaluator\n"
"directly, so a test can prove the seam dispatches rather than that the maths\n"
"works. The tree uses the same registry.");

static PyObject *py_provider_eval(PyObject *Py_UNUSED(self), PyObject *args) {
    const char *name, *path;
    PyObject *board_obj;
    if (!PyArg_ParseTuple(args, "ssO", &name, &path, &board_obj)) return NULL;
    const CBoard *board = unwrap_cboard(board_obj);
    if (!board) return NULL;

    const CaeValueProvider *vp = cae_value_provider_by_name(name);
    if (!vp) {
        PyErr_Format(PyExc_ValueError, "no value provider named %s", name);
        return NULL;
    }
    char err[512] = {0};
    void *ctx = vp->init(path, err, sizeof(err));
    if (!ctx) {
        PyErr_Format(PyExc_ValueError, "provider %s init failed: %s", vp->name, err);
        return NULL;
    }
    int32_t value = 0;
    int status = cae_value_eval(vp, ctx, board, &value);
    vp->destroy(ctx);
    if (status != CAE_VALUE_OK) { raise_status(status); return NULL; }
    return PyLong_FromLong((long)value);
}

PyDoc_STRVAR(arm_eval_doc,
"arm_eval(name, weights_path, boards) -> (values, stats)\n\n"
"Evaluate a LIST of positions through ONE resolver-backed arm context, then\n"
"report that context's own accumulated counters.\n\n"
"⚑ The counters come off the ctx the provider's eval() wrote them into — the\n"
"consumer's own state — not off anything this function recomputed. A caller can\n"
"therefore ask 'how much work did check resolution actually do' and get an\n"
"answer produced by the code that did it. One ctx across the whole list is the\n"
"point: the accumulation path (the atomic merge) is exercised, and the in-check\n"
"leaf fraction is a fraction of something.\n\n"
"stats keys: calls, calls_in_check, nodes, resolved_leaves, terminal_mate,\n"
"terminal_draw, depth_cutoffs, max_depth_seen, qnodes, qterminal_draw,\n"
"qply_cutoffs, qmax_ply_seen, plus resolver_max_depth / qsearch_max_ply /\n"
"qsearch_check_plies.\n\n"
"⚑ Those last three are read off THE CONTEXT THAT RAN, not off the module\n"
"globals set_arm_config() writes. A context snapshots its configuration at\n"
"init(), so after a set_arm_config() the globals and a long-lived context can\n"
"legitimately disagree — and the number that governed this batch is the\n"
"context's. Reporting the global here would be a knob that looks applied.");

/* Resolve a provider name to one of the resolver-backed arms. */
static const CaeValueProvider *arm_provider_by_name(const char *name) {
    const CaeValueProvider *vp = cae_value_provider_by_name(name);
    if (!vp) {
        PyErr_Format(PyExc_ValueError, "no value provider named %s", name);
        return NULL;
    }
    if (!cae_provider_is_arm(vp)) {
        PyErr_Format(PyExc_ValueError,
                     "provider '%s' is not a resolver-backed arm and reports no "
                     "resolver statistics; use provider_eval for it", vp->name);
        return NULL;
    }
    return vp;
}

static PyObject *arm_stats_dict(const CaeArmCtx *ctx) {
    const CaeArmStats *s = &ctx->totals;
    return Py_BuildValue(
        "{s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:I,s:K,s:K,s:K,s:I,s:i,s:i,s:i}",
        "calls", (unsigned long long)s->resolver.calls,
        "calls_in_check", (unsigned long long)s->resolver.calls_in_check,
        "nodes", (unsigned long long)s->resolver.nodes,
        "resolved_leaves", (unsigned long long)s->resolver.resolved_leaves,
        "terminal_mate", (unsigned long long)s->resolver.terminal_mate,
        "terminal_draw", (unsigned long long)s->resolver.terminal_draw,
        "depth_cutoffs", (unsigned long long)s->resolver.depth_cutoffs,
        "max_depth_seen", (unsigned int)s->resolver.max_depth_seen,
        "qnodes", (unsigned long long)s->qnodes,
        "qterminal_draw", (unsigned long long)s->qterminal_draw,
        "qply_cutoffs", (unsigned long long)s->qply_cutoffs,
        "qmax_ply_seen", (unsigned int)s->qmax_ply_seen,
        "resolver_max_depth", ctx->resolver_max_depth,
        "qsearch_max_ply", ctx->qsearch_max_ply,
        "qsearch_check_plies", ctx->qsearch_check_plies);
}

PyDoc_STRVAR(set_arm_config_doc,
"set_arm_config(resolver_max_depth, qsearch_max_ply, qsearch_check_plies) -> dict\n\n"
"Set the configuration NEW arm contexts will be built with, returning what is in\n"
"force. qsearch_max_ply=0 collapses quiescence to a stand-pat, which makes the\n"
"qsearch arm's leaf identical to the static arm's — the arm's own negative\n"
"control.\n\n"
"⚑ Takes effect at the next init(), not on contexts that already exist: they\n"
"snapshot their configuration when they are built. Read what a batch actually\n"
"used out of arm_eval()'s stats, not back out of this function.");

static PyObject *py_set_arm_config(PyObject *Py_UNUSED(self), PyObject *args) {
    int depth, max_ply, check_plies;
    if (!PyArg_ParseTuple(args, "iii", &depth, &max_ply, &check_plies)) return NULL;
    char err[256] = {0};
    if (cae_arm_set_config(depth, max_ply, check_plies, err, sizeof(err)) != 0) {
        PyErr_SetString(PyExc_ValueError, err);
        return NULL;
    }
    return Py_BuildValue("{s:i,s:i,s:i}",
                         "resolver_max_depth", g_arm_resolver_max_depth,
                         "qsearch_max_ply", g_arm_qsearch_max_ply,
                         "qsearch_check_plies", g_arm_qsearch_check_plies);
}

static PyObject *py_arm_eval(PyObject *Py_UNUSED(self), PyObject *args) {
    const char *name, *path;
    PyObject *seq;
    if (!PyArg_ParseTuple(args, "ssO", &name, &path, &seq)) return NULL;

    const CaeValueProvider *vp = arm_provider_by_name(name);
    if (!vp) return NULL;

    PyObject *fast = PySequence_Fast(seq, "boards must be a sequence of CBoard");
    if (!fast) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);

    /* Unwrap every board BEFORE loading weights: a TypeError halfway through
     * would otherwise leak the ctx, and the type check is the cheap part. */
    const CBoard **boards = NULL;
    if (n > 0) {
        boards = (const CBoard **)PyMem_Malloc((size_t)n * sizeof(*boards));
        if (!boards) { Py_DECREF(fast); return PyErr_NoMemory(); }
        for (Py_ssize_t i = 0; i < n; i++) {
            boards[i] = unwrap_cboard(PySequence_Fast_GET_ITEM(fast, i));
            if (!boards[i]) { PyMem_Free(boards); Py_DECREF(fast); return NULL; }
        }
    }
    Py_DECREF(fast);

    char err[512] = {0};
    void *ctx = vp->init(path, err, sizeof(err));
    if (!ctx) {
        PyMem_Free(boards);
        PyErr_Format(PyExc_ValueError, "provider %s init failed: %s", vp->name, err);
        return NULL;
    }

    int32_t *raw = NULL;
    if (n > 0) {
        raw = (int32_t *)PyMem_Malloc((size_t)n * sizeof(*raw));
        if (!raw) { vp->destroy(ctx); PyMem_Free(boards); return PyErr_NoMemory(); }
    }

    /* ⚑ ONE GIL release around the WHOLE batch, not one per board. Timing this
     * against _nnue_ext.benchmark() is the point of the surface, and a
     * per-board acquire/release would bill the arms for GIL churn the baseline
     * does not pay — an overhead measurement that measures the harness. */
    Py_ssize_t failed_at = -1;
    int failed_status = CAE_VALUE_OK;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n; i++) {
        int status = cae_value_eval(vp, ctx, boards[i], &raw[i]);
        if (status != CAE_VALUE_OK) { failed_at = i; failed_status = status; break; }
    }
    Py_END_ALLOW_THREADS
    PyMem_Free(boards);

    if (failed_at >= 0) {
        raise_status(failed_status);
        PyMem_Free(raw);
        vp->destroy(ctx);
        return NULL;
    }

    PyObject *values = PyList_New(n);
    if (!values) { PyMem_Free(raw); vp->destroy(ctx); return NULL; }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyLong_FromLong((long)raw[i]);
        if (!item) { Py_DECREF(values); PyMem_Free(raw); vp->destroy(ctx); return NULL; }
        PyList_SET_ITEM(values, i, item);
    }
    PyMem_Free(raw);

    PyObject *stats = arm_stats_dict((const CaeArmCtx *)ctx);
    vp->destroy(ctx);
    if (!stats) { Py_DECREF(values); return NULL; }
    return Py_BuildValue("(NN)", values, stats);
}

/* ================================================================
 * Long-lived arm handles
 * ================================================================
 *
 * arm_eval() builds a context, uses it, and drops it. That is convenient and it
 * is NOT how a corpus generator works: the generator holds one context for a
 * whole run and the counters that matter are the run's, not a batch's. It is
 * also the only shape in which the configuration snapshot is OBSERVABLE — a
 * context outlives a set_arm_config(), and the numbers it reports must be the
 * ones it was built with. A surface where the two can never disagree cannot
 * demonstrate which of them is being reported.
 */

typedef struct {
    const CaeValueProvider *vp;
    void *ctx;
} ArmHandle;

static void arm_handle_destructor(PyObject *capsule) {
    ArmHandle *h = (ArmHandle *)PyCapsule_GetPointer(capsule, "cae.nnue.arm");
    if (!h) return;
    if (h->vp && h->ctx) h->vp->destroy(h->ctx);
    PyMem_Free(h);
}

static ArmHandle *arm_handle_from_capsule(PyObject *capsule) {
    ArmHandle *h = (ArmHandle *)PyCapsule_GetPointer(capsule, "cae.nnue.arm");
    if (!h) PyErr_SetString(PyExc_TypeError, "expected an arm handle from arm_open()");
    return h;
}

PyDoc_STRVAR(arm_open_doc,
"arm_open(name, weights_path) -> handle\n\n"
"Build one arm context and keep it. It snapshots set_arm_config()'s values NOW\n"
"and uses them for its whole life, so a later set_arm_config() does not retune\n"
"it — read arm_stats(handle) to see what it is really running.");

static PyObject *py_arm_open(PyObject *Py_UNUSED(self), PyObject *args) {
    const char *name, *path;
    if (!PyArg_ParseTuple(args, "ss", &name, &path)) return NULL;
    const CaeValueProvider *vp = arm_provider_by_name(name);
    if (!vp) return NULL;

    char err[512] = {0};
    void *ctx = vp->init(path, err, sizeof(err));
    if (!ctx) {
        PyErr_Format(PyExc_ValueError, "provider %s init failed: %s", vp->name, err);
        return NULL;
    }
    ArmHandle *h = (ArmHandle *)PyMem_Malloc(sizeof(*h));
    if (!h) { vp->destroy(ctx); return PyErr_NoMemory(); }
    h->vp = vp;
    h->ctx = ctx;
    PyObject *capsule = PyCapsule_New(h, "cae.nnue.arm", arm_handle_destructor);
    if (!capsule) { vp->destroy(ctx); PyMem_Free(h); return NULL; }
    return capsule;
}

PyDoc_STRVAR(arm_stats_doc,
"arm_stats(handle) -> dict\n\n"
"⚑ The counters and the configuration OF THAT CONTEXT, accumulated across every\n"
"evaluation it has done. The configuration keys are the context's own fields, so\n"
"they can differ from what set_arm_config() currently holds — and when they do,\n"
"the context's is the number that governed the work.");

static PyObject *py_arm_stats(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    ArmHandle *h = arm_handle_from_capsule(capsule);
    if (!h) return NULL;
    return arm_stats_dict((const CaeArmCtx *)h->ctx);
}

PyDoc_STRVAR(arm_handle_eval_doc,
"arm_handle_eval(handle, boards) -> list[int]\n\n"
"Evaluate through an arm_open() context, accumulating into its counters.");

static PyObject *py_arm_handle_eval(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *capsule, *seq;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &seq)) return NULL;
    ArmHandle *h = arm_handle_from_capsule(capsule);
    if (!h) return NULL;

    PyObject *fast = PySequence_Fast(seq, "boards must be a sequence of CBoard");
    if (!fast) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    PyObject *values = PyList_New(n);
    if (!values) { Py_DECREF(fast); return NULL; }

    for (Py_ssize_t i = 0; i < n; i++) {
        const CBoard *board = unwrap_cboard(PySequence_Fast_GET_ITEM(fast, i));
        if (!board) { Py_DECREF(values); Py_DECREF(fast); return NULL; }
        int32_t value = 0;
        int status;
        Py_BEGIN_ALLOW_THREADS
        status = cae_value_eval(h->vp, h->ctx, board, &value);
        Py_END_ALLOW_THREADS
        if (status != CAE_VALUE_OK) {
            raise_status(status);
            Py_DECREF(values);
            Py_DECREF(fast);
            return NULL;
        }
        PyObject *item = PyLong_FromLong((long)value);
        if (!item) { Py_DECREF(values); Py_DECREF(fast); return NULL; }
        PyList_SET_ITEM(values, i, item);
    }
    Py_DECREF(fast);
    return values;
}

PyDoc_STRVAR(provider_names_doc, "provider_names() -> tuple of registered provider names");

static PyObject *py_provider_names(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args)) {
    int n = 0;
    while (CAE_VALUE_PROVIDERS[n]) n++;
    PyObject *out = PyTuple_New(n);
    if (!out) return NULL;
    for (int i = 0; i < n; i++)
        PyTuple_SET_ITEM(out, i, PyUnicode_FromString(CAE_VALUE_PROVIDERS[i]->name));
    return out;
}

PyDoc_STRVAR(set_simd_doc,
"set_simd(enabled) -> bool\n\n"
"Select the AVX2 kernels (True) or the scalar reference kernels (False) at\n"
"runtime, returning the state actually in force. Both paths are in the same\n"
"binary on purpose: an unexercised SIMD path is an unchecked one, and the parity\n"
"gate has to be runnable against each. Raises if SIMD is requested but was not\n"
"compiled in. Not thread-safe — set it before starting worker threads.");

static PyObject *py_set_simd(PyObject *Py_UNUSED(self), PyObject *args) {
    int enabled;
    if (!PyArg_ParseTuple(args, "p", &enabled)) return NULL;
    if (cae_nnue_set_simd(enabled) != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "this build has no AVX2 kernels (compiled without __AVX2__)");
        return NULL;
    }
    return PyBool_FromLong(cae_nnue_simd_active());
}

PyDoc_STRVAR(simd_active_doc,
"simd_active() -> bool\n\n"
"Which kernels the evaluator will actually use, read off the live flag rather\n"
"than off whatever was requested.");

static PyObject *py_simd_active(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args)) {
    return PyBool_FromLong(cae_nnue_simd_active());
}

PyDoc_STRVAR(weight_cache_size_doc,
"weight_cache_size() -> int\n\n"
"How many distinct weight files THIS module's evaluator is holding mapped. The\n"
"cache is a static of the evaluator, so the count is per copy of the evaluator's\n"
"code — which makes it the instrument for 'did that other extension really call\n"
"our provider, or a second copy of the evaluator compiled into itself'. A second\n"
"copy would keep its own cache and leave this one reading 0.");

static PyObject *py_weight_cache_size(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args)) {
    return PyLong_FromLong((long)cae_nnue_cache_size());
}

static PyMethodDef module_methods[] = {
    {"load", py_load, METH_VARARGS, load_doc},
    {"set_simd", py_set_simd, METH_VARARGS, set_simd_doc},
    {"simd_active", py_simd_active, METH_NOARGS, simd_active_doc},
    {"weight_cache_size", py_weight_cache_size, METH_NOARGS, weight_cache_size_doc},
    {"info", py_info, METH_VARARGS, info_doc},
    {"source_sha256", py_source_sha256, METH_VARARGS, source_sha256_doc},
    {"evaluate", py_evaluate, METH_VARARGS, evaluate_doc},
    {"trace", py_trace, METH_VARARGS, trace_doc},
    {"active_features", py_active_features, METH_VARARGS, active_features_doc},
    {"benchmark", py_benchmark, METH_VARARGS, benchmark_doc},
    {"provider_eval", py_provider_eval, METH_VARARGS, provider_eval_doc},
    {"provider_names", py_provider_names, METH_NOARGS, provider_names_doc},
    {"arm_eval", py_arm_eval, METH_VARARGS, arm_eval_doc},
    {"set_arm_config", py_set_arm_config, METH_VARARGS, set_arm_config_doc},
    {"arm_open", py_arm_open, METH_VARARGS, arm_open_doc},
    {"arm_handle_eval", py_arm_handle_eval, METH_VARARGS, arm_handle_eval_doc},
    {"arm_stats", py_arm_stats, METH_VARARGS, arm_stats_doc},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef nnue_module = {
    PyModuleDef_HEAD_INIT,
    "_nnue_ext",
    "Native big-net Stockfish-NNUE evaluator (parity/bench/test surface).",
    -1,
    module_methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__nnue_ext(void) {
    cboard_init_all();
    cae_nnue_init_tables();

    PyObject *m = PyModule_Create(&nnue_module);
    if (!m) return NULL;

    NnueInCheckError = PyErr_NewExceptionWithDoc(
        "chess_anti_engine.nnue._nnue_ext.InCheckError",
        "The NNUE evaluation is undefined for a position in check.\n\n"
        "Callers must resolve check nodes recursively (search the evasions, which\n"
        "may themselves give check) before asking for a static evaluation. This\n"
        "exception is the enforcement backstop for that invariant.",
        PyExc_ValueError, NULL);
    if (!NnueInCheckError) { Py_DECREF(m); return NULL; }
    Py_INCREF(NnueInCheckError);
    if (PyModule_AddObject(m, "InCheckError", NnueInCheckError) < 0) {
        Py_DECREF(NnueInCheckError);
        Py_DECREF(m);
        return NULL;
    }
    /* ⚑ Publish the provider for OTHER extensions. Static storage: the capsule
     * hands out a pointer, so the struct must outlive every consumer, and both
     * fields point at objects that live as long as this module does. */
    static CaeValueProviderExport nnue_export;
    nnue_export.abi_version = CAE_VALUE_PROVIDER_ABI;
    nnue_export.struct_size = (uint32_t)sizeof(CaeValueProviderExport);
    nnue_export.provider = &CAE_NNUE_PROVIDER;
    nnue_export.in_check_error = (struct _object *)NnueInCheckError;

    PyObject *export_capsule = PyCapsule_New(&nnue_export,
                                             CAE_VALUE_PROVIDER_CAPSULE_NAME, NULL);
    if (!export_capsule) { Py_DECREF(m); return NULL; }
    if (PyModule_AddObject(m, "value_provider_capsule", export_capsule) < 0) {
        Py_DECREF(export_capsule);
        Py_DECREF(m);
        return NULL;
    }

    /* The two race arms, published the same way and under the same capsule name.
     * Same shape, same ABI, different vtable — so a consumer installs one by
     * handing the tree an attribute, with no consumer-side edit per arm. */
    static CaeValueProviderExport arm_exports[2];
    static const struct { const char *attr; const CaeValueProvider *vp; } arm_pub[2] = {
        {"static_arm_capsule", &CAE_ARM_STATIC_PROVIDER},
        {"qsearch_arm_capsule", &CAE_ARM_QSEARCH_PROVIDER},
    };
    for (int i = 0; i < 2; i++) {
        arm_exports[i].abi_version = CAE_VALUE_PROVIDER_ABI;
        arm_exports[i].struct_size = (uint32_t)sizeof(CaeValueProviderExport);
        arm_exports[i].provider = arm_pub[i].vp;
        arm_exports[i].in_check_error = (struct _object *)NnueInCheckError;
        PyObject *cap = PyCapsule_New(&arm_exports[i],
                                      CAE_VALUE_PROVIDER_CAPSULE_NAME, NULL);
        if (!cap) { Py_DECREF(m); return NULL; }
        if (PyModule_AddObject(m, arm_pub[i].attr, cap) < 0) {
            Py_DECREF(cap);
            Py_DECREF(m);
            return NULL;
        }
    }

    PyModule_AddIntConstant(m, "RESOLVER_EVAL_CLAMP", (long)CAE_RESOLVER_EVAL_CLAMP);
    PyModule_AddIntConstant(m, "RESOLVER_MATE_BASE", (long)CAE_RESOLVER_MATE_BASE);
    PyModule_AddIntConstant(m, "RESOLVER_MATE_PLY_STEP", (long)CAE_RESOLVER_MATE_PLY_STEP);
    PyModule_AddIntConstant(m, "RESOLVER_MAX_PLIES", (long)CAE_RESOLVER_MAX_PLIES);
    PyModule_AddIntConstant(m, "RESOLVER_MAX_DEPTH", (long)CAE_RESOLVER_DEFAULT_MAX_DEPTH);
    PyModule_AddIntConstant(m, "QSEARCH_MAX_PLY", (long)CAE_QSEARCH_DEFAULT_MAX_PLY);
    PyModule_AddIntConstant(m, "QSEARCH_CHECK_PLIES", (long)CAE_QSEARCH_DEFAULT_CHECK_PLIES);

    PyModule_AddIntConstant(m, "THREAT_DIMS", (long)CAE_NNUE_THREAT_DIMS);
    PyModule_AddIntConstant(m, "HALFKA_DIMS", (long)CAE_NNUE_HALFKA_DIMS);
    PyModule_AddIntConstant(m, "PACK_VERSION", (long)CAE_NNUE_PACK_VERSION);
    PyModule_AddIntConstant(m, "FILE_VERSION", (long)CAE_NNUE_FILE_VERSION);
    PyModule_AddIntConstant(m, "HAVE_AVX2", (long)CAE_NNUE_HAVE_AVX2);
    return m;
}
