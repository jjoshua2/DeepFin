/* Native perft over the same CBoard core used by _lc0_ext and the C tree.
 * This diagnostic extension deliberately does not link a second move generator
 * or call Python while expanding moves. The Python boundary snapshots only the
 * position: repetition/history and draw adjudication do not define perft leaves.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "_cboard_impl.h"

#define PERFT_MAX_DEPTH 64
#define PERFT_SIGNAL_INTERVAL 4096

typedef struct {
    PyThreadState *thread_state;
    unsigned int until_signal_check;
    int interrupted;
    int overflow;
} PerftContext;

/* Called with the GIL released. Check signals periodically, not per leaf. */
static int perft_poll(PerftContext *ctx) {
    if (--ctx->until_signal_check != 0) return 0;
    ctx->until_signal_check = PERFT_SIGNAL_INTERVAL;
    PyEval_RestoreThread(ctx->thread_state);
    ctx->interrupted = PyErr_CheckSignals() < 0;
    ctx->thread_state = PyEval_SaveThread();
    return ctx->interrupted;
}

static uint64_t perft_count(const CBoard *board, int depth, PerftContext *ctx) {
    if (depth == 0) return 1;
    if (perft_poll(ctx)) return 0;
    int indices[256];
    int count = cboard_legal_move_indices(board, indices, 0);
    if (depth == 1) return (uint64_t)count;  /* standard bulk leaf counting */
    uint64_t total = 0;
    for (int i = 0; i < count; i++) {
        CBoard child = *board;  /* CBoard has no unmake operation. */
        cboard_push_index(&child, indices[i]);
        uint64_t nodes = perft_count(&child, depth - 1, ctx);
        if (ctx->interrupted || ctx->overflow) return 0;
        if (UINT64_MAX - total < nodes) {
            ctx->overflow = 1;
            return 0;
        }
        total += nodes;
    }
    return total;
}

static int read_u64(PyObject *obj, const char *name, uint64_t *out) {
    PyObject *value = PyObject_GetAttrString(obj, name);
    if (!value) return -1;
    unsigned long long result = PyLong_AsUnsignedLongLong(value);
    Py_DECREF(value);
    if (PyErr_Occurred()) return -1;
    *out = (uint64_t)result;
    return 0;
}

static int read_int(PyObject *obj, const char *name, long *out) {
    PyObject *value = PyObject_GetAttrString(obj, name);
    if (!value) return -1;
    if (value == Py_None) {
        /* Only ep_square uses None in the CBoard interface. */
        Py_DECREF(value);
        *out = -1;
        return 0;
    }
    long result = PyLong_AsLong(value);
    Py_DECREF(value);
    if (PyErr_Occurred()) return -1;
    *out = result;
    return 0;
}

/* Read public CBoard properties once; never assume another .so's object ABI.
 * The caller retains its original board, including its entire history. The
 * private snapshot starts a new history because perft ignores draw rules.
 */
static int perft_snapshot(PyObject *obj, CBoard *board) {
    PyObject *module = PyImport_ImportModule("chess_anti_engine.encoding._lc0_ext");
    if (!module) return -1;
    PyObject *type = PyObject_GetAttrString(module, "CBoard");
    Py_DECREF(module);
    if (!type) return -1;
    int is_board = PyObject_IsInstance(obj, type);
    Py_DECREF(type);
    if (is_board < 0) return -1;
    if (!is_board) {
        PyErr_SetString(PyExc_TypeError, "board must be a CBoard");
        return -1;
    }
    memset(board, 0, sizeof(*board));
    cboard_reset_hist_ep(board);
    const char *pieces[6] = {"pawns", "knights", "bishops", "rooks", "queens", "kings"};
    for (int i = 0; i < 6; i++)
        if (read_u64(obj, pieces[i], &board->bb[i]) < 0) return -1;
    if (read_u64(obj, "occ_white", &board->occ[WHITE_C]) < 0 ||
        read_u64(obj, "occ_black", &board->occ[BLACK_C]) < 0) return -1;
    long turn, castling, ep, halfmove, ply;
    if (read_int(obj, "turn", &turn) < 0 ||
        read_int(obj, "castling", &castling) < 0 ||
        read_int(obj, "ep_square", &ep) < 0 ||
        read_int(obj, "halfmove_clock", &halfmove) < 0 ||
        read_int(obj, "ply", &ply) < 0) return -1;
    if (turn < 0 || turn > 1 || castling < 0 || castling > 15 ||
        ep < -1 || ep > 63 || halfmove < 0 || halfmove > 255 ||
        ply < 0 || ply > UINT16_MAX) {
        PyErr_SetString(PyExc_ValueError, "invalid CBoard position fields");
        return -1;
    }
    uint64_t occupied = 0;
    for (int i = 0; i < 6; i++) {
        if (occupied & board->bb[i]) {
            PyErr_SetString(PyExc_ValueError, "overlapping CBoard piece bitboards");
            return -1;
        }
        occupied |= board->bb[i];
    }
    if ((board->occ[0] & board->occ[1]) ||
        occupied != (board->occ[0] | board->occ[1]) ||
        popcount64(board->bb[KING] & board->occ[0]) != 1 ||
        popcount64(board->bb[KING] & board->occ[1]) != 1) {
        PyErr_SetString(PyExc_ValueError, "perft requires a standard-chess CBoard with one king per side");
        return -1;
    }
    board->turn = (int8_t)turn;
    board->castling = (uint8_t)castling;
    board->ep_square = (int8_t)ep;
    board->halfmove_clock = (uint8_t)halfmove;
    board->ply = (uint16_t)ply;
    board->hash = cboard_compute_hash(board);
    return 0;
}

static int perft_parse(PyObject *args, CBoard *board, int *depth, int divide) {
    PyObject *obj;
    if (!PyArg_ParseTuple(args, "Oi", &obj, depth)) return -1;
    int min_depth = divide ? 1 : 0;
    if (*depth < min_depth || *depth > PERFT_MAX_DEPTH) {
        PyErr_Format(PyExc_ValueError, "depth must be %d..%d", min_depth, PERFT_MAX_DEPTH);
        return -1;
    }
    return perft_snapshot(obj, board);
}

static int perft_finish(PerftContext *ctx) {
    PyEval_RestoreThread(ctx->thread_state);
    if (ctx->interrupted || PyErr_CheckSignals() < 0) return -1;
    if (ctx->overflow) {
        PyErr_SetString(PyExc_OverflowError, "perft node count exceeds uint64");
        return -1;
    }
    return 0;
}

static PyObject *py_perft(PyObject *self, PyObject *args) {
    CBoard board;
    int depth;
    if (perft_parse(args, &board, &depth, 0) < 0) return NULL;
    PerftContext ctx = {NULL, PERFT_SIGNAL_INTERVAL, 0, 0};
    ctx.thread_state = PyEval_SaveThread();
    uint64_t nodes = perft_count(&board, depth, &ctx);
    if (perft_finish(&ctx) < 0) return NULL;
    return PyLong_FromUnsignedLongLong(nodes);
}

/* UCI labels use real board coordinates, not side-oriented policy coordinates.
 * POLICY_LUT uses PROMO_MAYBE_QUEEN for ordinary pawn moves too; append 'q'
 * only for an actual pawn reaching its final rank. Explicit promotions use
 * python-chess piece numbers (2=knight, 3=bishop, 4=rook).
 */
static void perft_move_uci(const CBoard *board, int index, char uci[6]) {
    PolicyMove move = POLICY_LUT[board->turn][index];
    uci[0] = (char)('a' + sq_file(move.from_sq));
    uci[1] = (char)('1' + sq_rank(move.from_sq));
    uci[2] = (char)('a' + sq_file(move.to_sq));
    uci[3] = (char)('1' + sq_rank(move.to_sq));
    uci[4] = '\0';
    uci[5] = '\0';
    if (piece_type_at(board, move.from_sq) == PAWN &&
        (sq_rank(move.to_sq) == 0 || sq_rank(move.to_sq) == 7)) {
        if (move.promotion == 2) uci[4] = 'n';
        else if (move.promotion == 3) uci[4] = 'b';
        else if (move.promotion == 4) uci[4] = 'r';
        else uci[4] = 'q';
    }
}

static PyObject *py_perft_divide(PyObject *self, PyObject *args) {
    CBoard board;
    int depth;
    if (perft_parse(args, &board, &depth, 1) < 0) return NULL;
    int indices[256];
    uint64_t counts[256];
    int count = cboard_legal_move_indices(&board, indices, 1);
    PerftContext ctx = {NULL, PERFT_SIGNAL_INTERVAL, 0, 0};
    uint64_t total = 0;
    ctx.thread_state = PyEval_SaveThread();
    for (int i = 0; i < count; i++) {
        CBoard child = board;
        cboard_push_index(&child, indices[i]);
        counts[i] = perft_count(&child, depth - 1, &ctx);
        if (ctx.interrupted || ctx.overflow) break;
        if (UINT64_MAX - total < counts[i]) {
            ctx.overflow = 1;
            break;
        }
        total += counts[i];
    }
    if (perft_finish(&ctx) < 0) return NULL;
    PyObject *result = PyDict_New();
    if (!result) return NULL;
    for (int i = 0; i < count; i++) {
        char uci[6];
        perft_move_uci(&board, indices[i], uci);
        PyObject *value = PyLong_FromUnsignedLongLong(counts[i]);
        if (!value || PyDict_SetItemString(result, uci, value) < 0) {
            Py_XDECREF(value);
            Py_DECREF(result);
            return NULL;
        }
        Py_DECREF(value);
    }
    return result;
}

static PyMethodDef methods[] = {
    {"perft", py_perft, METH_VARARGS,
     "perft(board, depth, /) -> int. Count legal move sequences in native C."},
    {"perft_divide", py_perft_divide, METH_VARARGS,
     "perft_divide(board, depth, /) -> dict[str, int]. Root UCI move counts; depth >= 1."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_perft_ext", NULL, -1, methods
};

PyMODINIT_FUNC PyInit__perft_ext(void) {
    cboard_init_all();
    PyObject *module = PyModule_Create(&moduledef);
    if (!module) return NULL;
    if (PyModule_AddStringConstant(module, "SLIDER_BACKEND", DEEPFIN_SLIDER_BACKEND_NAME) < 0 ||
        PyModule_AddIntConstant(module, "MAX_DEPTH", PERFT_MAX_DEPTH) < 0) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
