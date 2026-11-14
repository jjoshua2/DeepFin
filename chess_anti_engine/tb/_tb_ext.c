/* _tb_ext: Syzygy WDL/DTZ probing in C (vendored Fathom), no python-chess.
 *
 * Nothing in production calls this yet: it exists so the python-chess prober
 * (chess.syzygy, used through chess_anti_engine/tablebase.py) can be replaced
 * call site by call site, each switch behind its own record. Equivalence with
 * python-chess and the WDL/DTZ conventions: docs/tablebase_prober.md.
 *
 * Positions come in as a CBoard (the C board the search already holds) or a
 * FEN string. Results, side-to-move point of view:
 *   probe_wdl -> -2 loss, -1 blessed loss, 0 draw, 1 cursed win, 2 win
 *   probe_dtz -> python-chess probe_dtz convention
 *   probe_root -> [(uci, dtz, rank, score), ...] per legal move, 50-move aware
 * None means "not probeable here": castling rights, more men than the loaded
 * tables, or a table missing from every directory. An illegal position or a
 * malformed FEN raises ValueError.
 *
 * THREAD SAFETY. Fathom's tables are process-global, so this module is a
 * process-wide singleton:
 *   - every probe releases the GIL and holds a shared (read) lock on the
 *     table set, so WDL/DTZ probes run concurrently (Fathom's lazy table
 *     loading is internally mutex-protected);
 *   - init()/free() take the same lock exclusively, so tables can never be
 *     unmapped under an in-flight probe;
 *   - probe_root additionally holds a module mutex: Fathom documents
 *     tb_probe_root as not thread-safe, and this wrapper makes concurrent
 *     calls serialise instead of trusting callers.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <pthread.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../encoding/_cboard_impl.h"
#include "_tb_fathom.h"

/* PyCBoard layout -- must match _lc0_ext.c's typedef exactly (the same
 * contract _mcts_tree.c relies on; checked by tp_basicsize at first use). */
typedef struct { PyObject_HEAD CBoard board; } PyCBoard;

static PyTypeObject *g_cboard_type = NULL;

static pthread_rwlock_t g_tables_lock = PTHREAD_RWLOCK_INITIALIZER;
static pthread_mutex_t g_root_lock = PTHREAD_MUTEX_INITIALIZER;
static PyObject *g_path = NULL;  /* str of the last successful init, or NULL */

static int ensure_cboard_type(void) {
    if (g_cboard_type) return 0;
    PyObject *mod = PyImport_ImportModule("chess_anti_engine.encoding._lc0_ext");
    if (!mod) return -1;
    PyObject *type_obj = PyObject_GetAttrString(mod, "CBoard");
    Py_DECREF(mod);
    if (!type_obj) return -1;
    if (!PyType_Check(type_obj) ||
        ((PyTypeObject *)type_obj)->tp_basicsize != (Py_ssize_t)sizeof(PyCBoard)) {
        Py_DECREF(type_obj);
        PyErr_SetString(PyExc_ImportError,
            "CBoard extension ABI mismatch; rebuild C extensions");
        return -1;
    }
    g_cboard_type = (PyTypeObject *)type_obj;
    return 0;
}

/* ---------------------------------------------------------------- inputs */

static int pos_from_cboard(PyObject *obj, DfTbPos *pos) {
    if (ensure_cboard_type() < 0) return -1;
    if (!PyObject_TypeCheck(obj, g_cboard_type)) {
        PyErr_SetString(PyExc_TypeError, "expected a CBoard");
        return -1;
    }
    const CBoard *b = &((PyCBoard *)obj)->board;
    pos->white = b->occ[WHITE_C];
    pos->black = b->occ[BLACK_C];
    pos->pawns = b->bb[0];
    pos->knights = b->bb[1];
    pos->bishops = b->bb[2];
    pos->rooks = b->bb[3];
    pos->queens = b->bb[4];
    pos->kings = b->bb[5];
    pos->rule50 = b->halfmove_clock;
    pos->castling = b->castling;
    pos->ep = (b->ep_square >= 0 && b->ep_square < 64) ? (unsigned)b->ep_square : 0u;
    pos->turn = b->turn == WHITE_C ? 1 : 0;
    return 0;
}

static int fen_square(const char *s) {
    if (s[0] < 'a' || s[0] > 'h' || s[1] < '1' || s[1] > '8') return -1;
    return (s[1] - '1') * 8 + (s[0] - 'a');
}

/* Minimal FEN reader: placement, side, castling, ep, optional halfmove clock.
 * Square numbering a1=0 .. h8=63, the same as CBoard and python-chess. */
static int pos_from_fen(const char *fen, DfTbPos *pos) {
    memset(pos, 0, sizeof(*pos));
    const char *p = fen;
    int rank = 7, file = 0;
    for (; *p && *p != ' '; p++) {
        char c = *p;
        if (c == '/') {
            if (file != 8 || rank == 0) goto bad;
            rank--;
            file = 0;
            continue;
        }
        if (c >= '1' && c <= '8') {
            file += c - '0';
            if (file > 8) goto bad;
            continue;
        }
        if (file >= 8) goto bad;
        uint64_t bit = 1ULL << (rank * 8 + file);
        int white = (c >= 'A' && c <= 'Z');
        switch (white ? c : (char)(c - 'a' + 'A')) {
            case 'P': pos->pawns |= bit; break;
            case 'N': pos->knights |= bit; break;
            case 'B': pos->bishops |= bit; break;
            case 'R': pos->rooks |= bit; break;
            case 'Q': pos->queens |= bit; break;
            case 'K': pos->kings |= bit; break;
            default: goto bad;
        }
        if (white) pos->white |= bit; else pos->black |= bit;
        file++;
    }
    if (rank != 0 || file != 8 || *p != ' ') goto bad;
    p++;
    if (*p == 'w') pos->turn = 1;
    else if (*p == 'b') pos->turn = 0;
    else goto bad;
    p++;
    if (*p != ' ') goto bad;
    p++;
    if (*p == '-') {
        p++;
    } else {
        for (; *p && *p != ' '; p++) {
            switch (*p) {
                case 'K': pos->castling |= 1u; break;
                case 'Q': pos->castling |= 2u; break;
                case 'k': pos->castling |= 4u; break;
                case 'q': pos->castling |= 8u; break;
                default: goto bad;
            }
        }
    }
    if (*p != ' ') goto bad;
    p++;
    /* Castling flags are kept RAW, never cleaned: python-chess's probe tests
     * board.castling_rights as parsed (a stale "K" with the king off e1 still
     * raises KeyError there) and CBoard.from_board copies the raw flags, so a
     * cleaned FEN reader would be the one path that disagrees. */
    if (*p == '-') {
        p++;
    } else {
        int sq = fen_square(p);
        if (sq < 0) goto bad;
        /* Any square parses; the probe gate decides whether it is possible
         * (a1 = square 0 is "none" internally, so it is mapped to the
         * always-impossible 64). */
        pos->ep = sq == 0 ? 64u : (unsigned)sq;
        p += 2;
    }
    if (*p == ' ') {
        p++;
        unsigned clock = 0;
        int digits = 0;
        while (*p >= '0' && *p <= '9') {
            clock = clock * 10u + (unsigned)(*p - '0');
            if (clock > 100000u) goto bad;
            digits++;
            p++;
        }
        if (!digits) goto bad;
        pos->rule50 = clock;
        /* An optional fullmove number, read and discarded; nothing after it. */
        if (*p == ' ') {
            p++;
            if (*p < '0' || *p > '9') goto bad;
            while (*p >= '0' && *p <= '9') p++;
        }
        if (*p != '\0') goto bad;
    } else if (*p != '\0') {
        goto bad;
    }
    return 0;
bad:
    PyErr_Format(PyExc_ValueError, "malformed FEN: %s", fen);
    return -1;
}

/* ------------------------------------------------------------ probe core */

typedef enum { KIND_WDL, KIND_DTZ } ProbeKind;

static PyObject *finish_scalar(int rc, int value) {
    if (rc == DF_TB_OK) return PyLong_FromLong(value);
    if (rc == DF_TB_UNAVAILABLE) Py_RETURN_NONE;
    PyErr_SetString(PyExc_ValueError, "not a legal chess position");
    return NULL;
}

static PyObject *probe_scalar(const DfTbPos *pos, ProbeKind kind) {
    int value = 0, rc;
    Py_BEGIN_ALLOW_THREADS
    pthread_rwlock_rdlock(&g_tables_lock);
    rc = (kind == KIND_WDL) ? df_tb_probe_wdl(pos, &value) : df_tb_probe_dtz(pos, &value);
    pthread_rwlock_unlock(&g_tables_lock);
    Py_END_ALLOW_THREADS
    return finish_scalar(rc, value);
}

static PyObject *probe_root_core(const DfTbPos *pos, int has_repeated, int use_rule50) {
    DfTbRootMove moves[DF_TB_MAX_ROOT_MOVES];
    int n = 0, rc;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&g_root_lock);
    pthread_rwlock_rdlock(&g_tables_lock);
    rc = df_tb_probe_root(pos, has_repeated, use_rule50, moves, &n);
    pthread_rwlock_unlock(&g_tables_lock);
    pthread_mutex_unlock(&g_root_lock);
    Py_END_ALLOW_THREADS
    if (rc == DF_TB_UNAVAILABLE) Py_RETURN_NONE;
    if (rc == DF_TB_INVALID) {
        PyErr_SetString(PyExc_ValueError, "not a legal chess position");
        return NULL;
    }
    if (rc != DF_TB_OK) return PyErr_NoMemory();
    static const char promo_chr[] = {0, 'q', 'r', 'b', 'n'};
    PyObject *list = PyList_New(n);
    if (!list) return NULL;
    for (int i = 0; i < n; i++) {
        char uci[6];
        const DfTbRootMove *m = &moves[i];
        uci[0] = (char)('a' + m->from_sq % 8);
        uci[1] = (char)('1' + m->from_sq / 8);
        uci[2] = (char)('a' + m->to_sq % 8);
        uci[3] = (char)('1' + m->to_sq / 8);
        int len = 4;
        if (m->promotes > 0 && m->promotes <= 4) uci[len++] = promo_chr[m->promotes];
        uci[len] = '\0';
        PyObject *item = Py_BuildValue("(siii)", uci, m->dtz, m->rank, m->score);
        if (!item) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, i, item);
    }
    return list;
}

/* ------------------------------------------------------------- functions */

#ifdef _WIN32
#define DF_TB_SEP ';'   /* Fathom's own SEP_CHAR on Windows */
#else
#define DF_TB_SEP ':'
#endif

/* Every component of a non-empty path must be a readable directory. Checked
 * BEFORE Fathom re-initialises, so a bad path leaves the old table set loaded.
 * Fathom itself would skip a missing directory silently and report success. */
static int check_components(const char *path) {
    const char *start = path;
    for (;;) {
        const char *end = strchr(start, DF_TB_SEP);
        size_t len = end ? (size_t)(end - start) : strlen(start);
        char dir[4096];
        if (len == 0 || len >= sizeof dir) {
            PyErr_Format(PyExc_ValueError,
                         "Syzygy path has an empty or oversized component: %s", path);
            return -1;
        }
        memcpy(dir, start, len);
        dir[len] = '\0';
        struct stat st;
        if (stat(dir, &st) != 0 || !S_ISDIR(st.st_mode) || access(dir, R_OK | X_OK) != 0) {
            PyErr_Format(PyExc_ValueError,
                         "Syzygy path component is not a readable directory: %s", dir);
            return -1;
        }
        if (!end) return 0;
        start = end + 1;
    }
}

static PyObject *tb_init_py(PyObject *self, PyObject *args) {
    const char *path;
    if (!PyArg_ParseTuple(args, "s", &path)) return NULL;
    if (path[0] != '\0' && check_components(path) < 0) return NULL;
    PyObject *keep = PyUnicode_FromString(path);
    if (!keep) return NULL;
    int rc;
    unsigned largest;
    Py_BEGIN_ALLOW_THREADS
    pthread_rwlock_wrlock(&g_tables_lock);
    rc = df_tb_init(path);
    largest = df_tb_largest();
    pthread_rwlock_unlock(&g_tables_lock);
    Py_END_ALLOW_THREADS
    if (rc != 0) {
        Py_DECREF(keep);
        Py_CLEAR(g_path);
        PyErr_Format(PyExc_RuntimeError, "Fathom tb_init failed for %s", path);
        return NULL;
    }
    if (path[0] != '\0' && largest == 0) {
        /* Accepted-and-ignored is this repo's signature defect: a path that
         * loads nothing is an error, not "no tablebases". The table set is
         * now empty, and path() says so. */
        Py_DECREF(keep);
        Py_CLEAR(g_path);
        PyErr_Format(PyExc_ValueError, "no Syzygy tables found under %s", path);
        return NULL;
    }
    if (path[0] == '\0') Py_CLEAR(keep);
    Py_XSETREF(g_path, keep);
    return PyLong_FromUnsignedLong(largest);
}

static PyObject *tb_free_py(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    Py_BEGIN_ALLOW_THREADS
    pthread_rwlock_wrlock(&g_tables_lock);
    df_tb_free();
    pthread_rwlock_unlock(&g_tables_lock);
    Py_END_ALLOW_THREADS
    Py_CLEAR(g_path);
    Py_RETURN_NONE;
}

static PyObject *tb_largest_py(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    unsigned largest;
    pthread_rwlock_rdlock(&g_tables_lock);
    largest = df_tb_largest();
    pthread_rwlock_unlock(&g_tables_lock);
    return PyLong_FromUnsignedLong(largest);
}

static PyObject *tb_path_py(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    if (!g_path) Py_RETURN_NONE;
    return Py_NewRef(g_path);
}

static PyObject *tb_probe_wdl_py(PyObject *self, PyObject *obj) {
    DfTbPos pos;
    if (pos_from_cboard(obj, &pos) < 0) return NULL;
    return probe_scalar(&pos, KIND_WDL);
}

static PyObject *tb_probe_dtz_py(PyObject *self, PyObject *obj) {
    DfTbPos pos;
    if (pos_from_cboard(obj, &pos) < 0) return NULL;
    return probe_scalar(&pos, KIND_DTZ);
}

static PyObject *tb_probe_wdl_fen_py(PyObject *self, PyObject *args) {
    const char *fen;
    DfTbPos pos;
    if (!PyArg_ParseTuple(args, "s", &fen)) return NULL;
    if (pos_from_fen(fen, &pos) < 0) return NULL;
    return probe_scalar(&pos, KIND_WDL);
}

static PyObject *tb_probe_dtz_fen_py(PyObject *self, PyObject *args) {
    const char *fen;
    DfTbPos pos;
    if (!PyArg_ParseTuple(args, "s", &fen)) return NULL;
    if (pos_from_fen(fen, &pos) < 0) return NULL;
    return probe_scalar(&pos, KIND_DTZ);
}

static PyObject *tb_probe_root_py(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"board", "has_repeated", "use_rule50", NULL};
    PyObject *obj;
    int has_repeated = 0, use_rule50 = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|pp", kwlist,
                                     &obj, &has_repeated, &use_rule50)) return NULL;
    DfTbPos pos;
    if (pos_from_cboard(obj, &pos) < 0) return NULL;
    return probe_root_core(&pos, has_repeated, use_rule50);
}

static PyObject *tb_probe_root_fen_py(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"fen", "has_repeated", "use_rule50", NULL};
    const char *fen;
    int has_repeated = 0, use_rule50 = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|pp", kwlist,
                                     &fen, &has_repeated, &use_rule50)) return NULL;
    DfTbPos pos;
    if (pos_from_fen(fen, &pos) < 0) return NULL;
    return probe_root_core(&pos, has_repeated, use_rule50);
}

/* fen_bitboards(fen) -> (white, black, kings, queens, rooks, bishops, knights,
 * pawns, ep, turn, castling, rule50): the parsed input, for tests that pin the
 * FEN reader against python-chess. */
static PyObject *tb_fen_bitboards_py(PyObject *self, PyObject *args) {
    const char *fen;
    DfTbPos p;
    if (!PyArg_ParseTuple(args, "s", &fen)) return NULL;
    if (pos_from_fen(fen, &p) < 0) return NULL;
    return Py_BuildValue("(KKKKKKKKIiII)",
        (unsigned long long)p.white, (unsigned long long)p.black,
        (unsigned long long)p.kings, (unsigned long long)p.queens,
        (unsigned long long)p.rooks, (unsigned long long)p.bishops,
        (unsigned long long)p.knights, (unsigned long long)p.pawns,
        p.ep, p.turn, p.castling, p.rule50);
}

/* cboard_bitboards(cb): the same tuple read from a CBoard. */
static PyObject *tb_cboard_bitboards_py(PyObject *self, PyObject *obj) {
    DfTbPos p;
    if (pos_from_cboard(obj, &p) < 0) return NULL;
    return Py_BuildValue("(KKKKKKKKIiII)",
        (unsigned long long)p.white, (unsigned long long)p.black,
        (unsigned long long)p.kings, (unsigned long long)p.queens,
        (unsigned long long)p.rooks, (unsigned long long)p.bishops,
        (unsigned long long)p.knights, (unsigned long long)p.pawns,
        p.ep, p.turn, p.castling, p.rule50);
}

static PyMethodDef tb_methods[] = {
    {"init", tb_init_py, METH_VARARGS,
     "init(path: str) -> int. Load Syzygy tables from a ':'-separated directory "
     "list (every directory is searched; the first holding a file wins). "
     "Replaces any earlier table set. Returns the largest man count found "
     "(0 if none; an empty path clears the tables)."},
    {"free", tb_free_py, METH_NOARGS, "Unmap every table (largest() becomes 0)."},
    {"largest", tb_largest_py, METH_NOARGS, "Largest man count with a WDL table."},
    {"path", tb_path_py, METH_NOARGS, "Path string of the current table set, or None."},
    {"probe_wdl", tb_probe_wdl_py, METH_O,
     "probe_wdl(cboard) -> int | None. WDL -2..2 for the side to move; ignores "
     "the halfmove clock (python-chess probe_wdl contract)."},
    {"probe_dtz", tb_probe_dtz_py, METH_O,
     "probe_dtz(cboard) -> int | None. DTZ for the side to move; ignores the "
     "halfmove clock (python-chess probe_dtz contract)."},
    {"probe_wdl_fen", tb_probe_wdl_fen_py, METH_VARARGS, "probe_wdl for a FEN string."},
    {"probe_dtz_fen", tb_probe_dtz_fen_py, METH_VARARGS, "probe_dtz for a FEN string."},
    {"probe_root", (PyCFunction)(void (*)(void))tb_probe_root_py, METH_VARARGS | METH_KEYWORDS,
     "probe_root(cboard, has_repeated=False, use_rule50=True) -> "
     "list[(uci, dtz, rank, score)] | None. Fathom root_probe_dtz over every "
     "legal move, using the board's halfmove clock. [] at a mate/stalemate root."},
    {"probe_root_fen", (PyCFunction)(void (*)(void))tb_probe_root_fen_py,
     METH_VARARGS | METH_KEYWORDS, "probe_root for a FEN string."},
    {"fen_bitboards", tb_fen_bitboards_py, METH_VARARGS,
     "Parsed (white, black, kings, queens, rooks, bishops, knights, pawns, ep, "
     "turn, castling, rule50) of a FEN, as the prober sees it."},
    {"cboard_bitboards", tb_cboard_bitboards_py, METH_O,
     "The same tuple read from a CBoard."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef tb_moduledef = {
    PyModuleDef_HEAD_INIT,
    "_tb_ext",
    "Syzygy tablebase probing in C (vendored Fathom). See docs/tablebase_prober.md.",
    -1,
    tb_methods,
};

PyMODINIT_FUNC PyInit__tb_ext(void) {
    /* Fathom builds its attack tables on the first tb_init. Its is_valid()
     * (the illegal-position check every probe runs) reads them, so without
     * this an in-check position read as legal until the first init(). */
    df_tb_init("");
    PyObject *module = PyModule_Create(&tb_moduledef);
    if (!module) return NULL;
    if (PyModule_AddIntConstant(module, "MAX_ROOT_MOVES", DF_TB_MAX_ROOT_MOVES) < 0 ||
        PyModule_AddStringConstant(module, "FATHOM_COMMIT",
                                   "c9c6fef0dddc05d2e242c183acf5833149ab676d") < 0) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
