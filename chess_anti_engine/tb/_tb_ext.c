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
#include <limits.h>
#include <stdlib.h>
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
/* C-owned metadata shares the table lock and lifecycle. Never wait for the
 * GIL while holding its write lock: largest()/path() hold the GIL while
 * acquiring a read lock. */
static char *g_path = NULL;

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

static int df_lsb(uint64_t b) { return b ? __builtin_ctzll(b) : -1; }
static int df_msb(uint64_t b) { return b ? 63 - __builtin_clzll(b) : -1; }

/* Castling letters exactly as python-chess's Board._set_castling_fen reads them
 * (standard letters only), so the FEN path refuses precisely the positions
 * python-chess's probe refuses (it tests the raw castling_rights, unclean):
 *   q/Q: the leftmost back-rank rook if left of the king; nothing if the king
 *        is present and there is no rook; else the a-file corner;
 *   k/K: the rightmost back-rank rook if right of the king; else the h-file
 *        corner (also when there is no rook -- python-chess's msb(0) is -1).
 * Bits: 1/2/4/8 for a right on h1/a1/h8/a8; 16 for a right on any other square
 * (a non-corner rook, which CBoard -- corner bits only -- cannot represent).
 * Like python-chess 1.11's FEN_CASTLING_REGEX: upper-case letters first, at
 * most TWO of each case ("QQQ" is malformed there and here). Shredder file
 * letters are refused as malformed. */
static int castling_from_fen(const char *f, size_t n, DfTbPos *pos) {
    int seen_lower = 0, upper = 0, lower = 0;
    for (size_t i = 0; i < n; i++) {
        char c = f[i];
        int white = (c == 'K' || c == 'Q');
        if (!white && c != 'k' && c != 'q') return -1;
        if (white) {
            if (seen_lower || ++upper > 2) return -1;
        } else {
            seen_lower = 1;
            if (++lower > 2) return -1;
        }
        const uint64_t own = white ? pos->white : pos->black;
        const uint64_t backrank = white ? 0xFFULL : 0xFF00000000000000ULL;
        const uint64_t rooks = own & pos->rooks & backrank;
        const int king = df_msb(own & pos->kings);
        uint64_t add;
        if (c == 'q' || c == 'Q') {
            if (king >= 0 && df_lsb(rooks) < king) add = rooks & (0 - rooks);
            else add = backrank & 0x0101010101010101ULL;
        } else {
            const int rook = df_msb(rooks);
            if (king >= 0 && king < rook) add = 1ULL << rook;
            else add = backrank & 0x8080808080808080ULL;
        }
        if (add & (1ULL << 7)) pos->castling |= 1u;
        if (add & 1ULL) pos->castling |= 2u;
        if (add & (1ULL << 63)) pos->castling |= 4u;
        if (add & (1ULL << 56)) pos->castling |= 8u;
        if (add & ~((1ULL << 7) | 1ULL | (1ULL << 63) | (1ULL << 56))) pos->castling |= 16u;
    }
    return 0;
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
        const char *flags = p;
        while (*p && *p != ' ') p++;
        if (castling_from_fen(flags, (size_t)(p - flags), pos) < 0) goto bad;
    }
    if (*p != ' ') goto bad;
    p++;
    if (*p == '-') {
        p++;
    } else {
        int sq = fen_square(p);
        if (sq < 0) goto bad;
        /* Any square parses; the probe gate decides whether it is possible.
         * a1 (square 0) means "none", as it does for python-chess
         * (ep_square 0 is falsy there) and for CBoard's 0-based reader. */
        pos->ep = (unsigned)sq;
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
 * Fathom itself would skip a missing directory silently and report success.
 *
 * Each component is resolved with realpath(): Fathom opens tables lazily, on
 * the first probe of each material, so a relative component would be
 * re-resolved against whatever the cwd is THEN -- after a chdir every probe
 * would silently return None. Returns a new str of the absolute path list. */
static PyObject *absolute_components(const char *path) {
    PyObject *parts = PyList_New(0);
    if (!parts) return NULL;
    const char *start = path;
    for (;;) {
        const char *end = strchr(start, DF_TB_SEP);
        size_t len = end ? (size_t)(end - start) : strlen(start);
        char dir[PATH_MAX];
        char resolved[PATH_MAX];
        if (len == 0 || len >= sizeof dir) {
            PyErr_Format(PyExc_ValueError,
                         "Syzygy path has an empty or oversized component: %s", path);
            goto fail;
        }
        memcpy(dir, start, len);
        dir[len] = '\0';
        struct stat st;
        if (realpath(dir, resolved) == NULL || stat(resolved, &st) != 0 || !S_ISDIR(st.st_mode) ||
            access(resolved, R_OK | X_OK) != 0 || strchr(resolved, DF_TB_SEP) != NULL) {
            PyErr_Format(PyExc_ValueError,
                         "Syzygy path component is not a readable directory: %s", dir);
            goto fail;
        }
        PyObject *item = PyUnicode_FromString(resolved);
        if (!item || PyList_Append(parts, item) < 0) {
            Py_XDECREF(item);
            goto fail;
        }
        Py_DECREF(item);
        if (!end) break;
        start = end + 1;
    }
    {
        char sep[2] = {DF_TB_SEP, '\0'};
        PyObject *glue = PyUnicode_FromString(sep);
        if (!glue) goto fail;
        PyObject *joined = PyUnicode_Join(glue, parts);
        Py_DECREF(glue);
        Py_DECREF(parts);
        return joined;
    }
fail:
    Py_DECREF(parts);
    return NULL;
}

static PyObject *tb_init_py(PyObject *self, PyObject *args) {
    const char *given;
    if (!PyArg_ParseTuple(args, "s", &given)) return NULL;
    PyObject *keep = given[0] != '\0' ? absolute_components(given) : PyUnicode_FromString("");
    if (!keep) return NULL;
    const char *path = PyUnicode_AsUTF8(keep);
    if (!path) {
        Py_DECREF(keep);
        return NULL;
    }
    char *new_path = path[0] != '\0' ? strdup(path) : NULL;
    if (path[0] != '\0' && new_path == NULL) {
        Py_DECREF(keep);
        return PyErr_NoMemory();
    }
    int rc;
    unsigned largest;
    Py_BEGIN_ALLOW_THREADS
    pthread_rwlock_wrlock(&g_tables_lock);
    rc = df_tb_init(path);
    largest = df_tb_largest();
    free(g_path);
    g_path = (rc == 0 && largest > 0) ? new_path : NULL;
    if (g_path != new_path) free(new_path);
    pthread_rwlock_unlock(&g_tables_lock);
    Py_END_ALLOW_THREADS
    if (rc != 0) {
        PyErr_Format(PyExc_RuntimeError, "Fathom tb_init failed for %s", path);
        Py_DECREF(keep);
        return NULL;
    }
    if (path[0] != '\0' && largest == 0) {
        /* Accepted-and-ignored is this repo's signature defect: a path that
         * loads nothing is an error, not "no tablebases". The table set is
         * now empty, and path() says so. */
        PyErr_Format(PyExc_ValueError, "no Syzygy tables found under %s", path);
        Py_DECREF(keep);
        return NULL;
    }
    Py_DECREF(keep);
    return PyLong_FromUnsignedLong(largest);
}

static PyObject *tb_free_py(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    Py_BEGIN_ALLOW_THREADS
    pthread_rwlock_wrlock(&g_tables_lock);
    df_tb_free();
    free(g_path);
    g_path = NULL;
    pthread_rwlock_unlock(&g_tables_lock);
    Py_END_ALLOW_THREADS
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
    pthread_rwlock_rdlock(&g_tables_lock);
    PyObject *result = g_path ? PyUnicode_FromString(g_path) : Py_NewRef(Py_None);
    pthread_rwlock_unlock(&g_tables_lock);
    return result;
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
