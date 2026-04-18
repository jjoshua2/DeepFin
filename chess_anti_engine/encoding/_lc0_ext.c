/*
 * _lc0_ext.c — C-accelerated LC0 plane encoding + legal move index generation.
 *
 * encode_piece_planes(): converts bitboards -> 96 oriented piece planes (~20x vs Python)
 * legal_move_policy_indices(): generates legal move policy indices from bitboards (~60x vs Python)
 *
 * Both avoid python-chess object overhead by working directly with uint64 bitboards.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

/* Pure-C CBoard implementation (bitboard utilities, attack tables, move gen, CBoard) */
#include "_cboard_impl.h"

/* Shared feature computation (pure C, no Python callbacks) */
#include "_features_impl.h"

/* Read a uint64 attribute from a python-chess object. On error, returns 0
 * and (if err != NULL) sets *err = 1; the Python exception is left in place.
 * Pass err = NULL for silent fallback (used when reading history snapshots
 * where missing attrs should just zero out). */
static inline uint64_t py_attr_u64(PyObject *obj, const char *attr, int *err) {
    PyObject *v = PyObject_GetAttrString(obj, attr);
    if (!v) { if (err) *err = 1; return 0; }
    uint64_t r = PyLong_AsUnsignedLongLong(v);
    Py_DECREF(v);
    if (err && r == (uint64_t)-1 && PyErr_Occurred()) *err = 1;
    return r;
}


/* ================================================================
 * encode_piece_planes: bitboards -> (n_steps*12, 8, 8) float32
 * ================================================================ */

/*
 * encode_piece_planes(bitboards, turns, n_steps)
 *
 * bitboards: uint64 array of shape (n_steps * 12,) -- piece bitboards for each history step
 *   Order per step: us_pawns, us_knights, us_bishops, us_rooks, us_queens, us_kings,
 *                   them_pawns, them_knights, them_bishops, them_rooks, them_queens, them_kings
 * turns: int32 array of shape (n_steps,) -- 1=WHITE, 0=BLACK for each step
 * n_steps: number of history steps (1..8)
 *
 * Returns: float32 array of shape (n_steps*12, 8, 8)
 */
static PyObject* py_encode_piece_planes(PyObject *self, PyObject *args) {
    PyArrayObject *bbs_arr, *turns_arr;
    int n_steps;

    if (!PyArg_ParseTuple(args, "O!O!i",
                          &PyArray_Type, &bbs_arr,
                          &PyArray_Type, &turns_arr,
                          &n_steps))
        return NULL;

    if (n_steps < 0 || n_steps > 8) {
        PyErr_SetString(PyExc_ValueError, "n_steps must be 0..8");
        return NULL;
    }

    uint64_t *bbs = (uint64_t*)PyArray_DATA(bbs_arr);
    int32_t *turns = (int32_t*)PyArray_DATA(turns_arr);

    npy_intp dims[3] = {n_steps * 12, 8, 8};
    PyArrayObject *out = (PyArrayObject*)PyArray_ZEROS(3, dims, NPY_FLOAT32, 0);
    if (!out) return NULL;

    float *out_data = (float*)PyArray_DATA(out);

    for (int step = 0; step < n_steps; step++) {
        int is_white = turns[step];
        uint64_t *step_bbs = bbs + step * 12;
        float *step_out = out_data + step * 12 * 64;

        for (int p = 0; p < 12; p++) {
            if (is_white)
                bitboard_to_plane_white(step_bbs[p], step_out + p * 64);
            else
                bitboard_to_plane_black(step_bbs[p], step_out + p * 64);
        }
    }

    return (PyObject*)out;
}


/* ================================================================
 * legal_move_policy_indices: generate legal moves -> policy indices
 * ================================================================ */

/*
 * legal_move_policy_indices(
 *     us_pawns, us_knights, us_bishops, us_rooks, us_queens, us_kings,
 *     them_pawns, them_knights, them_bishops, them_rooks, them_queens, them_kings,
 *     turn, castling_us_k, castling_us_q, castling_them_k, castling_them_q, ep_square
 * )
 * Returns: sorted int32 array of policy indices.
 */
static PyObject* py_legal_move_policy_indices(PyObject *self, PyObject *args) {
    uint64_t us_p, us_n, us_b, us_r, us_q, us_k;
    uint64_t th_p, th_n, th_b, th_r, th_q, th_k;
    int turn, castle_uk, castle_uq, castle_tk, castle_tq, ep_sq;

    if (!PyArg_ParseTuple(args, "KKKKKKKKKKKKiiiiii",
                          &us_p, &us_n, &us_b, &us_r, &us_q, &us_k,
                          &th_p, &th_n, &th_b, &th_r, &th_q, &th_k,
                          &turn, &castle_uk, &castle_uq, &castle_tk, &castle_tq, &ep_sq))
        return NULL;

    BoardState bs;
    bs.us_pawns = us_p; bs.us_knights = us_n; bs.us_bishops = us_b;
    bs.us_rooks = us_r; bs.us_queens = us_q; bs.us_kings = us_k;
    bs.them_pawns = th_p; bs.them_knights = th_n; bs.them_bishops = th_b;
    bs.them_rooks = th_r; bs.them_queens = th_q; bs.them_kings = th_k;
    bs.us_occ = us_p | us_n | us_b | us_r | us_q | us_k;
    bs.them_occ = th_p | th_n | th_b | th_r | th_q | th_k;
    bs.all_occ = bs.us_occ | bs.them_occ;
    bs.turn = turn;
    bs.us_castle_k = castle_uk;
    bs.us_castle_q = castle_uq;
    bs.them_castle_k = castle_tk;
    bs.them_castle_q = castle_tq;
    bs.ep_square = ep_sq;

    /* Find king square */
    if (bs.us_kings == 0) {
        PyErr_SetString(PyExc_ValueError, "No king found");
        return NULL;
    }
    bs.king_sq = lsb64(bs.us_kings);

    /* Max possible legal moves in chess is ~218 */
    int indices[256];
    int count = generate_legal_move_indices_sorted(&bs, indices);

    /* Build numpy array */
    npy_intp dims[1] = {count};
    PyArrayObject *arr = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT32);
    if (!arr) return NULL;
    memcpy(PyArray_DATA(arr), indices, count * sizeof(int));

    return (PyObject*)arr;
}


/* ================================================================
 * CBoard Python encoding wrappers (require NumPy)
 * ================================================================ */

static PyObject* cboard_encode_146(const CBoard *b) {
    npy_intp dims[3] = {146, 8, 8};
    PyArrayObject *arr = (PyArrayObject*)PyArray_ZEROS(3, dims, NPY_FLOAT32, 0);
    if (!arr) return NULL;
    float *out = (float*)PyArray_DATA(arr);

    cboard_fill_lc0_112(b, out);
    cboard_compute_features_34(b, out + 112 * 64);

    return (PyObject*)arr;
}

/* Full LC0 112-plane encoding for MCTS boards (no history).
 * Planes 0-11: piece planes, 12-95: zeros (no history),
 * 96-99: castling, 100: EP, 101: turn, 102: rule50,
 * 103-110: repetitions (all 0), 111: ones. */
static PyObject* cboard_encode_112(const CBoard *b) {
    npy_intp dims[3] = {112, 8, 8};
    PyArrayObject *arr = (PyArrayObject*)PyArray_ZEROS(3, dims, NPY_FLOAT32, 0);
    if (!arr) return NULL;
    float *out = (float*)PyArray_DATA(arr);

    cboard_fill_lc0_112(b, out);

    return (PyObject*)arr;
}


/* ================================================================
 * PyCBoard: Python type wrapping CBoard
 * ================================================================ */

typedef struct {
    PyObject_HEAD
    CBoard board;
} PyCBoard;

static PyTypeObject PyCBoardType;  /* forward declaration */

/* Constructor: CBoard.from_board(chess_board) */
static PyObject* PyCBoard_from_board(PyTypeObject *type, PyObject *args) {
    PyObject *py_board;
    if (!PyArg_ParseTuple(args, "O", &py_board)) return NULL;

    PyCBoard *self = (PyCBoard*)type->tp_alloc(type, 0);
    if (!self) return NULL;

    CBoard *b = &self->board;
    memset(b, 0, sizeof(CBoard));

    /* Read bitboards from python-chess board. On any attr-miss or conversion
     * failure, DECREF self and propagate the Python exception. */
    PyObject *val;  /* reused by the per-attr blocks further below */
    #define READ_UINT64(attr) ({ \
        int _err = 0; \
        uint64_t _v = py_attr_u64(py_board, attr, &_err); \
        if (_err) { Py_DECREF(self); return NULL; } \
        _v; })

    b->bb[PAWN]   = READ_UINT64("pawns");
    b->bb[KNIGHT] = READ_UINT64("knights");
    b->bb[BISHOP] = READ_UINT64("bishops");
    b->bb[ROOK]   = READ_UINT64("rooks");
    b->bb[QUEEN]  = READ_UINT64("queens");
    b->bb[KING]   = READ_UINT64("kings");

    /* occupied_co is a list-like indexed by color bool */
    PyObject *occ_co = PyObject_GetAttrString(py_board, "occupied_co");
    if (!occ_co) { Py_DECREF(self); return NULL; }
    PyObject *occ_w = PyObject_GetItem(occ_co, PyBool_FromLong(1));  /* WHITE=True */
    PyObject *occ_b = PyObject_GetItem(occ_co, PyBool_FromLong(0));  /* BLACK=False */
    Py_DECREF(occ_co);
    if (!occ_w || !occ_b) {
        Py_XDECREF(occ_w); Py_XDECREF(occ_b);
        Py_DECREF(self); return NULL;
    }
    b->occ[WHITE_C] = (uint64_t)PyLong_AsUnsignedLongLong(occ_w);
    b->occ[BLACK_C] = (uint64_t)PyLong_AsUnsignedLongLong(occ_b);
    Py_DECREF(occ_w); Py_DECREF(occ_b);

    /* Turn: True=WHITE, False=BLACK */
    val = PyObject_GetAttrString(py_board, "turn");
    if (!val) { Py_DECREF(self); return NULL; }
    b->turn = PyObject_IsTrue(val) ? WHITE_C : BLACK_C;
    Py_DECREF(val);

    /* Castling rights: read bitmask directly instead of 4 method calls */
    b->castling = 0;
    val = PyObject_GetAttrString(py_board, "castling_rights");
    if (!val) { Py_DECREF(self); return NULL; }
    {
        uint64_t cr = PyLong_AsUnsignedLongLong(val);
        Py_DECREF(val);
        if (cr == (uint64_t)-1 && PyErr_Occurred()) { Py_DECREF(self); return NULL; }
        if (cr & (1ULL << 7))  b->castling |= WK_CASTLE;  /* H1 */
        if (cr & (1ULL << 0))  b->castling |= WQ_CASTLE;  /* A1 */
        if (cr & (1ULL << 63)) b->castling |= BK_CASTLE;  /* H8 */
        if (cr & (1ULL << 56)) b->castling |= BQ_CASTLE;  /* A8 */
    }

    /* EP square */
    val = PyObject_GetAttrString(py_board, "ep_square");
    if (!val) { Py_DECREF(self); return NULL; }
    if (val == Py_None) b->ep_square = -1;
    else b->ep_square = (int8_t)PyLong_AsLong(val);
    Py_DECREF(val);

    /* Halfmove clock */
    val = PyObject_GetAttrString(py_board, "halfmove_clock");
    if (!val) { Py_DECREF(self); return NULL; }
    int hmc = (int)PyLong_AsLong(val);
    b->halfmove_clock = (uint8_t)(hmc > 255 ? 255 : hmc);
    Py_DECREF(val);

    /* Fullmove number -> ply (matches python-chess Board.ply()) */
    val = PyObject_GetAttrString(py_board, "fullmove_number");
    if (!val) { Py_DECREF(self); return NULL; }
    {
        int fmn = (int)PyLong_AsLong(val);
        Py_DECREF(val);
        b->ply = (uint16_t)((fmn - 1) * 2 + (b->turn == BLACK_C ? 1 : 0));
    }

    /* --- Compute Zobrist hash --- */
    b->hash = cboard_compute_hash(b);

    /* --- Extract history from python-chess board._stack --- */
    b->hist_len = 0;
    b->hist_head = 0;
    b->hash_stack_len = 0;
    PyObject *stack = PyObject_GetAttrString(py_board, "_stack");
    if (stack && PyList_Check(stack)) {
        Py_ssize_t stack_len = PyList_Size(stack);

        /* History: last 7 entries of _stack are the 7 previous positions.
         * Populate the circular buffer oldest-first so that hist_head ends
         * up pointing past the most-recent entry. */
        int n_hist = (int)stack_len < CBOARD_HISTORY_MAX ? (int)stack_len : CBOARD_HISTORY_MAX;
        for (int i = 0; i < n_hist; i++) {
            /* i=0 -> oldest of the kept entries, i=n_hist-1 -> most recent */
            int hi = n_hist - 1 - i;  /* hi into _stack (hi=0 = most recent) */
            PyObject *s = PyList_GetItem(stack, stack_len - 1 - hi); /* borrowed */
            if (!s) break;

            uint64_t s_pawns   = py_attr_u64(s, "pawns",      NULL);
            uint64_t s_knights = py_attr_u64(s, "knights",    NULL);
            uint64_t s_bishops = py_attr_u64(s, "bishops",    NULL);
            uint64_t s_rooks   = py_attr_u64(s, "rooks",      NULL);
            uint64_t s_queens  = py_attr_u64(s, "queens",     NULL);
            uint64_t s_kings   = py_attr_u64(s, "kings",      NULL);
            uint64_t s_occ_w   = py_attr_u64(s, "occupied_w", NULL);
            uint64_t s_occ_b   = py_attr_u64(s, "occupied_b", NULL);

            PyObject *s_turn_obj = PyObject_GetAttrString(s, "turn");
            int s_turn = (s_turn_obj && PyObject_IsTrue(s_turn_obj)) ? WHITE_C : BLACK_C;
            Py_XDECREF(s_turn_obj);

            int slot = i;  /* write oldest at slot 0, newest at slot n_hist-1 */
            b->hist_bb[slot][PAWN]   = s_pawns;
            b->hist_bb[slot][KNIGHT] = s_knights;
            b->hist_bb[slot][BISHOP] = s_bishops;
            b->hist_bb[slot][ROOK]   = s_rooks;
            b->hist_bb[slot][QUEEN]  = s_queens;
            b->hist_bb[slot][KING]   = s_kings;
            b->hist_occ[slot][WHITE_C] = s_occ_w;
            b->hist_occ[slot][BLACK_C] = s_occ_b;
            b->hist_turn[slot] = (int8_t)s_turn;
            b->hist_len++;

        }
        b->hist_head = n_hist % CBOARD_HISTORY_MAX;

        /* Hash stack: compute hashes for all positions since last irreversible
         * move. Walk _stack backwards from most recent. */
        for (Py_ssize_t si = stack_len - 1; si >= 0; si--) {
            PyObject *s = PyList_GetItem(stack, si); /* borrowed */
            if (!s) break;

            /* Read piece data and castling_rights to compute the hash; stop
             * when we've walked past the halfmove_clock limit below. */
            uint64_t h_bb[6], h_occ[2];
            h_bb[PAWN]   = py_attr_u64(s, "pawns",      NULL);
            h_bb[KNIGHT] = py_attr_u64(s, "knights",    NULL);
            h_bb[BISHOP] = py_attr_u64(s, "bishops",    NULL);
            h_bb[ROOK]   = py_attr_u64(s, "rooks",      NULL);
            h_bb[QUEEN]  = py_attr_u64(s, "queens",     NULL);
            h_bb[KING]   = py_attr_u64(s, "kings",      NULL);
            h_occ[WHITE_C] = py_attr_u64(s, "occupied_w", NULL);
            h_occ[BLACK_C] = py_attr_u64(s, "occupied_b", NULL);

            PyObject *t_obj = PyObject_GetAttrString(s, "turn");
            int h_turn = (t_obj && PyObject_IsTrue(t_obj)) ? WHITE_C : BLACK_C;
            Py_XDECREF(t_obj);

            /* Read castling_rights (int) */
            PyObject *cr_obj = PyObject_GetAttrString(s, "castling_rights");
            uint8_t h_castling = 0;
            if (cr_obj) {
                unsigned long cr = PyLong_AsUnsignedLong(cr_obj);
                Py_DECREF(cr_obj);
                /* python-chess castling_rights uses chess.BB_* bitmask:
                 * BB_H1=128, BB_A1=1, BB_H8=..., BB_A8=...
                 * We need to convert to our WK/WQ/BK/BQ bits.
                 * chess.BB_H1=128 -> WK, chess.BB_A1=1 -> WQ,
                 * chess.BB_H8=... -> BK, chess.BB_A8=... -> BQ */
                if (cr & (1ULL << 7))  h_castling |= WK_CASTLE;  /* H1 */
                if (cr & (1ULL << 0))  h_castling |= WQ_CASTLE;  /* A1 */
                if (cr & (1ULL << 63)) h_castling |= BK_CASTLE;  /* H8 */
                if (cr & (1ULL << 56)) h_castling |= BQ_CASTLE;  /* A8 */
            }

            /* EP square -- not stored directly on _BoardState, skip for hash
             * (matches _check_repetitions which omits EP) */
            uint64_t h_hash = cboard_hist_hash(h_bb, h_occ, h_turn, h_castling, -1);

            if (b->hash_stack_len < CBOARD_HASH_STACK_MAX)
                b->hash_stack[b->hash_stack_len++] = h_hash;

            /* Stop at irreversible boundary: if this state's halfmove was 0,
             * no repetition can cross it */
            /* We approximate: stop after reading halfmove_clock entries
             * (b->halfmove_clock tells us how many reversible plies back) */
            if ((int)(stack_len - 1 - si) >= (int)b->halfmove_clock) break;
        }
    }
    Py_XDECREF(stack);

    return (PyObject*)self;
}

/* Constructor: CBoard.from_raw(pawns, knights, bishops, rooks, queens, kings,
 *                               occ_white, occ_black, turn, castling,
 *                               ep_square, halfmove_clock)
 * Accepts pre-extracted integers -- no Python attribute access needed. */
static PyObject* PyCBoard_from_raw(PyTypeObject *type, PyObject *args) {
    uint64_t pawns, knights, bishops, rooks, queens, kings;
    uint64_t occ_w, occ_b;
    int turn_int, castling_int, ep_sq, hmc;

    if (!PyArg_ParseTuple(args, "KKKKKKKKiiii",
        &pawns, &knights, &bishops, &rooks, &queens, &kings,
        &occ_w, &occ_b, &turn_int, &castling_int, &ep_sq, &hmc))
        return NULL;

    PyCBoard *self = (PyCBoard*)type->tp_alloc(type, 0);
    if (!self) return NULL;
    CBoard *b = &self->board;
    memset(b, 0, sizeof(CBoard));

    b->bb[PAWN]   = pawns;
    b->bb[KNIGHT] = knights;
    b->bb[BISHOP] = bishops;
    b->bb[ROOK]   = rooks;
    b->bb[QUEEN]  = queens;
    b->bb[KING]   = kings;
    b->occ[WHITE_C] = occ_w;
    b->occ[BLACK_C] = occ_b;
    b->turn = turn_int ? WHITE_C : BLACK_C;
    b->castling = (uint8_t)castling_int;
    b->ep_square = (int8_t)ep_sq;
    b->halfmove_clock = (uint8_t)(hmc > 255 ? 255 : hmc);

    b->hash = cboard_compute_hash(b);

    b->hist_len = 0;
    b->hist_head = 0;
    b->hash_stack_len = 0;

    return (PyObject*)self;
}

/* copy() -> new CBoard */
static PyObject* PyCBoard_copy(PyCBoard *self, PyObject *Py_UNUSED(args)) {
    PyCBoard *cp = (PyCBoard*)PyCBoardType.tp_alloc(&PyCBoardType, 0);
    if (!cp) return NULL;
    memcpy(&cp->board, &self->board, sizeof(CBoard));
    return (PyObject*)cp;
}

/* push_index(policy_index) -- in place */
static PyObject* PyCBoard_push_index(PyCBoard *self, PyObject *args) {
    int policy_index;
    if (!PyArg_ParseTuple(args, "i", &policy_index)) return NULL;
    if (policy_index < 0 || policy_index >= 4672) {
        PyErr_SetString(PyExc_ValueError, "policy_index out of range");
        return NULL;
    }
    cboard_push_index(&self->board, policy_index);
    Py_RETURN_NONE;
}

/* is_game_over() -> bool */
static PyObject* PyCBoard_is_game_over(PyCBoard *self, PyObject *Py_UNUSED(args)) {
    return PyBool_FromLong(cboard_is_game_over(&self->board));
}

/* terminal_value() -> float */
static PyObject* PyCBoard_terminal_value(PyCBoard *self, PyObject *Py_UNUSED(args)) {
    return PyFloat_FromDouble((double)cboard_terminal_value(&self->board));
}

/* fen() -> str */
static PyObject* PyCBoard_fen(PyCBoard *self, PyObject *Py_UNUSED(args)) {
    char buf[128];
    cboard_to_fen(&self->board, buf, sizeof(buf));
    return PyUnicode_FromString(buf);
}

/* result() -> str */
static PyObject* PyCBoard_result(PyCBoard *self, PyObject *Py_UNUSED(args)) {
    char buf[16];
    cboard_result(&self->board, buf);
    return PyUnicode_FromString(buf);
}

/* is_checkmate() -> bool */
static PyObject* PyCBoard_is_checkmate(PyCBoard *self, PyObject *Py_UNUSED(args)) {
    return PyBool_FromLong(cboard_is_checkmate(&self->board));
}

/* is_stalemate() -> bool */
static PyObject* PyCBoard_is_stalemate(PyCBoard *self, PyObject *Py_UNUSED(args)) {
    return PyBool_FromLong(cboard_is_stalemate(&self->board));
}

/* legal_move_indices() -> numpy int32 array */
static PyObject* PyCBoard_legal_move_indices(PyCBoard *self, PyObject *Py_UNUSED(args)) {
    int indices[256];
    int count = cboard_legal_move_indices(&self->board, indices, /*sorted=*/1);
    npy_intp dims[1] = {count};
    PyArrayObject *arr = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT32);
    if (!arr) return NULL;
    if (count > 0) memcpy(PyArray_DATA(arr), indices, count * sizeof(int));
    return (PyObject*)arr;
}

/* encode_planes() -> numpy (112, 8, 8) float32 */
static PyObject* PyCBoard_encode_planes(PyCBoard *self, PyObject *Py_UNUSED(args)) {
    return cboard_encode_112(&self->board);
}

/* encode_146() -> numpy (146, 8, 8) float32 */
static PyObject* PyCBoard_encode_146(PyCBoard *self, PyObject *Py_UNUSED(args)) {
    return cboard_encode_146(&self->board);
}

/* encode_146_and_legal() -> (numpy(146,8,8), numpy(N,) int32)
 * Fused encode + legal move generation: one Python->C call instead of two,
 * avoids redundant cboard_to_boardstate construction. */
static PyObject* PyCBoard_encode_146_and_legal(PyCBoard *self, PyObject *Py_UNUSED(args)) {
    PyObject *enc = cboard_encode_146(&self->board);
    if (!enc) return NULL;
    int indices[256];
    int count = cboard_legal_move_indices(&self->board, indices, /*sorted=*/1);
    npy_intp dims[1] = {count};
    PyArrayObject *legal = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT32);
    if (!legal) { Py_DECREF(enc); return NULL; }
    if (count > 0)
        memcpy(PyArray_DATA(legal), indices, count * sizeof(int));
    PyObject *result = PyTuple_Pack(2, enc, (PyObject*)legal);
    Py_DECREF(enc);
    Py_DECREF(legal);
    return result;
}

/* Properties: turn, ep_square, halfmove_clock, castling, and bitboards */
static PyObject* PyCBoard_get_turn(PyCBoard *self, void *closure) {
    return PyBool_FromLong(self->board.turn == WHITE_C);
}
static PyObject* PyCBoard_get_ep_square(PyCBoard *self, void *closure) {
    if (self->board.ep_square < 0) Py_RETURN_NONE;
    return PyLong_FromLong(self->board.ep_square);
}
static PyObject* PyCBoard_get_halfmove_clock(PyCBoard *self, void *closure) {
    return PyLong_FromLong(self->board.halfmove_clock);
}
static PyObject* PyCBoard_get_castling(PyCBoard *self, void *closure) {
    return PyLong_FromLong(self->board.castling);
}
static PyObject* PyCBoard_get_pawns(PyCBoard *self, void *closure) {
    return PyLong_FromUnsignedLongLong(self->board.bb[PAWN]);
}
static PyObject* PyCBoard_get_knights(PyCBoard *self, void *closure) {
    return PyLong_FromUnsignedLongLong(self->board.bb[KNIGHT]);
}
static PyObject* PyCBoard_get_bishops(PyCBoard *self, void *closure) {
    return PyLong_FromUnsignedLongLong(self->board.bb[BISHOP]);
}
static PyObject* PyCBoard_get_rooks(PyCBoard *self, void *closure) {
    return PyLong_FromUnsignedLongLong(self->board.bb[ROOK]);
}
static PyObject* PyCBoard_get_queens(PyCBoard *self, void *closure) {
    return PyLong_FromUnsignedLongLong(self->board.bb[QUEEN]);
}
static PyObject* PyCBoard_get_kings(PyCBoard *self, void *closure) {
    return PyLong_FromUnsignedLongLong(self->board.bb[KING]);
}
static PyObject* PyCBoard_get_occ_white(PyCBoard *self, void *closure) {
    return PyLong_FromUnsignedLongLong(self->board.occ[WHITE_C]);
}
static PyObject* PyCBoard_get_occ_black(PyCBoard *self, void *closure) {
    return PyLong_FromUnsignedLongLong(self->board.occ[BLACK_C]);
}
static PyObject* PyCBoard_get_zobrist_hash(PyCBoard *self, void *closure) {
    return PyLong_FromUnsignedLongLong(self->board.hash);
}
static PyObject* PyCBoard_get_hist_len(PyCBoard *self, void *closure) {
    return PyLong_FromLong(self->board.hist_len);
}
static PyObject* PyCBoard_get_hash_stack_len(PyCBoard *self, void *closure) {
    return PyLong_FromLong(self->board.hash_stack_len);
}
static PyObject* PyCBoard_get_ply(PyCBoard *self, void *closure) {
    return PyLong_FromLong(self->board.ply);
}

static PyMethodDef PyCBoard_methods[] = {
    {"from_board", (PyCFunction)PyCBoard_from_board, METH_VARARGS | METH_CLASS,
     "Create CBoard from a python-chess Board"},
    {"from_raw", (PyCFunction)PyCBoard_from_raw, METH_VARARGS | METH_CLASS,
     "Create CBoard from raw integer arguments (no Python attribute access)"},
    {"copy", (PyCFunction)PyCBoard_copy, METH_NOARGS, "Copy board"},
    {"push_index", (PyCFunction)PyCBoard_push_index, METH_VARARGS,
     "Apply move by policy index (in-place)"},
    {"is_game_over", (PyCFunction)PyCBoard_is_game_over, METH_NOARGS,
     "Check if game is over"},
    {"terminal_value", (PyCFunction)PyCBoard_terminal_value, METH_NOARGS,
     "Get terminal value (-1, 0)"},
    {"legal_move_indices", (PyCFunction)PyCBoard_legal_move_indices, METH_NOARGS,
     "Get sorted legal move policy indices"},
    {"fen", (PyCFunction)PyCBoard_fen, METH_NOARGS, "FEN string"},
    {"result", (PyCFunction)PyCBoard_result, METH_NOARGS, "Game result string"},
    {"is_checkmate", (PyCFunction)PyCBoard_is_checkmate, METH_NOARGS, "Check if checkmate"},
    {"is_stalemate", (PyCFunction)PyCBoard_is_stalemate, METH_NOARGS, "Check if stalemate"},
    {"encode_planes", (PyCFunction)PyCBoard_encode_planes, METH_NOARGS,
     "Encode as (112, 8, 8) float32 LC0 planes"},
    {"encode_146", (PyCFunction)PyCBoard_encode_146, METH_NOARGS,
     "Encode as (146, 8, 8) float32 LC0+feature planes"},
    {"encode_146_and_legal", (PyCFunction)PyCBoard_encode_146_and_legal, METH_NOARGS,
     "Encode (146,8,8) and return legal move indices in one call"},
    {NULL}
};

static PyGetSetDef PyCBoard_getset[] = {
    {"turn", (getter)PyCBoard_get_turn, NULL, "Side to move (True=WHITE)", NULL},
    {"ep_square", (getter)PyCBoard_get_ep_square, NULL, "En passant square or None", NULL},
    {"halfmove_clock", (getter)PyCBoard_get_halfmove_clock, NULL, "Halfmove clock", NULL},
    {"castling", (getter)PyCBoard_get_castling, NULL, "Castling rights (4-bit)", NULL},
    {"pawns", (getter)PyCBoard_get_pawns, NULL, "Pawns bitboard", NULL},
    {"knights", (getter)PyCBoard_get_knights, NULL, "Knights bitboard", NULL},
    {"bishops", (getter)PyCBoard_get_bishops, NULL, "Bishops bitboard", NULL},
    {"rooks", (getter)PyCBoard_get_rooks, NULL, "Rooks bitboard", NULL},
    {"queens", (getter)PyCBoard_get_queens, NULL, "Queens bitboard", NULL},
    {"kings", (getter)PyCBoard_get_kings, NULL, "Kings bitboard", NULL},
    {"occ_white", (getter)PyCBoard_get_occ_white, NULL, "White occupancy", NULL},
    {"occ_black", (getter)PyCBoard_get_occ_black, NULL, "Black occupancy", NULL},
    {"zobrist_hash", (getter)PyCBoard_get_zobrist_hash, NULL, "Zobrist hash of current position", NULL},
    {"hist_len", (getter)PyCBoard_get_hist_len, NULL, "Number of history entries", NULL},
    {"hash_stack_len", (getter)PyCBoard_get_hash_stack_len, NULL, "Number of hash stack entries", NULL},
    {"ply", (getter)PyCBoard_get_ply, NULL, "Half-moves from game start", NULL},
    {NULL}
};

static PyTypeObject PyCBoardType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_lc0_ext.CBoard",
    .tp_basicsize = sizeof(PyCBoard),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Lightweight C chess board for MCTS",
    .tp_methods = PyCBoard_methods,
    .tp_getset = PyCBoard_getset,
    .tp_new = PyType_GenericNew,
};


/* ================================================================
 * Module definition
 * ================================================================ */

static PyMethodDef methods[] = {
    {"encode_piece_planes", py_encode_piece_planes, METH_VARARGS,
     "Convert bitboards to oriented piece planes. "
     "encode_piece_planes(bitboards_u64, turns_i32, n_steps) -> float32(n_steps*12, 8, 8)"},
    {"legal_move_policy_indices", py_legal_move_policy_indices, METH_VARARGS,
     "Generate sorted legal move policy indices from bitboard state. "
     "legal_move_policy_indices(us_p, us_n, us_b, us_r, us_q, us_k, "
     "th_p, th_n, th_b, th_r, th_q, th_k, turn, ck, cq, tk, tq, ep) -> int32[]"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_lc0_ext", NULL, -1, methods
};

PyMODINIT_FUNC PyInit__lc0_ext(void) {
    import_array();
    cboard_init_all();
    init_tables_features();

    PyObject *m = PyModule_Create(&moduledef);
    if (!m) return NULL;

    if (PyType_Ready(&PyCBoardType) < 0) return NULL;
    Py_INCREF(&PyCBoardType);
    PyModule_AddObject(m, "CBoard", (PyObject*)&PyCBoardType);

    return m;
}
