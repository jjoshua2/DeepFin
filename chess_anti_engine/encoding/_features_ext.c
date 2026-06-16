/*
 * _features_ext.c — Python binding for the extra feature planes
 * (34 for v1, 63 for v2_threats, 67 for v3_checks, 69 for v3_xray).
 *
 * All bitboard tables, attack helpers, pin/pawn-structure logic, and the
 * compute_features_ext() driver live in _features_impl.h (shared with
 * _lc0_ext.c and _mcts_tree.c). This file only exposes the driver to Python.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdint.h>
#include <string.h>

#include "_features_impl.h"

static PyObject* py_compute_features(PyObject *self, PyObject *args) {
    PyArrayObject *pieces_us_arr, *pieces_them_arr;
    uint64_t occupied;
    int king_sq_us, king_sq_them, turn_white, ep_square;
    int n_extra = FEAT_EXTRA_V1;

    if (!PyArg_ParseTuple(args, "O!O!Kiipi|i",
            &PyArray_Type, &pieces_us_arr,
            &PyArray_Type, &pieces_them_arr,
            &occupied,
            &king_sq_us, &king_sq_them,
            &turn_white, &ep_square,
            &n_extra))
        return NULL;

    if (n_extra != FEAT_EXTRA_V1 && n_extra != FEAT_EXTRA_V2
            && n_extra != FEAT_EXTRA_V3_CHECKS && n_extra != FEAT_EXTRA_V3_XRAY) {
        PyErr_Format(PyExc_ValueError,
                     "n_extra must be %d (v1), %d (v2_threats), %d (v3_checks), "
                     "or %d (v3_xray), got %d",
                     FEAT_EXTRA_V1, FEAT_EXTRA_V2, FEAT_EXTRA_V3_CHECKS,
                     FEAT_EXTRA_V3_XRAY, n_extra);
        return NULL;
    }

    uint64_t *pus = (uint64_t *)PyArray_DATA(pieces_us_arr);
    uint64_t *pthem = (uint64_t *)PyArray_DATA(pieces_them_arr);

    uint64_t us_pieces[6], them_pieces[6];
    for (int i = 0; i < 6; i++) {
        us_pieces[i] = pus[i];
        them_pieces[i] = pthem[i];
    }

    npy_intp dims[3] = {n_extra, 8, 8};
    PyObject *out_arr = PyArray_ZEROS(3, dims, NPY_FLOAT32, 0);
    if (!out_arr) return NULL;

    compute_features_ext(
        us_pieces, them_pieces, occupied,
        king_sq_us, king_sq_them, turn_white, ep_square, n_extra,
        (float *)PyArray_DATA((PyArrayObject *)out_arr));

    return out_arr;
}

static PyObject* py_compute_relations(PyObject *self, PyObject *args) {
    PyArrayObject *pieces_us_arr, *pieces_them_arr;
    uint64_t occupied;
    int king_sq_us, king_sq_them, turn_white;

    if (!PyArg_ParseTuple(args, "O!O!Kiip",
            &PyArray_Type, &pieces_us_arr,
            &PyArray_Type, &pieces_them_arr,
            &occupied,
            &king_sq_us, &king_sq_them,
            &turn_white))
        return NULL;

    uint64_t *pus = (uint64_t *)PyArray_DATA(pieces_us_arr);
    uint64_t *pthem = (uint64_t *)PyArray_DATA(pieces_them_arr);
    uint64_t us_pieces[6], them_pieces[6];
    for (int i = 0; i < 6; i++) {
        us_pieces[i] = pus[i];
        them_pieces[i] = pthem[i];
    }

    npy_intp dims[3] = {FEAT_RELATION_COUNT, 64, 64};
    PyArrayObject *out_arr = (PyArrayObject *)PyArray_SimpleNew(3, dims, NPY_UINT8);
    if (!out_arr) return NULL;

    compute_relations(us_pieces, them_pieces, occupied,
                      king_sq_us, king_sq_them, turn_white,
                      (uint8_t *)PyArray_DATA(out_arr));
    return (PyObject *)out_arr;
}

static PyMethodDef methods[] = {
    {"compute_relation_matrices", py_compute_relations, METH_VARARGS,
     "Compute dynamic board-relation matrices from bitboard data.\n\n"
     "Args: pieces_us(uint64[6]), pieces_them(uint64[6]), occupied(uint64),\n"
     "      king_sq_us(int), king_sq_them(int), turn_white(bool)\n"
     "Returns: ndarray (5, 64, 64) uint8"},
    {"compute_extra_features", py_compute_features, METH_VARARGS,
     "Compute extra feature planes from bitboard data.\n\n"
     "Args: pieces_us(uint64[6]), pieces_them(uint64[6]), occupied(uint64),\n"
     "      king_sq_us(int), king_sq_them(int), turn_white(bool), ep_square(int)\n"
     "      [, n_extra(int) = 34 (v1), 63 (v2_threats), 67 (v3_checks),\n"
     "         or 69 (v3_xray)]\n"
     "Returns: ndarray (n_extra, 8, 8) float32"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_features_ext",
    "C-accelerated feature plane computation",
    -1,
    methods,
};

PyMODINIT_FUNC PyInit__features_ext(void) {
    PyObject *m = PyModule_Create(&moduledef);
    if (!m) return NULL;
    import_array();
    init_tables_features();
    return m;
}
