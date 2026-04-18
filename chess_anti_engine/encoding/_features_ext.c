/*
 * _features_ext.c — Python binding for 34 extra feature planes.
 *
 * All bitboard tables, attack helpers, pin/pawn-structure logic, and the
 * compute_features_34() driver live in _features_impl.h, which is shared
 * with _lc0_ext.c and _mcts_tree.c. This file just exposes the driver to
 * Python.
 *
 * Layout (34 planes, 8x8 float32):
 *   [0:10]  king safety
 *   [10:16] pins / x-rays / discovered attacks
 *   [16:24] pawn structure
 *   [24:30] mobility
 *   [30:34] outpost / space
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

    if (!PyArg_ParseTuple(args, "O!O!Kiipi",
            &PyArray_Type, &pieces_us_arr,
            &PyArray_Type, &pieces_them_arr,
            &occupied,
            &king_sq_us, &king_sq_them,
            &turn_white, &ep_square))
        return NULL;

    uint64_t *pus = (uint64_t *)PyArray_DATA(pieces_us_arr);
    uint64_t *pthem = (uint64_t *)PyArray_DATA(pieces_them_arr);

    uint64_t us_pieces[6], them_pieces[6];
    for (int i = 0; i < 6; i++) {
        us_pieces[i] = pus[i];
        them_pieces[i] = pthem[i];
    }

    npy_intp dims[3] = {34, 8, 8};
    PyObject *out_arr = PyArray_ZEROS(3, dims, NPY_FLOAT32, 0);
    if (!out_arr) return NULL;

    compute_features_34(
        us_pieces, them_pieces, occupied,
        king_sq_us, king_sq_them, turn_white, ep_square,
        (float *)PyArray_DATA((PyArrayObject *)out_arr));

    return out_arr;
}

static PyMethodDef methods[] = {
    {"compute_extra_features", py_compute_features, METH_VARARGS,
     "Compute 34 extra feature planes from bitboard data.\n\n"
     "Args: pieces_us(uint64[6]), pieces_them(uint64[6]), occupied(uint64),\n"
     "      king_sq_us(int), king_sq_them(int), turn_white(bool), ep_square(int)\n"
     "Returns: ndarray (34, 8, 8) float32"},
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
