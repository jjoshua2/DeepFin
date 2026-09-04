#ifndef DEEPFIN_NATIVE_BUILD_ATTESTATION_H
#define DEEPFIN_NATIVE_BUILD_ATTESTATION_H

/* Portable builds remain usable but are intentionally ineligible for
 * decision-grade evidence.  scripts/build_production_extensions.py supplies
 * all three macros from a clean, revision-bound dependency snapshot. */
#ifndef DEEPFIN_BUILD_MODULE_NAME
#define DEEPFIN_BUILD_MODULE_NAME "unattested"
#endif
#ifndef DEEPFIN_BUILD_GIT_SHA
#define DEEPFIN_BUILD_GIT_SHA "unattested"
#endif
#ifndef DEEPFIN_BUILD_INPUT_SHA256
#define DEEPFIN_BUILD_INPUT_SHA256 "unattested"
#endif

static int deepfin_add_native_build_attestation(PyObject *module) {
    if (PyModule_AddStringConstant(module, "BUILD_ATTESTATION_SCHEMA",
                                   "deepfin.native_build.v1") < 0 ||
        PyModule_AddStringConstant(module, "BUILD_MODULE_NAME",
                                   DEEPFIN_BUILD_MODULE_NAME) < 0 ||
        PyModule_AddStringConstant(module, "BUILD_SOURCE_GIT_SHA",
                                   DEEPFIN_BUILD_GIT_SHA) < 0 ||
        PyModule_AddStringConstant(module, "BUILD_INPUT_SHA256",
                                   DEEPFIN_BUILD_INPUT_SHA256) < 0) {
        return -1;
    }
    return 0;
}

#endif
