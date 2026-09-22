/* Transport test double only; never linked into the native model product. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "model_contract.h"
static const float *first_input;
static float *first_output;
static unsigned calls;
uint32_t deepfin_model_open_batch(void) { return DEEPFIN_MODEL_PROFILE; }
int deepfin_model_run_batch(const float *x, uint32_t rows, float *y, uint32_t count) {
    if (getenv("DEEPFIN_BATCH_TEST_FAIL")) { fputs("requested test backend failure\n", stderr); return 1; }
    const uint32_t stride = DEEPFIN_MODEL_CHANNELS * 64;
    if (!calls) { first_input = x; first_output = y; }
    if (x != first_input || y != first_output || !rows || rows > DEEPFIN_MODEL_BATCH || count != rows * 1861)
        return 1;
    ++calls;
    const uint32_t step = calls == 5 ? 1 : calls;
    for (uint32_t i = 0; i < rows * stride; ++i)
        if (x[i] != (float)((i + step * 7) % 31) * 0.0625f) return 1;
    if (!isnan(x[rows * stride]) || !isnan(y[count])) return 1;
    for (uint32_t row = 0; row < rows; ++row)
        for (uint32_t i = 0; i < 1861; ++i) y[row * 1861 + i] = x[row * stride] + (float)i;
    fprintf(stderr, "transport-call=%u real=%u physical=%u\n", calls, rows, DEEPFIN_MODEL_BATCH);
    return 0;
}
