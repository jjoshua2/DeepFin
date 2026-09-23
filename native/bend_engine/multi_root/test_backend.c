/* Explicit qualification double. Never linked into the LibTorch runner.
 * Row-independent logits depend on actual input features, not row/order/root ID.
 * Trace v2 records the padded model tensor and only real output rows. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "model_contract.h"
static FILE *trace;
static uint32_t calls;
static const float *first_input;
static float *first_output;
static float physical[16 * 175 * 64];
static uint32_t input_changes, output_changes;
static void closed(void) {
    if (trace && fclose(trace)) abort();
    if (getenv("DEEPFIN_BEND_BUFFER_AUDIT"))
        fprintf(stderr, "native-buffer-audit calls=%u input_changes=%u output_changes=%u input_tensor_allocations=1\n",
                calls, input_changes, output_changes);
}
uint32_t deepfin_model_open_batch(void) {
    const char *marker = getenv("DEEPFIN_MULTI_TEST_OPEN");
    if (marker) { FILE *f = fopen(marker, "wx"); if (!f) abort(); fclose(f); }
    const char *path = getenv("DEEPFIN_BEND_MODEL_TRACE");
    if (path) { trace = fopen(path, "wbx"); if (!trace) abort(); }
    atexit(closed);
    return DEEPFIN_MODEL_PROFILE;
}
int deepfin_model_run_batch(const float *x, uint32_t rows, float *y, uint32_t size) {
    const uint32_t width = DEEPFIN_MODEL_CHANNELS * 64;
    if (!rows || rows > DEEPFIN_MODEL_BATCH || size != rows * 1861) return 1;
    if (!calls) { first_input = x; first_output = y; }
    else { input_changes += (first_input != x); output_changes += (first_output != y); }
    ++calls;
    memset(physical, 0, DEEPFIN_MODEL_BATCH * width * sizeof(float));
    memcpy(physical, x, rows * width * sizeof(float));
    const char *fault = getenv("DEEPFIN_MULTI_TEST_FAULT");
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t i = 0; i < 1861; ++i) {
            uint32_t from = fault && !strcmp(fault,"swap") ? (r+1)%rows : r;
            y[r*1861+i] = physical[from*width+(i*7+11)%width] * 0.03125f
                          + (float)((int)(i%31)-15) * 0.015625f;
        }
    if (fault && !strcmp(fault,"fail")) { y[0] = 123.0f; return 1; }
    if (fault && !strcmp(fault,"nan")) y[(rows-1)*1861] = NAN;
    if (trace) {
        uint32_t header[6] = {0x44464232,calls,DEEPFIN_MODEL_BATCH,rows,DEEPFIN_MODEL_CHANNELS,1861};
        if (fwrite(header, sizeof(header), 1, trace) != 1
            || fwrite(physical,sizeof(float),DEEPFIN_MODEL_BATCH*width,trace) != DEEPFIN_MODEL_BATCH*width
            || fwrite(y,sizeof(float),size,trace) != size) abort();
    }
    return 0;
}
