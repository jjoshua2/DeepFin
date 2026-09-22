/* Test-only backend: checks the real generated effect's addresses and contents.
 * Not linked by the engine build, and never a neural fallback. */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
uint32_t deepfin_model_open(void) { return 1; }
int deepfin_model_run(const float *x, uint32_t count, float *y, uint32_t capacity) {
    static const void *first_input;
    static const void *first_output;
    static uint32_t calls;
    if (getenv("DEEPFIN_PROBE_FAIL")) return 1;
    if (!calls) { first_input = x; first_output = y; }
    if (first_input != x || first_output != y || capacity != 1861) return 1;
    ++calls;
    for (uint32_t i = 0; i < count; ++i) {
        float v;
        memcpy(&v, (const char *)x + i * sizeof(float), sizeof v);
        if (v != (float)calls) return 1;
    }
    for (uint32_t i = 0; i < capacity; ++i) {
        float v = (float)(calls + i);
        memcpy((char *)y + i * sizeof(float), &v, sizeof v);
    }
    fprintf(stderr, "buffer-reuse-call=%u rows=%u\n", calls, count);
    return 0;
}
