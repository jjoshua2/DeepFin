/* Test-only block around the existing deterministic callback. Never linked into
 * the model product. Pipe signaling has no access to tree/root/cancellation data. */
#include <errno.h>
#include <limits.h>
#include <unistd.h>
#define deepfin_model_run_batch deterministic_model_run_batch
#include "test_backend.c"
#undef deepfin_model_run_batch
static int descriptor(const char *name) {
    const char *s = getenv(name); char *end = NULL;
    if (!s || !*s) abort();
    long n = strtol(s, &end, 10);
    if (*end || n < 3 || n > INT_MAX) abort();
    return (int)n;
}
int deepfin_model_run_batch(const float *x, uint32_t rows, float *y, uint32_t size) {
    const int event = descriptor("DEEPFIN_TEST_EVENT_FD");
    const int release = descriptor("DEEPFIN_TEST_RELEASE_FD");
    ssize_t n; char c = 0;
    do { n = write(event, "S", 1); } while (n < 0 && errno == EINTR);
    if (n != 1) abort();
    do { n = read(release, &c, 1); } while (n < 0 && errno == EINTR);
    if (n != 1 || c != 'R') abort();
    return deterministic_model_run_batch(x, rows, y, size);
}
