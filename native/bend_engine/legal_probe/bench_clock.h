#ifndef DEEPFIN_BEND_BENCH_CLOCK_H
#define DEEPFIN_BEND_BENCH_CLOCK_H
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Only tree traversal is timed. Initialization, validation and printing are not. */
static struct timespec perft_started;
static void perft_clock_start(void) {
    if (clock_gettime(CLOCK_MONOTONIC, &perft_started)) {
        perror("clock_gettime"); exit(2);
    }
    __asm__ __volatile__("" ::: "memory");
}
static void perft_clock_finish(const char *engine, unsigned depth, uint64_t nodes) {
    __asm__ __volatile__("" ::: "memory");
    struct timespec end;
    if (clock_gettime(CLOCK_MONOTONIC, &end)) {
        perror("clock_gettime"); exit(2);
    }
    int64_t ns = (int64_t)(end.tv_sec - perft_started.tv_sec) * INT64_C(1000000000)
        + (int64_t)end.tv_nsec - (int64_t)perft_started.tv_nsec;
    if (ns <= 0) { fputs("nonpositive benchmark interval\n", stderr); exit(2); }
    printf("bench %s %u %" PRIu64 " %" PRId64 "\n", engine, depth, nodes, ns);
}
#endif
