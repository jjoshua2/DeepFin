/* Shared bitboard-to-plane conversion used by CBoard and feature encoders. */

#ifndef BITBOARD_PLANES_IMPL_H
#define BITBOARD_PLANES_IMPL_H

#include <stdint.h>
#include <string.h>

/* Compile-time endianness detection. The LUT path reinterprets uint64_t
 * memory bytes directly and is only correct on little-endian hosts. */
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define BB_PLANE_USE_LUT 1
#elif defined(_WIN32) || defined(__x86_64__) || defined(__i386__) || \
      defined(__aarch64__) || defined(__arm__)
/* All common desktop/server targets are LE. */
#define BB_PLANE_USE_LUT 1
#else
#define BB_PLANE_USE_LUT 0
#endif

#ifdef __AVX2__
#include <immintrin.h>
#define BB_PLANE_USE_AVX2 1
#else
#define BB_PLANE_USE_AVX2 0
#endif

/* BYTE_TO_8FLOATS is the scalar-path LUT (8 KB, 256 entries x 8 floats).
 * AVX2 builds use _avx2_byte_to_8floats instead, so the table is dead
 * weight there -- only emit it on non-AVX2. */
#if !BB_PLANE_USE_AVX2
static float BYTE_TO_8FLOATS[256][8];

static void init_byte_to_float_lut(void) {
    for (int b = 0; b < 256; b++)
        for (int i = 0; i < 8; i++)
            BYTE_TO_8FLOATS[b][i] = (float)((b >> i) & 1);
}
#else
static inline void init_byte_to_float_lut(void) { /* no-op on AVX2 */ }
#endif

#if BB_PLANE_USE_AVX2
/* AVX2: expand one byte to 8 floats (0.0f or 1.0f) using SIMD.
 * Processes the entire 64-float plane in 8 iterations of pure SIMD ops. */
static const int32_t _avx2_bit_masks[8] __attribute__((aligned(32))) =
    {1, 2, 4, 8, 16, 32, 64, 128};

static inline void _avx2_byte_to_8floats(uint8_t byte, float *out) {
    __m256i masks = _mm256_load_si256((const __m256i *)_avx2_bit_masks);
    __m256i val = _mm256_set1_epi32(byte);
    __m256i and_r = _mm256_and_si256(val, masks);
    /* non-zero -> all-ones mask */
    __m256i cmp = _mm256_cmpeq_epi32(and_r, _mm256_setzero_si256());
    __m256 result = _mm256_andnot_ps(_mm256_castsi256_ps(cmp),
                                      _mm256_set1_ps(1.0f));
    _mm256_storeu_ps(out, result);
}

static void bitboard_to_plane_white(uint64_t bb, float *out) {
    if (bb == 0) { memset(out, 0, 64 * sizeof(float)); return; }
    const uint8_t *bytes = (const uint8_t *)&bb;
    for (int r = 0; r < 8; r++)
        _avx2_byte_to_8floats(bytes[r], out + r * 8);
}

static void bitboard_to_plane_black(uint64_t bb, float *out) {
    if (bb == 0) { memset(out, 0, 64 * sizeof(float)); return; }
    const uint8_t *bytes = (const uint8_t *)&bb;
    for (int r = 0; r < 8; r++)
        _avx2_byte_to_8floats(bytes[7 - r], out + r * 8);
}

#else /* non-AVX2 path */

static void bitboard_to_plane_white(uint64_t bb, float *out) {
    if (bb == 0) {
        memset(out, 0, 64 * sizeof(float));
        return;
    }
#if BB_PLANE_USE_LUT
    const uint8_t *bytes = (const uint8_t *)&bb;
    for (int r = 0; r < 8; r++)
        memcpy(out + r * 8, BYTE_TO_8FLOATS[bytes[r]], 8 * sizeof(float));
#else
    for (int r = 0; r < 8; r++)
        for (int f = 0; f < 8; f++)
            out[r * 8 + f] = (float)((bb >> (r * 8 + f)) & 1);
#endif
}

static void bitboard_to_plane_black(uint64_t bb, float *out) {
    if (bb == 0) {
        memset(out, 0, 64 * sizeof(float));
        return;
    }
#if BB_PLANE_USE_LUT
    const uint8_t *bytes = (const uint8_t *)&bb;
    for (int r = 0; r < 8; r++)
        memcpy(out + r * 8, BYTE_TO_8FLOATS[bytes[7 - r]], 8 * sizeof(float));
#else
    for (int r = 0; r < 8; r++)
        for (int f = 0; f < 8; f++)
            out[r * 8 + f] = (float)((bb >> ((7 - r) * 8 + f)) & 1);
#endif
}

#endif /* BB_PLANE_USE_AVX2 */

/* CBoard's legacy slider functions are defined immediately before this shared
 * header. CBoard builds rename those helpers to *_reference, then install the
 * PEXT/magic implementations here so all subsequent legal-move/search code
 * compiles against the table-backed versions. */
#ifdef DEEPFIN_FAST_SLIDERS
#include "_slider_attacks_impl.h"
#endif

#endif /* BITBOARD_PLANES_IMPL_H */
