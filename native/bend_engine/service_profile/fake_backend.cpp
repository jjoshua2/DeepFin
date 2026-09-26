// Test callback only; not a model and never linked by build.sh.
#include "model_contract.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

extern "C" uint32_t deepfin_model_open_batch() {
  const char* fault = std::getenv("SERVICE_TEST_FAULT");
  return fault && !std::strcmp(fault, "startup") ? 0 : DEEPFIN_MODEL_PROFILE;
}
extern "C" int deepfin_model_run_batch(const float* input, uint32_t rows, float* output, uint32_t n) {
  if (!rows || rows > DEEPFIN_MODEL_BATCH || n != rows * 1861) return 1;
  const char* fault = std::getenv("SERVICE_TEST_FAULT");
  if (fault && !std::strcmp(fault, "fail")) return 1;
  if (fault && !std::strcmp(fault, "throw")) throw std::runtime_error("test exception");
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t j = 0; j < 1861; ++j)
      output[r * 1861 + j] = input[r * DEEPFIN_MODEL_CHANNELS * 64] + float(j) / 4096;
  if (fault && !std::strcmp(fault, "nonfinite")) output[n - 1] = std::numeric_limits<float>::quiet_NaN();
  if (fault && !std::strcmp(fault, "mismatch")) output[n - 1] += 100;
  if (fault && !std::strcmp(fault, "tail")) output[n] = 1; // allocated guard cell, not OOB
  return 0;
}
