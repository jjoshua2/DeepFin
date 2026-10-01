// Test-only deterministic CPU callback. No production target includes this file.
#include <chrono>
#include <cstdint>
#include <thread>
extern "C" int deepfin_model_run(const float* x, uint32_t n, float* y, uint32_t count) {
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  if (n != 9344 || count != 1861) return 1;
  for (uint32_t i=0;i<n;++i) if (x[i] != 2.0f) return 1;
  for (uint32_t i=0;i<count;++i) y[i] = 2.0f;
  return 0;
}
