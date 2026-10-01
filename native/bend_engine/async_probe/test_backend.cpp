// Test-only deterministic CPU callback. No production target includes this file.
#include <chrono>
#include <cstdint>
#include "../standalone/search_forward_budget.h"
#include <thread>
namespace {
deepfin_native::SearchForwardBudget budget;
}
extern "C" uint32_t deepfin_model_begin_search(uint32_t epoch, uint32_t limit) {
  return budget.begin(epoch, limit);
}
extern "C" int deepfin_model_run(const float* x, uint32_t n, float* y, uint32_t count) {
  if (!x || !y || n != 9344 || count != 1861) return 1;
  auto admission = budget.admit();
  if (!admission) return 1;
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  for (uint32_t i=0;i<n;++i) if (x[i] != 2.0f) return 1;
  for (uint32_t i=0;i<count;++i) y[i] = 2.0f;
  return 0;
}
