// Opt-in CPU-only bridge. Model startup must precede first submission.
#include "async_batch.h"
#include "model_contract.h"
#include <cstdio>
#include <cstdlib>
extern "C" int deepfin_model_run_batch(const float*, uint32_t, float*, uint32_t);
namespace {
deepfin_native::AsyncBatch& instance() {
  static deepfin_native::AsyncBatch worker(DEEPFIN_MODEL_BATCH, DEEPFIN_MODEL_CHANNELS, deepfin_model_run_batch);
  return worker;
}
[[noreturn]] void invalid(const std::exception& e) {
  std::fprintf(stderr, "async batch contract: %s\n", e.what());
  std::exit(2);
}
}
extern "C" uint32_t deepfin_async_batch_submit(const float* x, uint32_t rows) {
  try { return instance().submit(x, rows); }
  catch (const std::exception& e) { invalid(e); }
}
extern "C" uint32_t deepfin_async_batch_poll(uint32_t token, float* y, uint32_t rows) {
  try { return instance().take(token, y, rows); }
  catch (const std::exception& e) { invalid(e); }
}
// At most one millisecond of requested waiting before Bend services control
// and deadlines again. Notifications can end this wait early; it never takes.
extern "C" void deepfin_async_batch_wait(uint32_t token) {
  try {
    if (instance().wait_ready(token, std::chrono::milliseconds(1)) == deepfin_native::AsyncBatch::unknown)
      throw std::invalid_argument("batch completion token");
  } catch (const std::exception& e) { invalid(e); }
}
extern "C" void deepfin_async_batch_shutdown() { instance().shutdown(); }
