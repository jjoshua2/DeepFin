// CPU-singleton adapter only. No CUDA path or model startup is changed here.
#include "async_slot.h"
#include <cstdio>
#include <cstdlib>

extern "C" int deepfin_model_run(const float*, uint32_t, float*, uint32_t);
namespace {
deepfin_native::AsyncSlot& slot() {
  // Constructed after model startup; destroyed (and joined) before that model.
  static deepfin_native::AsyncSlot instance(deepfin_model_run);
  return instance;
}
[[noreturn]] void invalid(const std::exception& error) {
  std::fprintf(stderr, "native async contract: %s\n", error.what());
  std::exit(2);
}
}
extern "C" uint32_t deepfin_async_submit(uint32_t epoch, uint32_t request, uint32_t node,
                                          const float* input, uint32_t count) {
  try { return slot().submit(epoch, request, node, input, count); }
  catch (const std::exception& error) { invalid(error); }
}
extern "C" uint32_t deepfin_async_poll(uint32_t token, uint32_t epoch, uint32_t request,
                                       uint32_t node, float* output, uint32_t count) {
  try { return static_cast<uint32_t>(slot().take({token, epoch, request, node}, output, count)); }
  catch (const std::exception& error) { invalid(error); }
}
extern "C" uint32_t deepfin_async_cancel(uint32_t token, uint32_t epoch, uint32_t request, uint32_t node) {
  return slot().cancel({token, epoch, request, node});
}
extern "C" void deepfin_async_shutdown() { slot().shutdown(); }
