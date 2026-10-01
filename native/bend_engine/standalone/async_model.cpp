// CPU-singleton adapter only. No CUDA path or model startup is changed here.
#include "async_slot.h"
#include <cstdio>
#include <cstdlib>
#include <mutex>

extern "C" uint32_t deepfin_model_begin_search(uint32_t, uint32_t);
extern "C" int deepfin_model_run(const float*, uint32_t, float*, uint32_t);
namespace {
// Serializes the session binding with submit before the slot takes ownership.
// Zero preserves the original unbound diagnostic adapter until a begin succeeds.
std::mutex admission_mutex;
uint32_t bound_epoch = 0;
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
extern "C" uint32_t deepfin_async_begin_search(uint32_t epoch, uint32_t limit) {
  try {
    std::lock_guard lock(admission_mutex);
    const bool begun = slot().when_idle([=] { return deepfin_model_begin_search(epoch, limit) != 0; });
    if (begun) bound_epoch = epoch;
    return begun;
  } catch (const std::exception& error) { invalid(error); }
}
extern "C" uint32_t deepfin_async_submit(uint32_t epoch, uint32_t request, uint32_t node,
                                          const float* input, uint32_t count) {
  try {
    std::lock_guard lock(admission_mutex);
    if (bound_epoch && epoch != bound_epoch) return 0;
    return slot().submit(epoch, request, node, input, count);
  } catch (const std::exception& error) { invalid(error); }
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
