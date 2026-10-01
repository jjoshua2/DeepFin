// Cheap native admission qualification. Uses the exact production guard and
// async adapter with deterministic callbacks; no model, LibTorch, or GPU.
#include "../standalone/search_forward_budget.h"
#include "../standalone/async_slot.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>

using namespace deepfin_native;
using namespace std::chrono_literals;
namespace {
unsigned checks = 0;
void require(bool value) {
  ++checks;
  if (!value) throw std::runtime_error("search forward budget test failed");
}
template<class Exception, class F> void rejects(F action) {
  bool rejected = false;
  try { action(); } catch (const Exception&) { rejected = true; }
  require(rejected);
}
struct Gate {
  std::mutex mutex;
  std::condition_variable wake;
  bool started = false, allowed = false;
  void enter() {
    std::unique_lock lock(mutex);
    started = true;
    wake.notify_all();
    if (!wake.wait_for(lock, 5s, [this] { return allowed; }))
      throw std::runtime_error("test callback release timed out");
  }
  void wait_started() {
    std::unique_lock lock(mutex);
    if (!wake.wait_for(lock, 5s, [this] { return started; }))
      throw std::runtime_error("test callback never started");
  }
  void release() {
    std::lock_guard lock(mutex);
    allowed = true;
    wake.notify_all();
  }
};
AsyncStatus wait_done(AsyncSlot& slot, AsyncKey key) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  for (;;) {
    const auto status = slot.status(key);
    if (status != AsyncStatus::pending) return status;
    if (std::chrono::steady_clock::now() >= deadline)
      throw std::runtime_error("test callback never completed");
    std::this_thread::yield();
  }
}
void bounds_and_no_refund() {
  SearchForwardBudget budget;
  require(!budget.begin(0, 1));
  require(!budget.begin(1, 0));
  require(!budget.begin(1, 65537));
  require(!budget.begin(1, UINT32_MAX));
  require(budget.calls() == 0 && budget.used() == 0);
  require(budget.begin(5, 2));
  require(budget.begin(5, 2));
  require(!budget.begin(5, 1));
  require(!budget.begin(5, 3));
  require(!budget.begin(4, 2));
  {
    auto first = budget.admit();
    require(bool(first) && first.sequence() == 1);
    require(budget.used() == 1 && budget.calls() == 1);
    require(!budget.begin(6, 2));
    require(!budget.begin(5, 2));
    require(!budget.admit());
  }
  require(budget.begin(5, 2));
  require(budget.used() == 1);  // repeating begin cannot refund a completed call
  {
    auto second = budget.admit();
    require(bool(second) && second.sequence() == 2);
  }
  require(budget.begin(5, 2));
  require(!budget.admit());
  require(budget.used() == 2 && budget.calls() == 2);
  require(budget.begin(6, 1));
  require(budget.used() == 0 && budget.calls() == 2);
  rejects<std::runtime_error>([&] {
    auto failure = budget.admit();
    require(bool(failure) && failure.sequence() == 3);
    throw std::runtime_error("injected execution failure");
  });
  require(budget.begin(6, 1));  // exception releases physical ownership only
  require(!budget.admit());
  require(budget.used() == 1 && budget.calls() == 3);
  require(!budget.begin(6, 2));
  require(!budget.begin(5, 1));
  require(budget.begin(UINT32_MAX, 1));
  {
    auto last_epoch = budget.admit();
    require(bool(last_epoch) && last_epoch.sequence() == 4);
  }
  require(!budget.begin(0, 1));
  require(!budget.begin(1, 1));
  require(!budget.begin(UINT32_MAX - 1, 1));
  require(budget.begin(UINT32_MAX, 1));
  require(!budget.admit());
}
void cumulative_and_diagnostic_bounds() {
  SearchForwardBudget diagnostic;
  for (uint32_t i = 1; i <= 65536; ++i) {
    auto call = diagnostic.admit();
    require(bool(call) && call.sequence() == i);
  }
  require(!diagnostic.admit());
  require(diagnostic.calls() == 65536);
  // Only an explicit epoch transitions the diagnostic route into sessions.
  require(diagnostic.begin(1, 1));
  { auto call = diagnostic.admit(); require(bool(call) && call.sequence() == 65537); }
  require(!diagnostic.admit());

  SearchForwardBudget sessions;
  for (uint32_t epoch = 1; epoch <= 2; ++epoch) {
    require(sessions.begin(epoch, 65536));
    for (uint32_t i = 1; i <= 65536; ++i) {
      auto call = sessions.admit();
      require(bool(call) && call.sequence() == (epoch - 1) * 65536 + i);
    }
    require(sessions.begin(epoch, 65536));
    require(!sessions.admit());
  }
  require(sessions.calls() == 131072 && sessions.used() == 65536);
}
void lifetime_exhaustion() {
  SearchForwardBudget budget(UINT32_MAX - 1);
  require(budget.begin(1, 65536));
  {
    auto last = budget.admit();
    require(bool(last) && last.sequence() == UINT32_MAX);
  }
  require(!budget.admit());
  require(!budget.begin(1, 65536));
  require(!budget.begin(2, 65536));
  require(budget.calls() == UINT32_MAX && budget.used() == 1);
}
void occupied_slot_blocks_reset() {
  // Both successful and cancelled completions remain occupied until take;
  // failures additionally poison the slot even after their result is taken.
  for (uint32_t disposition = 0; disposition < 4; ++disposition) {
    SearchForwardBudget budget;
    Gate gate;
    unsigned transitions = 0;
    AsyncSlot slot([&](const float*, uint32_t, float* out, uint32_t n) {
      auto admission = budget.admit();
      if (!admission) return 1;
      gate.enter();
      if (disposition == 3) throw std::runtime_error("injected failure");
      std::fill(out, out + n, 7.0f);
      return disposition == 2 ? 1 : 0;
    });
    auto begin = [&](uint32_t epoch) {
      return slot.when_idle([&] { ++transitions; return budget.begin(epoch, 1); });
    };
    require(begin(1));
    std::array<float, 9344> input{};
    std::array<float, 1861> output{};
    output.fill(-19.0f);
    AsyncKey key{slot.submit(1, 1, 1, input.data(), input.size()), 1, 1, 1};
    require(key.token != 0);
    // Check immediately after submit, before waiting for execution. The slot
    // must reject transition whether the admission is still queued or running.
    require(!begin(2));
    require(transitions == 1);
    gate.wait_started();
    require(slot.status(key) == AsyncStatus::pending);
    require(!begin(2));
    require(!budget.begin(2, 1));
    require(budget.calls() == 1);
    if (disposition == 1) {
      require(slot.cancel(key));
      require(!begin(2));
    }
    gate.release();
    const auto expected = disposition >= 2 ? AsyncStatus::failed
                        : disposition == 1 ? AsyncStatus::cancelled : AsyncStatus::complete;
    require(wait_done(slot, key) == expected);
    // Deterministically done, but deliberately not consumed yet.
    require(!begin(2));
    require(transitions == 1);
    require(slot.take({key.token, 2, 1, 1}, output.data(), output.size()) == AsyncStatus::unknown);
    require(!begin(2));
    if (disposition != 1)
      rejects<std::invalid_argument>([&] { slot.take(key, output.data(), 1860); });
    require(slot.take(key, output.data(), output.size()) == expected);
    require(std::all_of(output.begin(), output.end(), [&](float x) {
      return x == (disposition == 0 ? 7.0f : -19.0f);
    }));
    require(slot.status(key) == AsyncStatus::unknown);
    if (disposition < 2) {
      require(begin(1));
      require(budget.used() == 1 && !budget.admit()); // cancellation never refunded
      require(begin(2));
      require(budget.used() == 0 && budget.calls() == 1);
    } else {
      require(!begin(2)); // backend failure remains terminal for the slot
      require(transitions == 1);
    }
    slot.shutdown();
    require(!begin(3));
  }
}

// Backend symbols for the actual async_model.cpp adapter, using the same guard.
SearchForwardBudget adapter_budget;
Gate adapter_gate;
std::atomic<uint32_t> begin_calls{0};
}
extern "C" uint32_t deepfin_model_begin_search(uint32_t epoch, uint32_t limit) {
  ++begin_calls;
  return adapter_budget.begin(epoch, limit);
}
extern "C" int deepfin_model_run(const float* input, uint32_t count, float* output, uint32_t n) {
  if (!input || !output || count != 9344 || n != 1861) return 1;
  auto admission = adapter_budget.admit();
  if (!admission) return 1;
  adapter_gate.enter();
  std::fill(output, output + n, 7.0f);
  return 0;
}
extern "C" uint32_t deepfin_async_begin_search(uint32_t, uint32_t);
extern "C" uint32_t deepfin_async_submit(uint32_t, uint32_t, uint32_t, const float*, uint32_t);
extern "C" uint32_t deepfin_async_poll(uint32_t, uint32_t, uint32_t, uint32_t, float*, uint32_t);
extern "C" uint32_t deepfin_async_cancel(uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" void deepfin_async_shutdown();
namespace {
uint32_t wait_adapter(uint32_t token, uint32_t epoch, float* output) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  for (;;) {
    const uint32_t status = deepfin_async_poll(token, epoch, 1, 1, output, 1861);
    if (status != 1) return status;
    if (std::chrono::steady_clock::now() >= deadline)
      throw std::runtime_error("adapter did not complete");
    std::this_thread::yield();
  }
}
void actual_adapter() {
  struct Join { ~Join() { adapter_gate.release(); deepfin_async_shutdown(); } } join;
  std::array<float, 9344> input{};
  std::array<float, 1861> output{};
  output.fill(-19.0f);
  require(!deepfin_async_begin_search(0, 1));
  require(!deepfin_async_begin_search(1, 0));
  require(!deepfin_async_begin_search(1, 65537));
  require(deepfin_async_begin_search(10, 1));
  const auto token = deepfin_async_submit(10, 1, 1, input.data(), input.size());
  require(token != 0);
  adapter_gate.wait_started();
  const uint32_t prior_begins = begin_calls;
  require(!deepfin_async_begin_search(11, 1));
  require(begin_calls == prior_begins); // occupied adapter never calls backend begin
  require(deepfin_async_cancel(token, 10, 1, 1));
  require(!deepfin_async_begin_search(11, 1));
  adapter_gate.release();
  require(wait_adapter(token, 10, output.data()) == 3);
  require(std::all_of(output.begin(), output.end(), [](float x) { return x == -19.0f; }));
  require(adapter_budget.calls() == 1 && adapter_budget.used() == 1);
  require(deepfin_async_begin_search(10, 1));
  require(!adapter_budget.admit());
  require(!deepfin_async_begin_search(10, 2));
  require(!deepfin_async_begin_search(9, 1));
  require(deepfin_async_begin_search(11, 2));
  for (uint32_t i = 0; i < 2; ++i) {
    require(deepfin_async_begin_search(11, 2));
    const auto next = deepfin_async_submit(11, 1, 1, input.data(), input.size());
    require(next > token);
    require(wait_adapter(next, 11, output.data()) == 2);
  }
  require(adapter_budget.calls() == 3 && adapter_budget.used() == 2);
  require(deepfin_async_begin_search(11, 2));
  require(!adapter_budget.admit());
  deepfin_async_shutdown();
  require(!deepfin_async_begin_search(12, 1));
}
}
int main() {
  bounds_and_no_refund();
  cumulative_and_diagnostic_bounds();
  lifetime_exhaustion();
  occupied_slot_blocks_reset();
  actual_adapter();
  std::printf("{\"status\":\"passed\",\"assertions\":%u,\"session_admissions\":131072,"
              "\"scope\":\"production guard and async adapter; deterministic callbacks; no model\"}\n", checks);
}
