// Completion notification contracts, not a scheduler/model latency benchmark.
#include "async_batch.h"
#include <algorithm>
#include <chrono>
#include <future>
#include <iostream>
#include <string>

using deepfin_native::AsyncBatch;
using namespace std::chrono_literals;
namespace {
unsigned checks = 0;
void require(bool value, const char* what) {
  ++checks;
  if (!value) throw std::runtime_error(what);
}
struct Gate {
  std::mutex mutex;
  std::condition_variable cv;
  bool entered = false, released = false;
  void block() {
    std::unique_lock guard(mutex);
    entered = true;
    cv.notify_all();
    cv.wait(guard, [this] { return released; });
  }
  void started() {
    std::unique_lock guard(mutex);
    require(cv.wait_for(guard, 5s, [this] { return entered; }), "callback did not start");
  }
  void release() {
    std::lock_guard guard(mutex);
    released = true;
    cv.notify_all();
  }
};
// Declared after AsyncBatch so callback release precedes the worker's join,
// including exceptions from assertions while evaluation is deliberately held.
struct ReleaseOnExit {
  Gate& gate;
  ~ReleaseOnExit() { gate.release(); }
};
void case_run(uint32_t batch, uint32_t channels, unsigned fault, bool abort_held = false) {
  Gate gate;
  std::vector<float> input(size_t(batch) * channels * 64, 7.0f);
  std::vector<float> output(size_t(batch) * 1861 + 1, -99.0f);
  AsyncBatch worker(batch, channels, [&](const float* x, uint32_t rows, float* y, uint32_t count) {
    gate.block();
    if (count != rows * 1861) return 1;
    for (uint32_t i = 0; i < count; ++i) y[i] = x[0] + float(i);
    if (fault == 2) throw std::runtime_error("injected callback exception");
    return fault == 1 ? 1 : 0;
  });
  ReleaseOnExit release_on_exit{gate};
  require(worker.wait_ready(1, 0ms) == AsyncBatch::unknown, "idle wait accepted");
  const auto token = worker.submit(input.data(), batch);
  gate.started();
  if (abort_held) throw std::runtime_error("intentional held-callback assertion");
  // Only the native worker's snapshot may be referenced after submit returns.
  std::fill(input.begin(), input.end(), 999.0f);
  require(worker.wait_ready(token + 1, 1ms) == AsyncBatch::unknown, "wrong token wait accepted");
  require(worker.wait_ready(token, 0ms) == AsyncBatch::pending, "zero-budget wait took output");
  const auto began = std::chrono::steady_clock::now();
  require(worker.wait_ready(token, 1ms) == AsyncBatch::pending, "held callback completed");
  require(std::chrono::steady_clock::now() - began < 500ms, "bounded wait did not return to owner");
  require(worker.submit(input.data(), batch) == 0, "wait released occupied slot");
  require(worker.take(token, output.data(), batch) == AsyncBatch::pending, "wait consumed a pending token");
  require(output[0] == -99.0f && output.back() == -99.0f, "wait wrote output");
  bool bad = false;
  try { (void)worker.wait_ready(token, -1ms); }
  catch (const std::invalid_argument&) { bad = true; }
  require(bad, "negative timeout accepted");

  // A deliberately longer library wait makes missed notification observable
  // without asserting microsecond OS latency. The application bridge uses 1ms.
  std::promise<void> waiting;
  auto entered = waiting.get_future();
  auto future = std::async(std::launch::async, [&] {
    waiting.set_value();
    return worker.wait_ready(token, 3s);
  });
  entered.wait();
  const bool held = future.wait_for(20ms) == std::future_status::timeout;
  gate.release(); // Release before any assertion so failures cannot strand callback ownership.
  const bool notified = future.wait_for(750ms) == std::future_status::ready;
  const auto status = future.get();
  require(held, "wait returned before callback release");
  require(notified, "completion notification did not wake waiter");
  const auto expected = fault ? AsyncBatch::failed : AsyncBatch::complete;
  require(status == expected, "completion status differs");
  // Completion already visible before wait: repeated waits must neither sleep
  // until timeout nor retire/copy output. Includes failed/throwing callbacks.
  const auto ready_start = std::chrono::steady_clock::now();
  require(worker.wait_ready(token, 3s) == expected, "ready-before-wait status differs");
  require(std::chrono::steady_clock::now() - ready_start < 750ms, "ready-before-wait missed completion");
  require(worker.wait_ready(token, 0ms) == expected, "repeat wait consumed token");
  require(output[0] == -99.0f, "completion wait copied output");
  require(worker.take(token, output.data(), batch) == expected, "take after wait failed");
  require(output[0] == (fault ? -99.0f : 7.0f), "failed/successful output ownership differs");
  require(output.back() == -99.0f, "output canary overwritten");
  require(worker.wait_ready(token, 1ms) == AsyncBatch::unknown, "retired token revived");
  require(worker.take(token, output.data(), batch) == AsyncBatch::unknown, "duplicate retirement");
  if (!fault) {
    const auto next = worker.submit(input.data(), 1);
    require(next == token + 1, "nonmonotonic reuse token");
    require(worker.wait_ready(token, 1ms) == AsyncBatch::unknown, "old wait matched new batch");
    require(worker.wait_ready(next, 3s) == AsyncBatch::complete, "reused callback failed");
    require(worker.take(next, output.data(), 1) == AsyncBatch::complete, "reused token lost");
    require(output[0] == 999.0f, "reused input snapshot differs");
  }
  worker.shutdown();
  worker.shutdown();
  require(worker.wait_ready(token, 0ms) == AsyncBatch::unknown, "shutdown revived token");
}
}
int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--abort-held") {
      case_run(4, 146, 0, true);
      throw std::runtime_error("held-callback assertion did not execute");
    }
    if (argc != 1) throw std::invalid_argument("unexpected test argument");
    for (uint32_t batch : {1u, 2u, 4u, 8u, 16u})
      for (uint32_t channels : {146u, 175u}) case_run(batch, channels, 0);
    for (unsigned fault : {1u, 2u}) case_run(4, 146, fault);
    std::cout << "{\"status\":\"passed\",\"cases\":12,\"assertions\":" << checks << "}\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
