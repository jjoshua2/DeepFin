#pragma once
// One bounded execution worker, not a search scheduler. The worker never retains
// a Bend pointer: submission copies real input bytes into its own stable storage.
#include <array>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace deepfin_native {
struct AsyncKey {
  uint32_t token, epoch, request, node;
  bool operator==(const AsyncKey&) const = default;
};
enum class AsyncStatus : uint32_t { unknown = 0, pending = 1, complete = 2, cancelled = 3, failed = 4 };

class AsyncSlot {
 public:
  // Execute must complete ALL physical accesses before returning or throwing.
  // This adapter is CPU-singleton only; do not substitute a launch-only callback.
  using Execute = std::function<int(const float*, uint32_t, float*, uint32_t)>;
  explicit AsyncSlot(Execute execute, uint32_t last_token = 0)
      : execute_(std::move(execute)), last_token_(last_token), worker_([this] { work(); }) {}
  AsyncSlot(const AsyncSlot&) = delete;
  AsyncSlot& operator=(const AsyncSlot&) = delete;
  ~AsyncSlot() { shutdown(); }

  uint32_t submit(uint32_t epoch, uint32_t request, uint32_t node,
                  const float* source, uint32_t count) {
    if (!source || (count != 9344 && count != 11200))
      throw std::invalid_argument("async singleton input shape");
    std::lock_guard lock(mutex_);
    if (closing_ || poisoned_) throw std::runtime_error("async slot is closed or poisoned");
    if (occupied_) return 0;  // bounded backpressure, not a replacement/overwrite
    if (last_token_ == std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("async request identity exhausted; no recycling");
    std::memcpy(input_.data(), source, size_t(count) * sizeof(float));
    key_ = {++last_token_, epoch, request, node};
    count_ = count;
    occupied_ = queued_ = true;
    done_ = cancelled_ = failed_ = false;
    wake_.notify_one();
    return key_.token;
  }

  bool cancel(AsyncKey key) {
    std::lock_guard lock(mutex_);
    if (!occupied_ || key != key_) return false;
    cancelled_ = true;  // logical cancellation never frees physical storage
    return true;
  }

  AsyncStatus take(AsyncKey key, float* destination, uint32_t capacity) {
    std::lock_guard lock(mutex_);
    if (!occupied_ || key != key_) return AsyncStatus::unknown;
    // A malformed poll must not consume a valid result, even while still busy.
    if (!cancelled_ && (!destination || capacity != 1861))
      throw std::invalid_argument("async singleton output shape");
    if (!done_) return AsyncStatus::pending;
    const auto status = failed_ ? AsyncStatus::failed
                       : cancelled_ ? AsyncStatus::cancelled : AsyncStatus::complete;
    if (status == AsyncStatus::complete)
      std::memcpy(destination, output_.data(), 1861 * sizeof(float));
    occupied_ = false;  // consume once; token and all three search IDs must match
    return status;
  }

  void shutdown() {
    // Called only by the owner (never the execution callback). A running forward
    // is not preemptible. Join before the model or staging buffers can be freed.
    {
      std::lock_guard lock(mutex_);
      closing_ = true;
      wake_.notify_one();
    }
    if (worker_.joinable()) worker_.join();
  }

 private:
  void work() noexcept {
    for (;;) {
      uint32_t count;
      {
        std::unique_lock lock(mutex_);
        wake_.wait(lock, [this] { return queued_ || closing_; });
        if (!queued_) return;
        queued_ = false;
        count = count_;
      }
      bool failed = true;
      // Cancelled admissions still run. This keeps their committed compute and
      // lifetime explicit; cancellation does not silently refund a reservation.
      try { failed = execute_(input_.data(), count, output_.data(), 1861) != 0; }
      catch (...) { failed = true; }
      {
        std::lock_guard lock(mutex_);
        failed_ = failed;
        poisoned_ = poisoned_ || failed;
        done_ = true;
      }
    }
  }
  Execute execute_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::array<float, 16384> input_{};
  std::array<float, 1861> output_{};
  AsyncKey key_{};
  uint32_t count_ = 0, last_token_ = 0;
  bool occupied_ = false, queued_ = false, done_ = false;
  bool cancelled_ = false, failed_ = false, poisoned_ = false, closing_ = false;
  // Last member: nothing accessed by the thread is uninitialized at launch.
  std::thread worker_;
};
}  // namespace deepfin_native
