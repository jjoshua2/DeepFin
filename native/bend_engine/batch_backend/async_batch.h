#pragma once
// Physical batch ownership only. Row-to-root mapping/cancellation belong to Bend.
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace deepfin_native {
class AsyncBatch {
 public:
  // Callback returns only after all physical memory accesses finish.
  using Execute = std::function<int(const float*, uint32_t, float*, uint32_t)>;
  enum Status : uint32_t { unknown = 0, pending = 1, complete = 2, failed = 4 };
  AsyncBatch(uint32_t batch, uint32_t channels, Execute execute, uint32_t last = 0)
      : batch_(checked_batch(batch)), width_(checked_channels(channels) * 64),
        execute_(std::move(execute)), input_(size_t(batch_) * width_),
        output_(size_t(batch_) * 1861), last_(last), worker_([this] { work(); }) {}
  AsyncBatch(const AsyncBatch&) = delete;
  AsyncBatch& operator=(const AsyncBatch&) = delete;
  ~AsyncBatch() { shutdown(); }

  uint32_t submit(const float* source, uint32_t rows) {
    if (!source || !rows || rows > batch_) throw std::invalid_argument("batch input rows");
    std::lock_guard guard(mutex_);
    if (closing_ || poisoned_) throw std::runtime_error("batch worker closed or failed");
    if (occupied_) return 0;
    if (last_ == std::numeric_limits<uint32_t>::max()) throw std::overflow_error("batch token exhausted");
    std::memcpy(input_.data(), source, size_t(rows) * width_ * sizeof(float));
    rows_ = rows;
    ++last_;
    occupied_ = queued_ = true;
    done_ = false;
    wake_.notify_one();
    return last_;
  }

  Status take(uint32_t token, float* target, uint32_t rows) {
    std::lock_guard guard(mutex_);
    if (!occupied_ || token != last_) return unknown;
    if (!target || rows != rows_) throw std::invalid_argument("batch result rows");
    if (!done_) return pending;
    const auto state = poisoned_ ? failed : complete;
    if (state == complete) std::memcpy(target, output_.data(), size_t(rows) * 1861 * sizeof(float));
    occupied_ = false;
    return state;
  }

  // Non-consuming wait by the same owner that submits/takes. No tensor access,
  // token reuse, or ownership transfer; take() remains the only retirement path.
  Status wait_ready(uint32_t token, std::chrono::milliseconds budget) {
    if (budget.count() < 0) throw std::invalid_argument("negative batch wait budget");
    std::unique_lock guard(mutex_);
    if (!occupied_ || token != last_) return unknown;
    ready_.wait_for(guard, budget, [this, token] {
      return !occupied_ || token != last_ || done_;
    });
    if (!occupied_ || token != last_) return unknown;
    return done_ ? (poisoned_ ? failed : complete) : pending;
  }

  void shutdown() {
    // One owner calls shutdown, never the callback or a racing second shutdown.
    { std::lock_guard guard(mutex_); closing_ = true; wake_.notify_one(); }
    if (worker_.joinable()) worker_.join();
  }

 private:
  static uint32_t checked_batch(uint32_t n) {
    if (!n || n > 16 || (n & (n-1))) throw std::invalid_argument("batch size");
    return n;
  }
  static uint32_t checked_channels(uint32_t n) {
    if (n != 146 && n != 175) throw std::invalid_argument("batch channels");
    return n;
  }
  void work() noexcept {
    for (;;) {
      uint32_t rows;
      { std::unique_lock guard(mutex_);
        wake_.wait(guard, [this] { return queued_ || closing_; });
        if (!queued_) return;
        queued_ = false;
        rows = rows_;
      }
      bool bad = true;
      try { bad = execute_(input_.data(), rows, output_.data(), rows * 1861) != 0; }
      catch (...) { bad = true; }
      { std::lock_guard guard(mutex_); poisoned_ |= bad; done_ = true; }
      ready_.notify_one();
    }
  }
  uint32_t batch_, width_;
  Execute execute_;
  std::mutex mutex_;
  std::condition_variable wake_, ready_;
  std::vector<float> input_, output_;
  uint32_t rows_ = 0, last_;
  bool occupied_ = false, queued_ = false, done_ = false, closing_ = false, poisoned_ = false;
  std::thread worker_; // Launched after initialization of everything it uses.
};
} // namespace deepfin_native
