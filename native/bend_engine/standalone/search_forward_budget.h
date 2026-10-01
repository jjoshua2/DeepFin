#pragma once
// Native admission/physical ownership only. Bend owns scheduling and decides
// when a search is retired and which configured limit applies to its forwards.
#include <cstdint>
#include <limits>
#include <mutex>

namespace deepfin_native {
class SearchForwardBudget {
 public:
  static constexpr uint32_t maximum_limit = 65536;
  // The seed permits boundary qualification without billions of forwards.
  // Production constructs the process-lifetime guard at zero, exactly once.
  explicit SearchForwardBudget(uint32_t lifetime_calls = 0) : calls_(lifetime_calls) {}
  SearchForwardBudget(const SearchForwardBudget&) = delete;
  SearchForwardBudget& operator=(const SearchForwardBudget&) = delete;

  bool begin(uint32_t epoch, uint32_t limit) {
    std::lock_guard lock(mutex_);
    if (busy_ || !epoch || !limit || limit > maximum_limit
        || calls_ == std::numeric_limits<uint32_t>::max()) return false;
    if (epoch == epoch_) return limit == limit_;  // idempotent; never refund
    if (epoch < epoch_) return false;  // exhaustion cannot wrap to a fresh epoch
    epoch_ = epoch;
    limit_ = limit;
    used_ = 0;
    return true;
  }

  class Admission {
   public:
    explicit Admission(SearchForwardBudget& owner) : owner_(owner), sequence_(owner.enter()) {}
    Admission(const Admission&) = delete;
    Admission& operator=(const Admission&) = delete;
    ~Admission() { if (sequence_) owner_.leave(); }
    explicit operator bool() const { return sequence_ != 0; }
    uint32_t sequence() const { return sequence_; }
   private:
    SearchForwardBudget& owner_;
    const uint32_t sequence_;
  };
  Admission admit() { return Admission(*this); }
  uint32_t calls() const { std::lock_guard lock(mutex_); return calls_; }
  uint32_t used() const { std::lock_guard lock(mutex_); return used_; }

 private:
  uint32_t enter() {
    std::lock_guard lock(mutex_);
    if (busy_ || calls_ == std::numeric_limits<uint32_t>::max()) return 0;
    if (epoch_ ? used_ >= limit_ : calls_ >= maximum_limit) return 0;
    // Charge before executing, including failures and logically cancelled
    // callbacks. The unbound diagnostic path retains its original lifetime cap.
    ++used_;
    busy_ = true;
    return ++calls_;  // bounded sequence is also the trace/audit lifetime count
  }
  void leave() {
    std::lock_guard lock(mutex_);
    busy_ = false;  // releases physical ownership, not the spent admission
  }
  mutable std::mutex mutex_;
  uint32_t epoch_ = 0, limit_ = 0, used_ = 0, calls_ = 0;
  bool busy_ = false;
};
}  // namespace deepfin_native
