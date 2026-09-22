// Deterministic blocking callback controls, not a fake GPU or strength test.
#include "../standalone/async_slot.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <vector>

using namespace deepfin_native;
using namespace std::chrono_literals;
static unsigned checks = 0;
static void require(bool ok) { ++checks; if (!ok) throw std::runtime_error("async slot test failed"); }
template<class Exception, class F> static void rejects(F action) {
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
    started = true; wake.notify_all();
    wake.wait(lock, [this] { return allowed; });
  }
  void wait_started() {
    std::unique_lock lock(mutex);
    if (!wake.wait_for(lock, 5s, [this] { return started; })) throw std::runtime_error("worker never started");
  }
  void release() { std::lock_guard lock(mutex); allowed = true; wake.notify_all(); }
};
static AsyncStatus wait(AsyncSlot& slot, AsyncKey key, float* out = nullptr, uint32_t n = 0) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  for (;;) {
    auto status = slot.take(key, out, n);
    if (status != AsyncStatus::pending) return status;
    if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("completion never arrived");
    std::this_thread::sleep_for(1ms);
  }
}
static void lifetime_and_identity() {
  Gate gate;
  std::array<float,16384> input{}; input.fill(2.0f);
  std::array<float,2048> output{}; output.fill(-19.0f);
  std::atomic<unsigned> calls{0};
  AsyncSlot slot([&](const float* x,uint32_t n,float* y,uint32_t m) {
    gate.enter(); ++calls;
    if (n != 9344 || m != 1861) return 1;
    for (uint32_t i=0;i<n;++i) if (x[i] != 2.0f) return 1;
    std::fill(y,y+m,4.0f); return 0;
  });
  AsyncKey key{slot.submit(1,7,10,input.data(),9344),1,7,10};
  require(key.token == 1);
  input.fill(97.0f);  // worker must have an owned snapshot, not a Bend pointer
  gate.wait_started();
  require(slot.take(key,output.data(),1861) == AsyncStatus::pending);
  require(slot.submit(2,8,10,input.data(),9344) == 0);
  for (AsyncKey bad : {AsyncKey{0,1,7,10},AsyncKey{2,1,7,10},AsyncKey{1,2,7,10},
                       AsyncKey{1,1,8,10},AsyncKey{1,1,7,11}}) {
    require(slot.take(bad,output.data(),1861) == AsyncStatus::unknown);
    require(!slot.cancel(bad));
  }
  rejects<std::invalid_argument>([&] { slot.take(key,output.data(),1860); });
  require(std::all_of(output.begin(),output.end(),[](float x){return x == -19.0f;}));
  require(slot.cancel(key)); require(slot.cancel(key));
  require(slot.take(key,nullptr,0) == AsyncStatus::pending);
  require(slot.submit(1,7,10,input.data(),9344) == 0); // cancellation cannot free the slot
  gate.release();
  require(wait(slot,key) == AsyncStatus::cancelled);
  require(calls == 1);
  require(slot.take(key,output.data(),1861) == AsyncStatus::unknown);
  require(!slot.cancel(key));
  require(std::all_of(output.begin(),output.end(),[](float x){return x == -19.0f;}));
  input.fill(2.0f);
  AsyncKey next{slot.submit(1,7,10,input.data(),9344),1,7,10};
  require(next.token > key.token);
  require(slot.take(key,output.data(),1861) == AsyncStatus::unknown);
  require(wait(slot,next,output.data(),1861) == AsyncStatus::complete);
  require(std::all_of(output.begin(),output.begin()+1861,[](float x){return x == 4.0f;}));
  require(std::all_of(output.begin()+1861,output.end(),[](float x){return x == -19.0f;}));
  require(slot.take(next,output.data(),1861) == AsyncStatus::unknown);
  slot.shutdown(); slot.shutdown();
  rejects<std::runtime_error>([&] { slot.submit(1,7,10,input.data(),9344); });
}
static void failures() {
  for (bool throwing : {false,true}) {
    AsyncSlot slot([=](const float*,uint32_t,float* y,uint32_t) -> int {
      y[0] = 97.0f; y[17] = -102.0f;  // partial backend writes must remain private on failure
      if (throwing) throw std::runtime_error("injected failure");
      return 1;
    });
    std::array<float,9344> input{};
    std::array<float,1861> out{}; out.fill(12.0f);
    rejects<std::invalid_argument>([&] { slot.submit(1,1,1,nullptr,9344); });
    rejects<std::invalid_argument>([&] { slot.submit(1,1,1,input.data(),9343); });
    AsyncKey key{slot.submit(1,1,1,input.data(),9344),1,1,1};
    require(wait(slot,key,out.data(),1861) == AsyncStatus::failed);
    require(std::all_of(out.begin(),out.end(),[](float x){return x == 12.0f;}));
    rejects<std::runtime_error>([&] { slot.submit(1,1,1,input.data(),9344); });
  }
  Gate gate;
  AsyncSlot slot([&](const float*,uint32_t,float*,uint32_t){ gate.enter(); return 1; });
  std::array<float,9344> input{};
  AsyncKey key{slot.submit(1,1,1,input.data(),9344),1,1,1};
  gate.wait_started(); require(slot.cancel(key)); gate.release();
  require(wait(slot,key) == AsyncStatus::failed); // cancellation cannot hide a backend failure
}
static void shutdown_with_pending_work() {
  for (bool cancelled : {false, true}) {
    Gate gate;
    std::atomic<bool> finished{false};
    std::array<float,9344> input{};
    std::array<float,1861> output{}; output.fill(-19.f);
    AsyncSlot slot([&](const float*,uint32_t,float* y,uint32_t n) {
      gate.enter(); std::fill(y,y+n,4.f); finished=true; return 0;
    });
    AsyncKey key{slot.submit(1,1,1,input.data(),9344),1,1,1};
    gate.wait_started();
    if (cancelled) require(slot.cancel(key));
    std::thread release([&]{std::this_thread::sleep_for(10ms);gate.release();});
    slot.shutdown();
    const bool completed_before_return=finished.load();
    release.join();  // ensure test cleanup also works when checking a broken join
    require(completed_before_return);
    require(slot.take(key,output.data(),1861)==(cancelled ? AsyncStatus::cancelled:AsyncStatus::complete));
    require(std::all_of(output.begin(),output.end(),[=](float x){return x==(cancelled ? -19.f:4.f);}));
  }
}
static void competing_admission() {
  Gate gate;
  std::array<float,9344> input{};
  std::array<uint32_t,8> tokens{};
  AsyncSlot slot([&](const float*,uint32_t,float*,uint32_t){gate.enter();return 0;});
  std::vector<std::thread> threads;
  for (uint32_t i=0;i<8;++i)
    threads.emplace_back([&,i]{tokens[i]=slot.submit(2,i,3,input.data(),9344);});
  for (auto& t:threads) t.join();
  require(std::count_if(tokens.begin(),tokens.end(),[](auto x){return x != 0;}) == 1);
  auto winner=uint32_t(std::find_if(tokens.begin(),tokens.end(),[](auto x){return x != 0;})-tokens.begin());
  AsyncKey key{tokens[winner],2,winner,3};
  require(slot.cancel(key)); gate.release();
  require(wait(slot,key) == AsyncStatus::cancelled);
}
static void reuse_and_wrap() {
  const float* first_in = nullptr; float* first_out = nullptr;
  AsyncSlot slot([&](const float* x,uint32_t n,float* y,uint32_t m){
    if (!first_in) { first_in=x; first_out=y; }
    if (x!=first_in || y!=first_out || (n!=9344 && n!=11200)) return 1;
    for (uint32_t i=0;i<m;++i) y[i] = x[0] + float(i);
    return 0;
  });
  std::array<float,16384> input{};
  std::array<float,1861> out{};
  for (uint32_t i=1;i<=128;++i) {
    input[0]=float(i);
    AsyncKey key{slot.submit(1,1,1,input.data(),i%2 ? 9344:11200),1,1,1};
    require(key.token==i);
    require(wait(slot,key,out.data(),1861)==AsyncStatus::complete);
    require(out.front()==float(i) && out.back()==float(i+1860));
  }
  AsyncSlot wrap([](const float*,uint32_t,float* y,uint32_t n){std::fill(y,y+n,0.f);return 0;},UINT32_MAX-1);
  AsyncKey key{wrap.submit(1,1,1,input.data(),9344),1,1,1};
  require(key.token==UINT32_MAX);
  require(wait(wrap,key,out.data(),1861)==AsyncStatus::complete);
  rejects<std::overflow_error>([&]{wrap.submit(1,1,1,input.data(),9344);});
}
int main() {
  lifetime_and_identity(); failures(); shutdown_with_pending_work(); competing_admission(); reuse_and_wrap();
  std::printf("{\"status\":\"passed\",\"assertions\":%u,\"reuse_roundtrips\":128,\"scope\":\"real worker thread with deterministic test callbacks; no model execution\"}\n",checks);
}
