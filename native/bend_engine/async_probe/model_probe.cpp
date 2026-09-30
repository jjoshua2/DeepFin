// Real CPU singleton model execution through the async adapter, with a separate
// direct-call control process. No chess, search scheduler or artificial model.
#include "model_contract.h"
#include "../standalone/async_slot.h"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>

extern "C" uint32_t deepfin_model_open();
extern "C" int deepfin_model_run(const float*, uint32_t, float*, uint32_t);
extern "C" uint32_t deepfin_async_submit(uint32_t, uint32_t, uint32_t, const float*, uint32_t);
extern "C" uint32_t deepfin_async_poll(uint32_t, uint32_t, uint32_t, uint32_t, float*, uint32_t);
extern "C" uint32_t deepfin_async_cancel(uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" void deepfin_async_shutdown();

namespace {
constexpr uint32_t count = DEEPFIN_MODEL_CHANNELS * 64;
constexpr uint32_t width = 1861;
constexpr uint32_t steps = 6;
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void unchanged(const std::array<float,2048>& output, uint32_t begin) {
  require(std::all_of(output.begin()+begin,output.end(),[](float x){return x == -19.f;}),
          "caller output changed outside successful publication");
}
void prepare(std::array<float,16384>& input, uint32_t step) {
  // Repeated input 1 is separated by different inputs and a cancelled evaluation.
  const uint32_t pattern = step == 5 ? 1 : step;
  input.fill(91.f);  // unused host capacity is deliberately not model input
  for (uint32_t i=0;i<count;++i) {
    const auto integer = int32_t((i*17+pattern*13)%257)-128;
    input[i] = float(integer) / 128.f;
  }
}
struct Join {
  bool enabled;
  ~Join() { if (enabled) deepfin_async_shutdown(); }
};
void run(bool async) {
  static_assert(!DEEPFIN_MODEL_CUDA && DEEPFIN_MODEL_BATCH == 1);
  require(deepfin_model_open() == DEEPFIN_MODEL_PROFILE, "model startup failed");
  Join join{async};  // join before global model destruction, also on exceptions
  std::array<float,16384> input{};
  std::array<float,2048> output{};
  uint32_t prior = 0, cancellations = 0, complete = 0;
  for (uint32_t step=1;step<=steps;++step) {
    prepare(input, step); output.fill(-19.f);
    if (async) {
      const uint32_t token = deepfin_async_submit(step,step+9,step+19,input.data(),count);
      require(token > prior, "token recycled");
      require(deepfin_async_poll(prior,step,step+9,step+19,output.data(),width) == 0,
              "old token consumed new request");
      require(deepfin_async_poll(token,step+1,step+9,step+19,output.data(),width) == 0,
              "wrong epoch consumed request");
      require(deepfin_async_poll(token,step,step+10,step+19,output.data(),width) == 0,
              "wrong request consumed request");
      require(deepfin_async_poll(token,step,step+9,step+20,output.data(),width) == 0,
              "wrong node consumed request");
      require(!deepfin_async_cancel(token,step+1,step+9,step+19), "wrong epoch cancelled request");
      unchanged(output,0);
      input.fill(99.f); // async worker must use the admission snapshot
      const bool cancel = step == 3;
      if (cancel) require(deepfin_async_cancel(token,step,step+9,step+19),"cancellation rejected");
      const auto deadline = std::chrono::steady_clock::now()+std::chrono::seconds(20);
      uint32_t status;
      do {
        status=deepfin_async_poll(token,step,step+9,step+19,output.data(),width);
        if (status != 1) break;
        unchanged(output,0);
        require(std::chrono::steady_clock::now()<deadline,"model completion timed out");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } while (true);
      require(status == (cancel ? 3u:2u),"unexpected model disposition");
      require(deepfin_async_poll(token,step,step+9,step+19,output.data(),width) == 0,
              "duplicate completion consumed twice");
      require(!deepfin_async_cancel(token,step,step+9,step+19),"consumed request cancelled again");
      prior=token;
      if (cancel) { unchanged(output,0); ++cancellations; std::printf("cancelled %u\n",step); continue; }
    } else {
      require(deepfin_model_run(input.data(),count,output.data(),width)==0,"direct model failed");
    }
    unchanged(output,width); ++complete;
    std::printf("result %u",step);
    for (uint32_t i=0;i<width;++i) std::printf(" %08x",std::bit_cast<uint32_t>(output[i]));
    std::putchar('\n');
  }
  std::printf("done %u %u\n",complete,cancellations);
}
}
int main(int argc, char** argv) {
  try {
    require(argc == 2 && (!std::strcmp(argv[1],"sync") || !std::strcmp(argv[1],"async")),
            "usage: native-async-model-probe sync|async");
    run(!std::strcmp(argv[1],"async"));
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr,"async model probe: %s\n",error.what()); return 2;
  }
}
