// CPU F32 / optional CUDA BF16 external evaluator. No Python runtime, no chess/search logic.
// Private wire format: little-endian U32 headers and IEEE F32 tensor payloads.
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <c10/core/DeviceGuard.h>
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

constexpr uint32_t MAGIC = 0x44464e31;
static void read_exact(void* data, size_t size) {
  if (!std::cin.read(static_cast<char*>(data), static_cast<std::streamsize>(size)))
    throw std::runtime_error("truncated evaluator input");
}
static uint32_t read_word() {
  unsigned char b[4]; read_exact(b, 4);
  return uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24;
}
static void word(uint32_t x) {
  unsigned char b[4] = {static_cast<unsigned char>(x), static_cast<unsigned char>(x >> 8),
                       static_cast<unsigned char>(x >> 16), static_cast<unsigned char>(x >> 24)};
  std::cout.write(reinterpret_cast<char*>(b), 4);
}
static void tensor(const at::Tensor& output, uint32_t batch, uint32_t size, const at::Device& device) {
  if (output.device() != device || output.scalar_type() != at::kFloat
      || output.dim() != 2 || output.size(0) != batch || output.size(1) != size)
    throw std::runtime_error("wrong evaluator output shape, dtype or device");
  auto values = output.to(at::kCPU).contiguous(); // blocking copy before publishing a reply
  for (uint32_t i = 0; i < batch * size; ++i) {
    float value = values.data_ptr<float>()[i];
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite evaluator output");
    uint32_t bits; std::memcpy(&bits, &value, 4); word(bits);
  }
}
int main(int argc, char** argv) {
  try {
    if ((argc != 3 && argc != 4 && argc != 6) || (std::string(argv[2]) != "146" && std::string(argv[2]) != "175"))
      throw std::runtime_error("usage: aoti_worker PACKAGE.pt2 {146|175} [1|2|4|8|16 [cpu-float32|cuda-bfloat16 INDEX]]");
    const std::string batch_arg = argc >= 4 ? argv[3] : "1";
    if (batch_arg != "1" && batch_arg != "2" && batch_arg != "4" && batch_arg != "8" && batch_arg != "16")
      throw std::runtime_error("unsupported fixed evaluator batch");
    const uint32_t batch = std::stoul(batch_arg);
    const uint32_t channels = std::stoul(argv[2]), count = batch * channels * 64;
    const std::string target = argc == 6 ? argv[4] : "cpu-float32";
    const std::string index_arg = argc == 6 ? argv[5] : "0";
    if (index_arg.empty() || index_arg.size() > 3 || index_arg.find_first_not_of("0123456789") != std::string::npos)
      throw std::runtime_error("invalid evaluator device index");
    const auto index = std::stoul(index_arg);
    if (index > 127 || (target == "cpu-float32" && index != 0))
      throw std::runtime_error("invalid evaluator device index");
    if (target != "cpu-float32" && target != "cuda-bfloat16")
      throw std::runtime_error("unsupported evaluator device/dtype");
    const bool cuda = target == "cuda-bfloat16";
    if (cuda && !at::hasCUDA()) throw std::runtime_error("CUDA evaluator requires a CUDA LibTorch build");
    const at::Device device = cuda ? at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(index)) : at::Device(at::kCPU);
    c10::DeviceGuard guard(device);
    at::set_num_threads(2);
    at::set_num_interop_threads(1);
    c10::InferenceMode inference;
    torch::inductor::AOTIModelPackageLoader loader(argv[1], "model", false, 1, cuda ? static_cast<c10::DeviceIndex>(index) : -1);
    word(MAGIC); word(0); std::cout.flush();
    uint32_t sequence = 1;
    for (unsigned commands = 0; commands < 65536; ++commands) {
      if (read_word() != MAGIC) throw std::runtime_error("wrong evaluator input magic");
      uint32_t request = read_word(), n = read_word();
      if (request == 0 && n == 0) return 0;
      if (request != sequence++ || n != count) throw std::runtime_error("wrong evaluator sequence or input size");
      auto x = at::empty({batch, channels, 8, 8}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
      for (uint32_t i = 0; i < n; ++i) {
        uint32_t bits = read_word(); float v; std::memcpy(&v, &bits, 4);
        if (!std::isfinite(v)) throw std::runtime_error("nonfinite evaluator input");
        x.data_ptr<float>()[i] = v;
      }
      if (cuda) x = x.to(device, at::kBFloat16);
      std::vector<at::Tensor> inputs = {x};
      auto outputs = loader.run(inputs);
      if (outputs.size() != 2) throw std::runtime_error("expected tuple (policy_logits, wdl_logits)");
      // This endpoint intentionally only accepts the exported compact-policy wrapper.
      // Output order/encoding is pinned by the checked sidecar manifest.
      word(MAGIC); word(request); word(batch * 1858); word(batch * 3);
      tensor(outputs[0], batch, 1858, device); tensor(outputs[1], batch, 3, device);
      std::cout.flush();
      if (!std::cout) throw std::runtime_error("evaluator output failed");
    }
    throw std::runtime_error("evaluator command limit");
  } catch (const std::exception& e) {
    std::cerr << "aoti worker: " << e.what() << '\n'; return 2;
  }
}
