/*
 * Native AOTInductor bridge for the Bend architecture probe.
 *
 * Loads a .pt2 package with PyTorch's supported C++ AOTIModelPackageLoader.
 * No Python interpreter participates at runtime. CI uses a tiny CPU package;
 * the loader API is the same one used for CUDA packages.
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

namespace {

std::mutex g_loader_mutex;
std::unique_ptr<torch::inductor::AOTIModelPackageLoader> g_loader;
std::string g_loaded_path;

torch::inductor::AOTIModelPackageLoader* get_loader() {
  const char* raw_path = std::getenv("DEEPFIN_AOTI_PROBE_PACKAGE");
  if (raw_path == nullptr || raw_path[0] == '\0') {
    std::cerr << "DEEPFIN_AOTI_PROBE_PACKAGE is not set\n";
    return nullptr;
  }

  const std::string path(raw_path);
  std::lock_guard<std::mutex> lock(g_loader_mutex);
  if (!g_loader || g_loaded_path != path) {
    g_loader = std::make_unique<torch::inductor::AOTIModelPackageLoader>(path);
    g_loaded_path = path;
  }
  return g_loader.get();
}

torch::Tensor make_input(uint32_t seed) {
  auto x = torch::empty(
    {2, 4},
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
  );
  float* data = x.data_ptr<float>();
  for (int i = 0; i < 8; ++i) {
    // Small exact integers keep x*2+1 bit-exact across eager and AOTI.
    data[i] = static_cast<float>(static_cast<int64_t>(seed) * 8 + i - 7);
  }
  return x;
}

}  // namespace

extern "C" uint32_t deepfin_aoti_probe_eval(
  uint32_t seed,
  uint32_t* out_bits,
  uint32_t capacity
) {
  if (out_bits == nullptr || capacity == 0) return 0;

  try {
    c10::InferenceMode mode;
    auto* loader = get_loader();
    if (loader == nullptr) return 0;

    std::vector<at::Tensor> inputs = {make_input(seed)};
    std::vector<at::Tensor> outputs = loader->run(inputs);
    if (outputs.size() != 1) {
      std::cerr << "AOTI probe expected one output, got " << outputs.size() << "\n";
      return 0;
    }

    at::Tensor out = outputs[0].to(torch::kCPU).contiguous();
    if (out.scalar_type() != at::kFloat) {
      std::cerr << "AOTI probe expected float32 output\n";
      return 0;
    }

    const uint32_t count = static_cast<uint32_t>(out.numel());
    if (count > capacity) {
      std::cerr << "AOTI probe output exceeds bridge capacity\n";
      return 0;
    }

    const float* values = out.data_ptr<float>();
    for (uint32_t i = 0; i < count; ++i) {
      static_assert(sizeof(float) == sizeof(uint32_t));
      std::memcpy(&out_bits[i], &values[i], sizeof(uint32_t));
    }
    return count;
  } catch (const std::exception& exc) {
    std::cerr << "AOTI probe failed: " << exc.what() << "\n";
    return 0;
  }
}
