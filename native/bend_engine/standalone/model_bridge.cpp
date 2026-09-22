// Fixed CPU tensor execution only. All board/history/encoding, legal masking,
// probabilities, ticket ownership and search decisions live in Bend.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/version.h>
#include <openssl/evp.h>
#include "model_contract.h"
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace {
static_assert(sizeof(float) == 4 && std::endian::native == std::endian::little);
struct Runtime {
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader;
  std::filesystem::path workspace;
  // One synchronous caller owns this input for the lifetime of the bound model.
  // AOTI output tensors remain loader-owned; results are copied into Bend's
  // distinct reusable output buffer before either side can reuse its storage.
  at::Tensor input;
  std::vector<at::Tensor> inputs;
  int trace = -1;
  uint32_t calls = 0;
  bool audit = false;
  const void* first_input = nullptr;
  const void* first_output = nullptr;
  uint32_t input_changes = 0, output_changes = 0, input_tensor_allocations = 0;
  ~Runtime() {
    if (audit && loader) {
      std::cerr << "native-buffer-audit calls=" << calls << " input_changes=" << input_changes
                << " output_changes=" << output_changes << " input_tensor_allocations=" << input_tensor_allocations << '\n';
    }
    loader.reset();
    if (trace >= 0) ::close(trace);
    if (!workspace.empty()) { std::error_code ec; std::filesystem::remove_all(workspace, ec); }
  }
};
Runtime state;
void write_all(int fd, const void* data, size_t n) {
  const char* p = static_cast<const char*>(data);
  while (n) {
    const auto written = ::write(fd, p, n);
    if (written < 0 && errno == EINTR) continue;
    if (written <= 0) throw std::runtime_error("native model trace write failed");
    p += written; n -= static_cast<size_t>(written);
  }
}
std::string copy_verified(const char* source) {
  const char* tmp = std::getenv("TMPDIR");
  std::string pattern = std::string(tmp && *tmp ? tmp : "/tmp") + "/deepfin-model.XXXXXX";
  std::vector<char> name(pattern.begin(), pattern.end()); name.push_back(0);
  if (!mkdtemp(name.data())) throw std::runtime_error("cannot create private model workspace");
  state.workspace = name.data();
  const auto dest = state.workspace / "bound.pt2";
  // One source descriptor, private copy, and hash of precisely the bytes copied.
  std::ifstream input(source, std::ios::binary);
  std::ofstream output(dest, std::ios::binary | std::ios::trunc);
  if (!input || !output) throw std::runtime_error("cannot read/copy bound model package");
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1) throw std::runtime_error("SHA256 initialization failed");
  std::array<char, 65536> buf;
  while (input) {
    input.read(buf.data(), buf.size());
    const auto n = input.gcount();
    if (n) {
      output.write(buf.data(), n);
      if (!output || EVP_DigestUpdate(ctx.get(), buf.data(), static_cast<size_t>(n)) != 1)
        throw std::runtime_error("package copy/hash failed");
    }
  }
  if (!input.eof()) throw std::runtime_error("model package read failed");
  output.close();
  if (!output) throw std::runtime_error("model package close failed");
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest;
  unsigned int size = 0;
  if (EVP_DigestFinal_ex(ctx.get(), digest.data(), &size) != 1 || size != 32)
    throw std::runtime_error("model package hash failed");
  std::ostringstream hex;
  for (unsigned int i = 0; i < size; ++i) hex << std::hex << std::setw(2) << std::setfill('0') << unsigned(digest[i]);
  if (hex.str() != DEEPFIN_MODEL_SHA256) throw std::runtime_error("bound model package hash mismatch");
  return dest.string();
}
void copy_output(const at::Tensor& x, int64_t width, void* destination) {
  if (!x.device().is_cpu() || x.scalar_type() != at::kFloat || x.dim() != 2
      || x.size(0) != 1 || x.size(1) != width)
    throw std::runtime_error("native model output shape/dtype/device mismatch");
  const auto dense = x.contiguous();
  std::memcpy(destination, dense.const_data_ptr(), static_cast<size_t>(width) * sizeof(float));
}
}
extern "C" uint32_t deepfin_model_open() {
  try {
    if (state.loader) throw std::runtime_error("model is already open");
    if (std::string(TORCH_VERSION) != DEEPFIN_MODEL_TORCH_VERSION)
      throw std::runtime_error("bound package/LibTorch version mismatch");
    const char* source = std::getenv("DEEPFIN_BEND_MODEL_PACKAGE");
    if (!source || !*source) throw std::runtime_error("DEEPFIN_BEND_MODEL_PACKAGE is required; no material fallback");
    const auto path = copy_verified(source);
    at::set_num_threads(2);
    at::set_num_interop_threads(1);
    state.loader = std::make_unique<torch::inductor::AOTIModelPackageLoader>(path);
    state.input = at::empty({1, DEEPFIN_MODEL_CHANNELS, 8, 8},
                           at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    ++state.input_tensor_allocations;
    state.inputs = {state.input};
    const char* audit = std::getenv("DEEPFIN_BEND_BUFFER_AUDIT");
    if (audit && *audit && std::strcmp(audit, "0") && std::strcmp(audit, "1"))
      throw std::runtime_error("DEEPFIN_BEND_BUFFER_AUDIT must be 0 or 1");
    state.audit = audit && std::strcmp(audit, "1") == 0;
    const char* trace = std::getenv("DEEPFIN_BEND_MODEL_TRACE");
    if (trace && *trace) {
      state.trace = ::open(trace, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
      if (state.trace < 0) throw std::runtime_error("trace destination must be a NEW file");
    }
    return DEEPFIN_MODEL_PROFILE;
  } catch (const std::exception& error) {
    std::cerr << "native model startup: " << error.what() << '\n'; return 0;
  }
}
extern "C" int deepfin_model_run(const float* input, uint32_t count, float* output, uint32_t capacity) {
  try {
    if (!state.loader || !input || !output || count != DEEPFIN_MODEL_CHANNELS * 64 || capacity != 1861
        || state.calls >= 65536) throw std::runtime_error("native model input/call bound violated");
    if (state.audit) {
      if (!state.calls) { state.first_input = input; state.first_output = output; }
      state.input_changes += state.first_input != input;
      state.output_changes += state.first_output != output;
    }
    c10::InferenceMode inference;
    std::memcpy(state.input.mutable_data_ptr(), input, static_cast<size_t>(count) * sizeof(float));
    auto outputs = state.loader->run(state.inputs);
    if (outputs.size() != 2) throw std::runtime_error("expected (policy, wdl) output tuple");
    copy_output(outputs[0], 1858, output);
    copy_output(outputs[1], 3, reinterpret_cast<char*>(output) + 1858 * sizeof(float));
    ++state.calls;
    if (state.trace >= 0) {
      const uint32_t header[] = {0x44464c31, state.calls, count, 1861};
      write_all(state.trace, header, sizeof header);
      write_all(state.trace, input, count * sizeof(float));
      write_all(state.trace, output, 1861 * sizeof(float));
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "native model execution: " << error.what() << '\n'; return 1;
  }
}
