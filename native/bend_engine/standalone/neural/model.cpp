// AOTI tensor execution and byte integrity only. Application logic stays in Bend.
// Synchronous CPU singleton calls; not a batch scheduler or interruptible kernel.
#include "leaf_model_config.h"
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/version.h>
#include <openssl/evp.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <array>
#include <cmath>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
namespace {
struct Fd {
  int value;
  explicit Fd(int fd) : value(fd) { if (fd < 0) throw std::runtime_error("file open failed"); }
  ~Fd() { if (value >= 0) ::close(value); }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
};
void write_all(int fd, const void* data, size_t n) {
  auto p = static_cast<const char*>(data);
  while (n) {
    ssize_t written = ::write(fd, p, n);
    if (written < 0 && errno == EINTR) continue;
    if (written <= 0) throw std::runtime_error("model file/trace write failed");
    p += written; n -= static_cast<size_t>(written);
  }
}
struct Model {
  std::string directory;
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader;
  std::unique_ptr<Fd> trace;
  uint64_t next = 1;
  ~Model() {
    loader.reset();
    if (!directory.empty()) { std::error_code e; std::filesystem::remove_all(directory, e); }
  }
};
std::unique_ptr<Model> model;
void word(int fd, uint32_t v) {
  unsigned char b[4] = {static_cast<unsigned char>(v), static_cast<unsigned char>(v >> 8),
                       static_cast<unsigned char>(v >> 16), static_cast<unsigned char>(v >> 24)};
  write_all(fd, b, 4);
}
std::string verified_copy(Model& m, const char* path) {
  Fd src(::open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  struct stat st;
  if (::fstat(src.value, &st) || !S_ISREG(st.st_mode) || st.st_size <= 0)
    throw std::runtime_error("package must be a nonempty regular file");
  char temp[] = "/tmp/deepfin-bend-model-XXXXXX";
  if (!::mkdtemp(temp)) throw std::runtime_error("cannot create private model workspace");
  m.directory = temp;
  std::string copy = m.directory + "/model.pt2";
  Fd dst(::open(copy.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600));
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> hash(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!hash || EVP_DigestInit_ex(hash.get(), EVP_sha256(), nullptr) != 1)
    throw std::runtime_error("SHA256 initialization failed");
  std::array<char, 65536> bytes;
  uint64_t total = 0;
  for (;;) {
    ssize_t n = ::read(src.value, bytes.data(), bytes.size());
    if (n < 0 && errno == EINTR) continue;
    if (n < 0) throw std::runtime_error("package read failed");
    if (n == 0) break;
    total += static_cast<uint64_t>(n);
    if (total > static_cast<uint64_t>(st.st_size)) throw std::runtime_error("package grew during copy");
    write_all(dst.value, bytes.data(), static_cast<size_t>(n));
    if (EVP_DigestUpdate(hash.get(), bytes.data(), static_cast<size_t>(n)) != 1)
      throw std::runtime_error("SHA256 update failed");
  }
  unsigned char digest[32]; unsigned len = 0;
  if (total != static_cast<uint64_t>(st.st_size) || EVP_DigestFinal_ex(hash.get(), digest, &len) != 1 || len != 32)
    throw std::runtime_error("package hash/length failed");
  std::string hex; const char* digits = "0123456789abcdef";
  for (auto c : digest) { hex += digits[c >> 4]; hex += digits[c & 15]; }
  if (hex != DF_PACKAGE_SHA256) throw std::runtime_error("package SHA256 differs from build-bound identity");
  // The loader sees only the exact copied bytes, even if the supplied path changes.
  return copy;
}
void copy_head(const at::Tensor& tensor, uint32_t width, uint32_t* out) {
  if (!tensor.device().is_cpu() || tensor.scalar_type() != at::kFloat || tensor.dim() != 2
      || tensor.size(0) != DF_MODEL_BATCH || tensor.size(1) != width)
    throw std::runtime_error("model output shape/dtype/device mismatch");
  auto t = tensor.contiguous(); const float* values = t.data_ptr<float>();
  for (uint32_t i = 0; i < DF_MODEL_BATCH * width; ++i)
    if (!std::isfinite(values[i])) throw std::runtime_error("nonfinite model output");
  std::memcpy(out, values, width * sizeof(float));
}
} // namespace
extern "C" uint32_t df_leaf_open(uint32_t out[8]) {
  try {
    if (model) throw std::runtime_error("model already opened");
    if (std::string(TORCH_VERSION) != DF_TORCH_RELEASE) throw std::runtime_error("LibTorch release mismatch");
    const char* path = std::getenv("DEEPFIN_BEND_PACKAGE");
    if (!path || !*path) throw std::runtime_error("DEEPFIN_BEND_PACKAGE is required");
    auto next = std::make_unique<Model>();
    auto copy = verified_copy(*next, path);
    if (::setenv("AOTI_RUNTIME_CHECK_INPUTS", "1", 1)) throw std::runtime_error("cannot enable AOTI input guards");
    at::set_num_threads(2); at::set_num_interop_threads(1);
    c10::InferenceMode mode;
    next->loader = std::make_unique<torch::inductor::AOTIModelPackageLoader>(copy, "model", false, 1, -1);
    const char* trace = std::getenv("DEEPFIN_BEND_TRACE");
    if (trace && *trace) next->trace = std::make_unique<Fd>(::open(trace, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600));
    const uint32_t fields[] = {1, DF_MODEL_HISTORY, DF_MODEL_FEATURES, DF_MODEL_CHANNELS,
                              DF_MODEL_BATCH, 1858, 1, next->trace ? 1u : 0u};
    std::memcpy(out, fields, sizeof(fields)); model = std::move(next); return 8;
  } catch (const std::exception& e) { std::cerr << "Bend native model: " << e.what() << '\n'; model.reset(); return 0; }
}
extern "C" uint32_t df_leaf_infer(uint32_t sequence, uint32_t count,
                                  const uint32_t* input, uint32_t out[1861]) {
  try {
    if (!model || sequence != model->next || count != DF_MODEL_CHANNELS * 64u)
      throw std::runtime_error("model input identity/size mismatch");
    c10::InferenceMode mode;
    auto x = at::zeros({DF_MODEL_BATCH, DF_MODEL_CHANNELS, 8, 8}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    float* data = x.data_ptr<float>();
    std::memcpy(data, input, count * sizeof(float));
    for (uint32_t i = 0; i < count; ++i)
      if (!std::isfinite(data[i])) throw std::runtime_error("nonfinite model input");
    auto result = model->loader->run(std::vector<at::Tensor>{x});
    if (result.size() != 2) throw std::runtime_error("expected policy/WDL tuple");
    copy_head(result[0], 1858, out); copy_head(result[1], 3, out + 1858);
    if (model->trace) {
      int fd = model->trace->value;
      for (uint32_t w : {0x44464c31u, sequence, DF_MODEL_CHANNELS, DF_MODEL_BATCH, count, 1861u}) word(fd, w);
      for (uint32_t i = 0; i < count; ++i) word(fd, input[i]);
      for (uint32_t i = 0; i < 1861; ++i) word(fd, out[i]);
    }
    ++model->next; return 1861;
  } catch (const std::exception& e) { std::cerr << "Bend native model: " << e.what() << '\n'; model.reset(); return 0; }
}
extern "C" void df_leaf_close(void) { model.reset(); }
