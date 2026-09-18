/* CPU AOTI evaluator for the experimental session adapter. The accepted package
 * is an explicit (compact-policy logits, WDL logits) tuple, batch one. Python
 * exports/tests the package; this library does not link a Python interpreter. */
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

struct Evaluator {
  torch::inductor::AOTIModelPackageLoader loader;
  uint32_t planes;
  bool bf16;
  std::mutex mutex;
  Evaluator(const char* path, uint32_t planes_, bool bf16_)
      : loader(path), planes(planes_), bf16(bf16_) {}
};
static int error_out(char* error, uint32_t size, const char* text) {
  if (error && size) std::snprintf(error, size, "%s", text);
  return 0;
}
extern "C" void* df_aoti_open(const char* path, const char* output_spec,
    uint32_t planes, uint32_t bf16, char* error, uint32_t size) {
  try {
    if (!path || !output_spec || (planes != 146 && planes != 175 && planes != 179) || bf16 > 1)
      throw std::runtime_error("invalid CPU AOTI configuration");
    const char* guards = std::getenv("AOTI_RUNTIME_CHECK_INPUTS");
    if (!guards || std::strcmp(guards, "1") != 0)
      throw std::runtime_error("AOTI_RUNTIME_CHECK_INPUTS=1 is required before loading packages");
    at::set_num_threads(2);
    auto e = std::make_unique<Evaluator>(path, planes, bf16 != 0);
    const auto spec = e->loader.get_call_spec();
    if (spec.size() != 2 || spec[1] != output_spec)
      throw std::runtime_error("AOTI output treespec does not match the tuple contract");
    return e.release();
  } catch (const std::exception& ex) { error_out(error, size, ex.what()); return nullptr; }
}
extern "C" void df_aoti_close(void* handle) { delete static_cast<Evaluator*>(handle); }
extern "C" int df_aoti_run(void* handle, const float* planes, uint32_t plane_count,
    const uint32_t* actions, uint32_t count, float* policy_logits, float* wdl_logits,
    float* priors, float* wdl, char* error, uint32_t size) {
  try {
    auto* e = static_cast<Evaluator*>(handle);
    if (!e || !planes || !actions || !policy_logits || !wdl_logits || !priors || !wdl ||
        count == 0 || count > 256 || plane_count != e->planes * 64)
      throw std::runtime_error("invalid AOTI input buffers");
    std::lock_guard<std::mutex> guard(e->mutex);
    c10::InferenceMode mode;
    std::vector<int64_t> ids(actions, actions + count);
    if (std::any_of(ids.begin(), ids.end(), [](int64_t x) { return x >= 1858; }))
      throw std::runtime_error("compact policy index out of range");
    auto input = at::from_blob(const_cast<float*>(planes), {1, e->planes, 8, 8}, at::kFloat).clone();
    if (!at::isfinite(input).all().item<bool>()) throw std::runtime_error("nonfinite neural input");
    if (e->bf16) input = input.to(at::kBFloat16);
    auto outputs = e->loader.run({input});
    if (outputs.size() != 2 || outputs[0].sizes() != at::IntArrayRef({1, 1858}) ||
        outputs[1].sizes() != at::IntArrayRef({1, 3}) ||
        !outputs[0].device().is_cpu() || !outputs[1].device().is_cpu() ||
        !outputs[0].is_floating_point() || !outputs[1].is_floating_point())
      throw std::runtime_error("AOTI output shape/device violates compact-policy/WDL contract");
    auto policy = outputs[0].to(at::kFloat).contiguous();
    auto value = outputs[1].to(at::kFloat).contiguous();
    if (!at::isfinite(policy).all().item<bool>() || !at::isfinite(value).all().item<bool>())
      throw std::runtime_error("nonfinite AOTI output");
    auto index = at::from_blob(ids.data(), {static_cast<int64_t>(count)}, at::kLong);
    auto probabilities = policy.index_select(1, index).softmax(1).contiguous();
    auto values = value.softmax(1).contiguous();
    /* Nothing caller-visible is committed before successful validation. */
    std::memcpy(policy_logits, policy.data_ptr<float>(), 1858 * sizeof(float));
    std::memcpy(wdl_logits, value.data_ptr<float>(), 3 * sizeof(float));
    std::memcpy(priors, probabilities.data_ptr<float>(), count * sizeof(float));
    std::memcpy(wdl, values.data_ptr<float>(), 3 * sizeof(float));
    return 1;
  } catch (const std::exception& ex) { return error_out(error, size, ex.what()); }
}
