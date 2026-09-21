// Dense tensor/model infrastructure only. No board, moves, policy gathering,
// WDL interpretation or search logic. No Python API or process spawning.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/version.h>
#include <openssl/evp.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "model_contract.h"

namespace {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader;
uint32_t next_sequence = 1;
bool poisoned = false;
std::string sha256(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read model package");
    auto ctx = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("SHA256 initialization failed");
    std::array<char, 65536> bytes;
    while (input.read(bytes.data(), bytes.size()) || input.gcount())
        if (EVP_DigestUpdate(ctx.get(), bytes.data(), size_t(input.gcount())) != 1)
            throw std::runtime_error("SHA256 update failed");
    if (!input.eof()) throw std::runtime_error("model package read failed");
    std::array<unsigned char, 32> hash; unsigned size = 0;
    if (EVP_DigestFinal_ex(ctx.get(), hash.data(), &size) != 1 || size != 32)
        throw std::runtime_error("SHA256 finalization failed");
    std::ostringstream text; text << std::hex << std::setfill('0');
    for (auto byte : hash) text << std::setw(2) << unsigned(byte);
    return text.str();
}
void check_output(const at::Tensor &value, int width) {
    if (!value.device().is_cpu() || value.scalar_type() != at::kFloat || value.dim() != 2
        || value.size(0) != MODEL_BATCH || value.size(1) != width)
        throw std::runtime_error("model output shape, dtype or device mismatch");
}
}
extern "C" uint32_t deepfin_model_open(const char *path) {
    try {
        if (loader || poisoned) throw std::runtime_error("model is already opened or failed");
        if (std::string(TORCH_VERSION) != MODEL_TORCH_VERSION)
            throw std::runtime_error("LibTorch version differs from model build contract");
        if (sha256(path) != MODEL_SHA256) throw std::runtime_error("model package SHA256 mismatch");
        at::set_num_threads(2); at::set_num_interop_threads(1);
        c10::InferenceMode mode;
        loader = std::make_unique<torch::inductor::AOTIModelPackageLoader>(path, "model", false, 1, -1);
        // Package must be trusted and immutable. This detects ordinary replacement,
        // not adversarial concurrent mutation or a substitute for authentication.
        if (sha256(path) != MODEL_SHA256) throw std::runtime_error("model package changed while loading");
        return 0;
    } catch (const std::exception &ex) {
        poisoned = true; loader.reset();
        std::cerr << "Bend native model: " << ex.what() << '\n'; return 2;
    }
}
extern "C" uint32_t deepfin_model_input_count() { return MODEL_CHANNELS * 64; }
extern "C" void deepfin_model_spec(uint32_t words[4]) {
    words[0] = MODEL_LAYOUT; words[1] = MODEL_FEATURES;
    words[2] = MODEL_CHANNELS; words[3] = MODEL_BATCH;
}
extern "C" uint32_t deepfin_model_run(uint32_t sequence, const float *in, uint32_t n, float *out) {
    try {
        if (!loader || poisoned || sequence != next_sequence || sequence == UINT32_MAX
            || n != MODEL_CHANNELS * 64) throw std::runtime_error("invalid native model call state");
        for (uint32_t i = 0; i < n; ++i)
            if (!std::isfinite(in[i])) throw std::runtime_error("nonfinite model input");
        c10::InferenceMode mode;
        // Single real row; other rows are explicit zero padding, not batched games.
        auto x = at::zeros({MODEL_BATCH, MODEL_CHANNELS, 8, 8}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
        std::memcpy(x.data_ptr<float>(), in, n * sizeof(float));
        auto outputs = loader->run(std::vector<at::Tensor>{x});
        if (outputs.size() != 2) throw std::runtime_error("expected compact-policy and WDL logits");
        check_output(outputs[0], 1858); check_output(outputs[1], 3);
        auto policy = outputs[0].contiguous(); auto wdl = outputs[1].contiguous();
        std::memcpy(out, policy.data_ptr<float>(), 1858 * sizeof(float));
        std::memcpy(out + 1858, wdl.data_ptr<float>(), 3 * sizeof(float));
        // Bend checks all consumed logits (including illegal slots) before softmax.
        ++next_sequence;
        return 1861;
    } catch (const std::exception &ex) {
        poisoned = true;
        std::cerr << "Bend native model: " << ex.what() << '\n'; return 0;
    }
}
extern "C" void deepfin_model_close() { loader.reset(); poisoned = true; }
