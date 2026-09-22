// CPU execution of the same casting/padding helper used by the CUDA slot.
// This tests data contracts, NOT GPU launches, streams or DMA completion.
#include "../standalone/batch_inputs.h"
#include "../standalone/batch_outputs.h"
#include <array>
#include <iostream>
#include <limits>
#include <vector>

static void require(bool ok) {
  if (!ok) throw std::runtime_error("BF16 staging assertion failed");
}
int main() {
  using deepfin_native::stage_bfloat16_inputs;
  const auto bf = at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU);
  unsigned valid = 0, invalid = 0;
  for (uint32_t batch : {1u, 2u, 4u, 8u, 16u}) for (uint32_t channels : {146u, 175u}) {
    auto staging = at::empty({batch, channels, 8, 8}, bf);
    const auto* original = staging.const_data_ptr();
    std::vector<float> source(size_t(batch) * channels * 64);
    for (uint32_t rows : {batch, 1u, batch, 1u}) {
      std::fill(source.begin(), source.end(), std::numeric_limits<float>::quiet_NaN());
      const size_t real = size_t(rows) * channels * 64;
      for (size_t i = 0; i < real; ++i) source[i] = float(int(i % 71) - 35) / 71.0f;
      stage_bfloat16_inputs(source.data(), batch, rows, channels, staging);
      require(original == staging.const_data_ptr());
      const auto* data = staging.const_data_ptr<c10::BFloat16>();
      for (size_t i = 0; i < real; ++i) require(data[i].x == c10::BFloat16(source[i]).x);
      for (size_t i = real; i < source.size(); ++i) require(data[i].x == 0);
      ++valid;
    }
  }
  auto staging = at::empty({4, 146, 8, 8}, bf);
  std::vector<float> source(4 * 146 * 64, 0.0f);
  const std::array<float, 5> values = {1.00390625f, 1.01171875f, -0.0f, 0.0f, -1.00390625f};
  const std::array<uint16_t, 5> bits = {0x3f80, 0x3f82, 0x8000, 0, 0xbf80};
  std::copy(values.begin(), values.end(), source.begin());
  stage_bfloat16_inputs(source.data(), 4, 1, 146, staging);
  for (size_t i = 0; i < bits.size(); ++i) require(staging.const_data_ptr<c10::BFloat16>()[i].x == bits[i]);
  ++valid;
  for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::max()}) {
    source[0] = bad;
    bool rejected = false;
    try { stage_bfloat16_inputs(source.data(), 4, 1, 146, staging); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected); ++invalid;
  }
  source[0] = 1.0f;
  for (const auto [batch, rows, channels] : {std::array{4u,0u,146u}, {4u,5u,146u}, {3u,1u,146u}, {4u,1u,147u}}) {
    bool rejected = false;
    try { stage_bfloat16_inputs(source.data(), batch, rows, channels, staging); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected); ++invalid;
  }
  for (auto bad : {at::Tensor(), at::empty({4,146,8,8}), at::empty({4,146,8,8},bf.device(at::kMeta)),
                  at::empty({4,146,8,8},bf).transpose(2,3), at::empty({4,146,64},bf)}) {
    bool rejected = false;
    try { stage_bfloat16_inputs(source.data(), 4, 1, 146, bad); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected); ++invalid;
  }
  bool rejected = false;
  try { stage_bfloat16_inputs(nullptr, 4, 1, 146, staging); }
  catch (const std::runtime_error&) { rejected = true; }
  require(rejected); ++invalid;
  // An otherwise valid CPU output pair must not satisfy a CUDA device contract.
  for (int index : {0, 1, 127}) {
    rejected = false;
    try { deepfin_native::validate_batch_outputs({at::empty({4,1858}),at::empty({4,3})}, 4,
                                                at::Device(at::kCUDA, index)); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected); ++invalid;
  }
  std::cout << "{\"status\":\"passed\",\"scope\":\"CPU data contracts only\",\"valid\":" << valid
            << ",\"rejected\":" << invalid << "}\n";
}
