#pragma once
// Host-side conversion only; no chess or scheduling. Shared by CPU contract
// tests and the CUDA runtime's reusable pinned BF16 staging buffer.
#include <ATen/ATen.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace deepfin_native {
inline void stage_bfloat16_inputs(const float* source, uint32_t batch, uint32_t rows,
                                 uint32_t channels, at::Tensor& staging) {
  if (!source || !rows || rows > batch || (channels != 146 && channels != 175)
      || !(batch == 1 || batch == 2 || batch == 4 || batch == 8 || batch == 16)
      || !staging.defined() || !staging.device().is_cpu()
      || staging.scalar_type() != at::kBFloat16 || !staging.is_contiguous()
      || staging.sizes() != at::IntArrayRef({batch, channels, 8, 8}))
    throw std::runtime_error("invalid BF16 staging contract");
  // Clear physical padding after EVERY full/partial call. Bend's source tail
  // can be NaN or stale; never read it. Conversion is round-to-nearest-even.
  staging.zero_();
  auto* target = staging.data_ptr<c10::BFloat16>();
  for (size_t i = 0; i < size_t(rows) * channels * 64; ++i) {
    const c10::BFloat16 converted(source[i]);
    if (!std::isfinite(source[i]) || !std::isfinite(float(converted)))
      throw std::runtime_error("nonfinite or overflowing BF16 input");
    target[i] = converted;
  }
}
}
