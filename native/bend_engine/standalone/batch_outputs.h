#pragma once
// Raw CPU output layout only. Callers validate logits/legality before acceptance.
#include <ATen/ATen.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace deepfin_native {
inline void copy_batch_outputs(const std::vector<at::Tensor>& outputs, uint32_t batch,
                               uint32_t rows, void* destination) {
  if (!destination || !rows || rows > batch
      || !(batch == 1 || batch == 2 || batch == 4 || batch == 8 || batch == 16))
    throw std::runtime_error("native batch row/output bound violated");
  if (outputs.size() != 2) throw std::runtime_error("expected (policy, wdl) output tuple");
  // Validate BOTH tensors before changing any caller bytes. Flattened element
  // counts alone cannot distinguish swapped axes, wrong batch or wrong heads.
  for (size_t i = 0; i < 2; ++i) {
    const auto& x = outputs[i];
    if (!x.defined() || !x.device().is_cpu() || x.scalar_type() != at::kFloat || x.dim() != 2
        || x.size(0) != batch || x.size(1) != (i == 0 ? 1858 : 3))
      throw std::runtime_error("native batch output shape/dtype/device mismatch");
  }
  const auto policy = outputs[0].contiguous(), wdl = outputs[1].contiguous();
  for (uint32_t row = 0; row < rows; ++row) {
    auto* out = static_cast<char*>(destination) + size_t(row) * 1861 * sizeof(float);
    std::memcpy(out, static_cast<const char*>(policy.const_data_ptr()) + size_t(row) * 1858 * sizeof(float), 1858 * sizeof(float));
    std::memcpy(out + 1858 * sizeof(float), static_cast<const char*>(wdl.const_data_ptr()) + size_t(row) * 3 * sizeof(float), 3 * sizeof(float));
  }
}
}
