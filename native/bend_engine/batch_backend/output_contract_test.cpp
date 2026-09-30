// Small real-ATen tests of the exact packing helper used by the native backend.
#include "../standalone/batch_outputs.h"
#include <array>
#include <iostream>
#include <limits>

static void require(bool ok) {
  if (!ok) throw std::runtime_error("output contract assertion failed");
}
int main() {
  using deepfin_native::copy_batch_outputs;
  const auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  // Transposed storage deliberately exercises the contiguous conversion.
  const auto p = at::arange(4 * 1858, options).view({1858, 4}).transpose(0, 1);
  const auto w = at::arange(12, options).view({3, 4}).transpose(0, 1);
  std::array<float, 8192> dest;
  unsigned valid = 0, invalid = 0;
  for (uint32_t rows : {4u, 1u, 3u, 4u, 1u}) {
    dest.fill(-12345.0f);
    copy_batch_outputs({p, w}, 4, rows, dest.data());
    for (uint32_t row = 0; row < rows; ++row) {
      for (unsigned i = 0; i < 1858; ++i) require(dest[row * 1861 + i] == float(i * 4 + row));
      for (unsigned i = 0; i < 3; ++i) require(dest[row * 1861 + 1858 + i] == float(i * 4 + row));
    }
    for (size_t i = size_t(rows) * 1861; i < dest.size(); ++i) require(dest[i] == -12345.0f);
    ++valid;
  }
  const std::vector<std::vector<at::Tensor>> bad = {
    {}, {p}, {p, w, w}, {w, p}, {p.reshape({4 * 1858}), w},
    {p, w.reshape({2, 6})}, {p, w.to(at::kDouble)}, {p.to(at::kBFloat16), w},
    {p, at::empty({4, 3}, options.device(at::kMeta))}, {at::Tensor(), w},
    {p, at::zeros({4, 4}, options)}, {at::zeros({3, 1858}, options), w}
  };
  for (const auto& outputs : bad) {
    dest.fill(-12345.0f);
    bool rejected = false;
    try { copy_batch_outputs(outputs, 4, 3, dest.data()); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected);
    for (float v : dest) require(v == -12345.0f);
    ++invalid;
  }
  for (const auto [batch, rows] : {std::pair{4u, 0u}, {4u, 5u}, {0u, 1u}, {3u, 1u}, {32u, 1u}}) {
    dest.fill(-12345.0f);
    bool rejected = false;
    try { copy_batch_outputs({p, w}, batch, rows, dest.data()); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected);
    for (float v : dest) require(v == -12345.0f);
    ++invalid;
  }
  bool rejected = false;
  try { copy_batch_outputs({p, w}, 4, 1, nullptr); }
  catch (const std::runtime_error&) { rejected = true; }
  require(rejected); ++invalid;
  std::cout << "{\"status\":\"passed\",\"packed_batches\":" << valid
            << ",\"rejected_contracts\":" << invalid << "}\n";
}
