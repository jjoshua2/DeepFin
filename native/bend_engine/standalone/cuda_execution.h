#pragma once
// One synchronous execution slot. CUDA work is internal to the native model
// effect; Bend still owns scheduling and retains its arrays until completion.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace deepfin_native {
class CudaExecution final {
 public:
  CudaExecution(int index, uint32_t batch, uint32_t channels, bool trace);
  ~CudaExecution() noexcept;
  CudaExecution(const CudaExecution&) = delete;
  CudaExecution& operator=(const CudaExecution&) = delete;
  void open(const std::string& path);
  void run(const float* source, uint32_t rows, float* destination);
  const void* trace_data() const;
  std::string audit() const;
 private:
  c10::DeviceIndex index_;
  uint32_t batch_, channels_;
  c10::cuda::CUDAStream stream_;
  cudaEvent_t done_ = nullptr;
  bool failed_ = false;
  at::Tensor host_input_, input_, packed_, host_output_, trace_input_;
  std::vector<at::Tensor> inputs_, outputs_;
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader_;
  void drain() noexcept;
};
}
