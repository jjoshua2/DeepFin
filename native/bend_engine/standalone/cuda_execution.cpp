#include "cuda_execution.h"
#include "batch_inputs.h"
#include "batch_outputs.h"
#include "model_contract.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cstring>
#include <sstream>
#include <stdexcept>

// Deliberately require the bound release's CUDA major/minor at build time.
// Patch releases/driver compatibility and numerical behavior need qualification.
static_assert(!DEEPFIN_MODEL_CUDA || DEEPFIN_MODEL_CUDA_RUNTIME == CUDART_VERSION,
              "bound package CUDA version differs from build toolkit");
namespace deepfin_native {
namespace {
c10::DeviceIndex checked_device(int index) {
  int count = 0;
  const auto status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess || index < 0 || index >= count || index > 127)
    throw std::runtime_error("requested CUDA device is unavailable; no CPU fallback");
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, index));
  if (properties.major < 8)
    throw std::runtime_error("native CUDA backend requires hardware BF16 support (SM80+)");
  int runtime = 0;
  C10_CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  if (runtime != DEEPFIN_MODEL_CUDA_RUNTIME)
    throw std::runtime_error("bound package CUDA runtime version mismatch");
  return static_cast<c10::DeviceIndex>(index);
}
}
CudaExecution::CudaExecution(int index, uint32_t batch, uint32_t channels, bool trace)
    : index_(checked_device(index)), batch_(batch), channels_(channels),
      stream_(c10::cuda::getStreamFromPool(false, index_)) {
  c10::cuda::CUDAStreamGuard guard(stream_);
  if (!(batch == 1 || batch == 2 || batch == 4 || batch == 8 || batch == 16)
      || (channels != 146 && channels != 175) || stream_.stream() == nullptr)
    throw std::runtime_error("invalid CUDA slot shape or default stream");
  const auto host = at::TensorOptions().device(at::kCPU).pinned_memory(true);
  const auto device = at::TensorOptions().device(at::Device(at::kCUDA, index_));
  host_input_ = at::empty({batch, channels, 8, 8}, host.dtype(at::kBFloat16));
  input_ = at::empty({batch, channels, 8, 8}, device.dtype(at::kBFloat16));
  packed_ = at::empty({batch, 1861}, device.dtype(at::kFloat));
  host_output_ = at::empty({batch, 1861}, host.dtype(at::kFloat));
  if (trace) trace_input_ = at::empty_like(host_input_, host.dtype(at::kBFloat16));
  if (!host_input_.is_pinned() || !host_output_.is_pinned()
      || (trace && !trace_input_.is_pinned()))
    throw std::runtime_error("CUDA staging must use pinned host storage");
  inputs_ = {input_};
  C10_CUDA_CHECK(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming));
}
void CudaExecution::open(const std::string& path) {
  if (loader_ || failed_) throw std::runtime_error("CUDA slot already open or failed");
  c10::cuda::CUDAStreamGuard guard(stream_);
  try {
    loader_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(path, "model", false, 1, index_);
    stream_.synchronize();
  } catch (...) { failed_ = true; drain(); throw; }
}
void CudaExecution::drain() noexcept {
  // Exceptional paths retain all temporary outputs until queued work drains.
  // Any failure poisons the slot; the foreign effect exits rather than reusing it.
  // A catastrophic driver failure cannot be recovered by retrying this slot.
  try { c10::cuda::CUDAStreamGuard guard(stream_); (void)cudaStreamSynchronize(stream_.stream()); }
  catch (...) { }
}
CudaExecution::~CudaExecution() noexcept {
  drain();
  // Tear down model resources and the event under their original device guard.
  try {
    c10::cuda::CUDAStreamGuard guard(stream_);
    outputs_.clear(); loader_.reset();
    if (done_) (void)cudaEventDestroy(done_);
  } catch (...) { }
}
void CudaExecution::run(const float* source, uint32_t rows, float* destination) {
  if (!loader_ || failed_ || !destination || !rows || rows > batch_)
    throw std::runtime_error("CUDA slot is unavailable or row contract failed");
  c10::cuda::CUDAStreamGuard guard(stream_);
  try {
    stage_bfloat16_inputs(source, batch_, rows, channels_, host_input_);
    const auto input_bytes = size_t(batch_) * channels_ * 64 * sizeof(c10::BFloat16);
    C10_CUDA_CHECK(cudaMemcpyAsync(input_.mutable_data_ptr(), host_input_.const_data_ptr(),
                                  input_bytes, cudaMemcpyHostToDevice, stream_.stream()));
    // The explicit stream handle is essential: guarding ATen alone is not a
    // contract that AOTI's generated launches use that same stream.
    outputs_ = loader_->run(inputs_, stream_.stream());
    validate_batch_outputs(outputs_, batch_, at::Device(at::kCUDA, index_));
    packed_.narrow(1, 0, 1858).copy_(outputs_[0]);
    packed_.narrow(1, 1858, 3).copy_(outputs_[1]);
    const auto bytes = size_t(rows) * 1861 * sizeof(float);
    C10_CUDA_CHECK(cudaMemcpyAsync(host_output_.mutable_data_ptr(), packed_.const_data_ptr(),
                                  bytes, cudaMemcpyDeviceToHost, stream_.stream()));
    if (trace_input_.defined()) {
      // Optional D2H snapshot of the actual BF16 device tensor, not ignored
      // host-tail memory or a claim that pre-conversion floats reached the model.
      C10_CUDA_CHECK(cudaMemcpyAsync(trace_input_.mutable_data_ptr(), input_.const_data_ptr(),
                                    input_bytes, cudaMemcpyDeviceToHost, stream_.stream()));
    }
    C10_CUDA_CHECK(cudaEventRecord(done_, stream_.stream()));
    C10_CUDA_CHECK(cudaEventSynchronize(done_));
    outputs_.clear();
    // No caller-visible output is modified until both heads validated and the
    // stream's H2D/model/packing/D2H work has physically completed.
    std::memcpy(destination, host_output_.const_data_ptr(), bytes);
  } catch (...) { failed_ = true; drain(); outputs_.clear(); throw; }
}
const void* CudaExecution::trace_data() const {
  if (!trace_input_.defined() || failed_) throw std::runtime_error("CUDA trace snapshot unavailable");
  return trace_input_.const_data_ptr();
}
std::string CudaExecution::audit() const {
  std::ostringstream out;
  out << "native-cuda-audit device_index=" << int(index_) << " stream_nondefault=" << (stream_.stream() != nullptr)
      << " pinned_buffers=" << (trace_input_.defined() ? 3 : 2) << " completion_events=1";
  return out.str();
}
}
