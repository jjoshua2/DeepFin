// Explicit native CPU microbenchmark; never linked into a search executable.
// Input/reference validation and JSON/output processing are outside timed calls.
#include "model_contract.h"
#include <openssl/evp.h>
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" uint32_t deepfin_model_open_batch();
extern "C" int deepfin_model_run_batch(const float*, uint32_t, float*, uint32_t);

namespace {
using Clock = std::chrono::steady_clock;
constexpr uint32_t B = DEEPFIN_MODEL_BATCH, C = DEEPFIN_MODEL_CHANNELS, W = 1861;
constexpr double ATOL = 2e-6, RTOL = 2e-5;
static_assert(sizeof(float) == 4 && std::endian::native == std::endian::little);
static_assert(B == 1 || B == 2 || B == 4 || B == 8 || B == 16);
static_assert(C == 146 || C == 175);
static_assert(DEEPFIN_MODEL_CUDA == 0, "service profile is CPU-only; no CUDA qualification");
static_assert(Clock::is_steady);

uint32_t number(const char* s, uint32_t lo, uint32_t hi) {
  std::string v(s);
  uint32_t n = 0;
  const auto [p, ec] = std::from_chars(v.data(), v.data() + v.size(), n);
  if (v.empty() || (v.size() > 1 && v[0] == '0')
      || ec != std::errc{} || p != v.data() + v.size() || n < lo || n > hi)
    throw std::runtime_error("invalid bounded decimal argument");
  return n;
}
uint64_t elapsed(Clock::time_point start) {
  const auto n = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
  if (n <= 0) throw std::runtime_error("nonpositive steady-clock interval");
  return static_cast<uint64_t>(n);
}
std::vector<float> read_floats(const char* path, size_t count) {
  std::vector<float> v(count);
  std::ifstream f(path, std::ios::binary);
  const auto bytes = static_cast<std::streamsize>(count * sizeof(float));
  if (!f || !f.read(reinterpret_cast<char*>(v.data()), bytes)
      || f.peek() != std::char_traits<char>::eof() || f.bad())
    throw std::runtime_error("input/reference must have exact bound shape");
  if (std::any_of(v.begin(), v.end(), [](float x) { return !std::isfinite(x); }))
    throw std::runtime_error("nonfinite input/reference");
  return v;
}
std::string hash(const std::vector<float>& values) {
  std::array<unsigned char, EVP_MAX_MD_SIZE> out{};
  unsigned int n = 0;
  if (EVP_Digest(values.data(), values.size() * sizeof(float), out.data(), &n, EVP_sha256(), nullptr) != 1 || n != 32)
    throw std::runtime_error("SHA256 failed");
  std::ostringstream text;
  for (unsigned int i = 0; i < n; ++i)
    text << std::hex << std::setw(2) << std::setfill('0') << unsigned(out[i]);
  return text.str();
}
struct Sample {
  uint32_t sweep, rows;
  uint64_t ns;
};
}
int main(int argc, char** argv) {
  try {
    if (argc != 6) throw std::runtime_error("usage: probe INPUT.f32 REFERENCE.f32 WARMUPS_PER_SIZE SAMPLES_PER_SIZE ORDER_OFFSET");
    const uint32_t warm = number(argv[3], 1, 32), count = number(argv[4], 2, 256);
    const uint32_t offset = number(argv[5], 0, B - 1);
    // No trace I/O or audit-counter work may contaminate service measurements.
    for (const char* name : {"DEEPFIN_BEND_MODEL_TRACE", "DEEPFIN_BEND_BUFFER_AUDIT"}) {
      const char* s = std::getenv(name);
      if (s && *s) throw std::runtime_error("disable trace/audit during service measurement");
    }
    const auto input = read_floats(argv[1], size_t(B) * C * 64);
    const auto expected = read_floats(argv[2], size_t(B) * W);
    const auto input_hash = hash(input), expected_hash = hash(expected);
    std::vector<float> output(size_t(B) * W + 1);
    std::vector<Sample> observations;
    observations.reserve(size_t(warm + count) * B);
    const auto load = Clock::now();
    if (deepfin_model_open_batch() != DEEPFIN_MODEL_PROFILE)
      throw std::runtime_error("bound native model startup failed");
    const auto startup_ns = elapsed(load);
    double max_error = 0;
    const auto panel_start = Clock::now();
    for (uint32_t sweep = 0; sweep < warm + count; ++sweep) {
      // Rotate occupancy order between sweeps and process repetitions.
      for (uint32_t index = 0; index < B; ++index) {
        const uint32_t rows = 1 + (index + sweep + offset) % B;
        std::fill(output.begin(), output.end(), std::numeric_limits<float>::quiet_NaN());
        const auto start = Clock::now();
        const int result = deepfin_model_run_batch(input.data(), rows, output.data(), rows * W);
        const auto ns = elapsed(start); // callback physically returns; not async launch timing
        if (result) throw std::runtime_error("native forward failed; no successful profile");
        for (size_t i = 0; i < size_t(rows) * W; ++i) {
          const double actual = output[i], want = expected[i];
          const auto error = std::abs(actual - want);
          if (!std::isfinite(actual) || error > ATOL + RTOL * std::abs(want))
            throw std::runtime_error("native output disagrees with independent reference");
          max_error = std::max(max_error, error);
        }
        for (size_t i = size_t(rows) * W; i < output.size(); ++i)
          if (!std::isnan(output[i])) throw std::runtime_error("native output tail was overwritten");
        observations.push_back({sweep, rows, ns});
      }
    }
    const auto panel_ns = elapsed(panel_start);
    // Emit only after the whole run succeeds. Warmup observations are retained,
    // distinctly labeled, rather than silently dropped or mixed into throughput.
    std::cout << std::setprecision(17)
      << "{\"schema\":\"deepfin.native-service-samples.v1\",\"status\":\"passed\""
      << ",\"package_sha256\":\"" << DEEPFIN_MODEL_SHA256 << "\""
      << ",\"checkpoint_sha256\":\"" << DEEPFIN_CHECKPOINT_SHA256 << "\""
      << ",\"input_sha256\":\"" << input_hash << "\",\"reference_sha256\":\"" << expected_hash << "\""
      << ",\"batch\":" << B << ",\"channels\":" << C << ",\"output_width\":" << W
      << ",\"device\":\"cpu\",\"dtype\":\"float32\",\"torch_threads\":2,\"interop_threads\":1"
      << ",\"warmups_per_size\":" << warm << ",\"samples_per_size\":" << count << ",\"order_offset\":" << offset
      << ",\"model_open_ns\":" << startup_ns << ",\"panel_ns\":" << panel_ns
      << ",\"max_logit_error\":" << max_error << ",\"atol\":" << ATOL << ",\"rtol\":" << RTOL
      << ",\"accepted_neural_rows\":null,\"useful_eps\":null,\"samples\":[";
    bool first = true;
    for (const auto& r : observations) {
      if (!first) std::cout << ',';
      first = false;
      std::cout << "{\"sweep\":" << r.sweep << ",\"real_rows\":" << r.rows
        << ",\"warmup\":" << (r.sweep < warm ? "true" : "false") << ",\"service_ns\":" << r.ns << '}';
    }
    std::cout << "]}\n";
    if (!std::cout) throw std::runtime_error("report write failed");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "native service profile: " << error.what() << '\n';
    return 2;
  }
}
