#pragma once
#include <cstring>
#include <stdexcept>

namespace deepfin_native {
inline void check_dispatch_binding(const char* plan, const char* sha,
                                   const char* bound_sha, bool batch_api, bool cuda) {
  if (!plan && !sha) return;
  if (!plan || !*plan || !sha || !*sha || !batch_api || cuda)
    throw std::runtime_error("dispatch requires a CPU batch plan and its exact package SHA256");
  if (std::strlen(sha) != 64 || std::strcmp(sha, bound_sha) != 0)
    throw std::runtime_error("dispatch profile/model package SHA256 mismatch");
  for (const char* p = sha; *p; ++p)
    if (!(*p >= '0' && *p <= '9') && !(*p >= 'a' && *p <= 'f'))
      throw std::runtime_error("dispatch package SHA256 must be lowercase hexadecimal");
}
}
