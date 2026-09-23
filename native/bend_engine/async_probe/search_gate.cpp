// Test-only callback for the ACTUAL generated Bend application. Never linked
// into the shipped engine. A separate pipe controls physical callback completion;
// stdout remains exclusively the engine's UCI/accounting stream.
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unistd.h>

namespace {
int event_fd = -1, release_fd = -1;
int descriptor(const char* name) {
  const char* value = std::getenv(name);
  if (!value || !*value) std::abort();
  char* end = nullptr;
  long fd = std::strtol(value, &end, 10);
  if (*end || fd < 3 || fd > 1048575) std::abort();
  return static_cast<int>(fd);
}
char receive() {
  char c;
  ssize_t n;
  do { n = read(release_fd, &c, 1); } while (n < 0 && errno == EINTR);
  if (n != 1) std::abort();
  return c;
}
}
extern "C" int deepfin_model_open() {
  event_fd = descriptor("DEEPFIN_TEST_EVENT_FD");
  release_fd = descriptor("DEEPFIN_TEST_RELEASE_FD");
  return 4; // 175-plane root-legacy-meta/v2_threats, CPU F32 singleton
}
extern "C" int deepfin_model_run(const float* x, uint32_t count, float* y, uint32_t out) {
  if (!x || !y || count != 11200 || out != 1861) return 1;
  ssize_t n;
  do { n = write(event_fd, "S", 1); } while (n < 0 && errno == EINTR);
  if (n != 1) std::abort();
  const char command = receive(); // No release means physical storage stays owned.
  y[0] = 123.0f; // Failed partial writes must never reach a tree or caller.
  if (command == 'F') return 1;
  if (command != 'R' && command != 'N') std::abort();
  std::memset(y, 0, sizeof(float) * out);
  if (command == 'N') y[0] = std::numeric_limits<float>::quiet_NaN();
  return 0;
}
