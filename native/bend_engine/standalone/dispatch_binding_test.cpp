#include "dispatch_binding.h"
#include <cassert>
#include <iostream>
#include <string>
int main() {
  using deepfin_native::check_dispatch_binding;
  const std::string sha(64, 'a'), wrong(64, 'b'), upper(64, 'A');
  check_dispatch_binding(nullptr, nullptr, sha.c_str(), false, false);
  check_dispatch_binding("4 1", sha.c_str(), sha.c_str(), true, false);
  unsigned rejected = 0;
  const auto bad = [&](const char* plan, const char* named, const char* bound, bool batch, bool cuda) {
    try { check_dispatch_binding(plan, named, bound, batch, cuda); }
    catch (const std::runtime_error&) { ++rejected; return; }
    throw std::runtime_error("binding negative control accepted");
  };
  bad(nullptr, sha.c_str(), sha.c_str(), true, false);
  bad("", sha.c_str(), sha.c_str(), true, false);
  bad("4 1", nullptr, sha.c_str(), true, false);
  bad("4 1", "", sha.c_str(), true, false);
  bad("4 1", wrong.c_str(), sha.c_str(), true, false);
  bad("4 1", "a", sha.c_str(), true, false);
  bad("4 1", upper.c_str(), upper.c_str(), true, false);
  bad("4 1", sha.c_str(), sha.c_str(), false, false);
  bad("4 1", sha.c_str(), sha.c_str(), true, true);
  assert(rejected == 9);
  std::cout << "{\"status\":\"passed\",\"rejected\":9}\n";
}
