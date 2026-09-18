// Bend foreign effect wrapper for native AOTInductor inference.
//
// The C++ bridge returns float32 output words as raw U32 bit patterns. Bend
// never receives the tensor itself; it receives only the compact values needed
// by this architecture probe.

#include <stdint.h>

extern uint32_t deepfin_aoti_probe_eval(
  uint32_t seed,
  uint32_t* out_bits,
  uint32_t capacity
);

Term nn_eval_run(Env e, Term* f, IoWork* w) {
  (void)w;
  uint32_t words[64];
  uint32_t count = deepfin_aoti_probe_eval((uint32_t)f[0], words, 64);
  Term xs = term_pak(CID_NIL, 0);
  for (uint32_t i = count; i > 0; i--) {
    xs = io_node(e, CID_CON, (Term)words[i - 1], xs, 0);
  }
  return xs;
}

static void __attribute__((constructor)) nn_eval_use(void) {
  io_eff(CID_NN_EVAL, nn_eval_run, 0);
}
