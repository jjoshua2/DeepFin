// Bend foreign effect wrapper for the real DeepFin CUDA parity harness.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern uint32_t deepfin_cuda_parity_run(uint32_t* out, uint32_t capacity);

Term nn_run_run(Env e, Term* f, IoWork* w) {
  (void)f;
  (void)w;
  uint32_t values[16];
  uint32_t count = deepfin_cuda_parity_run(values, 16);
  if (count == 0) {
    fprintf(stderr, "nn_run_run: native CUDA parity returned no outputs\n");
    exit(1);
  }
  Term xs = term_pak(CID_NIL, 0);
  for (uint32_t i = count; i > 0; i--) {
    xs = io_node(e, CID_CON, (Term)values[i - 1], xs, 0);
  }
  return xs;
}

static void __attribute__((constructor)) nn_run_use(void) {
  io_eff(CID_NN_RUN, nn_run_run, 0);
}
