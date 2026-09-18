// Foreign Bend effects for the architecture probe.
//
// The scalar effect handles board lifecycle/status. The legal-move effect
// marshals one complete sorted action list into a Bend List<U32>, so recursive
// selection stays entirely inside pure Bend rather than crossing IO per child.

#include <stdint.h>

extern uint32_t deepfin_bend_probe_call(uint32_t op, uint32_t a, uint32_t b);
extern uint32_t deepfin_bend_probe_legal_moves(
  uint32_t handle, uint32_t* out, uint32_t capacity
);

Term chess_call_run(Env e, Term* f, IoWork* w) {
  (void)e;
  (void)w;
  return (Term)deepfin_bend_probe_call(
    (uint32_t)f[0],
    (uint32_t)f[1],
    (uint32_t)f[2]
  );
}

Term chess_legal_moves_run(Env e, Term* f, IoWork* w) {
  (void)w;
  uint32_t moves[256];
  uint32_t count = deepfin_bend_probe_legal_moves((uint32_t)f[0], moves, 256);
  Term xs = term_pak(CID_NIL, 0);
  for (uint32_t i = count; i > 0; i--) {
    xs = io_node(e, CID_CON, (Term)moves[i - 1], xs, 0);
  }
  return xs;
}

static void __attribute__((constructor)) chess_probe_effects_use(void) {
  io_eff(CID_CHESS_CALL, chess_call_run, 0);
  io_eff(CID_CHESS_LEGAL_MOVES, chess_legal_moves_run, 0);
}
