// Foreign Bend effect: one scalar dispatch surface into the separately-linked
// DeepFin CBoard bridge. Keeping the effect scalar is deliberate for PR 1: it
// measures the actual Bend <-> C effect boundary without teaching the probe a
// second board representation.

extern uint32_t deepfin_bend_probe_call(uint32_t op, uint32_t a, uint32_t b);

Term chess_call_run(Env e, Term* f, IoWork* w) {
  (void)e;
  (void)w;
  return (Term)deepfin_bend_probe_call(
    (uint32_t)f[0],
    (uint32_t)f[1],
    (uint32_t)f[2]
  );
}

static void __attribute__((constructor)) chess_call_use(void) {
  io_eff(CID_CHESS_CALL, chess_call_run, 0);
}
