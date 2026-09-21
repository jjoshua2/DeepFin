/* Generic list-of-U32 transport for one build-bound native tensor model.
 * No board, move, rule, probability, or search decision is made here. */
#include <stdint.h>

extern uint32_t df_leaf_open(uint32_t out[8]);
extern uint32_t df_leaf_infer(uint32_t sequence, uint32_t count,
                            const uint32_t *input, uint32_t out[1861]);
extern void df_leaf_close(void);

static Term df_leaf_words(Env e, const uint32_t *words, uint32_t count) {
  Term xs = term_pak(CID_NIL, 0);
  while (count) xs = io_node(e, CID_CON, (Term)words[--count], xs);
  return xs;
}
static Term dfmodel_open_run(Env e, Term *f, IoWork *w) {
  (void)f; (void)w;
  uint32_t words[8];
  uint32_t n = df_leaf_open(words);
  return df_leaf_words(e, words, n);
}
static Term dfmodel_infer_run(Env e, Term *f, IoWork *w) {
  (void)w;
  uint32_t input[11200], output[1861], n = 0;
  Term xs = f[1];
  while (term_aux(xs) == CID_CON && n < 11200) {
    Term fields[2];
    spare_free(e, cls_fit(2), ctr_take(e, xs, 2, fields));
    if (fields[0] > UINT32_MAX) err_fail("non-U32 model transport");
    input[n++] = (uint32_t)fields[0];
    xs = fields[1];
  }
  if (term_aux(xs) != CID_NIL) err_fail("oversized model transport");
  uint32_t count = df_leaf_infer((uint32_t)f[0], n, input, output);
  return df_leaf_words(e, output, count);
}
static Term dfmodel_close_run(Env e, Term *f, IoWork *w) {
  (void)e; (void)f; (void)w;
  df_leaf_close();
  return term_pak(CID_UNIT, 0);
}
static void __attribute__((constructor)) dfmodel_use(void) {
  io_eff(CID_DFMODEL_OPEN, dfmodel_open_run, 0);
  io_eff(CID_DFMODEL_INFER, dfmodel_infer_run, 0);
  io_eff(CID_DFMODEL_CLOSE, dfmodel_close_run, 0);
}
