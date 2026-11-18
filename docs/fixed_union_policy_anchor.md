# Frozen-reference policy KL for a fixed union

The opt-in entry is `scripts.core_selfplay_union_policy_anchor_stored_row_once`; its passive proof wrapper is `scripts.core_selfplay_union_policy_anchor_stored_row_once_proof`. They use the distinct `core_selfplay_union_policy_anchor_stored_row_once_v1` contract and checkpoint family. Existing unanchored entry points and defaults remain unchanged.

This entry reuses the Trainer's existing forward KL from the frozen reference policy to the student policy over legal moves. It keeps globally uniform sampling without replacement, the real final partial batch, and the existing main-policy/outcome-WDL objective. It does not enable an auxiliary or search-WDL blend, priority sampling, or bounded shuffle.

The contract and method bind the same validated `policy_anchor` specification and finite nonnegative coefficient. A positive reference must be the exact immutable checkpoint, architecture and recorded step selected for model-only bootstrap. A zero coefficient requires a null reference and takes the no-reference-load/no-reference-forward path. No coefficient is selected by these entry points.

Reference setup runs after the fresh model bootstrap or full-state resume, before the initial observation. The teacher remains in eval mode, has no gradients and owns no optimizer parameters. The saved identity and continuation checks bind the anchor specification; changing the coefficient or reference requires a separately identified experiment.

The actual optimization objective includes the weighted KL term. Existing TOTAL diagnostics retain their pre-anchor meaning, so inspect the separate anchor eligible-row counts, KL row sum, weighted sum and forward diagnostics. The real partial-tail factor applies to the whole objective, including KL.

The own-live certificate records actual student and reference forward counts, the complete initial reference state/configuration, the matching final reference fingerprint, frozen/eval/optimizer-ownership checks, training diagnostics, and the existing student/optimizer/save/RNG proof. The wrapper adds no qualification forward. Zero-reference certificates explicitly contain no teacher state or forward work.

Ten Tiny model/Trainer cases cover the zero-path comparison, positive KL reaching real optimization, frozen reference identity, continuous-versus-resumed state/RNG, own-live proof and tampered reference or method rejection. They run through the normal CPU validation suite. Source review and tests establish implementation behavior; playing-strength effects require a controlled comparison with sampling, dose and other recipe settings held fixed.
