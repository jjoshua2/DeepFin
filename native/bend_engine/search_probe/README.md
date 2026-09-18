# Bend Gumbel sequential-halving probe

PR 4 moves the first real search algorithm into Bend.

The existing pure-C CBoard still owns chess/U64 move generation. For each fixed
chess position it returns the complete legal policy-action list. Bend then owns:

1. deterministic synthetic priors, Gumbels, and leaf values;
2. Gumbel + log-prior top-8 root candidate sampling;
3. DeepFin's sequential-halving visit-budget arithmetic;
4. completed-by-mix-value Q completion and min/max rescaling;
5. root sigma `0.1 * (50 + max_visit)`, matching the current selfplay-linear
   root defaults;
6. survivor re-ranking and halving; and
7. final visit/survivor accounting.

The probe uses a fixed 64-simulation budget and standard divisor-2 halving.
With top-8 candidates that exercises the complete 8 -> 4 -> 2 -> 1 schedule.

The Python test oracle calls DeepFin's current
`halving_visits_per_action`, `halving_keep_count`, and
`_completed_q_transform`; it does not carry a second unrelated halving
implementation.

This is still root-only search. Tree storage, below-root descent, expansion,
backpropagation, solved propagation, transpositions, and NN batching are later
PRs.
