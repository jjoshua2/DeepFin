# Bend two-ply MCTS tree probe

PR 5 validates below-root search semantics before choosing a high-performance
tree representation.

For each fixed chess fixture, CBoard supplies real legal moves. The probe caps
the branch factor to the first four sorted legal actions at root and at each
root child, then runs 24 simulations.

Bend owns:

- root and child N/W/prior state;
- first-visit expansion state;
- PUCT selection with DeepFin's parent/child side-to-move sign convention;
- root FPU reduction 0.25 and below-root FPU reduction 0.15;
- first-visit child evaluation;
- depth-2 grandchild selection/evaluation;
- sign-alternating backup from leaf STM to root STM; and
- the complete selected-path sequence.

Priors and fake values are exact binary fractions so the production C tree's
double state and Bend's F32 state are not asked to be bit-identical. The parity
contract is the integer visit distribution, selected-path hash, expansion count,
best move, and a quarter-unit root-W code.

The Python oracle drives **DeepFin's real C `MCTSTree`** using
`add_root`, `expand`, `select_leaves`, `backprop`,
`get_children_visits`, `find_child`, and `is_expanded`.

This probe deliberately uses functional reusable Bend lists. It answers the
semantic question first. A later PR can replace the representation with arrays
or another compact native layout and benchmark throughput without conflating
representation bugs with MCTS-rule bugs.
