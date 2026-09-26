# Castling update proofs

This public proof suite covers the actual raw flag-two / promotion-zero update.

Registered laws:
- exact complete Board equality for arbitrary Board/raw U32 source and destination;
- representation preservation for the four orthodox coordinate routes when the input
  Board is representation-consistent and the rook landing square is initially empty.

The rook-target premise is intentional. The actual raw branch inserts the rook without
first clearing that landing square; #886 archives a consistent blocked-target
counterexample whose raw result is inconsistent.

This suite does **not** prove source king/rook existence, correct side to move,
castling rights, other path-square emptiness, attack-free king traversal, king safety,
legal reachability, clocks/history, or move-generation soundness/completeness.
Raw update refinement is not legal castling correctness.

The implementation-linked helper sources are byte-identical to the checked castling
review archived by #886.
