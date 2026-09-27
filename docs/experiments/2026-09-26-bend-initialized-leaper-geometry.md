# Initialized non-slider attack geometry

## Scope before complete qualification

Base: PR #895 at e78987ccad0dfeafd026548b392e8fb52e960472.
Compiler: unchanged aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae.

Three proposed source contracts connect independently specified natural file/rank
knight, king, white-pawn and black-pawn masks to actual computation, their stored
values after the entire extras loop, and actual Chess.attack after allocation and
an arbitrary preceding table loop. Exact final array values are part of the last
two results. The query theorem assumes only depth17, all64 extras and bounded
square; no expected mask or initial content certificate is supplied.

The source model uses explicit coordinate steps, not production U32 wrapping deltas.
Finite scalar-coordinate cases are checked inside Bend; structural accumulation and
write-preservation induction avoid enumerating arbitrary arrays or occupancies.
This does not yet compose all five attack-mask classes with the attack-witness
reduction, prove forward/reverse slider semantics, or establish singleton king
conditions through castling. Existing initialized-slider proofs remain unchanged.

Acceptance: exact safe source consumer; classified semantic/policy controls;
independent actual initialization/query tests in generic, portable, native-target
and UBSan; original compiler/pin checks; unchanged repository lint. No aggregate
whole-stack run, production change, perft increase, model/GPU/training or benchmark.
All statements below are added only for completed checks. Self-review only.
