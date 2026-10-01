# Forward attack composition candidate

Status: unqualified candidate. No new theorem credit until the complete bounded
positive, semantic-control and source/compiler-identity gate passes.

## Contract

The independent Forward.attacked definition scans all 64 source squares as two
32-bit limbs. It checks source ownership and piece membership, then forward
pawn/knight/king coordinate targets or strict-interior rook/bishop visibility.
Queens contribute both rook and bishop witnesses. Occupancy is shared in both
directions; occupied endpoints are allowed, while strict interior blockers stop
sliders. Pinned pieces still attack. No production attack query, table lookup or
legal-move filter defines the forward answer.

The complete consumer requests four equations:

1. Target-centred independent geometry equals this forward-source witness scan,
   for any Board, Boolean attacking side and bounded target
2. Actual initialized Chess.attacked returns the complete same array and that
   forward answer
3. Actual initialized singleton Chess.in_check returns the same array and the
   opposing side's forward answer
4. The qualified actual full-generator castling projection retains its exact
   array/list result when all three stage decisions are forward witnesses

The geometry equation needs no Board-validity premise: overlapping raw piece
planes are interpreted by the same OR-of-memberships relation on both sides.
That generality is an algebraic fact, not acceptance of malformed chess states.
The actual query equations retain symbolic depth17/128-block/64-extra premises;
in_check retains its singleton/bounded-square premise. The castling equation
retains valid Board, Boolean side and home-king singleton premises, deciding
both castling guards inside the theorem.

This directly composes the shared bridge into the prior castling milestone.
It does not close the actual zero-argument Tables.build conversion, independent
rights/history/reachability, arbitrary king-away castling domain, ordinary/EP
enumeration, general skip-check safety or the whole-generator theorem. Runtime,
compiler, Base, pin and every prior proof/gate remain unchanged.

## Proof construction

Existing qualified leaper membership reversal (with pawn color swap) and
blocker-aware slider reciprocity provide pointwise geometry. A first-order
sampling/reconstruction proof transports those equalities through two symbolic
limbs. Proof-only packed masks connect to the existing witness fold; the public
forward specification scans source cells directly. No duplicated closure,
assumed mask, assumed attack answer or whole-board enumeration supplies a proof.
A 64-case scalar certificate normalizes only bounded header-coordinate arithmetic,
with an impossible out-of-domain branch; every case is checked in Bend.

## Bounded qualification design

The new opt-in source gate has a 1,600-second total cgroup bound, 200% CPU quota,
6 GiB RAM and no swap. The core/helper baseline has 180 seconds, each of five
controls 60, and the full importing consumer 1,050. Controls run before the
expensive consumer, but PASS requires every intended rejection, exact safe
consumer output, and final unchanged source/compiler identities. Timeout,
signals, memory/parser/kind/ownership/unfilled-definition errors or unrelated
diagnostics never count as proof or a valid semantic rejection.

Small specification fixtures qualify pawn direction, attacking-side ownership,
upper-limb coverage, queen diagonals and slider blockers. Disposable mutations
reverse pawn color, erase the upper source limb, remove queen diagonals, ignore
blockers and corrupt actual reverse-pawn query routing. The runtime routing
control uses a small exact production equation consumed by the actual pair proof.
Finite fixtures are control baselines, not universal theorem credit or native tests.

Historical separate consumers took about29 seconds for leaper reversal and92
for slider reversal. The prior castling consumer took about579–763 seconds.
These support a bounded trial, not a guarantee that this candidate fits. Resource
failure will remain NOT_COMPLETED with exact receipt; no larger budget or checker
change follows automatically. No local executor/tests, GPU, model or training work
is part of this source-proof task.
