# Sequential initialized castling producer and full-filter checks

## Scope and acceptance recorded before hosted qualification

Base: PR #899 at `22aa6449cf4fc69506442e9396837f342cf061e6`.
Compiler: unchanged `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`.

This continuation composes the established per-stage initialized geometric check
results through the actual producer's start/transit calls and the actual full
filter's destination call. The table returned by each operation supplies the next
operation; complete returned-array equality is part of the source result.

The three related public contracts cover exact producer output with an arbitrary
input tail; exact one-route empty-tail producer-plus-full-filter output; and false
independent check results at all three stages for any retained member of that
isolated pipeline. The first contract does not certify pre-existing tail entries.
The full-filter pipeline is a proof/test composition of production functions, not
the full optimized `Chess.legal_moves` implementation.

Input assumptions remain initial representation consistency, unique moving king
at home, valid Boolean-encoded turn, and actual castling producer guard. Actual
initialized table depth is 17, slider block count 128, extras count 64, with arbitrary
seed. The public caller does not supply stage singletons, desired check answers,
intermediate array identity, or an assumed safe destination.

## Deciding checks and budget

Accept only the complete importing consumer with exit zero and exactly
`All terms check.`, together with the new rejection gate. Seven source controls
exercise actual retention/table updates or new certificate/decision connections;
eight policy checks protect manifests/imports, and one warning-output unit is
synthetic. No crash, timeout or incorrect-input typing error is a semantic pass.
The full source consumer is capped at 1,500 seconds and each rejection at 180.

Native acceptance requires generic, forced-portable, native-target and UBSan results
matching the retained external coordinate/set reference, including complete ordered
producer and filtered lists, all 19 fields of each child Board, and an unused table
marker after every batch. The two contexts are actual public Tables.build and a
patterned-seed complete initialization. Actual mutations omit start/transit/final
checking or replace the returned array. All must compile and execute before value
rejection. No training, model, GPU, search, benchmark or perft workload is added.

## Development results before publication

Internal Route/Wire and Decision modules checked successfully. Native tests passed
all four modes: 852 base requests in two initialization contexts, 1,704 executions,
3,414 complete child Boards (64,866 fields), 28 marker reads and nine malformed
batch rejections per mode. There are 432 true guards, 269 producer emissions and
238 filtered acceptances per base fixture set. Four destination-only attacked
cases survive the producer but are rejected by the following filter.

The complete public source consumer and final full rejection gate are not claimed
qualified here until actual final receipts are attached. Small examples establish
satisfiable input conditions for both colors and wings; they do not normalize a
closed enormous initialized table.

## Retained construction and harness history

Early Wire drafts exposed an equality-transport orientation mistake and affine
proof-parameter reuse; proof-only parameters were correctly marked erased, without
changing runtime table ownership. The final internal module passed.

A draft directly normalized all geometric check values for four concrete boards
and exceeded its 180-second bound. Its simpler input-premise examples passed; the
native reference supplies concrete safe/unsafe behavior evidence. No previously
accepted law or public geometry statement was weakened.

The initial rejection gate timed out while normalizing a changed transit call;
a later full zero-array replacement was killed (status -9). Neither is credited as
semantic rejection. Native mutations still test both actual behaviors. Source
controls instead target the explicit transit/destination certificate connection
and a single-slot actual returned-table mutation. A first single-slot mutation used
the wrong Array.set argument order and was rejected at that wrong location; the
runner rejected it as invalid evidence before its argument order was corrected.
Original available logs and partial records are preserved in the review package.

## Remaining semantic boundary

These contracts use independent target-centred coordinate/ray attack semantics.
Universal forward/reverse attack correspondence remains separate. They establish
safe-stage results for the isolated full-filter pipeline, not whole optimized
legal_moves safety, both-wing/ordinary-scan table preservation, historical rights,
reachability, metadata legality or complete move-generation soundness/completeness.
A marker test is not native full-buffer, allocator or lifetime verification.

No production source, earlier law, compiler input or permanent workflow changes.
No new Python application responsibility moves into Bend. Self-review only; the
checker/Base, native lowering/storage, ABI, toolchain, OS and hardware remain trust
boundaries. Existing strict-TypeScript, snapshot-lowering and literal closed-builder
limitations are not repaired by this increment.
