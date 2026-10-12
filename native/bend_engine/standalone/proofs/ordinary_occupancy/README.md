# Ordinary move occupancy normalization and actual ray subset

For actual Chess.make_move(board,Chess.Ply{src,dst,0,0}), Ordinary.normalization
proves unconditionally:

    Chess.occupied(post) =
      U64.or(U64.and_not(Chess.occupied(board),U64.bit(U32.to_nat(src))),
             U64.bit(U32.to_nat(dst)))

Board is arbitrary, including overlapping/inconsistent piece and color rows and
arbitrary raw U32 turn. No ownership, row-consistency, src!=dst or range premise
is added. The theorem fixes promotion=0 and flag=0; it makes no legality claim.

The source in legal_probe/Chess.bend computes occupied from white OR black.
Ordinary.unfold binds the actual move to Core.raw with the exact removal mask
Spec.removal(source,target) = OR(OR(source,target),OR(target,zero)).
One of the complementary color updates always inserts target. Algebra.colors
uses actual Chess.select_u64; merge structurally distributes clear over color OR;
reinsert proves erasing target again is absorbed by reinserting it. These are
checked Boolean truth cases, arbitrary Word induction and U64 limb lifts. The
proof is not placeholder reflexivity on a neutral board.

Ordinary.raw_occupied handles arbitrary mask/target/kind/color/metadata.
normalization uses checked Ordinary.unfold, congruence through actual occupied,
raw_occupied and reinsert. U64.and_not's actual operand order is a AND NOT b.

Ordinary.actual_ray uses actual Lattice.rewrite to replace the normalized
occupancy in PR1026 Clear.clear_then_insert with the actual post-move occupied.
The result is a subset of the old actual Tables.ray under the explicit premise
AND(source_bit, Path.attack(xs, Chess.occupied(board))) = zero. The input-only
Spec.trace, actual fuel, stopped=False/mask=False and arbitrary shared accumulator
are preserved. General source/destination U32s are allowed; no range assumption
is needed for their actual bit semantics.

The full legal-board/candidate/move/table invariant supplying the old observation
premise remains open, as do unconditional bypass retention, lookup framing and
full legal-move correctness. No raw Pro fragments were applied or credited.

The importing consumer checks generic normalization and actual ray subset.
Closed cases cover inconsistent piece rows/raw turn7, overlapping colors and
src==dst, absent source, off-range source/destination, and actual old/new ray
attacks6/2. Special flags deliberately differ: EP actual occupancy1024 versus
ordinary1028, and castling actual96 versus ordinary192. These are total-function
counterexamples, not assertions that the supplied moves are legal.

## Qualification

Exact stacked base PR1026 cedfc6be3a2dcf98d6a643f82d186d6a10711b8a,
tree7a36eb4edf8a5b87a6bd7cb56d05fe03d0a30461.

From the repository root, choose fresh external evidence paths:

    python3 -m native.bend_engine.standalone.proofs.ordinary_occupancy.qualify_ordinary_occupancy \
      /path/to/pinned-checker \
      --checker-manifest /path/to/checker-tree.json \
      --report /outside/checkout/qualification-001.json \
      --evidence-dir /outside/checkout/checks-001

Require Bend2.0.21+U64 aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae, Bun1.4.2,
all84 checker files, fingerprint
d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4.
Exact-base closure/check/evidence/mutation/strict-classifier helpers are reused,
hashed and pinned alongside source dependencies. A clean published head must
match every qualified Git blob; all reused dependencies must match the exact base.

Each positive/negative timeout is86400s. This lane uses only CPUs1,3, at most2CPU,
6GiB AS/RSS checks and16MiB per-output caps; P2 CPUs30-31 are excluded.
Prior evidence, qualified source and checker identity must remain unchanged.

Eight controls reject omitted source clearing, omitted destination insertion,
wrongly generalized EP/castle occupancy, ordinary flag1, disconnected actual ray,
vacuous old observation and loss of the shared accumulator. Only rejection at
the intended typed obligation counts; parser/import/resource/timeout/linearity
errors do not count. Independent internal review precedes stacked draft
publication. No merge, live adoption, runtime edit, P2/GPU or installation.
