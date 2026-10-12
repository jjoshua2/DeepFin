# Actual generated candidate fields and bypass classification

This increment stacks on PR1037 head
`4dd45ecec257763a5be9635fd123e3a1fea455b6`, tree
`f61e53a88ab1d926bbcf4c7a78161b2d97d57415`.

`scan_fields` classifies every full-Ply occurrence of actual `Chain.candidates`.
The exact alternatives are tag 0/flag 0, tag 0/flag 1, and the typed promotion
tags 1..4/flag 0. `prefilter_fields` adds both actual guarded castle producers;
`legal_fields` threads these certificates through actual `filter_prepare` and
therefore covers actual `Chess.legal_moves` outputs. The castle alternative
includes the actual producer guard and full-Ply equality, not just flag 2.

`prefilter_bypass` and `legal_bypass` derive exact ordinary or typed promotion
witnesses from actual generated membership and actual `filter_requires == False`.
Flag 1 always forces the full check path, including arbitrary promotion tags.
A guarded castle has an owned bounded king source, so the existing sensitive-mask
certificate forces its full check path. Raw flag 2 alone can bypass and is not
used as a forcing premise. The owned-king result applies to every promotion/flag.

The classification needs no board-validity, canonical-turn, nonempty-king,
target-geometry or lookup-contract assumptions. `Fields` uses exact full-Ply
existential witnesses. `Emission` follows actual `put_move` and `destinations`:
promotion emits all four tags with flag 0, overriding even a true EP input;
otherwise the implementation passes `Bool.to_u32(pawn && destination == ep)`.
Fuel and empty are the actual parameters. `Scan` threads the actual affine array
through dependent continuations and arbitrary target masks/source-key lists.
`Generated` composes the actual guarded side and retaining-filter consumers.

Caller-tail classification is an explicit per-occurrence premise. Arbitrary tails
are not sanitized by scan: an unclassified tail survives an empty target mask.
The multi-bit example includes EP, promotion and ordinary fields, an unchanged
distinguishable input array, and two preexisting equal entries. Together with
one newly emitted equal move, all three occurrences survive. Separate witnesses
check zero fuel and explicit empty behavior. The mixed producer fixture contains
all four promotions and both guarded castles and checks actual final list order
against an arbitrary zero attack table; it makes no attack-geometry claim.

## Remaining whole-program obligations

Legacy `castle_chain.Spec.ordinary` means flag != 2, including EP and promotions.
It must not be substituted for PR1035's tag 0/flag 0 scope. This suite discharges
the actual field-classification boundary and eliminates forced EP/guarded-castle
cases from the possible bypass shapes. Classification is not a no-check proof.

The next bridge must retain actual scan-origin membership for each noncastle
occurrence, invoke PR1035 or PR1037 at the exact original side and input table,
and supply the existing ordered-list equivalence's per-occurrence bypass-safe
premise. It must preserve the actual list order and duplicate accumulator entries.
Board consistency, canonical turn, nonempty original-side king, actual initial
no-check, prepared bypass and separate old/actual-post whole-pair rook/bishop
lookup contracts remain explicit in those consumers. Initialized/builder lookup
contracts are not derived here. The initial-check True branch already chooses
full filtering and needs no bypass-safe argument. No full chess legality, post
board validity or unconditional fast/full equivalence is claimed.

## Qualification

Use Bend 2.0.21 + U64 at
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bun 1.4.2, and the verified
84-file checker fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
Positive and negative checks each allow 86400 seconds, CPUs 1/3, two threads,
a 6 GiB address-space/RSS bound and a 16 MiB cap per output file. Evidence and
immutable source snapshots live outside the checkout. Mutated closure snapshots
are preserved; no declaration is removed from the full consumer check.

The 26 controls are classified as 15 contract-coupling checks and 11 false-witness
checks. They cover typed promotion selection, EP flags, tail/producer/filter
connections, actual membership/bypass/guard/owned-king premises, exact returned
array, duplicate multiplicity, actual final order and raw castle/EP forcing.
Universal false selector claims have a concrete instantiation, with a fixed
board where relevant; the EP flag selector fact is boardless.
Each control needs exactly one intended typed mismatch with differing expected
and observed types. Parse, import, linearity, resource and incidental dependency
failures receive no negative credit.

```sh
python -B -m native.bend_engine.standalone.proofs.candidate_bypass_classification.qualify_candidate_bypass_classification \
  /path/to/pinned-checker --checker-manifest /path/to/checker-tree.json \
  --report /path/to/fresh-report.json --evidence-dir /path/to/fresh-evidence
```
