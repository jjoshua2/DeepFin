# Supplementary terminal-path occupancy frame

This is a new checked foundation for the remaining independent P2 geometry proof.
It is archived review code, not an additional registered lookup contract and not
a proof of actual Tables.ray or Tables.slider. No production source or existing
proof was changed to obtain it.

## Exact theorem

For every finite list of Nat square identifiers and every pair of U64 occupancies,
if the occupancies agree at each path position before the final position, the
independent `Path.attack` result is equal. The final position may differ.

`Path.attack` includes the first blocker before stopping. `Path.agree` observes
only input occupancy bits along the list's interior; it contains no expected
attack values or assumed equality of attack computations. The proof proceeds by
structural induction over the path. The theorem also covers empty/singleton lists,
arbitrary list lengths and repeated square identifiers; it does not assume the
list already has valid geometric coordinates. Geometric validity and its
connection to the actual production traversal are separate obligations.

The importing consumer provides an arbitrary-occupancy singleton result, a
terminal-only blocker example, a first-blocker-inclusion example, and an interior
blocker difference rejected by the premise. This is not a vacuous no-occupancy
claim. The first-blocker-inclusion example is closed, not a separate universal
set-membership theorem.

## Executed checks

Pinned compiler: aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae, Bend2.0.21 plus U64,
Bun1.4.2; source fingerprint d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4.
The final consumer completed locally with exit0 and exactly `All terms check.`
in 1.181526679 seconds. No new axioms, unsafe/foreign dependencies or holes.

Two disposable premise mutations were rejected at `Frame.interior_only` with
ordinary expected/observed diagnostics: omitting the first interior observation,
and admitting every occupancy pair. A third mutation changes the independent
attack function to omit the blocker itself; it is rejected at `Frame.single`.
These are new supplementary controls only, not added to the public lookup suite's
sixteen controls. They mutate the independent specification/proof connection,
not production Tables code. Missing-file errors, crashes and timeouts do not
count as semantic rejection. No native execution of these new path modules is
claimed.

The initial draft lacked explicit duplication markers on Bool b and Nat x.
The checker rejected both before acceptance. Adding the markers changed neither
the theorem domain nor checker. Original diagnostics are retained in the
conversation review package; no invalid draft is claimed to have passed.

## Reproduction

Copy these three `.bend.txt` files into a disposable directory, removing only
`.txt`, and check `consumer.bend` with the exact compiler:

```sh
BEND_NO_TELEMETRY=1 bun /path/to/pinned/bend/bend2/main.ts consumer.bend
```

For controls, modify only a disposable copy of Path.bend and check Frame.bend:

1. Replace `Bool.and(eq(U64.test_bit(a,x),U64.test_bit(b,x)),agree(y <> rest,a,b))`
   with `agree(y <> rest,a,b)`.
2. Replace that same expression with `True{}` instead.
3. In a fresh copy, replace the attack step with
   `Bool.pick(U64,U64.test_bit(occ,x),U64.zero(),U64.or(U64.bit(x),attack(tail,occ)))`
   and check consumer.bend. This must fail at Frame.single.

The required rejection is status1 and an ordinary expected/observed diagnostic
at the intended proof, never a crash or missing dependency.

| Exact archived source | SHA-256 |
| --- | --- |
| Path.bend.txt | 92595303f046ac947264f179a3489b40cd7822b674807511d846955eda6269ad |
| Frame.bend.txt | eacd6b86d18f3ead0dd56f4843f36b134b5c61630ae16fc6147dae7be8f8ccc9 |
| consumer.bend.txt | 8d6e4d978adcb8a02369a9ab190c5c0d32a75573f1052625ba28620b77fd7bc2 |

Semantic diagnostic hashes, excluding final newline, in the order above:
379ffca0b96c7ea16a82bd1862bb6a05fbb7f3c2bc302bab0ea29c8bfef317d6,
38006ee3ffc2f587d880e9b0298da0f23b95d50608ce0b96c209030539b03aaa,
316d42e3c9ab1544731b9e247b347e3054fb54edc271f4f83897918522b2f01b.

## Remaining connection

To use this in the final P2 theorem, prove that actual Tables.ray agrees with this
independent list traversal for every supported square/direction/occupancy, and
that masking by the actual relevant mask preserves all interior observations.
The already-checked independent step and complete-mask identities are foundations
for those connections, not substitutes for them. In particular this record does
not establish production terminal-edge irrelevance merely from a specification
lemma.

Self-review only. This source proof trusts the pinned checker/Base; it does not
qualify native lowering, allocation/lifetime or hardware. No application logic
moved from Python, no engine/model/GPU/perft/training/benchmark run was added.
