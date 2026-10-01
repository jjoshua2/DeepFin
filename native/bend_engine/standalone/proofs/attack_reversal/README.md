# Non-slider forward/reverse attack membership

Two public source laws on the pinned compiler:

* `leaper_coordinate_membership_reverses`: for on-board source and target,
  membership in the accepted independent coordinate target list is reciprocal
  under `Geometry.reverse` (pawn colors swap; king and knight do not).
* `computed_leaper_mask_reverses`: the same equivalence for actual computed
  `Slots.value` mask bits, using the previous `Computed.actual` producer and a
  new structural mask/list reflection. No caller supplies a computed-mask answer.

The 256 square/class certificates check the inverse of each possible directed
step. List induction projects those certificates to arbitrary bounded endpoint
membership, and Boolean reflection establishes equality in both directions.
The finite certificate generator emits proof constructors, not trusted answers;
the pinned Bend checker checks every equality and every excluded off-board case.

Both endpoints are below 64. The coordinate lists contain the off-board marker
64, whereas `U64.bit(64)` is zero. Dropping the bound is not licensed by the fact
that the underlying array or machine word exists. The structural mask/list
reflection accepts arbitrary Nat list entries, but observes only bounded bits.

These are two related interfaces to one new reciprocity result, not two
independent mathematical discoveries. Slider blocker paths are not covered.
The new public law is about actual mask computation, not a newly combined
initialized `Chess.attacked` or complete-generator safe-acceptance theorem.
Existing storage/query certificates are unchanged; native checks additionally
exercise actual initialized `Chess.attack` in four initialization contexts.

Commands from repository root (BUN and CC may select existing executables):

```sh
python3 native/bend_engine/standalone/proofs/attack_reversal/focused.py /path/to/bend --report /tmp/reversal-source.json
python3 native/bend_engine/standalone/proofs/attack_reversal/verify_native.py /path/to/bend --report /tmp/reversal-native.json
python3 native/bend_engine/standalone/proofs/attack_reversal/test_harness.py
```

The new runners require complete outputs regardless of Python optimization.
They invalidate the requested report before external work. Mutation diagnostics
never count crashes, missing imports, parse/ownership errors or timeouts as
successful semantic rejection. Initial argument/import failures are outside the
report guarantee. Concurrent runs require distinct output paths.

Native tests reuse the unchanged actual `attack_geometry/probe.bend`; the proof
model is not executed in the candidate. Every returned mask is checked against
independent signed-coordinate geometry AND every endpoint pair against its
reverse-class query. Swapping both pawn storage directions preserves reciprocity
but fails the independent geometry oracle: reciprocity alone is insufficient.
Build modes repeat inputs. No native array-identity or lifetime result is claimed.
