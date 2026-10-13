# Conditional equality of actual fast and full filters

The premise is per candidate: when Chess.filter_requires(sensitive,m) is False,
the actual full-filter check answer Spec.checked(c,b,m) must be False.
Spec.checked reads Chess.in_check after Chess.make_move with the original
board's turn. This premise says the full filter would retain a bypassed move;
it does not assert chess legality or prove king safety independently.

Under C.every of that predicate on xs, actual_fast_full equates the entire
production filter_fast and filter_legal pairs for arbitrary ordered xs and
arbitrary tail, including duplicates. The tail needs no safety premise.
The one-step lemma shares one accumulator. The structural induction processes
the head, updates the accumulator, and then recurses on the remainder.

actual_prepare_full uses the corresponding predicate with exactly
color(b,get_turn(b)) AND (Spec.rays(c,b) OR get_kings(b)). Prepare's full branch
is reflexive; its fast branch uses the conditional list equality. Both paths
compose the actual Filter.prepare/full bridges with pair-lifted list equality.
Its full-filter accumulator starts at Nil, as the implementation does.

No board-validity or target-geometry premises are added. There is no
unconditional fast/full equivalence or legal-move correctness claim.
The fixed-table observations use Observe.Cells/Observe.pack exactly as the
existing checked production bridges do. Equality includes the exact table
values, full Ply fields, list order and duplicate/tail multiplicity.

## Reproduce

Use this suite from a clean published checkout, or its explicit base-overlay
development mode. The pinned base is PR1021 head
21c84fd6b8810f5849bd21abf80585c02baa48c0, tree
74b910360f569630a395792e394107bd2d1110f8. For a shallow checkout:

    git fetch --depth=1 origin 21c84fd6b8810f5849bd21abf80585c02baa48c0

All reused source/support files must match their exact base Git blobs.
A published HEAD must have clean tracked files and differ from the base only
in this suite's Equivalence.bend, consumer.bend, qualify_fast_full.py and README.md.
The scope guard disables Git rename detection, and every qualified file is
verified against its HEAD Git blob.

Use Bun 1.4.2 and pinned Bend 2.0.21+U64 revision
aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae with its verified 84-file manifest,
fingerprint d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4.
From the repository root, with fresh paths outside the checkout:

    python3 -m native.bend_engine.standalone.proofs.fast_full_equivalence.qualify_fast_full \
      /path/to/pinned-checker \
      --checker-manifest /path/to/checker-tree.json \
      --report /outside/checkout/qualification-001.json \
      --evidence-dir /outside/checkout/checks-001

The qualifier inherits the caller's affinity, selects at most two allowed CPUs,
passes that exact set to every checker command, and restores original affinity.
Use the CPU set assigned alongside active jobs. Checker allowance is 86400 seconds
for both the positive and each negative run, with 6 GiB address-space and 16 MiB
output caps. Resource/timeout/import/parser failures are never semantic rejections.

The eight declaration-local controls test wrong bypass antecedent, a rejected
bypass certificate, unequal head accumulators, removal of the candidate premise,
loss of tail, removal of the production fast result, a weakened prepare-mask
premise, and reordered duplicate-containing candidates.
Checked closed instances show the bypass premise and exact prepare mask are necessary
and that reordering distinguishable candidates changes the full output.
The wrong True antecedent and all-bits prepare predicate hold in their respective
false-conclusion counterexamples. Each control is accepted only at its expected obligation with one nonempty unequal
expected/observed diagnostic on a single line per field and exit 1.
Duplicate or empty diagnostic fields are rejected.

The raw Pro candidate is retained separately as unexecuted provenance, SHA256
b2818aae09c037e7daff21ec560acc0a9dce8225a684ea4abe003b8df154914d.
It was structurally invalid. This implementation reconstructs the proof from
actual source; it does not attribute checked work to Pro.
CI evidence must name the checked-out synthetic merge commit separately from
its associated branch head, even when their source trees match.
