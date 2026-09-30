# Bend castling producer geometry

## Scope

This increment stacks on PR #888 at `9db6725f77b48510650442caa725f814536905c4`.
Qualified source is `4f3f40c9a98ddac838bd851ddf7fe821e8cfbfe2`.

Six universal source laws formalize the four orthodox route constants used by the
castling producer:

- king source is home+4;
- king destination is home+6 or home+2;
- rook source agrees with the raw castling update's rook selector;
- king transit is exactly the rook landing square;
- the producer's between mask contains that transit/rook-landing square;
- the same between mask contains the king destination.

Each theorem is discharged by exhaustive four-route pattern matching. These are
coordinate/mask theorems, not chess-legality theorems.

## Actual-source linkage

The hosted gate additionally checks that the actual
`native/bend_engine/legal_probe/Chess.bend::castle_side` still contains exactly
the expected home/src/rook/dst/transit/right/between expressions and the
`occupied & between == 0` guard. This is fail-closed source identity evidence
connecting the independent geometry to the implementation. It is not itself a
formal list-membership theorem.

Combined with PR #888's public castling-update theorem, this establishes the
geometric fact needed next: the producer's path-clear mask includes the rook
landing square that raw update preservation requires to be empty.

## Hosted qualification

Run **36219695972**, job **108342486509**, passed:

- six-law importing consumer on the current pinned compiler;
- all eight actual `castle_side` source anchors;
- original compiler16-law/seven-control source gate;
- all12 compiler pin tests;
- unchanged Ruff/Basedpyright/Vulture repository lint.

No new native executable result is claimed here.

An attempted fresh run of the historical `legal_probe` correctly failed before
execution because its own bitboard-probe toolchain fingerprint differs from the
current standalone compiler fingerprint. That guard was not overridden and the
old probe was not silently upgraded. Its earlier fixed castling/transit parity
remains historical evidence only on its original compiler pin.

Artifact **10898586940**, ZIP SHA-256
`3de32e8fb5fe0aca48d60c41684f8ee0d5b062d74b639c4c52e8d55bd0ee5d7a`.

Coverage is modular **160 laws / 470 controls**: exact parent154/470 plus six new
geometry laws. No new negative controls were registered in this increment and the
complete160-law wrapper was not executed.

## Limits and next step

This increment does not prove that every flag-two move appearing in
`Chess.legal_moves` satisfies the producer guard, does not prove start/transit/
destination king safety, and does not establish legal-move soundness/completeness.

The next decisive source theorem should connect actual `castle_side` list output
to these route/guard facts—e.g. any newly retained flag-two move must come from a
true producer guard—then combine that with #888's conditional update-preservation
law. After that, the remaining castling legality work is attack-free start/transit/
destination and independent rights/metadata semantics.

No production code, Python responsibility, model/GPU path, search, training,
benchmark or perft budget changed. Self-review only.
