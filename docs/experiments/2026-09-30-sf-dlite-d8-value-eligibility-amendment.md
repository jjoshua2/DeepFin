# D-lite depth-8 value eligibility after full04 — 2026-09-30

This is a prospective eligibility amendment to the [Selected-E D-lite value
preregistration](sf_dlite_selected_e_value_prereg_20260930.md), recorded after
the fourth full-label attempt failed and before a fifth attempt. It changes
how repeated depth-8 scalar UCI emissions are admitted when they have the
**same value target**. The 2,500,000 selected rows, source-qualified roster,
Stockfish depth-8/Hash-8/full-history/six-man profile, calibrated WDL formula,
one final FP16 round, unchanged policy and other targets, direct depth-12
audit, paired training and 576-pair strength decision are unchanged. This
amendment gives no label, target, training or Elo credit by itself.

The fourth campaign used the reviewed six-worker PSS/anonymous-memory
envelope in the [dated operations record](2026-09-30-sf-dlite-d8-label-operations.md).
It stopped after 402.471 seconds when worker 1's frozen strict parser rejected
two different non-bound rank-1 depth-8 PV/score emissions. The failed launch
receipt is `labels/sf_dlite_legacy_g10_d8_full04_20260930/LAUNCH-1790796787321325022-FAILED.json`
(SHA-256 `14e6badc4547c06b4bdd170f182026816f780ef1cb9b10b3423aca8e5312d052`).
Its 85 sealed blocks contain 134,153 rows, but the entire failed root has
**zero admitted credit** and will not be adopted across operator versions.
Peak aggregate proportional set size was 8,553,768 KiB, below the 24 GiB
guard; the anonymous-memory, host-free, output and physical-I/O guards held.

A bounded, single-engine replay of the next source captured each raw UCI
response before parsing. It reproduced the same HOLD at source 37, physical
row 1,829, selected-roster index 621,597 (game 170317, ply 75). The first
non-bound depth-8 line reported mate −4 and PV starting `a8c8`; the second
reported mate −3 and PV starting `h5h4`. Stockfish's final `bestmove h5h4`
matched the second. Both lines had native UCI WDL `[0, 0, 1000]` and
byte-identical calibrated float32 WDL `[0, 0, 1]` (three-float SHA-256
`caccb9a8fbd2401135207066c4b53d6e88467839318ea2f1e8bfadf8fef6844e`).
The original strict parser correctly held on their different move/mate
distance. The zero-credit replay terminal is
`operations/sf-dlite-full04-source37-hold-diagnostic-run02-20260930/TERMINAL.json`
(SHA-256 `bef5c01a8beaa8d1aa87e3ffb23092f5363d5b465b62316cddad6354380754f3`);
its fsynced 1,451-attempt raw stream has SHA-256
`50f6781a90d073251fa12cae2c119ee7523d332eee8c9fb199783418cf14f60c`.
An earlier diagnostic imposed an additional 8 GiB virtual-address limit and
observed Stockfish exit; it is not evidence about this production failure.

For **scalar MultiPV=1 only**, the first complete, non-bound rank-1
exact-depth-8 emission still supplies CP/mate, nodes, first PV and the
calibrated SF value. A later scored depth-8 emission may differ in move, CP
or mate distance only if **every** such non-bound emission has the same three
native WDL integers and byte-identical three-element calibrated float32 WDL
as the first. Every emitted full PV must be legal from the authenticated
position. The final UCI `bestmove` must be legal and match the first move of
at least one such depth-8 PV. Bank every raw line, the first score/PV, the
last best move and the number of differing repeats. Earlier iterative depths
are diagnostic; they cannot alter the fixed-depth value. Keep the first
emission's value even if the final best move matches a later, value-identical
PV. This does not select the last score as a winner.

An absent complete depth-8 score, a scored depth-8 line without PV or native
WDL, a bounds-only result, an illegal full PV or best move, or **any**
native/calibrated value disagreement is HOLD. There is no score tolerance,
fallback depth, omitted or replacement row, or imputation. A parser HOLD
fsyncs a source/roster-bound raw trace before worker exit. Each complete
source block is fsynced within 600 seconds; an interrupted block is
recomputed on same-identity resume. A new operator version uses a fresh
full05 identity and output root rather than importing full04's sealed prefix.

The prepared worker v8 source has SHA-256
`2f847b2857431ffcb69811ae60bfe5f6f24de6c458d4069527577903b3e2e5b4`.
Its 13-case CPU fixture result has SHA-256
`4859d4e8d1681fc0c68155a1ecb9e95c21d94beb6668b0109143456f1fe161b2`:
the actual mate case and a deliberately saturated CP pair pass without
changing the first target; native/calibrated mismatches, missing WDL/PV,
bounds-only and partial streams, and illegal PV/best move hold. Independent
review also replayed the saved real stream and confirmed the first target
and refusal mutations. This is a parser and audit qualification, not a new
full campaign result. The exact local method amendment was fsynced before
full05 launch (SHA-256
`ab07b50be54dabd7857b182757457f98642e4b40eafa6c3aa1ce567af9b788ec`).
Root authorized the fresh six-worker operator at SHA-256
`ef72ea19d0f57b1009ccfb58c71fd50af00b43f6d0dc517419c8b607c1e6d09e`.
The durable prelaunch bundle at
`operations/sf-dlite-d8-full05-launch-proof-20260930/MANIFEST.json`
has SHA-256 `3f3c52e48859558223563fc7964982d915cd35df9eedcd55d69ac30e37e275c2`;
its exact command record has SHA-256
`006469b3d0fdb44da4fa8b089b0581de024acd8bf21c4f8e3c9de95543aefb3c`.
The launch receipt at
`labels/sf_dlite_legacy_g10_d8_full05_20260930/LAUNCH-1790801293986850286.json`
has SHA-256 `5ba5f55062cb401930a3e6ff682ec78c4366cc06e0d59ec27d04ad5ad0906a0b`.
At this publication point full05 is **running and unadmitted**. A complete
full05 terminal and independent all-row label audit remain required before
building targets.
