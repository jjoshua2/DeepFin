# Selected-E D-lite scalar depth-8 first-target eligibility amendment — 2026-09-30

The exact local method amendment was fsynced **before** full06 labeling
(SHA-256 `a076b546e2fbd945a7080a37f6704bd2974637d0ef42fea0d99a7f1fc0cb7b50`).
This public record describes that prospective correction to the eligibility
rule in the [full05 amendment](2026-09-30-sf-dlite-d8-value-eligibility-amendment.md),
whose exact pre-full05 local method SHA-256 is
`ab07b50be54dabd7857b182757457f98642e4b40eafa6c3aa1ce567af9b788ec`.
The selected 2,500,000-row roster, source-qualified joins, Stockfish
depth-8/Hash-8/full-history/six-man profile, calibrated WDL formula, single
FP16 target round, unchanged policy and other targets, paired training, direct
depth-12 audit, and 576-pair arena decision do not change.

The preceding [full04 attempt](2026-09-30-sf-dlite-d8-value-eligibility-amendment.md)
also failed with zero admitted labels: its 85 sealed blocks / 134,153 rows
are evidence only and are not adopted. Full05 failed after 453.04 seconds with `FAILED_NO_LABEL_CREDIT` (launch
failure receipt SHA-256
`6d3fa62fb97bd915c999d2e11bc4be3e3c0a16f36d44731d7e99d2b6377046bc`).
Its 97 sealed blocks / 153,419 rows remain failed-attempt evidence only;
none are adopted into full06. The fsynced failing source-40 trace has SHA-256
`021e930b174740a3b0820a0cc1ba2d79c722401109bbf9b9b19d435a1902997c`.
Its first complete non-bound rank-1 depth-8 emission reported `cp 1927`,
native WDL `[1000,0,0]` and calibrated float32 WDL approximately
`[0.99998045,0.000014925688,0.0000046343962]` (three-float SHA-256
`f797f1d878b6ad2d98f5f66d71f2745d08362cb07b44599ee6cb17a8806aa679`).
A later exact-depth-8 emission reported `mate 7`, the same native WDL and
calibrated `[1,0,0]` (three-float SHA-256
`480376c6bf738a0227f2bbf2b3506b7cde209152c0ba9a9077e5527169eb292e`).
Both legal PVs began `d2d6`, matching the final best move. Full05's
byte-equality rule correctly held; this ordinary CP-to-mate emission shows
that requiring later value equality does not scale as an eligibility rule.
The independent saved-stream review is SHA-256
`fa26abc0a88bfa030d3d660c59abff71997493f1857ea9659df6582a264e0da7`.

For **scalar MultiPV=1**, the target is the first complete, non-bound,
rank-1 exact-depth-8 scored emission. Its CP or mate score, nodes, native
WDL, full legal PV, and calibrated float32 WDL are banked; its calibrated WDL
alone supplies the SF third of the candidate main value target. The first
target is identical to the first-target definition used in full04/full05.
Every subsequent non-bound scored exact-depth-8 rank-1 emission must itself
have one valid CP-or-mate score, positive nodes, a native three-integer WDL
summing to 1,000, a complete PV legal from the authenticated board, and a
finite calibrated WDL. Its score, PV and value are **diagnostic only**: value
churn never changes the first target or determines eligibility. All raw UCI
lines are retained. The final UCI best move must be legal and match the first
move of at least one valid exact-depth-8 PV. Bound lines and earlier iterative
depths are banked as raw diagnostics and never define the target. A bounds-only
search, missing complete non-bound depth-8 score, malformed native WDL,
incomplete or illegal full PV, unsupported final best move, or partial stream
still holds and fsyncs the failed-row raw trace before exit.

Per row, bank the first and last exact-depth-8 native WDL and calibrated
float32 WDL, number of scored exact-depth-8 emissions, count of later
score/head changes and counts of later native/calibrated value changes from
the first, plus maximum absolute component drift from the first for native
permille and calibrated WDL. These fields quantify provisional-versus-final
value churn in the full label source and independent audit. Do not filter,
resample, average, choose the last score, apply a numerical tolerance,
substitute a deeper score, or impute a missing row. The direct depth-12 audit
and paired 576-opening arena retain their preregistered roles; no strength
claim follows from this method amendment.

Full06 uses a fresh source-pinned operator (SHA-256
`31a7730cac1f9b6b39f01b938ccf6772c895d5a2727f89eb5a969aacec24cf83`),
launcher (SHA-256
`47ccd2327a5419a9288153a09d3b746fba5b383dc6433c89b01829f5f53b1c82`),
authorization (SHA-256
`348e33a76e40ef29e58c249bfbf6e2b5f883d13a261f26a42f0bde05791ce134`),
output root and independent all-row auditor. The exact prelaunch source,
method, authorization and command bundle is sealed at
`operations/sf-dlite-d8-full06-launch-proof-20260930/SEALED/MANIFEST.json`
(SHA-256 `7d827868c1ebac1d557fc6e31f7238b4d2ba42832af5897e72cda95da5c52649`).
No full04 or full05 prefix is adopted across
operator versions. Sealed source blocks are resumable only within full06's
identical pinned identity, with a 600-second bound per unsealed block and a
four-hour campaign cap. Before target construction, independent audit must
rederive the first target and all emitted-frame validity/diagnostic counters
from banked raw UCI lines and the source-qualified roster. Full06 is running
and unadmitted at this publication point; there is no complete terminal or
independent all-row audit result yet. This document gives zero label, target,
training, Elo or production-admission credit.
