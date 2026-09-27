# BT4 own-game teacher source, September 23

The question was whether an ordinary generated-game bank could retain its
already computed BT4 root policy and native WDL as reusable teacher channels,
without a second inference pass or fabricated Stockfish targets. The deciding
gate was a fresh full-bank strict replay against the frozen ordinary stage,
followed by an exact-copy source publication. This readout establishes source
integrity only; cohort mixing, training targets, and playing strength remain
separate decisions.

The input was the completed `run04_ordinary_retry01/ordinary_opening` bank at
`~/chess-artifacts/operations/bt4-root-gpu-qualification-20260923`.
The frozen stage had 8 completed games, 0 discarded, 752 accepted rows, 6
natural and 2 rule50 Syzygy endings. The audit reused its plan SHA
`4bce3ad1ba52ae3c34410f58bb1adc017b4f209448716f48327857766c93bbda`
and full verifier SHA
`aa2e16ab1a78973f46839bc6228c4d783ceb6097db55f9f5a5d576fec3103d86`.
The revised audit producer SHA was
`b53b94cdbaa33f432225c9476dc4e9f90b6b0fbcefad9f5ace3ee4c0c61eba96`.

The CPU-only audit **passed**. It replayed all saved games through the frozen
strict verifier, matched every returned fact against the original terminal,
checked the summary and CUDA provider proof, and found the bank tree unchanged
before and after. The [audit receipt](evidence/bt4-own-game-source-20260923/ordinary_retry01_audit_v2.json)
has SHA `aa7d3f8fb01bc6ca5a4403622effbfbcf1c311e9bf5ff209032d5022ff0db625`.
The adapter then published one 752-row Zarr source from that exact receipt. Its
[manifest](evidence/bt4-own-game-source-20260923/manifest.json) has SHA
`02c5036714541a5a24c85fdbd2f41f2185e872c58c1a1094d701b412e92ffd4a`.
The source output is `/tmp/bt4-own-label-reuse-20260923/ordinary_retry01_source_v2`.
The source carries separate `bt4_policy`, `bt4_wdl_raw`, saved input `x`,
bank-qualified row identities, and observed outcomes. It declares
`training_ready: false`: SF policy/search WDL and Ceres labels are absent.

The full repository `scripts/lint.sh` gate passed with Ruff, basedpyright
(0 errors, 0 warnings), and Vulture. Twelve focused CUDA-hidden CPU tests
passed. The type gate used the project's pinned Python 3.10 environment and
four local native-extension links for worktree resolution; its temporary
config and links were removed afterward. No GPU work was launched by this
adapter or audit.
