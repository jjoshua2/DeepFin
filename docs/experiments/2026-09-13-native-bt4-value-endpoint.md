# Native BT4 value endpoint on the matched 35M corpus

Selected September 13, 2026, before reading the running Ceres100 policy match.
Status: preparation selected and metadata disk accounting completed; explicit
V100 qualification support is pending. No V100 corpus, training run or match has launched.

## Question and control

The [completed 35M comparison](2026-09-13-combined35m-value-transfer.md) favored
50% SF / 50% native BT4 WDL over SF-only WDL by **31.30 Elo [8.53, 54.34]**.
That establishes neither the best mixture nor whether SF improves on BT4-only
value supervision. Test this missing endpoint before refining small mixture doses.

Candidate **Combined35M_V100** changes only `search_wdl` to normalized native
BT4 side-to-move W/D/L probabilities, stored as float16. Use the existing writer's
`--alpha 1`; do not softmax probabilities again or derive them from policy logits.
The direct control is completed **Combined35M_V50**, checkpoint
`6d36f93d040c8babed040159419279af62a444f6466f42b720e12bb72d67ab02`.

Both use the identical frozen **35,314,577 observations / 4,321 shards / 21
sources**: 18,910,484 original observations and 16,404,093 selected G10 observations.
These are observations, not unique FENs. Native BT4 labels already cover this
selection; no new teacher inference or deeper SF search is required.

Retain B100 policy targets at teacher temperature 0.5 and every other stored
array, seed101 initialization, original frozen trainer/config/optimizer, one
game epoch, batch512 and two planner/two loader workers. The expected schedule
is **68,974 updates / 784 windows**, final window70, with canonical identity
`ca3922c459b321dd5e890f8b3e21f6fee6a24aa6c7f0d76653165616aeca67bd`.
Actual completion must establish that schedule rather than inherit its status
from the control. New host launches explicitly cap compiler concurrency at two;
that resource setting does not claim measured memory savings or bitwise equality.

## Head semantics

Both completed 35M controls report WDL target shares search1/SF0/outcome0.
Their categorical target is absent, categorical rebuilding is inert, and
categorical and SF-eval losses were zero in all 784 windows. The SF contribution
being tested is inside stored `search_wdl`, not the trainer's separate SF fraction.
V100 therefore means BT4-only **supervised WDL targets** under this configuration.
The coupled categorical output and shared policy/value features can still change
indirectly. This does not independently supervise every value-related output to
BT4. Preserve the same masks and objective; adding a categorical target or changing
auxiliary weights would be a different experiment.

## Preparation and compute limits

Use the original B100/SF-value input roots and their authenticated native-label
mappings. Do not feed already mixed V50 values back through a writer expecting
the original source. Extend existing admission and coordinator checks explicitly
for V100; a V100 corpus must never masquerade as a V50 receipt.

The supported writer copies complete shards. Account for actual source storage,
output growth and concurrent writers before launch; a value-only overlay is not
an admitted shortcut. The completed metadata census covers these exact 21 roots:
**26.245 GiB allocated**. Allow **32 GiB total output**, plus **8 GiB for other
writers** and the **150-GiB reserve**: require **190 GiB free at launch**. The
accounting snapshot had 201.812 GiB free; refresh before launch. These are planning
allowances, not a compressor bound. Enforce aggregate output and free-space limits
during the operation, and preserve protected controls.

[Compact readiness evidence](evidence/native100-readiness-20260913.json) records
the actual head masks/losses, exact-root metadata census and disk-budget derivation.
The metadata check took 36.114 seconds and read no array payloads; it is not a
dataset qualification or source-content verification.

- Full preparation: at most **4 hours** for sequential rewrites, with low-priority
  two-CPU execution and bounded memory. Completed comparable G10 work extrapolates
  to about 2.1 hours for 35M; this is a planning estimate, not a measured runtime.
- Dataset qualification and prospective schedule: at most **1,800 seconds each**.
- Training: **21,600 seconds**; coordinator **27,000 seconds**, plus at most
  30 seconds of outer termination allowance. One actual epoch, no automatic retry.
- CPU match preparation: **600 seconds inclusive**. Match: **7,200-second stage**
  within **12,030 seconds inclusive** for the enclosing operator.

Keep 48-GiB startup and 32-GiB running available-memory floors, owned-process
cleanup and STOP handling. Exact source/output/implementation pins and concrete
commands must be saved before each launch. Preserve incomplete outputs and
receipts for diagnosis; a failure does not authorize an automatic rerun or a
different cohort. The separate Ceres policy match continues unchanged.

## Deciding comparison

After valid completion, run **V100 versus V50, 512 games / 256 swapped pairs,
400 simulations, priors1, seed20260913**, with the same qualified search runtime,
16-ply openings, maximum300 plies, no tablebases, rolling128 and batch4096.
Reuse development panel
`14470ee9bcf5fdfc822bb19988941ecf4bbe2a1ea739d46487d3663b1735340c`.
It is already used for recipe development, not fresh confirmation. Keep the
existing paired-score estimator and its nominal 95% interval; read the completed
bank, not a rolling score.

A complete paired score interval wholly above0.5 supports V100 in this setting;
wholly below0.5 supports retaining the SF contribution in V50. If it crosses0.5,
the result is unresolved and V50 remains the provisional value recipe. No extra
games, mixture sweep, training seed or promotion follows automatically.

Retain the three historical validity limitations: no held-out purity receipt,
committed rather than current live-config premises, and game-epoch sampling
distinct from the historical replacement control. This one-seed comparison
does not settle 100M transfer, RL behavior, other teacher mixtures, draw-versus-Q
calibration, or value/policy interactions with a different policy teacher.
