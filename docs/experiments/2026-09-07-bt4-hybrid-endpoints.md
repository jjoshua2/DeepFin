# BT4 hybrid and global endpoints: next development screen

Registered September 7, 2026, after the [fresh C/E confirmation](2026-09-07-bt4-fresh-confirmation.md)
completed and before training or playing outcomes for the new arms. **Status:
CPU qualification and implementation in progress; no new training or arena launched.**
The previous screen is complete. This registration starts the next bounded research
stage; it does not commit a 168-hour queue or change production.

## Question and selected arms

C20T05 remains the incumbent. Its fresh seed-one advantage over E0T05 is
+37.67 Elo [18.72, 56.85]. C's earlier win over G20T05 compared complete recipes:
G20 both added global BT4 and weakened BT4's control within C's selected set.
It did not establish that outside coverage itself is harmful.

Let S be the legal-normalized stored SF target, B be the legal-normalized raw
BT4 policy sharpened at teacher temperature 0.5, and C be the legal-normalized
stored C20T05 target. Preserve all other training targets and fields.

| New arm | Target | Question |
| --- | --- | --- |
| H20 | 0.8 C + 0.2 B | Does broader BT4 help while retaining C's conditional distribution among selected moves? |
| B100 | B | Does the missing BT4-only policy endpoint outperform the current mixture? |
| G50 | 0.5 S + 0.5 B | Was the tested global dose too small? This corpus is already prepared. |

C's set A is the union of all stored SF maxima and d9 top-three moves within
20 effective cp. It can exceed three moves because stored maxima may reflect
quantization or saturation. C redistributes the set's entire existing mass using
BT4 T0.5, while preserving the original target outside A.

Before storage rounding, H20 preserves C's conditional distribution within A.
At the same global dose, H20 and G20 have identical outside targets, so their
comparison addresses selected-set redistribution. H20 versus C addresses the
broader change in allocation while preserving internal conditional probabilities.
These are complementary contrasts; do not subtract their Elo estimates to infer
another matchup. Outside mass changes by 0.2 times (B(outside) minus C(outside)),
which can be negative. A 20% mixture is not necessarily 20% outside probability.
Whole-row normalization of stored float16 C and subsequent storage cause small
additional deviations; qualify and report those rather than claiming byte identity.

## Preparation and training identities

Reuse the original 18,910,484-row, 2,309-shard, 97,968-game corpus and existing
raw history-aware BT4 and d9 rank sidecars. H20 must retain explicit original-SF,
C-parent, raw-BT4 and rank lineage; a generic bypass of source identity checks is
not acceptable. Its SF-agreement audit is descriptive: arithmetic, alignment,
legal mass and non-policy preservation decide preparation fidelity; SF agreement
does not replace the registered playing comparison.

The [initial H20 feasibility receipt](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/H20.feasibility.json)
(SHA256 `554fe4c09f73bce3bdb23269f96285a1c00d8b4cc4cf4eca1f5fcde431d9b6c3`)
reused 128 banked training rows from 43 games. It reconstructed stored C and G20
exactly, checked all 16 non-policy fields, and found H20/G20 outside differences
at most 0.0000610352, within the combined normalization/storage tolerance.
This is a bounded construction check, not a full-corpus or held-out strength check.
The existing mixer rejects C as a nested source and its ordinary global audit
reconstructs S, so explicit composition and provenance support must precede H20
materialization. No target corpus has been generated merely by publishing this plan.

Train each new arm once from scratch at **seed zero**, using the qualified frozen
wise-cloud runtime `7ec261509fb7345cf1ca0ad73809193fc2749bb1`: Python 3.10.12,
Torch 2.11.0+cu128, CUDA 12.8 and NumPy 1.26.2. Retain the pinned trainer/config,
optimizer and initialization procedure, batch 512, one complete game epoch,
16 planning/loading workers and 88-step windows. Expected rows and batches are
18,910,484 and 36,935. Require finite window diagnostics, zero nonfinite skips or
CUDA retries, complete realized rows/batches, and actual source-normalized schedule
`dc687fc333295dee565d19bb4f20da5aa95479dba3aacc5499c22a4004acc64f`.
Select the registered final checkpoint, not an intermediate winner.

Reuse qualified seed-zero anchors rather than retraining a seed-one G20 bridge:

| Anchor | Final checkpoint SHA256 |
| --- | --- |
| C20T05 | `8a355d29f7d3eee5deec4b3a16a6625d23baebe1302e3a8f2dea9136939e1db3` |
| G20T05 | `bd8c208a95247373f423be0649329e9c100db64ab1b5a5e68fd7a6aec3769a74` |

Their completed training/schedule/runtime evidence is in the
[published catalog](evidence/bt4-bootstrap/README.md). Bind new corpus summaries,
prospective schedules, actual executable/configuration hashes and reference
identities in the launch manifests before execution. Check actual staging after
training. Preserve the historical `valid_control:false` limitations: no held-out
purity receipt, committed configuration premise, and a game-epoch sampler different
from the old replacement-sampled control. These remain development comparisons.

## Fixed playing comparisons and order

Train H20 and complete its registered comparisons first, then B100 and its
comparisons, then G50 and its comparison. Target preparation for later arms can
proceed during earlier GPU work when memory and I/O permit. Existing raw BT4
labeling retains the GPU during preparation and between owned experiment stages.

| Candidate / reference | Simulations per side | Games / opening pairs | Role |
| --- | --- | --- | --- |
| H20 / C20T05 | 100 | 1,000 / 500 | Primary hybrid strength screen |
| H20 / G20T05 | 100 | 1,000 / 500 | Selected-set construction at equal global dose |
| H20 / C20T05 | 400 | 500 / 250 | Predeclared higher-search probe |
| B100 / C20T05 | 100 | 1,000 / 500 | Policy endpoint strength screen |
| B100 / C20T05 | 400 | 500 / 250 | Predeclared higher-search probe |
| G50 / C20T05 | 100 | 1,000 / 500 | Larger global-dose screen |

Run H20 and B100's higher-search probes after valid completed training even if
those candidates lose at 100 simulations. G50's first-pass conclusion is limited
to 100 simulations; this screen does not dismiss that family at higher budgets.
All three arms are selected in advance; a shallow H20 loss does not cancel B100.

Use the same qualified training search dictionaries with explicit prior temperature
1.0 on both sides. Keep move temperature 0.1, maximum 300 plies, compilation,
128 concurrent games and evaluator batch 4096. Use original development book seed
42 with 16 opening plies. The 400-simulation bank uses exactly the first 250
opening pairs of its candidate's 100-simulation comparison; verify endpoint and
color identities. Neither the seed-one reserved confirmation bank nor future
confirmation seeds are development inputs here. Bank original game rows, full
settings, identities and charges. Read each fixed bank only after it completes;
no rolling outcome decisions or outcome-dependent extension.

## Readout and allocation rules

Use candidate-minus-reference Elo from the opening-pair sample-variance normal
95% score interval, transforming score endpoints to Elo. Require exactly the
registered complete pairs, with no duplicates or orphans, before reading strength.
For the three primary candidate-versus-C screens:

- Estimate at least +15 Elo and lower bound above zero: promising.
- Otherwise upper bound at least +15: still competitive at this precision.
- Upper bound below +15: deprioritize this tested recipe/budget for ordinary
  refinement. This is not equivalence or a rejection of its entire family, and
  never cancels a predeclared higher-search probe.

For H20 versus G20, a wholly positive interval favors H20, a wholly negative one
favors G20, and crossing zero is unresolved. Interpret the result as a comparison
of the specified inside-set target constructions, conditional on shared outside
targets up to verified storage tolerance.

The 400 probes have lower precision and describe performance at that budget.
Any secondary 400-minus-100 score contrast must use the same 250 opening pairs
at both budgets, with an aligned-pair bootstrap (10,000 PCG64 replicates, seed
20260903); label it exploratory. A win at 400 alone is not proof of a growing
search advantage. Keep per-arm results visible, including losses and invalid runs.
Independent completed-bank/provenance review precedes the next research choice.

## Resource envelope and continuation

Expected compute is **12–14 GPU hours**. Hard stage caps, including termination
allowance, are 4.5 hours per training epoch, 1.5 hours per 100-simulation match,
and 1.5 hours per 500-game higher-search probe. These sum to **22.5 GPU hours**:
9 hours for H20, 7.5 for B100 and 6 for G50. The enclosing cap is **30 GPU hours**;
unused allowance does not create additional arms, retries or longer horizons.
Necessary GPU qualification, if any, is separately logged within that enclosing
cap and limited to 0.5 hour. Do not launch a stage without its remaining allowance.
CPU-only schedule verification is capped at 30 minutes per stage and releases the
GPU lease; lease waits do not count as GPU compute.

Keep the shared GPU lease, surviving timeout supervisor, owned-child cleanup,
STOP handling and durable completion/failure receipts. Preserve partial outputs
for explicit diagnosis rather than automatic retries. Use the established low
CPU/I/O priority and two numerical threads, preserve generators, and keep at least
150 GiB free. Provisionally allow 32 GiB for each new H20/B100 corpus, then check
actual growth; this is a planning allowance, not measured compression. G50 needs
no rematerialization. Account for training outputs and concurrent generation too.

Publish preparation findings, implementation PRs, launches and completed readouts
in this record and its linked compact evidence. A plan or successful target check
must not be reported as launched training. This stage replaces an open-ended grid
with three substantive candidates. Longer-training qualification, consistent G10
observation selection, fair search calibration and fresh finalist replication
remain the next decisions; no TailRL training, production adoption or 168-hour
allocation is implicitly started by this registration.
