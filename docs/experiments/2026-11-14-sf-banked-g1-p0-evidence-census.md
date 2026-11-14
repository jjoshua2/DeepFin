# Banked original-G1 Stockfish P0 evidence census

Status at **2026-11-14 11:48 UTC**: **PROSPECTIVE DATA-ONLY PREREGISTRATION —
SOURCE REVIEWED; 40 SYNTHETIC CASES PASSED, ACTUAL CENSUS UNRUN.** Root accepted
the closed synthetic validation; its independent actual review is pending. The
actual root input authority and census execution identities remain **UNBOUND**,
and the census is not execution-admitted. This record is prepared for main
publication alongside the reusable auditor; it does not launch work, admit
labels to training or select a teacher-training regimen.

Audit UUID: `f47bab16-fca5-46b8-bb7b-d1d2d3dff172`.

## Question and fixed population

What source-qualified Stockfish P0 evidence and missingness survive in the banked
**original G1 TRAIN** shards? Census the complete authenticated generation:
**256 physical games and 24,827 retained rows**, with producing numerical source
`090b5143315a0faef7ab0ac1a90e2d5cebf3b929`. Its fixed generation manifest is
`core-baseline-research-g01-audit-results-v1/generation.json`, SHA256
`35279e2ef19cdf808e4b4e25bb50fbb524750693ee035c1d44080d87a3db7878`.
There is no sampling, substitute game, favorable subset, later-generation pooling
or HOLD/archive inclusion. Actual label coverage is UNKNOWN before this census;
availability flags or previously sampled fractions are not P0 coverage evidence.

The source supports two distinct routes. Existing `sf_p0_policy_target` and
`sf_p0_regret` were shifted from the immediately preceding **actual ply** before
producer thinning. Separately, a retained row's attached P1 annotation can
associate with another retained row's P0 only when their actual plies differ by
one in the same physical game and full history. Adjacent retained row offsets
across a ply gap do not establish that association. The census measures which
route and evidence tier survives, without assuming a complete native P0 teacher.

## Fixed method and technical yardstick

Use the [read-only census CLI](../../scripts/sf_banked_p0_census.py) and its
[usage/schema guide](../sf_banked_p0_census.md), candidate source
`70db0b6f4c8b18a9005983e849c66802ec572334`, with the exact source/command templates
in the V2 packet below. Root must first authenticate the existing generation,
producing protocol/source, original evidence and immutable inventories. The
`ROOT_ACCEPTED_CORE_TRAIN_GENERATION_V1` name alone is insufficient: the CPU
producer writes that candidate before its final seal and owner closure. The
census therefore requires the separately hash-pinned
`ROOT_AUTHENTICATED_INPUT_AUTHORITY`, joined to the selection, exact generation,
source-audit protocol/readout, successful closed CPU audit owner and exact child
command. Original GPU producer closure also remains required. Root will bind the
legacy accepted G1 receipts; no such actual authority is issued by this draft or
by the child census.

Reconstruct full legal opening/played histories and raw-record/retained-row
identities. Check physical game, serialized game/ply, TRAIN/NET/selfplay roles,
STM/POV/clock context, input-F16 fingerprints, legal compact masks, source and
served-state lineage. Stored input hashes join the accepted original encoder
proofs; this is not a fresh encoding or F32-parity claim. Authenticate all shard
files before and after bounded per-game decoding. Wrong hashes, roles, sources,
duplicates, illegal/shifted histories, changing inputs or unsafe storage fail
technical qualification; they are not a clean negative coverage result.

Preserve every retained-row denominator and all structural raw records in
per-row evidence and per-game/generation summaries. Keep exclusive join statuses
separate from overlapping reason counts and support/ply/observed-effort
distributions. An unavailable label is not a negative teacher value. Malformed
or inconsistent reduced labels stay in the report with explicit invalid reasons;
no imputation, dropping or replacement is permitted. Both optional payload and
flag absent is unavailable; payload surviving without its flag is ambiguous and
invalidates that route even if another SF field is valid. Ordinary flag-zero
storage remains a separate absent case.

Respect the actual storage contract: F16 vectors, U8 masks/flags, up to 48 sparse
int16 MultiPV rows and int32 record metadata with source sentinels. Native-query
W/D rounded to permille and reconstructed L remain distinct from record-POV
flipped or producer-declared CP-logistic F16 WDL. Check the existing declared
conversion; select no new transform. CP/mate boundary values are possibly
clipped. Partial legal support and possible fallback are explicit; a normalized
policy or legal best move does not prove a full teacher distribution. Explicit
P0 regret is not a probability distribution, and its transform is not rebuilt by
this census. Bounds, original wire/query IDs/history, full PVs, escalation and
per-candidate requested/actual effort absent from CORE remain **UNKNOWN**.
Full wire is not required for the narrower source-reconstructed reduced-label
tier, but that tier is not wire-authenticated or exact/unbounded evaluation.

Technical completion requires the exact full population, successful custody and
correctness gates, all 40 separately qualified synthetic cases, complete output
denominators and independent review of the closed actual census. It does **not**
require any minimum SF availability or coverage percentage. The deciding output
is descriptive evidence/missingness, not a gain, calibration, confidence-interval
or teacher-strength test. Report games as physical clusters; do not invent
independent-row intervals or infer label adequacy from fit or source review.

## Engineering controls, resources and recovery

The authored synthetic suite uses independently expected small games and cases
for actual-ply gaps/thinning, native versus flipped/transformed WDL, missing or
malformed labels, partial support, fallback and unknown bounds. It also checks
wrong game/role/source/hash/history/POV/input identities and input custody. Ten
authority negatives cover absent/pending/foreign root authority, pending/failed/
unclosed CPU audit and wrong manifest/protocol/readout/command, before numeric
payload loads. Three flag-contract branches preserve rows while distinguishing
unavailable pairs from malformed P1/explicit-P0 evidence.

All **40 synthetic cases passed**, with zero failures, errors or skips, under the
separately admitted CPU owner. Root accepted its closed PASS result: 6.512032907
owner-seconds and 5.19 CPU-seconds. The fresh validation checkout was pinned to
`70db0b6f4c8b18a9005983e849c66802ec572334`; all project and auditor module origins
were confined there, no project `.so` was present or loaded, Torch used two CPU
threads and CUDA remained uninitialized. Source custody checked all 1,327 pinned
files unchanged before and after. This qualifies the exercised synthetic routes
in that import environment, not an installed native image, actual G1 label
coverage or full CI. Independent actual validation review is **pending** at this
cutoff. Static/source acceptance and the synthetic pass do not substitute for
the fixed-bank census or its separate input authority.

| Phase | CPU outer / inner seconds | GPU seconds | Current state |
| --- | ---: | ---: | --- |
| Synthetic qualification | 300 / 270 | 0 | CLOSED PASS: 40 cases; independent actual review pending |
| One fixed original-G1 census | 900 / 870 | 0 | UNRUN; input authority/execution bindings UNBOUND |

At most one numeric owner, two threads on cores 20,22, CUDA hidden. The maximum
prospective CPU sum is 1,200 owner-seconds, not an ETA, entitlement or automatic
retry budget. Source templates bound metadata to 128 MiB **per input metadata
file**, decoded arrays to 64 MiB per game, inventory to128,000 files per game and
`rows.jsonl` to 256 MiB. Stream one game at a time, without all-corpus arrays.
The public Python move helper can import its optional native module; no existing
ignored image is qualified by source equality or by not calling a native kernel.
The synthetic owner demonstrated an isolated environment with no project native
images; the eventual census must preserve that qualified import/source binding
or separately qualify any different environment. No P8 ignored image is adopted
by this evidence.

The exact proposed child argv are captured in `command.TEMPLATES.json`; the
read-only protocol/authority templates retain explicit UNBOUND fields. Source,
identity, owner, fresh-output and final protocol hashes must be reviewed and
bound before use. Reviewed reusable tooling and this record must be published
before the actual census; publication alone is not resource or execution
admission. Root alone launches, with fresh controls/resources/deadline and closed
prior ownership. This audit constructs no model, optimizer, Torch/ORT session or
TB handle and sends no SF/search/teacher query.

Preserve the original corpus, labels, proofs and every failed/partial attempt.
Failures after output allocation preserve the partial row stream and an external
failure marker; earlier gate refusals remain in the owning stage's logs. No
incomplete attempt receives technical or training credit. Any source/estimator
correction must preserve the earlier attempt and reuse these exact banked inputs;
there is no automatic rerun, extension or data substitution. If evidence is
insufficient, report the missing tier rather than silently issuing new queries.

## Interpretation and evidence

This is data qualification only. It selects no loss/teacher mix, model
initialization, optimizer, LR, replay, training dose or new learning experiment.
It supplies no data/model/champion promotion, CONF allocation, playing-gain or
external Elo claim. Earlier baseline and successor negative/unresolved results
remain unchanged; the long-term strength objective is not fulfilled by this
census. Any training or strength follow-up needs a separate reviewed decision.

Bulk evidence root: `/home/josh/chess-artifacts/operations/rl-week-one-20261112-v1`.
Paths below are relative to that root. No actual G1 payload was read to author
this record. V1 source findings were corrected prospectively in V2: the candidate
schema needed closed-audit/root authority, and payload-without-flag needed an
invalid classification. V1 and its review remain preserved; there was no numeric
result reroll.

| Evidence | Path | SHA256 |
| --- | --- | --- |
| Root data-only selection | `core-SF-banked-G1-data-qualification-selection-v1/decision.json` | `3278a9177246fdfdea2098f19b1504cd3112b457c5fe59868d0442c588a32e3a` |
| Source plan | `core-future-SF-P1-to-P0-banked-join-source-plan-v1/source-evidence-manifest.json` | `47d8ab8876025252dc6a755b271696caa8ae7a83ece175121257cbf1fe39476d` |
| Independent plan review | `core-future-SF-P1-to-P0-banked-join-independent-source-review-v1/review.json` | `7552bf4143454cf7159340233b85074b118d9c199791ffa7cd3df9c1c38251b9` |
| Corrected implementation V2 | `core-SF-banked-G1-auditor-implementation-v2/manifest.json` | `b1f84ebbc5e0a570fc61b4c5a4a349e27d892db4679cf3c0ec6cfd2c61cbe7d3` |
| Independent source V2 acceptance | `core-SF-banked-G1-auditor-independent-source-review-v2/review.json` | `e0b3beabd14c1917a6ce4b694a7e1b69997020f6eed0638d092cb94d5fc65db9` |
| V1 changes-required review, preserved | `core-SF-banked-G1-auditor-independent-source-review-v1/review.json` | `ed53111b5f407c2bdcfafeb3ccdeed1bedc6d43c4cead805e289ed816df47212` |
| Root synthetic validation acceptance | `core-SF-banked-G1-synthetic-root-validation-v1/closed-focused-tests.json` | `7b92f02348dd24246443495461d46324f46261dcfd314ff7bb57ee72f2a1f50f` |
| Closed synthetic CPU owner | `core-SF-banked-G1-synthetic-CPU-execution-v1/terminal.json` | `4f771f9a3b0f7a6052bdea1fbbe387d8919f6eefbcd4e777ab74d901ffafc9fd` |
| Synthetic import/source custody | `core-SF-banked-G1-synthetic-results-v1/custody.json` | `7024eaeb21b809746f836982830fd5904245d616ce89a9cd1d5bf17c54893649` |
| Actual 40-case JUnit | `core-SF-banked-G1-synthetic-results-v1/pytest.xml` | `fccbde42aebfbd6b2a1de5c06be5de2dd9749a8227916e1dfed30243329cc062` |

Actual root input authority and final census protocol/commands/admission remain
**UNBOUND**; actual G1 census observations remain **UNRUN**. Independent actual
review of the completed synthetic validation is pending at this cutoff.
