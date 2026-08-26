# FastQ-4+ — implementation record

`docs/fastq_design.md` is the spec and stays the authority. This file records what
was BUILT and what was MEASURED, including the three places the implementation
departs from the spec and why.

Arm name `nnue-fastq`. Default OFF everywhere: nothing in the training loop
selects it, and no config key reaches it. It is reachable only from
`_nnue_ext.arm_open("nnue-fastq", ...)`.

## Files

| File | What |
| --- | --- |
| `chess_anti_engine/nnue/_fastq_see.h` | §5 static exchange evaluation: full swap with x-ray reveal, en passant, promotion |
| `chess_anti_engine/nnue/_fastq_certificate.h` | §3.1 quiet certificate — the position-only predicate |
| `chess_anti_engine/nnue/_fastq_search.h` | §3 search, §4 DAG rules, §6 knobs, §7 counters |
| `chess_anti_engine/nnue/_nnue_dag_store.h` | `quiet_bits` node payload + the `CAE_DAG_CERT_*` bits |
| `chess_anti_engine/nnue/_arm_providers.h` | the `nnue-fastq` provider and its config snapshot |
| `scripts/fastq_reference_arm.py` | §8 reference-arm harness |
| `tests/test_fastq_see.py`, `tests/test_fastq_search.py` | §8's verification plan |

The certificate is a separate header from the search on purpose. §4.1 permits
caching it against a canonical node and §4.2 forbids caching anything the search
computed; what separates the two is whether the quantity is a function of the
position alone. `_fastq_certificate.h` has no access to a window, an alpha, a beta
or a context, so that boundary is checkable by reading the includes rather than
by trusting a comment.

## Measured — §8 reference arm

467-row parity corpus (`tests/test_qsearch_dag_parity.CORPUS`), production net,
both arms at qply 4 (`QSEARCH_MAX_PLY` is compiled at 4, so they are
depth-paired), FastQ at its §6 defaults (node_cap 32, delta_margin 200,
recapture exemption on).

    PYTHONPATH=. python3 scripts/fastq_reference_arm.py --pack <production.pack>

### NNUE evaluations per call

| arm | mean | median | p90 | p99 | max | <5 | 5–20 | >20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `nnue-qsearch` | 431.84 | 35 | 958 | 6018 | 14403 | 0.212 | 0.233 | 0.555 |
| `nnue-qsearch-dag` | 342.67 | 32 | 798 | 4388 | 8250 | 0.231 | 0.221 | 0.548 |
| `nnue-fastq` | **6.44** | 4 | 15 | 31 | 32 | 0.520 | 0.424 | 0.056 |

§1's target is 5–20 evaluations per leaf. FastQ's mean is 6.44 and 94.4% of calls
sit at 20 or below; the distribution's whole tail is the 32-node cap.

⚑ **431.84 IS NOT §1's "~72 AVERAGE", AND THE RATIO IS THE PART THAT TRANSFERS.**
This corpus is deliberately tactical-heavy — it is every capture child of a set of
tactical FENs, built to make the DAG share subtrees — so qsearch costs far more
here than on a general leaf population. The same-rows reduction is **67×**
(431.84 → 6.44); quoting FastQ's 6.44 against §1's 72 would be comparing two
different populations, which is a mistake this repo has made before.

The middle row separates the two independent changes: the DAG substrate alone
buys 431.84 → 342.67 (1.26×), and the move policy plus pruning buys the remaining
342.67 → 6.44 (53×). Attributing all 67× to either one would be wrong.

### Value agreement vs `nnue-qsearch`

| | |
| --- | ---: |
| **sign agreement (§8 primary)** | **458/467 = 0.9807** |
| identical | 379/467 = 0.8116 |
| median \|Δ\| | 0 |
| p95 \|Δ\| | 341 |
| \|Δ\| > 50 | 77/467 = 0.1649 |
| \|Δ\| > 100 | 63/467 = 0.1349 |
| \|Δ\| > 250 | 33/467 = 0.0707 |
| mate-band rows | 8 (7 differing) |
| non-mate rows | 459: mean \|Δ\| 41.95, p95 298, max 1211 |

Mate-band rows are reported separately because a mate score is not a large
centipawn number: one row where one arm found a mate moves a 467-row mean by
~200, swamping every genuine evaluation difference in the other 466. The
whole-corpus mean \|Δ\| including them is 1528.96 and means nothing.

`nnue-qsearch-dag` vs `nnue-qsearch` is bit-identical on all 467 rows, as #472
established — it is carried in the harness as the control that proves the
substrate is not what moves the numbers.

### Where the 16 differing rows come from — attributed, not assumed

**7 differing mate-band rows: the check policy, and it is a stated non-goal.**
All 7 offer a quiet check at the root (7/7, against a 111/467 = 23.8% base rate).
qsearch generates checks at its first quiescence ply (`QSEARCH_CHECK_PLIES` = 1)
and finds a forced mate; FastQ never generates a quiet check, so it returns a
quiet value. §9 defers quiet-check generation and §10 assigns mate detection to a
later verifier layer on the same DAG. This is the design working, not a defect —
but it does mean **FastQ must not be used as a mate oracle**.

**9 sign disagreements: the SEE gate and delta pruning, NOT the check policy and
NOT the budget.** Only 1 of the 9 offers a root quiet check — 11%, *below* the
23.8% base rate, so the check policy is ruled out rather than merely unmentioned.
They are ordinary pruning differences: qsearch searches every capture, FastQ
declines SEE-negative ones (0–13 SEE prunes on these rows) and delta-prunes
against the window (0–18).

**The node cap is not the driver.** It tripped on 1 of the 17 differing rows and
on 11 of the 450 agreeing ones. §3.4 calls it a tripwire rather than a tuned knob,
and `budget_trips` is what makes that claim checkable.

⚑ **AGREEMENT WITH qsearch IS NOT THE TARGET AND 0.9807 IS NOT A STRENGTH
RESULT.** §8 says so explicitly: the deciding readout for any production claim is
the downstream standardized primary against deep SF. qsearch-4 is a cheap
first-pass reference that answers one question — is FastQ resolving the same
tactics, or playing a different game — and nothing else.

## Deviations from the spec

**1. FastQ is not a fourth `CaeQsearchSubstrate`.** §8's integration notes
suggested extending the `CAE_QSUB_*` enum. The enum is documented in
`_arm_providers.h` as "where a node's stand-pat NUMBER comes from — nothing else
about the search varies with it", and `tests/test_nnue_incremental.py` asserts
each qsearch wrapper selects a distinct value of it. FastQ differs in move policy,
pruning, recursion shape and budget, and agrees with `nnue-qsearch-dag` on exactly
the one thing the enum names. Adding `CAE_QSUB_FASTQ` would have made the enum
mean two incompatible things and forced every qsearch branch to ask which search
it was in. FastQ owns its own eval entry point; the substrate enum stays at three.

**2. §8.4's crafted repetition fixture is not constructible, so that mutant
survives.** FastQ generates captures, promotions, and — in check — evasions.
Captures and promotions change the piece multiset, so a cycle would need every
edge to be an evasion and every node on it to be in check: an unbroken
mutual-perpetual-check loop. Measured four ways, all zero — including a
constructive hunt over 1,857,031 random sparse positions (831,189 of them in
check) DFS'd to depth 8 requiring every edge to both evade and give check. The
guard is KEPT anyway: §4.3 mandates it because the DAG admits back-edges, and §9's
deferred quiet-check generation makes reversible non-check edges exist the moment
it lands. `test_no_cycle_is_reachable_under_the_current_move_policy` asserts the
measured property so it FAILS the day the policy widens, at which point the
fixture becomes both constructible and required. Full reasoning is in that test's
header block.

**3. §5's "no full SEE exists today" premise is wrong, and the duplication is
deliberate.** `encoding/_features_impl.h` already has `feat_see_capture`. It is
not reused: it lives in a different extension, uses pawn units rather than
centipawns, handles neither en passant nor promotion, and — decisively — feeds the
`FEAT_EXTRA_V3_SEE` production model input planes, so widening it to satisfy §5
would silently change the training input distribution. The four reasons are in the
⚑⚑ block at the top of `_fastq_see.h`; read it before "unifying" them.

## Knobs (§6)

| key | default | effect |
| --- | ---: | --- |
| `max_qply` | 4 | tactical recursion depth; does NOT bound evasion recursion (§3.2) |
| `node_cap` | 32 | §3.4 tripwire, counted by `budget_trips`; 0 disables |
| `delta_margin` | 200 | delta pruning slack, in internal units |
| `see_recapture_exempt` | 1 | keep a SEE-negative capture on the square the parent captured on |

⚑ **A CONTEXT SNAPSHOTS THE KNOBS AT `init()`.** `fastq_set_config` affects
contexts created after it, never a running one, and `fastq_stats` reports the
context's OWN snapshot rather than the module globals — which is how a caller can
tell the two apart. Each knob has a test that sets an extreme value and reads the
effect out of the search's own counters; each of those tests was run against a
mutant that ignores the context and uses the compiled default, and each killed it.

## Counters (§7)

The evaluate-once identity is asserted, in the form that is actually true:

    nnue_evals + nodes_created_in_check == nodes_created

⚑ The spec words it as "NNUE evaluations must equal nodes created". That holds
only where every created node has a static value, and an in-check node
deliberately has none — the store publishes it with `value_valid = 0` because the
NNUE evaluation is undefined in check. Asserting the spec's wording verbatim would
fail on any position containing a check; the in-check term is the same claim,
stated so it can be checked. `scripts/fastq_reference_arm.py` exits non-zero if it
breaks, and `_assert_counter_identity` checks it in every search test.
