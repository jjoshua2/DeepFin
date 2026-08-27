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

    PYTHONPATH=. python3 scripts/fastq_reference_arm.py \
        --pack <production.pack> --dump fastq_rows.jsonl

⚑ `--dump` writes one JSON object per row — both arms' values, the sign and
mate-band flags, per-row eval count and wall time, whether the cap tripped, the
SEE/delta prune counts, and whether the root offered a quiet check. **Every
attribution below is re-derived from that file, not from the aggregates.** A
printed mean cannot confirm or refute a claim about WHICH rows differ, so the
claim and its evidence would otherwise live in different places.

### NNUE evaluations per call

| arm | mean | median | p90 | p99 | max | <5 | 5–20 | >20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `nnue-qsearch` | 431.84 | 35 | 958 | 6018 | 14403 | 0.212 | 0.233 | 0.555 |
| `nnue-qsearch-dag` | 342.67 | 32 | 798 | 4388 | 8250 | 0.231 | 0.221 | 0.548 |
| `nnue-fastq` | **6.37** | 4 | 14 | 31 | 32 | 0.520 | 0.424 | 0.056 |

§1's target is 5–20 evaluations per leaf. FastQ's mean is 6.37 and 94.4% of calls
sit at 20 or below; the distribution's whole tail is the 32-node cap.

### Wall per call

| arm | p50 | p99 | max |
| --- | ---: | ---: | ---: |
| `nnue-qsearch` | 199.7 µs | 36.8 ms | 89.1 ms |
| `nnue-qsearch-dag` | 302.2 µs | 66.4 ms | 419.3 ms |
| `nnue-fastq` | **47.0 µs** | **346.6 µs** | 8.9 ms |

⚑ **§7 ASKS FOR WALL p50/p99 AND IT IS MEASURED IN THE HARNESS, NOT AS A C
COUNTER.** Deliberate, and the reasoning is worth stating because "the doc
promises a counter that does not exist" is the failure being avoided: a per-call
`clock_gettime` in the hot path buys a number dominated by whatever else the
machine is running (usually live training), and a p50/p99 needs per-call samples
— a histogram or reservoir — inside a search whose entire purpose is to be
cheap. The harness already loops per call, so it takes the same measurement for
free and identically for every arm. **Eval count remains the deciding
instrument** because it is deterministic and reproducible; these figures were
taken alongside a live training run and should be read as ratios, not
absolutes.

⚑ **431.84 IS NOT §1's "~72 AVERAGE", AND THE RATIO IS THE PART THAT TRANSFERS.**
This corpus is deliberately tactical-heavy — it is every capture child of a set of
tactical FENs, built to make the DAG share subtrees — so qsearch costs far more
here than on a general leaf population. The same-rows reduction is **67×**
(431.84 → 6.37); quoting FastQ's 6.37 against §1's 72 would be comparing two
different populations, which is a mistake this repo has made before.

The middle row separates the two independent changes: the DAG substrate alone
buys 431.84 → 342.67 (1.26×), and the move policy plus pruning buys the remaining
342.67 → 6.37 (53×). Attributing all 67× to either one would be wrong.

### Value agreement vs `nnue-qsearch`

| | |
| --- | ---: |
| **sign agreement (§8 primary)** | **458/467 = 0.9807** |
| identical | 377/467 = 0.8073 |
| value-differing rows | 90/467 = 0.1928 |
| median \|Δ\| | 0 |
| p95 \|Δ\| (all rows) | 341 |
| \|Δ\| > 50 | 77/467 = 0.1649 |
| \|Δ\| > 100 | 62/467 = 0.1328 |
| \|Δ\| > 250 | 32/467 = 0.0685 |
| mate-band rows | 8 (7 differing) |
| non-mate rows (459) | mean \|Δ\| **40.54**, p95 **298**, max 1211 |

⚑ **THE TWO p95 FIGURES ARE NOT INTERCHANGEABLE.** 341 is over all 467 rows and
is pulled up by the mate band; 298 is over the 459 non-mate rows and is the one
that pairs with the 40.54 mean. Quoting the non-mate mean next to the all-rows
p95 mixes two populations.

Mate-band rows are reported separately because a mate score is not a large
centipawn number: one row where one arm found a mate moves a 467-row mean by
~200, swamping every genuine evaluation difference in the other 466. The
whole-corpus mean \|Δ\| including them is 1528.96 and means nothing.

`nnue-qsearch-dag` vs `nnue-qsearch` is bit-identical on all 467 rows, as #472
established — it is carried in the harness as the control that proves the
substrate is not what moves the numbers.

### Where the differing rows come from — re-derived from the dump

**90 of 467 rows differ in value at all (19.3%).** Of those, **74 differ only in
MAGNITUDE** — same sign, outside the mate band, mean \|Δ\| 216.12 and max 1211.
Those are ordinary pruning differences and are not evidence of anything beyond
"the two searches are not the same search".

The **16** rows discussed below are the sign-or-mate subset: 9 that disagree in
sign plus 7 mate-band rows that differ. Saying "16 differing rows" without the 90
understates how much the arms diverge; saying "90 differing rows" without the
split overstates how much of it matters.

**7 differing mate-band rows: the check policy, and it is a stated non-goal.**
All 7 offer a quiet check at the root (7/7, against a 111/467 = 23.8% base rate).
qsearch generates checks at its first quiescence ply (`QSEARCH_CHECK_PLIES` = 1)
and finds a forced mate; FastQ never generates a quiet check, so it returns a
quiet value. §9 defers quiet-check generation and §10 assigns mate detection to a
later verifier layer on the same DAG. This is the design working, not a defect —
but it does mean **FastQ must not be used as a mate oracle**.

**9 sign disagreements: the SEE gate and delta pruning, NOT the check policy and
NOT the budget.** Only 1 of the 9 offers a root quiet check — 11%, *below* the
111/467 = 23.8% base rate, so the check policy is ruled out rather than merely
unmentioned. They are ordinary pruning differences: qsearch searches every
capture, FastQ declines SEE-negative ones (0–13 SEE prunes on these rows) and
delta-prunes against the window (0–18).

**The node cap is not the driver.** It tripped on 11 of 467 rows in total: 1 of
the 16 sign-or-mate rows and 10 of the 450 sign-agreeing non-mate rows. §3.4
calls it a tripwire rather than a tuned knob, and `budget_trips` — now recorded
per row in the dump — is what makes that claim checkable rather than asserted.

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
not reused because it lives in a different extension, uses pawn units rather than
centipawns, and is square-based rather than move-based — so it handles neither en
passant (whose victim does not stand on the capture square) nor promotion, both
of which §5 requires.

⚑⚑ **AN EARLIER REVISION OF THIS PR CLAIMED A FOURTH, DECISIVE REASON — that
`feat_see_capture` feeds the live net's input planes — AND IT WAS FALSE.** Its
output reaches only the `FEAT_EXTRA_V3_SEE` planes, and `v3_see` is a separate
65-plane family from the 63-plane `v2_threats` that `configs/pbt2_small.yaml`
selects; the graded-SEE planes are appended after index 63 and exist only when
`v3_see` is chosen. Changing `feat_see_capture` would **not** move the live input
distribution. This is the "diff the file you measured against production" trap,
committed inside a PR whose whole subject is that class of defect. The true
residual reason is weaker: `v3_see` is a shipped encoder with a closed
experimental verdict, so changing it would silently cost that verdict's
reproducibility.

`tests/test_fastq_see.py::test_the_two_SEEs_in_this_repo_must_not_be_unified`
turns the comment into a tripwire — it pins one en-passant case where the two
MUST disagree (FastQ scores it 100; the `v3_see` planes are entirely zero because
the loop only visits occupied squares), with an ordinary capture asserted first
so the comparison is not two silent zeros.

**4. §5's move ordering is implemented as specified, after the first version was
not.** §5 says "used for BOTH ordering (best SEE first) and gating — one
computation. MVV-LVA exists only as the pre-SEE tiebreak." The first
implementation sorted by SEE descending and left ties in move-generation order,
which is not the same thing. Measured before deciding: **233 of the 348 corpus
nodes with two or more tactical moves (67.0%) contain an equal-SEE tie, and
MVV-LVA reorders a tie group at 120 of them (34.5%)** — so this was a silent
dependence on an unrelated implementation detail across a third of all nodes,
not a corner case. Adopting the tiebreak also measurably lowered the cost
(6.44 → 6.41 evaluations per call, 6.37 after the recapture-square fix below)
at unchanged sign agreement, so it is both
what the spec says and the better of the two.

## Four more found in review — rule draws, the recapture square, one-shot stats, an inverting knob

**An in-check node did not consult the draw rules.** `cae_resolve_node` tests
mate, then `cae_resolver_is_drawn`, then recurses; FastQ owns its evasion
recursion (§3.2) and had only the mate half. A position already over by 50-move,
repetition or insufficient material had its evasions searched and returned a live
score — measured at 2103 where the reference arm says 0.

⚑ **THE ORDER IS MATE-THEN-DRAW AND IT IS NOT INTERCHANGEABLE.**
`cae_resolver_is_drawn` is `cboard_search_terminal`, which answers 1 for
CHECKMATE as well — it reports "terminal", not "drawn". Testing it first scores
every mate as 0.

⚑ **THE FACT COMES FROM THE BOARD, NEVER FROM THE NODE.** Halfmove clock and
repetition history are exactly what the canonical DAG position EXCLUDES (§4.2),
so drawn-ness must not be cached in the payload: two search paths reaching the
same structural node can disagree about whether it is drawn, and both are right.

⚑ **THREE OF THE FOUR DRAW FIXTURES CANNOT FAIL, AND THEY SAY SO.** Insufficient
material is closed under moving — a position with no pawns cannot gain material —
so every evasion of an insufficient-material node is also drawn and the searched
answer coincides with 0. Only a 50-move node with a clock-RESETTING evasion
(a capture or pawn move) discriminates, because that child carries a real
evaluation. `test_the_in_check_draw_fixture_can_actually_fail` asserts the
property that makes it discriminating.

**The recapture exemption was granted by non-captures.** §3.4 exempts a
SEE-negative capture only on "the square just captured on", and every child's
destination was being recorded — so a non-capturing evasion, or a quiet
promotion, handed the next node an exemption for a square nothing was captured
on. Both routes have a fixture (swept: 0 exemptions with the capture test, 1 and
3 without). Cost: 6.41 → 6.37 evaluations per call, values unchanged.

⚑ The control (`test_a_real_recapture_is_still_exempt`) needed its own sweep: the
original recapture fixture fires 11 exemptions on the production net and **zero**
on the synthetic one, because a PSQT-only pack moves stand-pat, which moves
alpha, which delta-prunes the capture before the exemption is consulted. A
control that is vacuous under the mandatory pack is not a control.

**`arm_eval("nnue-fastq", ...)` returned the resolver's zeros** — the third
instance of the family fixed twice already in this PR (`arm_stats`,
`fastq_stats`). It builds its own context and never sees a handle, so neither
refusal covered it. Dispatched rather than refused: the evaluation itself was
correct and only the counter block was wrong.

**`delta_margin` inverted at the top of its range.** Delta pruning skips a
capture when `stand_pat + victim + margin <= alpha`; in int32 a near-INT_MAX
margin wraps that sum negative, so "prune less" silently became "prune
everything" — measured, INT32_MAX pruned 33 moves where the correct answer is 0.
The comparison is now int64 rather than the knob being capped, because a cap
turns an out-of-range request into a different in-range one, which is the same
class of silent substitution. This doubles as §6's extreme-value mutation for
`delta_margin`.

## Two defects this PR shipped and then fixed

Recorded because both are instances of the class the PR exists to guard against,
and because the second one's *obvious repair does not work*.

**An in-check budget trip returned a supermate.** An in-check node has no
stand-pat, so the evasion loop seeds `best` at `-CAE_FASTQ_INF`; when the budget
refused the FIRST evasion, that seed left the function as −200000 and the parent
negated it to **+200000 — twice `RESOLVER_MATE_BASE`**, a score better than
mate-in-0 from a node the search never looked at. The §8 harness classifies
anything past the eval clamp as a mate, so it would have been banked as FastQ
finding a forced win. Reachable on ordinary rows, not just crafted ones: 13
corpus rows returned ±200000 at `node_cap=2`.

⚑ Every budget test asserted only `budget_trips > 0` and an evaluation count. All
of them passed with the bug in place, because none of them looked at the number
that came back.

⚑⚑ **"Return `alpha`, mirroring the path-ceiling branch" REPRODUCES IT EXACTLY.**
`cae_arm_fastq_eval` enters the root with `beta = +CAE_FASTQ_INF`, the root passes
`beta` down unchanged, and `cae_fastq_child_value` negates it — so a
first-generation child's `alpha` **is** `-CAE_FASTQ_INF`. Run as a mutant, that
repair failed with the identical `returned 200000`. The path-ceiling branch had
the same defect and was fixed alongside. The fix returns `cae_resolver_clamp(beta)`:
`beta` so an unsearched move cannot be *promoted* (returning `alpha` claims a
fail-low, which the parent reads as a fail-HIGH on the move that reached the
node), and the clamp because `beta` is `-alpha_parent` and an in-check parent's
alpha starts at `-CAE_FASTQ_INF` too — without it the escape just moves up one
level. The invariant is: **a node the search declined to look at never emits a
mate-magnitude score.** Real mates are untouched; they come from the terminal
branch.

⚑ No corpus row reaches the clamp half — swept at `node_cap` 1/2/3/4 with the
clamp removed, zero escapes. It needs a root in check whose evasions give check
back, which is a composed shape, so
`test_a_trip_below_an_in_check_ROOT_stays_in_range_too` carries a crafted fixture.
"We could not find it" and "it cannot happen" are different claims.

**The stand-pat was not clamped where `cae_qsearch_node` clamps its own.** The DAG
stores the raw NNUE value by contract, so the clamp belongs at every reader.
Unobservable with the production net (largest raw evaluation on the corpus is
4546 against a clamp of 32000) — which is exactly why it needed a test rather
than an argument: at a synthetic PSQT magnitude of 200,000, **147 of 444 non-check
rows evaluate past the clamp and reach 89,044**, inside the mate band.

**`fastq_stats()` answered zeros for any non-FastQ handle** — the mirror of the
`arm_stats()` refusal added in this PR's fifth commit, and a reminder that fixing
one direction of a defect and leaving the other is this codebase's documented
failure mode for exactly this class. Both directions now raise.

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

⚑ **`beta_cutoffs` WAS SPLIT INTO `stand_pat_cutoffs` AND `move_cutoffs`.** One
name covered two different events — a node that never generated a move, and a
node that searched at least one and then cut — so the counter could not answer
the only question it gets asked, which is how often the move loop paid off. A
counter that cannot answer its own question is the same defect as one that reads
zero.

⚑ **`arm_stats()` and `fastq_stats()` each REFUSE the other arm's handle.**
FastQ's counters live in a separate block from the check resolver's, and reading
the wrong one returns a fully-populated struct of zeros — including
`nnue_evals` — which looks like an answer. Both directions raise instead.

The evaluate-once identity is asserted, in the form that is actually true:

    nnue_evals + nodes_created_in_check == nodes_created

⚑ The spec words it as "NNUE evaluations must equal nodes created". That holds
only where every created node has a static value, and an in-check node
deliberately has none — the store publishes it with `value_valid = 0` because the
NNUE evaluation is undefined in check. Asserting the spec's wording verbatim would
fail on any position containing a check; the in-check term is the same claim,
stated so it can be checked. `scripts/fastq_reference_arm.py` exits non-zero if it
breaks, and `_assert_counter_identity` checks it in every search test.
