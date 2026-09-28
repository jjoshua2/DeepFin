# FIFO integration into the bounded cohort scheduler

## Preregistration

Base: PR #863 `bb1847d2d7f3c4e8b94e5dfb9f64bfc87e0c3390`, not the
main-targeted collections PR. Reuse the exact independently written Queue.bend
from #864 `2f2f137e346658a60682b574c63679ebc88cfd74`, SHA-256
`643a330324cfb637b8bacb74d3fad7ac41d9081452bb74253c6592b65b061c9f`.
No third-party collections source is copied. The ring and search early-exit
change are not pulled into this stack; they remain in #864.

Hypothesis: the ready-root FIFO can replace list-append rotation without
changing scheduling order, ownership, selected-leaf execution or accounting.
The earlier owning microbenchmark motivates this integration, not a claimed
full-runner speedup. This readout decides semantic equivalence only.

Implementation: ready roots and the unvisited tail are single-owner two-list
queues in both sync and async execution. Gather removes roots, keeping their
owning state/ticket/history/row together in existing tasks. Scatter returns
processed roots after the unvisited tail in original order. Roots are never
requeued during gathering, so an available sweep cannot revisit one root.
Mask scans and cancellation transform both physical lists without reversing
or flattening them. Only final reporting converts the queue to a flat list.
Existing admission remains 1..16 roots, one pending leaf per root, one physical
batch slot and 4,096-node arenas. No live-admission API or larger cohort is added.

Controls: the exact parent generated coordinator C from #863, SHA-256
`fcb787241c29f10c0fa33fbd8983f9630d3ab52914c109de2449b52a93251a24`,
retained in artifact 10775514198 from run 35918586010; the unchanged independent
CBoard/Python-chess tree oracle; existing blocked cancellation/deadline tests.
Compiler stays aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae, with the unchanged
84-file fingerprint d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4.

Deciding gates: exact chronological path/reply/batch/rule events, complete final
tree fields, report root order and all non-time work fields match the parent.
Only six named measured time/rate fields are excluded after both reports pass
the existing strict parser. New/unknown work fields stay in comparison. Run
five fixtures (single, mixed-three, mixed-sixteen, neural-budget and terminal)
in both sync/async modes at batches 1/2/4/8/16 and 146/175 planes. Add the two
batch-four UBSan comparisons. Any disagreement blocks qualification; no
numerical tolerance is relaxed. These are deterministic callbacks, not models.

Adapter checks cover sizes 0/1/3/16, removal counts 0/1/2/4/16/17, seven rotations,
and five cancellation masks. They exercise the actual queue adapters and live
mask scan, preserve an existing mask bit, and check root state/order/length.
Run normal and UBSan; a compiled reversed-requeue mutation must exit normally
but fail the unchanged oracle. Existing async and deadline matrices stay intact.

Budget: isolated hosted CPU-only qualification, two Torch threads and one
compiler/build job; at most 25 minutes per full attempt. No model export,
training, GPU, live process, live checkout/config or deployment access.
Recovery: source-only stacked PR, no merge. Reverting its four coordinator
files returns the parent list scheduler. Self-review only, not independent
review or a formal proof. No end-to-end speed, latency, EPS or Elo claim.

## Reproduction

```sh
BUN=bun CC=clang-18 CXX=clang++-18 \
  bash native/bend_engine/multi_root/qualify_fifo.sh \
  VERIFIED_COMPILER PINNED_PARENT.c NEW_OUTPUT
python -m pytest tests/test_bend_fifo_scheduler.py
```

The script rejects a mismatched parent C hash before running. The parent C
is external build evidence, never committed. It is regenerated from the pinned
parent with the same compiler if the retained artifact is unavailable; confirm
the exact hash instead of substituting a candidate build as the control.

## Local checks before hosted execution

The actual changed runner passed Bend --check-only. The pure adapter compiled
and produced 840 exact rows (120 cases) under local Clang 17 UBSan. All 15 new
Python cases passed in a separate test directory. A normal repository pytest
attempt lacked python-chess and failed in its conftest; no repository pytest
pass is claimed locally. Full local coordinator generation was killed with
exit -9 after 118.46 seconds. That is not a successful build; the hosted source
compiler must generate the complete runner before native qualification.

## Hosted readout

Completed qualification is recorded below; pending execution is superseded.


### Preserved transport setup failure

Run 35942974064 stopped before applying source or running any test because two characters were transcribed incorrectly in the compressed transport. The corrected transport was independently compared with the local authored patch and verified against its unchanged SHA-256. No source, oracle or tolerance was changed to recover it.


### Completed qualification

Run [35943091398](https://github.com/jjoshua2/DeepFin/actions/runs/35943091398), job `107455083191`, completed every stage. All 209 focused Python cases passed without skips (15 new); explicit static checks and whole-repository Ruff, Basedpyright and Vulture passed.

The freshly generated FIFO coordinator passed the unchanged CBoard/Python-chess matrix: ten batch/width configurations in synchronous and asynchronous modes, two UBSan configurations, and four held-forward cancellation/stop/quit configurations. Existing deadline controls passed in four configurations; compiled timer/compaction checks passed in normal and UBSan modes. The unchanged native worker passed 1,456 assertions and 160 reuse calls per normal/ASan+UBSan mode.

The additional 120 parent/FIFO pairs matched chronological path/reply/batch/rule events, final report root order, every populated final tree field and every non-time work field. Only the six named time/rate values were excluded after strict parsing. These comprise five fixture workloads in sync/async execution across ten normal configurations and two batch-four UBSan configurations. Adapter checks passed 120 scenarios / 840 exact rows per normal and UBSan build; the compiled reversed-requeue mutation was rejected. Repeated modes use the same fixtures, not independent games.

Representation rationale (self-review, not a formal proof): the logical queue is front followed by reversed rear. Removing from its front and appending retired tasks in their original order preserves the old tail-plus-returned-roots schedule. Cancellation is a pointwise transform; the live-root mask is a commutative OR across both lists. Neither operation needs to flatten the ready queue. Pending tasks remain outside it until retirement.

No real model, GPU, full-runner performance, deadline-latency bound, EPS, Elo, formal proof or independent review is claimed. The existing opt-in cohort runner is the integration target, not production UCI or live training. Initial capacity remains 16 roots. Nothing is merged or deployed.

Raw build/test reports are retained in artifact `10785862321`; ZIP SHA-256 `8e892da29782d004e9209d65732a5fdaf95a439be7733f77e0a49874f6a799c7`. Candidate generated C SHA-256 `7f348a6ad58c7e3abebbf876f4ada8862c9f6dbc5856f445e1e2310346d68b74`. The exact source manifest, compact controls and 120 equivalence observations are committed under `evidence/fifo-cohort-integration/`. The source manifest describes the tested preregistration before this documentation-only readout; executable files are unchanged.
