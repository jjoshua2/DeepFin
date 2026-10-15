# Research repository

New experiment prose lives in [DeepFin-research](https://github.com/jjoshua2/DeepFin-research).
The optional research/ submodule pins the exact revision used by this checkout.
[Experiment navigation](experiments/README.md) links moved records and retained contracts.

## Working with records

Initialize only when needed:

    git submodule update --init research
    rg -n -i 'TOPIC_OR_SOURCE_IDENTITY' research/docs/experiments research/scratchpad

Default production clones and CI do not need recursive submodule checkout. Research edits
are committed and reviewed in DeepFin-research; update the main gitlink deliberately
when a consumer needs the new revision. Avoid automatic tracking of its moving main
branch. An uninitialized submodule must not break production or existing qualification.

Keep each hypothesis, preregistration, amendment, negative/inconclusive result and readout
together. Link follow-ups to earlier records and retain source-qualified game/shard/
position/checkpoint identities. Store large datasets, labels, weights, checkpoints and
bulk logs in existing artifact storage; record hashes and retrieval locations here.

Reusable tools, maintained proof source, essential contracts and machine-read qualification
inputs remain with their implementation until a reviewed reader migration proves that a
different location works. Do not add new one-off research payloads to main's scratchpad.
Do not publish private/live operational receipts under this policy.

## What this first migration does

Source revision: **32370b099f6d8d4ec965de135e70a752fc95a7a9**. The research repository imports **251 already-public
Markdown files / 2,153,780 source bytes**, including **41 scratchpad records**. Its first import
commit **8536ec99471478efee257e9b7bbeb4d3e614854b** preserves original bytes and modes, with source path/blob/
SHA-256 identities in its provenance/source-manifest.tsv. A later navigation commit adds
attribution and portable links; original code blocks and historical commands remain evidence.

This draft removes **121 independent dated prose files / 834,985 bytes** from main and
updates their Markdown navigation links. It retains **130 imported source copies** wherever
machine-readable receipts, native/proof docs or scratchpad references make removal unsafe.
It leaves all non-Markdown evidence, all executable verifiers, maintained Bend proof source,
frozen proof receipts and all runtime/queued paths unchanged. Original Git history remains
in DeepFin and is linked from research provenance.

The original two roots contain **133,666,800 bytes**; this first removal is only **0.62%**
of that payload. Most bloat remains: JSON contributes 95,383,943 bytes, JSONL 13,681,622,
and compressed .gz banks 17,516,110. This draft establishes the living destination and
prevents new prose accumulation; it does not complete the payload migration or shrink
existing Git history.

## Remaining bundle map

The research repository's **provenance/remaining-bundles.tsv** lists every retained bundle
from the two roots, with source file counts/bytes, disposition and smallest compatible
next step. Its classifications are source-based; unknown active/queued readers are
explicitly unresolved rather than assumed inactive.

| Bundle type | What remains | Minimum next change |
| --- | --- | --- |
| Current/future runtime or queue inputs | scratchpad operator/runtime mixtures | Owning thread checks job/source/queue/recovery paths; configure explicit replacement input locations before byte-verified inactive-only removal. |
| Executable reader dependency | global_run02 verifier and sf_close_run02 qualification | Promote the reusable verifier to a supported entrypoint; give readers explicit pinned evidence paths and exercise missing/wrong-input behavior. |
| Frozen source qualification | Bend receipt source.json and source_sha256s chains | Preserve bytes/source identities beside implementation; separate scoped qualification before any path migration. Maintained proof source is outside extraction. |
| Closed compact historical evidence | JSON/JSONL/CSV and small receipts without proven consumers | Confirm no active or source-hash binding, copy exact bytes, verify destination tree/readback, and update prose links; unknown bindings remain a hold. |
| Bulk observations or compressed banks | dataset-like .gz/.npz/.raw/.zst payloads | Keep or relocate to existing artifact storage with hashes, provenance and restore locations; do not turn the new research Git repo into bulk storage. |
| Reference copies of frozen prose | Records retained because receipts/native/scratch paths mention them | Keep original source bytes stable; use research copies for new follow-ups and close each historical reference before removal. |

Historical proof attempt reports are eligible for a later per-record review; that does not
authorize moving maintained proof modules or qualifications. Do not rewrite frozen receipt
paths or source-hash manifests as a link cleanup. The known Bend qualifiers consume the
king-away receipt and require 249 original source identities; the current paths remain valid.

## Search, checks and rollback

Search current research in the initialized submodule. Search earlier source history with:

    git log --all -- docs/experiments scratchpad/bt4_joint20 docs/experiment_ledger.md
    git grep -n -i TOPIC 32370b099f6d8d4ec965de135e70a752fc95a7a9 -- docs/experiments scratchpad/bt4_joint20

The import manifest and exact original import commit make source bytes recoverable even
after navigation edits. No prior history was rewritten or force-pushed. File removal from
the current tree does not reclaim historical Git blobs.

For future executable/evidence changes, use existing scope-appropriate tests and the
project review process. Coordinate local CPU/GPU validation with the current owner;
a documentation migration does not authorize a new controller, guard or live workload.
This draft changes prose/navigation and a gitlink, with all executable/evidence source
bytes retained. No test-suite result is claimed for it.

Rollback is a normal revert of the migration commit. The original files remain in DeepFin
history and the byte-preserving research import. Main's merge and live deployment are
separate actions; this draft requests neither a merge nor a training restart.
