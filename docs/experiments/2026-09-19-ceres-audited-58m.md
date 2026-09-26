# Ceres labels for the audited 58M bootstrap union

The authorized 2x2 screen compares BT4 versus equally mixed BT4/Ceres policy
(each teacher T=0.5), and SF/BT4 V50 versus equal SF/BT4/Ceres values.
All four use the same 58,090,688 rows, initialization, order and training budget.

The existing Ceres banks cover 35,314,577 rows (4,321 shards). The remaining
22,776,111 rows (2,787 shards) belong to 14 audited, finalized derived subsets
of still-growing raw corpora. These subsets must retain their explicit audit
admission; they must never be relabeled as complete raw corpora.

This change carries audited-source identity into each Ceres bank binding,
checks saved evidence during collection, and revalidates admission before the
completion receipt. It also brings in two already-used dependencies: audited
source inventory support from a71c40c91, and configurable inference batches
from b032c98e1. Existing fixed32 and common-G10 paths remain supported.

Use batch512, two CPU threads, a 16GiB ORT arena and four shards per invocation,
retaining raw legal policy logits and both raw WDL logit heads. Past complete
pipeline throughput (~1,006 rows/s) suggests ~6.3 GPU hours for the missing
rows; this is an estimate, not a measured new run. Collection follows the
current active trainer, without changing its sources or runtime.

Coverage mapping and prepared collection artifacts live at
`scratchpad/bt4_joint20/factorial58_20260919/`. Collection completion is not
training admission: the target producer must check exact row identities and
calibrations and qualify all four complete corpora before training.
