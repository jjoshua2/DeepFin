# Native batch service-time screen

## Preregistration — September 25, 2026

Parent #884: `f53c908a0af23267f61cee303944d11b878620de`.
Continue PR5's measured dispatch work, not live training or a new search policy.

Question: how does native synchronous CPU callback latency change with bound
physical batch and actual occupancy? Can a provenance-bound offline comparison
identify distinct equal-work choices without assuming that larger is faster?

Control: same saved untrained 5,043,005-parameter 175-plane CPU-F32 checkpoint
SHA256 `bb9934134c0d9a39c515c10fa870122c3dfd28259bece392be026e32a9ca60d7`.
Reuse its singleton package SHA256
`9654a1e334fb5f875d164b82068846bfeab3df2ef2a994737e44789c2623ee6e`.
Export one batch-four package from those exact weights and encoding. No GPU or
private/trained checkpoint, architecture change, training, or match arena.

Plan: three native processes per package, rotating package order; four warmup
sweeps and 24 measured sweeps per process. Each sweep measures all real occupancies
1..B once in rotating order, using fixed root-tensor prefixes. For B1+B4 this is
420 forwards and 924 real rows including warmup, with 360 measured forwards and
792 measured real rows. These are repeated synthetic-workload measurements, not
independent games or 792 search simulations. Every output, including warmups, is
checked against eager singleton reference outside its timing interval.

Deciding gates: model/encoding/CPU/binary/input/reference identities reconcile;
all native and parser failure controls pass; every row stays within inherited
CPU-F32 tolerances; all prescribed samples are preserved with separate warmup
labels and physical/real/padded counts. Any failure stops a successful report.
There is no speedup threshold or favorable-run selection; keep all timings and
process variation. A timing screen cannot establish Elo or end-to-end EPS.

Deliverables: a separate native benchmark using the unmodified model bridge;
strict profile parsing/aggregation; offline equal-work fixed-package plans from
measured occupancy curves; reproducible checks and compact results. The comparison
uses summed sample p95 as a descriptive cost estimate, not a probabilistic
deadline guarantee. It does not switch packages in the running engine or modify
the scheduler. No assumption that a CPU result transfers to the 5090.

Budget: one bounded hosted CPU job, two Torch threads and one compiler job.
No Bend generation is needed; only the small C++ benchmark and existing bridge
are compiled. First run the cheap parser/native/static controls. Reuse existing
artifacts, avoid repeating successful model work for estimator-only corrections,
and retain any failed evidence. No original oracle or tolerance may be relaxed.

Independent review is not available in this tool environment; self-review and
limits must be explicit. No merge, deployment or live-setting change is authorized.

## Readout

Not yet measured. Completed native/model results belong here after the gates run.
