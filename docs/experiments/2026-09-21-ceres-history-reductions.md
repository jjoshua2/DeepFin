# Ceres CPU history reductions — 2026-09-21

## Preregistered question

Can converting already-validated binary history to uint8 reduce Ceres TPG encoding CPU time while preserving every teacher feed byte and rejection rule? Existing chunk-aligned source reads are already on main. No inference, batch/backend change, active runtime edit or teacher relabeling is involved.

Saved completion for factorial58 extra184_run06_g10 shard000000 reported 32,768 rows: CPU preparation/postprocessing 4.409 s, source reads 0.493 s, output/verification 1.446 s and synchronous session.run 17.018 s (total 23.962 s). CPU encoding profile on the first 1,024 original 20M rows took 0.127 s, with 0.038 s in sums and 0.018 s in argmax. These are observations on different workloads, not an attribution of the entire collector residual.

## Fixed comparison and decision

Freeze baseline source from origin/main `7939609c9`. Read the first 1,024 rows from one original 20M, one history-qualified G10, and one raw-adapted source shard. Preserve input digests. Compare original versus byte-history encoding in the actual 32-row call shape; one untimed warmup and five alternating-order paired measurements per source. Keep source reads and equality hashing outside timing. Retain each timing and exact output digest; compare each complete output byte array and read-only input preservation. Require all outputs equal and median per-source encoding time at least 15% lower on every source. Otherwise do not promote the change.

CPU-only: affinity 12–13, nice 19, OMP/MKL/OpenBLAS 2, hidden GPU, 15-minute aggregate benchmark cap. One small frozen panel per source; no corpus scan or teacher calls. Interpret as warm CPU encoder evidence only; any end-to-end labeling gain remains unmeasured. Validate rejection behavior including near-zero/near-one fractional history, nonfinite input and existing history/castling/en-passant cases against independent Board encoding.

## Completed CPU readout

The precommitted gate passed on all three 1,024-row panels; all 3,072 teacher feeds were byte-identical to the frozen baseline and read-only inputs remained unchanged. Each measurement retained the original 32-row inference-call shape.

| Source family | Baseline median seconds | Candidate median seconds | Encoder time saved |
| --- | ---: | ---: | ---: |
| Original20M stored history | 0.1272 | 0.0983 | 22.7% |
| Qualified G10 history | 0.2044 | 0.1425 | 30.3% |
| Raw-adapted source | 0.1277 | 0.0909 | 28.8% |

[All paired measurements and byte digests](evidence/ceres-history-cpu-20260921.json) retain the five alternating pairs per source. The complete bounded benchmark took 5.49 s. Raw benchmark/profile scripts and receipt are banked in `~/chess-artifacts/operations/ceres-history-cpu-20260921/`. Measurements used Python 3.10's deployed NumPy 1.26.2 on cores 12–13 at nice 19; no new environment or GPU work was performed. Host load varied during the panel; medians describe this small warm CPU screen, not a cross-machine guarantee.

The change narrows history only after the original binary-value check. Valid history contains exactly 0/1, so reductions and argmax preserve values; fractional, negative and overflowing values still fail before narrowing. At batch 32 the additional byte-history allocation is 208 KiB. The caller's float16 stored inputs, output uint8 feed, row order, batch padding, teacher backend and teacher calls remain unchanged. The existing producer hash identifies the new encoder revision rather than reusing old provenance.

Full collector throughput and teacher numerical outputs have not been remeasured. Equal feed bytes establish the encoder contract; any deployment uses a newly qualified runtime, while existing frozen collectors remain unchanged. The measured 22.7–30.3% improvement applies only to this encoder component.

## Validation

118 focused tests passed across the stored TPG encoder, independent Board encoder and Ceres sidecar collector suites. Scoped Ruff, basedpyright and Vulture checks passed. Whole-repository Ruff and Vulture passed, but basedpyright reported 14 type errors in six unchanged baseline test files under the host dependency environment. `git diff origin/main` verifies those six files are unchanged; none of the findings names the edited encoder or test. Tests used existing native modules copied into the isolated worktree; no native rebuild or production deployment accompanies this change.
