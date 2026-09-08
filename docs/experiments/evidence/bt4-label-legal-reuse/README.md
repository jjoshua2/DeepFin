# BT4 legal-move reuse evidence

See the [readout](../../2026-09-07-bt4-label-legal-reuse.md).
`manifest.json` maps every copied artifact to its original path and SHA-256.
Absolute paths inside historical plans and commands describe the measured machine;
they are not portable defaults or instructions to rerun a live job.

The baseline program files used by `ab_script.py` are recoverable from commit
`bb01468b1a0eb0e6a97102e069525911ef04f76a`: save
`scripts/bt4_raw_corpus_sidecar.py` as `baseline_raw.py` and
`scripts/bt4_policy_dump.py` as `baseline_dump.py`, then verify their manifest hashes.
The original copied plans/scripts pin the measured candidate and runtime. Source
rows, full profile data and mapping banks remain external with explicit hashes.

The earlier independent review accurately marked the benchmark and whole lint
pending at its creation. The later `author_validation.json` and `ab_completed.json`
close those items without changing that historical receipt. Import failures are
preserved separately; neither executed a timed numerical comparison.
