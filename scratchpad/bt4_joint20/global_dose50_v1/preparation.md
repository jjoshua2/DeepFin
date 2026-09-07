# G50T05 data preparation

Prospective CPU-only preparation, September 6, 2026. This record binds target
construction and its descriptive audit. It does not authorize a training launch,
select a winner, change the current SF-close grid or consume the fresh confirmation
opening bank. A training/match preregistration must specify the actual comparator,
budget and stopping rule before GPU launch.

## Purpose and target

The completed 20% global BT4 screen favors sharpened BT4 over raw BT4 at all
three registered search budgets against common S0. Whether 20% is a good mixture
weight remains unresolved. Prepare one coarse higher dose for that question:

`G50T05 = 0.5 * normalize(stored SF policy) + 0.5 * normalize(raw BT4 policy ** 2)`

Normalize over legal actions. Use mixer scope `global`, alpha 0.5, BT4 temperature
0.5. Store through the same float16 policy path as G20T05. Copy all non-policy
arrays and source row identities unchanged. This is a global distribution, not
redistribution only among exact ties or SF-close moves. G20T05 is the natural
same-temperature dose control; the final training design remains open until the
currently registered SF-close screen is read.

Source: `data/nnue_derived/armB/qtemp_0.0005_hist_20m`, 18,910,484 rows in 2,309
shards; derive summary SHA256
`391837e49773465edced77bfd13f4084edc60feeff0484078280873d942e50ef`.
Raw BT4 sidecar: `data/lc0/bt4_policy_sidecars/armB_qtemp0005_hist20m`;
summary SHA256 `68b32a41e89c03737aa28c89310d9ac744f6b1e5afcbfba198d2a0155bd646b3`.
Use frozen mixer `/tmp/deepfin-bt4-toolchain/scripts/bt4_policy_mix.py`, SHA256
`5fe8202f1e9681b042caeea06614cdbe3b55cf41ec2100182ce3e58e95e377c7`.
Output: `data/nnue_derived/armB/qtemp_0.0005_hist_20m_bt4_global_G50T05`.
An existing output or `.writing` directory is a stop for this fresh preparation,
not permission to overwrite, delete or silently resume it.

## Audit and resource bounds

Reuse the existing frozen 4,000-position audit, d9 labels and BT4 cache. The audit
uses FEN-only teacher inputs; the full training sidecars retain actual history.
Deep-SF regret is descriptive because agreement with SF does not decide the
global policy-tail/search hypothesis. Use `--sf-audit-mode descriptive` and bind
this unchanged record in both audit and mix. The mixer's legal support, finite
probability, storage-mass and source/sidecar lineage checks remain mandatory.
Record all audit outcomes; do not tune alpha from this audit.

No GPU inference or training. Use nice 19, at most two numeric/compression threads,
one corpus materializer, and a four-hour CPU-stage wall-clock cap. Require at least
300 GiB free before starting the full copy. The existing 150 GiB reserve remains;
the extra admission margin accommodates the dense policy output and generation.
Preserve both generators, the disk monitor, and the current trainer/coordinator.
On timeout or failure, preserve the partial directory and error output. Never
reinterpret a failed preparation as a negative playing result.

Publication must have the expected row/shard counts and exact treatment/source/
teacher/record identities. Before training, verify the ordered game schedule and
non-policy copy integrity using the existing methods. Data publication alone is
not evidence of a completed epoch or playing strength.
