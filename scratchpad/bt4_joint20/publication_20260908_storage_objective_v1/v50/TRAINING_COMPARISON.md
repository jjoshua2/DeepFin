# B100V50 versus B100: prospective value-only comparison

Selected before any V50 training or matches. Compare 50% separately normalized
stored SF WDL plus 50% native BT4 WDL against the completed B100 SF-value control.
Keep actual B100 policy target bytes, every other target, rows and training
schedule fixed. Shared-trunk learning may change learned policy even though its
supervision stays fixed. This tests a substantial value dose, not an optimal dose
or a separately identified teacher-calibration mechanism.

## Training and eligibility

Require the completed 2309-shard, 18,910,484-row bank and reviewed V50 corpus
publication before training. The new alpha-aware coordinator is pinned at
5afc1e2f0923704b2bb8f1aa7ffe45cc15141757; the actual original trainer/environment
must match the B100 control. A coordinator update is not permission to change the
training objective or schedule. Use schema3 training_only profile B100V50.

Fresh seed0 initialization, batch512, game-aware sampling without replacement,
16 planner/16 loader workers, 88-update windows, 36935 updates and420 finite windows.
Require canonical source schedule
dc687fc333295dee565d19bb4f20da5aa95479dba3aacc5499c22a4004acc64f,
complete realized coverage and no skipped/retried batches. Preserve historical
control limitations, including allow-invalid-control; do not convert this into
held-out or fresh-seed confirmation. If actual trainer compatibility fails, stop
and select a genuinely matched control rather than label a changed-runtime result
as value-only.

B100 reference checkpoint SHA256:
b30ab345d0cf3acfb51bea6c90a91aef3c1dd5edb78da3c92d3a504fb2735d62.
Reference completed-training receipt SHA256:
fa2b63bfa7e4d4039a8fe1f788378416eee4568ec301c6b83e1a357724979799.
Candidate checkpoint identity is pending actual training, not invented here.

## Playing decision

Use the established original-runtime two-budget recipe comparison, direction
V50 minus B100, common training-shape search and prior temperature1.0. Bind the
actual qualified book and canonical development panel in the executable manifest.

- 100 simulations: ordered SPRT H0=0/H1=+15 Elo, alpha.05/beta.10,
  first128 pairs then64-pair looks, cap500 pairs, lookahead64 pairs, pool256.
- Protected400 simulations: fixed first128 paired openings, pool128; run after
  any valid low outcome. Operationally invalid low stops and preserves evidence.
- Both: evaluation batch4096, seed42, move temperature.1, maximum300 plies,
  matching full16-ply opening history and qualified CUDA arena.

Use the existing reader for canonical stopping, fixed high paired interval and
aligned first128-pair high-minus-low score contrast (10000 PCG64 bootstrap draws,
seed20260903). Low stopped Elo intervals are descriptive, not sequential CIs.
H1 warrants further consideration; H0 retains B100 unless the protected high
probe supplies a reason to investigate the distinct depth behavior. An uncrossed
cap is inconclusive, not a reason for automatic extension. A small or uncertain
high difference alone does not justify repeatedly extending the same match.

## Resources and recovery

Training GPU-stage ceiling16200s; each arena ceiling5400s, total new training
plus matches at most27000s. Arena internal deadline5340s, TERM5370s plus30s
grace. Report actual costs rather than calling this a7.5-hour end-to-end budget.
CPU qualification up to600s, prospective and realized schedule verification
up to1800s each, arena CPU preparation up to300s; all separately accounted.
Corpus rewrite already has its separate six-hour producer allocation.

Require current RAM/CPU/GPU and SSD headroom checks, the shared exclusive GPU
lease, STOP markers and exclusive output identities. Keep numerical pools at2;
original trainer16/16 workers require a host resource check. Maintain150GiB
SSD reserve and account for transfers/other pending outputs. Preserve failed
partials and receipts; no blind retry, silent budget extension or promotion.

This is prospective scientific/resource registration. Actual corpus qualification,
prospective schedule, exact command/runtime/input hashes and final launch review
remain pending. No executable manifest or queue is created by this document.
