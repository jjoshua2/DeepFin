# Soft-SF cached-bank readiness

The4K audit bank retains112,172 raw d9 lines (3,072 explicit mate lines); it lacks the source-qualified training-row/history join and actual stored C targets required for this training-only comparison. No temperature was selected from that audit bank.

Reuse of the independently reviewed128-row,43-game,one-shard training sample gives the following float16-stored then renormalized target statistics. No calculation was rerun and no new training temperature is selected. Existing10cp is merely the closest exploratory mean C entropy among10/20/40/80; exact ties prefer cooler.

| Target | Entropy nats | Top1 mass | Raw-cp maxima mass | Positive support | Outside C selected mass |
|---|---:|---:|---:|---:|---:|
| C20T05 | 0.660261 | 0.771762 | 0.727854 | 8.656250 | 0.083078 |
| cp10_stored | 0.543010 | 0.784891 | 0.840906 | 9.882812 | 0.041652 |
| cp20_stored | 0.877249 | 0.681654 | 0.735407 | 15.507812 | 0.138634 |
| cp40_stored | 1.386118 | 0.540343 | 0.588654 | 19.484375 | 0.285998 |
| cp80_stored | 1.928890 | 0.390679 | 0.433283 | 21.617188 | 0.439884 |

The source readout and independent-review hashes are pinned in readiness.json, along with the4K original audit/rank/cache identities. Scores preserve existing effective-cp and mate-distance semantics; missing scores are not filled from stored probabilities. The128-row sample is selected seen-training data, not representative. Source-qualified larger training sampling and its C-target/raw-observation join remain the next evidence gap; there is no GPU-arm choice or strength claim.

Reproduce this metadata/JSON-bank inspection only in a fresh output directory (the script refuses to overwrite readiness.json):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
nice -n 19 ionice -c 3 taskset -c 4,5 \
/usr/bin/timeout --signal=TERM --kill-after=30s 270s python3 inspect_bank.py
```

Actual invocation exited0; recorded body wall3.149s, with no corpus decoding, inference, materialization or training.
