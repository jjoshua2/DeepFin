# Cap40 stored-input prediction drift, 2026-11-12

This independently reviewed diagnostic compares donor and cap40 WDL predictions on the exact same 254 stored rows from one metadata-selected, corrected-history, pure-selfplay producer shard. It is **target-free descriptive drift**, not evidence of improved playing strength, correctness, calibration or loss.

- Source revision: `53f8908522e0659b0661f484586dc1e35c55e309`.
- Three supplied producer-game groups contain 70, 87 and 97 rows. The shard was training-ingested; exact sampled-row exposure is unknown. There is no heldout or independence claim.
- Five sequential CPU passes: donor batch 16, identical donor repeat, donor batch 1, cap40 batch 16 and cap40 batch 1. Repeat and batch controls passed at `atol=1e-6`, `rtol=1e-5`.
- Candidate-minus-donor expected-score prediction shift: +0.8194 percentage points row-weighted; +0.7235 points equal-group weighted. Mean **absolute** expected-score change was 6.0688 points; 133 rows moved up and 121 moved down. A small signed mean masks substantial bidirectional changes.
- The largest individual WDL probability change was 54.9444 points. This is a prediction change, not a measured accuracy change.
- Each child recorded 16 GiB address-space, 60 CPU-second and 90 wall-second limits, one CPU thread and CUDA disabled. These are execution settings, not measured peak-resource telemetry.
- W,D,L order and side-to-move perspective are source-declared. Real semantic POV validation remains unperformed.

`independent-review.json` is the unchanged reviewed compact receipt (SHA256 `9da1295da264a3a9488bde613f357d62420fa8a607db002398bca58cb13b8e9d`). It binds checkpoint, input, saved-output and plan identities and independently recomputed metrics. Raw input rows and probability datasets are not published here. Local paths in the receipt identify retained technical evidence, not publicly accessible artifacts. No training, GPU workload or model update occurred.
