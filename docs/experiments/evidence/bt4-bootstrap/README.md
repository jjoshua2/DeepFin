# BT4 bootstrap publication evidence

These are historical scientific artifacts, copied byte-for-byte from the recorded
experiment state. They accompany the [experiment records](../../README.md).
The [manifest](manifest.json) maps every repository-relative source to its published
path, byte count and SHA256. Preserving originals keeps the hashes in preregistrations
and reviews meaningful. Source `scratchpad/bt4_joint20/` paths are retained; training
and data summaries are under `scratchpad/bt4_joint20/publication_originals/`, with
original paths recorded in the manifest.

Historical absolute paths, commands, PIDs, initial launch observations and committed
configuration identities are provenance. They are neither portable execution
instructions nor current runtime status. Do not execute archived drivers or adopt
old manifests to resume work. The live runtime and development checkout are separate.

## Coverage

September 8 preparation/readiness supplements have their own manifests:

- [Completed corrected G10 block](g10-next-completion-manifest.json): 1,053,018
  common rows, original stage logs and independent accounting.
- [Next worker01 inputs](g10-worker01-readiness-manifest.json): metadata-only
  readiness for 2,126,537 raw rows; no pipeline launch.
- [Training-horizon runtime](training-horizon-runtime-manifest.json): tiny real
  two-epoch CPU tests and preserved diagnostic failures/limits.
- [Soft-SF bank readiness](soft-sf-bank-readiness-manifest.json): audit-bank
  provenance check and reuse of the existing small training sample.

The bundle contains all nine completed global treatment cells, all three SF-close
cells, both direct development comparisons, the fresh paired confirmation, and both
prior-temperature calibration banks: **17 complete 1,000-game banks**. The earlier
sampler diagnostic is also retained, separately labeled by its original filename;
it is not one of those 17 primary/calibration banks. The original JSONL headers and
all game rows are preserved, along with results, completion/protocol receipts,
preregistrations, readouts and independent reviews. Reserved history-bearing opening
FENs, their generation and exclusion evidence are included.

Completed model training summaries and matched schedules permit inspection of
training completion, skipped/retried windows, source identities and ordering without
shipping weights or training corpora. Target derivation/mix summaries, bounded formula
checks and the originally cited six-arm/global/G50 audits document recipe construction.
Superseded registrations and the sampler diagnostic retain their historical limits;
their presence is not a claim that discarded plans or proposed G50 training ran.

## Raw banks

| Bank | Original game rows | File |
| --- | --- | --- |
| `af_run01/E0_sampler_diagnostic.games.jsonl` | 384 | [JSONL](../../../../scratchpad/bt4_joint20/af_run01/E0_sampler_diagnostic.games.jsonl) |
| `confirmation_seed1_E0T05_vs_C20T05_v1/arena.games.jsonl` | 1000 | [JSONL](../../../../scratchpad/bt4_joint20/confirmation_seed1_E0T05_vs_C20T05_v1/arena.games.jsonl) |
| `direct_close_global_run01/arena/arena.games.jsonl` | 1000 | [JSONL](../../../../scratchpad/bt4_joint20/direct_close_global_run01/arena/arena.games.jsonl) |
| `global_run03/calibration/E0.s100.arena.games.jsonl` | 1000 | [JSONL](../../../../scratchpad/bt4_joint20/global_run03/calibration/E0.s100.arena.games.jsonl) |
| `global_run03/calibration/S0.s100.arena.games.jsonl` | 1000 | [JSONL](../../../../scratchpad/bt4_joint20/global_run03/calibration/S0.s100.arena.games.jsonl) |
| `global_run03/treatments/E0.s100.arena.games.jsonl` | 1000 | [JSONL](../../../../scratchpad/bt4_joint20/global_run03/treatments/E0.s100.arena.games.jsonl) |
| `global_run03/treatments/E0.s25.arena.games.jsonl` | 1000 | [JSONL](../../../../scratchpad/bt4_joint20/global_run03/treatments/E0.s25.arena.games.jsonl) |
| `global_run03/treatments/E0.s400.arena.games.jsonl` | 1000 | [JSONL](../../../../scratchpad/bt4_joint20/global_run03/treatments/E0.s400.arena.games.jsonl) |
| `global_run03/treatments/G20T05.s100.arena.games.jsonl` | 1000 | [JSONL](../../../../scratchpad/bt4_joint20/global_run03/treatments/G20T05.s100.arena.games.jsonl) |
| `global_run03/treatments/G20T05.s25.arena.games.jsonl` | 1000 | [JSONL](../../../../scratchpad/bt4_joint20/global_run03/treatments/G20T05.s25.arena.games.jsonl) |
| `global_run03/treatments/G20T05.s400.arena.games.jsonl` | 1000 | [JSONL](../../../../scratchpad/bt4_joint20/global_run03/treatments/G20T05.s400.arena.games.jsonl) |
| `global_run03/treatments/G20T1.s100.arena.games.jsonl` | 1000 | [JSONL](../../../../scratchpad/bt4_joint20/global_run03/treatments/G20T1.s100.arena.games.jsonl) |
| `global_run03/treatments/G20T1.s25.arena.games.jsonl` | 1000 | [JSONL](../../../../scratchpad/bt4_joint20/global_run03/treatments/G20T1.s25.arena.games.jsonl) |
| `global_run03/treatments/G20T1.s400.arena.games.jsonl` | 1000 | [JSONL](../../../../scratchpad/bt4_joint20/global_run03/treatments/G20T1.s400.arena.games.jsonl) |
| `sf_close_run02/sf-close/C20T05.s100.arena.games.jsonl` | 1000 | [JSONL](../../../../scratchpad/bt4_joint20/sf_close_run02/sf-close/C20T05.s100.arena.games.jsonl) |
| `sf_close_run02/sf-close/C20T05.s25.arena.games.jsonl` | 1000 | [JSONL](../../../../scratchpad/bt4_joint20/sf_close_run02/sf-close/C20T05.s25.arena.games.jsonl) |
| `sf_close_run02/sf-close/C20T05.s400.arena.games.jsonl` | 1000 | [JSONL](../../../../scratchpad/bt4_joint20/sf_close_run02/sf-close/C20T05.s400.arena.games.jsonl) |
| `sharpened_ties_run01/experiment/arena/arena.games.jsonl` | 1000 | [JSONL](../../../../scratchpad/bt4_joint20/sharpened_ties_run01/experiment/arena/arena.games.jsonl) |

## Checks and boundaries

Selected files were inspected before copying: JSON/JSONL structure was parsed,
credential/private-key/token patterns and populated secret fields were checked, and
source/configuration fields were inspected for unrelated host/account material.
No such content was found. Copies were then hash-checked against stable source
metadata. This is publication integrity, not a new independent scientific verdict.
Individual accepted reviews state the provenance they did and did not recheck.

Corpus shards, model/optimizer checkpoints, ONNX/native binaries, bulk logs and
operational monitors remain external. Historical scripts outside the repository are
identified by their original paths/revisions/hashes; normal project code remains in
Git history. The bundle does not make those runtime prerequisites self-contained.

The manifest lists each 64-character artifact reference found in the dated BT4
records at collection. A populated `published_path` identifies the exact matching
bytes. Unmapped hashes remain visible: many identify external weights/runtime code,
a canonical schedule digest rather than a file, or source payloads intentionally
excluded here. Later document edits can add references after that collection snapshot.
Fresh confirmation's [final independent completed-result review](../../../../scratchpad/bt4_joint20/confirmation_registration_v1/independent_completed_confirmation_review.json)
is included at SHA256 `cab8e9ccbf5d379d6dc03c9fa5c0502d2b865b50a649d5bf6c3f8093661505e0`.
It distinguishes the accepted complete-bank result from the earlier training/launch reviews.

## Recompute without the original machine

From the repository root, the following uses only Python's standard library. It
verifies every cataloged file and recomputes the chosen complete bank's candidate
score and nominal paired normal interval. Change `bank` to another complete bank
from the table. Read its header for candidate/reference roles: direct opponents
differ across screens. This single-bank interval does not reproduce the global
screen's registered paired-bootstrap contrasts or validate training provenance.

```python
import hashlib, json, math, statistics
from pathlib import Path

catalog = json.loads(Path("docs/experiments/evidence/bt4-bootstrap/manifest.json").read_text())
for item in catalog["files"]:
    data = Path(item["published_path"]).read_bytes()
    assert len(data) == item["bytes"]
    assert hashlib.sha256(data).hexdigest() == item["sha256"]

bank = Path("scratchpad/bt4_joint20/confirmation_seed1_E0T05_vs_C20T05_v1/arena.games.jsonl")
rows = [json.loads(line) for line in bank.read_text().splitlines()]
header = [row for row in rows if row["kind"] == "header"]
assert len(header) == 1
print(header[0]["settings"]["candidate"], "minus", header[0]["settings"]["reference"])
pairs = {}
for row in rows:
    if row["kind"] != "game":
        continue
    white = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}[row["result"]]
    score = white if row["a_is_white"] else 1.0 - white
    assert score == row["score_candidate"]
    pair = pairs.setdefault(row["pair_id"], {})
    assert row["half"] not in pair
    pair[row["half"]] = (score, row["a_is_white"], row["opening_fen"])
assert len(pairs) == 500
for pair in pairs.values():
    assert set(pair) == {0, 1}
    assert pair[0][1] != pair[1][1] and pair[0][2] == pair[1][2]
scores = [statistics.mean(value[0] for value in pair.values()) for pair in pairs.values()]
score = statistics.mean(scores)
se = math.sqrt(statistics.variance(scores) / len(scores))
ci = [max(0.0, score - 1.96 * se), min(1.0, score + 1.96 * se)]
def elo(p):
    return -math.inf if p == 0 else math.inf if p == 1 else 400 * math.log10(p / (1 - p))
print({"candidate_score": score, "score_ci95": ci, "elo": elo(score), "elo_ci95": list(map(elo, ci))})
```

The four follow-up catalogs above point to frozen originals under
`scratchpad/bt4_joint20/publication_20260908_followup/`. Catalog `path` values
name published repository-relative artifacts; `source_path` values preserve the
original producing-checkout-relative provenance. The originals retain historical
absolute paths and are evidence, not portable executable instructions.

PR559 initially placed these 84 artifacts below `docs/`, outside the existing
historical-archive convention. Its author/reviewer receipts are preserved byte
for byte; their validation did not close the tracked-file path-hygiene check. The
follow-up relocation corrects that publication gap without redaction, changes to
scientific results, or changes to the existing hygiene scanner and exemptions.

[Qualified Soft-SF training sample](soft-sf-qualified-sample-manifest.json) includes
the fixed sampling plan, original completion, row-level metrics and lossless raw d9
observations; array-bank parts remain external with published identities.

[Worker01 failure and larger-data proposal](g10-worker01-failure-and-large-readiness-manifest.json)
preserves the failed rank admission and incomplete common-input status separately
from the unregistered metadata-only proposal.

## Completed B100 two-budget screen

The [completed result manifest](b100-two-budget-results-manifest.json) preserves
both match banks, successful readouts, stage/launch identities and the original
reader failure. The [dated readout](../../2026-09-08-bt4-pure-policy-endpoint.md)
reports the shallow sequential decision, protected deep result, aligned search
contrast and actual charged time, with same-seed and stopped-estimate limits.
