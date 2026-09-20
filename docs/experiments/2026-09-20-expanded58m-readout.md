# Completed 58M continuation: paired readout against its 50M predecessor

Status: queued on 2026-09-20 after independent review; no result yet. The match follows the BT4 batch benchmark and precedes final factorial preparation. Existing labeling and CPU preparation continue unchanged.

The question is whether the completed 58M continuation improves playing strength over the exact 50M checkpoint it resumed. Both training exposure and data coverage changed; this does not isolate a data-size effect.

One fixed 256-game match (128 opening pairs), 400 simulations, seed 2026092001, matched search settings and prior temperature 1.0. The unchanged validated runner checks checkpoint lineage, completed training receipts, runtime/source/book pins, paired game-bank completeness and resource bounds. Report the Elo estimate and nominal 95% opening-paired interval, including an interval crossing zero. There is no automatic extension or promotion decision.

Expected active duration is about 20 minutes; hard outer cap is 2,490 seconds. Models and match output stay outside GitHub. Output: `/home/josh/chess-artifacts/runs/expanded58m_vs50m_20260920`.

The [original preregistration](artifacts/2026-09-20-expanded58m-readout/PREREGISTRATION.md), [plan](artifacts/2026-09-20-expanded58m-readout/plan.json), [binding qualification](artifacts/2026-09-20-expanded58m-readout/qualification.json), [registered descriptor](artifacts/2026-09-20-expanded58m-readout/registered_command.json), and [atomic adoption receipt](artifacts/2026-09-20-expanded58m-readout/queued.json) preserve exact settings and identities. The preregistration retains its original pre-adoption status; the later receipt records queue adoption.

Validation: parent reviewer independently inspected runner, descriptor, checkpoint binding and matched settings and approved this bounded run. All plan pins and both checkpoint bindings were checked again immediately before insertion; actual operator registration and launch-budget validation passed. Queue insertion held `gpu.lock`, required preparation still queued, and preserved every other queue item.
