# Bootstrap research toward restarting RL

## Intended outcome

The user reaffirmed on September 6 that this research serves a strong bootstrap
for restarting the RL loop in roughly one to two months. The deliverable is a
selected data/target/loss recipe, its trained checkpoint, reproducible provenance
and search settings, and evidence that it is ready for an RL restart. The current
20M-position experiments are a development scale; the planned larger bootstrap
uses approximately 100M positions.

Choose further work by how much it can change that selection or make the restart
ready. Keep useful data generation running throughout. The time horizon is a
working research window, not a claim that every method will be exhaustively tested.

## Decisions that remain connected

| Decision | Relevant comparisons |
| --- | --- |
| Data quality and cost | SF search depth/allocation, coverage and distribution; keep generator/label provenance |
| Teacher and mixing recipe | SF, BT4, exact ties, SF-close moves, global mixing, mixture weight and BT4 sharpening |
| Training objective | Existing CE versus a controlled objective change, crossed with selected teacher/mixing recipes |
| Search scaling | Matched shallow/deeper search and realized cost; teacher agreement is descriptive |
| Larger bootstrap and RL restart | Confirm a selected recipe, scale its data/training, then check the actual restart path |

These decisions interact. A mixture selected under ordinary CE may not be best
under another loss. A sharper policy can help one search budget and hurt another.
BT4 disagreement with SF is not itself a playing loss. A reward map can discard
score differences before the loss sees them. Record these dependencies rather than
promoting a recipe from one favorable diagnostic.

## Current order of work

1. Finish the registered global BT4 screen and the queued sharpened SF-close
   comparison against exact ties. Preserve their frozen training/search protocols.
2. Use those playing results and cheap banked-data analyses to select a small set
   of informative comparisons across teacher/mixing recipe and training objective.
   Pure BT4, larger mixing weights and a graded objective remain candidate avenues;
   banking their targets does not queue a full training grid.
3. Compare competitive trained recipes directly and confirm the selected candidate
   with a fresh training seed and opening bank. Change this design if evidence
   shows another comparison would better resolve the restart decision.
4. Apply the supported recipe at the larger bootstrap scale. Verify the resulting
   checkpoint at relevant search budgets and exercise a bounded RL restart with
   recoverable state before committing to sustained online training.

Preregister each compute-backed comparison at its actual decision point. Prefer
substantive alternatives over repeatedly refining a clearly positive-versus-SF
score. Conversely, an unfinished search-scaling question still needs its matched
search evidence. Stop expanding a method when its likely decision value is below
the cost of delaying confirmation, larger-scale training or the restart.

## Evidence navigation

- [Global BT4 screen](2026-09-05-bt4-global-search-scaling.md): registered source
  recipes and search-budget comparison; later prior-temperature correction and
  completed runtime artifacts live under `scratchpad/bt4_joint20/global_run03`.
- [Graded TailRL bootstrap screen](2026-09-06-tailrl-bootstrap.md): objective
  qualification, real-data mechanism evidence and teacher/mixing interactions.
- Live continuation state: `scratchpad/bt4_joint20/global_run03/temperature_transition_status.json`.
  Inspect actual processes and completion artifacts before resuming a stage.

The next scientific selection remains open. No bootstrap recipe or restart
checkpoint has been declared optimal by this roadmap.
