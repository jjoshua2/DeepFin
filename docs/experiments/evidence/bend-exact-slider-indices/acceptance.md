# Exact U32 compact-index continuation — acceptance before implementation

Base: PR #819 40803892f3204a54a33a87e2e7b25fa48fed2e1a.
Use the unchanged compiler fingerprint, existing LAWS and all negative controls.
New acceptance: generic low32 value preservation below mathematical 2^32;
actual Sliders.pext_index equals full PEXT value for every valid chess key/occupancy;
connect that exact U32 index to the imported subset ordinal and coverage;
reject an in-range but wrong constant index in a disposable copy.
No production changes, model work, perft, merge or deployment.
Budget: one compiler at a time; focused source/native probe, aggregate proof gate,
original pin/source checks, one whole-repository lint attempt. No ordinary CI additions.
Do not claim table storage/offset or ray refinement from scalar correspondence.
