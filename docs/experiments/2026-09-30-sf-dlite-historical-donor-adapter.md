# D-lite historical donor adaptation: reviewed CPU source packet

Status: prepared and CPU-reviewed source packet. No D-lite training, target
effect, or playing-strength result is claimed. The 2.5M-row paired pack,
direct deep-Stockfish target audit, exact launch profile and sole-GPU warmup
must pass before either training arm starts.

The short screen must train both control and candidate from the same completed
Selected-E seed-121 donor checkpoint. Its historical `game_epoch` runner
preserves the required one-row-per-game sampling and 88-step windows but had
no exact donor-load gate. The modern fixed-epoch runner can load a donor, yet
uses a different row-order route. The patch here adds donor admission to the
**pinned historical runtime only**; it is archived on main as an immutable
research patch so the historical branch's unrelated commits do not become a
development PR.

Decompress [adapter-v3.patch.gz](evidence/sf-dlite-historical-donor-adapter-20260930/adapter-v3.patch.gz)
and apply the resulting patch
to an isolated checkout at historical commit
`502cd02e072471c901255f3fdb580d6ea7b826d0`. The patch SHA-256 is
`5920b63d525c4a0b2b530bb3f7cfa795c3c908b1688e146d7972b3b23a35e303`.
The stored gzip SHA-256 is
`671bd11b5e581bd74fdbcbb4603fe2188e79895eca5cd4c7fdefbc1cc4fb3568`.
It changes `scripts/lc0_control_train.py` and adds
`scripts/paired_donor_adapter.py` plus focused tests. The resulting source
SHA-256 values are respectively
`c6935a9169a2734bcf3acf8a8c6ff7cef3579e70bb7d9d7f2037e61fb8cc7f15`,
`70b489f2e9c44eaf6b94cece98df2ba4d4ebee58ebee4655eda5c8a22b51531c`,
and `924ed19cbf97a1f4ca96fd0e9c5cb32fd3fbff407953e4ac590cced9eaef17f5`.
The original historical runtime remains unchanged.

The opt-in adapter rejects a missing or changed donor, incomplete optimizer
slots, model/optimizer/scheduler/step/ZClip remapping or reset, and any paired
pack without the independently qualified source/roster/target-scheme manifest.
It requires the selected 384 native `.zarr` directories and their exact file
and roster-sidecar hashes before training. Both arms retain the donor's
optimizer moments, LR schedule and global step. The adapted initial model
digest is recorded **after** donor load; the fresh-build digest has a separate
name. In donor mode Python, NumPy and Torch start from the same seed 121.
For every consumed batch the adapter banks roster row order, nonmain arrays,
main WDL targets, mirror RNG and realized loss/mask route; it requires the
2,500,000-row, 4,883-batch one-epoch schedule and fails on dropped rows or
active outcome/SF-auxiliary losses. The main WDL target is the only declared
control-versus-candidate difference.

CPU validation of the frozen v3 patch passed 10 focused tests, including
deliberate donor-state mismatch and paired schedule/target refusals. The
reviewer independently reran those 10 tests. Changed-file Ruff and Vulture
passed; scoped basedpyright returned 0 errors/0 warnings. A whole historical
checkout type check has six pre-existing errors in
`tests/test_bootstrap_recovery.py` at lines 50, 53, 68, 69, 81 and 98, with no
new adapter errors. These are baseline findings in the old runtime, not a
green whole-checkout type gate.

A bounded CPU-only admission read confirmed the actual donor checkpoint SHA
`154e768ce1a76dda98b0077244227df1591ef3d4a4e2e438e16c49b1a58ac203`,
historical config SHA
`413dbea9dcde2774eafc2fde706e639fef9e944e301717b938b39b4729633de2`,
and donor step 113459 in 4.8 seconds at 950,360 KiB peak RSS with CUDA hidden.
It did **not** run the full model/optimizer load, compile CUDA, or certify
actual paired-pack consumption. The donor LR is at a cosine-cycle floor, but
the inherited next 88-step `sqrt_release` window resets to the active base;
no LR override is registered for this pair.

Each arm is a complete short adaptation with an external 3,300-second hard
wall cap covering startup, compile, training and save. An interrupted arm
restarts from the same donor in a new output directory; partial checkpoints
carry no scientific credit. If a complete arm cannot fit inside that bound,
the experiment holds until exact model/optimizer/scheduler/RNG/data-cursor
resume is implemented and tested. This short-arm restart contract must not
be mistaken for the exact fixed-epoch resume developed separately for a
multi-day 500M-row campaign.
