# Actual pawn scan occurrence count

**Status: CHECKED.** The pinned Bend 2.0.21 + U64 qualification in
evidence/2026-10-03/qualification.json passes the positive consumer and all
seven semantic negative controls. The earlier target-pair Data/Type failure is
retained separately as failed-target-pair-receipt.json and is not the final
qualification result.

This proof composes the actual Chess.scan_step pawn branch with the checked
scan_after full-Ply occurrence count. The target mask is expanded to the
source's exact terms: one-square push with its range/occupancy guard, eligible
two-square push with the initial-rank and both-square-clear guards, and pawn
attack squares intersected with enemy non-King occupancy or Chess.ep_target.
The actual piece_targets result is proved equal to this formula; target_exact
supplies the exact formula-to-(table, targets) pair used by the count.

The en-passant set is the implementation's validated target: in range, on the
side-to-move's expected rank, empty, and backed by the opposing pawn bit at
target xor 8. The initialized table is exactly Tables.extras(64,0,
Tables.tables(n,key,at,Array.new(U64,17,seed))), expressed through
Attack.Initialized.run.

The query remains a full Ply: the shared count theorem accounts for the four
back-rank promotions with flag zero, ordinary destinations with the
implementation's pawn/EP flag, and exact multiplicity from any supplied
duplicate-containing tail. This proves pseudo-attack emission/counting, not
legal-move or king-safety correctness.

An intermediate isolated pair-type scratch probe was used to diagnose early failures and is not part of the published proof.
