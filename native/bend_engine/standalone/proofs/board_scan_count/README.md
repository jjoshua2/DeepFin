# Actual destinations and board scan occurrence count

This proof adds the generic full-Ply count bridge from arbitrary target bitboards through the actual `Chess.destinations` and `Chess.scan_after` implementation, then composes it with the actual recursive `Chess.scan`.

## Destination and scan_after factorization

`AfterCount.scan_after_count` receives the actual `(table, targets)` pair used by `Chess.scan_after`. It proves that the resulting move list has exactly the `Spec.tally` occurrence count over `Chess.bit_squares(64, is_zero(targets), targets, Nil)`, plus the count already present in the arbitrary tail. `AfterCount.actual_key_frequency` and `actual_keys_below64` instantiate PR1000's all-U32 frequency and range theorems for that exact target enumeration. This establishes each counted destination is an in-range square whose bit is set.

The emitted key is matched as the complete four-field `Chess.Ply`: source, destination, promotion, and flag. The producer's actual rule is preserved: a pawn destination on either back rank emits promotions 1 through 4, each with flag 0; otherwise it emits promotion 0 and flag `Bool(pawn && destination == ep)`. `Emission` connects the count formula to `Chess.destinations` using the existing promotion-choice consumer. No target-geometry premise is introduced.

## Actual source traversal

`scan_count_cells` handles every source-list shape, including repeated U32 entries and the empty list. Its exact count is the left-to-right fold of each actual `scan_step` / `piece_targets` contribution. The arbitrary initial tail may contain duplicate full-`Ply` values; the proof preserves every occurrence and never sorts or deduplicates.

`TableTargets.step` and `TableTargets.scan` establish the actual unchanged table projection. The fold uses each source's actual `piece_targets` result against the preserved table; it assumes no cache identity beyond that checked table projection.

`legal_moves_raw_scan_count` instantiates the exact source list passed by `Chess.legal_moves`. `actual_legal_moves_scan_input` connects that scan to the caller before castle/filter preparation. It does not count the later filtered result and makes no board-validity, legal-move, or king-safety claim.

## Qualification

Run `qualify_board_scan_count.py` with the pinned checker, verified checker manifest, PR1000 base-source manifest, private receipt path, and private evidence directory. The qualifier checks the positive consumer with an 86400-second wall limit and six negative controls: wrong promotion fields, wrong en-passant flag, duplicate source, omitted source, wrong initial tail, and disconnected production consumer. Each checker process is capped to two CPUs, 6 GiB address space, and 64 KiB combined output.

The source overlay was checked against PR1000 head `1690865e2fc59091d0b8947364fb2a5e64ea40b9`: every one of the 76 pre-existing dependency files matches by remote Git blob ID and the source snapshot file hash. `AfterCount.bend` and the board-scan proof are the new files added on that base.

Raw diagnostics stay local. The committed qualification summary records commands, pins, result hashes, and source hashes only.
