# Actual board-scan full-Ply occurrence count

This proof composes the arbitrary-target Chess.scan_after destination factorization with the actual recursive Chess.scan traversal.

scan_count_cells handles every source-list shape, including repeated U32 entries and the empty list. Its exact count is the left-to-right fold of the per-source scan_step contribution. The initial tail is arbitrary and may contain duplicate Chess.Ply values. The count therefore preserves both newly emitted multiplicity and each occurrence already in the tail; it never sorts, deduplicates, or changes the searched value from the complete four-field Ply.

The recursive proof invokes actual Chess.scan_step on each source and the existing per-source scan_after count lemma. TableTargets.step and TableTargets.scan establish the actual unchanged table projection. The fold records each source's actual piece_targets result against the preserved table. This states only the checked table refinement for these read-only lookups; it assumes no cache identity beyond the proved table projection.

legal_moves_raw_scan_count instantiates the exact source list passed by Chess.legal_moves. actual_legal_moves_scan_input connects that scan to the caller, before its castle/filter preparation. The count theorem does not count the later filtered result and makes no board-validity, target-geometry, legal-move, or king-safety claim.

## Qualification

Run qualify_board_scan_count.py with the pinned checker, the verified checker manifest, a private receipt path, and a private evidence directory. The qualifier checks the positive consumer with an 86400-second wall limit, then runs four 120-second semantic controls: duplicated source, omitted source, wrong initial tail, and disconnected production consumer. Both positive and negative checker processes are capped to two CPUs, 6 GiB address space, and 64 KiB combined stdout/stderr files. It verifies the Bend and Bun pins, the 84-file checker fingerprint, and source hashes before and after qualification.

Raw logs and the complete receipt stay local because they contain checker diagnostics and local execution paths. The committed qualification summary contains only sanitized commands and digests.
