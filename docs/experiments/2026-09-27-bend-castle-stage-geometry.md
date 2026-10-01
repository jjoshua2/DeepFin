# Initialized geometric checks through castling stages

## Result and exact source

PR #899 stacks on #898 at `e8430e5cf9c33bcc2856f3a61bf19b44ec46c388`. The new one-law composition and focused runner were checked at `017fadcd25ccda6973b88e240ab50aaae38cf1bd`, tree `ae0491d6a0da6a704d633e56337689af9116a25d`. Archive successor `9347be02ae28f16698318be7ba571d54f29e4377` changes documentation only. No production function or previously accepted law changed.

The public initialized_castling_stage_check_matches_geometry law connects actual in_check at each starting/transit/completed Board to the independent target-centred coordinate/ray witness and the complete initialized array. This closes the per-stage composition of the previously separate king-location and initialized-query results. It does not assert that the returned Boolean is false.

Input conditions are explicit: initial partition consistency, exactly one moving-side king at its home square, raw turn equal to a Boolean side, and the actual producer guard. Initialization has depth17,128 slider blocks and64 extras with arbitrary seed. Inputs.singleton derives the stage singleton; Check.bounded derives its square bound. The caller supplies no output singleton, stored mask, correct query answer or empty rook target.

Both children are computed from the original Board: ordinary source-to-transit and flag2 source-to-destination, not transit followed by castling. All queries continue to use the original moving side even though the child turn flips. Each stage starts from the same initialized table expression. A sequential three-check table-threading theorem and full legal_moves safe-acceptance theorem remain separate.

## Fresh hosted qualification

Run36295872282/job108554494045 passed every stage on its first attempt. The complete importing consumer returned exactly All terms check. in659.488 seconds. Its206-file source closure includes the actual initialized attack and castling-stage producer bodies; this is not merely a statement with assumed query certificates.

All six controls passed: five expected/observed refinement rejections at the new Inputs.singleton certificate (omitted consistency, omitted valid side, omitted input singleton, omitted producer guard, wrong selected color), plus one synthetic warning-output unit. The five fast rejection checks intentionally target certificate wiring rather than claiming to mutate every outer initialized proof. The synthetic unit is not a compiler execution. Crashes, missing files, syntax/ownership failures and timeouts do not count as semantic rejection.

The unchanged castle_kings native verifier reran in generic,forced-portable,native-target and UBSan modes:1,148 distinct requests,3,444 complete stage Boards and82,656 observed fields per mode; eight malformed batches per mode and three compiled/executed actual-code corruptions were rejected.451 inputs satisfy transit premises and437 satisfy final-stage premises. Its complete report matches the parent report except the allowed recorded compiler field. These are repeated fixture modes, not disjoint datasets or new independent proof functions executing natively.

Original compiler16-law/seven-control checks and12 pin tests passed. Unchanged locked CPU repository Ruff/Basedpyright/Vulture lint passed at its configured scope. The new focused Python script was executed but this is not represented as an added all-native-Python type-check. Compiler pin aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae remains unchanged.

All601 inherited source hashes and all607 candidate source hashes match. Active evidence is modular191 laws/606 controls:190/600 retained from #898 and newly executed1/6. The full aggregate wrapper was not run. Complete source array equality is not native allocation,ownership,lifetime or full-buffer correctness.

## Saved local variant preserved without replacement

All14 primary files from saved local commit7df89c34d03149b6ff00ebd5456ef96be2d901ee are committed byte-for-byte under saved-singleton-7df89c34/source/ with a .txt suffix. Original readout,publication metadata and selected source/native receipts accompany them. Archive-only run36296376219 verified all20 payload entries,13 focused suite dependencies and38 inherited dependencies. It did not rerun the saved proof or native suite.

The saved five-law/18-control castle_singleton suite and active four-law/19-control castle_kings suite have different domains. The saved version follows raw U32 side behavior in several statements; the active version takes Boolean/valid-turn premises. Neither is silently replaced or counted twice, and no formal equivalence is claimed. The saved191/599 historical total is not the active191/606 total. The saved local missing-tools lint failure remains a failure.

An initial archive-only workflow had an indentation error before any archive job ran; the corrected archive job succeeded without changing payload or proof bytes. Local parent-tree reconstruction initially omitted178 already-tracked ignored files; restoring the exact archive path list recovered the full4,050-file parent tree. Neither metadata recovery issue changed proof sources.

## Remaining scope

This is independent target-centred geometric check correctness for the certified stage Boards, not a claim that arbitrary guarded castling is safe. Universal attacker-origin/target-centred reversal, actual accepted-path/table-threading composition, opposing-side invariants,historical rights and whole-generator soundness/completeness remain separate.

Self-review only. No independent reviewer is claimed. Pinned checker/Base,native lowering/storage,ABI,toolchain,libraries,OS and hardware remain trust boundaries. Existing compiler,snapshot-lowering and literal closed-builder limitations remain unchanged. No permanent workflow,production code,Python application responsibility,model/GPU,training,search,benchmark,perft increase,merge,force push or deployment.
