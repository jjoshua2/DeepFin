# Optimization-safe verifier publication and follow-up integration

## Scope and result

Run **36332433712** fully qualified source `a1a90374fe9ef320bb913872137a45453d8ddd32` on parent `012d0b0bc883065745433baa5436078083c4310e`. This updates existing PR #910 without replacing its five-law table-preservation suite, its two-law generator-follow-up composition, or its saved-source archive. The final source tree is `220c52b0f5b2fee5a7e0002e587811be4ffb7a56`. Evidence publication changes documentation only.

The saved local verifier repair at2a926846c843d4b829091f5415376bc8b5a71c78 is now active: table-preservation result validation remains enabled under Python -O and PYTHONOPTIMIZE, complete buffers are required before counting any compared cell, and failed reruns invalidate stale PASS receipts at the requested output path. No active Bend law, consumer, native probe, production function or fixture definition changed.

## Confirmed old verifier defect and integration repair

A parser-only reproduction supplied64 complete before cells, one matching after cell and no done terminator. The original runner rejects that in normal Python but accepts it under -O, incorrectly counting64 compared cells with zero changes. The repaired parser rejects it in all tested modes. This is a verification-harness defect under optional Python optimization, not a counterexample to the Bend theorem or grounds to relabel the original normal-mode hosted qualification a failure.

The latest generator_followup runners load table_preservation runners by file path under arbitrary module names. A naive sibling import in the saved repair failed there with ModuleNotFoundError before CLI startup. Integration tests caught that before changing the PR. Direct/spec-loaded runners now load the exact sibling validation file, rejecting its absence rather than falling back to a same-named module on sys.path. Both follow-up dependency manifests explicitly include that transitive helper. These two manifest changes do not change the follow-up proof or reference logic.

Both table-preservation entry points replace the specifically requested report with NOT_COMPLETED before external execution and replace it atomically with PASS only after all checks. Unrelated historical receipts are untouched. Concurrent runs still require different paths. This guarantee is not claimed for every other repository runner; generator_followup still uses its existing normal-Python validation and report behavior.

## Fresh integrated checks

All59 host test methods passed in normal,-O and PYTHONOPTIMIZE=2 modes. This includes the saved50 methods, four real CLI failure-path tests and five import/consumer compatibility methods. They are repeated host tests, not new formal laws or registered compiler rejection controls.

The entire five-law table-preservation consumer and17 controls passed in both normal and optimized Python. The full two-law generator-follow-up consumer and15 controls also passed in normal Python with the repaired imported runner. Both manifests include the shared helper. Formal source bodies are unchanged; these are fresh executions of existing contracts, not seven new discoveries.

The full preservation native driver passed generic,forced-portable,native-target and UBSan with Python optimization level2:173 requests and2,313,088 U64 cells/4,626,176 limb comparisons per mode, seven malformed batches per mode and all three actual-code corruptions. The follow-up driver passed its four modes in normal Python:368 requests per mode, exact generated-list comparisons, independent following-check answers and all four behavioral corruptions. Complete mode/mutation observations match their original respective reports, with current source manifests recorded separately.

The original compiler16-law/seven-control suite and all12 pin tests passed. Locked CPU tools passed unchanged configured Ruff/Basedpyright/Vulture lint. New runtime validation was not achieved by disabling a checker or suppressing compiler diagnostics. The compiler remains aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae.

All unchanged files from the643-entry current native manifest remain identical; two new Python helper/test files make645. The already-published14-file alternate source archive was compared byte-for-byte with the saved local package and retained once, at evidence/bend-table-preservation-continuation/saved-a67267f9/source/. No alternate implementation is silently overwritten or counted twice.

The earlier standalone qualification run36331753111 passed54 tests in three host modes, the five-law gate twice, four native modes under optimization, compiler gates and lint on the old83045b38 baseline. It is retained as component evidence, not substituted for this integrated run. The old local missing-tool receipts and original archives are unmodified.

## Remaining scope and review

Active modular coverage remains200 laws/655 controls from the two already-published suites. No new formal law or registered proof control is added by this hardening. The full aggregate wrapper and independent review were not run. These checks do not prove whole-generator move legality, universal attack-direction correspondence, historical rights, native pointer identity or lifetime safety. Source value preservation and native observations remain distinct.

Self-review only. No merge,force push,deployment,approval-gate bypass,model/GPU work,training,search,benchmark,perft increase or application responsibility change. Ordinary PR CI is distinct from this dedicated qualification.
