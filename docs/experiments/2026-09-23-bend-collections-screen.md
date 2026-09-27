# Owning FIFO and search early-exit screen

## Preregistration

Base: `4e8d463e1df6871a2b73b066c95506258d028800` on main. Compiler remains
`jjoshua2/bend@aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae` (2.0.21 + U64).

Hypothesis: consumed-state two-list queues avoid repeated list-append traversal,
and completed search traversals need not consume remaining fuel. These are
separate mechanisms, not a combined promised engine speedup.

Baseline/control: existing public Search step semantics and the independent
root/session CBoard oracle; list-append rotation and Python deque for FIFO work.
Deciding gates: exact FIFO values and length, exact traversal IDs/visit/value
bits, unchanged existing session/root-advance behavior, and rejection of a
compiled LIFO mutation. Any semantic mismatch kills adoption. Static failures
must be fixed or documented without weakening checks. Compiler remains pinned.

Budget: one isolated hosted CPU qualification, two Torch threads, one compiler
job, no model export, training, GPU, production settings or live process access.
Native checks cover generic/native/UBSan FIFO and generic/UBSan traversal.
The timing screen has six alternating pairs (first warmup) at each of three
sizes; retain every checksum, length and millisecond observation. Sub-20-ms
samples are not grounds for a precise ratio. No useful-EPS or Elo inference.

Recovery: source-only PR targeting main; no merge/deployment. Queue is not wired
into scheduling. Reverting the small Search traversal change restores the old
fuel-consuming loops. Source and compiler identities accompany the readout.

## Scope and reuse

The upstream library was used as a design comparison, not copied. This adds an
independent, ownership-polymorphic FIFO supporting array-owning elements,
which addresses the upstream deque's Data-only restriction. No crypto, String
hash map, growable arena, upstream dependency or compiler upgrade is adopted.
See [probe documentation](../../native/bend_engine/collections_probe/README.md).

## Readout

The preregistered checks below were executed; raw observations are retained.


### Preserved setup failures

Run 35922817981 stopped at the first static gate: four Ruff findings in the new test file (one list-construction style issue and three broad exception assertions). Whole-repository type checking reported zero errors/warnings and Vulture had no findings. Hosted Python and native tests had not run. The corrected candidate uses list.extend and checks exception messages; no gate or expectation was suppressed.

Run 35923458566 passed focused Ruff/Basedpyright and all 66 Python cases, then stopped before compilation: the new probe mistakenly imported a legacy compiler guard. The existing bitboard/session manifest pins 57bc84ed with fingerprint c178489e, whereas the CI installer and this preregistration pin aaeb9bc9 with fingerprint d9550e30. Neither manifest nor compiler source was changed. The new screen now verifies the intended 84-file fingerprint itself, and session_check.py builds the same existing session/CBoard sources and invokes the unchanged Python-chess/session/root-advance oracles under that verified compiler. It does not monkeypatch the legacy guard or call its build function with a mismatched compiler. These setup failures are not native coverage. The original 17 pure Python tests also passed locally.

Run 35923899467 again passed the 66 Python cases and focused static checks, then the first Bend parser rejected computed-tuple destructuring in the owning trace emitter. The emitter and the analogous benchmark length read now pass the computed tuple to dedicated parameter-destructuring helpers. No compiler, behavior expectation or test fixture changed; no native runtime was executed in that failed attempt.

Run 35924258475 passed the three-mode owning FIFO, two-mode traversal and executable LIFO rejection checks, but the benchmark checker rejected reuse of its unannotated order Boolean. The shared Boolean now has a + annotation. This was not a queue/traversal behavior failure; benchmark timings and full qualification were not completed in that attempt. The final attempt reruns all checks rather than relabeling partial success.

Run 35924444086 passed whole-repository lint, the 66 Python cases, all native FIFO/traversal/mutation checks and all timing checksums. Its new session-build wrapper incorrectly rejected the existing driver's exact 18-definition foreign-I/O dependency notice despite compiler exit zero. The session driver already imports Job.load, Command.read and Reply.read through C; this is a native integration test, not a source-proof gate. The corrected wrapper requires exactly that diagnostic and caller list, preserves it in logs/report, and still rejects every unexpected message/nonzero status. Eight added unit cases check this boundary; the pure collection/traversal compiler checks remain strict. No existing driver, oracle, compiler or foreign implementation changed.


### Completed hosted qualification

Run: https://github.com/jjoshua2/DeepFin/actions/runs/35924956435

Whole-repository lint and explicit new-file static checks passed. 74 focused Python tests passed without skips. FIFO: 6151 exact output rows in each of generic/native/UBSan modes. Traversal: 368 exact output rows in each of generic/UBSan modes. The compiled LIFO mutation was rejected. The unchanged independent root/session verifier passed in its default modes, using the separate exact-current-pin build wrapper. These repeated modes are not disjoint datasets.

Raw timing samples and exact source hashes are in evidence/bend-collections-screen/. Short samples remain flagged. This is not an owning-root scheduler benchmark, GPU/model run, formal proof, independent review, EPS result or Elo result. Self-review only. No third-party source was imported.
