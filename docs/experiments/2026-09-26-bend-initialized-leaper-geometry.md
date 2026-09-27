# Initialized non-slider attack geometry

## Scope before complete qualification

Base: PR #895 at e78987ccad0dfeafd026548b392e8fb52e960472.
Compiler: unchanged aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae.

Three proposed source contracts connect independently specified natural file/rank
knight, king, white-pawn and black-pawn masks to actual computation, their stored
values after the entire extras loop, and actual Chess.attack after allocation and
an arbitrary preceding table loop. Exact final array values are part of the last
two results. The query theorem assumes only depth17, all64 extras and bounded
square; no expected mask or initial content certificate is supplied.

The source model uses explicit coordinate steps, not production U32 wrapping deltas.
Finite scalar-coordinate cases are checked inside Bend; structural accumulation and
write-preservation induction avoid enumerating arbitrary arrays or occupancies.
This does not yet compose all five attack-mask classes with the attack-witness
reduction, prove forward/reverse slider semantics, or establish singleton king
conditions through castling. Existing initialized-slider proofs remain unchanged.

Acceptance: exact safe source consumer; classified semantic/policy controls;
independent actual initialization/query tests in generic, portable, native-target
and UBSan; original compiler/pin checks; unchanged repository lint. No aggregate
whole-stack run, production change, perft increase, model/GPU/training or benchmark.
All statements below are added only for completed checks. Self-review only.


## Completed hosted qualification

Hosted run **36284142030** passes all three public laws, importing consumer, nineteen controls, four native initialization/query modes, three compiled/executed actual-code corruptions, original compiler checks and unchanged repository lint on source `6fd072fe09f3e507940f3723719588e54b06c062`.

The three related contracts establish independently specified non-slider masks at actual computation, complete extras storage and initialized Chess.attack boundaries. The last two results include the complete final array. Actual allocation and any preceding Tables.tables loop supply shape; initial slot contents and occupancy are arbitrary. Depth17 is supported, not claimed minimal. No correctness of preceding slider contents follows from the non-slider result.

Geometry uses bounded natural file/rank steps. Sixty-four scalar coordinate tuples cover all four classes, with structural accumulation rather than host-generated expected bitboards. Address separation and one-block frames compose by induction through all later writes. The natural schedule is explicitly equated with actual Tables.extras. Each selected square is bound into the same complete loop, not a selected-key-dependent runtime.

The consumer also specializes the actual reversed pawn query for both Boolean attacker colors. This is query-direction wiring, not a new universal forward/reverse attack-membership theorem. Existing initialized-slider and attack-witness results are unchanged; composing the complete initialized attacked result and singleton conditions through castling remains separate.

Each native mode passes3,072 distinct requests/6,144 U32 mask fields:256 masks for each of four initialization contexts and three occupancies. Contexts are actual full Tables.build, all-ones initialization plus full extras, a partial preceding table loop plus full extras, and a planted last-slot sentinel plus full extras. Nine malformed batches are rejected per mode. Modes repeat fixtures; queries do not inspect the entire array or prove native lifetime. Candidate code never imports the proof model or receives expected answers.

Actual wrong king-slot, shifted black-pawn storage and reversed white-pawn storage corruptions compile and execute before the independent signed-coordinate reference rejects their values. Ten source/refinement controls and eight policy controls pass; one synthetic warning-output unit is not a compiler execution. Crashes, missing imports, syntax/ownership errors and timeouts are not semantic rejection.

All553 parent source entries are retained unchanged; all571 candidate hashes match. Parent180/543 plus newly executed3/19 yields modular183 laws/562 controls, not a full aggregate run. Original compiler16-laws/seven-controls and12 pin tests pass. The hosted native report matches the local result except C compiler identity. The source dependency manifest matches exactly.

Direct complete-mask normalization exceeded a240-second local bound. Two dependent draft checks were then deliberately stopped without a result. A structural accumulation/scalar-coordinate replacement passed, without changing the geometry domain or any accepted prior law. Initial reserved-name/Data/match-order and probe duplication errors were corrected before the completed runs. Original logs/drafts remain in the review package. Local lint failed for missing Ruff/Basedpyright/Vulture; hosted locked CPU tools pass the unchanged lint command, without relabeling the local failure.

The symbolic depth and full-extras certificates avoid expanding a closed131,072-cell term. No new formal equality for the enormous literal Tables.build expression is claimed; its real native execution is a test context. Source and compiler inputs are unchanged after checking; later publication changes only documentation/evidence.

Self-review only. Pinned checker/Base, native lowering/storage, ABI, C/C++ toolchain, libraries, OS and hardware remain boundaries. Existing compiler strict-TypeScript, structural-snapshot lowering and literal closed-builder limitations remain unchanged. No production function, earlier proof, permanent workflow, routine perft, search, model/GPU, training or benchmark change. No Python application responsibility moved into Bend. No merge, force push, deployment or live-process action.
