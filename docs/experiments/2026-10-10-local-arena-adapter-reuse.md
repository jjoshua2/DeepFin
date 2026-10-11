# Local arena adapter reuse: isolated source packet

Status: repository-ready source/evidence draft; no live adoption, arena result or
measured production speedup. The original armed evaluation and training retain
their source pins and operational ownership.

The fresh epoch-one arena adapter copied 21 of 23 top-level definitions from the
existing short-E adapter. Their full diff changed two PGN names and the pair-bound
shim path. The candidate keeps the short-E implementation and replaces the fresh
copy with a 28-line pinned caller. Four shared functions change to accept these
existing differences and bind authority to the executing wrapper. The production
diff removes 484 net lines; there is no new scheduler, CLI option or class.

The [source packet](evidence/arena-adapter-reuse-20261010/) includes both candidate
files, 11 self-contained stdlib tests, byte-pinned baseline fixtures, the original
local patch, and its verification/review receipt. Candidate production bytes are
unchanged from independently reviewed local revision
`870d9aa84d58e48a90e44fafac24bf263fb65167`. The shared source SHA-256 is
`5e7133d525db7854ded59af42a309f58316082e8410e6d1d92755f1e967fb497`.
Test portability changes replace the local Git-history dependency with exact
original source fixtures; their raw hashes are checked before use.

Repository guidance requires compact experiment evidence on main and allows code
and small fixtures in checkouts. Existing precedent includes
`evidence/factorial58-20260919/operating_scripts/` and the archived
[D-lite historical adapter](2026-09-30-sf-dlite-historical-donor-adapter.md).
This placement records frozen operational source without promoting its absolute
runtime paths, 2,606-pair science protocol or historical serving dependencies into
a general package API. Bulk models, panels, logs and run outputs remain external.

The source packet preserves search/count/time limits, source/manifest and inference
relations, owner PID/start-ticks/boot identity, inherited GPU/IO descriptors,
create-only attempt/log admission, and code 3 only as an incomplete arena attempt.
Existing receipt publication and external lifecycle logic are unchanged.

Original local validation passed 11 tests in 0.421 seconds. An independent reviewer
ran the initial 11 tests and confirmed the final cleanup revision with no remaining
findings. These are mocked adapter-contract tests, not an end-to-end arena result.
Repository-packet test and review results are recorded separately in
`packet-verification.json` when complete.

Future adoption must follow [ADOPTION.md](evidence/arena-adapter-reuse-20261010/ADOPTION.md).
It creates a new reviewed future-launch packet; it never changes or repins the
currently armed controller, its dependency closure or an existing manifest.
