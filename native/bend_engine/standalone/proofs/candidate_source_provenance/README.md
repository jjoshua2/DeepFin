# Bounded owned-source provenance for actual pre-castling candidates

Source.candidates proves castle_chain.Spec.every(Source.source(board),moves) for
the actual moves extracted from castle_chain.Spec.candidates(table,board), for
an arbitrary affine input table and arbitrary Board. Source.source is the pair:

    Nat.is_lt(U32.to_nat(castle_safety.Spec.source(move)),64n) == True
    U64.test_bit(Chess.color(board,Chess.get_turn(board)),U32.to_nat(source(move))) == True

This closes source ownership and strict source-index provenance left explicit in
PR1028. It does not claim legal-board invariants, move completeness/uniqueness,
lookup geometry, destination coverage, or full chess legality.

Source.put preserves the caller-tail predicate in both actual put_move branches:
promotion emits four same-source plies, while the ordinary branch emits one.
The tail can contain other bounded owned sources and arbitrary duplicates; no
all-output-sources-equal-to-src statement is made. Source.destinations follows the
actual fuel and empty flags, ctz/clear_lsb recursion and promotion/EP fields.

Source.after consumes the actual scan_after (table,targets) pair. Source.step calls
the actual piece_targets result with the actual pawn test and EP metadata.
Source.scan is a CPS theorem threading the caller accumulator predicate through
actual Chess.scan, assuming both AllEmittedSound.all_below64(keys) and
all_bits_set(keys,Chess.color(board,Chess.get_turn(board))). Duplicate input keys
and duplicate caller moves are permitted. The strict key-bound and ownership
premises are necessary for the general scan theorem.

Source.initial exactly matches actual C.candidates: own=color(board,get_turn),
bit_squares(64n,is_zero(own),own,Nil), and scan from (table,Nil).
Existing checked All.actual_bit_squares_below64 and All.actual_bit_squares_all_set
close the two key certificates. The source module does not replace the generator.

Source.member obtains both certificates for a full actual Ply via the existing
castle_chain.Lists.extract and castle_emission.Spec.member. The importing
use_exclusion consumer derives the PR1028 source-bit/rays disjointness under
actual filter_requires(Spec.sensitive(board,rays),move)==False, without caller
ownership or index certificates. use_ray composes the actual ordinary
promotion=0/flag=0 post-move Tables.ray subset. Actual input trace, fuel,
stopped=False/mask=False, arbitrary shared accumulator and explicit OLD
Path.attack subset rays certificate are retained.

Concrete total-function witnesses check actual candidate2->10 and its derived
ownership/bound/exclusion using a zero-leaf input table; an arbitrary-key scan
on an unowned pawn emits the same source2->10 and its source is not owned.
These witnesses make no input-table geometry or legal-board claim.
A mixed-source caller tail with two identical source3 moves survives source2
ordinary and promotion emission. The ordinary output retains tail count2.
An out-of-range source64 caller-tail move fails the required source bound;
the opposite color plane does not supply the actual ownership certificate.

## Qualification

Exact base PR1028 117fcf3909e4c086cb9242a29852766c54f2d321,
tree8de12e3ee3f9cec56ab3c2ddeb2f51b4f038a7ab.

From the repository root, select fresh paths outside every checkout:

    python3 -m native.bend_engine.standalone.proofs.candidate_source_provenance.qualify_candidate_source_provenance /path/to/pinned-checker --checker-manifest /path/to/checker-tree.json --report /outside/checkout/qualification-001.json --evidence-dir /outside/checkout/checks-001

Require Bend2.0.21+U64 aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae,
Bun1.4.2, all84 checker files and source fingerprint
d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4.
All reused source/check/mutation/classifier/closure helpers match the exact base.
Every qualified source on a clean published head must match its HEAD Git blob.

Each positive and semantic negative allowance is86400 seconds, CPUs1,3,
maximum2CPU,6GiB AS/RSS and16MiB per-output caps. P2 CPUs30-31 are excluded.
Qualified source, compiler identity and all prior evidence must stay unchanged.

Twelve declaration-local controls reject missing tail/key/membership premises,
a disconnected actual candidate producer, falsely owned arbitrary-key output,
a falsely bounded source64 tail, dropped duplicate tail, wrong color plane,
vacuous actual bypass, omitted old-path coverage and a disconnected actual
ordinary ray. Only expected typed-obligation failures count; parser/import,
linearity, resource/timeout/signal failures do not count.
Internal lightweight independent theorem-boundary review and a clean-head pinned
receipt precede stacked draft publication. No merge, adoption or runtime changes.
