# Independent corrected-candidate review

Reviewer: `/root/independent_king_away_review`, a separate read-only agent.
Base: `dd187ac09c8e60061bfcd2f8f00d748a7543c9ca`.
Verdict: **no actionable findings in the corrected candidate**.

The public KingAway.exact theorem retains the arbitrary affine array and arbitrary
Board domain with exactly four premises: Boolean side encoded in turn, selected
king singleton U64.bit(sq), sq<64, and square unequal to that side's home. It
concludes CastleSpec.result(Chess.legal_moves(table,b))==(table,Nil{}).

The reviewer traced both actual wing rejections, actual ordinary scan, full/fast
filtering, flag-2 projection, and complete-array preservation. Corrected internal
Representation.reify/Observe.Cells transport follows the existing storage pattern
and introduces no public certificate premise or initialized-array assumption.
No new axioms, unsafe annotations, holes or unfilled candidate laws were found.

Corrected proof files, importing consumer, qualifier, wrapper tests, theorem guide
and proposed CI workflow were reviewed. The completed receipt was independently inspected;
all 249 recorded source hashes were compared with current files, with zero
mismatches. Guard and ordinary helpers passed in 15.593 and 15.826 seconds; the
complete importing consumer passed in 591.629 seconds within its 900-second bound.
Removing the actual home-king guard rejected at side; injecting flag 2 in ordinary
put_move rejected at put. These were intended semantic rejections, not backend
or ownership failures. The unchanged original compiler identity was confirmed.

Qualification receipt SHA-256:
`6e20c442ca16b43a80803c3d2156bf6471da7414a6080d5507cccda6e432f800`.
Host-validation receipt SHA-256:
`c01d51e80a2cb3603a11f7ee47b804067bf5750cac5444579c290b68d68cb07c`.
The reviewer verified the three saved lint-log hashes and inspected focused ruff,
basedpyright and vulture success. Whole-repository lint has 277 base-equivalent
environment errors and is not credited as passing.

Reviewed final source SHA-256 identities:

```text
KingAway.bend             11673120972387b865988f3c1befb41551c36c516df276e71f23b45892d5f968
KingAwayFilter.bend       fe74edfee8f305c3b9f5580c90bea0e5ee121c26abb77e2d01bc932b7cac1492
KingAwayGuard.bend        7e38881ebad4a11f9277375290169cdfcc1e2137ca6e23b18617c962bd9f8587
KingAwayWings.bend        a7f453f5e2346fbc0b35bef367b211c814c5a3731f46b7dec5f49599bbf0de09
king_away_consumer.bend   10b4dae51e77a7803d20b433ac21a72f0ec02e180dc9f16b13453a0d79d927ae
qualify_king_away.py      678c56828223edefeeddaa0573ac67b556bd3ce86240ce8a4c1767b9a74b473f
test_qualify_king_away.py 9eb31d9b316982ef6e34147c401fe5d57108b50c6f2afdd99f599d8f3c2a43b9
bend-king-away.yml       39455a1349394bb50aa60617322d52816d9a03ac73349b15a11c60bca021439d
KING_AWAY.md             6c065b84bb3c920818b678ff88a2d1a8d6ee84a9f8f489b1049fd1a139170bab
```

Limits: this was read-only source and receipt review. The reviewer did not rerun
the checker or tests and made no writes. Native lowering, ABI, independent history
and complete legal-move correctness remain outside this component and review.
The earlier static review explicitly left checker acceptance pending; its initial
array-reuse candidate was rejected by the checker and superseded by this reviewed,
fully accepted affine-safe candidate.


Publication note added by the author: GitHub lacks workflow OAuth scope. The
reviewed bend-king-away.yml is preserved byte-for-byte at proposed-workflow.yml
in this evidence directory; it is not registered as an automatic workflow. The
source proof/gate identities and qualification receipt remain unchanged.
