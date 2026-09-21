# Bend policy mapping and complete-input composition

## Plan before execution

Base #802, 5032ba20d5c6e45630300d409a063de590f2ac21. Preserve current
Bend2.0.21 U64 compiler aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae and its manifest.
An existing unpublished policy prototype at bf2d50ada2d0b3d718a987c4d24de7a010ce1ad2
was discovered on feat/bend-owned-policy. Reuse its mapping and verifier rather
than overwrite that branch or replay its older pin/Protocol changes onto #802.

Hypothesis: the Bend policy vocabulary can compose with the complete 146/175-plane
encoder without any runtime Python or C chess helper. Add a typed prepared-input
value pairing an explicitly shaped tensor with exact legal moves and both index
spaces, constructed from ONE validated Game. Keep search/material defaults unchanged.

Acceptance: exhaustive geometry/permutation tables match project Python mappings;
legal moves and all promotions map and resolve exactly with independent CBoard
cross-checks. Prepared requests must match the existing complete-input diagnostic
bit-for-bit and match legal policies from the same hypothetical descendant.
Same-board/different-history inputs must differ while policy IDs agree. Reject
malformed/late-illegal paths before output; preserve root/history and busy behavior.
No array capacity as model width, sentinel as move, geometry as legality, or missing
features as zeros. Explicitly test zero legal moves. No inference/softmax claim.

Budget: isolated CPU, at most two C compilers, one engine thread. One opt-in
15-minute hosted confirmation, retries only to correct findings. Generic, portable,
native, UBSan and static/empty-chroot runtime. Preserve old perft/UCI/rule/history/
complete-input verifiers and 12 pin tests; no recurring workflow or deeper perft.
Python and C oracles remain external. No GPU, training, checkpoint, production,
merge or deployment. Recovery discards the isolated new branch, preserving earlier
work. Self-review only; no independent reviewer is available in this session.
