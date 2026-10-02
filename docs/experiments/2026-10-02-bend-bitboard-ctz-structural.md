# Actual U64 ctz structural selector

This proof increment follows the actual U64 zero-predicate bridge in PR #991. The question was whether the actual ctz implementation can be connected to the first set bit structurally and whether the selected bit is present for every nonzero U64.

The proof first defines an LSB-first Word traversal. For every width and word, including zero, WordCtzCore.popcount_lowbit_index proves that the popcount of the modeled lowbit minus one equals the traversal index. Checked bridges then connect the actual U64.lsb, subtraction, and popcount to that lemma. WordCtzCore.actual_ctz_index proves the actual U64.ctz equality for every U64 without a nonzero premise.

Two consequences use exactly the premise U64.is_zero(a) == False: WordCtzCore.actual_ctz_range proves ctz is below 64, and TestBitBridge.actual_ctz_selects_bit proves actual U64.test_bit is true at ctz. The latter uses an explicit actual-shift bridge checked for all indices 0 through 63, plus the actual comparator/zero and AND bridges.

The unchanged pinned Bend 2.0.21 checker accepted all four proof entries. Two disposable-copy controls replacing actual ctz with zero and actual test_bit with false were rejected at their intended obligations. Exact source and dependency identities, commands, output digests, and control results are in the qualification receipt. The internal review receipt is in the adjacent evidence folder.

This scope establishes the ctz/index/range/selected-bit bridge only. It does not claim a whole Chess.legal_moves or castling result. Scratch prototypes and the invalid first control harness are not part of this published evidence.
