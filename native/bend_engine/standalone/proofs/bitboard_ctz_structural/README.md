# Actual U64 ctz structural proof

This checked proof extends the actual U64 zero-predicate bridge in PR #991. It builds the first-set-bit model over Base Word, connects the model to the unchanged U64 implementation, and proves ctz range and selected-bit properties. It does not prove the full chess legal-move generator.

## Theorems

- WordCtzCore.popcount_lowbit_index is universal over widths and words, including zero: the popcount of lowbit_model(width, word) minus one equals first_set(width, word).
- WordCtzCore.actual_ctz_index is universal over U64 inputs and equates actual U64.ctz with the structural first-set index.
- WordCtzCore.actual_ctz_range requires only U64.is_zero(a) == False and establishes the result is below 64.
- TestBitBridge.actual_ctz_selects_bit has the same sole nonzero premise and establishes actual U64.test_bit at actual ctz is true.

The proofs reuse actual U64.lsb, subtraction, AND, popcount, comparator, zero, shift, and test_bit definitions through the checked Word bridges in PR #991 and the existing inventory/successor proof modules. U64BitProbe.bit_word_exact expands exactly indices 0 through 63 against the actual shift and proves the out-of-range case by Domain.impossible.

## Reproduction

Run each file below with the pinned Bend checker and --check-only. The checked commands, source/dependency hashes, output digests, and valid negative controls are recorded in the evidence qualification. Each check used CPU affinity to CPUs 0 and 1 with a 180-second timeout. No axioms, holes, admits, unsafe annotations, or compiler/source changes were added.

## Review and provenance

An internal independent review verified theorem assumptions, bridges, the 64 shift cases, transitive imports, source hashes, and two corrected mutation controls; it also independently reran all four checks. The reviewer could verify the pinned checker's base.bend and main.ts hashes, but the pin directory contains no Git metadata, so the recorded compiler commit identifier could not be authenticated from that directory alone. See the review receipt in the evidence folder.
