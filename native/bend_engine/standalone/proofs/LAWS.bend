# Accepted source contracts. Do not weaken them to repair a failed proof.
import Base
import ../Subsets.bend as Subsets

# Mathematical recurrence of the step called by Tables.fill, not another step
# implementation. Nat counts do not wrap at 2^32 or 2^64. Connecting this sequence
# to affine table writes/offsets is a separate, still-open refinement obligation.
def at(n: Nat, +mask: U64) -> U64:
  match n:
    case 0n: U64.zero()
    case 1n+p: Subsets.next(at(p, mask), mask)

law step_in_mask:
  for +subset: U64
  for +mask: U64
  {U64.and(Subsets.next(subset, mask), mask) == Subsets.next(subset, mask) : U64}

law sequence_in_mask:
  for n: Nat
  for +mask: U64
  {U64.and(at(n, mask), mask) == at(n, mask) : U64}

# Reuse the pinned U64 theorem, now applied to actual enumerator states.
# This is injectivity of their extracted representation, NOT enumeration order.
law sequence_extract_deposit:
  for n: Nat
  for +mask: U64
  {U64.pdep(U64.pext(at(n, mask), mask), mask) == at(n, mask) : U64}

law empty_mask_sequence:
  for n: Nat
  {at(n, U64.zero()) == U64.zero() : U64}

# Closed witnesses exercise actual subtraction and the 32-/64-bit boundaries.
# They do not establish the quantified compact-index/order theorem.
law cross_low_half:
  {Subsets.next(U64.one(), U64.from_parts(2147483649, 1))
    == U64.from_parts(1, 0) : U64}

law cross_bit63:
  {Subsets.next(U64.from_parts(1, 1), U64.from_parts(2147483649, 1))
    == U64.from_parts(2147483648, 0) : U64}

law mask_cycle_end:
  {Subsets.next(U64.from_parts(2147483649, 1), U64.from_parts(2147483649, 1))
    == U64.zero() : U64}

law full_word_wrap:
  {Subsets.next(U64.from_parts(4294967295, 4294967295), U64.from_parts(4294967295, 4294967295))
    == U64.zero() : U64}
