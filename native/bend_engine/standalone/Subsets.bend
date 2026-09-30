# Production carry-rippler step called by Tables.fill.
# Kept as two pure functions so membership proofs observe the actual arithmetic
# without hard-coding its body; closed boundary laws also constrain subtraction.
import Base

def difference(subset: U64, mask: U64) -> U64:
  U64.sub(subset, mask)

def next(+subset: U64, +mask: U64) -> U64:
  U64.and(difference(subset, mask), mask)
