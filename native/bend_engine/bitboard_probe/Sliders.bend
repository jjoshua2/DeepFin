import Base

# The offset and index must address a table built with this SAME indexing
# scheme. Array.get wraps out-of-range indices; the parity harness checks
# the unwrapped index separately. Occupancies/attacks retain all 64 bits.
def pext_index(occupied: U64, mask: U64) -> U32:
  U64.low(U64.pext(occupied, mask))

def magic_index(occupied: U64, mask: U64, magic: U64, shift: Nat) -> U32:
  U64.low(U64.shrn(U64.mul(U64.and(occupied, mask), magic), shift))

def lookup(table: Array<U64>, offset: U32, index: U32) -> Array<U64> & U64:
  Array.get(U64, table, U32.add(offset, index))
