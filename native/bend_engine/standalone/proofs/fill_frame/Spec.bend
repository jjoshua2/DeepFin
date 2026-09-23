import Base
import ../prefix/Spec.bend as Prefix
import ../layout/Spec.bend as Layout
import ../../Tables.bend as Tables

# Mathematical count plus observed start; not modular arithmetic and not a read equality.
def room(n: Nat,at: U32,end: U32) -> Bool:
  Nat.is_le(Nat.add(n,U32.to_nat(at)),U32.to_nat(end))

def outside(+q: U32,at: U32,end: U32) -> Bool:
  Bool.or(U32.is_lt(q,at),U32.is_le(end,q))

# The actual fill invocation made by one slider block, excluding its header writes.
def block(a: Array<U64>,+k: Nat) -> Array<U64>:
  +key = U32.from_nat(k)
  Tables.fill(U32.to_nat(Layout.size(key)),Prefix.prefix(k),U32.mod(key,64),
    U32.is_ge(key,64),Layout.mask(key),U64.zero(),a)
