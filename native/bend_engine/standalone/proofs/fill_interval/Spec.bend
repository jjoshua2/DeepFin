# One real production slider block, retaining Tables.fill's actual count and step.
import Base
import ../../Tables.bend as Tables
import ../prefix/Spec.bend as Prefix
import ../layout/Spec.bend as Layout

def block(+k: Nat,a: Array<U64>) -> Array<U64>:
  +key = U32.from_nat(k)
  Tables.fill(U32.to_nat(Layout.size(key)),Prefix.prefix(k),U32.mod(key,64),U32.is_ge(key,64),Layout.mask(key),U64.zero(),a)
