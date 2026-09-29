# Source address expression: actual compact PEXT index plus certified prefix.
# This is not yet a theorem that buffer-stored metadata equals that prefix.
import Base
import ../prefix/Spec.bend as Prefix
import ../layout/Spec.bend as Layout
import ../../../bitboard_probe/Sliders.bend as Sliders

def address(k: Nat,r: U32) -> U32:
  U32.add(Prefix.prefix(k),r)

def index(+k: Nat,occ: U64) -> U32:
  Sliders.pext_index(occ,Layout.mask(U32.from_nat(k)))

def interval(+k: Nat,+r: U32) -> Type:
  {U32.is_le(Prefix.prefix(k),address(k,r)) == True{} : Bool}
  & {U32.is_lt(address(k,r),Prefix.prefix(1n+k)) == True{} : Bool}
  & {U32.is_lt(address(k,r),131072) == True{} : Bool}
