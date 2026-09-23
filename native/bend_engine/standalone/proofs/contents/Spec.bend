# Pure positions in the actual increment/subset schedule; no candidate table values.
import Base
import ../../Subsets.bend as Subsets

def address(n: Nat, +at: U32) -> U32:
  match n:
    case 0n: at
    case 1n+p: address(p,U32.inc(at))

def state(n: Nat,+mask: U64,+subset: U64) -> U64:
  match n:
    case 0n: subset
    case 1n+p: state(p,mask,Subsets.next(subset,mask))
