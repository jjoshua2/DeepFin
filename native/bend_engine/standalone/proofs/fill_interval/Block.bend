import Base
import ./Naturals.bend as N
import ./Sum.bend as Sum
import ../relative/Add.bend as Add
import ../relative/Bounds.bend as Bounds
import ../prefix/Order.bend as Order

# A previously proved widened endpoint gives an exact mathematical loop budget.
def count(+start: U32,+size: U32,+end: U32,
  +wide: {U64.add(U64.from_u32(start),U64.from_u32(size)) == U64.from_u32(end) : U64}) ->
  {Nat.add(U32.to_nat(size),U32.to_nat(start)) == U32.to_nat(end) : Nat}:
  Equal.trans(Nat,Nat.add(U32.to_nat(size),U32.to_nat(start)),Nat.add(U32.to_nat(start),U32.to_nat(size)),U32.to_nat(end),
    N.commute(U32.to_nat(size),U32.to_nat(start)),
    Equal.trans(Nat,Nat.add(U32.to_nat(start),U32.to_nat(size)),U32.to_nat(U32.add(start,size)),U32.to_nat(end),
      Equal.sym(Nat,U32.to_nat(U32.add(start,size)),Nat.add(U32.to_nat(start),U32.to_nat(size)),
        Sum.u32(start,size,Add.safe_from_wide(start,size,end,wide))),
      Equal.cong(U32,Nat,x => U32.to_nat(x),U32.add(start,size),end,Bounds.endpoint(start,size,end,wide))))

def budget(+start: U32,+size: U32,+end: U32,
  wide: {U64.add(U64.from_u32(start),U64.from_u32(size)) == U64.from_u32(end) : U64}) ->
  {Nat.is_le(Nat.add(U32.to_nat(size),U32.to_nat(start)),U32.to_nat(end)) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_le(Nat.add(U32.to_nat(size),U32.to_nat(start)),U32.to_nat(end)),
    Nat.is_le(U32.to_nat(end),U32.to_nat(end)),True{},
    Equal.cong(Nat,Bool,n => Nat.is_le(n,U32.to_nat(end)),
      Nat.add(U32.to_nat(size),U32.to_nat(start)),U32.to_nat(end),count(start,size,end,wide)),Order.nat_refl(U32.to_nat(end)))
