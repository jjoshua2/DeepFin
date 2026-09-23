# The actual-size and prefix certificates are produced by the accepted source laws.
import Base
import ./Spec.bend as Spec
import ./Nats.bend as N
import ./AddNat.bend as AddNat
import ../prefix/Spec.bend as Prefix
import ../prefix/LAWS.bend as PrefixLaws
import ../prefix/PROOF.bend as PrefixProof
import ../prefix/Nats.bend as P
import ../relative/Add.bend as Add
import ../relative/Bounds.bend as Bounds
import ../layout/Spec.bend as Layout

def observed_count(+at: U32,+size: U32,+end: U32,
  +wide: {U64.add(U64.from_u32(at),U64.from_u32(size)) == U64.from_u32(end) : U64}) ->
  {Nat.add(U32.to_nat(size),U32.to_nat(at)) == U32.to_nat(end) : Nat}:
  Equal.trans(Nat,Nat.add(U32.to_nat(size),U32.to_nat(at)),Nat.add(U32.to_nat(at),U32.to_nat(size)),U32.to_nat(end),
    N.comm(U32.to_nat(size),U32.to_nat(at)),
    Equal.trans(Nat,Nat.add(U32.to_nat(at),U32.to_nat(size)),U32.to_nat(U32.add(at,size)),U32.to_nat(end),
      Equal.sym(Nat,U32.to_nat(U32.add(at,size)),Nat.add(U32.to_nat(at),U32.to_nat(size)),
        AddNat.exact(at,size,Add.safe_from_wide(at,size,end,wide))),
      Equal.cong(U32,Nat,x=>U32.to_nat(x),U32.add(at,size),end,Bounds.endpoint(at,size,end,wide))))

def room(+k: Nat,valid: {Nat.is_lt(k,128n) == True{} : Bool}) ->
  {Spec.room(U32.to_nat(Layout.size(U32.from_nat(k))),Prefix.prefix(k),Prefix.prefix(1n+k)) == True{} : Bool}:
  Equal.trans(Bool,
    Spec.room(U32.to_nat(Layout.size(U32.from_nat(k))),Prefix.prefix(k),Prefix.prefix(1n+k)),
    Nat.is_le(U32.to_nat(Prefix.prefix(1n+k)),U32.to_nat(Prefix.prefix(1n+k))),True{},
    Equal.cong(Nat,Bool,x=>Nat.is_le(x,U32.to_nat(Prefix.prefix(1n+k))),
      Nat.add(U32.to_nat(Layout.size(U32.from_nat(k))),U32.to_nat(Prefix.prefix(k))),U32.to_nat(Prefix.prefix(1n+k)),
      observed_count(Prefix.prefix(k),Layout.size(U32.from_nat(k)),Prefix.prefix(1n+k),PrefixLaws.step_no_overflow(k,valid))),
    N.le_self(U32.to_nat(Prefix.prefix(1n+k))))

def endpoint(+k: Nat,valid: {Nat.is_lt(k,128n) == True{} : Bool}) ->
  {U32.is_lt(Prefix.prefix(1n+k),131072) == True{} : Bool}:
  Bounds.upper(1n+k,PrefixLaws.prefix_in_allocation(1n+k,valid))
