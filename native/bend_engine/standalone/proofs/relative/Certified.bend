# Public certificate producers use already discharged parent laws.
import Base
import ./Spec.bend as S
import ./Bounds.bend as Bounds
import ./Add.bend as Add
import ./Order.bend as Order
import ../prefix/Spec.bend as Prefix
import ../prefix/LAWS.bend as PrefixLaws
import ../prefix/PROOF.bend as PrefixProof
import ../layout/Spec.bend as Layout
import ../layout/Domain.bend as Domain
import ../layout/LAWS.bend as LayoutLaws
import ../layout/PROOF.bend as LayoutProof

def certificate(+k: Nat,+r: U32,+valid: {Nat.is_lt(k,128n) == True{} : Bool},
  rb: {U32.is_lt(r,Layout.size(U32.from_nat(k))) == True{} : Bool}) ->
  S.interval(k,r) & {U64.from_u32(S.address(k,r)) == U64.add(U64.from_u32(Prefix.prefix(k)),U64.from_u32(r)) : U64}:
  +wide = PrefixLaws.step_no_overflow(k,valid)
  Bounds.finish(k,r,wide,Bounds.upper(1n+k,PrefixLaws.prefix_in_allocation(1n+k,valid)),
    Add.increasing32(Prefix.prefix(k),r,Layout.size(U32.from_nat(k)),
      Add.safe_from_wide(Prefix.prefix(k),Layout.size(U32.from_nat(k)),Prefix.prefix(1n+k),wide),rb))

# Convert the existing arbitrary-occupancy Nat certificate, not a sampled PEXT oracle.
def lookup_index(+k: Nat,+occ: U64,+valid: {Nat.is_lt(k,128n) == True{} : Bool}) ->
  {U32.is_lt(S.index(k,occ),Layout.size(U32.from_nat(k))) == True{} : Bool}:
  +key = U32.from_nat(k)
  +key_valid = Equal.trans(Bool,Nat.is_lt(U32.to_nat(key),128n),Nat.is_lt(k,128n),True{},
    Equal.cong(Nat,Bool,n => Nat.is_lt(n,128n),U32.to_nat(key),k,Domain.exact(k,valid)),valid)
  Order.from_nat_lt(S.index(k,occ),Layout.size(key),LayoutLaws.lookup_index_bound(key,occ,key_valid))
