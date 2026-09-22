# Actual-size and endpoint facts are obtained from the accepted proof producer.
import Base
import ./Spec.bend as Spec
import ./Actual.bend as Actual
import ./Block.bend as Block
import ../prefix/Spec.bend as Prefix
import ../prefix/LAWS.bend as PrefixLaws
import ../prefix/PROOF.bend as PrefixProof
import ../prefix/Order.bend as Order
import ../relative/Bounds.bend as Bounds
import ../layout/Spec.bend as Layout
import ../complete/Spec.bend as Complete
import ../storage/Observe.bend as O
import ../separation/Spec.bend as S

def before(a: Array<U64>,+k: Nat,+q: U32,
  +valid: {Nat.is_lt(k,128n) == True{} : Bool},
  shape: {O.shape(a) == Complete.full(17n) : O.Shape},
  qb: {U32.is_lt(q,Prefix.prefix(k)) == True{} : Bool}) ->
  {S.value(Array.get(U64,Spec.block(k,a),q)) == S.value(Array.get(U64,a,q)) : U64}:
  +key = U32.from_nat(k)
  Actual.frame_before(a,U32.to_nat(Layout.size(key)),Prefix.prefix(k),Prefix.prefix(1n+k),q,
    U32.mod(key,64),U32.is_ge(key,64),Layout.mask(key),U64.zero(),shape,
    Bounds.upper(1n+k,PrefixLaws.prefix_in_allocation(1n+k,valid)),qb,
    Block.budget(Prefix.prefix(k),Layout.size(key),Prefix.prefix(1n+k),PrefixLaws.step_no_overflow(k,valid)))

def after(a: Array<U64>,+k: Nat,+q: U32,
  +valid: {Nat.is_lt(k,128n) == True{} : Bool},
  shape: {O.shape(a) == Complete.full(17n) : O.Shape},
  start: {U32.is_le(Prefix.prefix(1n+k),q) == True{} : Bool},
  qb: {U32.is_lt(q,131072) == True{} : Bool}) ->
  {S.value(Array.get(U64,Spec.block(k,a),q)) == S.value(Array.get(U64,a,q)) : U64}:
  +key = U32.from_nat(k)
  Actual.frame_after(a,U32.to_nat(Layout.size(key)),Prefix.prefix(k),q,
    U32.mod(key,64),U32.is_ge(key,64),Layout.mask(key),U64.zero(),shape,qb,
    Order.nat_trans(Nat.add(U32.to_nat(Layout.size(key)),U32.to_nat(Prefix.prefix(k))),U32.to_nat(Prefix.prefix(1n+k)),U32.to_nat(q),
      Block.budget(Prefix.prefix(k),Layout.size(key),Prefix.prefix(1n+k),PrefixLaws.step_no_overflow(k,valid)),
      Order.to_nat(Prefix.prefix(1n+k),q,start)))
