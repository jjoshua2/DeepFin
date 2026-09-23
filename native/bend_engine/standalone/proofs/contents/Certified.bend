# Full-block count/end facts are obtained from the accepted prefix proof.
import Base
import ./Actual.bend as Actual
import ../fill_interval/Spec.bend as Block
import ../fill_interval/Block.bend as Budget
import ../prefix/Spec.bend as Prefix
import ../prefix/LAWS.bend as PrefixLaws
import ../prefix/PROOF.bend as PrefixProof
import ../relative/Bounds.bend as Bounds
import ../layout/Spec.bend as Layout
import ../complete/Spec.bend as C
import ../storage/Observe.bend as O
import ../LAWS.bend as Sequence
import ../PROOF.bend as SequenceProof
import ../../Tables.bend as Tables

def block(a: Array<U64>,+k: Nat,+i: Nat,
  +valid: {Nat.is_lt(k,128n) == True{} : Bool},
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  ib: {Nat.is_lt(i,U32.to_nat(Layout.size(U32.from_nat(k)))) == True{} : Bool}) ->
  {Array.get(U64,Block.block(k,a),U32.add(Prefix.prefix(k),U32.from_nat(i))) ==
    (Block.block(k,a),Tables.slider(U32.mod(U32.from_nat(k),64),U32.is_ge(U32.from_nat(k),64),Sequence.at(i,Layout.mask(U32.from_nat(k))),False{})) : Array<U64> & U64}:
  +key = U32.from_nat(k)
  Actual.complete_pair(a,U32.to_nat(Layout.size(key)),i,Prefix.prefix(k),Prefix.prefix(1n+k),U32.mod(key,64),U32.is_ge(key,64),Layout.mask(key),shape,ib,
    Bounds.upper(1n+k,PrefixLaws.prefix_in_allocation(1n+k,valid)),
    Budget.budget(Prefix.prefix(k),Layout.size(key),Prefix.prefix(1n+k),PrefixLaws.step_no_overflow(k,valid)))
