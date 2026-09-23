# Computed inside-block values, not yet complete stored-header/ray refinement.
import Base
import ./Spec.bend as Spec
import ../../Tables.bend as Tables
import ../LAWS.bend as Sequence
import ../complete/Spec.bend as C
import ../storage/Observe.bend as O
import ../separation/Spec.bend as S
import ../fill_interval/Spec.bend as Block
import ../prefix/Spec.bend as Prefix
import ../layout/Spec.bend as Layout

law bounded_fill_entry:
  for a: Array<U64>
  for +n: Nat
  for +i: Nat
  for +at: U32
  for +end: U32
  for +sq: U32
  for +bishop: Bool
  for +mask: U64
  for +subset: U64
  for shape: {O.shape(a) == C.full(17n) : O.Shape}
  for ib: {Nat.is_lt(i,n) == True{} : Bool}
  for eb: {U32.is_lt(end,131072) == True{} : Bool}
  for budget: {Nat.is_le(Nat.add(n,U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool}
  {S.value(Array.get(U64,Tables.fill(n,at,sq,bishop,mask,subset,a),U32.add(at,U32.from_nat(i)))) ==
    Tables.slider(sq,bishop,Spec.state(i,mask,subset),False{}) : U64}

law bounded_zero_fill_read:
  for a: Array<U64>
  for +n: Nat
  for +i: Nat
  for +at: U32
  for +end: U32
  for +sq: U32
  for +bishop: Bool
  for +mask: U64
  for shape: {O.shape(a) == C.full(17n) : O.Shape}
  for ib: {Nat.is_lt(i,n) == True{} : Bool}
  for eb: {U32.is_lt(end,131072) == True{} : Bool}
  for budget: {Nat.is_le(Nat.add(n,U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool}
  {Array.get(U64,Tables.fill(n,at,sq,bishop,mask,U64.zero(),a),U32.add(at,U32.from_nat(i))) ==
    (Tables.fill(n,at,sq,bishop,mask,U64.zero(),a),Tables.slider(sq,bishop,Sequence.at(i,mask),False{})) : Array<U64> & U64}

law full_block_entry:
  for a: Array<U64>
  for +k: Nat
  for +i: Nat
  for +valid: {Nat.is_lt(k,128n) == True{} : Bool}
  for shape: {O.shape(a) == C.full(17n) : O.Shape}
  for ib: {Nat.is_lt(i,U32.to_nat(Layout.size(U32.from_nat(k)))) == True{} : Bool}
  {Array.get(U64,Block.block(k,a),U32.add(Prefix.prefix(k),U32.from_nat(i))) ==
    (Block.block(k,a),Tables.slider(U32.mod(U32.from_nat(k),64),U32.is_ge(U32.from_nat(k),64),Sequence.at(i,Layout.mask(U32.from_nat(k))),False{})) : Array<U64> & U64}
