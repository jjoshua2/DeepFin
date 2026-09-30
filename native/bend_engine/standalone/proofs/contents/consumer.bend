import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Spec.bend as Spec
import ./Schedule.bend as Schedule
import ../../Tables.bend as Tables
import ../LAWS.bend as Sequence
import ../PROOF.bend as SequenceProof
import ../complete/Spec.bend as C
import ../complete/Actual.bend as Complete
import ../storage/Observe.bend as O
import ../separation/Spec.bend as S
import ../fill_interval/Spec.bend as Block
import ../prefix/Spec.bend as Prefix
import ../layout/Spec.bend as Layout

# Every valid key and every valid relative index: no caller-supplied endpoint,
# actual block-size equality, clear-write certificate, or expected read value.
def use_block(a: Array<U64>,+k: Nat,+i: Nat,
  valid: {Nat.is_lt(k,128n) == True{} : Bool},
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  ib: {Nat.is_lt(i,U32.to_nat(Layout.size(U32.from_nat(k)))) == True{} : Bool}) ->
  {Array.get(U64,Block.block(k,a),U32.add(Prefix.prefix(k),U32.from_nat(i))) ==
    (Block.block(k,a),Tables.slider(U32.mod(U32.from_nat(k),64),U32.is_ge(U32.from_nat(k),64),Sequence.at(i,Layout.mask(U32.from_nat(k))),False{})) : Array<U64> & U64}:
  Laws.full_block_entry(a,k,i,valid,shape,ib)

# Complete-array inhabitants are provided by the existing universal allocation
# law. Keep the array symbolic in the value-sensitive consumer to avoid eagerly
# normalizing a depth-17 constructor value merely to check the example.
def initialized_shape(+d: Nat,+seed: U64) ->
  {O.shape(Array.new(U64,d,seed)) == C.full(d) : O.Shape}:
  Complete.allocation(d,seed)

def middle(a: Array<U64>,+mask: U64,shape: {O.shape(a) == C.full(17n) : O.Shape}) ->
  {Array.get(U64,Tables.fill(3n,512,0,False{},mask,U64.zero(),a),513) ==
    (Tables.fill(3n,512,0,False{},mask,U64.zero(),a),Tables.slider(0,False{},Sequence.at(1n,mask),False{})) : Array<U64> & U64}:
  Laws.bounded_zero_fill_read(a,3n,1n,512,515,0,False{},mask,shape,{==},{==},{==})

# The schedule bridge itself is modular; bounded contents do not pretend that
# U32 arithmetic remains nonwrapping without the numeric budget.
def address_wrap() -> {Spec.address(2n,4294967295) == 1 : U32}:
  Schedule.address(2n,4294967295)

def excluded_index() -> {Nat.is_lt(3n,3n) == False{} : Bool}: {==}

def excluded_overrun() ->
  {Nat.is_le(Nat.add(3n,U32.to_nat(2)),U32.to_nat(4)) == False{} : Bool}: {==}


def excluded_allocation_end() -> {U32.is_lt(131072,131072) == False{} : Bool}: {==}
