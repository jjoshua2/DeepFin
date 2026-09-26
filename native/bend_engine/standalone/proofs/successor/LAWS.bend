# Accepted laws about the actual Subsets.next / Sequence.at and public U64.sub.
# Capacity uses unbounded Nat; no chess-only restriction or 64-bit power overflow.
import Base
import ../../Subsets.bend as Subsets
import ../LAWS.bend as Sequence
import ../index/Bits.bend as Bits
import ./Spec.bend as Spec

law subtraction_refinement:
  for +a: U64
  for +b: U64
  {U64.to_word(U64.sub(a,b)) == Word.sub(64n,U64.to_word(a),U64.to_word(b)) : Word(64n)}

law step_successor:
  for +s: U64
  for +mask: U64
  for member: {U64.and(s,mask) == s : U64}
  {Bits.value(U64.pext(Subsets.next(s,mask),mask))
    == Spec.cycle(Bits.value(U64.pext(s,mask)),Bits.capacity(mask)) : Nat}

law sequence_ordinal:
  for +i: Nat
  for +mask: U64
  for +bounded: {Nat.is_lt(i,Bits.capacity(mask)) == True{} : Bool}
  {Bits.value(U64.pext(Sequence.at(i,mask),mask)) == i : Nat}

law sequence_nonduplicating:
  for +i: Nat
  for +j: Nat
  for +mask: U64
  for ib: {Nat.is_lt(i,Bits.capacity(mask)) == True{} : Bool}
  for jb: {Nat.is_lt(j,Bits.capacity(mask)) == True{} : Bool}
  for same: {Sequence.at(i,mask) == Sequence.at(j,mask) : U64}
  {i == j : Nat}

# The witness is the actual extracted index, not an assumed enumeration index.
law sequence_coverage:
  for +s: U64
  for +mask: U64
  for member: {U64.and(s,mask) == s : U64}
  exs i: Nat
  {Nat.is_lt(i,Bits.capacity(mask)) == True{} : Bool} & {Sequence.at(i,mask) == s : U64}

law cycle_endpoint:
  for +mask: U64
  {Sequence.at(Bits.capacity(mask),mask) == U64.zero() : U64}

law sequence_periodic:
  for +i: Nat
  for +mask: U64
  {Sequence.at(Nat.add(i,Bits.capacity(mask)),mask) == Sequence.at(i,mask) : U64}
