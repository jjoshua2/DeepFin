# Actual builder observations; computed values are not an independent ray oracle.
import Base
import ../../Tables.bend as Tables
import ../prefix/Spec.bend as P
import ../layout/Spec.bend as Layout
import ../LAWS.bend as Sequence

def count(before: Nat,after: Nat) -> Nat:
  Nat.add(before,1n+after)

def query(+k: Nat,i: Nat) -> U32:
  U32.add(P.prefix(k),U32.from_nat(i))

def expected(+k: Nat,i: Nat) -> U64:
  +key = U32.from_nat(k)
  Tables.slider(U32.mod(key,64),U32.is_ge(key,64),Sequence.at(i,Layout.mask(key)),False{})

def table(a: Array<U64>,before: Nat,after: Nat,+k: Nat) -> Array<U64>:
  Tables.tables(count(before,after),U32.from_nat(k),P.prefix(k),a)

def run(a: Array<U64>,before: Nat,after: Nat,k: Nat,extra: Nat,start: Nat) -> Array<U64>:
  Tables.extras(extra,U32.from_nat(start),table(a,before,after,k))
