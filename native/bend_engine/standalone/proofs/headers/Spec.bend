# Public specification names only actual production calls and intended headers.
import Base
import ../../Tables.bend as Tables
import ../layout/Spec.bend as Layout
import ../prefix/Spec.bend as Prefix

def count(before: Nat,after: Nat) -> Nat:
  Nat.add(before,1n+after)

def key(k: Nat) -> U32:
  U32.from_nat(k)

def query(+k: Nat,which: Bool) -> U32:
  Bool.pick(U32,which,U32.add(128,key(k)),key(k))

def expected(+k: Nat,which: Bool) -> U64:
  Bool.pick(U64,which,U64.from_u32(Prefix.prefix(k)),Layout.mask(key(k)))

def table(a: Array<U64>,before: Nat,after: Nat,+k: Nat) -> Array<U64>:
  Tables.tables(count(before,after),key(k),Prefix.prefix(k),a)

def run(a: Array<U64>,before: Nat,after: Nat,k: Nat,extra: Nat,start: Nat) -> Array<U64>:
  Tables.extras(extra,U32.from_nat(start),table(a,before,after,k))
