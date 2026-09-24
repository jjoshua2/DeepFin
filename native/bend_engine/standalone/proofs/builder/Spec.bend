# Symbolic form of the public build recipe. Public-body linkage is separately guarded.
import Base
import ../../Tables.bend as Tables

def run(+d: Nat,seed: U64,n: Nat,extra: Nat,start: Nat) -> Array<U64>:
  Tables.extras(extra,U32.from_nat(start),Tables.tables(n,0,512,Array.new(U64,d,seed)))
