# The same actual initialized array for every class, square, Board and query.
import Base
import ../builder/Spec.bend as B

def run(d: Nat,seed: U64,n: Nat,extra: Nat) -> Array<U64>:
  B.run(d,seed,n,extra,0n)

def extras_bound(+extra: Nat,full: {extra == 64n : Nat}) ->
  {Nat.is_le(Nat.add(extra,0n),64n) == True{} : Bool}:
  Equal.cong(Nat,Bool,x => Nat.is_le(Nat.add(x,0n),64n),extra,64n,full)
