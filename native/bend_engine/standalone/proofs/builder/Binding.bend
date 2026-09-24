# Scalar-domain binding to the actual full construction schedule.
import Base
import ./Spec.bend as B
import ./Domain.bend as D
import ../data/Spec.bend as Data

def count(+key: Nat,cert: D.statement(key)) -> {Data.count(key,D.suffix(key)) == 128n : Nat}:
  (same,rest) = cert
  same

def budget(+key: Nat,cert: D.statement(key)) ->
  {Nat.is_lt(Nat.add(Data.count(key,D.suffix(key)),0n),129n) == True{} : Bool}:
  (same,rest) = cert
  (selected,bound) = rest
  bound

def image(+d: Nat,+seed: U64,+n: Nat,+key: Nat,+extra: Nat,+start: Nat,
  valid: {Nat.is_lt(key,128n) == True{} : Bool},full: {n == 128n : Nat}) ->
  {Data.run(Array.new(U64,d,seed),key,D.suffix(key),0n,extra,start) == B.run(d,seed,n,extra,start) : Array<U64>}:
  Equal.cong(Nat,Array<U64>,size => B.run(d,seed,size,extra,start),Data.count(key,D.suffix(key)),n,
    Equal.trans(Nat,Data.count(key,D.suffix(key)),128n,n,count(key,D.full_configuration(key,valid)),Equal.sym(Nat,n,128n,full)))
