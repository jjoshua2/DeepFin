import Base
import ./Spec.bend as S
import ./Path.bend as P
import ./Rays.bend as R
import ./Trace.bend as Trace
import ./Traversal.bend as T
import ./Fold.bend as Fold
import ../../Tables.bend as Tables

def ray(+sq: Nat,+dir: Nat,+occ: U64,+acc: U64,
  sb: {Nat.is_lt(sq,64n) == True{} : Bool},db: {Nat.is_lt(dir,8n) == True{} : Bool}) ->
  {Tables.ray(7n,False{},U32.from_nat(sq),U32.from_nat(dir),occ,False{},acc) ==
    U64.or(acc,P.attack(R.path(7n,sq,dir),occ)) : U64}:
  Equal.trans(U64,Tables.ray(7n,False{},U32.from_nat(sq),U32.from_nat(dir),occ,False{},acc),
    S.scan(R.path(7n,sq,dir),occ,acc),U64.or(acc,P.attack(R.path(7n,sq,dir),occ)),
    T.actual(7n,U32.from_nat(sq),U32.from_nat(dir),R.path(7n,sq,dir),occ,acc,Trace.finite(sq,dir,sb,db)),
    Fold.accumulator(R.path(7n,sq,dir),occ,acc))

def initial(+sq: Nat,+dir: Nat,+occ: U64,
  sb: {Nat.is_lt(sq,64n) == True{} : Bool},db: {Nat.is_lt(dir,8n) == True{} : Bool}) ->
  {Tables.ray(7n,False{},U32.from_nat(sq),U32.from_nat(dir),occ,False{},U64.zero()) == P.attack(R.path(7n,sq,dir),occ) : U64}:
  Equal.trans(U64,Tables.ray(7n,False{},U32.from_nat(sq),U32.from_nat(dir),occ,False{},U64.zero()),
    S.scan(R.path(7n,sq,dir),occ,U64.zero()),P.attack(R.path(7n,sq,dir),occ),
    T.actual(7n,U32.from_nat(sq),U32.from_nat(dir),R.path(7n,sq,dir),occ,U64.zero(),Trace.finite(sq,dir,sb,db)),
    Fold.initial(R.path(7n,sq,dir),occ))
