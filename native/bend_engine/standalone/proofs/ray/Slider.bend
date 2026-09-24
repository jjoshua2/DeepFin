import Base
import ./Spec.bend as S
import ./Path.bend as P
import ./Rays.bend as Rays
import ./Refinement.bend as R
import ../../Tables.bend as T
import ./Metadata.bend as Meta

def combine(+a: U64,+b: U64,+c: U64,+d: U64,ab: {a == b : U64},cd: {c == d : U64}) ->
  {U64.or(a,c) == U64.or(b,d) : U64}:
  Equal.trans(U64,U64.or(a,c),U64.or(b,c),U64.or(b,d),
    Equal.cong(U64,U64,x => U64.or(x,c),a,b,ab),Equal.cong(U64,U64,x => U64.or(b,x),c,d,cd))

def actual_four(+s: Nat,+d: Nat,+occ: U64) -> U64:
  U64.or(U64.or(T.ray(7n,False{},U32.from_nat(s),U32.from_nat(d),occ,False{},U64.zero()),T.ray(7n,False{},U32.from_nat(s),U32.from_nat(1n+d),occ,False{},U64.zero())),
    U64.or(T.ray(7n,False{},U32.from_nat(s),U32.from_nat(2n+d),occ,False{},U64.zero()),T.ray(7n,False{},U32.from_nat(s),U32.from_nat(3n+d),occ,False{},U64.zero())))

def four(+s: Nat,+d: Nat,+occ: U64,+sb: {Nat.is_lt(s,64n) == True{} : Bool},
  d0: {Nat.is_lt(d,8n) == True{} : Bool},d1: {Nat.is_lt(1n+d,8n) == True{} : Bool},
  d2: {Nat.is_lt(2n+d,8n) == True{} : Bool},d3: {Nat.is_lt(3n+d,8n) == True{} : Bool}) ->
  {actual_four(s,d,occ) == S.directional(s,d,occ) : U64}:
  combine(
    U64.or(T.ray(7n,False{},U32.from_nat(s),U32.from_nat(d),occ,False{},U64.zero()),T.ray(7n,False{},U32.from_nat(s),U32.from_nat(1n+d),occ,False{},U64.zero())),
    U64.or(P.attack(Rays.path(7n,s,d),occ),P.attack(Rays.path(7n,s,1n+d),occ)),
    U64.or(T.ray(7n,False{},U32.from_nat(s),U32.from_nat(2n+d),occ,False{},U64.zero()),T.ray(7n,False{},U32.from_nat(s),U32.from_nat(3n+d),occ,False{},U64.zero())),
    U64.or(P.attack(Rays.path(7n,s,2n+d),occ),P.attack(Rays.path(7n,s,3n+d),occ)),
    combine(T.ray(7n,False{},U32.from_nat(s),U32.from_nat(d),occ,False{},U64.zero()),P.attack(Rays.path(7n,s,d),occ),
      T.ray(7n,False{},U32.from_nat(s),U32.from_nat(1n+d),occ,False{},U64.zero()),P.attack(Rays.path(7n,s,1n+d),occ),R.initial(s,d,occ,sb,d0),R.initial(s,1n+d,occ,sb,d1)),
    combine(T.ray(7n,False{},U32.from_nat(s),U32.from_nat(2n+d),occ,False{},U64.zero()),P.attack(Rays.path(7n,s,2n+d),occ),
      T.ray(7n,False{},U32.from_nat(s),U32.from_nat(3n+d),occ,False{},U64.zero()),P.attack(Rays.path(7n,s,3n+d),occ),R.initial(s,2n+d,occ,sb,d2),R.initial(s,3n+d,occ,sb,d3)))

def square(+s: Nat,+bishop: Bool,+occ: U64,sb: {Nat.is_lt(s,64n) == True{} : Bool}) ->
  {T.slider(U32.from_nat(s),bishop,occ,False{}) == S.directional(s,Bool.pick(Nat,bishop,4n,0n),occ) : U64}:
  match bishop:
    case False{}: four(s,0n,occ,sb,{==},{==},{==},{==})
    case True{}: four(s,4n,occ,sb,{==},{==},{==},{==})

def fields(+k: Nat,+occ: U64,e: Meta.mapping(k)) -> {S.actual_slider(k,occ) == S.rays(k,occ) : U64}:
  (sq,tail) = e
  (bishop,bound) = tail
  Equal.trans(U64,S.actual_slider(k,occ),
    T.slider(U32.from_nat(Nat.mod(k,64n)),U32.is_ge(U32.from_nat(k),64),occ,False{}),S.rays(k,occ),
    Equal.cong(U32,U64,s => T.slider(s,U32.is_ge(U32.from_nat(k),64),occ,False{}),
      U32.mod(U32.from_nat(k),64),U32.from_nat(Nat.mod(k,64n)),sq),
    Equal.trans(U64,T.slider(U32.from_nat(Nat.mod(k,64n)),U32.is_ge(U32.from_nat(k),64),occ,False{}),
      T.slider(U32.from_nat(Nat.mod(k,64n)),Nat.is_ge(k,64n),occ,False{}),S.rays(k,occ),
      Equal.cong(Bool,U64,b => T.slider(U32.from_nat(Nat.mod(k,64n)),b,occ,False{}),U32.is_ge(U32.from_nat(k),64),Nat.is_ge(k,64n),bishop),
      square(Nat.mod(k,64n),Nat.is_ge(k,64n),occ,bound)))

def actual(+k: Nat,+occ: U64,valid: {Nat.is_lt(k,128n) == True{} : Bool}) -> {S.actual_slider(k,occ) == S.rays(k,occ) : U64}:
  fields(k,occ,Meta.finite(k,valid))
