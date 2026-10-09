import Base
import ./Spec.bend as S
import ./Cases.bend as F
import ./Algebra.bend as A
import ../attack_geometry/Initialized.bend as I
import ../attack_geometry/Geometry.bend as G
import ../child_metadata/Range.bend as R
import ../u64/Words.bend as Old
import ../../../legal_probe/Chess.bend as C

def and_left(+a: Bool,+b: Bool,e: {Bool.and(a,b) == True{} : Bool}) -> {a == True{} : Bool}:
  match a:
    case True{}: {==}
    case False{}: Old.false_true({False{} == True{} : Bool},e)

def and_right(+a: Bool,+b: Bool,e: {Bool.and(a,b) == True{} : Bool}) -> {b == True{} : Bool}:
  match a:
    case True{}: e
    case False{}: Old.false_true({b == True{} : Bool},e)

def attack(+d: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,+b: C.Board,+s: U32,+w: Bool,
  +depth: {d == 17n : Nat},+full: {extra == 64n : Nat},+below: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},
  white: {U32.is_eq(C.get_turn(b),1) == w : Bool}) ->
  {C.attack(0,I.run(d,seed,n,key,at,extra),s,C.get_turn(b),C.occupied(b)) ==
   (I.run(d,seed,n,key,at,extra),G.mask(S.kind(w),U32.to_nat(s))) : Array<U64> & U64}:
  match w:
    case True{}:
      %F.recover(s,U32.to_nat(s),{==},below) :
        {C.attack(0,I.run(d,seed,n,key,at,extra),_,C.get_turn(b),C.occupied(b)) ==
          (I.run(d,seed,n,key,at,extra),G.mask(G.WhitePawn{},U32.to_nat(s))) : Array<U64> & U64}
      %Equal.sym(Bool,U32.is_eq(C.get_turn(b),1),True{},white) :
        {Array.get(U64,I.run(d,seed,n,key,at,extra),U32.add(U32.from_nat(U32.to_nat(s)),Bool.pick(U32,_,384,448))) ==
          (I.run(d,seed,n,key,at,extra),G.mask(G.WhitePawn{},U32.to_nat(s))) : Array<U64> & U64}
      I.query(d,seed,n,key,at,extra,G.WhitePawn{},U32.to_nat(s),C.occupied(b),depth,full,below)
    case False{}:
      %F.recover(s,U32.to_nat(s),{==},below) :
        {C.attack(0,I.run(d,seed,n,key,at,extra),_,C.get_turn(b),C.occupied(b)) ==
          (I.run(d,seed,n,key,at,extra),G.mask(G.BlackPawn{},U32.to_nat(s))) : Array<U64> & U64}
      %Equal.sym(Bool,U32.is_eq(C.get_turn(b),1),False{},white) :
        {Array.get(U64,I.run(d,seed,n,key,at,extra),U32.add(U32.from_nat(U32.to_nat(s)),Bool.pick(U32,_,384,448))) ==
          (I.run(d,seed,n,key,at,extra),G.mask(G.BlackPawn{},U32.to_nat(s))) : Array<U64> & U64}
      I.query(d,seed,n,key,at,extra,G.BlackPawn{},U32.to_nat(s),C.occupied(b),depth,full,below)

def targets(+d: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,+b: C.Board,+s: U32,
  depth: {d == 17n : Nat},full: {extra == 64n : Nat},below: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool}) ->
  {C.pawn_targets(I.run(d,seed,n,key,at,extra),b,s) == (I.run(d,seed,n,key,at,extra),S.targets(b,s)) : Array<U64> & U64}:
  Equal.cong(Array<U64> & U64,Array<U64> & U64,r => C.pawn_after(U64.or(S.one(b,s),S.two(b,s)),S.capturable(b),r),
    C.attack(0,I.run(d,seed,n,key,at,extra),s,C.get_turn(b),C.occupied(b)),
    (I.run(d,seed,n,key,at,extra),G.mask(S.kind(S.white(b)),U32.to_nat(s))),
    attack(d,seed,n,key,at,extra,b,s,S.white(b),depth,full,below,{==}))

def model_facts(+b: C.Board,+s: U32,+below: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},
  hit: {U64.test_bit(S.targets(b,s),U32.to_nat(U32.xor(s,16))) == True{} : Bool},
  fields: F.facts(s,S.white(b))) -> S.double(b,s,U32.xor(s,16)):
  ((pawn_absent,first_absent),+bound)=fields
  +i=U32.to_nat(U32.xor(s,16))
  +guard=A.selected_true(S.empty2(b,s),U64.bit(U32.to_nat(S.second(s,S.white(b)))),i,
    A.second(S.one(b,s),S.two(b,s),G.mask(S.kind(S.white(b)),U32.to_nat(s)),S.capturable(b),i,bound,
      A.selected_absent(S.empty1(b,s),U64.bit(U32.to_nat(S.first(s,S.white(b)))),i,first_absent),pawn_absent,hit))
  (guard,F.destination(s,U32.to_nat(s),S.white(b),{==},below,
    and_left(S.rank(b,s),Bool.not(U64.test_bit(C.occupied(b),U32.to_nat(S.second(s,S.white(b))))),
      and_right(S.empty1(b,s),Bool.and(S.rank(b,s),Bool.not(U64.test_bit(C.occupied(b),U32.to_nat(S.second(s,S.white(b)))))),guard))))

def model(+b: C.Board,+s: U32,+below: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},
  hit: {U64.test_bit(S.targets(b,s),U32.to_nat(U32.xor(s,16))) == True{} : Bool}) -> S.double(b,s,U32.xor(s,16)):
  model_facts(b,s,below,hit,F.finite(s,U32.to_nat(s),S.white(b),{==},below))

def generated(+b: C.Board,+s: U32,+dst: U32,+below: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},
  +trigger: {U32.is_eq(U32.xor(s,dst),16) == True{} : Bool},
  hit: {U64.test_bit(S.targets(b,s),U32.to_nat(dst)) == True{} : Bool}) -> S.double(b,s,dst):
  %Equal.sym(U32,dst,U32.xor(s,16),R.destination(s,dst,trigger)) : S.double(b,s,_)
  model(b,s,below,Equal.trans(Bool,U64.test_bit(S.targets(b,s),U32.to_nat(U32.xor(s,16))),
    U64.test_bit(S.targets(b,s),U32.to_nat(dst)),True{},
    Equal.cong(U32,Bool,x => U64.test_bit(S.targets(b,s),U32.to_nat(x)),U32.xor(s,16),dst,
      Equal.sym(U32,dst,U32.xor(s,16),R.destination(s,dst,trigger))),hit))

def initialized(+d: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,+b: C.Board,+s: U32,+dst: U32,
  +depth: {d == 17n : Nat},+full: {extra == 64n : Nat},+below: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},
  trigger: {U32.is_eq(U32.xor(s,dst),16) == True{} : Bool},
  hit: {U64.test_bit(S.mask(C.pawn_targets(I.run(d,seed,n,key,at,extra),b,s)),U32.to_nat(dst)) == True{} : Bool}) -> S.double(b,s,dst):
  generated(b,s,dst,below,trigger,
    Equal.trans(Bool,U64.test_bit(S.targets(b,s),U32.to_nat(dst)),
      U64.test_bit(S.mask(C.pawn_targets(I.run(d,seed,n,key,at,extra),b,s)),U32.to_nat(dst)),True{},
      Equal.sym(Bool,U64.test_bit(S.mask(C.pawn_targets(I.run(d,seed,n,key,at,extra),b,s)),U32.to_nat(dst)),U64.test_bit(S.targets(b,s),U32.to_nat(dst)),
        Equal.cong(Array<U64> & U64,Bool,r => U64.test_bit(S.mask(r),U32.to_nat(dst)),
          C.pawn_targets(I.run(d,seed,n,key,at,extra),b,s),(I.run(d,seed,n,key,at,extra),S.targets(b,s)),
          targets(d,seed,n,key,at,extra,b,s,depth,full,below))),hit))
