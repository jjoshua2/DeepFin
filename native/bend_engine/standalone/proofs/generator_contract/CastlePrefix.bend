# Totalize both initialized wing producers and retain the real ordinary scan.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../board/Spec.bend as B
import ../castle_kings/Spec.bend as K
import ../castle_emission/Spec.bend as E
import ../castle_emission/Flow.bend as Flow
import ../castle_sequence/Spec.bend as S
import ../castle_sequence/Composition.bend as Sequence
import ../castle_chain/Spec.bend as C
import ../castle_chain/Scan.bend as Scan
import ../castle_chain/Sides.bend as Sides
import ../castle_chain/Lists.bend as Lists
import ../castle_chain/Chain.bend as Chain
import ../table_preservation/Targets.bend as Targets
import ../accepted_castle/Generated.bend as Generated
import ../initialized_attacks/Build.bend as Init
import ./CastleSpec.bend as P

def side_case(ok: Bool,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,+ks: Bool,
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +good: {B.valid(b) == True{} : Bool},+turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64},
  guard: {E.guard(b,ks) == ok : Bool},+tail: List<&2,Chess.Ply>) ->
  {Chess.castle_side(b,ks,(Init.run(d,seed,n,extra),tail)) ==
    (Init.run(d,seed,n,extra),P.before_guard(ok,b,white,ks,tail)) : Array<U64> & List<&2,Chess.Ply>}:
  match ok:
    case False{}: Flow.rejected(b,ks,Init.run(d,seed,n,extra),tail,guard)
    case True{}: Sequence.producer(d,seed,n,extra,b,white,ks,depth,tables,full,good,turn,one,guard,tail)

def side(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,+ks: Bool,
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +good: {B.valid(b) == True{} : Bool},+turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64},+tail: List<&2,Chess.Ply>) ->
  {Chess.castle_side(b,ks,(Init.run(d,seed,n,extra),tail)) ==
    (Init.run(d,seed,n,extra),P.before(b,white,ks,tail)) : Array<U64> & List<&2,Chess.Ply>}:
  side_case(E.guard(b,ks),d,seed,n,extra,b,white,ks,depth,tables,full,good,turn,one,{==},tail)

def wings(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +good: {B.valid(b) == True{} : Bool},+turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64},+tail: List<&2,Chess.Ply>) ->
  {Chess.castle_side(b,False{},Chess.castle_side(b,True{},(Init.run(d,seed,n,extra),tail))) ==
    (Init.run(d,seed,n,extra),P.before(b,white,False{},P.before(b,white,True{},tail))) : Array<U64> & List<&2,Chess.Ply>}:
  Equal.trans(Array<U64> & List<&2,Chess.Ply>,
    Chess.castle_side(b,False{},Chess.castle_side(b,True{},(Init.run(d,seed,n,extra),tail))),
    Chess.castle_side(b,False{},(Init.run(d,seed,n,extra),P.before(b,white,True{},tail))),
    (Init.run(d,seed,n,extra),P.before(b,white,False{},P.before(b,white,True{},tail))),
    Equal.cong(Array<U64> & List<&2,Chess.Ply>,Array<U64> & List<&2,Chess.Ply>,r => Chess.castle_side(b,False{},r),
      Chess.castle_side(b,True{},(Init.run(d,seed,n,extra),tail)),
      (Init.run(d,seed,n,extra),P.before(b,white,True{},tail)),
      side(d,seed,n,extra,b,white,True{},depth,tables,full,good,turn,one,tail)),
    side(d,seed,n,extra,b,white,False{},depth,tables,full,good,turn,one,P.before(b,white,True{},tail)))

def scan_pair(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board) ->
  {C.candidates(Init.run(d,seed,n,extra),b) ==
    (Init.run(d,seed,n,extra),E.moves(C.candidates(Init.run(d,seed,n,extra),b))) : Array<U64> & List<&2,Chess.Ply>}:
  +own = Chess.color(b,Chess.get_turn(b))
  +keys = Chess.bit_squares(64n,U64.is_zero(own),own,Nil{})
  Generated.list_pair(Init.run(d,seed,n,extra),C.candidates(Init.run(d,seed,n,extra),b),
    Targets.scan(keys,b,(Init.run(d,seed,n,extra),Nil{})))

def exact(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +good: {B.valid(b) == True{} : Bool},+turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64}) ->
  {P.prefix(Init.run(d,seed,n,extra),b) ==
    (Init.run(d,seed,n,extra),P.before(b,white,False{},P.before(b,white,True{},E.moves(C.candidates(Init.run(d,seed,n,extra),b))))) : Array<U64> & List<&2,Chess.Ply>}:
  Equal.trans(Array<U64> & List<&2,Chess.Ply>,
    P.prefix(Init.run(d,seed,n,extra),b),
    Chess.castle_side(b,False{},Chess.castle_side(b,True{},(Init.run(d,seed,n,extra),E.moves(C.candidates(Init.run(d,seed,n,extra),b))))),
    (Init.run(d,seed,n,extra),P.before(b,white,False{},P.before(b,white,True{},E.moves(C.candidates(Init.run(d,seed,n,extra),b))))),
    Equal.cong(Array<U64> & List<&2,Chess.Ply>,Array<U64> & List<&2,Chess.Ply>,
      r => Chess.castle_side(b,False{},Chess.castle_side(b,True{},r)),
      C.candidates(Init.run(d,seed,n,extra),b),
      (Init.run(d,seed,n,extra),E.moves(C.candidates(Init.run(d,seed,n,extra),b))),
      scan_pair(d,seed,n,extra,b)),
    wings(d,seed,n,extra,b,white,depth,tables,full,good,turn,one,E.moves(C.candidates(Init.run(d,seed,n,extra),b))))

def classified_from_scan(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,
  plain: C.every(C.ordinary,E.moves(r))) ->
  C.every(m => C.classified(b,m),E.moves(Chess.castle_side(b,False{},Chess.castle_side(b,True{},r)))):
  (table,+xs) = r
  Sides.side(m => C.classified(b,m),
    z => C.every(m => C.classified(b,m),E.moves(Chess.castle_side(b,False{},z))),
    b,True{},(table,xs),Lists.lift(b,xs,plain),g => Inr{Inl{(g,{==})}},
    z => pz => Sides.side(m => C.classified(b,m),
      z => C.every(m => C.classified(b,m),E.moves(z)),
      b,False{},z,pz,g => Inr{Inr{(g,{==})}},z => pz => pz))

def classified(table: Array<U64>,+b: Chess.Board) ->
  C.every(m => C.classified(b,m),E.moves(P.prefix(table,b))):
  Scan.initial(z => C.every(m => C.classified(b,m),E.moves(Chess.castle_side(b,False{},Chess.castle_side(b,True{},z)))),
    table,b,z => plain => classified_from_scan(b,z,plain))

def ordinary_project(xs: List<&2,Chess.Ply>,plain: C.every(C.ordinary,xs)) ->
  {P.project(xs) == Nil{} : List<&2,Chess.Ply>}:
  match xs:
    case Nil{}: {==}
    case Con{+m,+tail}:
      (head,rest) = plain
      %Equal.sym(Bool,U32.is_eq(C.flag(m),2),False{},head) :
        {P.proj_bit(_,m,P.project(tail)) == Nil{} : List<&2,Chess.Ply>}
      ordinary_project(tail,rest)

def select_project(reject: Bool,+white: Bool,+ks: Bool,+tail: List<&2,Chess.Ply>) ->
  {P.project(S.select(reject,S.move(white,ks),tail)) ==
    S.select(reject,S.move(white,ks),P.project(tail)) : List<&2,Chess.Ply>}:
  match reject:
    case False{}: {==}
    case True{}: {==}

def before_project_case(ok: Bool,+b: Chess.Board,+white: Bool,+ks: Bool,+tail: List<&2,Chess.Ply>) ->
  {P.project(P.before_guard(ok,b,white,ks,tail)) ==
    P.before_guard(ok,b,white,ks,P.project(tail)) : List<&2,Chess.Ply>}:
  match ok:
    case False{}: {==}
    case True{}:
      select_project(Bool.or(S.checked(b,white,ks,K.Initial{}),S.checked(b,white,ks,K.Transit{})),white,ks,tail)

def before_project(+b: Chess.Board,+white: Bool,+ks: Bool,+tail: List<&2,Chess.Ply>) ->
  {P.project(P.before(b,white,ks,tail)) == P.before(b,white,ks,P.project(tail)) : List<&2,Chess.Ply>}:
  before_project_case(E.guard(b,ks),b,white,ks,tail)

def two_project(+b: Chess.Board,+white: Bool,+tail: List<&2,Chess.Ply>,
  plain: C.every(C.ordinary,tail)) ->
  {P.project(P.before(b,white,False{},P.before(b,white,True{},tail))) ==
    P.before(b,white,False{},P.before(b,white,True{},Nil{})) : List<&2,Chess.Ply>}:
  Equal.trans(List<&2,Chess.Ply>,
    P.project(P.before(b,white,False{},P.before(b,white,True{},tail))),
    P.before(b,white,False{},P.project(P.before(b,white,True{},tail))),
    P.before(b,white,False{},P.before(b,white,True{},Nil{})),
    before_project(b,white,False{},P.before(b,white,True{},tail)),
    Equal.cong(List<&2,Chess.Ply>,List<&2,Chess.Ply>,xs => P.before(b,white,False{},xs),
      P.project(P.before(b,white,True{},tail)),P.before(b,white,True{},Nil{}),
      Equal.trans(List<&2,Chess.Ply>,P.project(P.before(b,white,True{},tail)),
        P.before(b,white,True{},P.project(tail)),P.before(b,white,True{},Nil{}),
        before_project(b,white,True{},tail),
        Equal.cong(List<&2,Chess.Ply>,List<&2,Chess.Ply>,xs => P.before(b,white,True{},xs),
          P.project(tail),Nil{},ordinary_project(tail,plain)))))
