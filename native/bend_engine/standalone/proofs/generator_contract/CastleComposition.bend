# Compose actual scan, both wings, optimized filtering, and initialized geometry.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../board/Spec.bend as B
import ../castle_kings/Spec.bend as K
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as C
import ../castle_chain/Chain.bend as Chain
import ../table_preservation/Core.bend as T
import ../table_preservation/Generation.bend as Generation
import ../initialized_attacks/Build.bend as Init
import ./CastleSpec.bend as P
import ./CastlePrefix.bend as Prefix
import ./CastleProjection.bend as Projection
import ./CastleFinish.bend as Finish
import ./CastleUnique.bend as Unique

def transport(-b: Chess.Board,-x: Array<U64> & List<&2,Chess.Ply>,-y: Array<U64> & List<&2,Chess.Ply>,
  same: {x == y : Array<U64> & List<&2,Chess.Ply>},
  good: C.every(m => C.classified(b,m),E.moves(x))) ->
  C.every(m => C.classified(b,m),E.moves(y)):
  %same : C.every(m => C.classified(b,m),E.moves(_))
  good

def projected(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +good: {B.valid(b) == True{} : Bool},+turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64}) ->
  {P.project(E.moves(Chess.legal_moves(Init.run(d,seed,n,extra),b))) ==
    E.moves(Chess.filter_legal(
      P.project(P.before(b,white,False{},P.before(b,white,True{},E.moves(C.candidates(Init.run(d,seed,n,extra),b))))),
      b,(Init.run(d,seed,n,extra),Nil{}))) : List<&2,Chess.Ply>}:
  +xs = P.before(b,white,False{},P.before(b,white,True{},E.moves(C.candidates(Init.run(d,seed,n,extra),b))))
  Equal.trans(List<&2,Chess.Ply>,
    P.project(E.moves(Chess.legal_moves(Init.run(d,seed,n,extra),b))),
    P.project(E.moves(Chess.filter_prepare(b,(Init.run(d,seed,n,extra),xs)))),
    E.moves(Chess.filter_legal(P.project(xs),b,(Init.run(d,seed,n,extra),Nil{}))),
    Equal.cong(Array<U64> & List<&2,Chess.Ply>,List<&2,Chess.Ply>,
      r => P.project(E.moves(Chess.filter_prepare(b,r))),
      P.prefix(Init.run(d,seed,n,extra),b),(Init.run(d,seed,n,extra),xs),
      Prefix.exact(d,seed,n,extra,b,white,depth,tables,full,good,turn,one)),
    Projection.actual(Init.run(d,seed,n,extra),b,xs,
      transport(b,P.prefix(Init.run(d,seed,n,extra),b),(Init.run(d,seed,n,extra),xs),
        Prefix.exact(d,seed,n,extra,b,white,depth,tables,full,good,turn,one),
        Prefix.classified(Init.run(d,seed,n,extra),b))))

def moves(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +good: {B.valid(b) == True{} : Bool},+turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64}) ->
  {P.project(E.moves(Chess.legal_moves(Init.run(d,seed,n,extra),b))) ==
    P.expected(b,white) : List<&2,Chess.Ply>}:
  +tail = E.moves(C.candidates(Init.run(d,seed,n,extra),b))
  +own = Chess.color(b,Chess.get_turn(b))
  +keys = Chess.bit_squares(64n,U64.is_zero(own),own,Nil{})
  Equal.trans(List<&2,Chess.Ply>,
    P.project(E.moves(Chess.legal_moves(Init.run(d,seed,n,extra),b))),
    E.moves(Chess.filter_legal(P.project(P.before(b,white,False{},P.before(b,white,True{},tail))),b,(Init.run(d,seed,n,extra),Nil{}))),
    P.expected(b,white),
    projected(d,seed,n,extra,b,white,depth,tables,full,good,turn,one),
    Equal.trans(List<&2,Chess.Ply>,
      E.moves(Chess.filter_legal(P.project(P.before(b,white,False{},P.before(b,white,True{},tail))),b,(Init.run(d,seed,n,extra),Nil{}))),
      E.moves(Chess.filter_legal(P.before(b,white,False{},P.before(b,white,True{},Nil{})),b,(Init.run(d,seed,n,extra),Nil{}))),
      P.expected(b,white),
      Equal.cong(List<&2,Chess.Ply>,List<&2,Chess.Ply>,
        xs => E.moves(Chess.filter_legal(xs,b,(Init.run(d,seed,n,extra),Nil{}))),
        P.project(P.before(b,white,False{},P.before(b,white,True{},tail))),
        P.before(b,white,False{},P.before(b,white,True{},Nil{})),
        Prefix.two_project(b,white,tail,Chain.scan_plain(keys,b,Init.run(d,seed,n,extra)))),
      Equal.cong(Array<U64> & List<&2,Chess.Ply>,List<&2,Chess.Ply>,r => E.moves(r),
        Chess.filter_legal(P.before(b,white,False{},P.before(b,white,True{},Nil{})),b,(Init.run(d,seed,n,extra),Nil{})),
        (Init.run(d,seed,n,extra),P.expected(b,white)),
        Finish.both(d,seed,n,extra,b,white,depth,tables,full,good,turn,one))))

def pair(-a: Array<U64>,+wanted: List<&2,Chess.Ply>,r: Array<U64> & List<&2,Chess.Ply>,
  table: {T.table(List<&2,Chess.Ply>,r) == a : Array<U64>},
  output: {P.project(E.moves(r)) == wanted : List<&2,Chess.Ply>}) ->
  {P.result(r) == (a,wanted) : Array<U64> & List<&2,Chess.Ply>}:
  (t,+xs) = r
  %table : {(t,P.project(xs)) == (_,wanted) : Array<U64> & List<&2,Chess.Ply>}
  %output : {(t,P.project(xs)) == (t,_) : Array<U64> & List<&2,Chess.Ply>}
  {==}

def exact(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +good: {B.valid(b) == True{} : Bool},+turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64}) ->
  {P.result(Chess.legal_moves(Init.run(d,seed,n,extra),b)) ==
    (Init.run(d,seed,n,extra),P.expected(b,white)) : Array<U64> & List<&2,Chess.Ply>}:
  pair(Init.run(d,seed,n,extra),P.expected(b,white),Chess.legal_moves(Init.run(d,seed,n,extra),b),
    Generation.legal(Init.run(d,seed,n,extra),b),
    moves(d,seed,n,extra,b,white,depth,tables,full,good,turn,one))

def no_duplicates(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +good: {B.valid(b) == True{} : Bool},+turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64}) ->
  {Unique.unique(P.project(E.moves(Chess.legal_moves(Init.run(d,seed,n,extra),b)))) == True{} : Bool}:
  Equal.trans(Bool,
    Unique.unique(P.project(E.moves(Chess.legal_moves(Init.run(d,seed,n,extra),b)))),
    Unique.unique(P.expected(b,white)),True{},
    Equal.cong(List<&2,Chess.Ply>,Bool,xs => Unique.unique(xs),
      P.project(E.moves(Chess.legal_moves(Init.run(d,seed,n,extra),b))),P.expected(b,white),
      moves(d,seed,n,extra,b,white,depth,tables,full,good,turn,one)),
    Unique.expected(b,white))
