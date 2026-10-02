# Exact empty flag-2 projection, with internal proved affine-array transport.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_kings/Spec.bend as K
import ../castle_emission/Spec.bend as E
import ../table_preservation/Core.bend as T
import ../table_preservation/Generation.bend as Generation
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ./CastleSpec.bend as P
import ./CastlePrefix.bend as Prefix
import ./KingAwayFilter.bend as Filter

def moves_cells(+c: O.Cells,+b: Chess.Board,+white: Bool,+sq: Nat,
  turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  one: {K.kings(b,white) == U64.bit(sq) : U64},
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  away: {Nat.is_eq(sq,U32.to_nat(K.src(white))) == False{} : Bool}) ->
  {P.project(E.moves(Chess.legal_moves(O.pack(c),b))) == Nil{} : List<&2,Chess.Ply>}:
  Prefix.ordinary_project(E.moves(Chess.legal_moves(O.pack(c),b)),
    Filter.plain(O.pack(c),b,white,sq,turn,one,bound,away))

def moves_lift(-table: Array<U64>,+b: Chess.Board,+white: Bool,+sq: Nat,
  turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  one: {K.kings(b,white) == U64.bit(sq) : U64},
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  away: {Nat.is_eq(sq,U32.to_nat(K.src(white))) == False{} : Bool},
  cert: Exists(O.Cells,c => {O.pack(c) == table : Array<U64>})) ->
  {P.project(E.moves(Chess.legal_moves(table,b))) == Nil{} : List<&2,Chess.Ply>}:
  (+c,same) = cert
  %same : {P.project(E.moves(Chess.legal_moves(_,b))) == Nil{} : List<&2,Chess.Ply>}
  moves_cells(c,b,white,sq,turn,one,bound,away)

def moves(table: Array<U64>,+b: Chess.Board,+white: Bool,+sq: Nat,
  turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  one: {K.kings(b,white) == U64.bit(sq) : U64},
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  away: {Nat.is_eq(sq,U32.to_nat(K.src(white))) == False{} : Bool}) ->
  {P.project(E.moves(Chess.legal_moves(table,b))) == Nil{} : List<&2,Chess.Ply>}:
  moves_lift(table,b,white,sq,turn,one,bound,away,R.reify(table))

def pair(-table: Array<U64>,r: Array<U64> & List<&2,Chess.Ply>,
  same: {T.table(List<&2,Chess.Ply>,r) == table : Array<U64>},
  empty: {P.project(E.moves(r)) == Nil{} : List<&2,Chess.Ply>}) ->
  {P.result(r) == (table,Nil{}) : Array<U64> & List<&2,Chess.Ply>}:
  (a,+xs) = r
  %same : {(a,P.project(xs)) == (_,Nil{}) : Array<U64> & List<&2,Chess.Ply>}
  %empty : {(a,P.project(xs)) == (a,_) : Array<U64> & List<&2,Chess.Ply>}
  {==}

def exact_cells(+c: O.Cells,+b: Chess.Board,+white: Bool,+sq: Nat,
  turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  one: {K.kings(b,white) == U64.bit(sq) : U64},
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  away: {Nat.is_eq(sq,U32.to_nat(K.src(white))) == False{} : Bool}) ->
  {P.result(Chess.legal_moves(O.pack(c),b)) == (O.pack(c),Nil{}) : Array<U64> & List<&2,Chess.Ply>}:
  pair(O.pack(c),Chess.legal_moves(O.pack(c),b),Generation.legal(O.pack(c),b),
    moves_cells(c,b,white,sq,turn,one,bound,away))

def exact_lift(-table: Array<U64>,+b: Chess.Board,+white: Bool,+sq: Nat,
  turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  one: {K.kings(b,white) == U64.bit(sq) : U64},
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  away: {Nat.is_eq(sq,U32.to_nat(K.src(white))) == False{} : Bool},
  cert: Exists(O.Cells,c => {O.pack(c) == table : Array<U64>})) ->
  {P.result(Chess.legal_moves(table,b)) == (table,Nil{}) : Array<U64> & List<&2,Chess.Ply>}:
  (+c,e) = cert
  +same: {O.pack(c) == table : Array<U64>} = e
  %same : {P.result(Chess.legal_moves(_,b)) == (table,Nil{}) : Array<U64> & List<&2,Chess.Ply>}
  %same : {P.result(Chess.legal_moves(O.pack(c),b)) == (_,Nil{}) : Array<U64> & List<&2,Chess.Ply>}
  exact_cells(c,b,white,sq,turn,one,bound,away)

def exact(table: Array<U64>,+b: Chess.Board,+white: Bool,+sq: Nat,
  turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  one: {K.kings(b,white) == U64.bit(sq) : U64},
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  away: {Nat.is_eq(sq,U32.to_nat(K.src(white))) == False{} : Bool}) ->
  {P.result(Chess.legal_moves(table,b)) == (table,Nil{}) : Array<U64> & List<&2,Chess.Ply>}:
  exact_lift(table,b,white,sq,turn,one,bound,away,R.reify(table))
