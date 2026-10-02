# Preserve ordinary provenance through the real off-home suffix, for any array.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_kings/Spec.bend as K
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as C
import ../castle_chain/Scan.bend as Scan
import ../castle_chain/Filter.bend as Filter
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ./KingAwayWings.bend as Wings

def filtered(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,
  plain: C.every(C.ordinary,E.moves(r))) ->
  C.every(C.ordinary,E.moves(Chess.filter_prepare(b,r))):
  Filter.prepare(C.ordinary,z => C.every(C.ordinary,E.moves(z)),b,r,plain,z => pz => pz)

def suffix_cells(+c: O.Cells,+b: Chess.Board,+white: Bool,+sq: Nat,
  turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  one: {K.kings(b,white) == U64.bit(sq) : U64},
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  away: {Nat.is_eq(sq,U32.to_nat(K.src(white))) == False{} : Bool},
  +xs: List<&2,Chess.Ply>,plain: C.every(C.ordinary,xs)) ->
  C.every(C.ordinary,E.moves(C.suffix(b,(O.pack(c),xs)))):
  %Equal.sym(Array<U64> & List<&2,Chess.Ply>,
    Chess.castle_side(b,False{},Chess.castle_side(b,True{},(O.pack(c),xs))),(O.pack(c),xs),
    Wings.cells(c,b,white,sq,turn,one,bound,away,xs)) :
    C.every(C.ordinary,E.moves(Chess.filter_prepare(b,_)))
  filtered(b,(O.pack(c),xs),plain)

def suffix_lift(-table: Array<U64>,+b: Chess.Board,+white: Bool,+sq: Nat,
  turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  one: {K.kings(b,white) == U64.bit(sq) : U64},
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  away: {Nat.is_eq(sq,U32.to_nat(K.src(white))) == False{} : Bool},
  +xs: List<&2,Chess.Ply>,plain: C.every(C.ordinary,xs),
  cert: Exists(O.Cells,c => {O.pack(c) == table : Array<U64>})) ->
  C.every(C.ordinary,E.moves(C.suffix(b,(table,xs)))):
  (+c,same) = cert
  %same : C.every(C.ordinary,E.moves(C.suffix(b,(_,xs))))
  suffix_cells(c,b,white,sq,turn,one,bound,away,xs,plain)

def suffix(+b: Chess.Board,+white: Bool,+sq: Nat,
  turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  one: {K.kings(b,white) == U64.bit(sq) : U64},
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  away: {Nat.is_eq(sq,U32.to_nat(K.src(white))) == False{} : Bool},
  r: Array<U64> & List<&2,Chess.Ply>,plain: C.every(C.ordinary,E.moves(r))) ->
  C.every(C.ordinary,E.moves(C.suffix(b,r))):
  (table,+xs) = r
  suffix_lift(table,b,white,sq,turn,one,bound,away,xs,plain,R.reify(table))

def plain(table: Array<U64>,+b: Chess.Board,+white: Bool,+sq: Nat,
  +turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {K.kings(b,white) == U64.bit(sq) : U64},
  +bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  +away: {Nat.is_eq(sq,U32.to_nat(K.src(white))) == False{} : Bool}) ->
  C.every(C.ordinary,E.moves(Chess.legal_moves(table,b))):
  Scan.initial(z => C.every(C.ordinary,E.moves(C.suffix(b,z))),table,b,
    z => pz => suffix(b,white,sq,turn,one,bound,away,z,pz))
