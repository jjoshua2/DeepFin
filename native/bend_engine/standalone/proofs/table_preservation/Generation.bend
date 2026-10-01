# Whole production generator composition, not a replacement generator.
import Base
import ./Core.bend as C
import ./Castles.bend as Castles
import ./Filters.bend as Filters
import ./Targets.bend as Targets
import ../storage/Observe.bend as O
import ../../../legal_probe/Chess.bend as Chess

def wings(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>) ->
  {C.table(List<&2,Chess.Ply>,Chess.castle_side(b,False{},Chess.castle_side(b,True{},r))) == C.table(List<&2,Chess.Ply>,r) : Array<U64>}:
  (table,+moves) = r
  C.apply(List<&2,Chess.Ply>,a => Chess.castle_side(b,False{},Chess.castle_side(b,True{},(a,moves))),table,+c =>
    Equal.trans(Array<U64>,
      C.table(List<&2,Chess.Ply>,Chess.castle_side(b,False{},Chess.castle_side(b,True{},(O.pack(c),moves)))),
      C.table(List<&2,Chess.Ply>,Chess.castle_side(b,True{},(O.pack(c),moves))),O.pack(c),
      Castles.side(b,False{},Chess.castle_side(b,True{},(O.pack(c),moves))),Castles.side(b,True{},(O.pack(c),moves))))

def suffix(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>) ->
  {C.table(List<&2,Chess.Ply>,Chess.filter_prepare(b,Chess.castle_side(b,False{},Chess.castle_side(b,True{},r)))) == C.table(List<&2,Chess.Ply>,r) : Array<U64>}:
  (table,+moves) = r
  C.apply(List<&2,Chess.Ply>,a => Chess.filter_prepare(b,Chess.castle_side(b,False{},Chess.castle_side(b,True{},(a,moves)))),table,+c =>
    Equal.trans(Array<U64>,
      C.table(List<&2,Chess.Ply>,Chess.filter_prepare(b,Chess.castle_side(b,False{},Chess.castle_side(b,True{},(O.pack(c),moves))))),
      C.table(List<&2,Chess.Ply>,Chess.castle_side(b,False{},Chess.castle_side(b,True{},(O.pack(c),moves)))),O.pack(c),
      Filters.prepare(b,Chess.castle_side(b,False{},Chess.castle_side(b,True{},(O.pack(c),moves)))),wings(b,(O.pack(c),moves))))

def legal(table: Array<U64>,+b: Chess.Board) ->
  {C.table(List<&2,Chess.Ply>,Chess.legal_moves(table,b)) == table : Array<U64>}:
  +own = Chess.color(b,Chess.get_turn(b))
  +squares = Chess.bit_squares(64n,U64.is_zero(own),own,Nil{})
  C.apply(List<&2,Chess.Ply>,a => Chess.legal_moves(a,b),table,+c =>
    Equal.trans(Array<U64>,C.table(List<&2,Chess.Ply>,Chess.legal_moves(O.pack(c),b)),
      C.table(List<&2,Chess.Ply>,Chess.scan(squares,b,(O.pack(c),Nil{}))),O.pack(c),
      suffix(b,Chess.scan(squares,b,(O.pack(c),Nil{}))),Targets.scan(squares,b,(O.pack(c),Nil{}))))
