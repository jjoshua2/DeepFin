import Base
import ./Spec.bend as S
import ./Words.bend as U
import ./Rights.bend as Rights
import ../protocol_replay_rows/Apply.bend as Prior
import ../storage/Observe.bend as O
import ../table_preservation/Core.bend as T
import ../table_preservation/Generation.bend as Tables
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

# The result applies the actual first text match, with its complete Ply fields.
def moved(+limit: U32,+g: Pos.Game,+maybe: Maybe<&2,Chess.Ply>,
  good: U.subset(limit,Chess.get_rights(Pos.board(g)))) -> S.within(limit,Protocol.move_apply(g,maybe)):
  match maybe:
    case None{}: Unit{}
    case Some{+m}:
      %Equal.sym(Chess.Board,Pos.board(Pos.apply(g,m)),Chess.make_move(Pos.board(g),m),Prior.board(g,m)) :
        U.subset(limit,Chess.get_rights(_))
      U.trans(limit,Chess.get_rights(Pos.board(g)),Chess.get_rights(Chess.make_move(Pos.board(g),m)),good,Rights.subset(Pos.board(g),m))
def found(+c: O.Cells,+limit: U32,+g: Pos.Game,+text: String,r: Array<U64> & List<&2,Chess.Ply>,
  kept: {T.table(List<&2,Chess.Ply>,r) == O.pack(c) : Array<U64>},
  good: U.subset(limit,Chess.get_rights(Pos.board(g)))) -> S.root(c,limit,Protocol.move_found(g,text,r)):
  (a,+xs)=r
  (kept,moved(limit,g,Pos.find_move(xs,text),good))
def one(+c: O.Cells,+limit: U32,+maybe: Maybe<&2,Pos.Game>,+text: String,good: S.within(limit,maybe)) ->
  S.root(c,limit,Protocol.apply_one(text,maybe,O.pack(c))):
  match maybe:
    case None{}: ({==},Unit{})
    case Some{+g}:
      found(c,limit,g,text,Chess.legal_moves(O.pack(c),Pos.board(g)),Tables.legal(O.pack(c),Pos.board(g)),good)
