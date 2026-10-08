import Base
import ./Mask.bend as M
import ./Emission.bend as Emission
import ./Spec.bend as S
import ../protocol_replay_rows/Apply.bend as Prior
import ../protocol_replay_rows/Find.bend as Find
import ../protocol_replay_rows/Spec.bend as Selected
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../storage/Observe.bend as O
import ../table_preservation/Core.bend as T
import ../table_preservation/Generation.bend as Tables
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

def moved(+g: Pos.Game,maybe: Maybe<&2,Chess.Ply>,
  good: Selected.selected(m => M.safe(Chess.make_move(Pos.board(g),m)),maybe)) -> S.good(Protocol.move_apply(g,maybe)):
  match maybe:
    case None{}: Unit{}
    case Some{+m}:
      (child,unit)=good
      %Equal.sym(Chess.Board,Pos.board(Pos.apply(g,m)),Chess.make_move(Pos.board(g),m),Prior.board(g,m)) : M.safe(_)
      child
def found(+c: O.Cells,+g: Pos.Game,+text: String,r: Array<U64> & List<&2,Chess.Ply>,
  kept: {T.table(List<&2,Chess.Ply>,r) == O.pack(c) : Array<U64>},parent: M.safe(Pos.board(g)),
  emitted: C.every(m => M.emitted(Pos.board(g),m),E.moves(r))) -> S.root(c,Protocol.move_found(g,text,r)):
  (table,+xs)=r
  (kept,moved(g,Pos.find_move(xs,text),Find.all(m => M.safe(Chess.make_move(Pos.board(g),m)),xs,text,
    Emission.children(Pos.board(g),xs,parent,emitted))))
def legal(+c: O.Cells,+g: Pos.Game,+text: String,
  +parent: {U64.is_zero(U64.and(Chess.get_pawns(Pos.board(g)),M.mask())) == True{} : Bool}) ->
  S.root(c,Protocol.move_found(g,text,Chess.legal_moves(O.pack(c),Pos.board(g)))):
  Emission.legal(r => {T.table(List<&2,Chess.Ply>,r) == O.pack(c) : Array<U64>} ->
    S.root(c,Protocol.move_found(g,text,r)),O.pack(c),Pos.board(g),parent,
    r => emitted => kept => found(c,g,text,r,kept,parent,emitted))(Tables.legal(O.pack(c),Pos.board(g)))
def one(+c: O.Cells,+maybe: Maybe<&2,Pos.Game>,+text: String,parent: S.good(maybe)) ->
  S.root(c,Protocol.apply_one(text,maybe,O.pack(c))):
  match maybe:
    case None{}: ({==},Unit{})
    case Some{+g}: legal(c,g,text,parent)
