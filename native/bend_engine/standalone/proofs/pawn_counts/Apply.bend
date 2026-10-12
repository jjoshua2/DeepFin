import Base
import ./Actual.bend as A
import ../color_counts/Emission.bend as Emission
import ../color_counts/Actual.bend as Sources
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

def children_each(+b: Chess.Board,+w: Nat,+bl: Nat,xs: List<&2,Chess.Ply>,
  +zero: {Chess.get_rights(b) == 0 : U32},+pw: {U32.is_le(U64.popcount(U64.and(Chess.get_pawns(b),Chess.get_white(b))),8) == True{} : Bool},
  +pb: {U32.is_le(U64.popcount(U64.and(Chess.get_pawns(b),Chess.get_black(b))),8) == True{} : Bool},
  +cw: {Nat.is_le(A.white(b),w) == True{} : Bool},+cb: {Nat.is_le(A.black(b),bl) == True{} : Bool},
  good: C.every(m => Sources.emitted(b,m),xs)) -> C.every(m => S.child_state(w,bl,Chess.make_move(b,m)),xs):
  match xs:
    case Nil{}: Unit{}
    case Con{+m,t}:
      (head,tail)=good
      (S.changed(b,m,w,bl,(zero,(pw,pb)),(cw,cb),A.post(b,m,head)),children_each(b,w,bl,t,zero,pw,pb,cw,cb,tail))
def children(+b: Chess.Board,+w: Nat,+bl: Nat,xs: List<&2,Chess.Ply>,parent: S.board_good(b),prior: S.board_within(w,bl,b),
  good: C.every(m => Sources.emitted(b,m),xs)) -> C.every(m => S.child_state(w,bl,Chess.make_move(b,m)),xs):
  (zero,bounds)=parent
  (pw,pb)=bounds
  (cw,cb)=prior
  children_each(b,w,bl,xs,zero,pw,pb,cw,cb,good)
def moved(+g: Pos.Game,+w: Nat,+bl: Nat,maybe: Maybe<&2,Chess.Ply>,
  good: Selected.selected(m => S.child_state(w,bl,Chess.make_move(Pos.board(g),m)),maybe)) -> S.state(w,bl,Protocol.move_apply(g,maybe)):
  match maybe:
    case None{}: (Unit{},Unit{})
    case Some{+m}:
      (child,unit)=good
      %Equal.sym(Chess.Board,Pos.board(Pos.apply(g,m)),Chess.make_move(Pos.board(g),m),Prior.board(g,m)) : S.child_state(w,bl,_)
      child
def found(+c: O.Cells,+g: Pos.Game,+text: String,+w: Nat,+bl: Nat,r: Array<U64> & List<&2,Chess.Ply>,
  kept: {T.table(List<&2,Chess.Ply>,r) == O.pack(c) : Array<U64>},parent: S.board_good(Pos.board(g)),before: S.board_within(w,bl,Pos.board(g)),
  emitted: C.every(m => Sources.emitted(Pos.board(g),m),E.moves(r))) -> S.root(c,w,bl,Protocol.move_found(g,text,r)):
  (table,+xs)=r
  (kept,moved(g,w,bl,Pos.find_move(xs,text),Find.all(m => S.child_state(w,bl,Chess.make_move(Pos.board(g),m)),xs,text,
    children(Pos.board(g),w,bl,xs,parent,before,emitted))))
def legal(+c: O.Cells,+g: Pos.Game,+text: String,+w: Nat,+bl: Nat,parent: S.board_good(Pos.board(g)),before: S.board_within(w,bl,Pos.board(g))) ->
  S.root(c,w,bl,Protocol.move_found(g,text,Chess.legal_moves(O.pack(c),Pos.board(g)))):
  (+zero,bounds)=parent
  Emission.legal(r => {T.table(List<&2,Chess.Ply>,r) == O.pack(c) : Array<U64>} -> S.root(c,w,bl,Protocol.move_found(g,text,r)),O.pack(c),Pos.board(g),zero,
    r => emitted => kept => found(c,g,text,w,bl,r,kept,(zero,bounds),before,emitted))(Tables.legal(O.pack(c),Pos.board(g)))
def one(+c: O.Cells,+maybe: Maybe<&2,Pos.Game>,+text: String,+w: Nat,+bl: Nat,parent: S.state(w,bl,maybe)) ->
  S.root(c,w,bl,Protocol.apply_one(text,maybe,O.pack(c))):
  match maybe:
    case None{}: ({==},(Unit{},Unit{}))
    case Some{+g}:
      (good,before)=parent
      legal(c,g,text,w,bl,good,before)
