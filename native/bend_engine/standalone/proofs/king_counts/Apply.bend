import Base
import ./Spec.bend as S
import ./Actual.bend as K
import ./Emission.bend as Emission
import ../protocol_replay_ep_coherence/Apply.bend as EPApply
import ../protocol_replay_rows/Apply.bend as Prior
import ../protocol_replay_rows/Find.bend as Find
import ../protocol_replay_rows/Spec.bend as Selected
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../board/Spec.bend as Board
import ../storage/Observe.bend as O
import ../table_preservation/Core.bend as T
import ../table_preservation/Generation.bend as Tables
import ../initialized_attacks/Build.bend as Init
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

def children_each(+b: Chess.Board,+w: Nat,+bl: Nat,xs: List<&2,Chess.Ply>,+rows: {Board.valid(b) == True{} : Bool},
  +zero: {Chess.get_rights(b) == 0 : U32},+cw: {K.white(b) == w : Nat},+cb: {K.black(b) == bl : Nat},
  good: C.every(m => K.emitted(b,m),xs)) -> C.every(m => S.child_counts(w,bl,Chess.make_move(b,m)),xs):
  match xs:
    case Nil{}: Unit{}
    case Con{+m,t}:
      (head,tail)=good
      (S.changed(b,m,w,bl,zero,(cw,cb),K.post(b,m,rows,head)),children_each(b,w,bl,t,rows,zero,cw,cb,tail))
def children(+b: Chess.Board,+w: Nat,+bl: Nat,xs: List<&2,Chess.Ply>,rows: {Board.valid(b) == True{} : Bool},
  zero: {Chess.get_rights(b) == 0 : U32},before: S.board_counts(w,bl,b),good: C.every(m => K.emitted(b,m),xs)) ->
  C.every(m => S.child_counts(w,bl,Chess.make_move(b,m)),xs):
  (cw,cb)=before
  children_each(b,w,bl,xs,rows,zero,cw,cb,good)
def moved(+g: Pos.Game,+w: Nat,+bl: Nat,maybe: Maybe<&2,Chess.Ply>,
  good: Selected.selected(m => S.child_counts(w,bl,Chess.make_move(Pos.board(g),m)),maybe)) -> S.naked(w,bl,Protocol.move_apply(g,maybe)):
  match maybe:
    case None{}: (Unit{},Unit{})
    case Some{+m}:
      (child,unit)=good
      %Equal.sym(Chess.Board,Pos.board(Pos.apply(g,m)),Chess.make_move(Pos.board(g),m),Prior.board(g,m)) : S.child_counts(w,bl,_)
      child
def found(+c: O.Cells,+g: Pos.Game,+text: String,+w: Nat,+bl: Nat,r: Array<U64> & List<&2,Chess.Ply>,
  kept: {T.table(List<&2,Chess.Ply>,r) == O.pack(c) : Array<U64>},rows: {Board.valid(Pos.board(g)) == True{} : Bool},
  zero: {Chess.get_rights(Pos.board(g)) == 0 : U32},before: S.board_counts(w,bl,Pos.board(g)),good: C.every(m => K.emitted(Pos.board(g),m),E.moves(r))) ->
  S.naked_root(c,w,bl,Protocol.move_found(g,text,r)):
  (table,+xs)=r
  (kept,moved(g,w,bl,Pos.find_move(xs,text),Find.all(m => S.child_counts(w,bl,Chess.make_move(Pos.board(g),m)),xs,text,children(Pos.board(g),w,bl,xs,rows,zero,before,good))))
def legal(+c: O.Cells,+g: Pos.Game,+text: String,+w: Nat,+bl: Nat,+rows: {Board.valid(Pos.board(g)) == True{} : Bool},
  ep: {Pos.valid_ep(Chess.get_ep(Pos.board(g)),Pos.board(g)) == True{} : Bool},+zero: {Chess.get_rights(Pos.board(g)) == 0 : U32},before: S.board_counts(w,bl,Pos.board(g))) ->
  S.naked_root(c,w,bl,Protocol.move_found(g,text,Chess.legal_moves(O.pack(c),Pos.board(g)))):
  Emission.legal(r => {T.table(List<&2,Chess.Ply>,r) == O.pack(c) : Array<U64>} -> S.naked_root(c,w,bl,Protocol.move_found(g,text,r)),
    O.pack(c),Pos.board(g),rows,ep,zero,r => good => kept => found(c,g,text,w,bl,r,kept,rows,zero,before,good))(Tables.legal(O.pack(c),Pos.board(g)))
def one(+c: O.Cells,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+w: Nat,+bl: Nat,+maybe: Maybe<&2,Pos.Game>,+text: String,
  built: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>},depth: {d == 17n : Nat},full: {extra == 64n : Nat},parent: S.state(w,bl,maybe)) ->
  S.root(c,w,bl,Protocol.apply_one(text,maybe,O.pack(c))):
  match maybe:
    case None{}: ({==},(Unit{},(Unit{},(Unit{},Unit{}))))
    case Some{+g}:
      ((+rows,canonical),(coherent,(zero,before)))=parent
      S.joined(c,w,bl,Protocol.move_found(g,text,Chess.legal_moves(O.pack(c),Pos.board(g))),
        legal(c,g,text,w,bl,rows,coherent,zero,before),EPApply.legal(c,d,seed,n,extra,g,text,built,depth,full,rows,canonical))
