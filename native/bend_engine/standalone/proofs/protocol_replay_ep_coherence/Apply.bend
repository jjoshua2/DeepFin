import Base
import ./Spec.bend as S
import ./Members.bend as Members
import ../protocol_replay_metadata/Spec.bend as Metadata
import ../protocol_replay_metadata/Apply.bend as MetadataApply
import ../protocol_replay_rows/Spec.bend as Selected
import ../protocol_replay_rows/Find.bend as Find
import ../protocol_replay_rows/Apply.bend as Prior
import ../child_ep_coherence/Coherence.bend as EP
import ../fen_validated_full/Root.bend as Root
import ../fen_validated_full/Turn.bend as Turn
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../board/Spec.bend as B
import ../storage/Observe.bend as O
import ../initialized_attacks/Build.bend as Init
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

def combine(+out: Maybe<&2,Pos.Game>,metadata: Metadata.output(out),coherent: S.ep_output(out)) -> S.output(out):
  match out:
    case None{}: Unit{}
    case Some{g}: (metadata,coherent)
def joined(+c: O.Cells,r: Array<U64> & Maybe<&2,Pos.Game>,
  checked: Metadata.root(c,r),coherent: S.ep_output(Root.value(r))) -> S.root(c,r):
  (kept,metadata)=checked
  (kept,combine(Root.value(r),metadata,coherent))

def moved(+g: Pos.Game,maybe: Maybe<&2,Chess.Ply>,
  good: Selected.selected(m => {EP.child(Pos.board(g),m) == True{} : Bool},maybe)) ->
  S.ep_output(Protocol.move_apply(g,maybe)):
  match maybe:
    case None{}: Unit{}
    case Some{+m}:
      (coherent,unit)=good
      %Equal.sym(Chess.Board,Pos.board(Pos.apply(g,m)),Chess.make_move(Pos.board(g),m),Prior.board(g,m)) :
        {EP.valid(_) == True{} : Bool}
      coherent
def found(+g: Pos.Game,+text: String,r: Array<U64> & List<&2,Chess.Ply>,
  fields: C.every(m => {EP.child(Pos.board(g),m) == True{} : Bool},E.moves(r))) ->
  S.ep_output(Root.value(Protocol.move_found(g,text,r))):
  (table,+xs)=r
  moved(g,Pos.find_move(xs,text),Find.all(m => {EP.child(Pos.board(g),m) == True{} : Bool},xs,text,fields))

def legal(+c: O.Cells,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+g: Pos.Game,+text: String,
  built: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},full: {extra == 64n : Nat},
  +rows: {B.valid(Pos.board(g)) == True{} : Bool},
  +canonical: {Chess.get_turn(Pos.board(g)) == Bool.to_u32(U32.is_eq(Chess.get_turn(Pos.board(g)),1)) : U32}) ->
  S.root(c,Protocol.move_found(g,text,Chess.legal_moves(O.pack(c),Pos.board(g)))):
  joined(c,Protocol.move_found(g,text,Chess.legal_moves(O.pack(c),Pos.board(g))),
    MetadataApply.legal(c,g,text,rows,canonical),
    found(g,text,Chess.legal_moves(O.pack(c),Pos.board(g)),
      Members.legal(c,d,seed,n,extra,Pos.board(g),U32.is_eq(Chess.get_turn(Pos.board(g)),1),built,depth,full,canonical)))
def one(+c: O.Cells,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+maybe: Maybe<&2,Pos.Game>,+text: String,
  built: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},full: {extra == 64n : Nat},good: Metadata.parent(maybe)) ->
  S.root(c,Protocol.apply_one(text,maybe,O.pack(c))):
  match maybe:
    case None{}: ({==},Unit{})
    case Some{+g}:
      (rows,canonical)=good
      legal(c,d,seed,n,extra,g,text,built,depth,full,rows,canonical)
