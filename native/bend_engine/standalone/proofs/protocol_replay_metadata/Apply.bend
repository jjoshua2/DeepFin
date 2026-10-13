import Base
import ./Spec.bend as S
import ../protocol_replay_rows/Find.bend as Find
import ../protocol_replay_rows/Spec.bend as Selected
import ../protocol_replay_rows/Apply.bend as Prior
import ../fen_validated_full/Turn.bend as Turn
import ../child_metadata/Source.bend as Source
import ../child_metadata/Post.bend as Post
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../board/Spec.bend as B
import ../storage/Observe.bend as O
import ../table_preservation/Core.bend as T
import ../table_preservation/Generation.bend as Tables
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

def turn_certificate(+t: U32,+white: Bool,same: {t == Bool.to_u32(white) : U32}) -> Turn.canonical(t):
  %Equal.sym(U32,t,Bool.to_u32(white),same) : Turn.canonical(_)
  Turn.boolean(white)
def converted(+b: Chess.Board,+white: Bool,good: Post.ready(b,white)) -> S.ready(b):
  (rows,(turn,bound))=good
  (rows,(turn_certificate(Chess.get_turn(b),white,turn),bound))
def moved(+g: Pos.Game,+white: Bool,maybe: Maybe<&2,Chess.Ply>,
  good: Selected.selected(m => Post.child(Pos.board(g),m,white),maybe)) ->
  S.output(Protocol.move_apply(g,maybe)):
  match maybe:
    case None{}: Unit{}
    case Some{+m}:
      (child,unit)=good
      %Equal.sym(Chess.Board,Pos.board(Pos.apply(g,m)),Chess.make_move(Pos.board(g),m),Prior.board(g,m)) : S.ready(_)
      converted(Chess.make_move(Pos.board(g),m),Bool.not(white),child)
def found(+c: O.Cells,+g: Pos.Game,+white: Bool,+text: String,
  r: Array<U64> & List<&2,Chess.Ply>,
  kept: {T.table(List<&2,Chess.Ply>,r) == O.pack(c) : Array<U64>},
  +rows: {B.valid(Pos.board(g)) == True{} : Bool},
  +canonical: {Chess.get_turn(Pos.board(g)) == Bool.to_u32(white) : U32},
  fields: C.every(m => Source.generated(Pos.board(g),m),E.moves(r))) ->
  S.root(c,Protocol.move_found(g,text,r)):
  (table,+xs)=r
  (kept,moved(g,white,Pos.find_move(xs,text),Find.all(m => Post.child(Pos.board(g),m,white),xs,text,
    Post.occurrences(Pos.board(g),xs,white,rows,canonical,fields))))
def legal(+c: O.Cells,+g: Pos.Game,+text: String,
  +rows: {B.valid(Pos.board(g)) == True{} : Bool},
  canonical: Turn.canonical(Chess.get_turn(Pos.board(g)))) ->
  S.root(c,Protocol.move_found(g,text,Chess.legal_moves(O.pack(c),Pos.board(g)))):
  Source.legal(r => {T.table(List<&2,Chess.Ply>,r) == O.pack(c) : Array<U64>} ->
      S.root(c,Protocol.move_found(g,text,r)),O.pack(c),Pos.board(g),
    r => fields => kept => found(c,g,U32.is_eq(Chess.get_turn(Pos.board(g)),1),text,r,kept,rows,canonical,fields))(Tables.legal(O.pack(c),Pos.board(g)))
def one(+c: O.Cells,+maybe: Maybe<&2,Pos.Game>,+text: String,
  good: S.parent(maybe)) -> S.root(c,Protocol.apply_one(text,maybe,O.pack(c))):
  match maybe:
    case None{}: ({==},Unit{})
    case Some{+g}:
      (rows,canonical)=good
      legal(c,g,text,rows,canonical)
