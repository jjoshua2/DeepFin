# Row partition follows the selected full Ply through the actual apply consumer.
import Base
import ./Spec.bend as S
import ./Find.bend as Find
import ../fen_validated_full/Root.bend as Root
import ../board/Spec.bend as B
import ../generated_child_validity/Post.bend as Post
import ../candidate_bypass_classification/Generated.bend as G
import ../candidate_bypass_classification/Fields.bend as Fields
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../storage/Observe.bend as O
import ../table_preservation/Core.bend as T
import ../table_preservation/Generation.bend as Tables
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

def board(+g: Pos.Game,+m: Chess.Ply) ->
  {Pos.board(Pos.apply(g,m)) == Chess.make_move(Pos.board(g),m) : Chess.Board}:
  Pos.Game{+b,half,full,history}=g
  Chess.Ply{src,dst,promotion,flag}=m
  {==}

def moved(+g: Pos.Game,maybe: Maybe<&2,Chess.Ply>,
  good: S.selected(m => Post.valid(Pos.board(g),m),maybe)) ->
  Root.rows(Protocol.move_apply(g,maybe)):
  match maybe:
    case None{}: Unit{}
    case Some{+m}:
      (child,unit)=good
      %Equal.sym(Chess.Board,Pos.board(Pos.apply(g,m)),Chess.make_move(Pos.board(g),m),board(g,m)) :
        {B.valid(_) == True{} : Bool}
      child

def found(+c: O.Cells,+g: Pos.Game,+text: String,r: Array<U64> & List<&2,Chess.Ply>,
  kept: {T.table(List<&2,Chess.Ply>,r) == O.pack(c) : Array<U64>},
  +good: {B.valid(Pos.board(g)) == True{} : Bool},
  fields: C.every(m => Fields.generated(Pos.board(g),m),E.moves(r))) ->
  S.root(c,Protocol.move_found(g,text,r)):
  (table,+xs)=r
  (kept,moved(g,Pos.find_move(xs,text),Find.all(m => Post.valid(Pos.board(g),m),xs,text,
    Post.occurrences(Pos.board(g),xs,good,fields))))

# The field and table certificates are about the same actual legal_moves pair.
def legal(+c: O.Cells,+g: Pos.Game,+text: String,
  +good: {B.valid(Pos.board(g)) == True{} : Bool}) ->
  S.root(c,Protocol.move_found(g,text,Chess.legal_moves(O.pack(c),Pos.board(g)))):
  G.legal(r => {T.table(List<&2,Chess.Ply>,r) == O.pack(c) : Array<U64>} ->
      S.root(c,Protocol.move_found(g,text,r)),O.pack(c),Pos.board(g),
    r => fields => kept => found(c,g,text,r,kept,good,fields))(Tables.legal(O.pack(c),Pos.board(g)))

def one(+c: O.Cells,+maybe: Maybe<&2,Pos.Game>,+text: String,
  good: Root.rows(maybe)) -> S.root(c,Protocol.apply_one(text,maybe,O.pack(c))):
  match maybe:
    case None{}: ({==},Unit{})
    case Some{+g}: legal(c,g,text,good)

def result(+c: O.Cells,+text: String,r: Array<U64> & Maybe<&2,Pos.Game>,
  checked: S.root(c,r)) -> S.root(c,Protocol.moves(Con{text,Nil{}},r)):
  (a,+maybe)=r
  (same,good)=checked
  %Equal.sym(Array<U64>,a,O.pack(c),same) : S.root(c,Protocol.apply_one(text,maybe,_))
  one(c,maybe,text,good)
