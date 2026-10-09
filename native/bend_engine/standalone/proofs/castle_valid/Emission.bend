# Import the checked ordinary scan and compose both actual castle producers and
# both actual final filtering paths. The predicate compares complete Plies.
import Base
import ../king_counts/Actual.bend as K
import ../king_counts/Emission.bend as Scan
import ../board/Spec.bend as Board
import ../castle_chain/Spec.bend as C
import ../castle_chain/Chain.bend as Chain
import ../castle_chain/Lists.bend as Lists
import ../castle_emission/Spec.bend as E
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

def certificate(-b: Chess.Board,-m: Chess.Ply) -> Type:
  K.emitted(b,m) | C.castle(b,m)
def lift(-b: Chess.Board,xs: List<&2,Chess.Ply>,good: C.every(m => K.emitted(b,m),xs)) ->
  C.every(m => certificate(b,m),xs):
  match xs:
    case Nil{}: Unit{}
    case Con{h,t}:
      (ph,pt)=good
      (Inl{ph},lift(b,t,pt))
def suffix(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,
  good: C.every(m => K.emitted(b,m),E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => certificate(b,m),E.moves(r)) -> Goal(r)) -> Goal(C.suffix(b,r)):
  (table,+xs)=r
  Chain.suffix(m => certificate(b,m),Goal,b,(table,xs),lift(b,xs,good),
    g => Inr{Inl{(g,{==})}},g => Inr{Inr{(g,{==})}},next)
def legal(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,table: Array<U64>,+b: Chess.Board,
  rows: {Board.valid(b) == True{} : Bool},coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => certificate(b,m),E.moves(r)) -> Goal(r)) -> Goal(Chess.legal_moves(table,b)):
  Scan.initial(z => Goal(C.suffix(b,z)),table,b,rows,coherent,z => good => suffix(Goal,b,z,good,next))
def member(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,rows: {Board.valid(b) == True{} : Bool},
  coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},here: E.member(m,E.moves(Chess.legal_moves(table,b)))) -> certificate(b,m):
  legal(r => E.member(m,E.moves(r)) -> certificate(b,m),table,b,rows,coherent,
    r => good => Lists.extract(x => certificate(b,x),r,m,good))(here)
