import Base
import ../../../legal_probe/Chess.bend as Chess
import ./Spec.bend as S
import ./Lists.bend as L
import ./Scan.bend as Scan
import ./Both.bend as Both
import ../castle_emission/Spec.bend as E
import ../castle_emission/Guard.bend as Guard
import ../board/Spec.bend as Board
import ../u64/Words.bend as W

def then(table: Array<U64>,+b: Chess.Board,
  -Q: @-r: (Array<U64> & List<&2,Chess.Ply>) -> Type,
  next: @r: (Array<U64> & List<&2,Chess.Ply>) -> S.all(m => S.classified(b,m),E.moves(r)) -> Q(r)) -> Q(Chess.legal_moves(table,b)):
  +own = Chess.color(b,Chess.get_turn(b))
  Scan.loop(Chess.bit_squares(64n,U64.is_zero(own),own,Nil{}),b,(table,Nil{}),
    out => Q(S.filtered(b,out)),Unit{},out => p => Both.classified_then(b,out,Q,p,next))

def classified(table: Array<U64>,+b: Chess.Board) -> S.all(m => S.classified(b,m),E.moves(Chess.legal_moves(table,b))):
  then(table,b,out => S.all(m => S.classified(b,m),E.moves(out)),out => ps => ps)

def extract(+b: Chess.Board,+m: Chess.Ply,is_castle: {S.flag(m) == 2 : U32},p: S.classified(b,m)) -> S.certified(b,m):
  match p:
    case Inl{no}:
      W.false_true(S.certified(b,m),Equal.trans(Bool,False{},U32.is_eq(S.flag(m),2),True{},
        Equal.sym(Bool,U32.is_eq(S.flag(m),2),False{},no),
        Equal.cong(U32,Bool,f => U32.is_eq(f,2),S.flag(m),2,is_castle)))
    case Inr{yes}: yes

def observe(+b: Chess.Board,+m: Chess.Ply,is_castle: {S.flag(m) == 2 : U32},
  r: Array<U64> & List<&2,Chess.Ply>,ps: S.all(x => S.classified(b,x),E.moves(r))) -> E.member(m,E.moves(r)) -> S.certified(b,m):
  (table,xs) = r
  here => extract(b,m,is_castle,L.member(xs,x => S.classified(b,x),m,ps,here))

def castle(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,
  here: E.member(m,E.moves(Chess.legal_moves(table,b))),is_castle: {S.flag(m) == 2 : U32}) -> S.certified(b,m):
  then(table,b,out => E.member(m,E.moves(out)) -> S.certified(b,m),out => ps => observe(b,m,is_castle,out,ps))(here)

def preserve(+b: Chess.Board,+m: Chess.Ply,good: {Board.valid(b) == True{} : Bool},p: S.certified(b,m)) -> {Board.valid(Chess.make_move(b,m)) == True{} : Bool}:
  match p:
    case Inl{king}: Guard.consume(b,True{},m,good,king)
    case Inr{queen}: Guard.consume(b,False{},m,good,queen)

def legal_preserves(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,
  good: {Board.valid(b) == True{} : Bool},here: E.member(m,E.moves(Chess.legal_moves(table,b))),is_castle: {S.flag(m) == 2 : U32}) -> {Board.valid(Chess.make_move(b,m)) == True{} : Bool}:
  preserve(b,m,good,castle(table,b,m,here,is_castle))
