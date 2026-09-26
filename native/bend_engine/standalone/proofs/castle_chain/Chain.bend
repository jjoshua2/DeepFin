# Compose actual ordinary scan, both side producers, and full/fast final filtering.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_emission/Spec.bend as E
import ../castle_emission/Guard.bend as Guard
import ../board/Spec.bend as B
import ./Spec.bend as S
import ./Lists.bend as L
import ./Scan.bend as Scan
import ./Sides.bend as Sides
import ./Filter.bend as Filter

def suffix(-P: Chess.Ply -> Type,-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,good: S.every(P,E.moves(r)),
  king: {E.guard(b,True{}) == True{} : Bool} -> P(E.move(b,True{})),
  queen: {E.guard(b,False{}) == True{} : Bool} -> P(E.move(b,False{})),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(P,E.moves(r)) -> Goal(r)) ->
  Goal(S.suffix(b,r)):
  Sides.side(P,z => Goal(Chess.filter_prepare(b,Chess.castle_side(b,False{},z))),b,True{},r,good,king,
    z => pz => Sides.side(P,z => Goal(Chess.filter_prepare(b,z)),b,False{},z,pz,queen,
      z => pz => Filter.prepare(P,Goal,b,z,pz,next)))

def from_scan(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,good: S.every(S.ordinary,E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(m => S.classified(b,m),E.moves(r)) -> Goal(r)) ->
  Goal(S.suffix(b,r)):
  (table,+xs) = r
  suffix(m => S.classified(b,m),Goal,b,(table,xs),L.lift(b,xs,good),
    g => Inr{Inl{(g,{==})}},g => Inr{Inr{(g,{==})}},next)

def legal(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,table: Array<U64>,+b: Chess.Board,
  next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(m => S.classified(b,m),E.moves(r)) -> Goal(r)) ->
  Goal(Chess.legal_moves(table,b)):
  Scan.initial(z => Goal(S.suffix(b,z)),table,b,z => pz => from_scan(Goal,b,z,pz,next))

def scan_plain(keys: List<&2,U32>,+b: Chess.Board,table: Array<U64>) ->
  S.every(S.ordinary,E.moves(Chess.scan(keys,b,(table,Nil{})))):
  Scan.scan(z => S.every(S.ordinary,E.moves(z)),keys,b,(table,Nil{}),Unit{},z => pz => pz)

def filter_member(+b: Chess.Board,table: Array<U64>,+xs: List<&2,Chess.Ply>,+m: Chess.Ply,
  here: E.member(m,E.moves(Chess.filter_prepare(b,(table,xs))))) -> E.member(m,xs):
  Filter.prepare(x => E.member(x,xs),z => E.member(m,E.moves(z)) -> E.member(m,xs),b,(table,xs),L.self(xs),
    z => pz => L.extract(x => E.member(x,xs),z,m,pz))(here)

def suffix_member(+b: Chess.Board,table: Array<U64>,+xs: List<&2,Chess.Ply>,+m: Chess.Ply,
  here: E.member(m,E.moves(S.suffix(b,(table,xs))))) -> E.member(m,xs) | S.castle(b,m):
  suffix(x => E.member(x,xs) | S.castle(b,x),
    z => E.member(m,E.moves(z)) -> (E.member(m,xs) | S.castle(b,m)),b,(table,xs),L.old_tail(b,xs,xs,L.self(xs)),
    g => Inr{Inl{(g,{==})}},g => Inr{Inr{(g,{==})}},
    z => pz => L.extract(x => E.member(x,xs) | S.castle(b,x),z,m,pz))(here)

def legal_member(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,
  here: E.member(m,E.moves(Chess.legal_moves(table,b))),tag: {S.flag(m) == 2 : U32}) -> S.castle(b,m):
  L.castling(b,m,tag,
    legal(z => E.member(m,E.moves(z)) -> S.classified(b,m),table,b,
      z => pz => L.extract(x => S.classified(b,x),z,m,pz))(here))

def preserve_certificate(+b: Chess.Board,+m: Chess.Ply,good: {B.valid(b) == True{} : Bool},p: S.castle(b,m)) ->
  {B.valid(Chess.make_move(b,m)) == True{} : Bool}:
  match p:
    case Inl{k}: Guard.consume(b,True{},m,good,k)
    case Inr{q}: Guard.consume(b,False{},m,good,q)

def legal_preserves(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,
  good: {B.valid(b) == True{} : Bool},here: E.member(m,E.moves(Chess.legal_moves(table,b))),tag: {S.flag(m) == 2 : U32}) ->
  {B.valid(Chess.make_move(b,m)) == True{} : Bool}:
  preserve_certificate(b,m,good,legal_member(table,b,m,here,tag))
