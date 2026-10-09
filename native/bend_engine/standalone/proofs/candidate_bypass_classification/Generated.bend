# Preserve exact field alternatives through the actual guarded producers/filters.
import Base
import ./Fields.bend as Fields
import ./Scan.bend as Scan
import ../castle_chain/Spec.bend as Chain
import ../castle_chain/Chain.bend as Old
import ../castle_chain/Sides.bend as Sides
import ../castle_chain/Lists.bend as Lists
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as Chess

def sides(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>) -> Array<U64> & List<&2,Chess.Ply>:
  Chess.castle_side(b,False{},Chess.castle_side(b,True{},r))

def produced(table: Array<U64>,+b: Chess.Board) -> Array<U64> & List<&2,Chess.Ply>:
  sides(b,Chain.candidates(table,b))

def lift(-b: Chess.Board,xs: List<&2,Chess.Ply>,good: Chain.every(m => Fields.scanned(m),xs)) ->
  Chain.every(m => Fields.generated(b,m),xs):
  match xs:
    case Nil{}: Unit{}
    case Con{h,t}:
      (ph,pt) = good
      (Inl{ph},lift(b,t,pt))

def before(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,good: Chain.every(m => Fields.scanned(m),E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> Chain.every(m => Fields.generated(b,m),E.moves(r)) -> Goal(r)) ->
  Goal(sides(b,r)):
  (table,+xs) = r
  Sides.side(m => Fields.generated(b,m),z => Goal(Chess.castle_side(b,False{},z)),b,True{},(table,xs),lift(b,xs,good),
    g => Inr{Inl{(g,{==})}},
    z => pz => Sides.side(m => Fields.generated(b,m),Goal,b,False{},z,pz,g => Inr{Inr{(g,{==})}},next))

def after(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,good: Chain.every(m => Fields.scanned(m),E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> Chain.every(m => Fields.generated(b,m),E.moves(r)) -> Goal(r)) ->
  Goal(Chain.suffix(b,r)):
  (table,+xs) = r
  Old.suffix(m => Fields.generated(b,m),Goal,b,(table,xs),lift(b,xs,good),
    g => Inr{Inl{(g,{==})}},g => Inr{Inr{(g,{==})}},next)

def prefilter(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,table: Array<U64>,+b: Chess.Board,
  next: @r: Array<U64> & List<&2,Chess.Ply> -> Chain.every(m => Fields.generated(b,m),E.moves(r)) -> Goal(r)) ->
  Goal(produced(table,b)):
  Scan.initial(z => Goal(sides(b,z)),table,b,z => pz => before(Goal,b,z,pz,next))

def legal(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,table: Array<U64>,+b: Chess.Board,
  next: @r: Array<U64> & List<&2,Chess.Ply> -> Chain.every(m => Fields.generated(b,m),E.moves(r)) -> Goal(r)) ->
  Goal(Chess.legal_moves(table,b)):
  Scan.initial(z => Goal(Chain.suffix(b,z)),table,b,z => pz => after(Goal,b,z,pz,next))

def all_prefilter(table: Array<U64>,+b: Chess.Board) ->
  Chain.every(m => Fields.generated(b,m),E.moves(produced(table,b))):
  prefilter(r => Chain.every(m => Fields.generated(b,m),E.moves(r)),table,b,r => good => good)

def all_legal(table: Array<U64>,+b: Chess.Board) ->
  Chain.every(m => Fields.generated(b,m),E.moves(Chess.legal_moves(table,b))):
  legal(r => Chain.every(m => Fields.generated(b,m),E.moves(r)),table,b,r => good => good)

def member_prefilter(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,
  here: E.member(m,E.moves(produced(table,b)))) -> Fields.generated(b,m):
  prefilter(r => E.member(m,E.moves(r)) -> Fields.generated(b,m),table,b,
    r => good => Lists.extract(x => Fields.generated(b,x),r,m,good))(here)

def member_legal(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,
  here: E.member(m,E.moves(Chess.legal_moves(table,b)))) -> Fields.generated(b,m):
  legal(r => E.member(m,E.moves(r)) -> Fields.generated(b,m),table,b,
    r => good => Lists.extract(x => Fields.generated(b,x),r,m,good))(here)
