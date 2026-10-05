# The actual affine table is threaded by dependent continuations.
import Base
import ./Fields.bend as Fields
import ./Emission.bend as Emission
import ../castle_chain/Spec.bend as Chain
import ../castle_chain/Lists.bend as Lists
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as Chess

def after(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +src: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>,r: Array<U64> & U64,
  good: Chain.every(m => Fields.scanned(m),tail),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> Chain.every(m => Fields.scanned(m),E.moves(r)) -> Goal(r)) ->
  Goal(Chess.scan_after(src,pawn,ep,tail,r)):
  (table,+targets) = r
  next((table,Chess.destinations(64n,U64.is_zero(targets),targets,src,pawn,ep,tail)),
    Emission.destinations(64n,U64.is_zero(targets),targets,src,pawn,ep,tail,good))

def step(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +b: Chess.Board,+src: U32,r: Array<U64> & List<&2,Chess.Ply>,
  good: Chain.every(m => Fields.scanned(m),E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> Chain.every(m => Fields.scanned(m),E.moves(r)) -> Goal(r)) ->
  Goal(Chess.scan_step(b,src,r)):
  (table,tail) = r
  +pawn = U64.test_bit(Chess.get_pawns(b),U32.to_nat(src))
  after(Goal,src,pawn,Chess.get_ep(b),tail,Chess.piece_targets(pawn,table,b,src),good,next)

def scan(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  keys: List<&2,U32>,+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,
  good: Chain.every(m => Fields.scanned(m),E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> Chain.every(m => Fields.scanned(m),E.moves(r)) -> Goal(r)) ->
  Goal(Chess.scan(keys,b,r)):
  match keys:
    case Nil{}: next(r,good)
    case Con{src,+rest}:
      step(z => Goal(Chess.scan(rest,b,z)),b,src,r,good,
        z => pz => scan(Goal,rest,b,z,pz,next))

def initial(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,table: Array<U64>,+b: Chess.Board,
  next: @r: Array<U64> & List<&2,Chess.Ply> -> Chain.every(m => Fields.scanned(m),E.moves(r)) -> Goal(r)) ->
  Goal(Chain.candidates(table,b)):
  +own = Chess.color(b,Chess.get_turn(b))
  scan(Goal,Chess.bit_squares(64n,U64.is_zero(own),own,Nil{}),b,(table,Nil{}),Unit{},next)

def candidates(table: Array<U64>,+b: Chess.Board) ->
  Chain.every(m => Fields.scanned(m),E.moves(Chain.candidates(table,b))):
  initial(r => Chain.every(m => Fields.scanned(m),E.moves(r)),table,b,r => good => good)

def member(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,
  here: E.member(m,E.moves(Chain.candidates(table,b)))) -> Fields.scanned(m):
  initial(r => E.member(m,E.moves(r)) -> Fields.scanned(m),table,b,
    r => good => Lists.extract(m => Fields.scanned(m),r,m,good))(here)
