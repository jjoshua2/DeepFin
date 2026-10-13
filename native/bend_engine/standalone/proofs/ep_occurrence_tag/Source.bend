# Preserve the predicate of every incoming full-Ply node, including duplicates.
import Base
import ./Spec.bend as S
import ../bitboard_ctz_structural/WordCtzCore.bend as Ctz
import ../castle_chain/Spec.bend as Chain
import ../castle_chain/Chain.bend as Pipeline
import ../castle_chain/Lists.bend as Lists
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as C

def put(promote: Bool,+b: C.Board,+s: U32,+d: U32,+p: Bool,+ep: U32,+f: Bool,+tail: List<&2,C.Ply>,
  bound: {Nat.is_lt(U32.to_nat(d),64n) == True{} : Bool},
  same_pawn: {p == S.pawn(b,s) : Bool},same_ep: {ep == C.get_ep(b) : U32},
  fields: S.bits(f,p,U32.is_eq(d,ep)),good: Chain.every(m => S.property(b,m),tail)) ->
  Chain.every(m => S.property(b,m),C.put_move(promote,s,d,Bool.to_u32(f),tail)):
  match promote:
    case False{}: (S.selected(f,b,s,d,p,ep,bound,same_pawn,same_ep,fields),good)
    case True{}: (Unit{},(Unit{},(Unit{},(Unit{},good))))

def destinations(+b: C.Board,n: Nat,empty: Bool,+bb: U64,+s: U32,+p: Bool,+ep: U32,+tail: List<&2,C.Ply>,
  same: {empty == U64.is_zero(bb) : Bool},
  +same_pawn: {p == S.pawn(b,s) : Bool},+same_ep: {ep == C.get_ep(b) : U32},
  good: Chain.every(m => S.property(b,m),tail)) ->
  Chain.every(m => S.property(b,m),C.destinations(n,empty,bb,s,p,ep,tail)):
  match n:
    case 0n: good
    case 1n+fuel:
      match empty:
        case True{}: good
        case False{}:
          +d=U64.ctz(bb)
          +rank=U32.div(d,8)
          +rest=U64.clear_lsb(bb)
          +promote=Bool.and(p,Bool.or(U32.is_zero(rank),U32.is_eq(rank,7)))
          +f=Bool.and(p,U32.is_eq(d,ep))
          destinations(b,fuel,U64.is_zero(rest),rest,s,p,ep,C.put_move(promote,s,d,Bool.to_u32(f),tail),
            {==},same_pawn,same_ep,
            put(promote,b,s,d,p,ep,f,tail,Ctz.actual_ctz_range(bb,Equal.sym(Bool,False{},U64.is_zero(bb),same)),
              same_pawn,same_ep,S.correlate(p,U32.is_eq(d,ep)),good))

def after(-Goal: @-r: Array<U64> & List<&2,C.Ply> -> Type,
  +b: C.Board,+s: U32,+p: Bool,+ep: U32,+tail: List<&2,C.Ply>,r: Array<U64> & U64,
  same_pawn: {p == S.pawn(b,s) : Bool},same_ep: {ep == C.get_ep(b) : U32},
  good: Chain.every(m => S.property(b,m),tail),
  next: @r: Array<U64> & List<&2,C.Ply> -> Chain.every(m => S.property(b,m),E.moves(r)) -> Goal(r)) ->
  Goal(C.scan_after(s,p,ep,tail,r)):
  (table,+targets)=r
  next((table,C.destinations(64n,U64.is_zero(targets),targets,s,p,ep,tail)),
    destinations(b,64n,U64.is_zero(targets),targets,s,p,ep,tail,{==},same_pawn,same_ep,good))

def step(-Goal: @-r: Array<U64> & List<&2,C.Ply> -> Type,
  +b: C.Board,+s: U32,r: Array<U64> & List<&2,C.Ply>,good: Chain.every(m => S.property(b,m),E.moves(r)),
  next: @r: Array<U64> & List<&2,C.Ply> -> Chain.every(m => S.property(b,m),E.moves(r)) -> Goal(r)) -> Goal(C.scan_step(b,s,r)):
  (table,tail)=r
  +p=S.pawn(b,s)
  after(Goal,b,s,p,C.get_ep(b),tail,C.piece_targets(p,table,b,s),{==},{==},good,next)

def scan(-Goal: @-r: Array<U64> & List<&2,C.Ply> -> Type,
  keys: List<&2,U32>,+b: C.Board,r: Array<U64> & List<&2,C.Ply>,good: Chain.every(m => S.property(b,m),E.moves(r)),
  next: @r: Array<U64> & List<&2,C.Ply> -> Chain.every(m => S.property(b,m),E.moves(r)) -> Goal(r)) -> Goal(C.scan(keys,b,r)):
  match keys:
    case Nil{}: next(r,good)
    case Con{+s,+rest}:
      step(z => Goal(C.scan(rest,b,z)),b,s,r,good,z => cert => scan(Goal,rest,b,z,cert,next))

def initial(-Goal: @-r: Array<U64> & List<&2,C.Ply> -> Type,table: Array<U64>,+b: C.Board,
  next: @r: Array<U64> & List<&2,C.Ply> -> Chain.every(m => S.property(b,m),E.moves(r)) -> Goal(r)) -> Goal(Chain.candidates(table,b)):
  +own=C.color(b,C.get_turn(b))
  scan(Goal,C.bit_squares(64n,U64.is_zero(own),own,Nil{}),b,(table,Nil{}),Unit{},next)

def castle(+b: C.Board,ks: Bool) -> S.property(b,E.move(b,ks)):
  Unit{}

def suffix(-Goal: @-r: Array<U64> & List<&2,C.Ply> -> Type,
  +b: C.Board,r: Array<U64> & List<&2,C.Ply>,good: Chain.every(m => S.property(b,m),E.moves(r)),
  next: @r: Array<U64> & List<&2,C.Ply> -> Chain.every(m => S.property(b,m),E.moves(r)) -> Goal(r)) -> Goal(Chain.suffix(b,r)):
  Pipeline.suffix(m => S.property(b,m),Goal,b,r,good,
    guard => castle(b,True{}),guard => castle(b,False{}),next)

def legal(-Goal: @-r: Array<U64> & List<&2,C.Ply> -> Type,table: Array<U64>,+b: C.Board,
  next: @r: Array<U64> & List<&2,C.Ply> -> Chain.every(m => S.property(b,m),E.moves(r)) -> Goal(r)) -> Goal(C.legal_moves(table,b)):
  initial(z => Goal(Chain.suffix(b,z)),table,b,z => good => suffix(Goal,b,z,good,next))

def member(table: Array<U64>,+b: C.Board,+m: C.Ply,here: E.member(m,E.moves(C.legal_moves(table,b)))) -> S.property(b,m):
  legal(r => E.member(m,E.moves(r)) -> S.property(b,m),table,b,r => good => Lists.extract(x => S.property(b,x),r,m,good))(here)
