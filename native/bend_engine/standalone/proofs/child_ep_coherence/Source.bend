# One affine-table continuation carries source bounds and exact full-Ply fields.
import Base
import ../child_metadata/Range.bend as R
import ./Spec.bend as S
import ../bitboard_membership/AllEmittedSound.bend as All
import ../candidate_bypass_classification/Fields.bend as Fields
import ../promotion/Spec.bend as P
import ../castle_chain/Spec.bend as C
import ../castle_chain/Chain.bend as Chain
import ../castle_chain/Lists.bend as Lists
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as Chess

def scanned(-m: Chess.Ply) -> Type:
  R.bounded(m) & S.scanned(m)

def generated(-b: Chess.Board,-m: Chess.Ply) -> Type:
  R.bounded(m) & S.generated(b,m)

def put(promote: Bool,+s: U32,+d: U32,ep: Bool,+tail: List<&2,Chess.Ply>,
  +below: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},ranking: S.rank_gate(promote,d),
  good: C.every(m => scanned(m),tail)) -> C.every(m => scanned(m),Chess.put_move(promote,s,d,Bool.to_u32(ep),tail)):
  match promote:
    case False{}:
      match ep:
        case False{}: ((below,Inl{(s,(d,{==}))}),good)
        case True{}: ((below,Inr{Inl{(s,(d,{==}))}}),good)
    case True{}:
      +high: {S.backrank(d) == True{} : Bool}=ranking
      ((below,Inr{Inr{(high,(s,(d,(P.Knight{},{==}))))}}),
        ((below,Inr{Inr{(high,(s,(d,(P.Bishop{},{==}))))}}),
          ((below,Inr{Inr{(high,(s,(d,(P.Rook{},{==}))))}}),
            ((below,Inr{Inr{(high,(s,(d,(P.Queen{},{==}))))}}),good))))

def destinations(n: Nat,empty: Bool,+bb: U64,+s: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>,
  +below: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},good: C.every(m => scanned(m),tail)) ->
  C.every(m => scanned(m),Chess.destinations(n,empty,bb,s,pawn,ep,tail)):
  match n:
    case 0n: good
    case 1n+p:
      match empty:
        case True{}: good
        case False{}:
          +d=U64.ctz(bb)
          +rank=U32.div(d,8)
          +rest=U64.clear_lsb(bb)
          +promote=Bool.and(pawn,Bool.or(U32.is_zero(rank),U32.is_eq(rank,7)))
          +f=Bool.and(pawn,U32.is_eq(d,ep))
          destinations(p,U64.is_zero(rest),rest,s,pawn,ep,Chess.put_move(promote,s,d,Bool.to_u32(f),tail),
            below,put(promote,s,d,f,tail,below,S.correlate(pawn,d),good))

def after(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +s: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>,r: Array<U64> & U64,
  +below: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},good: C.every(m => scanned(m),tail),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => scanned(m),E.moves(r)) -> Goal(r)) ->
  Goal(Chess.scan_after(s,pawn,ep,tail,r)):
  (table,+targets)=r
  next((table,Chess.destinations(64n,U64.is_zero(targets),targets,s,pawn,ep,tail)),
    destinations(64n,U64.is_zero(targets),targets,s,pawn,ep,tail,below,good))

def step(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +b: Chess.Board,+s: U32,r: Array<U64> & List<&2,Chess.Ply>,
  below: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},good: C.every(m => scanned(m),E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => scanned(m),E.moves(r)) -> Goal(r)) ->
  Goal(Chess.scan_step(b,s,r)):
  (table,tail)=r
  +pawn=U64.test_bit(Chess.get_pawns(b),U32.to_nat(s))
  after(Goal,s,pawn,Chess.get_ep(b),tail,Chess.piece_targets(pawn,table,b,s),below,good,next)

def scan(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  keys: List<&2,U32>,+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,
  +below: {All.all_below64(keys) == True{} : Bool},good: C.every(m => scanned(m),E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => scanned(m),E.moves(r)) -> Goal(r)) ->
  Goal(Chess.scan(keys,b,r)):
  match keys:
    case Nil{}: next(r,good)
    case Con{+s,+rest}:
      +head=All.bool_and_true_left(Nat.is_lt(U32.to_nat(s),64n),All.all_below64(rest),below)
      +tail=All.bool_and_true_right(Nat.is_lt(U32.to_nat(s),64n),All.all_below64(rest),below)
      step(z => Goal(Chess.scan(rest,b,z)),b,s,r,head,good,
        z => pz => scan(Goal,rest,b,z,tail,pz,next))

def initial(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,table: Array<U64>,+b: Chess.Board,
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => scanned(m),E.moves(r)) -> Goal(r)) ->
  Goal(C.candidates(table,b)):
  +own=Chess.color(b,Chess.get_turn(b))
  scan(Goal,Chess.bit_squares(64n,U64.is_zero(own),own,Nil{}),b,(table,Nil{}),
    All.actual_bit_squares_below64(own),Unit{},next)

def home_bound(white: Bool) ->
  {Nat.is_lt(U32.to_nat(U32.add(Bool.pick(U32,white,0,56),4)),64n) == True{} : Bool}:
  match white:
    case False{}: {==}
    case True{}: {==}

def castle_bound(+b: Chess.Board,ks: Bool) -> R.bounded(E.move(b,ks)):
  home_bound(U32.is_eq(Chess.get_turn(b),1))

def lift(-b: Chess.Board,xs: List<&2,Chess.Ply>,good: C.every(m => scanned(m),xs)) ->
  C.every(m => generated(b,m),xs):
  match xs:
    case Nil{}: Unit{}
    case Con{h,t}:
      ((bound,fields),rest)=good
      ((bound,Inl{fields}),lift(b,t,rest))

def suffix(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,good: C.every(m => scanned(m),E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => generated(b,m),E.moves(r)) -> Goal(r)) ->
  Goal(C.suffix(b,r)):
  (table,+xs)=r
  Chain.suffix(m => generated(b,m),Goal,b,(table,xs),lift(b,xs,good),
    guard => (castle_bound(b,True{}),Inr{Inl{(guard,{==})}}),
    guard => (castle_bound(b,False{}),Inr{Inr{(guard,{==})}}),next)

def legal(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,table: Array<U64>,+b: Chess.Board,
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => generated(b,m),E.moves(r)) -> Goal(r)) ->
  Goal(Chess.legal_moves(table,b)):
  initial(z => Goal(C.suffix(b,z)),table,b,z => good => suffix(Goal,b,z,good,next))

def member(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,
  here: E.member(m,E.moves(Chess.legal_moves(table,b)))) -> generated(b,m):
  legal(r => E.member(m,E.moves(r)) -> generated(b,m),table,b,
    r => good => Lists.extract(x => generated(b,x),r,m,good))(here)
