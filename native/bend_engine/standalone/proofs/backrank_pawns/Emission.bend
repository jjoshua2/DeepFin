import Base
import ./Mask.bend as M
import ./Rank.bend as Rank
import ../candidate_king_destination/Algebra.bend as A
import ../ray_clear_disjoint/Algebra.bend as D
import ../castle_chain/Spec.bend as S
import ../castle_chain/Chain.bend as Chain
import ../castle_chain/Lists.bend as Lists
import ../castle_emission/Spec.bend as E
import ../bitboard_ctz_structural/WordCtzCore.bend as Ctz
import ../../../legal_probe/Chess.bend as Chess

def selected(+target: U64,put: Bool,clear: D.disjoint(target,M.mask())) ->
  D.disjoint(Chess.select_u64(put,target,U64.zero()),M.mask()):
  match put:
    case True{}: clear
    case False{}: D.left_zero(M.mask())

# Keep both the actual source pawn Boolean and the actual rank decision.
def put(+b: Chess.Board,+src: U32,+dst: U32,+flag: U32,pawn: Bool,rank: Bool,
  source: {M.pawn(b,src) == pawn : Bool},rank_same: {M.backrank(dst) == rank : Bool},
  bound: {Nat.is_lt(U32.to_nat(dst),64n) == True{} : Bool},+tail: List<&2,Chess.Ply>,
  good: S.every(m => M.emitted(b,m),tail)) ->
  S.every(m => M.emitted(b,m),Chess.put_move(Bool.and(pawn,rank),src,dst,flag,tail)):
  match pawn rank:
    case False{} _: (M.nonpawn(b,src,dst,flag,source),good)
    case True{} False{}:
      (selected(U64.bit(U32.to_nat(dst)),U32.is_eq(Chess.piece(b,src),0),Rank.bounded(dst,bound,rank_same)),good)
    case True{} True{}:
      (D.left_zero(M.mask()),(D.left_zero(M.mask()),(D.left_zero(M.mask()),(D.left_zero(M.mask()),good))))

# Arbitrary actual masks, fuel and certified caller tails. Every physical cons is retained.
def destinations(+b: Chess.Board,n: Nat,+empty: Bool,+bb: U64,+src: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>,
  flag: {empty == U64.is_zero(bb) : Bool},+source: {M.pawn(b,src) == pawn : Bool},
  good: S.every(m => M.emitted(b,m),tail)) ->
  S.every(m => M.emitted(b,m),Chess.destinations(n,empty,bb,src,pawn,ep,tail)):
  match n:
    case 0n: good
    case 1n+p:
      match empty:
        case True{}: good
        case False{}:
          +dst=U64.ctz(bb)
          +rest=U64.clear_lsb(bb)
          +rank=M.backrank(dst)
          +tag=Bool.to_u32(Bool.and(pawn,U32.is_eq(dst,ep)))
          destinations(b,p,U64.is_zero(rest),rest,src,pawn,ep,
            Chess.put_move(Bool.and(pawn,rank),src,dst,tag,tail),{==},source,
            put(b,src,dst,tag,pawn,rank,source,{==},Ctz.actual_ctz_range(bb,Equal.sym(Bool,False{},U64.is_zero(bb),flag)),tail,good))
def after(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +b: Chess.Board,+src: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>,r: Array<U64> & U64,
  source: {M.pawn(b,src) == pawn : Bool},good: S.every(m => M.emitted(b,m),tail),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(m => M.emitted(b,m),E.moves(r)) -> Goal(r)) ->
  Goal(Chess.scan_after(src,pawn,ep,tail,r)):
  (table,+targets)=r
  next((table,Chess.destinations(64n,U64.is_zero(targets),targets,src,pawn,ep,tail)),
    destinations(b,64n,U64.is_zero(targets),targets,src,pawn,ep,tail,{==},source,good))
def step(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,+b: Chess.Board,+src: U32,r: Array<U64> & List<&2,Chess.Ply>,
  good: S.every(m => M.emitted(b,m),E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(m => M.emitted(b,m),E.moves(r)) -> Goal(r)) -> Goal(Chess.scan_step(b,src,r)):
  (table,+tail)=r
  +pawn=M.pawn(b,src)
  after(Goal,b,src,pawn,Chess.get_ep(b),tail,Chess.piece_targets(pawn,table,b,src),{==},good,next)
def scan(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,+keys: List<&2,U32>,+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,
  good: S.every(m => M.emitted(b,m),E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(m => M.emitted(b,m),E.moves(r)) -> Goal(r)) -> Goal(Chess.scan(keys,b,r)):
  match keys:
    case Nil{}: next(r,good)
    case Con{+src,+rest}:
      step(z => Goal(Chess.scan(rest,b,z)),b,src,r,good,z => pz => scan(Goal,rest,b,z,pz,next))
def initial(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,table: Array<U64>,+b: Chess.Board,
  next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(m => M.emitted(b,m),E.moves(r)) -> Goal(r)) -> Goal(S.candidates(table,b)):
  +own=Chess.color(b,Chess.get_turn(b))
  scan(Goal,Chess.bit_squares(64n,U64.is_zero(own),own,Nil{}),b,(table,Nil{}),Unit{},next)
def suffix(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,
  +parent: {U64.is_zero(U64.and(Chess.get_pawns(b),M.mask())) == True{} : Bool},good: S.every(m => M.emitted(b,m),E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(m => M.emitted(b,m),E.moves(r)) -> Goal(r)) -> Goal(S.suffix(b,r)):
  Chain.suffix(m => M.emitted(b,m),Goal,b,r,good,g => M.castle(b,True{},parent),g => M.castle(b,False{},parent),next)
def legal(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,table: Array<U64>,+b: Chess.Board,+parent: {U64.is_zero(U64.and(Chess.get_pawns(b),M.mask())) == True{} : Bool},
  next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(m => M.emitted(b,m),E.moves(r)) -> Goal(r)) -> Goal(Chess.legal_moves(table,b)):
  initial(z => Goal(S.suffix(b,z)),table,b,z => pz => suffix(Goal,b,z,parent,pz,next))
def legal_member(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,+parent: {U64.is_zero(U64.and(Chess.get_pawns(b),M.mask())) == True{} : Bool},
  here: E.member(m,E.moves(Chess.legal_moves(table,b)))) -> M.safe(Chess.make_move(b,m)):
  M.post(b,m,parent,legal(z => E.member(m,E.moves(z)) -> M.emitted(b,m),table,b,parent,
    z => pz => Lists.extract(m => M.emitted(b,m),z,m,pz))(here))
def children(+b: Chess.Board,xs: List<&2,Chess.Ply>,+parent: {U64.is_zero(U64.and(Chess.get_pawns(b),M.mask())) == True{} : Bool},good: S.every(m => M.emitted(b,m),xs)) ->
  S.every(m => M.safe(Chess.make_move(b,m)),xs):
  match xs:
    case Nil{}: Unit{}
    case Con{+m,t}:
      (head,tail)=good
      (M.post(b,m,parent,head),children(b,t,parent,tail))
