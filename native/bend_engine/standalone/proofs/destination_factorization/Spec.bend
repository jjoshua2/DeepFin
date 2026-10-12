# Independent emission and full-Ply multiplicity for a supplied square list.
# Square-list coverage and legal geometry are separate obligations.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../promotion/Spec.bend as Promotion

def promotes(+pawn: Bool,+dst: U32) -> Bool:
  +rank = U32.div(dst,8)
  Bool.and(pawn,Bool.or(U32.is_zero(rank),U32.is_eq(rank,7)))

def flag(+pawn: Bool,+dst: U32,+ep: U32) -> U32:
  Bool.to_u32(Bool.and(pawn,U32.is_eq(dst,ep)))

def block(promote: Bool,+src: U32,+dst: U32,tag: U32) -> List<&2,Chess.Ply>:
  match promote:
    case True{}: Promotion.choices(src,dst)
    case False{}: Con{Chess.Ply{src,dst,0,tag},Nil{}}

def expand(keys: List<&2,U32>,+src: U32,+pawn: Bool,+ep: U32,tail: List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  match keys:
    case Nil{}: tail
    case Con{+dst,rest}:
      List.append(&2,Chess.Ply,block(promotes(pawn,dst),src,dst,flag(pawn,dst,ep)),expand(rest,src,pawn,ep,tail))

def same(a: Chess.Ply,b: Chess.Ply) -> Bool:
  match a:
    case Chess.Ply{src,dst,qp,tag}:
      match b:
        case Chess.Ply{qs,qd,qpart,qtag}:
          Bool.and(U32.is_eq(qp,qpart),Bool.and(U32.is_eq(tag,qtag),
            Bool.and(U32.is_eq(src,qs),U32.is_eq(dst,qd))))

def bump(hit: Bool,n: Nat) -> Nat:
  match hit:
    case False{}: n
    case True{}: 1n+n

def count(xs: List<&2,Chess.Ply>,+query: Chess.Ply) -> Nat:
  match xs:
    case Nil{}: 0n
    case Con{head,rest}: bump(same(head,query),count(rest,query))

def contribution(promote: Bool,+src: U32,+dst: U32,tag: U32,+query: Chess.Ply,total: Nat) -> Nat:
  match promote:
    case True{}:
      bump(same(Chess.Ply{src,dst,1,0},query),bump(same(Chess.Ply{src,dst,2,0},query),
        bump(same(Chess.Ply{src,dst,3,0},query),bump(same(Chess.Ply{src,dst,4,0},query),total))))
    case False{}: bump(same(Chess.Ply{src,dst,0,tag},query),total)

def tally(keys: List<&2,U32>,+src: U32,+pawn: Bool,+ep: U32,+query: Chess.Ply,total: Nat) -> Nat:
  match keys:
    case Nil{}: total
    case Con{+dst,rest}:
      contribution(promotes(pawn,dst),src,dst,flag(pawn,dst,ep),query,tally(rest,src,pawn,ep,query,total))
