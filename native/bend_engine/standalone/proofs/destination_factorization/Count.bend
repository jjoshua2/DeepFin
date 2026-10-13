# Exact multiplicity of the complete (source,destination,promotion,flag) tuple.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ./Spec.bend as S
import ./Emission.bend as E

def block(promote: Bool,+src: U32,+dst: U32,+tag: U32,+query: Chess.Ply,+tail: List<&2,Chess.Ply>) ->
  {S.count(List.append(&2,Chess.Ply,S.block(promote,src,dst,tag),tail),query) ==
    S.contribution(promote,src,dst,tag,query,S.count(tail,query)) : Nat}:
  match promote:
    case True{}: {==}
    case False{}: {==}

def keys(xs: List<&2,U32>,+src: U32,+pawn: Bool,+ep: U32,+query: Chess.Ply,+tail: List<&2,Chess.Ply>) ->
  {S.count(S.expand(xs,src,pawn,ep,tail),query) == S.tally(xs,src,pawn,ep,query,S.count(tail,query)) : Nat}:
  match xs:
    case Nil{}: {==}
    case Con{+dst,+rest}:
      +promo = S.promotes(pawn,dst)
      +tag = S.flag(pawn,dst,ep)
      Equal.trans(Nat,S.count(S.expand(Con{dst,rest},src,pawn,ep,tail),query),
        S.contribution(promo,src,dst,tag,query,S.count(S.expand(rest,src,pawn,ep,tail),query)),
        S.tally(Con{dst,rest},src,pawn,ep,query,S.count(tail,query)),
        block(promo,src,dst,tag,query,S.expand(rest,src,pawn,ep,tail)),
        Equal.cong(Nat,Nat,n => S.contribution(promo,src,dst,tag,query,n),
          S.count(S.expand(rest,src,pawn,ep,tail),query),S.tally(rest,src,pawn,ep,query,S.count(tail,query)),
          keys(rest,src,pawn,ep,query,tail)))

def destinations(+n: Nat,+empty: Bool,+bb: U64,+src: U32,+pawn: Bool,+ep: U32,+query: Chess.Ply,+tail: List<&2,Chess.Ply>) ->
  {S.count(Chess.destinations(n,empty,bb,src,pawn,ep,tail),query) ==
    S.tally(Chess.bit_squares(n,empty,bb,Nil{}),src,pawn,ep,query,S.count(tail,query)) : Nat}:
  Equal.trans(Nat,S.count(Chess.destinations(n,empty,bb,src,pawn,ep,tail),query),
    S.count(S.expand(Chess.bit_squares(n,empty,bb,Nil{}),src,pawn,ep,tail),query),
    S.tally(Chess.bit_squares(n,empty,bb,Nil{}),src,pawn,ep,query,S.count(tail,query)),
    Equal.cong(List<&2,Chess.Ply>,Nat,ms => S.count(ms,query),
      Chess.destinations(n,empty,bb,src,pawn,ep,tail),S.expand(Chess.bit_squares(n,empty,bb,Nil{}),src,pawn,ep,tail),
      E.destinations(n,empty,bb,src,pawn,ep,tail)),
    keys(Chess.bit_squares(n,empty,bb,Nil{}),src,pawn,ep,query,tail))

def promotion_excludes_zero(+src: U32,+dst: U32,+tag: U32,+qs: U32,+qd: U32,+qt: U32) ->
  {S.contribution(True{},src,dst,tag,Chess.Ply{qs,qd,0,qt},0n) == 0n : Nat}:
  {==}
