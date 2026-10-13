import Base
import ./Spec.bend as S
import ./Apply.bend as Apply
import ../storage/Observe.bend as O
import ../initialized_attacks/Build.bend as Init
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol

def tail(+c: O.Cells,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+w: Nat,+bl: Nat,xs: List<&2,String>,r: Array<U64> & Maybe<&2,Pos.Game>,
  +built: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>},+depth: {d == 17n : Nat},+full: {extra == 64n : Nat},parent: S.root(c,w,bl,r)) ->
  S.root(c,w,bl,Protocol.moves(xs,r)):
  match xs:
    case Nil{}: parent
    case Con{+text,+rest}:
      (table,+maybe)=r
      (same,good)=parent
      %Equal.sym(Array<U64>,table,O.pack(c),same) : S.root(c,w,bl,Protocol.moves(rest,Protocol.apply_one(text,maybe,_)))
      tail(c,d,seed,n,extra,w,bl,rest,Protocol.apply_one(text,maybe,O.pack(c)),built,depth,full,Apply.one(c,d,seed,n,extra,w,bl,maybe,text,built,depth,full,good))
