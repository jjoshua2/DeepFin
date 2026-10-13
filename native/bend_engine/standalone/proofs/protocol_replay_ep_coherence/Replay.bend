import Base
import ./Spec.bend as S
import ./Apply.bend as Apply
import ../protocol_replay_metadata/Spec.bend as Metadata
import ../fen_validated_full/Root.bend as Root
import ../storage/Observe.bend as O
import ../initialized_attacks/Build.bend as Init
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol

def tail(+c: O.Cells,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,texts: List<&2,String>,
  r: Array<U64> & Maybe<&2,Pos.Game>,
  +built: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>},
  +depth: {d == 17n : Nat},+full: {extra == 64n : Nat},checked: S.root(c,r)) ->
  S.root(c,Protocol.moves(texts,r)):
  match texts:
    case Nil{}: checked
    case Con{+text,+rest}:
      (a,+maybe)=r
      (same,good)=checked
      %Equal.sym(Array<U64>,a,O.pack(c),same) :
        S.root(c,Protocol.moves(rest,Protocol.apply_one(text,maybe,_)))
      tail(c,d,seed,n,extra,rest,Protocol.apply_one(text,maybe,O.pack(c)),built,depth,full,
        Apply.one(c,d,seed,n,extra,maybe,text,built,depth,full,S.weaken(maybe,good)))
def first(+c: O.Cells,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+maybe: Maybe<&2,Pos.Game>,+text: String,+rest: List<&2,String>,
  +built: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>},
  +depth: {d == 17n : Nat},+full: {extra == 64n : Nat},good: Metadata.parent(maybe)) ->
  S.root(c,Protocol.moves(Con{text,rest},(O.pack(c),maybe))):
  tail(c,d,seed,n,extra,rest,Protocol.apply_one(text,maybe,O.pack(c)),built,depth,full,
    Apply.one(c,d,seed,n,extra,maybe,text,built,depth,full,good))
def validated(+c: O.Cells,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+text: String,+rest: List<&2,String>,
  r: Array<U64> & Maybe<&2,Pos.Game>,checked: Root.root(c,r),
  built: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},full: {extra == 64n : Nat}) ->
  S.root(c,Protocol.moves(Con{text,rest},r)):
  (a,+maybe)=r
  (same,good)=checked
  %Equal.sym(Array<U64>,a,O.pack(c),same) : S.root(c,Protocol.moves(Con{text,rest},(_,maybe)))
  first(c,d,seed,n,extra,maybe,text,rest,built,depth,full,Metadata.accepted(maybe,good))
