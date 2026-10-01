# Complete actual initialized result pairs composed with forward witnesses.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../initialized_attacks/Build.bend as Init
import ../initialized_attacks/Spec.bend as Geometry
import ../initialized_attacks/Composition.bend as Initialized
import ./Spec.bend as Forward
import ./Bridge.bend as Bridge
import ./Runtime.bend as Runtime

def attacked(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+target: Nat,+by: Bool,
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  +bound: {Nat.is_lt(target,64n) == True{} : Bool}) ->
  {Chess.attacked(Init.run(d,seed,n,extra),b,U32.from_nat(target),Bool.to_u32(by)) ==
    (Init.run(d,seed,n,extra),Forward.attacked(b,target,by)) : Array<U64> & Bool}:
  Equal.trans(Array<U64> & Bool,
    Chess.attacked(Init.run(d,seed,n,extra),b,U32.from_nat(target),Bool.to_u32(by)),
    Runtime.route(Init.run(d,seed,n,extra),b,U32.from_nat(target),Bool.to_u32(by)),
    (Init.run(d,seed,n,extra),Forward.attacked(b,target,by)),
    Runtime.unfold(Init.run(d,seed,n,extra),b,U32.from_nat(target),Bool.to_u32(by)),
    Equal.trans(Array<U64> & Bool,
      Runtime.route(Init.run(d,seed,n,extra),b,U32.from_nat(target),Bool.to_u32(by)),
      (Init.run(d,seed,n,extra),Geometry.attacked(b,target,by)),
      (Init.run(d,seed,n,extra),Forward.attacked(b,target,by)),
      Initialized.attacked(d,seed,n,extra,b,target,by,depth,tables,full,bound),
      Equal.cong(Bool,Array<U64> & Bool,answer => (Init.run(d,seed,n,extra),answer),
        Geometry.attacked(b,target,by),Forward.attacked(b,target,by),
        Bridge.geometry(b,target,by,bound))))

def checked(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+target: Nat,+side: Bool,
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  +bound: {Nat.is_lt(target,64n) == True{} : Bool},
  one: {U64.and(Chess.get_kings(b),Chess.color(b,Bool.to_u32(side))) == U64.bit(target) : U64}) ->
  {Chess.in_check(Init.run(d,seed,n,extra),b,Bool.to_u32(side)) ==
    (Init.run(d,seed,n,extra),Forward.attacked(b,target,Bool.not(side))) : Array<U64> & Bool}:
  Equal.trans(Array<U64> & Bool,
    Chess.in_check(Init.run(d,seed,n,extra),b,Bool.to_u32(side)),
    (Init.run(d,seed,n,extra),Geometry.attacked(b,target,Bool.not(side))),
    (Init.run(d,seed,n,extra),Forward.attacked(b,target,Bool.not(side))),
    Initialized.checked(d,seed,n,extra,b,target,side,depth,tables,full,bound,one),
    Equal.cong(Bool,Array<U64> & Bool,answer => (Init.run(d,seed,n,extra),answer),
      Geometry.attacked(b,target,Bool.not(side)),Forward.attacked(b,target,Bool.not(side)),
      Bridge.geometry(b,target,Bool.not(side),bound)))
