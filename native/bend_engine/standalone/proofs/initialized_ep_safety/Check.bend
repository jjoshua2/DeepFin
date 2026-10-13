# Initialized query certificate at the preserved original king.
import Base
import ./Frame.bend as Frame
import ./Finish.bend as Finish
import ../ep_king_frame/Frame.bend as F
import ../initialized_attacks/Build.bend as Init
import ../initialized_attacks/Composition.bend as Composition
import ../initialized_attacks/Spec.bend as G
import ../../../legal_probe/Chess.bend as Chess

def selected(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+m: Chess.Ply,+q: Nat,+side: Bool,
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  bound: {Nat.is_lt(q,64n) == True{} : Bool},
  +turn: {Chess.get_turn(b) == Bool.to_u32(side) : U32},
  one: {F.plane(b,Bool.to_u32(side)) == U64.bit(q) : U64},
  frame: Frame.preservation(b,m),
  no: {Chess.in_check(Init.run(d,seed,n,extra),Chess.make_move(b,m),Chess.get_turn(b)) ==
    (Init.run(d,seed,n,extra),False{}) : Array<U64> & Bool}) ->
  {G.attacked(Chess.make_move(b,m),q,Bool.not(side)) == False{} : Bool}:
  Finish.geometry(Init.run(d,seed,n,extra),Chess.make_move(b,m),q,side,
    Composition.checked(d,seed,n,extra,Chess.make_move(b,m),q,side,
      depth,tables,full,bound,Frame.singleton(b,m,q,side,turn,one,frame)),
    Finish.old_turn(Init.run(d,seed,n,extra),b,m,side,turn,no))

