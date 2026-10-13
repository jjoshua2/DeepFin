# Actual retained EP supplies both acceptance and the king frame.
import Base
import ./Check.bend as Check
import ./Flag.bend as Flag
import ../accepted_ep/Public.bend as Accepted
import ../accepted_castle/Generated.bend as Split
import ../ep_king_frame/consumer.bend as EP
import ../ep_king_frame/Frame.bend as F
import ../board/Spec.bend as Board
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as C
import ../initialized_attacks/Build.bend as Init
import ../initialized_attacks/Spec.bend as G
import ../../../legal_probe/Chess.bend as Chess
import ../../Position.bend as Position

def actual(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+m: Chess.Ply,+q: Nat,+side: Bool,
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  good: {Board.valid(b) == True{} : Bool},
  coherent: {Position.valid_ep(Chess.get_ep(b),b) == True{} : Bool},
  turn: {Chess.get_turn(b) == Bool.to_u32(side) : U32},
  bound: {Nat.is_lt(q,64n) == True{} : Bool},
  one: {F.plane(b,Bool.to_u32(side)) == U64.bit(q) : U64},
  here: E.member(m,E.moves(Chess.legal_moves(Init.run(d,seed,n,extra),b))),
  +tag: {C.flag(m) == 1 : U32}) ->
  {G.attacked(Chess.make_move(b,m),q,Bool.not(side)) == False{} : Bool}:
  Split.split_member({G.attacked(Chess.make_move(b,m),q,Bool.not(side)) == False{} : Bool},
    E.moves(Chess.legal_moves(Init.run(d,seed,n,extra),b)),m,here,origin => accepted =>
      Check.selected(d,seed,n,extra,b,m,q,side,depth,tables,full,bound,turn,one,
        EP.legal_member(Init.run(d,seed,n,extra),b,m,good,coherent,origin,Flag.required(m,tag)),
        Accepted.legal(Init.run(d,seed,n,extra),b,m,accepted,tag)))
