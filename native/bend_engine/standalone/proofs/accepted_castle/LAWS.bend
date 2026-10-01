# Accepted destination checks: operational result first, initialized geometry second.
import Base
import ./Spec.bend as S
import ./Destination.bend as D
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as C
import ../board/Spec.bend as B
import ../castle_kings/Spec.bend as K
import ../initialized_attacks/Build.bend as Init
import ../initialized_attacks/Spec.bend as G
import ../../../legal_probe/Chess.bend as Chess

law prepared_castle_check_returns_false:
  for a: Array<U64>
  for +b: Chess.Board
  for +xs: List<&2,Chess.Ply>
  for +m: Chess.Ply
  for castle: C.castle(b,m)
  for here: E.member(m,E.moves(Chess.filter_prepare(b,(a,xs))))
  {Chess.in_check(a,Chess.make_move(b,m),Chess.get_turn(b)) == (a,False{}) : Array<U64> & Bool}

law generated_castle_check_returns_false:
  for a: Array<U64>
  for +b: Chess.Board
  for +m: Chess.Ply
  for here: E.member(m,E.moves(Chess.legal_moves(a,b)))
  for tag: {C.flag(m) == 2 : U32}
  {Chess.in_check(a,Chess.make_move(b,m),Chess.get_turn(b)) == (a,False{}) : Array<U64> & Bool}

law initialized_generated_castle_destination_is_unattacked:
  for +d: Nat
  for +seed: U64
  for +n: Nat
  for +extra: Nat
  for +b: Chess.Board
  for +white: Bool
  for +m: Chess.Ply
  for depth: {d == 17n : Nat}
  for tables: {n == 128n : Nat}
  for full: {extra == 64n : Nat}
  for good: {B.valid(b) == True{} : Bool}
  for turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32}
  for one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64}
  for here: E.member(m,E.moves(Chess.legal_moves(Init.run(d,seed,n,extra),b)))
  for tag: {C.flag(m) == 2 : U32}
  {G.attacked(Chess.make_move(b,m),U32.to_nat(D.square(m)),Bool.not(white)) == False{} : Bool}
