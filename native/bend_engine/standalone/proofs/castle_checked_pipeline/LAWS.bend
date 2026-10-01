import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_kings/Spec.bend as K
import ../castle_emission/Spec.bend as E
import ../board/Spec.bend as B
import ../initialized_attacks/Build.bend as Init
import ./Spec.bend as S

law initialized_castle_producer_matches_checks:
  for +d: Nat
  for +seed: U64
  for +n: Nat
  for +extra: Nat
  for +b: Chess.Board
  for +white: Bool
  for +ks: Bool
  for +tail: List<&2,Chess.Ply>
  for depth: {d == 17n : Nat}
  for tables: {n == 128n : Nat}
  for full: {extra == 64n : Nat}
  for good: {B.valid(b) == True{} : Bool}
  for turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32}
  for one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64}
  for guard: {E.guard(b,ks) == True{} : Bool}
  {Chess.castle_side(b,ks,(Init.run(d,seed,n,extra),tail)) == (Init.run(d,seed,n,extra),S.keep(Bool.or(S.check(b,white,ks,K.Initial{}),S.check(b,white,ks,K.Transit{})),S.move(white,ks),tail)) : Array<U64> & List<&2,Chess.Ply>}

law initialized_castle_pipeline_matches_three_checks:
  for +d: Nat
  for +seed: U64
  for +n: Nat
  for +extra: Nat
  for +b: Chess.Board
  for +white: Bool
  for +ks: Bool
  for depth: {d == 17n : Nat}
  for tables: {n == 128n : Nat}
  for full: {extra == 64n : Nat}
  for good: {B.valid(b) == True{} : Bool}
  for turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32}
  for one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64}
  for guard: {E.guard(b,ks) == True{} : Bool}
  {S.run(Init.run(d,seed,n,extra),b,ks) == (Init.run(d,seed,n,extra),S.keep(S.rejected(b,white,ks),S.move(white,ks),Nil{})) : Array<U64> & List<&2,Chess.Ply>}

law accepted_castle_pipeline_has_safe_stages:
  for +d: Nat
  for +seed: U64
  for +n: Nat
  for +extra: Nat
  for +b: Chess.Board
  for +white: Bool
  for +ks: Bool
  for +m: Chess.Ply
  for depth: {d == 17n : Nat}
  for tables: {n == 128n : Nat}
  for full: {extra == 64n : Nat}
  for good: {B.valid(b) == True{} : Bool}
  for turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32}
  for one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64}
  for guard: {E.guard(b,ks) == True{} : Bool}
  for here: E.member(m,E.moves(S.run(Init.run(d,seed,n,extra),b,ks)))
  {S.check(b,white,ks,K.Initial{}) == False{} : Bool} & ({S.check(b,white,ks,K.Transit{}) == False{} : Bool} & {S.check(b,white,ks,K.Completed{}) == False{} : Bool})
