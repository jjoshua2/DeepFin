# Complete initialized attack results, with no assumed lookup answers.
import Base
import ./Build.bend as B
import ./Spec.bend as S
import ../board/Spec.bend as P
import ../../../legal_probe/Chess.bend as Chess

law initialized_piece_attack_matches_geometry:
  for +d: Nat
  for +seed: U64
  for +n: Nat
  for +extra: Nat
  for +kind: P.Piece
  for +sq: Nat
  for +white: Bool
  for +occ: U64
  for depth: {d == 17n : Nat}
  for tables: {n == 128n : Nat}
  for full: {extra == 64n : Nat}
  for bound: {Nat.is_lt(sq,64n) == True{} : Bool}
  {Chess.attack(P.tag(kind),B.run(d,seed,n,extra),U32.from_nat(sq),Bool.to_u32(white),occ) ==
    (B.run(d,seed,n,extra),S.mask(kind,sq,white,occ)) : Array<U64> & U64}

law initialized_attacked_matches_geometry:
  for +d: Nat
  for +seed: U64
  for +n: Nat
  for +extra: Nat
  for +b: Chess.Board
  for +sq: Nat
  for +by: Bool
  for depth: {d == 17n : Nat}
  for tables: {n == 128n : Nat}
  for full: {extra == 64n : Nat}
  for bound: {Nat.is_lt(sq,64n) == True{} : Bool}
  {Chess.attacked(B.run(d,seed,n,extra),b,U32.from_nat(sq),Bool.to_u32(by)) ==
    (B.run(d,seed,n,extra),S.attacked(b,sq,by)) : Array<U64> & Bool}

law initialized_single_king_check_matches_geometry:
  for +d: Nat
  for +seed: U64
  for +n: Nat
  for +extra: Nat
  for +b: Chess.Board
  for +sq: Nat
  for +side: Bool
  for depth: {d == 17n : Nat}
  for tables: {n == 128n : Nat}
  for full: {extra == 64n : Nat}
  for bound: {Nat.is_lt(sq,64n) == True{} : Bool}
  for one: {U64.and(Chess.get_kings(b),Chess.color(b,Bool.to_u32(side))) == U64.bit(sq) : U64}
  {Chess.in_check(B.run(d,seed,n,extra),b,Bool.to_u32(side)) ==
    (B.run(d,seed,n,extra),S.attacked(b,sq,Bool.not(side))) : Array<U64> & Bool}
