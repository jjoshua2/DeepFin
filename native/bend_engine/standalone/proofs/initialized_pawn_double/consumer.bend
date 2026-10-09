# Public producer: actual initialized targets; no legal-list membership claim.
import Base
import ./Producer.bend as P
import ./Spec.bend as S
import ../attack_geometry/Initialized.bend as I
import ../../../legal_probe/Chess.bend as C

def initialized_double(+d: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,+b: C.Board,+src: U32,+dst: U32,
  depth: {d == 17n : Nat},full: {extra == 64n : Nat},below: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool},
  trigger: {U32.is_eq(U32.xor(src,dst),16) == True{} : Bool},
  hit: {U64.test_bit(S.mask(C.pawn_targets(I.run(d,seed,n,key,at,extra),b,src)),U32.to_nat(dst)) == True{} : Bool}) -> S.double(b,src,dst):
  P.initialized(d,seed,n,key,at,extra,b,src,dst,depth,full,below,trigger,hit)
