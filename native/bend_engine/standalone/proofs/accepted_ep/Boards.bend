# Literal fixtures from the prior EP proofs; no theorem adds these as premises.
import Base
import ../../../legal_probe/Chess.bend as C
def white() -> C.Board:
  C.Board{U64.or(U64.bit(36n),U64.bit(35n)),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.or(U64.bit(0n),U64.bit(63n)),U64.or(U64.bit(0n),U64.bit(36n)),U64.or(U64.bit(63n),U64.bit(35n)),1,0,43}
def black() -> C.Board:
  C.Board{U64.or(U64.bit(27n),U64.bit(28n)),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.or(U64.bit(0n),U64.bit(63n)),U64.or(U64.bit(0n),U64.bit(28n)),U64.or(U64.bit(63n),U64.bit(27n)),0,0,20}
def discovered() -> C.Board:
  C.Board{U64.or(U64.bit(36n),U64.bit(35n)),U64.zero(),U64.zero(),U64.bit(39n),U64.zero(),U64.or(U64.bit(32n),U64.bit(63n)),U64.or(U64.bit(32n),U64.bit(36n)),U64.or(U64.or(U64.bit(63n),U64.bit(35n)),U64.bit(39n)),1,0,43}
