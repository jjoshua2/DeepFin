# Finite certificates contain arithmetic and single-bit facts, never board planes.
import Base
import ./Files.bend as Files
import ./Move.bend as Move
import ../initialized_pawn_double/Spec.bend as D

def origin(+e: U32,+w: Bool) -> U32:
  Bool.pick(U32,U32.is_eq(Bool.to_u32(Bool.not(w)),1),U32.add(e,8),U32.sub(e,8))
def rank(+e: U32,+w: Bool) -> Bool:
  U32.is_eq(U32.div(e,8),Bool.pick(U32,U32.is_eq(Bool.to_u32(Bool.not(w)),1),5,2))

def facts(+s: U32,+d: U32,+e: U32,+w: Bool) -> Type:
  {rank(e,w) == True{} : Bool} &
  ({origin(e,w) == s : U32} &
  ({U32.xor(e,8) == d : U32} &
  ({D.first(s,w) == e : U32} &
  ({Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool} &
  ({Nat.is_lt(U32.to_nat(d),64n) == True{} : Bool} &
  ({Nat.is_lt(U32.to_nat(e),64n) == True{} : Bool} &
  ({U64.test_bit(U64.bit(U32.to_nat(s)),U32.to_nat(s)) == True{} : Bool} &
  ({U64.test_bit(U64.bit(U32.to_nat(d)),U32.to_nat(d)) == True{} : Bool} &
  ({U64.test_bit(U64.bit(U32.to_nat(d)),U32.to_nat(s)) == False{} : Bool} &
   {U64.test_bit(U64.bit(U32.to_nat(d)),U32.to_nat(e)) == False{} : Bool})))))))))

def file(+f: Files.File,+w: Bool) -> facts(Files.source(f,w),D.second(Files.source(f,w),w),Move.midpoint(Files.source(f,w),D.second(Files.source(f,w),w)),w):
  match f w:
    case Files.A{} True{}: ({==},({==},({==},({==},({==},({==},({==},({==},({==},({==},{==}))))))))))
    case Files.A{} False{}: ({==},({==},({==},({==},({==},({==},({==},({==},({==},({==},{==}))))))))))
    case Files.B{} True{}: ({==},({==},({==},({==},({==},({==},({==},({==},({==},({==},{==}))))))))))
    case Files.B{} False{}: ({==},({==},({==},({==},({==},({==},({==},({==},({==},({==},{==}))))))))))
    case Files.C{} True{}: ({==},({==},({==},({==},({==},({==},({==},({==},({==},({==},{==}))))))))))
    case Files.C{} False{}: ({==},({==},({==},({==},({==},({==},({==},({==},({==},({==},{==}))))))))))
    case Files.D{} True{}: ({==},({==},({==},({==},({==},({==},({==},({==},({==},({==},{==}))))))))))
    case Files.D{} False{}: ({==},({==},({==},({==},({==},({==},({==},({==},({==},({==},{==}))))))))))
    case Files.E{} True{}: ({==},({==},({==},({==},({==},({==},({==},({==},({==},({==},{==}))))))))))
    case Files.E{} False{}: ({==},({==},({==},({==},({==},({==},({==},({==},({==},({==},{==}))))))))))
    case Files.F{} True{}: ({==},({==},({==},({==},({==},({==},({==},({==},({==},({==},{==}))))))))))
    case Files.F{} False{}: ({==},({==},({==},({==},({==},({==},({==},({==},({==},({==},{==}))))))))))
    case Files.G{} True{}: ({==},({==},({==},({==},({==},({==},({==},({==},({==},({==},{==}))))))))))
    case Files.G{} False{}: ({==},({==},({==},({==},({==},({==},({==},({==},({==},({==},{==}))))))))))
    case Files.H{} True{}: ({==},({==},({==},({==},({==},({==},({==},({==},({==},({==},{==}))))))))))
    case Files.H{} False{}: ({==},({==},({==},({==},({==},({==},({==},({==},({==},({==},{==}))))))))))

def first(+s: U32,+d: U32,+e: U32,+w: Bool,cert: facts(s,d,e,w)) -> {D.first(s,w) == e : U32}:
  (rank_ok,(origin_same,(victim_same,(first_same,rest))))=cert
  first_same
