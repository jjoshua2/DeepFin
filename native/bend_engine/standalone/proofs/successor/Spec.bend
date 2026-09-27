# Mathematical successor specification only; no engine or compiler replacement.
import Base

# Structural mathematical observation, never called by the engine.
def increment(n: Nat, x: Word(n), mask: Word(n)) -> Word(n):
  match n:
    case 0n: WNil{}
    case 1n+p:
      match x mask:
        case WCon{b,t} WCon{False{},m}: WCon{False{},increment(p,t,m)}
        case WCon{False{},t} WCon{True{},m}: WCon{True{},Word.and(p,t,m)}
        case WCon{True{},t} WCon{True{},m}: WCon{False{},increment(p,t,m)}

# For 0 <= v < capacity this is (v+1) mod capacity, without machine powers.
def cycle(+v: Nat, cap: Nat) -> Nat:
  Bool.pick(Nat,Nat.is_lt(1n+v,cap),1n+v,0n)
