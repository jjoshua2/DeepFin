# Mathematical compact-index specification; not a replacement enumerator.
# Nat powers are deliberately unbounded, including a mask population of 64.
import Base

def power(k: Nat) -> Nat:
  match k:
    case 0n: 1n
    case 1n+p: Nat.double(power(p))

def value(x: U64) -> Nat:
  Word.to_nat(64n, U64.to_word(x))

def width(mask: U64) -> Nat:
  Word.count(64n, U64.to_word(mask))

def capacity(mask: U64) -> Nat:
  power(width(mask))

# Zero the bits above a mathematical compact width. This total reference
# accepts k > n as well; theorems use the mask's actual population.
def clip(n: Nat, k: Nat, w: Word(n)) -> Word(n):
  match n:
    case 0n: WNil{}
    case 1n+p:
      match k:
        case 0n: Word.zero(1n+p)
        case 1n+q:
          match w:
            case WCon{b, t}: WCon{b, clip(p, q, t)}
