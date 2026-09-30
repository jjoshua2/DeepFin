# Structural reduction of the real address mask to the existing low-bit clip.
import Base
import ../index/Bits.bend as Bits

def and_zero(+n: Nat, +w: Word(n)) ->
  {Word.and(n,w,Word.zero(n)) == Word.zero(n) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w:
        case WCon{False{},t}:
          Equal.cong(Word(p),Word(1n+p),v => WCon{False{},v},Word.and(p,t,Word.zero(p)),Word.zero(p),and_zero(p,t))
        case WCon{True{},t}:
          Equal.cong(Word(p),Word(1n+p),v => WCon{False{},v},Word.and(p,t,Word.zero(p)),Word.zero(p),and_zero(p,t))

def clip_mask(+n: Nat, +k: Nat) -> Word(n):
  Bits.clip(n,k,Word.not(n,Word.zero(n)))

def and_clip(+n: Nat, +k: Nat, +w: Word(n)) ->
  {Word.and(n,w,clip_mask(n,k)) == Bits.clip(n,k,w) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match k:
        case 0n: and_zero(1n+p,w)
        case 1n+q:
          match w:
            case WCon{False{},t}:
              Equal.cong(Word(p),Word(1n+p),v => WCon{False{},v},Word.and(p,t,clip_mask(p,q)),Bits.clip(p,q,t),and_clip(p,q,t))
            case WCon{True{},t}:
              Equal.cong(Word(p),Word(1n+p),v => WCon{True{},v},Word.and(p,t,clip_mask(p,q)),Bits.clip(p,q,t),and_clip(p,q,t))
