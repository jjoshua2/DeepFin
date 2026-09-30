# Mask identity derived directly from unsigned Word comparison, not unary 2^32.
import Base
import ./Compare.bend as Compare
import ./Mask.bend as Mask
import ../index/Bits.bend as Bits

def limit(n: Nat, k: Nat) -> Word(n):
  match n k:
    case 0n _: WNil{}
    case 1n+p 0n: WCon{True{},Word.zero(p)}
    case 1n+p 1n+q: WCon{False{},limit(p,q)}

def bounded(+n: Nat, +k: Nat, +w: Word(n),
  +e: {Cmp.is_lt(Word.cmp(n,w,limit(n,k))) == True{} : Bool}) ->
  {Word.and(n,w,Mask.clip_mask(n,k)) == w : Word(n)}:
  match n k:
    case 0n _:
      match w:
        case WNil{}: {==}
    case 1n+p 0n:
      match w:
        case WCon{b,t}:
          Equal.trans(Word(1n+p),Word.and(1n+p,WCon{b,t},Word.zero(1n+p)),Word.zero(1n+p),WCon{b,t},
            Mask.and_zero(1n+p,WCon{b,t}),
            Equal.sym(Word(1n+p),WCon{b,t},Word.zero(1n+p),
              Compare.below_one_finish(p,b,t,Compare.le_zero(p,t,Compare.tail_of_below_one(b,Word.cmp(p,t,Word.zero(p)),e)),e)))
    case 1n+p 1n+q:
      match w:
        case WCon{False{},t}:
          Equal.cong(Word(p),Word(1n+p),x => WCon{False{},x},Word.and(p,t,Mask.clip_mask(p,q)),t,
            bounded(p,q,t,Compare.strip_low(False{},Word.cmp(p,t,limit(p,q)),e)))
        case WCon{True{},t}:
          Equal.cong(Word(p),Word(1n+p),x => WCon{True{},x},Word.and(p,t,Mask.clip_mask(p,q)),t,
            bounded(p,q,t,Compare.strip_low(True{},Word.cmp(p,t,limit(p,q)),e)))

def table_address(+i: U32, e: {U32.is_lt(i,131072) == True{} : Bool}) ->
  {U32.and(i,U32.sub(131072,1)) == i : U32}:
  match i:
    case U32{w}:
      Equal.cong(Word(32n),U32,x => U32{x},Word.and(32n,w,Mask.clip_mask(32n,17n)),w,bounded(32n,17n,w,e))
