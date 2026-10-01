# Structural reconstruction of a Word from its bounded bit observations.
import Base
import ../decoder/Bits.bend as B

def word(+width: Nat,n: Nat,+i: Nat,+w: Word(width)) -> Word(n):
  match n:
    case 0n: WNil{}
    case 1n+p: WCon{B.get(width,w,i),word(width,p,1n+i,w)}

def cons(+n: Nat,+a: Bool,+b: Bool,+x: Word(n),+y: Word(n),ab: {a == b : Bool},xy: {x == y : Word(n)}) ->
  {WCon{a,x} == WCon{b,y} : Word(1n+n)}:
  Equal.trans(Word(1n+n),WCon{a,x},WCon{b,x},WCon{b,y},
    Equal.cong(Bool,Word(1n+n),h => WCon{h,x},a,b,ab),
    Equal.cong(Word(n),Word(1n+n),t => WCon{b,t},x,y,xy))

def tail(+width: Nat,n: Nat,+i: Nat,+h: Bool,+w: Word(width)) ->
  {word(1n+width,n,1n+i,WCon{h,w}) == word(width,n,i,w) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      Equal.cong(Word(p),Word(1n+p),t => WCon{B.get(width,w,i),t},
        word(1n+width,p,2n+i,WCon{h,w}),word(width,p,1n+i,w),tail(width,p,1n+i,h,w))

def identity(+n: Nat,+w: Word(n)) -> {word(n,n,0n,w) == w : Word(n)}:
  match n:
    case 0n:
      match w:
        case WNil{}: {==}
    case 1n+p:
      match w:
        case WCon{+h,+t}:
          Equal.cong(Word(p),Word(1n+p),xs => WCon{h,xs},
            word(1n+p,p,1n,WCon{h,t}),t,
            Equal.trans(Word(p),word(1n+p,p,1n,WCon{h,t}),word(p,p,0n,t),t,
              tail(p,p,0n,h,t),identity(p,t)))
