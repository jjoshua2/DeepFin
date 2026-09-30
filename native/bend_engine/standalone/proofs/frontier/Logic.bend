import Base
import ../ray/Frame.bend as F
import ../ray/Masking.bend as J
import ../u64/Words.bend as Old

def left(+a: Bool,+b: Bool,e: {Bool.and(a,b) == True{} : Bool}) -> {a == True{} : Bool}:
  F.left(a,b,e)
def right(+a: Bool,+b: Bool,e: {Bool.and(a,b) == True{} : Bool}) -> {b == True{} : Bool}:
  F.right(a,b,e)
def both(+a: Bool,+b: Bool,x: {a == True{} : Bool},y: {b == True{} : Bool}) -> {Bool.and(a,b) == True{} : Bool}:
  J.and_true(a,b,x,y)
def impossible(-T: Type,e: {False{} == True{} : Bool}) -> T:
  Old.false_true(T,e)

def covers(a: Bool,b: Bool) -> Bool:
  Bool.or(Bool.not(a),b)

def sub(+n: Nat,a: Word(n),b: Word(n)) -> Bool:
  match n:
    case 0n: True{}
    case 1n+p:
      match a b:
        case WCon{ah,at} WCon{bh,bt}: Bool.and(covers(ah,bh),sub(p,at,bt))

def apart(+n: Nat,a: Word(n),b: Word(n)) -> Bool:
  match n:
    case 0n: True{}
    case 1n+p:
      match a b:
        case WCon{ah,at} WCon{bh,bt}: Bool.and(Bool.not(Bool.and(ah,bh)),apart(p,at,bt))

def subset(a: U64,b: U64) -> Bool:
  match a b:
    case U64{U32{al},U32{ah}} U64{U32{bl},U32{bh}}: Bool.and(sub(32n,al,bl),sub(32n,ah,bh))

def disjoint(a: U64,b: U64) -> Bool:
  match a b:
    case U64{U32{al},U32{ah}} U64{U32{bl},U32{bh}}: Bool.and(apart(32n,al,bl),apart(32n,ah,bh))
