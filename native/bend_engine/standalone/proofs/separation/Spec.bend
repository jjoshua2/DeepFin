# Pure access-path specification. No expected read values occur in separation.
import Base
import ../storage/Observe.bend as O

def size(t: O.Shape) -> U32:
  match t:
    case O.Leaf{}: 1
    case O.Node{xs,ys}: U32.shl(size(xs))

def choice(l: Bool, r: Bool, a: Bool, b: Bool) -> Bool:
  match a b:
    case True{} True{}: l
    case True{} False{}: True{}
    case False{} True{}: True{}
    case False{} False{}: r

def apart(t: O.Shape, +n: U32, +i: U32, +j: U32) -> Bool:
  match t:
    case O.Leaf{}: False{}
    case O.Node{xs,ys}:
      choice(apart(xs,U32.shr(n),i,j),
        apart(ys,U32.shr(n),U32.sub(i,U32.shr(n)),U32.sub(j,U32.shr(n))),
        U32.is_lt(i,U32.shr(n)),U32.is_lt(j,U32.shr(n)))

def disjoint(+t: O.Shape, i: U32, j: U32) -> Bool:
  +n = size(t)
  apart(t,n,U32.and(i,U32.sub(n,1)),U32.and(j,U32.sub(n,1)))

def separate(a: Array<U64>, i: U32, j: U32) -> Bool:
  disjoint(O.shape(a),i,j)

def value(r: Array<U64> & U64) -> U64:
  (a,v) = r
  v

# Pure frame condition for the existing incrementing Tables.fill addresses.
def clear(+t: O.Shape, n: Nat, +at: U32, +j: U32) -> Bool:
  match n:
    case 0n: True{}
    case 1n+p: Bool.and(disjoint(t,at,j),clear(t,p,U32.inc(at),j))
