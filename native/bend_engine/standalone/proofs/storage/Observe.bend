# Duplicable specification of array contents; pack only builds constructors.
# No host oracle, foreign storage or application runtime changes.
import Base

type Cells is Data:
  Cell{value: U64}
  Branch{left: Cells, right: Cells}

type Shape is Data:
  Leaf{}
  Node{left: Shape, right: Shape}

def pack(c: Cells) -> Array<U64>:
  match c:
    case Cell{x}: ALeaf{x}
    case Branch{xs,ys}: ANode{pack(xs),pack(ys)}

def unpack(a: Array<U64>) -> Cells:
  match a:
    case ALeaf{x}: Cell{x}
    case ANode{xs,ys}: Branch{unpack(xs),unpack(ys)}

def shape(a: Array<U64>) -> Shape:
  match a:
    case ALeaf{x}: Leaf{}
    case ANode{xs,ys}: Node{shape(xs),shape(ys)}

def storage(r: Array<U64> & U64) -> Array<U64>:
  (a,v) = r
  a

def sized(r: Array<U64> & U32) -> Array<U64>:
  (a,n) = r
  a

def count(r: Array<U64> & U32) -> U32:
  (a,n) = r
  n

def size(c: Cells) -> U32:
  match c:
    case Cell{x}: 1
    case Branch{xs,ys}: U32.shl(size(xs))

def set_go(a: Array<U64>, n: U32, i: U32, v: U64) -> Array<U64>:
  Array.set.fin(U64,Array.swap.go(U64,a,n,i,v))
