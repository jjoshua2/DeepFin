# Proof-only structural image, not a runtime substitute for Base Array APIs.
import Base
import ./Observe.bend as O

def update(c: O.Cells, +n: U32, +i: U32, +v: U64) -> O.Cells:
  match c:
    case O.Cell{x}: O.Cell{v}
    case O.Branch{+xs,+ys}:
      Bool.pick(O.Cells,U32.is_lt(i,U32.shr(n)),
        O.Branch{update(xs,U32.shr(n),i,v),ys},
        O.Branch{xs,update(ys,U32.shr(n),U32.sub(i,U32.shr(n)),v)})

def read(c: O.Cells, +n: U32, +i: U32) -> U64:
  match c:
    case O.Cell{x}: x
    case O.Branch{xs,ys}:
      Bool.pick(U64,U32.is_lt(i,U32.shr(n)),
        read(xs,U32.shr(n),i),read(ys,U32.shr(n),U32.sub(i,U32.shr(n))))

def normalized(+c: O.Cells, i: U32) -> U32:
  U32.and(i,U32.sub(O.size(c),1))

def updated(+c: O.Cells, +i: U32, v: U64) -> O.Cells:
  update(c,O.size(c),normalized(c,i),v)

def read_sized(c: O.Cells, +n: U32, i: U32) -> U64:
  read(c,n,U32.and(i,U32.sub(n,1)))
