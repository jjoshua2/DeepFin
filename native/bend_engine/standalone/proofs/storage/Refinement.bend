# Connect actual affine read/write operations to a duplicable structural image.
import Base
import ./Observe.bend as O
import ./Read.bend as Read
import ./Model.bend as M

def set_lo(-ys: Array<U64>, r: Array<U64> & U64) ->
  {Array.set.fin(U64,Array.swap.lo(U64,ys,r)) == ANode{Array.set.fin(U64,r),ys} : Array<U64>}:
  (xs,v) = r
  {==}

def set_hi(-xs: Array<U64>, r: Array<U64> & U64) ->
  {Array.set.fin(U64,Array.swap.hi(U64,xs,r)) == ANode{xs,Array.set.fin(U64,r)} : Array<U64>}:
  (ys,v) = r
  {==}

def set_branch(+xs: O.Cells, +ys: O.Cells, +h: U32, +i: U32, +v: U64, z: Bool,
  left: {O.set_go(O.pack(xs),h,i,v) == O.pack(M.update(xs,h,i,v)) : Array<U64>},
  right: {O.set_go(O.pack(ys),h,U32.sub(i,h),v) == O.pack(M.update(ys,h,U32.sub(i,h),v)) : Array<U64>}) ->
  {Array.set.fin(U64,Array.swap.if(U64,O.pack(xs),O.pack(ys),h,i,v,z)) ==
    O.pack(Bool.pick(O.Cells,z,O.Branch{M.update(xs,h,i,v),ys},O.Branch{xs,M.update(ys,h,U32.sub(i,h),v)})) : Array<U64>}:
  match z:
    case True{}:
      Equal.trans(Array<U64>,Array.set.fin(U64,Array.swap.lo(U64,O.pack(ys),Array.swap.go(U64,O.pack(xs),h,i,v))),
        ANode{O.set_go(O.pack(xs),h,i,v),O.pack(ys)},O.pack(O.Branch{M.update(xs,h,i,v),ys}),
        set_lo(O.pack(ys),Array.swap.go(U64,O.pack(xs),h,i,v)),
        Equal.cong(Array<U64>,Array<U64>,a => ANode{a,O.pack(ys)},O.set_go(O.pack(xs),h,i,v),O.pack(M.update(xs,h,i,v)),left))
    case False{}:
      Equal.trans(Array<U64>,Array.set.fin(U64,Array.swap.hi(U64,O.pack(xs),Array.swap.go(U64,O.pack(ys),h,U32.sub(i,h),v))),
        ANode{O.pack(xs),O.set_go(O.pack(ys),h,U32.sub(i,h),v)},O.pack(O.Branch{xs,M.update(ys,h,U32.sub(i,h),v)}),
        set_hi(O.pack(xs),Array.swap.go(U64,O.pack(ys),h,U32.sub(i,h),v)),
        Equal.cong(Array<U64>,Array<U64>,a => ANode{O.pack(xs),a},O.set_go(O.pack(ys),h,U32.sub(i,h),v),O.pack(M.update(ys,h,U32.sub(i,h),v)),right))

def set_go(+c: O.Cells, +n: U32, +i: U32, +v: U64) ->
  {O.set_go(O.pack(c),n,i,v) == O.pack(M.update(c,n,i,v)) : Array<U64>}:
  match c:
    case O.Cell{x}: {==}
    case O.Branch{xs,ys}:
      set_branch(xs,ys,U32.shr(n),i,v,U32.is_lt(i,U32.shr(n)),
        set_go(xs,U32.shr(n),i,v),set_go(ys,U32.shr(n),U32.sub(i,U32.shr(n)),v))

def read_branch(+xs: O.Cells, +ys: O.Cells, +h: U32, +i: U32, z: Bool,
  left: {Array.get.go(U64,O.pack(xs),h,i) == (O.pack(xs),M.read(xs,h,i)) : Array<U64> & U64},
  right: {Array.get.go(U64,O.pack(ys),h,U32.sub(i,h)) == (O.pack(ys),M.read(ys,h,U32.sub(i,h))) : Array<U64> & U64}) ->
  {Array.get.if(U64,O.pack(xs),O.pack(ys),h,i,z) ==
    (O.pack(O.Branch{xs,ys}),Bool.pick(U64,z,M.read(xs,h,i),M.read(ys,h,U32.sub(i,h)))) : Array<U64> & U64}:
  match z:
    case True{}:
      %Equal.sym(Array<U64> & U64,Array.get.go(U64,O.pack(xs),h,i),(O.pack(xs),M.read(xs,h,i)),left) :
        {Array.swap.lo(U64,O.pack(ys),_) == (O.pack(O.Branch{xs,ys}),M.read(xs,h,i)) : Array<U64> & U64}
      {==}
    case False{}:
      %Equal.sym(Array<U64> & U64,Array.get.go(U64,O.pack(ys),h,U32.sub(i,h)),(O.pack(ys),M.read(ys,h,U32.sub(i,h))),right) :
        {Array.swap.hi(U64,O.pack(xs),_) == (O.pack(O.Branch{xs,ys}),M.read(ys,h,U32.sub(i,h))) : Array<U64> & U64}
      {==}

def read_go(+c: O.Cells, +n: U32, +i: U32) ->
  {Array.get.go(U64,O.pack(c),n,i) == (O.pack(c),M.read(c,n,i)) : Array<U64> & U64}:
  match c:
    case O.Cell{x}: {==}
    case O.Branch{xs,ys}:
      read_branch(xs,ys,U32.shr(n),i,U32.is_lt(i,U32.shr(n)),
        read_go(xs,U32.shr(n),i),read_go(ys,U32.shr(n),U32.sub(i,U32.shr(n))))

def size_branch(+xs: O.Cells, +ys: O.Cells, +h: U32, +i: U32, +v: U64, z: Bool,
  left: {O.size(M.update(xs,h,i,v)) == O.size(xs) : U32}) ->
  {O.size(Bool.pick(O.Cells,z,O.Branch{M.update(xs,h,i,v),ys},O.Branch{xs,M.update(ys,h,U32.sub(i,h),v)})) == U32.shl(O.size(xs)) : U32}:
  match z:
    case True{}:
      Equal.cong(U32,U32,x => U32.shl(x),O.size(M.update(xs,h,i,v)),O.size(xs),left)
    case False{}: {==}

def size_update(+c: O.Cells, +n: U32, +i: U32, +v: U64) ->
  {O.size(M.update(c,n,i,v)) == O.size(c) : U32}:
  match c:
    case O.Cell{x}: {==}
    case O.Branch{xs,ys}:
      size_branch(xs,ys,U32.shr(n),i,v,U32.is_lt(i,U32.shr(n)),size_update(xs,U32.shr(n),i,v))

def model_branch(+xs: O.Cells, +ys: O.Cells, +n: U32, +i: U32, +v: U64,
  z: Bool, same: {U32.is_lt(i,U32.shr(n)) == z : Bool},
  left: {M.read(M.update(xs,U32.shr(n),i,v),U32.shr(n),i) == v : U64},
  right: {M.read(M.update(ys,U32.shr(n),U32.sub(i,U32.shr(n)),v),U32.shr(n),U32.sub(i,U32.shr(n))) == v : U64}) ->
  {M.read(Bool.pick(O.Cells,z,O.Branch{M.update(xs,U32.shr(n),i,v),ys},O.Branch{xs,M.update(ys,U32.shr(n),U32.sub(i,U32.shr(n)),v)}),n,i) == v : U64}:
  match z:
    case True{}:
      %Equal.sym(Bool,U32.is_lt(i,U32.shr(n)),True{},same) :
        {Bool.pick(U64,_,M.read(M.update(xs,U32.shr(n),i,v),U32.shr(n),i),M.read(ys,U32.shr(n),U32.sub(i,U32.shr(n)))) == v : U64}
      left
    case False{}:
      %Equal.sym(Bool,U32.is_lt(i,U32.shr(n)),False{},same) :
        {Bool.pick(U64,_,M.read(xs,U32.shr(n),i),M.read(M.update(ys,U32.shr(n),U32.sub(i,U32.shr(n)),v),U32.shr(n),U32.sub(i,U32.shr(n)))) == v : U64}
      right

def model_read_update(+c: O.Cells, +n: U32, +i: U32, +v: U64) ->
  {M.read(M.update(c,n,i,v),n,i) == v : U64}:
  match c:
    case O.Cell{x}: {==}
    case O.Branch{xs,ys}:
      model_branch(xs,ys,n,i,v,U32.is_lt(i,U32.shr(n)),{==},
        model_read_update(xs,U32.shr(n),i,v),model_read_update(ys,U32.shr(n),U32.sub(i,U32.shr(n)),v))
