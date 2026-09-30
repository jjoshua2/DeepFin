import Base
import ./Observe.bend as O
import ./Read.bend as Read
import ./Representation.bend as R

def shape_lo(-ys: Array<U64>, r: Array<U64> & U64) ->
  {O.shape(Array.set.fin(U64,Array.swap.lo(U64,ys,r))) == O.Node{O.shape(Array.set.fin(U64,r)),O.shape(ys)} : O.Shape}:
  (xs,v) = r
  {==}

def shape_hi(-xs: Array<U64>, r: Array<U64> & U64) ->
  {O.shape(Array.set.fin(U64,Array.swap.hi(U64,xs,r))) == O.Node{O.shape(xs),O.shape(Array.set.fin(U64,r))} : O.Shape}:
  (ys,v) = r
  {==}

def branch(+xs: O.Cells, +ys: O.Cells, +h: U32, +i: U32, +v: U64, z: Bool,
  left: {O.shape(O.set_go(O.pack(xs),h,i,v)) == O.shape(O.pack(xs)) : O.Shape},
  right: {O.shape(O.set_go(O.pack(ys),h,U32.sub(i,h),v)) == O.shape(O.pack(ys)) : O.Shape}) ->
  {O.shape(Array.set.fin(U64,Array.swap.if(U64,O.pack(xs),O.pack(ys),h,i,v,z)))
    == O.shape(O.pack(O.Branch{xs,ys})) : O.Shape}:
  match z:
    case True{}:
      Equal.trans(O.Shape,O.shape(Array.set.fin(U64,Array.swap.lo(U64,O.pack(ys),Array.swap.go(U64,O.pack(xs),h,i,v)))),
        O.Node{O.shape(O.set_go(O.pack(xs),h,i,v)),O.shape(O.pack(ys))},O.shape(O.pack(O.Branch{xs,ys})),
        shape_lo(O.pack(ys),Array.swap.go(U64,O.pack(xs),h,i,v)),
        Equal.cong(O.Shape,O.Shape,s => O.Node{s,O.shape(O.pack(ys))},
          O.shape(O.set_go(O.pack(xs),h,i,v)),O.shape(O.pack(xs)),left))
    case False{}:
      Equal.trans(O.Shape,O.shape(Array.set.fin(U64,Array.swap.hi(U64,O.pack(xs),Array.swap.go(U64,O.pack(ys),h,U32.sub(i,h),v)))),
        O.Node{O.shape(O.pack(xs)),O.shape(O.set_go(O.pack(ys),h,U32.sub(i,h),v))},O.shape(O.pack(O.Branch{xs,ys})),
        shape_hi(O.pack(xs),Array.swap.go(U64,O.pack(ys),h,U32.sub(i,h),v)),
        Equal.cong(O.Shape,O.Shape,s => O.Node{O.shape(O.pack(xs)),s},
          O.shape(O.set_go(O.pack(ys),h,U32.sub(i,h),v)),O.shape(O.pack(ys)),right))

def go(+c: O.Cells, +n: U32, +i: U32, +v: U64) ->
  {O.shape(O.set_go(O.pack(c),n,i,v)) == O.shape(O.pack(c)) : O.Shape}:
  match c:
    case O.Cell{x}: {==}
    case O.Branch{xs,ys}:
      branch(xs,ys,U32.shr(n),i,v,U32.is_lt(i,U32.shr(n)),
        go(xs,U32.shr(n),i,v),go(ys,U32.shr(n),U32.sub(i,U32.shr(n)),v))

def model(+c: O.Cells, +i: U32, +v: U64) ->
  {O.shape(Array.set(U64,O.pack(c),i,v)) == O.shape(O.pack(c)) : O.Shape}:
  %Equal.sym(Array<U64> & U32,Array.size(U64,O.pack(c)),(O.pack(c),O.size(c)),Read.size_pair(c)) :
    {O.shape(Array.set.fin(U64,Array.swap.at(U64,i,v,_))) == O.shape(O.pack(c)) : O.Shape}
  go(c,O.size(c),U32.and(i,U32.sub(O.size(c),1)),v)

def lift(-a: Array<U64>, +i: U32, +v: U64,
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {O.shape(Array.set(U64,a,i,v)) == O.shape(a) : O.Shape}:
  (+c,e) = cert
  +same: {O.pack(c) == a : Array<U64>} = e
  %same : {O.shape(Array.set(U64,_,i,v)) == O.shape(a) : O.Shape}
  %same : {O.shape(Array.set(U64,O.pack(c),i,v)) == O.shape(_) : O.Shape}
  model(c,i,v)

def write(a: Array<U64>, +i: U32, +v: U64) ->
  {O.shape(Array.set(U64,a,i,v)) == O.shape(a) : O.Shape}:
  lift(a,i,v,R.reify(a))
