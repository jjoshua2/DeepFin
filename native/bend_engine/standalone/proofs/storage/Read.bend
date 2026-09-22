import Base
import ./Representation.bend as R
import ./Observe.bend as O

def sized_node(-ys: Array<U64>, r: Array<U64> & U32) ->
  {O.sized(Array.size.node(U64,ys,r)) == ANode{O.sized(r),ys} : Array<U64>}:
  (xs,n) = r
  {==}

def storage_lo(-ys: Array<U64>, r: Array<U64> & U64) ->
  {O.storage(Array.swap.lo(U64,ys,r)) == ANode{O.storage(r),ys} : Array<U64>}:
  (xs,v) = r
  {==}

def storage_hi(-xs: Array<U64>, r: Array<U64> & U64) ->
  {O.storage(Array.swap.hi(U64,xs,r)) == ANode{xs,O.storage(r)} : Array<U64>}:
  (ys,v) = r
  {==}

def size_pair(+c: O.Cells) -> {Array.size(U64,O.pack(c)) == (O.pack(c),O.size(c)) : Array<U64> & U32}:
  match c:
    case O.Cell{x}: {==}
    case O.Branch{xs,ys}:
      %size_pair(xs) :
        {Array.size.node(U64,O.pack(ys),Array.size(U64,O.pack(xs))) == Array.size.node(U64,O.pack(ys),_) : Array<U64> & U32}
      {==}

def get_if(+xs: O.Cells, +ys: O.Cells, +h: U32, +i: U32, z: Bool,
  left: {O.storage(Array.get.go(U64,O.pack(xs),h,i)) == O.pack(xs) : Array<U64>},
  right: {O.storage(Array.get.go(U64,O.pack(ys),h,U32.sub(i,h))) == O.pack(ys) : Array<U64>}) ->
  {O.storage(Array.get.if(U64,O.pack(xs),O.pack(ys),h,i,z)) == O.pack(O.Branch{xs,ys}) : Array<U64>}:
  match z:
    case True{}:
      Equal.trans(Array<U64>,O.storage(Array.swap.lo(U64,O.pack(ys),Array.get.go(U64,O.pack(xs),h,i))),
        ANode{O.storage(Array.get.go(U64,O.pack(xs),h,i)),O.pack(ys)},O.pack(O.Branch{xs,ys}),
        storage_lo(O.pack(ys),Array.get.go(U64,O.pack(xs),h,i)),
        Equal.cong(Array<U64>,Array<U64>,a => ANode{a,O.pack(ys)},O.storage(Array.get.go(U64,O.pack(xs),h,i)),O.pack(xs),left))
    case False{}:
      Equal.trans(Array<U64>,O.storage(Array.swap.hi(U64,O.pack(xs),Array.get.go(U64,O.pack(ys),h,U32.sub(i,h)))),
        ANode{O.pack(xs),O.storage(Array.get.go(U64,O.pack(ys),h,U32.sub(i,h)))},O.pack(O.Branch{xs,ys}),
        storage_hi(O.pack(xs),Array.get.go(U64,O.pack(ys),h,U32.sub(i,h))),
        Equal.cong(Array<U64>,Array<U64>,a => ANode{O.pack(xs),a},O.storage(Array.get.go(U64,O.pack(ys),h,U32.sub(i,h))),O.pack(ys),right))

def get_go(+c: O.Cells, +n: U32, +i: U32) ->
  {O.storage(Array.get.go(U64,O.pack(c),n,i)) == O.pack(c) : Array<U64>}:
  match c:
    case O.Cell{x}: {==}
    case O.Branch{xs,ys}:
      get_if(xs,ys,U32.shr(n),i,U32.is_lt(i,U32.shr(n)),
        get_go(xs,U32.shr(n),i),get_go(ys,U32.shr(n),U32.sub(i,U32.shr(n))))

def read(+c: O.Cells, +i: U32) ->
  {O.storage(Array.get(U64,O.pack(c),i)) == O.pack(c) : Array<U64>}:
  %Equal.sym(Array<U64> & U32,Array.size(U64,O.pack(c)),(O.pack(c),O.size(c)),size_pair(c)) :
    {O.storage(Array.get.at(U64,i,_)) == O.pack(c) : Array<U64>}
  get_go(c,O.size(c),U32.and(i,U32.sub(O.size(c),1)))


def lift(-a: Array<U64>, +i: U32, cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {O.storage(Array.get(U64,a,i)) == a : Array<U64>}:
  (+c,e) = cert
  +same: {O.pack(c) == a : Array<U64>} = e
  %same : {O.storage(Array.get(U64,_,i)) == a : Array<U64>}
  %same : {O.storage(Array.get(U64,O.pack(c),i)) == _ : Array<U64>}
  read(c,i)

def read_any(a: Array<U64>, +i: U32) ->
  {O.storage(Array.get(U64,a,i)) == a : Array<U64>}:
  lift(a,i,R.reify(a))
