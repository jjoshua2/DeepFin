# Consume an affine array exactly once and retain a complete Data image + proof.
# This is a proof witness, never called by the engine or used as an oracle.
import Base
import ./Observe.bend as O

def node(-xs: Array<U64>, -ys: Array<U64>,
  lx: Exists(O.Cells,c => {O.pack(c) == xs : Array<U64>}),
  ly: Exists(O.Cells,c => {O.pack(c) == ys : Array<U64>})) ->
  Exists(O.Cells,c => {O.pack(c) == ANode{xs,ys} : Array<U64>}):
  (+cx,ex) = lx
  (+cy,ey) = ly
  (O.Branch{cx,cy},Equal.trans(Array<U64>,ANode{O.pack(cx),O.pack(cy)},ANode{xs,O.pack(cy)},ANode{xs,ys},
    Equal.cong(Array<U64>,Array<U64>,a => ANode{a,O.pack(cy)},O.pack(cx),xs,ex),
    Equal.cong(Array<U64>,Array<U64>,a => ANode{xs,a},O.pack(cy),ys,ey)))

def reify(a: Array<U64>) -> Exists(O.Cells,c => {O.pack(c) == a : Array<U64>}):
  match a:
    case ALeaf{+x}: (O.Cell{x},{==})
    case ANode{xs,ys}: node(xs,ys,reify(xs),reify(ys))

# Lift a live proof on constructor images to an arbitrary affine array. The
# original storage is consumed only by reify; equality transport is erased.
def shape_transfer(-a: Array<U64>, -f: Array<U64> -> Array<U64>,
  rule: @+c: O.Cells -> {O.shape(f(O.pack(c))) == O.shape(O.pack(c)) : O.Shape},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {O.shape(f(a)) == O.shape(a) : O.Shape}:
  (+c,e) = cert
  +same: {O.pack(c) == a : Array<U64>} = e
  %same : {O.shape(f(_)) == O.shape(a) : O.Shape}
  %same : {O.shape(f(O.pack(c))) == O.shape(_) : O.Shape}
  rule(c)
