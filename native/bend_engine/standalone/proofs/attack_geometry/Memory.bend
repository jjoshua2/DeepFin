# Proof-only affine transport. Reify consumes the actual array once.
import Base
import ../storage/Observe.bend as O
import ../complete/Spec.bend as C

def equal(-T: Type,-a: Array<U64>,-left: Array<U64> -> T,-right: Array<U64> -> T,
  rule: @+c: O.Cells -> {left(O.pack(c)) == right(O.pack(c)) : T},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {left(a) == right(a) : T}:
  (+c,e) = cert
  +same: {O.pack(c) == a : Array<U64>} = e
  %same : {left(_) == right(a) : T}
  %same : {left(O.pack(c)) == right(_) : T}
  rule(c)

def shaped(-T: Type,-a: Array<U64>,-left: Array<U64> -> T,-right: Array<U64> -> T,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  rule: @+c: O.Cells -> {O.shape(O.pack(c)) == C.full(17n) : O.Shape} ->
    {left(O.pack(c)) == right(O.pack(c)) : T},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {left(a) == right(a) : T}:
  (+c,e) = cert
  +same: {O.pack(c) == a : Array<U64>} = e
  %same : {left(_) == right(a) : T}
  %same : {left(O.pack(c)) == right(_) : T}
  rule(c)(Equal.trans(O.Shape,O.shape(O.pack(c)),O.shape(a),C.full(17n),
    Equal.cong(Array<U64>,O.Shape,x => O.shape(x),O.pack(c),a,same),shape))
