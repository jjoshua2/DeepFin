# Proof-only first-component projection and affine-to-Data transport.
import Base
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R

def table(-V: Type,r: Array<U64> & V) -> Array<U64>:
  (a,v) = r
  a

def lift(-V: Type,-f: @a: Array<U64> -> (Array<U64> & V),-a: Array<U64>,
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>}),
  proof: @c: O.Cells -> {table(V,f(O.pack(c))) == O.pack(c) : Array<U64>}) ->
  {table(V,f(a)) == a : Array<U64>}:
  (+c,e) = cert
  +same: {O.pack(c) == a : Array<U64>} = e
  %same : {table(V,f(_)) == a : Array<U64>}
  %same : {table(V,f(O.pack(c))) == _ : Array<U64>}
  proof(c)

def apply(-V: Type,-f: @a: Array<U64> -> (Array<U64> & V),a: Array<U64>,
  proof: @c: O.Cells -> {table(V,f(O.pack(c))) == O.pack(c) : Array<U64>}) ->
  {table(V,f(a)) == a : Array<U64>}:
  lift(V,f,a,R.reify(a),proof)
