import Base
import ./Observe.bend as O
import ./Read.bend as Read
import ./Model.bend as M
import ./Refinement.bend as Ref
import ./Representation.bend as R

def set(+c: O.Cells, +i: U32, +v: U64) ->
  {Array.set(U64,O.pack(c),i,v) == O.pack(M.updated(c,i,v)) : Array<U64>}:
  %Equal.sym(Array<U64> & U32,Array.size(U64,O.pack(c)),(O.pack(c),O.size(c)),Read.size_pair(c)) :
    {Array.set.fin(U64,Array.swap.at(U64,i,v,_)) == O.pack(M.updated(c,i,v)) : Array<U64>}
  Ref.set_go(c,O.size(c),M.normalized(c,i),v)

def get(+c: O.Cells, +i: U32) ->
  {Array.get(U64,O.pack(c),i) == (O.pack(c),M.read_sized(c,O.size(c),i)) : Array<U64> & U64}:
  %Equal.sym(Array<U64> & U32,Array.size(U64,O.pack(c)),(O.pack(c),O.size(c)),Read.size_pair(c)) :
    {Array.get.at(U64,i,_) == (O.pack(c),M.read_sized(c,O.size(c),i)) : Array<U64> & U64}
  Ref.read_go(c,O.size(c),M.normalized(c,i))

def model_value(+c: O.Cells, +i: U32, +v: U64) ->
  {M.read_sized(M.updated(c,i,v),O.size(M.updated(c,i,v)),i) == v : U64}:
  %Equal.sym(U32,O.size(M.updated(c,i,v)),O.size(c),Ref.size_update(c,O.size(c),M.normalized(c,i),v)) :
    {M.read_sized(M.updated(c,i,v),_,i) == v : U64}
  Ref.model_read_update(c,O.size(c),M.normalized(c,i),v)

def model(+c: O.Cells, +i: U32, +v: U64) ->
  {Array.get(U64,Array.set(U64,O.pack(c),i,v),i) == (Array.set(U64,O.pack(c),i,v),v) : Array<U64> & U64}:
  +stored = set(c,i,v)
  %Equal.sym(Array<U64>,Array.set(U64,O.pack(c),i,v),O.pack(M.updated(c,i,v)),stored) :
    {Array.get(U64,_,i) == (Array.set(U64,O.pack(c),i,v),v) : Array<U64> & U64}
  %Equal.sym(Array<U64>,Array.set(U64,O.pack(c),i,v),O.pack(M.updated(c,i,v)),stored) :
    {Array.get(U64,O.pack(M.updated(c,i,v)),i) == (_,v) : Array<U64> & U64}
  Equal.trans(Array<U64> & U64,Array.get(U64,O.pack(M.updated(c,i,v)),i),
    (O.pack(M.updated(c,i,v)),M.read_sized(M.updated(c,i,v),O.size(M.updated(c,i,v)),i)),
    (O.pack(M.updated(c,i,v)),v),get(M.updated(c,i,v),i),
    Equal.cong(U64,Array<U64> & U64,x => (O.pack(M.updated(c,i,v)),x),
      M.read_sized(M.updated(c,i,v),O.size(M.updated(c,i,v)),i),v,model_value(c,i,v)))

def lift(-a: Array<U64>, +i: U32, +v: U64,
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {Array.get(U64,Array.set(U64,a,i,v),i) == (Array.set(U64,a,i,v),v) : Array<U64> & U64}:
  (+c,e) = cert
  +same: {O.pack(c) == a : Array<U64>} = e
  %same : {Array.get(U64,Array.set(U64,_,i,v),i) == (Array.set(U64,a,i,v),v) : Array<U64> & U64}
  %same : {Array.get(U64,Array.set(U64,O.pack(c),i,v),i) == (Array.set(U64,_,i,v),v) : Array<U64> & U64}
  model(c,i,v)

def write_read(a: Array<U64>, +i: U32, +v: U64) ->
  {Array.get(U64,Array.set(U64,a,i,v),i) == (Array.set(U64,a,i,v),v) : Array<U64> & U64}:
  lift(a,i,v,R.reify(a))
