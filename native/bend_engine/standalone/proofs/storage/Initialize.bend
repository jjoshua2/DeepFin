import Base
import ./Observe.bend as O
import ./Model.bend as M
import ./Roundtrip.bend as Roundtrip

def uniform(d: Nat, +v: U64) -> O.Cells:
  match d:
    case 0n: O.Cell{v}
    case 1n+p:
      +c = uniform(p,v)
      O.Branch{c,c}

def pack(+d: Nat, +v: U64) -> {Array.new(U64,d,v) == O.pack(uniform(d,v)) : Array<U64>}:
  match d:
    case 0n: {==}
    case 1n+p:
      +e = pack(p,v)
      Equal.trans(Array<U64>,ANode{Array.new(U64,p,v),Array.new(U64,p,v)},
        ANode{O.pack(uniform(p,v)),Array.new(U64,p,v)},O.pack(uniform(1n+p,v)),
        Equal.cong(Array<U64>,Array<U64>,a => ANode{a,Array.new(U64,p,v)},Array.new(U64,p,v),O.pack(uniform(p,v)),e),
        Equal.cong(Array<U64>,Array<U64>,a => ANode{O.pack(uniform(p,v)),a},Array.new(U64,p,v),O.pack(uniform(p,v)),e))

def pick(z: Bool, +v: U64) -> {Bool.pick(U64,z,v,v) == v : U64}:
  match z:
    case False{}: {==}
    case True{}: {==}

def value(+d: Nat, +v: U64, +n: U32, +i: U32) ->
  {M.read(uniform(d,v),n,i) == v : U64}:
  match d:
    case 0n: {==}
    case 1n+p:
      %Equal.sym(U64,M.read(uniform(p,v),U32.shr(n),i),v,value(p,v,U32.shr(n),i)) :
        {Bool.pick(U64,U32.is_lt(i,U32.shr(n)),_,M.read(uniform(p,v),U32.shr(n),U32.sub(i,U32.shr(n)))) == v : U64}
      %Equal.sym(U64,M.read(uniform(p,v),U32.shr(n),U32.sub(i,U32.shr(n))),v,value(p,v,U32.shr(n),U32.sub(i,U32.shr(n)))) :
        {Bool.pick(U64,U32.is_lt(i,U32.shr(n)),v,_) == v : U64}
      pick(U32.is_lt(i,U32.shr(n)),v)

def read(+d: Nat, +v: U64, +i: U32) ->
  {Array.get(U64,Array.new(U64,d,v),i) == (Array.new(U64,d,v),v) : Array<U64> & U64}:
  +same = pack(d,v)
  %Equal.sym(Array<U64>,Array.new(U64,d,v),O.pack(uniform(d,v)),same) :
    {Array.get(U64,_,i) == (Array.new(U64,d,v),v) : Array<U64> & U64}
  %Equal.sym(Array<U64>,Array.new(U64,d,v),O.pack(uniform(d,v)),same) :
    {Array.get(U64,O.pack(uniform(d,v)),i) == (_,v) : Array<U64> & U64}
  Equal.trans(Array<U64> & U64,Array.get(U64,O.pack(uniform(d,v)),i),
    (O.pack(uniform(d,v)),M.read_sized(uniform(d,v),O.size(uniform(d,v)),i)),(O.pack(uniform(d,v)),v),
    Roundtrip.get(uniform(d,v),i),
    Equal.cong(U64,Array<U64> & U64,x => (O.pack(uniform(d,v)),x),
      M.read_sized(uniform(d,v),O.size(uniform(d,v)),i),v,
      value(d,v,O.size(uniform(d,v)),M.normalized(uniform(d,v),i))))
