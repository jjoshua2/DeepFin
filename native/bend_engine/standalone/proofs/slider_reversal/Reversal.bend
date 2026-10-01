import Base
import ./Spec.bend as S
import ./Booleans.bend as B
import ./Directed.bend as D
import ./Reflect.bend as Reflect
import ./Computed.bend as Computed

def rook(+src: Nat,+dst: Nat,+occ: U64,+sb: {Nat.is_lt(src,64n) == True{} : Bool},+tb: {Nat.is_lt(dst,64n) == True{} : Bool}) ->
  {S.answer(S.Rook{},S.paths(src,dst),occ) == S.answer(S.Rook{},S.paths(dst,src),occ) : Bool}:
  Equal.trans(Bool,S.answer(S.Rook{},S.paths(src,dst),occ),
    S.four(S.edge(dst,src,1n,occ),S.edge(dst,src,0n,occ),S.edge(dst,src,3n,occ),S.edge(dst,src,2n,occ)),S.answer(S.Rook{},S.paths(dst,src),occ),
    B.four_cong(S.edge(src,dst,0n,occ),S.edge(src,dst,1n,occ),S.edge(src,dst,2n,occ),S.edge(src,dst,3n,occ),
      S.edge(dst,src,1n,occ),S.edge(dst,src,0n,occ),S.edge(dst,src,3n,occ),S.edge(dst,src,2n,occ),
      D.bools(src,dst,0n,occ,sb,tb,{==}),D.bools(src,dst,1n,occ,sb,tb,{==}),D.bools(src,dst,2n,occ,sb,tb,{==}),D.bools(src,dst,3n,occ,sb,tb,{==})),
    B.four_swap(S.edge(dst,src,0n,occ),S.edge(dst,src,1n,occ),S.edge(dst,src,2n,occ),S.edge(dst,src,3n,occ)))

def bishop(+src: Nat,+dst: Nat,+occ: U64,+sb: {Nat.is_lt(src,64n) == True{} : Bool},+tb: {Nat.is_lt(dst,64n) == True{} : Bool}) ->
  {S.answer(S.Bishop{},S.paths(src,dst),occ) == S.answer(S.Bishop{},S.paths(dst,src),occ) : Bool}:
  Equal.trans(Bool,S.answer(S.Bishop{},S.paths(src,dst),occ),
    S.four(S.edge(dst,src,7n,occ),S.edge(dst,src,6n,occ),S.edge(dst,src,5n,occ),S.edge(dst,src,4n,occ)),S.answer(S.Bishop{},S.paths(dst,src),occ),
    B.four_cong(S.edge(src,dst,4n,occ),S.edge(src,dst,5n,occ),S.edge(src,dst,6n,occ),S.edge(src,dst,7n,occ),
      S.edge(dst,src,7n,occ),S.edge(dst,src,6n,occ),S.edge(dst,src,5n,occ),S.edge(dst,src,4n,occ),
      D.bools(src,dst,4n,occ,sb,tb,{==}),D.bools(src,dst,5n,occ,sb,tb,{==}),D.bools(src,dst,6n,occ,sb,tb,{==}),D.bools(src,dst,7n,occ,sb,tb,{==})),
    B.four_reverse(S.edge(dst,src,4n,occ),S.edge(dst,src,5n,occ),S.edge(dst,src,6n,occ),S.edge(dst,src,7n,occ)))

def family(+kind: S.Slider,+src: Nat,+dst: Nat,+occ: U64,+sb: {Nat.is_lt(src,64n) == True{} : Bool},+tb: {Nat.is_lt(dst,64n) == True{} : Bool}) ->
  {S.answer(kind,S.paths(src,dst),occ) == S.answer(kind,S.paths(dst,src),occ) : Bool}:
  match kind:
    case S.Rook{}: rook(src,dst,occ,sb,tb)
    case S.Bishop{}: bishop(src,dst,occ,sb,tb)
    case S.Queen{}:
      B.or_cong(S.answer(S.Rook{},S.paths(src,dst),occ),S.answer(S.Bishop{},S.paths(src,dst),occ),
        S.answer(S.Rook{},S.paths(dst,src),occ),S.answer(S.Bishop{},S.paths(dst,src),occ),
        rook(src,dst,occ,sb,tb),bishop(src,dst,occ,sb,tb))

def geometry(+kind: S.Slider,+src: Nat,+dst: Nat,+occ: U64,+sb: {Nat.is_lt(src,64n) == True{} : Bool},+tb: {Nat.is_lt(dst,64n) == True{} : Bool}) ->
  {U64.test_bit(S.geometry(kind,src,occ),dst) == U64.test_bit(S.geometry(kind,dst,occ),src) : Bool}:
  Equal.trans(Bool,U64.test_bit(S.geometry(kind,src,occ),dst),S.answer(kind,S.paths(src,dst),occ),U64.test_bit(S.geometry(kind,dst,occ),src),
    Reflect.geometry(kind,src,dst,occ,tb),
    Equal.trans(Bool,S.answer(kind,S.paths(src,dst),occ),S.answer(kind,S.paths(dst,src),occ),U64.test_bit(S.geometry(kind,dst,occ),src),
      family(kind,src,dst,occ,sb,tb),
      Equal.sym(Bool,U64.test_bit(S.geometry(kind,dst,occ),src),S.answer(kind,S.paths(dst,src),occ),Reflect.geometry(kind,dst,src,occ,sb))))

def computed(+kind: S.Slider,+src: Nat,+dst: Nat,+occ: U64,+sb: {Nat.is_lt(src,64n) == True{} : Bool},+tb: {Nat.is_lt(dst,64n) == True{} : Bool}) ->
  {U64.test_bit(S.computed(kind,src,occ),dst) == U64.test_bit(S.computed(kind,dst,occ),src) : Bool}:
  Equal.trans(Bool,U64.test_bit(S.computed(kind,src,occ),dst),U64.test_bit(S.geometry(kind,src,occ),dst),U64.test_bit(S.computed(kind,dst,occ),src),
    Equal.cong(U64,Bool,m => U64.test_bit(m,dst),S.computed(kind,src,occ),S.geometry(kind,src,occ),Computed.computed(kind,src,occ,sb)),
    Equal.trans(Bool,U64.test_bit(S.geometry(kind,src,occ),dst),U64.test_bit(S.geometry(kind,dst,occ),src),U64.test_bit(S.computed(kind,dst,occ),src),
      geometry(kind,src,dst,occ,sb,tb),
      Equal.cong(U64,Bool,m => U64.test_bit(m,src),S.geometry(kind,dst,occ),S.computed(kind,dst,occ),
        Equal.sym(U64,S.computed(kind,dst,occ),S.geometry(kind,dst,occ),Computed.computed(kind,dst,occ,tb)))))
