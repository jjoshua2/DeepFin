# Lift pointwise reversal through the complete two-limb witness representation.
import Base
import ./Spec.bend as S
import ./Sample.bend as Sample
import ./Index.bend as Index
import ./Pointwise.bend as Point
import ./Packing.bend as Pack
import ../decoder/Bits.bend as Bit
import ../prefix/Nats.bend as N
import ../attack_witness/Spec.bend as W
import ../initialized_attacks/Spec.bend as I

def word(+n: Nat,+i: Nat,+high: Bool,+kind: S.Kind,+dst: Nat,+by: Bool,+occ: U64,
  +budget: {Nat.is_lt(Nat.add(n,i),33n) == True{} : Bool},+tb: {Nat.is_lt(dst,64n) == True{} : Bool}) ->
  {Sample.word(32n,n,i,Index.limb(Point.reversed(kind,dst,by,occ),high)) == S.word(n,Index.source(high,i),kind,dst,by,occ) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+t:
      +a = Point.reversed(kind,dst,by,occ)
      +ib = N.strip(t,i,32n,budget)
      Sample.cons(t,Bit.get(32n,Index.limb(a,high),i),S.edge(kind,Index.source(high,i),dst,by,occ),
        Sample.word(32n,t,1n+i,Index.limb(a,high)),S.word(t,1n+Index.source(high,i),kind,dst,by,occ),
        Equal.trans(Bool,Bit.get(32n,Index.limb(a,high),i),Bit.get64(a,Index.source(high,i)),S.edge(kind,Index.source(high,i),dst,by,occ),
          Index.get(a,high,i,ib),Point.edge(kind,Index.source(high,i),dst,by,occ,Index.bounded(high,i,ib),tb)),
        Equal.trans(Word(t),Sample.word(32n,t,1n+i,Index.limb(a,high)),S.word(t,Index.source(high,1n+i),kind,dst,by,occ),S.word(t,1n+Index.source(high,i),kind,dst,by,occ),
          word(t,1n+i,high,kind,dst,by,occ,Index.next_bound(t,i,budget),tb),
          Equal.cong(Nat,Word(t),src => S.word(t,src,kind,dst,by,occ),Index.source(high,1n+i),1n+Index.source(high,i),Index.advance(high,i))))

def limb(+high: Bool,+kind: S.Kind,+dst: Nat,+by: Bool,+occ: U64,tb: {Nat.is_lt(dst,64n) == True{} : Bool}) ->
  {Index.limb(Point.reversed(kind,dst,by,occ),high) == S.word(32n,Index.source(high,0n),kind,dst,by,occ) : Word(32n)}:
  +a = Index.limb(Point.reversed(kind,dst,by,occ),high)
  Equal.trans(Word(32n),a,Sample.word(32n,32n,0n,a),S.word(32n,Index.source(high,0n),kind,dst,by,occ),
    Equal.sym(Word(32n),Sample.word(32n,32n,0n,a),a,Sample.identity(32n,a)),
    word(32n,0n,high,kind,dst,by,occ,{==},tb))

def finish(+a: U64,+lo: Word(32n),+hi: Word(32n),
  low: {Index.limb(a,False{}) == lo : Word(32n)},high: {Index.limb(a,True{}) == hi : Word(32n)}) ->
  {a == U64{U32{lo},U32{hi}} : U64}:
  match a:
    case U64{U32{+al},U32{+ah}}:
      Equal.trans(U64,U64{U32{al},U32{ah}},U64{U32{lo},U32{ah}},U64{U32{lo},U32{hi}},
        Equal.cong(Word(32n),U64,w => U64{U32{w},U32{ah}},al,lo,low),
        Equal.cong(Word(32n),U64,w => U64{U32{lo},U32{w}},ah,hi,high))

def mask(+kind: S.Kind,+dst: Nat,+by: Bool,+occ: U64,+tb: {Nat.is_lt(dst,64n) == True{} : Bool}) ->
  {Point.reversed(kind,dst,by,occ) == Pack.mask(kind,dst,by,occ) : U64}:
  finish(Point.reversed(kind,dst,by,occ),S.word(32n,0n,kind,dst,by,occ),S.word(32n,32n,kind,dst,by,occ),
    limb(False{},kind,dst,by,occ,tb),limb(True{},kind,dst,by,occ,tb))

def masks(+dst: Nat,+by: Bool,+occ: U64,+tb: {Nat.is_lt(dst,64n) == True{} : Bool}) ->
  {I.masks(dst,by,occ) == Pack.masks(dst,by,occ) : W.Masks}:
  %Equal.sym(U64,Point.reversed(S.Pawn{},dst,by,occ),Pack.mask(S.Pawn{},dst,by,occ),mask(S.Pawn{},dst,by,occ,tb)) :
    {W.Masks{_,Point.reversed(S.Knight{},dst,by,occ),Point.reversed(S.King{},dst,by,occ),Point.reversed(S.Rook{},dst,by,occ),Point.reversed(S.Bishop{},dst,by,occ)} == Pack.masks(dst,by,occ) : W.Masks}
  %Equal.sym(U64,Point.reversed(S.Knight{},dst,by,occ),Pack.mask(S.Knight{},dst,by,occ),mask(S.Knight{},dst,by,occ,tb)) :
    {W.Masks{Pack.mask(S.Pawn{},dst,by,occ),_,Point.reversed(S.King{},dst,by,occ),Point.reversed(S.Rook{},dst,by,occ),Point.reversed(S.Bishop{},dst,by,occ)} == Pack.masks(dst,by,occ) : W.Masks}
  %Equal.sym(U64,Point.reversed(S.King{},dst,by,occ),Pack.mask(S.King{},dst,by,occ),mask(S.King{},dst,by,occ,tb)) :
    {W.Masks{Pack.mask(S.Pawn{},dst,by,occ),Pack.mask(S.Knight{},dst,by,occ),_,Point.reversed(S.Rook{},dst,by,occ),Point.reversed(S.Bishop{},dst,by,occ)} == Pack.masks(dst,by,occ) : W.Masks}
  %Equal.sym(U64,Point.reversed(S.Rook{},dst,by,occ),Pack.mask(S.Rook{},dst,by,occ),mask(S.Rook{},dst,by,occ,tb)) :
    {W.Masks{Pack.mask(S.Pawn{},dst,by,occ),Pack.mask(S.Knight{},dst,by,occ),Pack.mask(S.King{},dst,by,occ),_,Point.reversed(S.Bishop{},dst,by,occ)} == Pack.masks(dst,by,occ) : W.Masks}
  %Equal.sym(U64,Point.reversed(S.Bishop{},dst,by,occ),Pack.mask(S.Bishop{},dst,by,occ),mask(S.Bishop{},dst,by,occ,tb)) :
    {W.Masks{Pack.mask(S.Pawn{},dst,by,occ),Pack.mask(S.Knight{},dst,by,occ),Pack.mask(S.King{},dst,by,occ),Pack.mask(S.Rook{},dst,by,occ),_} == Pack.masks(dst,by,occ) : W.Masks}
  {==}
