# Actual four writes, own-value correctness and other-square frame.
import Base
import ./Geometry.bend as G
import ./Slots.bend as S
import ./Facts.bend as F
import ../storage/Observe.bend as O
import ../storage/Write.bend as Write
import ../storage/Roundtrip.bend as Round
import ../storage/Build.bend as Build
import ../complete/Spec.bend as C
import ../complete/Actual.bend as Complete
import ../separation/Spec.bend as V

def read(a: Array<U64>,q: U32) -> U64:
  V.value(Array.get(U64,a,q))

def frame(a: Array<U64>,+kind: G.Leaper,+sq: Nat,+query: G.Leaper,+at: Nat,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  +sb: {Nat.is_lt(sq,64n) == True{} : Bool},+qb: {Nat.is_lt(at,64n) == True{} : Bool},
  ne: {S.overlaps(kind,query,sq,at) == False{} : Bool}) ->
  {read(Array.set(U64,a,S.index(kind,sq),S.value(kind,sq)),S.index(query,at)) == read(a,S.index(query,at)) : U64}:
  Equal.cong(Array<U64> & U64,U64,x => V.value(x),
    Array.get(U64,Array.set(U64,a,S.index(kind,sq),S.value(kind,sq)),S.index(query,at)),
    (Array.set(U64,a,S.index(kind,sq),S.value(kind,sq)),read(a,S.index(query,at))),
    Complete.frame(a,S.index(kind,sq),S.index(query,at),S.value(kind,sq),shape,
      F.bounded(kind,sq,F.fixed(kind,sq,sb)),F.bounded(query,at,F.fixed(query,at,qb)),F.distinct(kind,query,sq,at,sb,qb,ne)))

def same(a: Array<U64>,+kind: G.Leaper,+sq: Nat) ->
  {read(Array.set(U64,a,S.index(kind,sq),S.value(kind,sq)),S.index(kind,sq)) == S.value(kind,sq) : U64}:
  Equal.cong(Array<U64> & U64,U64,x => V.value(x),
    Array.get(U64,Array.set(U64,a,S.index(kind,sq),S.value(kind,sq)),S.index(kind,sq)),
    (Array.set(U64,a,S.index(kind,sq),S.value(kind,sq)),S.value(kind,sq)),
    Round.write_read(a,S.index(kind,sq),S.value(kind,sq)))

def stage0(+c: O.Cells,+sq: Nat) -> Array<U64>:
  O.pack(c)

def stage1(+c: O.Cells,+sq: Nat) -> Array<U64>:
  Array.set(U64,stage0(c,sq),S.index(G.Knight{},sq),S.value(G.Knight{},sq))

def stage2(+c: O.Cells,+sq: Nat) -> Array<U64>:
  Array.set(U64,stage1(c,sq),S.index(G.King{},sq),S.value(G.King{},sq))

def stage3(+c: O.Cells,+sq: Nat) -> Array<U64>:
  Array.set(U64,stage2(c,sq),S.index(G.WhitePawn{},sq),S.value(G.WhitePawn{},sq))

def stage4(+c: O.Cells,+sq: Nat) -> Array<U64>:
  Array.set(U64,stage3(c,sq),S.index(G.BlackPawn{},sq),S.value(G.BlackPawn{},sq))

def sh0(+c: O.Cells,+sq: Nat,sh: {O.shape(O.pack(c)) == C.full(17n) : O.Shape}) ->
  {O.shape(stage0(c,sq)) == C.full(17n) : O.Shape}:
  sh

def sh1(+c: O.Cells,+sq: Nat,sh: {O.shape(O.pack(c)) == C.full(17n) : O.Shape}) ->
  {O.shape(stage1(c,sq)) == C.full(17n) : O.Shape}:
  Equal.trans(O.Shape,O.shape(stage1(c,sq)),O.shape(stage0(c,sq)),C.full(17n),
    Write.write(stage0(c,sq),S.index(G.Knight{},sq),S.value(G.Knight{},sq)),sh0(c,sq,sh))

def sh2(+c: O.Cells,+sq: Nat,sh: {O.shape(O.pack(c)) == C.full(17n) : O.Shape}) ->
  {O.shape(stage2(c,sq)) == C.full(17n) : O.Shape}:
  Equal.trans(O.Shape,O.shape(stage2(c,sq)),O.shape(stage1(c,sq)),C.full(17n),
    Write.write(stage1(c,sq),S.index(G.King{},sq),S.value(G.King{},sq)),sh1(c,sq,sh))

def sh3(+c: O.Cells,+sq: Nat,sh: {O.shape(O.pack(c)) == C.full(17n) : O.Shape}) ->
  {O.shape(stage3(c,sq)) == C.full(17n) : O.Shape}:
  Equal.trans(O.Shape,O.shape(stage3(c,sq)),O.shape(stage2(c,sq)),C.full(17n),
    Write.write(stage2(c,sq),S.index(G.WhitePawn{},sq),S.value(G.WhitePawn{},sq)),sh2(c,sq,sh))

def sh4(+c: O.Cells,+sq: Nat,sh: {O.shape(O.pack(c)) == C.full(17n) : O.Shape}) ->
  {O.shape(stage4(c,sq)) == C.full(17n) : O.Shape}:
  Equal.trans(O.Shape,O.shape(stage4(c,sq)),O.shape(stage3(c,sq)),C.full(17n),
    Write.write(stage3(c,sq),S.index(G.BlackPawn{},sq),S.value(G.BlackPawn{},sq)),sh3(c,sq,sh))

def actual_block(+c: O.Cells,+sq: Nat) ->
  {S.block(O.pack(c),sq) == Build.black(c,U32.from_nat(sq)) : Array<U64>}:
  {==}

def own(+c: O.Cells,+sq: Nat,+kind: G.Leaper,
  +shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape},
  +bound: {Nat.is_lt(sq,64n) == True{} : Bool}) ->
  {read(S.block(O.pack(c),sq),S.index(kind,sq)) == S.value(kind,sq) : U64}:
  match kind:
    case G.Knight{}: Equal.trans(U64,read(stage4(c,sq),S.index(G.Knight{},sq)),read(stage3(c,sq),S.index(G.Knight{},sq)),S.value(G.Knight{},sq),frame(stage3(c,sq),G.BlackPawn{},sq,G.Knight{},sq,sh3(c,sq,shape),bound,bound,{==}),Equal.trans(U64,read(stage3(c,sq),S.index(G.Knight{},sq)),read(stage2(c,sq),S.index(G.Knight{},sq)),S.value(G.Knight{},sq),frame(stage2(c,sq),G.WhitePawn{},sq,G.Knight{},sq,sh2(c,sq,shape),bound,bound,{==}),Equal.trans(U64,read(stage2(c,sq),S.index(G.Knight{},sq)),read(stage1(c,sq),S.index(G.Knight{},sq)),S.value(G.Knight{},sq),frame(stage1(c,sq),G.King{},sq,G.Knight{},sq,sh1(c,sq,shape),bound,bound,{==}),same(stage0(c,sq),G.Knight{},sq))))
    case G.King{}: Equal.trans(U64,read(stage4(c,sq),S.index(G.King{},sq)),read(stage3(c,sq),S.index(G.King{},sq)),S.value(G.King{},sq),frame(stage3(c,sq),G.BlackPawn{},sq,G.King{},sq,sh3(c,sq,shape),bound,bound,{==}),Equal.trans(U64,read(stage3(c,sq),S.index(G.King{},sq)),read(stage2(c,sq),S.index(G.King{},sq)),S.value(G.King{},sq),frame(stage2(c,sq),G.WhitePawn{},sq,G.King{},sq,sh2(c,sq,shape),bound,bound,{==}),same(stage1(c,sq),G.King{},sq)))
    case G.WhitePawn{}: Equal.trans(U64,read(stage4(c,sq),S.index(G.WhitePawn{},sq)),read(stage3(c,sq),S.index(G.WhitePawn{},sq)),S.value(G.WhitePawn{},sq),frame(stage3(c,sq),G.BlackPawn{},sq,G.WhitePawn{},sq,sh3(c,sq,shape),bound,bound,{==}),same(stage2(c,sq),G.WhitePawn{},sq))
    case G.BlackPawn{}: same(stage3(c,sq),G.BlackPawn{},sq)

def other(+c: O.Cells,+sq: Nat,+kind: G.Leaper,+at: Nat,
  +shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape},
  +sb: {Nat.is_lt(sq,64n) == True{} : Bool},+qb: {Nat.is_lt(at,64n) == True{} : Bool},
  +ne: {Nat.is_eq(sq,at) == False{} : Bool}) ->
  {read(S.block(O.pack(c),sq),S.index(kind,at)) == read(O.pack(c),S.index(kind,at)) : U64}:
  Equal.trans(U64,read(stage4(c,sq),S.index(kind,at)),read(stage3(c,sq),S.index(kind,at)),read(O.pack(c),S.index(kind,at)),frame(stage3(c,sq),G.BlackPawn{},sq,kind,at,sh3(c,sq,shape),sb,qb,F.other_square(G.BlackPawn{},kind,sq,at,ne)),Equal.trans(U64,read(stage3(c,sq),S.index(kind,at)),read(stage2(c,sq),S.index(kind,at)),read(O.pack(c),S.index(kind,at)),frame(stage2(c,sq),G.WhitePawn{},sq,kind,at,sh2(c,sq,shape),sb,qb,F.other_square(G.WhitePawn{},kind,sq,at,ne)),Equal.trans(U64,read(stage2(c,sq),S.index(kind,at)),read(stage1(c,sq),S.index(kind,at)),read(O.pack(c),S.index(kind,at)),frame(stage1(c,sq),G.King{},sq,kind,at,sh1(c,sq,shape),sb,qb,F.other_square(G.King{},kind,sq,at,ne)),frame(stage0(c,sq),G.Knight{},sq,kind,at,sh0(c,sq,shape),sb,qb,F.other_square(G.Knight{},kind,sq,at,ne)))))
