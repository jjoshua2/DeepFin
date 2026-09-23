import Base
import ./Facts.bend as Facts
import ./Spec.bend as S
import ./Primitive.bend as P
import ../../Tables.bend as Tables
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../storage/Build.bend as Build
import ../storage/Write.bend as Write
import ../separation/Spec.bend as Sep
import ../complete/Spec.bend as C

def lift_write(-a: Array<U64>,+i: U32,+q: U32,+v: U64,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  ib: Facts.bounds(i),qb: {U32.is_lt(q,131072) == True{} : Bool},e: {S.outside(q) == True{} : Bool},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {Sep.value(Array.get(U64,Array.set(U64,a,i,v),q)) == Sep.value(Array.get(U64,a,q)) : U64}:
  (+c,eq) = cert
  +same: {O.pack(c) == a : Array<U64>} = eq
  %same : {Sep.value(Array.get(U64,Array.set(U64,_,i,v),q)) == Sep.value(Array.get(U64,a,q)) : U64}
  %same : {Sep.value(Array.get(U64,Array.set(U64,O.pack(c),i,v),q)) == Sep.value(Array.get(U64,_,q)) : U64}
  P.write(c,i,q,v,Equal.trans(O.Shape,O.shape(O.pack(c)),O.shape(a),C.full(17n),
    Equal.cong(Array<U64>,O.Shape,x => O.shape(x),O.pack(c),a,same),shape),ib,qb,e)

def write(a: Array<U64>,+i: U32,+q: U32,+v: U64,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  ib: Facts.bounds(i),qb: {U32.is_lt(q,131072) == True{} : Bool},e: {S.outside(q) == True{} : Bool}) ->
  {Sep.value(Array.get(U64,Array.set(U64,a,i,v),q)) == Sep.value(Array.get(U64,a,q)) : U64}:
  lift_write(a,i,q,v,shape,ib,qb,e,R.reify(a))

def knight_shape(+c: O.Cells,+sq: U32,shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape}) ->
  {O.shape(Build.knight(c,sq)) == C.full(17n) : O.Shape}:
  Equal.trans(O.Shape,O.shape(Build.knight(c,sq)),O.shape(O.pack(c)),C.full(17n),
    Write.model(c,U32.add(256,sq),Tables.leaps(8n,sq,0,False{},U64.zero())),shape)

def king_shape(+c: O.Cells,+sq: U32,shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape}) ->
  {O.shape(Build.king(c,sq)) == C.full(17n) : O.Shape}:
  Equal.trans(O.Shape,O.shape(Build.king(c,sq)),O.shape(Build.knight(c,sq)),C.full(17n),
    Write.write(Build.knight(c,sq),U32.add(320,sq),Tables.leaps(8n,sq,0,True{},U64.zero())),knight_shape(c,sq,shape))

def white_shape(+c: O.Cells,+sq: U32,shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape}) ->
  {O.shape(Build.white(c,sq)) == C.full(17n) : O.Shape}:
  Equal.trans(O.Shape,O.shape(Build.white(c,sq)),O.shape(Build.king(c,sq)),C.full(17n),
    Write.write(Build.king(c,sq),U32.add(384,sq),Tables.pawns(sq,1)),king_shape(c,sq,shape))

def frame(+c: O.Cells,+k: Nat,+q: U32,
  +shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape},
  facts: Facts.step(k),+qb: {U32.is_lt(q,131072) == True{} : Bool},+e: {S.outside(q) == True{} : Bool}) ->
  {Sep.value(Array.get(U64,Build.black(c,U32.from_nat(k)),q)) == Sep.value(Array.get(U64,O.pack(c),q)) : U64}:
  (inc,(b0,(b1,(b2,b3)))) = facts
  +sq = U32.from_nat(k)
  Equal.trans(U64,Sep.value(Array.get(U64,Build.black(c,sq),q)),Sep.value(Array.get(U64,Build.white(c,sq),q)),Sep.value(Array.get(U64,O.pack(c),q)),
    write(Build.white(c,sq),U32.add(448,sq),q,Tables.pawns(sq,4294967295),white_shape(c,sq,shape),b3,qb,e),
    Equal.trans(U64,Sep.value(Array.get(U64,Build.white(c,sq),q)),Sep.value(Array.get(U64,Build.king(c,sq),q)),Sep.value(Array.get(U64,O.pack(c),q)),
      write(Build.king(c,sq),U32.add(384,sq),q,Tables.pawns(sq,1),king_shape(c,sq,shape),b2,qb,e),
      Equal.trans(U64,Sep.value(Array.get(U64,Build.king(c,sq),q)),Sep.value(Array.get(U64,Build.knight(c,sq),q)),Sep.value(Array.get(U64,O.pack(c),q)),
        write(Build.knight(c,sq),U32.add(320,sq),q,Tables.leaps(8n,sq,0,True{},U64.zero()),knight_shape(c,sq,shape),b1,qb,e),
        P.write(c,U32.add(256,sq),q,Tables.leaps(8n,sq,0,False{},U64.zero()),shape,b0,qb,e))))
