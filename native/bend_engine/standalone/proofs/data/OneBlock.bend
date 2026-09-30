# Metadata precedes the actual fill; the completed fill supplies the entry.
import Base
import ./Spec.bend as S
import ../headers/Header.bend as H
import ../contents/Certified.bend as Contents
import ../prefix/Spec.bend as P
import ../layout/Spec.bend as Layout
import ../complete/Spec.bend as C
import ../storage/Observe.bend as O

def model(+c: O.Cells,+k: Nat,+i: Nat,valid: {Nat.is_lt(k,128n) == True{} : Bool},
  shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape},
  ib: {Nat.is_lt(i,U32.to_nat(Layout.size(U32.from_nat(k)))) == True{} : Bool}) ->
  {Array.get(U64,P.block(O.pack(c),k),S.query(k,i)) == (P.block(O.pack(c),k),S.expected(k,i)) : Array<U64> & U64}:
  Contents.block(H.headers(O.pack(c),k,P.prefix(k)),k,i,valid,H.shape(c,k,P.prefix(k),shape),ib)
