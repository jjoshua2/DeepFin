import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Spec.bend as Spec
import ../../Tables.bend as Tables
import ../complete/Spec.bend as C
import ../storage/Observe.bend as O
import ../separation/Spec.bend as S
import ../prefix/Spec.bend as Prefix

# Both sides of a positive-length fill; no assumed path or read-value premise.
def left(a: Array<U64>,+sq: U32,+bishop: Bool,+mask: U64,+subset: U64,
  shape: {O.shape(a) == C.full(17n) : O.Shape}) ->
  {S.value(Array.get(U64,Tables.fill(2n,512,sq,bishop,mask,subset,a),511)) == S.value(Array.get(U64,a,511)) : U64}:
  Laws.bounded_fill_preserves_before(a,2n,512,514,511,sq,bishop,mask,subset,shape,{==},{==},{==})

def right(a: Array<U64>,+sq: U32,+bishop: Bool,+mask: U64,+subset: U64,
  shape: {O.shape(a) == C.full(17n) : O.Shape}) ->
  {S.value(Array.get(U64,Tables.fill(2n,512,sq,bishop,mask,subset,a),514)) == S.value(Array.get(U64,a,514)) : U64}:
  Laws.bounded_fill_preserves_after(a,2n,512,514,sq,bishop,mask,subset,shape,{==},{==})

# Universal real-block composition; the public producer discharges count facts.
def block_right(a: Array<U64>,+k: Nat,+q: U32,
  valid: {Nat.is_lt(k,128n) == True{} : Bool},shape: {O.shape(a) == C.full(17n) : O.Shape},
  start: {U32.is_le(Prefix.prefix(1n+k),q) == True{} : Bool},qb: {U32.is_lt(q,131072) == True{} : Bool}) ->
  {S.value(Array.get(U64,Spec.block(k,a),q)) == S.value(Array.get(U64,a,q)) : U64}:
  Laws.block_fill_preserves_after(a,k,q,valid,shape,start,qb)

# Overlap is excluded, and unbounded U32 arithmetic cannot replace the Nat budget.
def overlap() -> {Nat.is_le(Nat.add(2n,512n),513n) == False{} : Bool}: {==}
def wrapping_budget() -> {Nat.is_le(Nat.add(2n,4294967295n),131071n) == False{} : Bool}: {==}
