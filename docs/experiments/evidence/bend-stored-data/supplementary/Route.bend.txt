# Supplementary composition only: both headers are supplied as explicit read certificates.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../../../bitboard_probe/Sliders.bend as Sliders
import ../u64/LAWS.bend as U
import ../u64/PROOF.bend as UP
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R

def model(+c: O.Cells,+key: U32,+occ: U64,+mask: U64,+at: U32,
  mh: {Array.get(U64,O.pack(c),key) == (O.pack(c),mask) : Array<U64> & U64},
  oh: {Array.get(U64,O.pack(c),U32.add(128,key)) == (O.pack(c),U64.from_u32(at)) : Array<U64> & U64}) ->
  {Chess.slide(O.pack(c),key,occ) == Array.get(U64,O.pack(c),U32.add(at,Sliders.pext_index(occ,mask))) : Array<U64> & U64}:
  Equal.trans(Array<U64> & U64,Chess.slide(O.pack(c),key,occ),Chess.slide_mask(key,occ,(O.pack(c),mask)),
    Array.get(U64,O.pack(c),U32.add(at,Sliders.pext_index(occ,mask))),
    Equal.cong(Array<U64> & U64,Array<U64> & U64,r => Chess.slide_mask(key,occ,r),
      Array.get(U64,O.pack(c),key),(O.pack(c),mask),mh),
    Equal.trans(Array<U64> & U64,Chess.slide_mask(key,occ,(O.pack(c),mask)),
      Chess.slide_offset(Sliders.pext_index(occ,mask),(O.pack(c),U64.from_u32(at))),
      Array.get(U64,O.pack(c),U32.add(at,Sliders.pext_index(occ,mask))),
      Equal.cong(Array<U64> & U64,Array<U64> & U64,r => Chess.slide_offset(Sliders.pext_index(occ,mask),r),
        Array.get(U64,O.pack(c),U32.add(128,key)),(O.pack(c),U64.from_u32(at)),oh),
      Equal.cong(U32,Array<U64> & U64,p => Array.get(U64,O.pack(c),U32.add(p,Sliders.pext_index(occ,mask))),
        U64.low(U64.from_u32(at)),at,U.widening_low(at))))

def lift(-a: Array<U64>,+key: U32,+occ: U64,+mask: U64,+at: U32,
  mh: {Array.get(U64,a,key) == (a,mask) : Array<U64> & U64},
  oh: {Array.get(U64,a,U32.add(128,key)) == (a,U64.from_u32(at)) : Array<U64> & U64},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {Chess.slide(a,key,occ) == Array.get(U64,a,U32.add(at,Sliders.pext_index(occ,mask))) : Array<U64> & U64}:
  (+c,eq) = cert
  +same: {O.pack(c) == a : Array<U64>} = eq
  %same : {Chess.slide(_,key,occ) == Array.get(U64,a,U32.add(at,Sliders.pext_index(occ,mask))) : Array<U64> & U64}
  %same : {Chess.slide(O.pack(c),key,occ) == Array.get(U64,_,U32.add(at,Sliders.pext_index(occ,mask))) : Array<U64> & U64}
  model(c,key,occ,mask,at,
    %Equal.sym(Array<U64>,O.pack(c),a,same) : {Array.get(U64,_,key) == (O.pack(c),mask) : Array<U64> & U64}
    %Equal.sym(Array<U64>,O.pack(c),a,same) : {Array.get(U64,a,key) == (_,mask) : Array<U64> & U64}
    mh,
    %Equal.sym(Array<U64>,O.pack(c),a,same) : {Array.get(U64,_,U32.add(128,key)) == (O.pack(c),U64.from_u32(at)) : Array<U64> & U64}
    %Equal.sym(Array<U64>,O.pack(c),a,same) : {Array.get(U64,a,U32.add(128,key)) == (_,U64.from_u32(at)) : Array<U64> & U64}
    oh)

def actual(a: Array<U64>,+key: U32,+occ: U64,+mask: U64,+at: U32,
  mh: {Array.get(U64,a,key) == (a,mask) : Array<U64> & U64},
  oh: {Array.get(U64,a,U32.add(128,key)) == (a,U64.from_u32(at)) : Array<U64> & U64}) ->
  {Chess.slide(a,key,occ) == Array.get(U64,a,U32.add(at,Sliders.pext_index(occ,mask))) : Array<U64> & U64}:
  lift(a,key,occ,mask,at,mh,oh,R.reify(a))
