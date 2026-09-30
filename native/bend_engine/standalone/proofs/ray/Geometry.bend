import Base
import ./Spec.bend as S
import ./Slider.bend as Slider
import ./Masking.bend as M
import ./MaskCertificate.bend as Cert
import ./Masks.bend as Masks
import ./Path.bend as P
import ./Rays.bend as R
import ../layout/Spec.bend as Layout

def masked_paths(+k: Nat,+occ: U64,e: Cert.four(k)) ->
  {S.rays(k,U64.and(occ,R.mask(k))) == S.rays(k,occ) : U64}:
  (e0,rest) = e
  (e1,last) = rest
  (e2,e3) = last
  +s = Nat.mod(k,64n)
  +d = Bool.pick(Nat,Nat.is_ge(k,64n),4n,0n)
  +mask = R.mask(k)
  Slider.combine(
    U64.or(P.attack(R.path(7n,s,d),U64.and(occ,mask)),P.attack(R.path(7n,s,1n+d),U64.and(occ,mask))),
    U64.or(P.attack(R.path(7n,s,d),occ),P.attack(R.path(7n,s,1n+d),occ)),
    U64.or(P.attack(R.path(7n,s,2n+d),U64.and(occ,mask)),P.attack(R.path(7n,s,3n+d),U64.and(occ,mask))),
    U64.or(P.attack(R.path(7n,s,2n+d),occ),P.attack(R.path(7n,s,3n+d),occ)),
    Slider.combine(P.attack(R.path(7n,s,d),U64.and(occ,mask)),P.attack(R.path(7n,s,d),occ),
      P.attack(R.path(7n,s,1n+d),U64.and(occ,mask)),P.attack(R.path(7n,s,1n+d),occ),
      M.attack(R.path(7n,s,d),occ,mask,e0),M.attack(R.path(7n,s,1n+d),occ,mask,e1)),
    Slider.combine(P.attack(R.path(7n,s,2n+d),U64.and(occ,mask)),P.attack(R.path(7n,s,2n+d),occ),
      P.attack(R.path(7n,s,3n+d),U64.and(occ,mask)),P.attack(R.path(7n,s,3n+d),occ),
      M.attack(R.path(7n,s,2n+d),occ,mask,e2),M.attack(R.path(7n,s,3n+d),occ,mask,e3)))

def masked_slider(+k: Nat,+occ: U64,+valid: {Nat.is_lt(k,128n) == True{} : Bool}) ->
  {S.actual_slider(k,U64.and(occ,Layout.mask(U32.from_nat(k)))) == S.rays(k,occ) : U64}:
  Equal.trans(U64,S.actual_slider(k,U64.and(occ,Layout.mask(U32.from_nat(k)))),
    S.rays(k,U64.and(occ,Layout.mask(U32.from_nat(k)))),S.rays(k,occ),
    Slider.actual(k,U64.and(occ,Layout.mask(U32.from_nat(k))),valid),
    Equal.trans(U64,S.rays(k,U64.and(occ,Layout.mask(U32.from_nat(k)))),S.rays(k,U64.and(occ,R.mask(k))),S.rays(k,occ),
      Equal.cong(U64,U64,mask => S.rays(k,U64.and(occ,mask)),Layout.mask(U32.from_nat(k)),R.mask(k),Masks.finite(k,valid)),
      masked_paths(k,occ,Cert.finite(k,valid))))
