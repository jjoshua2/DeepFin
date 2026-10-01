# Commuting Boolean conjunction exposes source ownership before geometry.
import Base
import ./Spec.bend as S
import ../attack_witness/Spec.bend as W
import ../slider_reversal/Booleans.bend as B

def five(a: Bool,b: Bool,c: Bool,d: Bool,e: Bool) -> Bool:
  Bool.or(Bool.or(Bool.or(Bool.or(a,b),c),d),e)

def five_cong(+a: Bool,+b: Bool,+c: Bool,+d: Bool,+e: Bool,+f: Bool,+g: Bool,+h: Bool,+i: Bool,+j: Bool,
  af: {a == f : Bool},bg: {b == g : Bool},ch: {c == h : Bool},di: {d == i : Bool},ej: {e == j : Bool}) ->
  {five(a,b,c,d,e) == five(f,g,h,i,j) : Bool}:
  B.or_cong(Bool.or(Bool.or(Bool.or(a,b),c),d),e,Bool.or(Bool.or(Bool.or(f,g),h),i),j,
    B.or_cong(Bool.or(Bool.or(a,b),c),d,Bool.or(Bool.or(f,g),h),i,
      B.or_cong(Bool.or(a,b),c,Bool.or(f,g),h,B.or_cong(a,b,f,g,af,bg),ch),di),ej)

def same(+own: Bool,+p: Bool,+n: Bool,+b: Bool,+r: Bool,+q: Bool,+k: Bool,+pa: Bool,+na: Bool,+ka: Bool,+ra: Bool,+ba: Bool) ->
  {W.row(own,p,n,b,r,q,k,pa,na,ka,ra,ba) == S.row(own,p,n,b,r,q,k,pa,na,ka,ra,ba) : Bool}:
  +old = five(Bool.and(pa,p),Bool.and(na,n),Bool.and(ka,k),Bool.and(ra,Bool.or(r,q)),Bool.and(ba,Bool.or(b,q)))
  +new = five(Bool.and(p,pa),Bool.and(n,na),Bool.and(k,ka),Bool.and(Bool.or(r,q),ra),Bool.and(Bool.or(b,q),ba))
  Equal.trans(Bool,W.row(own,p,n,b,r,q,k,pa,na,ka,ra,ba),Bool.and(own,old),S.row(own,p,n,b,r,q,k,pa,na,ka,ra,ba),
    B.and_comm(old,own),
    Equal.cong(Bool,Bool,x => Bool.and(own,x),old,new,
      five_cong(Bool.and(pa,p),Bool.and(na,n),Bool.and(ka,k),Bool.and(ra,Bool.or(r,q)),Bool.and(ba,Bool.or(b,q)),
        Bool.and(p,pa),Bool.and(n,na),Bool.and(k,ka),Bool.and(Bool.or(r,q),ra),Bool.and(Bool.or(b,q),ba),
        B.and_comm(pa,p),B.and_comm(na,n),B.and_comm(ka,k),B.and_comm(ra,Bool.or(r,q)),B.and_comm(ba,Bool.or(b,q)))))
