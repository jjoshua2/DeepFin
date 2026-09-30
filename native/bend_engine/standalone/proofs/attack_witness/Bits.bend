import Base
import ./Spec.bend as S

def any(+d: Nat,w: Word(d)) -> Bool:
  match d:
    case 0n: False{}
    case 1n+t:
      match w:
        case WCon{h,tail}: Bool.or(h,any(t,tail))

def de_morgan(+a: Bool,+b: Bool) ->
  {Bool.not(Bool.and(a,b)) == Bool.or(Bool.not(a),Bool.not(b)) : Bool}:
  match a b:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def split(+h: Bool,+c: Cmp) ->
  {Bool.not(Cmp.is_eq(Word.cmp.fin(h,False{},c))) == Bool.or(h,Bool.not(Cmp.is_eq(c))) : Bool}:
  match h c:
    case False{} LT{}: {==}
    case False{} EQ{}: {==}
    case False{} GT{}: {==}
    case True{} LT{}: {==}
    case True{} EQ{}: {==}
    case True{} GT{}: {==}

def nonzero(+d: Nat,+w: Word(d)) ->
  {Bool.not(Cmp.is_eq(Word.cmp(d,w,Word.zero(d)))) == any(d,w) : Bool}:
  match d:
    case 0n: {==}
    case 1n+t:
      match w:
        case WCon{+h,+tail}:
          Equal.trans(Bool,
            Bool.not(Cmp.is_eq(Word.cmp(1n+t,WCon{h,tail},Word.zero(1n+t)))),
            Bool.or(h,Bool.not(Cmp.is_eq(Word.cmp(t,tail,Word.zero(t))))),
            any(1n+t,WCon{h,tail}),
            split(h,Word.cmp(t,tail,Word.zero(t))),
            Equal.cong(Bool,Bool,x => Bool.or(h,x),
              Bool.not(Cmp.is_eq(Word.cmp(t,tail,Word.zero(t)))),any(t,tail),nonzero(t,tail)))

def fold(+d: Nat,+own: Word(d),+p: Word(d),+n: Word(d),+b: Word(d),+r: Word(d),+q: Word(d),+k: Word(d),+pa: Word(d),+na: Word(d),+ka: Word(d),+ra: Word(d),+ba: Word(d)) -> Word(d):
  Word.and(d,Word.or(d,Word.or(d,Word.or(d,Word.or(d,Word.and(d,pa,p),Word.and(d,na,n)),Word.and(d,ka,k)),Word.and(d,ra,Word.or(d,r,q))),Word.and(d,ba,Word.or(d,b,q))),own)

def pointwise(+d: Nat,+own: Word(d),+p: Word(d),+n: Word(d),+b: Word(d),+r: Word(d),+q: Word(d),+k: Word(d),+pa: Word(d),+na: Word(d),+ka: Word(d),+ra: Word(d),+ba: Word(d)) ->
  {any(d,fold(d,own,p,n,b,r,q,k,pa,na,ka,ra,ba)) == S.scan(d,own,p,n,b,r,q,k,pa,na,ka,ra,ba) : Bool}:
  match d:
    case 0n: {==}
    case 1n+t:
      match own p n b r q k pa na ka ra ba:
        case WCon{+ownh,+ownt} WCon{+ph,+pt} WCon{+nh,+nt} WCon{+bh,+bt} WCon{+rh,+rt} WCon{+qh,+qt} WCon{+kh,+kt} WCon{+pah,+pat} WCon{+nah,+nat} WCon{+kah,+kat} WCon{+rah,+rat} WCon{+bah,+bat}:
          Equal.cong(Bool,Bool,x => Bool.or(S.row(ownh,ph,nh,bh,rh,qh,kh,pah,nah,kah,rah,bah),x),
            any(t,fold(t,ownt,pt,nt,bt,rt,qt,kt,pat,nat,kat,rat,bat)),S.scan(t,ownt,pt,nt,bt,rt,qt,kt,pat,nat,kat,rat,bat),
            pointwise(t,ownt,pt,nt,bt,rt,qt,kt,pat,nat,kat,rat,bat))

def reflected(+d: Nat,+own: Word(d),+p: Word(d),+n: Word(d),+b: Word(d),+r: Word(d),+q: Word(d),+k: Word(d),+pa: Word(d),+na: Word(d),+ka: Word(d),+ra: Word(d),+ba: Word(d)) ->
  {Bool.not(Cmp.is_eq(Word.cmp(d,fold(d,own,p,n,b,r,q,k,pa,na,ka,ra,ba),Word.zero(d)))) == S.scan(d,own,p,n,b,r,q,k,pa,na,ka,ra,ba) : Bool}:
  Equal.trans(Bool,
    Bool.not(Cmp.is_eq(Word.cmp(d,fold(d,own,p,n,b,r,q,k,pa,na,ka,ra,ba),Word.zero(d)))),
    any(d,fold(d,own,p,n,b,r,q,k,pa,na,ka,ra,ba)),S.scan(d,own,p,n,b,r,q,k,pa,na,ka,ra,ba),
    nonzero(d,fold(d,own,p,n,b,r,q,k,pa,na,ka,ra,ba)),pointwise(d,own,p,n,b,r,q,k,pa,na,ka,ra,ba))
