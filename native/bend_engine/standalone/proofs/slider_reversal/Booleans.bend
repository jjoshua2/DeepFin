import Base
import ./Spec.bend as S

def and_true(+a: Bool) -> {Bool.and(a,True{}) == a : Bool}:
  match a:
    case False{}: {==}
    case True{}: {==}

def and_false(+a: Bool) -> {Bool.and(a,False{}) == False{} : Bool}:
  match a:
    case False{}: {==}
    case True{}: {==}

def and_comm(+a: Bool,+b: Bool) -> {Bool.and(a,b) == Bool.and(b,a) : Bool}:
  match a b:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def and_assoc(+a: Bool,+b: Bool,+c: Bool) ->
  {Bool.and(Bool.and(a,b),c) == Bool.and(a,Bool.and(b,c)) : Bool}:
  match a:
    case False{}: {==}
    case True{}: {==}

def or_cong(+a: Bool,+b: Bool,+c: Bool,+d: Bool,ac: {a == c : Bool},bd: {b == d : Bool}) ->
  {Bool.or(a,b) == Bool.or(c,d) : Bool}:
  Equal.trans(Bool,Bool.or(a,b),Bool.or(c,b),Bool.or(c,d),
    Equal.cong(Bool,Bool,x => Bool.or(x,b),a,c,ac),
    Equal.cong(Bool,Bool,x => Bool.or(c,x),b,d,bd))

def or_comm(+a: Bool,+b: Bool) -> {Bool.or(a,b) == Bool.or(b,a) : Bool}:
  match a b:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def four_cong(+a: Bool,+b: Bool,+c: Bool,+d: Bool,+aa: Bool,+bb: Bool,+cc: Bool,+dd: Bool,
  ea: {a == aa : Bool},eb: {b == bb : Bool},ec: {c == cc : Bool},ed: {d == dd : Bool}) ->
  {S.four(a,b,c,d) == S.four(aa,bb,cc,dd) : Bool}:
  or_cong(Bool.or(a,b),Bool.or(c,d),Bool.or(aa,bb),Bool.or(cc,dd),
    or_cong(a,b,aa,bb,ea,eb),or_cong(c,d,cc,dd,ec,ed))

def four_swap(+a: Bool,+b: Bool,+c: Bool,+d: Bool) ->
  {S.four(b,a,d,c) == S.four(a,b,c,d) : Bool}:
  or_cong(Bool.or(b,a),Bool.or(d,c),Bool.or(a,b),Bool.or(c,d),or_comm(b,a),or_comm(d,c))

def four_reverse(+a: Bool,+b: Bool,+c: Bool,+d: Bool) ->
  {S.four(d,c,b,a) == S.four(a,b,c,d) : Bool}:
  Equal.trans(Bool,S.four(d,c,b,a),S.four(b,a,d,c),S.four(a,b,c,d),
    or_comm(Bool.or(d,c),Bool.or(b,a)),four_swap(a,b,c,d))

def append_clear(+xs: List<&2,Nat>,+ys: List<&2,Nat>,+occ: U64) ->
  {S.clear(S.append(xs,ys),occ) == Bool.and(S.clear(xs,occ),S.clear(ys,occ)) : Bool}:
  match xs:
    case Nil{}: {==}
    case +h <> +t:
      Equal.trans(Bool,S.clear(S.append(h <> t,ys),occ),
        Bool.and(Bool.not(U64.test_bit(occ,h)),Bool.and(S.clear(t,occ),S.clear(ys,occ))),
        Bool.and(S.clear(h <> t,occ),S.clear(ys,occ)),
        Equal.cong(Bool,Bool,x => Bool.and(Bool.not(U64.test_bit(occ,h)),x),
          S.clear(S.append(t,ys),occ),Bool.and(S.clear(t,occ),S.clear(ys,occ)),append_clear(t,ys,occ)),
        Equal.sym(Bool,Bool.and(S.clear(h <> t,occ),S.clear(ys,occ)),
          Bool.and(Bool.not(U64.test_bit(occ,h)),Bool.and(S.clear(t,occ),S.clear(ys,occ))),
          and_assoc(Bool.not(U64.test_bit(occ,h)),S.clear(t,occ),S.clear(ys,occ))))

def reverse_clear(+xs: List<&2,Nat>,+occ: U64) ->
  {S.clear(S.reverse(xs),occ) == S.clear(xs,occ) : Bool}:
  match xs:
    case Nil{}: {==}
    case +h <> +t:
      Equal.trans(Bool,S.clear(S.reverse(h <> t),occ),
        Bool.and(S.clear(S.reverse(t),occ),S.clear(h <> Nil{},occ)),S.clear(h <> t,occ),
        append_clear(S.reverse(t),h <> Nil{},occ),
        Equal.trans(Bool,Bool.and(S.clear(S.reverse(t),occ),S.clear(h <> Nil{},occ)),
          Bool.and(S.clear(t,occ),S.clear(h <> Nil{},occ)),S.clear(h <> t,occ),
          Equal.cong(Bool,Bool,x => Bool.and(x,S.clear(h <> Nil{},occ)),
            S.clear(S.reverse(t),occ),S.clear(t,occ),reverse_clear(t,occ)),
          Equal.trans(Bool,Bool.and(S.clear(t,occ),S.clear(h <> Nil{},occ)),
            Bool.and(S.clear(t,occ),Bool.not(U64.test_bit(occ,h))),S.clear(h <> t,occ),
            Equal.cong(Bool,Bool,x => Bool.and(S.clear(t,occ),x),S.clear(h <> Nil{},occ),
              Bool.not(U64.test_bit(occ,h)),and_true(Bool.not(U64.test_bit(occ,h)))),
            and_comm(S.clear(t,occ),Bool.not(U64.test_bit(occ,h))))))

def mirror_visible(+v: S.Segment,+occ: U64) ->
  {S.visible(S.mirror(v),occ) == S.visible(v,occ) : Bool}:
  match v:
    case S.Missing{}: {==}
    case S.Between{xs}: reverse_clear(xs,occ)
