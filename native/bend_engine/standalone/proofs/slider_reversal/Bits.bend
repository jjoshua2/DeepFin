import Base
import ./Spec.bend as S
import ./Booleans.bend as BoolProof
import ../decoder/Bits.bend as B
import ../attack_reversal/Mask.bend as Packed
import ../attack_reversal/Spec.bend as Member
import ../ray/Algebra.bend as A
import ../ray/Path.bend as Path

def zero(+i: Nat,+bound: {Nat.is_lt(i,64n) == True{} : Bool}) ->
  {U64.test_bit(U64.zero(),i) == False{} : Bool}:
  Equal.trans(Bool,U64.test_bit(U64.zero(),i),B.get64(Packed.packed(Nil{}),i),False{},
    B.test64(U64.zero(),i,bound),Packed.get(Nil{},i,bound))

def singleton_encoding(+h: Nat) -> {U64.bit(h) == Packed.packed(h <> Nil{}) : U64}:
  Equal.trans(U64,U64.bit(h),U64.or(U64.bit(h),U64.zero()),Packed.packed(h <> Nil{}),
    Equal.sym(U64,U64.or(U64.bit(h),U64.zero()),U64.bit(h),A.zero(U64.bit(h))),Packed.head(h,Nil{}))

def singleton(+h: Nat,+i: Nat,+bound: {Nat.is_lt(i,64n) == True{} : Bool}) ->
  {U64.test_bit(U64.bit(h),i) == Nat.is_eq(i,h) : Bool}:
  Equal.trans(Bool,U64.test_bit(U64.bit(h),i),U64.test_bit(Packed.packed(h <> Nil{}),i),Nat.is_eq(i,h),
    Equal.cong(U64,Bool,m => U64.test_bit(m,i),U64.bit(h),Packed.packed(h <> Nil{}),singleton_encoding(h)),
    Equal.trans(Bool,U64.test_bit(Packed.packed(h <> Nil{}),i),B.get64(Packed.packed(h <> Nil{}),i),Nat.is_eq(i,h),
      B.test64(Packed.packed(h <> Nil{}),i,bound),
      Equal.trans(Bool,B.get64(Packed.packed(h <> Nil{}),i),Member.contains(i,h <> Nil{}),Nat.is_eq(i,h),
        Packed.get(h <> Nil{},i,bound),B.or_false(Nat.is_eq(i,h)))))

def pick(+stop: Bool,+tail: U64,+i: Nat,+bound: {Nat.is_lt(i,64n) == True{} : Bool}) ->
  {U64.test_bit(Bool.pick(U64,stop,U64.zero(),tail),i) == Bool.and(Bool.not(stop),U64.test_bit(tail,i)) : Bool}:
  match stop:
    case False{}: {==}
    case True{}: zero(i,bound)

def prepend_visible(+h: Nat,+v: S.Segment,+occ: U64) ->
  {Bool.and(Bool.not(U64.test_bit(occ,h)),S.visible(v,occ)) == S.visible(S.prepend(h,v),occ) : Bool}:
  match v:
    case S.Missing{}: BoolProof.and_false(Bool.not(U64.test_bit(occ,h)))
    case S.Between{xs}: {==}

def choice(+same: Bool,+h: Nat,+v: S.Segment,+occ: U64) ->
  {Bool.or(same,Bool.and(Bool.not(U64.test_bit(occ,h)),S.visible(v,occ))) ==
    S.visible(Bool.pick(S.Segment,same,S.Between{Nil{}},S.prepend(h,v)),occ) : Bool}:
  match same:
    case True{}: {==}
    case False{}: prepend_visible(h,v,occ)

def path(+xs: List<&2,Nat>,+occ: U64,+dst: Nat,+bound: {Nat.is_lt(dst,64n) == True{} : Bool}) ->
  {U64.test_bit(Path.attack(xs,occ),dst) == S.visible(S.cut(dst,xs),occ) : Bool}:
  match xs:
    case Nil{}: zero(dst,bound)
    case +h <> +tail:
      +rest = Bool.pick(U64,U64.test_bit(occ,h),U64.zero(),Path.attack(tail,occ))
      Equal.trans(Bool,U64.test_bit(Path.attack(h <> tail,occ),dst),
        Bool.or(U64.test_bit(U64.bit(h),dst),U64.test_bit(rest,dst)),S.visible(S.cut(dst,h <> tail),occ),
        B.test64_or(U64.bit(h),rest,dst,bound),
        Equal.trans(Bool,Bool.or(U64.test_bit(U64.bit(h),dst),U64.test_bit(rest,dst)),
          Bool.or(Nat.is_eq(dst,h),Bool.and(Bool.not(U64.test_bit(occ,h)),U64.test_bit(Path.attack(tail,occ),dst))),
          S.visible(S.cut(dst,h <> tail),occ),
          BoolProof.or_cong(U64.test_bit(U64.bit(h),dst),U64.test_bit(rest,dst),Nat.is_eq(dst,h),
            Bool.and(Bool.not(U64.test_bit(occ,h)),U64.test_bit(Path.attack(tail,occ),dst)),
            singleton(h,dst,bound),pick(U64.test_bit(occ,h),Path.attack(tail,occ),dst,bound)),
          Equal.trans(Bool,Bool.or(Nat.is_eq(dst,h),Bool.and(Bool.not(U64.test_bit(occ,h)),U64.test_bit(Path.attack(tail,occ),dst))),
            Bool.or(Nat.is_eq(dst,h),Bool.and(Bool.not(U64.test_bit(occ,h)),S.visible(S.cut(dst,tail),occ))),
            S.visible(S.cut(dst,h <> tail),occ),
            Equal.cong(Bool,Bool,x => Bool.or(Nat.is_eq(dst,h),Bool.and(Bool.not(U64.test_bit(occ,h)),x)),
              U64.test_bit(Path.attack(tail,occ),dst),S.visible(S.cut(dst,tail),occ),path(tail,occ,dst,bound)),
            choice(Nat.is_eq(dst,h),h,S.cut(dst,tail),occ))))
