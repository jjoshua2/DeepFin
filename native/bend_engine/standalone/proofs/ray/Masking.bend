# Mask preservation observes only the mask bits, not desired attacks.
import Base
import ./Path.bend as P
import ./Frame.bend as Frame
import ./Algebra.bend as A
import ./Rays.bend as R

def keeps(xs: List<&2,Nat>,+mask: U64) -> Type:
  match xs:
    case Nil{}: {True{} == True{} : Bool}
    case +x <> tail:
      match tail:
        case Nil{}: {True{} == True{} : Bool}
        case y <> rest:
          {U64.and(mask,U64.bit(x)) == U64.bit(x) : U64} & keeps(y <> rest,mask)

def bit_preserved(+occ: U64,+mask: U64,+x: Nat,
  e: {U64.and(mask,U64.bit(x)) == U64.bit(x) : U64}) ->
  {U64.test_bit(U64.and(occ,mask),x) == U64.test_bit(occ,x) : Bool}:
  Equal.cong(U64,Bool,w => U64.is_ne(w,U64.zero()),U64.and(U64.and(occ,mask),U64.bit(x)),U64.and(occ,U64.bit(x)),
    Equal.trans(U64,U64.and(U64.and(occ,mask),U64.bit(x)),U64.and(occ,U64.and(mask,U64.bit(x))),U64.and(occ,U64.bit(x)),
      A.and_assoc(occ,mask,U64.bit(x)),
      Equal.cong(U64,U64,w => U64.and(occ,w),U64.and(mask,U64.bit(x)),U64.bit(x),e)))

def reflexive(+b: Bool) -> {P.eq(b,b) == True{} : Bool}:
  match b:
    case False{}: {==}
    case True{}: {==}

def observes(+occ: U64,+mask: U64,+x: Nat,
  e: {U64.and(mask,U64.bit(x)) == U64.bit(x) : U64}) ->
  {P.eq(U64.test_bit(occ,x),U64.test_bit(U64.and(occ,mask),x)) == True{} : Bool}:
  Equal.trans(Bool,P.eq(U64.test_bit(occ,x),U64.test_bit(U64.and(occ,mask),x)),P.eq(U64.test_bit(occ,x),U64.test_bit(occ,x)),True{},
    Equal.cong(Bool,Bool,b => P.eq(U64.test_bit(occ,x),b),U64.test_bit(U64.and(occ,mask),x),U64.test_bit(occ,x),bit_preserved(occ,mask,x,e)),
    reflexive(U64.test_bit(occ,x)))

def and_true(+a: Bool,+b: Bool,ea: {a == True{} : Bool},eb: {b == True{} : Bool}) ->
  {Bool.and(a,b) == True{} : Bool}:
  Equal.trans(Bool,Bool.and(a,b),b,True{},Equal.cong(Bool,Bool,x => Bool.and(x,b),a,True{},ea),eb)

def agreement(+xs: List<&2,Nat>,+occ: U64,+mask: U64,e: keeps(xs,mask)) ->
  {P.agree(xs,occ,U64.and(occ,mask)) == True{} : Bool}:
  match xs:
    case Nil{}: {==}
    case +x <> tail:
      match tail:
        case Nil{}: {==}
        case +y <> +rest:
          (here,there) = e
          and_true(P.eq(U64.test_bit(occ,x),U64.test_bit(U64.and(occ,mask),x)),P.agree(y <> rest,occ,U64.and(occ,mask)),
            observes(occ,mask,x,here),agreement(y <> rest,occ,mask,there))

def attack(+xs: List<&2,Nat>,+occ: U64,+mask: U64,e: keeps(xs,mask)) ->
  {P.attack(xs,U64.and(occ,mask)) == P.attack(xs,occ) : U64}:
  Equal.sym(U64,P.attack(xs,occ),P.attack(xs,U64.and(occ,mask)),
    Frame.interior_only(xs,occ,U64.and(occ,mask),agreement(xs,occ,mask,e)))

def extend_left(+extra: U64,+mask: U64,+bit: U64,+e: {U64.and(mask,bit) == bit : U64}) ->
  {U64.and(U64.or(extra,mask),bit) == bit : U64}:
  Equal.trans(U64,U64.and(U64.or(extra,mask),bit),U64.and(U64.or(extra,mask),U64.and(mask,bit)),bit,
    Equal.cong(U64,U64,x => U64.and(U64.or(extra,mask),x),bit,U64.and(mask,bit),Equal.sym(U64,U64.and(mask,bit),bit,e)),
    Equal.trans(U64,U64.and(U64.or(extra,mask),U64.and(mask,bit)),U64.and(mask,bit),bit,A.extend_left(extra,mask,bit),e))

def extend_right(+mask: U64,+extra: U64,+bit: U64,+e: {U64.and(mask,bit) == bit : U64}) ->
  {U64.and(U64.or(mask,extra),bit) == bit : U64}:
  Equal.trans(U64,U64.and(U64.or(mask,extra),bit),U64.and(U64.or(mask,extra),U64.and(mask,bit)),bit,
    Equal.cong(U64,U64,x => U64.and(U64.or(mask,extra),x),bit,U64.and(mask,bit),Equal.sym(U64,U64.and(mask,bit),bit,e)),
    Equal.trans(U64,U64.and(U64.or(mask,extra),U64.and(mask,bit)),U64.and(mask,bit),bit,A.extend_right(mask,extra,bit),e))

def add_left(+xs: List<&2,Nat>,+mask: U64,+extra: U64,e: keeps(xs,mask)) -> keeps(xs,U64.or(extra,mask)):
  match xs:
    case Nil{}: {==}
    case +x <> tail:
      match tail:
        case Nil{}: {==}
        case +y <> +rest:
          (here,there) = e
          (extend_left(extra,mask,U64.bit(x),here),add_left(y <> rest,mask,extra,there))

def add_right(+xs: List<&2,Nat>,+mask: U64,+extra: U64,e: keeps(xs,mask)) -> keeps(xs,U64.or(mask,extra)):
  match xs:
    case Nil{}: {==}
    case +x <> tail:
      match tail:
        case Nil{}: {==}
        case +y <> +rest:
          (here,there) = e
          (extend_right(mask,extra,U64.bit(x),here),add_right(y <> rest,mask,extra,there))

def own(+xs: List<&2,Nat>) -> keeps(xs,R.bits(R.interior(xs))):
  match xs:
    case Nil{}: {==}
    case +x <> tail:
      match tail:
        case Nil{}: {==}
        case +y <> +rest:
          (A.absorb_left(U64.bit(x),R.bits(R.interior(y <> rest))),
            add_left(y <> rest,R.bits(R.interior(y <> rest)),U64.bit(x),own(y <> rest)))
