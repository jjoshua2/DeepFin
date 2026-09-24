import Base
import ./Path.bend as P

def same_bool(+a: Bool,+b: Bool,e: {P.eq(a,b) == True{} : Bool}) -> {a == b : Bool}:
  match a b:
    case False{} False{}: {==}
    case True{} True{}: {==}
    case False{} True{}: %e : {False{} == _ : Bool} {==}
    case True{} False{}: %e : {_ == False{} : Bool} {==}

def left(+a: Bool,+b: Bool,e: {Bool.and(a,b) == True{} : Bool}) -> {a == True{} : Bool}:
  match a:
    case False{}: e
    case True{}: {==}

def right(+a: Bool,+b: Bool,e: {Bool.and(a,b) == True{} : Bool}) -> {b == True{} : Bool}:
  match a b:
    case False{} False{}: e
    case False{} True{}: {==}
    case True{} False{}: e
    case True{} True{}: {==}

def choose_zero(b: Bool) -> {Bool.pick(U64,b,U64.zero(),U64.zero()) == U64.zero() : U64}:
  match b:
    case False{}: {==}
    case True{}: {==}

def single(+x: Nat,+a: U64) -> {P.attack(x <> Nil{},a) == U64.or(U64.bit(x),U64.zero()) : U64}:
  Equal.cong(U64,U64,v => U64.or(U64.bit(x),v),
    Bool.pick(U64,U64.test_bit(a,x),U64.zero(),U64.zero()),U64.zero(),choose_zero(U64.test_bit(a,x)))

def interior_only(+xs: List<&2,Nat>,+a: U64,+b: U64,
  +e: {P.agree(xs,a,b) == True{} : Bool}) -> {P.attack(xs,a) == P.attack(xs,b) : U64}:
  match xs:
    case Nil{}: {==}
    case +x <> tail:
      match tail:
        case Nil{}:
          Equal.trans(U64,P.attack(x <> Nil{},a),U64.or(U64.bit(x),U64.zero()),P.attack(x <> Nil{},b),
            single(x,a),Equal.sym(U64,P.attack(x <> Nil{},b),U64.or(U64.bit(x),U64.zero()),single(x,b)))
        case +y <> +rest:
          +ab = P.eq(U64.test_bit(a,x),U64.test_bit(b,x))
          +later = P.agree(y <> rest,a,b)
          Equal.cong(U64,U64,v => U64.or(U64.bit(x),v),
            Bool.pick(U64,U64.test_bit(a,x),U64.zero(),P.attack(y <> rest,a)),
            Bool.pick(U64,U64.test_bit(b,x),U64.zero(),P.attack(y <> rest,b)),
            Equal.trans(U64,
              Bool.pick(U64,U64.test_bit(a,x),U64.zero(),P.attack(y <> rest,a)),
              Bool.pick(U64,U64.test_bit(b,x),U64.zero(),P.attack(y <> rest,a)),
              Bool.pick(U64,U64.test_bit(b,x),U64.zero(),P.attack(y <> rest,b)),
              Equal.cong(Bool,U64,z => Bool.pick(U64,z,U64.zero(),P.attack(y <> rest,a)),
                U64.test_bit(a,x),U64.test_bit(b,x),same_bool(U64.test_bit(a,x),U64.test_bit(b,x),left(ab,later,e))),
              Equal.cong(U64,U64,v => Bool.pick(U64,U64.test_bit(b,x),U64.zero(),v),
                P.attack(y <> rest,a),P.attack(y <> rest,b),interior_only(y <> rest,a,b,right(ab,later,e)))))
