# Independent path-prefix semantics, including the first occupied square.
import Base

def attack(xs: List<&2,Nat>, +occ: U64) -> U64:
  match xs:
    case Nil{}: U64.zero()
    case +x <> tail:
      U64.or(U64.bit(x),Bool.pick(U64,U64.test_bit(occ,x),U64.zero(),attack(tail,occ)))

def eq(+a: Bool,+b: Bool) -> Bool:
  Bool.pick(Bool,a,b,Bool.not(b))

# Compare only occupancy observations before the last path square.
# No desired attack value or equality of attack functions is an input.
def agree(xs: List<&2,Nat>,+a: U64,+b: U64) -> Bool:
  match xs:
    case Nil{}: True{}
    case +x <> tail:
      match tail:
        case Nil{}: True{}
        case +y <> +rest:
          Bool.and(eq(U64.test_bit(a,x),U64.test_bit(b,x)),agree(y <> rest,a,b))
