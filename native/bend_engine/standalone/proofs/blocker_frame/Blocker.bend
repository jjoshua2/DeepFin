import Base
import ../ray/Path.bend as P
import ../ray/Frame.bend as F
import ../ray/Spec.bend as S
import ../ray/Fold.bend as Fold
import ../ray/Traversal.bend as T
import ../../Tables.bend as Tables
import ../u64/Words.bend as W

# Only observations before the last square can affect an attack.
# At a shared blocker the suffix contributes no observations.
def agree_step(a: Bool,b: Bool,later: Unit -> Bool) -> Bool:
  match a b:
    case False{} False{}: later(Unit{})
    case True{} True{}: True{}
    case False{} True{}: False{}
    case True{} False{}: False{}

def agree(xs: List<&2,Nat>,+a: U64,+b: U64) -> Bool:
  match xs:
    case Nil{}: True{}
    case +x <> tail:
      match tail:
        case Nil{}: True{}
        case y <> rest:
          agree_step(U64.test_bit(a,x),U64.test_bit(b,x),u => agree(y <> rest,a,b))

def head_value(+x: Nat,occupied: Bool,tail: U64) -> U64:
  U64.or(U64.bit(x),Bool.pick(U64,occupied,U64.zero(),tail))

# The continuation is required only in the both-clear branch.
def head_equal(+a: Bool,+b: Bool,+later: Bool,+x: Nat,+va: U64,+vb: U64,
  e: {agree_step(a,b,u => later) == True{} : Bool},
  next: {later == True{} : Bool} -> {va == vb : U64}) ->
  {head_value(x,a,va) == head_value(x,b,vb) : U64}:
  match a b:
    case False{} False{}:
      Equal.cong(U64,U64,v => U64.or(U64.bit(x),v),va,vb,next(e))
    case True{} True{}: {==}
    case False{} True{}:
      W.false_true({head_value(x,False{},va) == head_value(x,True{},vb) : U64},e)
    case True{} False{}:
      W.false_true({head_value(x,True{},va) == head_value(x,False{},vb) : U64},e)

def attack_equal(+xs: List<&2,Nat>,+a: U64,+b: U64,
  e: {agree(xs,a,b) == True{} : Bool}) -> {P.attack(xs,a) == P.attack(xs,b) : U64}:
  match xs:
    case Nil{}: {==}
    case +x <> tail:
      match tail:
        case Nil{}:
          Equal.trans(U64,P.attack(x <> Nil{},a),U64.or(U64.bit(x),U64.zero()),P.attack(x <> Nil{},b),
            F.single(x,a),Equal.sym(U64,P.attack(x <> Nil{},b),U64.or(U64.bit(x),U64.zero()),F.single(x,b)))
        case +y <> +rest:
          head_equal(U64.test_bit(a,x),U64.test_bit(b,x),agree(y <> rest,a,b),x,
            P.attack(y <> rest,a),P.attack(y <> rest,b),e,
            certificate => attack_equal(y <> rest,a,b,certificate))

def step_from_interior(+a: Bool,+b: Bool,+later: Bool,
  same: {P.eq(a,b) == True{} : Bool},rest: {later == True{} : Bool}) ->
  {agree_step(a,b,u => later) == True{} : Bool}:
  match a b:
    case False{} False{}: rest
    case True{} True{}: {==}
    case False{} True{}: W.false_true({agree_step(False{},True{},u => later) == True{} : Bool},same)
    case True{} False{}: W.false_true({agree_step(True{},False{},u => later) == True{} : Bool},same)

# The old premise implies the new one; the closed consumer shows strict weakening.
def interior_implies_agree(+xs: List<&2,Nat>,+a: U64,+b: U64,
  +e: {P.agree(xs,a,b) == True{} : Bool}) -> {agree(xs,a,b) == True{} : Bool}:
  match xs:
    case Nil{}: {==}
    case +x <> tail:
      match tail:
        case Nil{}: {==}
        case +y <> +rest:
          +same = P.eq(U64.test_bit(a,x),U64.test_bit(b,x))
          +later = P.agree(y <> rest,a,b)
          step_from_interior(U64.test_bit(a,x),U64.test_bit(b,x),agree(y <> rest,a,b),
            F.left(same,later,e),interior_implies_agree(y <> rest,a,b,F.right(same,later,e)))

def scan_equal(+xs: List<&2,Nat>,+a: U64,+b: U64,+acc: U64,
  e: {agree(xs,a,b) == True{} : Bool}) -> {S.scan(xs,a,acc) == S.scan(xs,b,acc) : U64}:
  Equal.trans(U64,S.scan(xs,a,acc),U64.or(acc,P.attack(xs,a)),S.scan(xs,b,acc),
    Fold.accumulator(xs,a,acc),
    Equal.trans(U64,U64.or(acc,P.attack(xs,a)),U64.or(acc,P.attack(xs,b)),S.scan(xs,b,acc),
      Equal.cong(U64,U64,v => U64.or(acc,v),P.attack(xs,a),P.attack(xs,b),attack_equal(xs,a,b,e)),
      Equal.sym(U64,S.scan(xs,b,acc),U64.or(acc,P.attack(xs,b)),Fold.accumulator(xs,b,acc))))

# trace observes actual coordinate steps and fuel, not occupancy or desired attacks.
# Initial stopped=False, mask=False and one shared accumulator match the bridge.
def actual_ray(+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+a: U64,+b: U64,+acc: U64,
  +route: {S.trace(n,sq,dir,xs) == True{} : Bool},
  e: {agree(xs,a,b) == True{} : Bool}) ->
  {Tables.ray(n,False{},sq,dir,a,False{},acc) ==
    Tables.ray(n,False{},sq,dir,b,False{},acc) : U64}:
  Equal.trans(U64,Tables.ray(n,False{},sq,dir,a,False{},acc),S.scan(xs,a,acc),
    Tables.ray(n,False{},sq,dir,b,False{},acc),
    T.actual(n,sq,dir,xs,a,acc,route),
    Equal.trans(U64,S.scan(xs,a,acc),S.scan(xs,b,acc),Tables.ray(n,False{},sq,dir,b,False{},acc),
      scan_equal(xs,a,b,acc,e),
      Equal.sym(U64,Tables.ray(n,False{},sq,dir,b,False{},acc),S.scan(xs,b,acc),
        T.actual(n,sq,dir,xs,b,acc,route))))
