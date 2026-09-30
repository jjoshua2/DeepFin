import Base
import ./Algebra.bend as A
import ./Path.bend as P
import ./Spec.bend as S

def choose(+b: Bool,+acc: U64,+bit: U64,+tail: U64) ->
  {Bool.pick(U64,b,U64.or(acc,bit),U64.or(U64.or(acc,bit),tail)) ==
    U64.or(acc,U64.or(bit,Bool.pick(U64,b,U64.zero(),tail))) : U64}:
  match b:
    case True{}:
      Equal.sym(U64,U64.or(acc,U64.or(bit,U64.zero())),U64.or(acc,bit),
        Equal.cong(U64,U64,x => U64.or(acc,x),U64.or(bit,U64.zero()),bit,A.zero(bit)))
    case False{}: A.or_assoc(acc,bit,tail)

def accumulator(+xs: List<&2,Nat>,+occ: U64,+acc: U64) ->
  {S.scan(xs,occ,acc) == U64.or(acc,P.attack(xs,occ)) : U64}:
  match xs:
    case Nil{}: Equal.sym(U64,U64.or(acc,U64.zero()),acc,A.zero(acc))
    case +x <> tail:
      Equal.trans(U64,S.scan(x <> tail,occ,acc),
        Bool.pick(U64,U64.test_bit(occ,x),U64.or(acc,U64.bit(x)),U64.or(U64.or(acc,U64.bit(x)),P.attack(tail,occ))),
        U64.or(acc,P.attack(x <> tail,occ)),
        Equal.cong(U64,U64,v => Bool.pick(U64,U64.test_bit(occ,x),U64.or(acc,U64.bit(x)),v),
          S.scan(tail,occ,U64.or(acc,U64.bit(x))),U64.or(U64.or(acc,U64.bit(x)),P.attack(tail,occ)),accumulator(tail,occ,U64.or(acc,U64.bit(x)))),
        choose(U64.test_bit(occ,x),acc,U64.bit(x),P.attack(tail,occ)))

def initial(+xs: List<&2,Nat>,+occ: U64) -> {S.scan(xs,occ,U64.zero()) == P.attack(xs,occ) : U64}:
  Equal.trans(U64,S.scan(xs,occ,U64.zero()),U64.or(U64.zero(),P.attack(xs,occ)),P.attack(xs,occ),
    accumulator(xs,occ,U64.zero()),A.left_zero(P.attack(xs,occ)))
