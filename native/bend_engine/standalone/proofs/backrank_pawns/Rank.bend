import Base
import ./Mask.bend as M
import ../layout/Domain.bend as Domain
import ../ray_clear_disjoint/Algebra.bend as D
import ../u64/Words.bend as Old

# Only the ctz-derived destination range is used; no board geometry premise.
def cases(+d: U32,n: Nat,+same: {U32.to_nat(d) == n : Nat},below: {Nat.is_lt(n,64n) == True{} : Bool},clear: {M.backrank(d) == False{} : Bool}) -> D.disjoint(U64.bit(U32.to_nat(d)),M.mask()):
  match n:
    case 0n:
      %Equal.sym(U32,d,0,Domain.injective(d,0,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      Old.false_true(D.disjoint(U64.bit(0n),M.mask()),Equal.sym(Bool,True{},False{},
        Equal.trans(Bool,M.backrank(0),M.backrank(d),False{},
          Equal.cong(U32,Bool,M.backrank,0,d,Equal.sym(U32,d,0,Domain.injective(d,0,same))),clear)))
    case 1n:
      %Equal.sym(U32,d,1,Domain.injective(d,1,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      Old.false_true(D.disjoint(U64.bit(1n),M.mask()),Equal.sym(Bool,True{},False{},
        Equal.trans(Bool,M.backrank(1),M.backrank(d),False{},
          Equal.cong(U32,Bool,M.backrank,1,d,Equal.sym(U32,d,1,Domain.injective(d,1,same))),clear)))
    case 2n:
      %Equal.sym(U32,d,2,Domain.injective(d,2,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      Old.false_true(D.disjoint(U64.bit(2n),M.mask()),Equal.sym(Bool,True{},False{},
        Equal.trans(Bool,M.backrank(2),M.backrank(d),False{},
          Equal.cong(U32,Bool,M.backrank,2,d,Equal.sym(U32,d,2,Domain.injective(d,2,same))),clear)))
    case 3n:
      %Equal.sym(U32,d,3,Domain.injective(d,3,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      Old.false_true(D.disjoint(U64.bit(3n),M.mask()),Equal.sym(Bool,True{},False{},
        Equal.trans(Bool,M.backrank(3),M.backrank(d),False{},
          Equal.cong(U32,Bool,M.backrank,3,d,Equal.sym(U32,d,3,Domain.injective(d,3,same))),clear)))
    case 4n:
      %Equal.sym(U32,d,4,Domain.injective(d,4,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      Old.false_true(D.disjoint(U64.bit(4n),M.mask()),Equal.sym(Bool,True{},False{},
        Equal.trans(Bool,M.backrank(4),M.backrank(d),False{},
          Equal.cong(U32,Bool,M.backrank,4,d,Equal.sym(U32,d,4,Domain.injective(d,4,same))),clear)))
    case 5n:
      %Equal.sym(U32,d,5,Domain.injective(d,5,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      Old.false_true(D.disjoint(U64.bit(5n),M.mask()),Equal.sym(Bool,True{},False{},
        Equal.trans(Bool,M.backrank(5),M.backrank(d),False{},
          Equal.cong(U32,Bool,M.backrank,5,d,Equal.sym(U32,d,5,Domain.injective(d,5,same))),clear)))
    case 6n:
      %Equal.sym(U32,d,6,Domain.injective(d,6,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      Old.false_true(D.disjoint(U64.bit(6n),M.mask()),Equal.sym(Bool,True{},False{},
        Equal.trans(Bool,M.backrank(6),M.backrank(d),False{},
          Equal.cong(U32,Bool,M.backrank,6,d,Equal.sym(U32,d,6,Domain.injective(d,6,same))),clear)))
    case 7n:
      %Equal.sym(U32,d,7,Domain.injective(d,7,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      Old.false_true(D.disjoint(U64.bit(7n),M.mask()),Equal.sym(Bool,True{},False{},
        Equal.trans(Bool,M.backrank(7),M.backrank(d),False{},
          Equal.cong(U32,Bool,M.backrank,7,d,Equal.sym(U32,d,7,Domain.injective(d,7,same))),clear)))
    case 8n:
      %Equal.sym(U32,d,8,Domain.injective(d,8,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 9n:
      %Equal.sym(U32,d,9,Domain.injective(d,9,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 10n:
      %Equal.sym(U32,d,10,Domain.injective(d,10,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 11n:
      %Equal.sym(U32,d,11,Domain.injective(d,11,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 12n:
      %Equal.sym(U32,d,12,Domain.injective(d,12,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 13n:
      %Equal.sym(U32,d,13,Domain.injective(d,13,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 14n:
      %Equal.sym(U32,d,14,Domain.injective(d,14,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 15n:
      %Equal.sym(U32,d,15,Domain.injective(d,15,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 16n:
      %Equal.sym(U32,d,16,Domain.injective(d,16,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 17n:
      %Equal.sym(U32,d,17,Domain.injective(d,17,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 18n:
      %Equal.sym(U32,d,18,Domain.injective(d,18,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 19n:
      %Equal.sym(U32,d,19,Domain.injective(d,19,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 20n:
      %Equal.sym(U32,d,20,Domain.injective(d,20,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 21n:
      %Equal.sym(U32,d,21,Domain.injective(d,21,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 22n:
      %Equal.sym(U32,d,22,Domain.injective(d,22,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 23n:
      %Equal.sym(U32,d,23,Domain.injective(d,23,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 24n:
      %Equal.sym(U32,d,24,Domain.injective(d,24,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 25n:
      %Equal.sym(U32,d,25,Domain.injective(d,25,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 26n:
      %Equal.sym(U32,d,26,Domain.injective(d,26,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 27n:
      %Equal.sym(U32,d,27,Domain.injective(d,27,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 28n:
      %Equal.sym(U32,d,28,Domain.injective(d,28,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 29n:
      %Equal.sym(U32,d,29,Domain.injective(d,29,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 30n:
      %Equal.sym(U32,d,30,Domain.injective(d,30,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 31n:
      %Equal.sym(U32,d,31,Domain.injective(d,31,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 32n:
      %Equal.sym(U32,d,32,Domain.injective(d,32,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 33n:
      %Equal.sym(U32,d,33,Domain.injective(d,33,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 34n:
      %Equal.sym(U32,d,34,Domain.injective(d,34,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 35n:
      %Equal.sym(U32,d,35,Domain.injective(d,35,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 36n:
      %Equal.sym(U32,d,36,Domain.injective(d,36,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 37n:
      %Equal.sym(U32,d,37,Domain.injective(d,37,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 38n:
      %Equal.sym(U32,d,38,Domain.injective(d,38,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 39n:
      %Equal.sym(U32,d,39,Domain.injective(d,39,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 40n:
      %Equal.sym(U32,d,40,Domain.injective(d,40,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 41n:
      %Equal.sym(U32,d,41,Domain.injective(d,41,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 42n:
      %Equal.sym(U32,d,42,Domain.injective(d,42,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 43n:
      %Equal.sym(U32,d,43,Domain.injective(d,43,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 44n:
      %Equal.sym(U32,d,44,Domain.injective(d,44,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 45n:
      %Equal.sym(U32,d,45,Domain.injective(d,45,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 46n:
      %Equal.sym(U32,d,46,Domain.injective(d,46,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 47n:
      %Equal.sym(U32,d,47,Domain.injective(d,47,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 48n:
      %Equal.sym(U32,d,48,Domain.injective(d,48,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 49n:
      %Equal.sym(U32,d,49,Domain.injective(d,49,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 50n:
      %Equal.sym(U32,d,50,Domain.injective(d,50,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 51n:
      %Equal.sym(U32,d,51,Domain.injective(d,51,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 52n:
      %Equal.sym(U32,d,52,Domain.injective(d,52,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 53n:
      %Equal.sym(U32,d,53,Domain.injective(d,53,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 54n:
      %Equal.sym(U32,d,54,Domain.injective(d,54,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 55n:
      %Equal.sym(U32,d,55,Domain.injective(d,55,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      {==}
    case 56n:
      %Equal.sym(U32,d,56,Domain.injective(d,56,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      Old.false_true(D.disjoint(U64.bit(56n),M.mask()),Equal.sym(Bool,True{},False{},
        Equal.trans(Bool,M.backrank(56),M.backrank(d),False{},
          Equal.cong(U32,Bool,M.backrank,56,d,Equal.sym(U32,d,56,Domain.injective(d,56,same))),clear)))
    case 57n:
      %Equal.sym(U32,d,57,Domain.injective(d,57,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      Old.false_true(D.disjoint(U64.bit(57n),M.mask()),Equal.sym(Bool,True{},False{},
        Equal.trans(Bool,M.backrank(57),M.backrank(d),False{},
          Equal.cong(U32,Bool,M.backrank,57,d,Equal.sym(U32,d,57,Domain.injective(d,57,same))),clear)))
    case 58n:
      %Equal.sym(U32,d,58,Domain.injective(d,58,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      Old.false_true(D.disjoint(U64.bit(58n),M.mask()),Equal.sym(Bool,True{},False{},
        Equal.trans(Bool,M.backrank(58),M.backrank(d),False{},
          Equal.cong(U32,Bool,M.backrank,58,d,Equal.sym(U32,d,58,Domain.injective(d,58,same))),clear)))
    case 59n:
      %Equal.sym(U32,d,59,Domain.injective(d,59,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      Old.false_true(D.disjoint(U64.bit(59n),M.mask()),Equal.sym(Bool,True{},False{},
        Equal.trans(Bool,M.backrank(59),M.backrank(d),False{},
          Equal.cong(U32,Bool,M.backrank,59,d,Equal.sym(U32,d,59,Domain.injective(d,59,same))),clear)))
    case 60n:
      %Equal.sym(U32,d,60,Domain.injective(d,60,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      Old.false_true(D.disjoint(U64.bit(60n),M.mask()),Equal.sym(Bool,True{},False{},
        Equal.trans(Bool,M.backrank(60),M.backrank(d),False{},
          Equal.cong(U32,Bool,M.backrank,60,d,Equal.sym(U32,d,60,Domain.injective(d,60,same))),clear)))
    case 61n:
      %Equal.sym(U32,d,61,Domain.injective(d,61,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      Old.false_true(D.disjoint(U64.bit(61n),M.mask()),Equal.sym(Bool,True{},False{},
        Equal.trans(Bool,M.backrank(61),M.backrank(d),False{},
          Equal.cong(U32,Bool,M.backrank,61,d,Equal.sym(U32,d,61,Domain.injective(d,61,same))),clear)))
    case 62n:
      %Equal.sym(U32,d,62,Domain.injective(d,62,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      Old.false_true(D.disjoint(U64.bit(62n),M.mask()),Equal.sym(Bool,True{},False{},
        Equal.trans(Bool,M.backrank(62),M.backrank(d),False{},
          Equal.cong(U32,Bool,M.backrank,62,d,Equal.sym(U32,d,62,Domain.injective(d,62,same))),clear)))
    case 63n:
      %Equal.sym(U32,d,63,Domain.injective(d,63,same)) : D.disjoint(U64.bit(U32.to_nat(_)),M.mask())
      Old.false_true(D.disjoint(U64.bit(63n),M.mask()),Equal.sym(Bool,True{},False{},
        Equal.trans(Bool,M.backrank(63),M.backrank(d),False{},
          Equal.cong(U32,Bool,M.backrank,63,d,Equal.sym(U32,d,63,Domain.injective(d,63,same))),clear)))
    case 64n+p: Domain.impossible(D.disjoint(U64.bit(U32.to_nat(d)),M.mask()),p,below)

def bounded(+d: U32,below: {Nat.is_lt(U32.to_nat(d),64n) == True{} : Bool},clear: {M.backrank(d) == False{} : Bool}) -> D.disjoint(U64.bit(U32.to_nat(d)),M.mask()):
  cases(d,U32.to_nat(d),{==},below,clear)
