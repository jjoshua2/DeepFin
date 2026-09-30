# Finite input-only certificates, evaluated from actual steps and independent paths.
# The scaffold enumerates domain cases, never occupancy or expected attack values.
import Base
import ./Grid.bend as G
import ./Rays.bend as R
import ../layout/Domain.bend as D

def endpoint(xs: List<&2,Nat>,start: Nat) -> Nat:
  match xs:
    case Nil{}: start
    case x <> tail: endpoint(tail,x)

def statement(+s: Nat,+d: Nat) -> Type:
  {G.next(endpoint(R.path(7n,s,d),s),d) == 64n : Nat}


def square_0(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(0n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(0n,8n+p),p,db)

def square_1(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(1n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(1n,8n+p),p,db)

def square_2(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(2n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(2n,8n+p),p,db)

def square_3(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(3n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(3n,8n+p),p,db)

def square_4(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(4n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(4n,8n+p),p,db)

def square_5(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(5n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(5n,8n+p),p,db)

def square_6(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(6n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(6n,8n+p),p,db)

def square_7(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(7n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(7n,8n+p),p,db)

def square_8(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(8n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(8n,8n+p),p,db)

def square_9(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(9n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(9n,8n+p),p,db)

def square_10(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(10n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(10n,8n+p),p,db)

def square_11(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(11n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(11n,8n+p),p,db)

def square_12(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(12n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(12n,8n+p),p,db)

def square_13(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(13n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(13n,8n+p),p,db)

def square_14(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(14n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(14n,8n+p),p,db)

def square_15(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(15n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(15n,8n+p),p,db)

def square_16(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(16n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(16n,8n+p),p,db)

def square_17(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(17n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(17n,8n+p),p,db)

def square_18(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(18n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(18n,8n+p),p,db)

def square_19(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(19n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(19n,8n+p),p,db)

def square_20(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(20n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(20n,8n+p),p,db)

def square_21(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(21n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(21n,8n+p),p,db)

def square_22(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(22n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(22n,8n+p),p,db)

def square_23(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(23n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(23n,8n+p),p,db)

def square_24(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(24n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(24n,8n+p),p,db)

def square_25(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(25n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(25n,8n+p),p,db)

def square_26(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(26n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(26n,8n+p),p,db)

def square_27(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(27n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(27n,8n+p),p,db)

def square_28(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(28n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(28n,8n+p),p,db)

def square_29(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(29n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(29n,8n+p),p,db)

def square_30(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(30n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(30n,8n+p),p,db)

def square_31(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(31n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(31n,8n+p),p,db)

def square_32(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(32n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(32n,8n+p),p,db)

def square_33(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(33n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(33n,8n+p),p,db)

def square_34(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(34n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(34n,8n+p),p,db)

def square_35(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(35n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(35n,8n+p),p,db)

def square_36(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(36n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(36n,8n+p),p,db)

def square_37(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(37n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(37n,8n+p),p,db)

def square_38(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(38n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(38n,8n+p),p,db)

def square_39(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(39n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(39n,8n+p),p,db)

def square_40(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(40n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(40n,8n+p),p,db)

def square_41(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(41n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(41n,8n+p),p,db)

def square_42(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(42n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(42n,8n+p),p,db)

def square_43(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(43n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(43n,8n+p),p,db)

def square_44(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(44n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(44n,8n+p),p,db)

def square_45(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(45n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(45n,8n+p),p,db)

def square_46(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(46n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(46n,8n+p),p,db)

def square_47(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(47n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(47n,8n+p),p,db)

def square_48(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(48n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(48n,8n+p),p,db)

def square_49(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(49n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(49n,8n+p),p,db)

def square_50(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(50n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(50n,8n+p),p,db)

def square_51(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(51n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(51n,8n+p),p,db)

def square_52(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(52n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(52n,8n+p),p,db)

def square_53(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(53n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(53n,8n+p),p,db)

def square_54(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(54n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(54n,8n+p),p,db)

def square_55(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(55n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(55n,8n+p),p,db)

def square_56(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(56n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(56n,8n+p),p,db)

def square_57(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(57n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(57n,8n+p),p,db)

def square_58(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(58n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(58n,8n+p),p,db)

def square_59(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(59n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(59n,8n+p),p,db)

def square_60(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(60n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(60n,8n+p),p,db)

def square_61(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(61n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(61n,8n+p),p,db)

def square_62(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(62n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(62n,8n+p),p,db)

def square_63(+d: Nat,db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(63n,d):
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: D.impossible(statement(63n,8n+p),p,db)

def finite(+s: Nat,+d: Nat,sb: {Nat.is_lt(s,64n) == True{} : Bool},db: {Nat.is_lt(d,8n) == True{} : Bool}) -> statement(s,d):
  match s:
    case 0n: square_0(d,db)
    case 1n: square_1(d,db)
    case 2n: square_2(d,db)
    case 3n: square_3(d,db)
    case 4n: square_4(d,db)
    case 5n: square_5(d,db)
    case 6n: square_6(d,db)
    case 7n: square_7(d,db)
    case 8n: square_8(d,db)
    case 9n: square_9(d,db)
    case 10n: square_10(d,db)
    case 11n: square_11(d,db)
    case 12n: square_12(d,db)
    case 13n: square_13(d,db)
    case 14n: square_14(d,db)
    case 15n: square_15(d,db)
    case 16n: square_16(d,db)
    case 17n: square_17(d,db)
    case 18n: square_18(d,db)
    case 19n: square_19(d,db)
    case 20n: square_20(d,db)
    case 21n: square_21(d,db)
    case 22n: square_22(d,db)
    case 23n: square_23(d,db)
    case 24n: square_24(d,db)
    case 25n: square_25(d,db)
    case 26n: square_26(d,db)
    case 27n: square_27(d,db)
    case 28n: square_28(d,db)
    case 29n: square_29(d,db)
    case 30n: square_30(d,db)
    case 31n: square_31(d,db)
    case 32n: square_32(d,db)
    case 33n: square_33(d,db)
    case 34n: square_34(d,db)
    case 35n: square_35(d,db)
    case 36n: square_36(d,db)
    case 37n: square_37(d,db)
    case 38n: square_38(d,db)
    case 39n: square_39(d,db)
    case 40n: square_40(d,db)
    case 41n: square_41(d,db)
    case 42n: square_42(d,db)
    case 43n: square_43(d,db)
    case 44n: square_44(d,db)
    case 45n: square_45(d,db)
    case 46n: square_46(d,db)
    case 47n: square_47(d,db)
    case 48n: square_48(d,db)
    case 49n: square_49(d,db)
    case 50n: square_50(d,db)
    case 51n: square_51(d,db)
    case 52n: square_52(d,db)
    case 53n: square_53(d,db)
    case 54n: square_54(d,db)
    case 55n: square_55(d,db)
    case 56n: square_56(d,db)
    case 57n: square_57(d,db)
    case 58n: square_58(d,db)
    case 59n: square_59(d,db)
    case 60n: square_60(d,db)
    case 61n: square_61(d,db)
    case 62n: square_62(d,db)
    case 63n: square_63(d,db)
    case 64n+p: D.impossible(statement(64n+p,d),p,sb)
