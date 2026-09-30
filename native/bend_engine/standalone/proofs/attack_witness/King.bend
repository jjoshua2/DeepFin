import Base
import ../../../legal_probe/Chess.bend as Chess
import ../layout/Domain.bend as Domain
import ./Queries.bend as Q
import ./Flow.bend as Flow

def bit_index(+sq: Nat,e: {Nat.is_lt(sq,64n) == True{} : Bool}) ->
  {U64.ctz(U64.bit(sq)) == U32.from_nat(sq) : U32}:
  match sq:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n: {==}
    case 9n: {==}
    case 10n: {==}
    case 11n: {==}
    case 12n: {==}
    case 13n: {==}
    case 14n: {==}
    case 15n: {==}
    case 16n: {==}
    case 17n: {==}
    case 18n: {==}
    case 19n: {==}
    case 20n: {==}
    case 21n: {==}
    case 22n: {==}
    case 23n: {==}
    case 24n: {==}
    case 25n: {==}
    case 26n: {==}
    case 27n: {==}
    case 28n: {==}
    case 29n: {==}
    case 30n: {==}
    case 31n: {==}
    case 32n: {==}
    case 33n: {==}
    case 34n: {==}
    case 35n: {==}
    case 36n: {==}
    case 37n: {==}
    case 38n: {==}
    case 39n: {==}
    case 40n: {==}
    case 41n: {==}
    case 42n: {==}
    case 43n: {==}
    case 44n: {==}
    case 45n: {==}
    case 46n: {==}
    case 47n: {==}
    case 48n: {==}
    case 49n: {==}
    case 50n: {==}
    case 51n: {==}
    case 52n: {==}
    case 53n: {==}
    case 54n: {==}
    case 55n: {==}
    case 56n: {==}
    case 57n: {==}
    case 58n: {==}
    case 59n: {==}
    case 60n: {==}
    case 61n: {==}
    case 62n: {==}
    case 63n: {==}
    case 64n+p: Domain.impossible({U64.ctz(U64.bit(64n+p)) == U32.from_nat(64n+p) : U32},p,e)

def selected(+b: Chess.Board,+side: U32,+sq: Nat,bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  one: {U64.and(Chess.get_kings(b),Chess.color(b,side)) == U64.bit(sq) : U64}) ->
  {U64.ctz(U64.and(Chess.get_kings(b),Chess.color(b,side))) == U32.from_nat(sq) : U32}:
  Equal.trans(U32,U64.ctz(U64.and(Chess.get_kings(b),Chess.color(b,side))),U64.ctz(U64.bit(sq)),U32.from_nat(sq),
    Equal.cong(U64,U32,x => U64.ctz(x),U64.and(Chess.get_kings(b),Chess.color(b,side)),U64.bit(sq),one),bit_index(sq,bound))

def checked(table: Array<U64>,+b: Chess.Board,+side: U32,+sq: Nat,bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  one: {U64.and(Chess.get_kings(b),Chess.color(b,side)) == U64.bit(sq) : U64}) ->
  {Chess.in_check(table,b,side) == Q.run(table,b,U32.from_nat(sq),U32.xor(side,1)) : Array<U64> & Bool}:
  %Equal.sym(U32,U64.ctz(U64.and(Chess.get_kings(b),Chess.color(b,side))),U32.from_nat(sq),selected(b,side,sq,bound,one)) :
    {Chess.attacked(table,b,_,U32.xor(side,1)) == Q.run(table,b,U32.from_nat(sq),U32.xor(side,1)) : Array<U64> & Bool}
  Flow.attacked(table,b,U32.from_nat(sq),U32.xor(side,1))
