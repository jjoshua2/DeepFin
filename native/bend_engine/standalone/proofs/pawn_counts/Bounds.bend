import Base
import ./Actual.bend as A
import ../color_counts/U64Count.bend as U
import ../color_counts/Naturals.bend as N
import ../u64/Words.bend as Old
import ../../../legal_probe/Chess.bend as Chess

def valid(-b: Chess.Board) -> Type:
  {U32.is_le(U64.popcount(U64.and(Chess.get_pawns(b),Chess.get_white(b))),8) == True{} : Bool} &
  {U32.is_le(U64.popcount(U64.and(Chess.get_pawns(b),Chess.get_black(b))),8) == True{} : Bool}

# Finite count-value bridge, not an enumeration of boards or bitboards.
def to_nat8(+k: Nat,bound: {Nat.is_le(k,64n) == True{} : Bool},
  known: {U32.is_le(U32.from_nat(k),8) == True{} : Bool}) -> {Nat.is_le(k,8n) == True{} : Bool}:
  match k:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n: {==}
    case 9n: Old.false_true({Nat.is_le(9n,8n) == True{} : Bool},known)
    case 10n: Old.false_true({Nat.is_le(10n,8n) == True{} : Bool},known)
    case 11n: Old.false_true({Nat.is_le(11n,8n) == True{} : Bool},known)
    case 12n: Old.false_true({Nat.is_le(12n,8n) == True{} : Bool},known)
    case 13n: Old.false_true({Nat.is_le(13n,8n) == True{} : Bool},known)
    case 14n: Old.false_true({Nat.is_le(14n,8n) == True{} : Bool},known)
    case 15n: Old.false_true({Nat.is_le(15n,8n) == True{} : Bool},known)
    case 16n: Old.false_true({Nat.is_le(16n,8n) == True{} : Bool},known)
    case 17n: Old.false_true({Nat.is_le(17n,8n) == True{} : Bool},known)
    case 18n: Old.false_true({Nat.is_le(18n,8n) == True{} : Bool},known)
    case 19n: Old.false_true({Nat.is_le(19n,8n) == True{} : Bool},known)
    case 20n: Old.false_true({Nat.is_le(20n,8n) == True{} : Bool},known)
    case 21n: Old.false_true({Nat.is_le(21n,8n) == True{} : Bool},known)
    case 22n: Old.false_true({Nat.is_le(22n,8n) == True{} : Bool},known)
    case 23n: Old.false_true({Nat.is_le(23n,8n) == True{} : Bool},known)
    case 24n: Old.false_true({Nat.is_le(24n,8n) == True{} : Bool},known)
    case 25n: Old.false_true({Nat.is_le(25n,8n) == True{} : Bool},known)
    case 26n: Old.false_true({Nat.is_le(26n,8n) == True{} : Bool},known)
    case 27n: Old.false_true({Nat.is_le(27n,8n) == True{} : Bool},known)
    case 28n: Old.false_true({Nat.is_le(28n,8n) == True{} : Bool},known)
    case 29n: Old.false_true({Nat.is_le(29n,8n) == True{} : Bool},known)
    case 30n: Old.false_true({Nat.is_le(30n,8n) == True{} : Bool},known)
    case 31n: Old.false_true({Nat.is_le(31n,8n) == True{} : Bool},known)
    case 32n: Old.false_true({Nat.is_le(32n,8n) == True{} : Bool},known)
    case 33n: Old.false_true({Nat.is_le(33n,8n) == True{} : Bool},known)
    case 34n: Old.false_true({Nat.is_le(34n,8n) == True{} : Bool},known)
    case 35n: Old.false_true({Nat.is_le(35n,8n) == True{} : Bool},known)
    case 36n: Old.false_true({Nat.is_le(36n,8n) == True{} : Bool},known)
    case 37n: Old.false_true({Nat.is_le(37n,8n) == True{} : Bool},known)
    case 38n: Old.false_true({Nat.is_le(38n,8n) == True{} : Bool},known)
    case 39n: Old.false_true({Nat.is_le(39n,8n) == True{} : Bool},known)
    case 40n: Old.false_true({Nat.is_le(40n,8n) == True{} : Bool},known)
    case 41n: Old.false_true({Nat.is_le(41n,8n) == True{} : Bool},known)
    case 42n: Old.false_true({Nat.is_le(42n,8n) == True{} : Bool},known)
    case 43n: Old.false_true({Nat.is_le(43n,8n) == True{} : Bool},known)
    case 44n: Old.false_true({Nat.is_le(44n,8n) == True{} : Bool},known)
    case 45n: Old.false_true({Nat.is_le(45n,8n) == True{} : Bool},known)
    case 46n: Old.false_true({Nat.is_le(46n,8n) == True{} : Bool},known)
    case 47n: Old.false_true({Nat.is_le(47n,8n) == True{} : Bool},known)
    case 48n: Old.false_true({Nat.is_le(48n,8n) == True{} : Bool},known)
    case 49n: Old.false_true({Nat.is_le(49n,8n) == True{} : Bool},known)
    case 50n: Old.false_true({Nat.is_le(50n,8n) == True{} : Bool},known)
    case 51n: Old.false_true({Nat.is_le(51n,8n) == True{} : Bool},known)
    case 52n: Old.false_true({Nat.is_le(52n,8n) == True{} : Bool},known)
    case 53n: Old.false_true({Nat.is_le(53n,8n) == True{} : Bool},known)
    case 54n: Old.false_true({Nat.is_le(54n,8n) == True{} : Bool},known)
    case 55n: Old.false_true({Nat.is_le(55n,8n) == True{} : Bool},known)
    case 56n: Old.false_true({Nat.is_le(56n,8n) == True{} : Bool},known)
    case 57n: Old.false_true({Nat.is_le(57n,8n) == True{} : Bool},known)
    case 58n: Old.false_true({Nat.is_le(58n,8n) == True{} : Bool},known)
    case 59n: Old.false_true({Nat.is_le(59n,8n) == True{} : Bool},known)
    case 60n: Old.false_true({Nat.is_le(60n,8n) == True{} : Bool},known)
    case 61n: Old.false_true({Nat.is_le(61n,8n) == True{} : Bool},known)
    case 62n: Old.false_true({Nat.is_le(62n,8n) == True{} : Bool},known)
    case 63n: Old.false_true({Nat.is_le(63n,8n) == True{} : Bool},known)
    case 64n: Old.false_true({Nat.is_le(64n,8n) == True{} : Bool},known)
    case 65n+p: Old.false_true({Nat.is_le(65n+p,8n) == True{} : Bool},bound)

def from_nat8(+k: Nat,known: {Nat.is_le(k,8n) == True{} : Bool}) ->
  {U32.is_le(U32.from_nat(k),8) == True{} : Bool}:
  match k:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n: {==}
    case 9n+p: Old.false_true({U32.is_le(U32.from_nat(9n+p),8) == True{} : Bool},known)

def preserved(+old: U64,+new: U64,change: {Nat.is_le(U.count(new),U.count(old)) == True{} : Bool},
  parent: {U32.is_le(U64.popcount(old),8) == True{} : Bool}) -> {U32.is_le(U64.popcount(new),8) == True{} : Bool}:
  from_nat8(U.count(new),N.transitive(U.count(new),U.count(old),8n,change,
    to_nat8(U.count(old),Old.count_bound(64n,U64.to_word(old)),parent)))

def post(+b: Chess.Board,+m: Chess.Ply,parent: valid(b),changes: A.decreases(b,m)) -> valid(Chess.make_move(b,m)):
  (pw,pb)=parent
  (cw,cb)=changes
  (preserved(U64.and(Chess.get_pawns(b),Chess.get_white(b)),U64.and(Chess.get_pawns(Chess.make_move(b,m)),Chess.get_white(Chess.make_move(b,m))),cw,pw),
    preserved(U64.and(Chess.get_pawns(b),Chess.get_black(b)),U64.and(Chess.get_pawns(Chess.make_move(b,m)),Chess.get_black(Chess.make_move(b,m))),cb,pb))
