import Base
import ./Actual.bend as A
import ./U64Count.bend as U
import ./Naturals.bend as N
import ../u64/Words.bend as Old
import ../../../legal_probe/Chess.bend as Chess

def valid(-b: Chess.Board) -> Type:
  {U32.is_le(U64.popcount(Chess.get_white(b)),16) == True{} : Bool} &
  {U32.is_le(U64.popcount(Chess.get_black(b)),16) == True{} : Bool}

# Count-value cases bridge the actual U32 comparator; no bitboard enumeration.
def to_nat16(+k: Nat,bound: {Nat.is_le(k,64n) == True{} : Bool},
  known: {U32.is_le(U32.from_nat(k),16) == True{} : Bool}) -> {Nat.is_le(k,16n) == True{} : Bool}:
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
    case 9n: {==}
    case 10n: {==}
    case 11n: {==}
    case 12n: {==}
    case 13n: {==}
    case 14n: {==}
    case 15n: {==}
    case 16n: {==}
    case 17n: Old.false_true({Nat.is_le(17n,16n) == True{} : Bool},known)
    case 18n: Old.false_true({Nat.is_le(18n,16n) == True{} : Bool},known)
    case 19n: Old.false_true({Nat.is_le(19n,16n) == True{} : Bool},known)
    case 20n: Old.false_true({Nat.is_le(20n,16n) == True{} : Bool},known)
    case 21n: Old.false_true({Nat.is_le(21n,16n) == True{} : Bool},known)
    case 22n: Old.false_true({Nat.is_le(22n,16n) == True{} : Bool},known)
    case 23n: Old.false_true({Nat.is_le(23n,16n) == True{} : Bool},known)
    case 24n: Old.false_true({Nat.is_le(24n,16n) == True{} : Bool},known)
    case 25n: Old.false_true({Nat.is_le(25n,16n) == True{} : Bool},known)
    case 26n: Old.false_true({Nat.is_le(26n,16n) == True{} : Bool},known)
    case 27n: Old.false_true({Nat.is_le(27n,16n) == True{} : Bool},known)
    case 28n: Old.false_true({Nat.is_le(28n,16n) == True{} : Bool},known)
    case 29n: Old.false_true({Nat.is_le(29n,16n) == True{} : Bool},known)
    case 30n: Old.false_true({Nat.is_le(30n,16n) == True{} : Bool},known)
    case 31n: Old.false_true({Nat.is_le(31n,16n) == True{} : Bool},known)
    case 32n: Old.false_true({Nat.is_le(32n,16n) == True{} : Bool},known)
    case 33n: Old.false_true({Nat.is_le(33n,16n) == True{} : Bool},known)
    case 34n: Old.false_true({Nat.is_le(34n,16n) == True{} : Bool},known)
    case 35n: Old.false_true({Nat.is_le(35n,16n) == True{} : Bool},known)
    case 36n: Old.false_true({Nat.is_le(36n,16n) == True{} : Bool},known)
    case 37n: Old.false_true({Nat.is_le(37n,16n) == True{} : Bool},known)
    case 38n: Old.false_true({Nat.is_le(38n,16n) == True{} : Bool},known)
    case 39n: Old.false_true({Nat.is_le(39n,16n) == True{} : Bool},known)
    case 40n: Old.false_true({Nat.is_le(40n,16n) == True{} : Bool},known)
    case 41n: Old.false_true({Nat.is_le(41n,16n) == True{} : Bool},known)
    case 42n: Old.false_true({Nat.is_le(42n,16n) == True{} : Bool},known)
    case 43n: Old.false_true({Nat.is_le(43n,16n) == True{} : Bool},known)
    case 44n: Old.false_true({Nat.is_le(44n,16n) == True{} : Bool},known)
    case 45n: Old.false_true({Nat.is_le(45n,16n) == True{} : Bool},known)
    case 46n: Old.false_true({Nat.is_le(46n,16n) == True{} : Bool},known)
    case 47n: Old.false_true({Nat.is_le(47n,16n) == True{} : Bool},known)
    case 48n: Old.false_true({Nat.is_le(48n,16n) == True{} : Bool},known)
    case 49n: Old.false_true({Nat.is_le(49n,16n) == True{} : Bool},known)
    case 50n: Old.false_true({Nat.is_le(50n,16n) == True{} : Bool},known)
    case 51n: Old.false_true({Nat.is_le(51n,16n) == True{} : Bool},known)
    case 52n: Old.false_true({Nat.is_le(52n,16n) == True{} : Bool},known)
    case 53n: Old.false_true({Nat.is_le(53n,16n) == True{} : Bool},known)
    case 54n: Old.false_true({Nat.is_le(54n,16n) == True{} : Bool},known)
    case 55n: Old.false_true({Nat.is_le(55n,16n) == True{} : Bool},known)
    case 56n: Old.false_true({Nat.is_le(56n,16n) == True{} : Bool},known)
    case 57n: Old.false_true({Nat.is_le(57n,16n) == True{} : Bool},known)
    case 58n: Old.false_true({Nat.is_le(58n,16n) == True{} : Bool},known)
    case 59n: Old.false_true({Nat.is_le(59n,16n) == True{} : Bool},known)
    case 60n: Old.false_true({Nat.is_le(60n,16n) == True{} : Bool},known)
    case 61n: Old.false_true({Nat.is_le(61n,16n) == True{} : Bool},known)
    case 62n: Old.false_true({Nat.is_le(62n,16n) == True{} : Bool},known)
    case 63n: Old.false_true({Nat.is_le(63n,16n) == True{} : Bool},known)
    case 64n: Old.false_true({Nat.is_le(64n,16n) == True{} : Bool},known)
    case 65n+p: Old.false_true({Nat.is_le(65n+p,16n) == True{} : Bool},bound)
def from_nat16(+k: Nat,known: {Nat.is_le(k,16n) == True{} : Bool}) ->
  {U32.is_le(U32.from_nat(k),16) == True{} : Bool}:
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
    case 9n: {==}
    case 10n: {==}
    case 11n: {==}
    case 12n: {==}
    case 13n: {==}
    case 14n: {==}
    case 15n: {==}
    case 16n: {==}
    case 17n+p: Old.false_true({U32.is_le(U32.from_nat(17n+p),16) == True{} : Bool},known)

def preserved(+old: U64,+new: U64,change: {Nat.is_le(U.count(new),U.count(old)) == True{} : Bool},
  parent: {U32.is_le(U64.popcount(old),16) == True{} : Bool}) -> {U32.is_le(U64.popcount(new),16) == True{} : Bool}:
  from_nat16(U.count(new),N.transitive(U.count(new),U.count(old),16n,change,
    to_nat16(U.count(old),Old.count_bound(64n,U64.to_word(old)),parent)))
def post(+b: Chess.Board,+m: Chess.Ply,parent: valid(b),changes: A.decreases(b,m)) -> valid(Chess.make_move(b,m)):
  (pw,pb)=parent
  (cw,cb)=changes
  (preserved(Chess.get_white(b),Chess.get_white(Chess.make_move(b,m)),cw,pw),
    preserved(Chess.get_black(b),Chess.get_black(Chess.make_move(b,m)),cb,pb))
