# Partition is a per-square representation premise, separate from accepted roots.
import Base
import ../board/Spec.bend as B
import ../decoder/Spec.bend as Row
import ../decoder/Projection.bend as Projection
import ../decoder/Bits.bend as Bits
import ../candidate_king_destination/Algebra.bend as A
import ../u64/Words.bend as Old
import ../../../legal_probe/Chess.bend as C

def pawn(row: B.Row) -> Bool:
  B.Row{p,n,b,r,q,k,w,bl}=row
  p
def king(row: B.Row) -> Bool:
  B.Row{p,n,b,r,q,k,w,bl}=row
  k

def pawn_clear(+row: B.Row,good: {B.row_valid(row) == True{} : Bool},yes: {pawn(row) == True{} : Bool}) -> {king(row) == False{} : Bool}:
  match row:
    case B.Row{p,n,b,r,q,False{},w,bl}: {==}
    case B.Row{False{},n,b,r,q,True{},w,bl}: Old.false_true({True{} == False{} : Bool},yes)
    case B.Row{True{},False{},False{},False{},False{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},False{},False{},False{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},False{},False{},False{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},False{},False{},False{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},False{},False{},True{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},False{},False{},True{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},False{},False{},True{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},False{},False{},True{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},False{},True{},False{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},False{},True{},False{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},False{},True{},False{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},False{},True{},False{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},False{},True{},True{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},False{},True{},True{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},False{},True{},True{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},False{},True{},True{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},True{},False{},False{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},True{},False{},False{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},True{},False{},False{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},True{},False{},False{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},True{},False{},True{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},True{},False{},True{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},True{},False{},True{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},True{},False{},True{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},True{},True{},False{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},True{},True{},False{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},True{},True{},False{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},True{},True{},False{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},True{},True{},True{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},True{},True{},True{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},True{},True{},True{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},True{},True{},True{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},False{},False{},False{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},False{},False{},False{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},False{},False{},False{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},False{},False{},False{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},False{},False{},True{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},False{},False{},True{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},False{},False{},True{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},False{},False{},True{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},False{},True{},False{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},False{},True{},False{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},False{},True{},False{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},False{},True{},False{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},False{},True{},True{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},False{},True{},True{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},False{},True{},True{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},False{},True{},True{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},True{},False{},False{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},True{},False{},False{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},True{},False{},False{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},True{},False{},False{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},True{},False{},True{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},True{},False{},True{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},True{},False{},True{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},True{},False{},True{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},True{},True{},False{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},True{},True{},False{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},True{},True{},False{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},True{},True{},False{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},True{},True{},True{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},True{},True{},True{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},True{},True{},True{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},True{},True{},True{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)

# The decoder can return zero only through its first pawn branch.
def decoded_pawn(+row: B.Row,zero: {Row.selected_tag(row) == 0 : U32}) -> {pawn(row) == True{} : Bool}:
  match row:
    case B.Row{True{},n,b,r,q,k,w,bl}: {==}
    case B.Row{False{},True{},b,r,q,k,w,bl}:
      Old.false_true({False{} == True{} : Bool},Equal.cong(U32,Bool,U32.is_zero,1,0,zero))
    case B.Row{False{},False{},True{},r,q,k,w,bl}:
      Old.false_true({False{} == True{} : Bool},Equal.cong(U32,Bool,U32.is_zero,2,0,zero))
    case B.Row{False{},False{},False{},True{},q,k,w,bl}:
      Old.false_true({False{} == True{} : Bool},Equal.cong(U32,Bool,U32.is_zero,3,0,zero))
    case B.Row{False{},False{},False{},False{},True{},k,w,bl}:
      Old.false_true({False{} == True{} : Bool},Equal.cong(U32,Bool,U32.is_zero,4,0,zero))
    case B.Row{False{},False{},False{},False{},False{},k,w,bl}:
      Old.false_true({False{} == True{} : Bool},Equal.cong(U32,Bool,U32.is_zero,5,0,zero))

def pawn_square(+b: C.Board,+s: U32,good: {B.valid(b) == True{} : Bool},bound: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},yes: {U64.test_bit(C.get_pawns(b),U32.to_nat(s)) == True{} : Bool}) -> {U64.and(U64.bit(U32.to_nat(s)),C.get_kings(b)) == U64.zero() : U64}:
  A.test_zero(C.get_kings(b),U32.to_nat(s),pawn_clear(Row.row(b,s),Projection.valid(b,s,good,bound),yes))

def decoded_square(+b: C.Board,+s: U32,good: {B.valid(b) == True{} : Bool},bound: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},zero: {C.piece(b,s) == 0 : U32}) -> {U64.and(U64.bit(U32.to_nat(s)),C.get_kings(b)) == U64.zero() : U64}:
  pawn_square(b,s,good,bound,decoded_pawn(Row.row(b,s),zero))

# Actual U64.bit clamps indices at64; an out-of-range test cannot be true.
def and_zero(a: U64) -> {U64.and(a,U64.zero()) == U64.zero() : U64}:
  match a:
    case U64{U32{lo},U32{hi}}:
      Equal.trans(U64,U64{U32{Word.and(32n,lo,Word.zero(32n))},U32{Word.and(32n,hi,Word.zero(32n))}},U64{0,U32{Word.and(32n,hi,Word.zero(32n))}},U64.zero(),
        Equal.cong(Word(32n),U64,x => U64{U32{x},U32{Word.and(32n,hi,Word.zero(32n))}},Word.and(32n,lo,Word.zero(32n)),Word.zero(32n),Bits.and_zero(32n,lo)),
        Equal.cong(Word(32n),U64,x => U64{0,U32{x}},Word.and(32n,hi,Word.zero(32n)),Word.zero(32n),Bits.and_zero(32n,hi)))

def min_zero(p: Nat) -> {Nat.min(p,0n) == 0n : Nat}:
  match p:
    case 0n: {==}
    case 1n+r: {==}

def bit_large(+p: Nat) -> {U64.bit(64n+p) == U64.zero() : U64}:
  Equal.cong(Nat,U64,x => U64.shln.go(64n+x,U64.from_u32(1)),Nat.min(p,0n),0n,min_zero(p))

def beyond_false(+a: U64,+p: Nat) -> {U64.test_bit(a,64n+p) == False{} : Bool}:
  Equal.trans(Bool,U64.test_bit(a,64n+p),Bool.not(U64.is_eq(U64.and(a,U64.zero()),U64.zero())),False{},
    Equal.cong(U64,Bool,x => Bool.not(U64.is_eq(U64.and(a,x),U64.zero())),U64.bit(64n+p),U64.zero(),bit_large(p)),
    Equal.cong(U64,Bool,x => Bool.not(U64.is_eq(x,U64.zero())),U64.and(a,U64.zero()),U64.zero(),and_zero(a)))

def true_bound(+a: U64,i: Nat,yes: {U64.test_bit(a,i) == True{} : Bool}) -> {Nat.is_lt(i,64n) == True{} : Bool}:
  match i:
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
    case 64n+p:
      Old.false_true({Nat.is_lt(64n+p,64n) == True{} : Bool},Equal.trans(Bool,False{},U64.test_bit(a,64n+p),True{},Equal.sym(Bool,U64.test_bit(a,64n+p),False{},beyond_false(a,p)),yes))
