import Base
import ../prepared_ray_coverage/Query.bend as Q
import ../bitboard_ctz_structural/WordCtzCore.bend as Ctz
import ../bitboard_ctz_structural/TestBitBridge.bend as Test
import ../layout/Domain.bend as Domain
import ../castle_kings/Kind.bend as Kind
import ../castle_kings/Rows.bend as Rows
import ../board/Spec.bend as Board
import ../decoder/Spec.bend as Decoder
import ../decoder/Bits.bend as Bits
import ../decoder/Projection.bend as Projection
import ../ray/Frame.bend as Frame
import ../castle_safety/Mask.bend as Mask
import ../castle_safety/Spec.bend as Sensitive
import ../../../legal_probe/Chess.bend as Chess

# This is precisely the mask selected by production filter_checked(False).
# Board.valid is a partition invariant, so presence is kept separate.
def plane(+b: Chess.Board) -> U64:
  U64.and(Chess.get_kings(b),Chess.color(b,Chess.get_turn(b)))
def index(+b: Chess.Board) -> Nat:
  U32.to_nat(Q.king(b))

def bound(+b: Chess.Board,nonempty: {U64.is_zero(plane(b)) == False{} : Bool}) ->
  {Nat.is_lt(index(b),64n) == True{} : Bool}:
  Ctz.actual_ctz_range(plane(b),nonempty)

def chosen(+b: Chess.Board,+nonempty: {U64.is_zero(plane(b)) == False{} : Bool}) ->
  {Q.king(b) == U32.from_nat(index(b)) : U32}:
  Domain.injective(Q.king(b),U32.from_nat(index(b)),
    Equal.sym(Nat,U32.to_nat(U32.from_nat(index(b))),index(b),
      Ctz.exact_u32_below64(index(b),bound(b,nonempty))))

def present(+b: Chess.Board,nonempty: {U64.is_zero(plane(b)) == False{} : Bool}) ->
  {U64.test_bit(plane(b),index(b)) == True{} : Bool}:
  Test.actual_ctz_selects_bit(plane(b),nonempty)

def king_bit(+b: Chess.Board,+nonempty: {U64.is_zero(plane(b)) == False{} : Bool}) ->
  {U64.test_bit(Chess.get_kings(b),index(b)) == True{} : Bool}:
  Kind.test_left(Chess.get_kings(b),Chess.color(b,Chess.get_turn(b)),index(b),
    bound(b,nonempty),present(b,nonempty))

# The right AND projection is structural, including both U64 limbs.
def word_right(+n: Nat,+a: Word(n),+b: Word(n),+i: Nat,
  e: {Bits.get(n,Word.and(n,a,b),i) == True{} : Bool}) ->
  {Bits.get(n,b,i) == True{} : Bool}:
  match n:
    case 0n: e
    case 1n+p:
      match a b i:
        case WCon{+ah,at} WCon{+bh,bt} 0n: Frame.right(ah,bh,e)
        case WCon{ah,at} WCon{bh,bt} 1n+j: word_right(p,at,bt,j,e)

def choose_right(+al: Word(32n),+ah: Word(32n),+bl: Word(32n),+bh: Word(32n),+i: Nat,low: Bool,
  e: {Bool.pick(Bool,low,Bits.get(32n,Word.and(32n,al,bl),i),Bits.get(32n,Word.and(32n,ah,bh),Nat.sub(i,32n))) == True{} : Bool}) ->
  {Bool.pick(Bool,low,Bits.get(32n,bl,i),Bits.get(32n,bh,Nat.sub(i,32n))) == True{} : Bool}:
  match low:
    case True{}: word_right(32n,al,bl,i,e)
    case False{}: word_right(32n,ah,bh,Nat.sub(i,32n),e)

def get64_right(+a: U64,+b: U64,+i: Nat,
  e: {Bits.get64(U64.and(a,b),i) == True{} : Bool}) ->
  {Bits.get64(b,i) == True{} : Bool}:
  match a b:
    case U64{U32{al},U32{ah}} U64{U32{bl},U32{bh}}:
      choose_right(al,ah,bl,bh,i,Nat.is_lt(i,32n),e)

def owned(+b: Chess.Board,+nonempty: {U64.is_zero(plane(b)) == False{} : Bool}) ->
  {U64.test_bit(Chess.color(b,Chess.get_turn(b)),index(b)) == True{} : Bool}:
  Equal.trans(Bool,U64.test_bit(Chess.color(b,Chess.get_turn(b)),index(b)),
    Bits.get64(Chess.color(b,Chess.get_turn(b)),index(b)),True{},
    Bits.test64(Chess.color(b,Chess.get_turn(b)),index(b),bound(b,nonempty)),
    get64_right(Chess.get_kings(b),Chess.color(b,Chess.get_turn(b)),index(b),
      Equal.trans(Bool,Bits.get64(plane(b),index(b)),U64.test_bit(plane(b),index(b)),True{},
        Equal.sym(Bool,U64.test_bit(plane(b),index(b)),Bits.get64(plane(b),index(b)),
          Bits.test64(plane(b),index(b),bound(b,nonempty))),present(b,nonempty))))

# Consistency is used only to exclude earlier decoder kinds colliding with a king.
def kind(+b: Chess.Board,+nonempty: {U64.is_zero(plane(b)) == False{} : Bool},
  good: {Board.valid(b) == True{} : Bool}) ->
  {Chess.piece(b,Q.king(b)) == 5 : U32}:
  Rows.kind(Decoder.row(b,Q.king(b)),
    Projection.valid(b,Q.king(b),good,bound(b,nonempty)),king_bit(b,nonempty))

def required(+b: Chess.Board,+rays: U64,+dst: U32,+promotion: U32,+flag: U32,
  +nonempty: {U64.is_zero(plane(b)) == False{} : Bool}) ->
  {Chess.filter_requires(Sensitive.sensitive(b,rays),Chess.Ply{Q.king(b),dst,promotion,flag}) == True{} : Bool}:
  Mask.owned(b,rays,Q.king(b),dst,promotion,flag,bound(b,nonempty),present(b,nonempty))
