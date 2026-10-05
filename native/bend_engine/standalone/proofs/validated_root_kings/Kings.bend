import Base
import ../bitboard_inventory/ClearStep.bend as Clear
import ../bitboard_zero_bridge/Bridge.bend as Zero
import ../bitboard_ctz_structural/WordCtzCore.bend as Ctz
import ../bitboard_membership/IndexEquality.bend as Eq
import ../u64/Words.bend as Old
import ../ray/Frame.bend as Frame
import ../prepared_king_selection/Selection.bend as K
import ../prepared_king_selection/consumer.bend as Prior
import ../prepared_ray_coverage/Query.bend as Q
import ../../../legal_probe/Chess.bend as Chess
import ../../Position.bend as Pos

def zero_count(+n: Nat,+w: Word(n),e: {Clear.is_zero_word(n,w) == True{} : Bool}) ->
  {Word.count(n,w) == 0n : Nat}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w:
        case WCon{False{},tail}: zero_count(p,tail,e)
        case WCon{True{},tail}: Old.false_true({Word.count(1n+p,w) == 0n : Nat},e)

def zero_popcount(+a: U64,e: {U64.is_zero(a) == True{} : Bool}) ->
  {U64.popcount(a) == 0 : U32}:
  Equal.cong(Nat,U32,x => U32.from_nat(x),Word.count(64n,U64.to_word(a)),0n,
    zero_count(64n,U64.to_word(a),Equal.trans(Bool,Clear.is_zero_word(64n,U64.to_word(a)),U64.is_zero(a),True{},
      Equal.sym(Bool,U64.is_zero(a),Clear.is_zero_word(64n,U64.to_word(a)),Zero.u64_is_zero_word(a)),e)))

def count_cases(+a: U64,empty: Bool,same: {U64.is_zero(a) == empty : Bool},
  one: {U64.popcount(a) == 1 : U32}) -> {U64.is_zero(a) == False{} : Bool}:
  match empty:
    case False{}: same
    case True{}:
      Old.false_true({U64.is_zero(a) == False{} : Bool},
        Equal.cong(U32,Bool,x => U32.is_eq(x,1),0,1,
          Equal.trans(U32,0,U64.popcount(a),1,
            Equal.sym(U32,U64.popcount(a),0,zero_popcount(a,same)),one)))

def count_one_nonempty(+a: U64,one: {U64.popcount(a) == 1 : U32}) ->
  {U64.is_zero(a) == False{} : Bool}:
  count_cases(a,U64.is_zero(a),{==},one)

def material(+b: Chess.Board) -> Bool:
  Bool.and(U32.is_le(U64.popcount(Chess.get_white(b)),16),
    Bool.and(U32.is_le(U64.popcount(Chess.get_black(b)),16),
      Bool.and(U32.is_le(U64.popcount(U64.and(Chess.get_pawns(b),Chess.get_white(b))),8),
        Bool.and(U32.is_le(U64.popcount(U64.and(Chess.get_pawns(b),Chess.get_black(b))),8),
          U64.is_zero(U64.and(Chess.get_pawns(b),U64.from_parts(4278190080,255)))))))

def king_counts(+b: Chess.Board,+good: {Pos.structural(b) == True{} : Bool}) ->
  {U64.popcount(U64.and(Chess.get_kings(b),Chess.get_white(b))) == 1 : U32} &
  {U64.popcount(U64.and(Chess.get_kings(b),Chess.get_black(b))) == 1 : U32}:
  +w=U32.is_eq(U64.popcount(U64.and(Chess.get_kings(b),Chess.get_white(b))),1)
  +bl=U32.is_eq(U64.popcount(U64.and(Chess.get_kings(b),Chess.get_black(b))),1)
  (Eq.u32_eq_true_same(U64.popcount(U64.and(Chess.get_kings(b),Chess.get_white(b))),1,
      Frame.left(w,Bool.and(bl,material(b)),good)),
    Eq.u32_eq_true_same(U64.popcount(U64.and(Chess.get_kings(b),Chess.get_black(b))),1,
      Frame.left(bl,material(b),Frame.right(w,Bool.and(bl,material(b)),good))))

def metadata_valid(+b: Chess.Board) -> Bool:
  Bool.and(Pos.valid_ep(Chess.get_ep(b),b),
    Bool.and(Pos.valid_right(b,1,7,4,1),Bool.and(Pos.valid_right(b,2,0,4,1),
      Bool.and(Pos.valid_right(b,4,63,60,0),Pos.valid_right(b,8,56,60,0)))))

def structural(+b: Chess.Board,good: {Pos.valid(b) == True{} : Bool}) ->
  {Pos.structural(b) == True{} : Bool}:
  Frame.left(Pos.structural(b),metadata_valid(b),good)

def choose_plane(+k: U64,+w: U64,+bl: U64,white: Bool,
  ew: {U64.is_zero(U64.and(k,w)) == False{} : Bool},
  eb: {U64.is_zero(U64.and(k,bl)) == False{} : Bool}) ->
  {U64.is_zero(U64.and(k,Chess.select_u64(white,w,bl))) == False{} : Bool}:
  match white:
    case False{}: eb
    case True{}: ew

# No canonical-turn or row-partition premise is needed for actual color selection.
def side_from_counts(+b: Chess.Board,side: U32,
  counts: {U64.popcount(U64.and(Chess.get_kings(b),Chess.get_white(b))) == 1 : U32} &
    {U64.popcount(U64.and(Chess.get_kings(b),Chess.get_black(b))) == 1 : U32}) ->
  {U64.is_zero(U64.and(Chess.get_kings(b),Chess.color(b,side))) == False{} : Bool}:
  (w,bl)=counts
  choose_plane(Chess.get_kings(b),Chess.get_white(b),Chess.get_black(b),U32.is_eq(side,1),
    count_one_nonempty(U64.and(Chess.get_kings(b),Chess.get_white(b)),w),
    count_one_nonempty(U64.and(Chess.get_kings(b),Chess.get_black(b)),bl))

def side_nonempty(+b: Chess.Board,side: U32,good: {Pos.valid(b) == True{} : Bool}) ->
  {U64.is_zero(U64.and(Chess.get_kings(b),Chess.color(b,side))) == False{} : Bool}:
  side_from_counts(b,side,king_counts(b,structural(b,good)))

def selected(-b: Chess.Board) -> Type:
  {Nat.is_lt(K.index(b),64n) == True{} : Bool} &
  ({Q.king(b) == U32.from_nat(K.index(b)) : U32} &
  ({U64.test_bit(Chess.get_kings(b),K.index(b)) == True{} : Bool} &
    {U64.test_bit(Chess.color(b,Chess.get_turn(b)),K.index(b)) == True{} : Bool}))

def selection(+b: Chess.Board,good: {Pos.valid(b) == True{} : Bool}) -> selected(b):
  Prior.use_selection(b,side_nonempty(b,Chess.get_turn(b),good))

def query_index(+b: Chess.Board,side: U32) -> Nat:
  U32.to_nat(U64.ctz(U64.and(Chess.get_kings(b),Chess.color(b,side))))

def safe_query(-b: Chess.Board,-side: U32) -> Type:
  {Nat.is_lt(query_index(b,side),64n) == True{} : Bool}

def query_bound(+b: Chess.Board,+side: U32,good: {Pos.valid(b) == True{} : Bool}) -> safe_query(b,side):
  Ctz.actual_ctz_range(U64.and(Chess.get_kings(b),Chess.color(b,side)),side_nonempty(b,side,good))

def required(+b: Chess.Board,+rays: U64,+dst: U32,+promotion: U32,+flag: U32,
  good: {Pos.valid(b) == True{} : Bool}) ->
  {Chess.filter_requires(U64.and(Chess.color(b,Chess.get_turn(b)),U64.or(rays,Chess.get_kings(b))),
    Chess.Ply{Q.king(b),dst,promotion,flag}) == True{} : Bool}:
  Prior.use_required(b,rays,dst,promotion,flag,side_nonempty(b,Chess.get_turn(b),good))
