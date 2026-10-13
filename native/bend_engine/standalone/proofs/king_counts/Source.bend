import Base
import ../decoder/consumer.bend as Decoder
import ../decoder/Spec.bend as Row
import ../decoder/Projection.bend as Projection
import ../board/Spec.bend as Board
import ../ep_king_frame/Rows.bend as Pawn
import ../ep_king_frame/Parent.bend as EP
import ../occupancy_insertion/Lattice.bend as L
import ../u64/Words.bend as Old
import ../../../legal_probe/Chess.bend as Chess

def king(row: Board.Row) -> Bool:
  Board.Row{p,n,b,r,q,k,w,bl}=row
  k
def black(row: Board.Row) -> Bool:
  Board.Row{p,n,b,r,q,k,w,bl}=row
  bl

def pick_test(+w: U64,+bl: U64,+i: Nat,side: Bool) ->
  {U64.test_bit(Chess.select_u64(side,w,bl),i) == Bool.pick(Bool,side,U64.test_bit(w,i),U64.test_bit(bl,i)) : Bool}:
  match side:
    case True{}: {==}
    case False{}: {==}
def or_true(+w: Bool) -> {Bool.or(w,True{}) == True{} : Bool}:
  match w:
    case True{}: {==}
    case False{}: {==}
def picked_or(+w: Bool,+bl: Bool,side: Bool,owned: {Bool.pick(Bool,side,w,bl) == True{} : Bool}) ->
  {Bool.or(w,bl) == True{} : Bool}:
  match side:
    case True{}:
      %Equal.sym(Bool,w,True{},owned) : {Bool.or(_,bl) == True{} : Bool}
      {==}
    case False{}:
      %Equal.sym(Bool,bl,True{},owned) : {Bool.or(w,_) == True{} : Bool}
      or_true(w)

def present(+b: Chess.Board,+s: U32,owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)) == True{} : Bool}) ->
  {U64.test_bit(Chess.occupied(b),U32.to_nat(s)) == True{} : Bool}:
  Equal.trans(Bool,U64.test_bit(Chess.occupied(b),U32.to_nat(s)),
    Bool.or(U64.test_bit(Chess.get_white(b),U32.to_nat(s)),U64.test_bit(Chess.get_black(b),U32.to_nat(s))),True{},
    L.test_or(Chess.get_white(b),Chess.get_black(b),U32.to_nat(s)),
    picked_or(U64.test_bit(Chess.get_white(b),U32.to_nat(s)),U64.test_bit(Chess.get_black(b),U32.to_nat(s)),U32.is_eq(Chess.get_turn(b),1),
      Equal.trans(Bool,Bool.pick(Bool,U32.is_eq(Chess.get_turn(b),1),U64.test_bit(Chess.get_white(b),U32.to_nat(s)),U64.test_bit(Chess.get_black(b),U32.to_nat(s))),
        U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)),True{},
        Equal.sym(Bool,U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)),
          Bool.pick(Bool,U32.is_eq(Chess.get_turn(b),1),U64.test_bit(Chess.get_white(b),U32.to_nat(s)),U64.test_bit(Chess.get_black(b),U32.to_nat(s))),
          pick_test(Chess.get_white(b),Chess.get_black(b),U32.to_nat(s),U32.is_eq(Chess.get_turn(b),1))),owned)))

def row(+b: Chess.Board,+s: U32,+good: {Board.valid(b) == True{} : Bool},
  +sb: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},
  owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)) == True{} : Bool}) ->
  {Row.encode(Row.Occupant{Chess.piece(b,s),U64.test_bit(Chess.get_white(b),U32.to_nat(s))}) == Row.row(b,s) : Board.Row}:
  Decoder.reconstruct_occupied_row(b,s,good,sb,present(b,s,owned))

def decoded(+b: Chess.Board,+s: U32,good: {Board.valid(b) == True{} : Bool},
  sb: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},
  owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)) == True{} : Bool}) ->
  {U32.is_eq(Chess.piece(b,s),5) == U64.test_bit(Chess.get_kings(b),U32.to_nat(s)) : Bool}:
  Equal.cong(Board.Row,Bool,king,Row.encode(Row.Occupant{Chess.piece(b,s),U64.test_bit(Chess.get_white(b),U32.to_nat(s))}),Row.row(b,s),row(b,s,good,sb,owned))

def color_pair(+w: Bool,+bl: Bool,side: Bool,
  +owned: {Bool.pick(Bool,side,w,bl) == True{} : Bool},same: {bl == Bool.not(w) : Bool}) ->
  {w == side : Bool} & {bl == Bool.not(side) : Bool}:
  match side:
    case True{}:
      (owned,Equal.trans(Bool,bl,Bool.not(w),False{},same,Equal.cong(Bool,Bool,Bool.not,w,True{},owned)))
    case False{}:
      (EP.not_true(w,Equal.trans(Bool,Bool.not(w),bl,True{},Equal.sym(Bool,bl,Bool.not(w),same),owned)),owned)

def colors(+b: Chess.Board,+s: U32,good: {Board.valid(b) == True{} : Bool},
  sb: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},
  +owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)) == True{} : Bool}) ->
  {U64.test_bit(Chess.get_white(b),U32.to_nat(s)) == U32.is_eq(Chess.get_turn(b),1) : Bool} &
  {U64.test_bit(Chess.get_black(b),U32.to_nat(s)) == Bool.not(U32.is_eq(Chess.get_turn(b),1)) : Bool}:
  color_pair(U64.test_bit(Chess.get_white(b),U32.to_nat(s)),U64.test_bit(Chess.get_black(b),U32.to_nat(s)),U32.is_eq(Chess.get_turn(b),1),
    Equal.trans(Bool,Bool.pick(Bool,U32.is_eq(Chess.get_turn(b),1),U64.test_bit(Chess.get_white(b),U32.to_nat(s)),U64.test_bit(Chess.get_black(b),U32.to_nat(s))),
      U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)),True{},
      Equal.sym(Bool,U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)),
        Bool.pick(Bool,U32.is_eq(Chess.get_turn(b),1),U64.test_bit(Chess.get_white(b),U32.to_nat(s)),U64.test_bit(Chess.get_black(b),U32.to_nat(s))),
        pick_test(Chess.get_white(b),Chess.get_black(b),U32.to_nat(s),U32.is_eq(Chess.get_turn(b),1))),owned),
    Equal.sym(Bool,Bool.not(U64.test_bit(Chess.get_white(b),U32.to_nat(s))),U64.test_bit(Chess.get_black(b),U32.to_nat(s)),
      Equal.cong(Board.Row,Bool,black,Row.encode(Row.Occupant{Chess.piece(b,s),U64.test_bit(Chess.get_white(b),U32.to_nat(s))}),Row.row(b,s),row(b,s,good,sb,owned))))

def pawn_clear(+b: Chess.Board,+s: U32,good: {Board.valid(b) == True{} : Bool},
  sb: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},pawn: {U64.test_bit(Chess.get_pawns(b),U32.to_nat(s)) == True{} : Bool}) ->
  {U64.test_bit(Chess.get_kings(b),U32.to_nat(s)) == False{} : Bool}:
  Pawn.pawn_clear(Row.row(b,s),Projection.valid(b,s,good,sb),pawn)
