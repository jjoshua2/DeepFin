# Geometry and king clearance derived from rows and the actual producer guard.
import Base
import ./CastleRows.bend as Rows
import ../board/Spec.bend as B
import ../castle_emission/Spec.bend as E
import ../castle_emission/Guard.bend as G
import ../castle_emission/Fresh.bend as H
import ../castle_kings/Fresh.bend as Fresh
import ../castle_kings/Kind.bend as Kind
import ../backrank_pawns/Mask.bend as M
import ../ray/Frame.bend as F
import ../candidate_king_destination/Algebra.bend as A
import ../../../legal_probe/Chess.bend as Chess

def home(white: Bool) -> U32: Bool.pick(U32,white,0,56)
def src(white: Bool) -> U32: U32.add(home(white),4)
def dst(+white: Bool,ks: Bool) -> U32: U32.add(home(white),Bool.pick(U32,ks,6,2))
def rook(+white: Bool,ks: Bool) -> U32: U32.add(home(white),Bool.pick(U32,ks,7,0))
def between(+white: Bool,ks: Bool) -> U64:
  U64.shln(U64.from_u32(Bool.pick(U32,ks,96,14)),U32.to_nat(home(white)))
def sb(+white: Bool) -> {Nat.is_lt(U32.to_nat(src(white)),64n) == True{} : Bool}:
  match white:
    case False{}: {==}
    case True{}: {==}
def db(+white: Bool,+ks: Bool) -> {Nat.is_lt(U32.to_nat(dst(white,ks)),64n) == True{} : Bool}:
  match white ks:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}
def rb(+white: Bool,+ks: Bool) -> {Nat.is_lt(U32.to_nat(rook(white,ks)),64n) == True{} : Bool}:
  match white ks:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}
def destination_subset(+white: Bool,+ks: Bool) ->
  {H.subset(U64.bit(U32.to_nat(dst(white,ks))),between(white,ks)) == True{} : Bool}:
  match white ks:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}
def mask(+white: Bool,+ks: Bool) ->
  {M.remove(Chess.Ply{src(white),dst(white,ks),0,2}) ==
   U64.or(U64.or(U64.bit(U32.to_nat(src(white))),U64.bit(U32.to_nat(dst(white,ks)))),
     U64.or(U64.bit(U32.to_nat(dst(white,ks))),U64.bit(U32.to_nat(rook(white,ks))))) : U64}:
  match white ks:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def path_fresh(+b: Chess.Board,+ks: Bool,guard: {E.guard(b,ks) == True{} : Bool}) ->
  {B.fresh_mask(b,E.between(b,ks)) == True{} : Bool}:
  Equal.trans(Bool,B.fresh_mask(b,E.between(b,ks)),E.path_clear(b,ks),True{},
    Equal.sym(Bool,E.path_clear(b,ks),B.fresh_mask(b,E.between(b,ks)),H.zero_board(b,E.between(b,ks))),
    G.clear(b,ks,guard))

def destination(+b: Chess.Board,+ks: Bool,rows: {B.valid(b) == True{} : Bool},
  guard: {E.guard(b,ks) == True{} : Bool}) ->
  {U64.and(U64.bit(U32.to_nat(E.dst(b,ks))),Chess.get_kings(b)) == U64.zero() : U64}:
  Equal.trans(U64,U64.and(U64.bit(U32.to_nat(E.dst(b,ks))),Chess.get_kings(b)),
    U64.and(Chess.get_kings(b),U64.bit(U32.to_nat(E.dst(b,ks)))),U64.zero(),
    A.and_commute(U64.bit(U32.to_nat(E.dst(b,ks))),Chess.get_kings(b)),
    Fresh.board(b,U64.bit(U32.to_nat(E.dst(b,ks))),rows,
      H.shrink(b,U64.bit(U32.to_nat(E.dst(b,ks))),E.between(b,ks),path_fresh(b,ks,guard),
        destination_subset(U32.is_eq(Chess.get_turn(b),1),ks))))

def king_guard(+b: Chess.Board,+ks: Bool,guard: {E.guard(b,ks) == True{} : Bool}) ->
  {E.king_ok(b) == True{} : Bool}:
  F.left(E.king_ok(b),Bool.and(E.rook_ok(b,ks),E.path_clear(b,ks)),
    F.right(E.rights_ok(b,ks),Bool.and(E.king_ok(b),Bool.and(E.rook_ok(b,ks),E.path_clear(b,ks))),guard))
def rook_guard(+b: Chess.Board,+ks: Bool,guard: {E.guard(b,ks) == True{} : Bool}) ->
  {E.rook_ok(b,ks) == True{} : Bool}:
  F.left(E.rook_ok(b,ks),E.path_clear(b,ks),
    F.right(E.king_ok(b),Bool.and(E.rook_ok(b,ks),E.path_clear(b,ks)),
      F.right(E.rights_ok(b,ks),Bool.and(E.king_ok(b),Bool.and(E.rook_ok(b,ks),E.path_clear(b,ks))),guard)))

def rook_source(+b: Chess.Board,+ks: Bool,rows: {B.valid(b) == True{} : Bool},
  guard: {E.guard(b,ks) == True{} : Bool}) ->
  {U64.and(U64.bit(U32.to_nat(E.rook(b,ks))),Chess.get_kings(b)) == U64.zero() : U64}:
  Rows.square(b,E.rook(b,ks),rows,rb(U32.is_eq(Chess.get_turn(b),1),ks),
    Kind.test_left(Chess.get_rooks(b),Chess.color(b,Chess.get_turn(b)),U32.to_nat(E.rook(b,ks)),
      rb(U32.is_eq(Chess.get_turn(b),1),ks),rook_guard(b,ks,guard)))

def source_owned(+b: Chess.Board,+ks: Bool,guard: {E.guard(b,ks) == True{} : Bool}) ->
  {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(E.src(b))) == True{} : Bool}:
  Kind.test_left(Chess.color(b,Chess.get_turn(b)),Chess.get_kings(b),U32.to_nat(E.src(b)),sb(U32.is_eq(Chess.get_turn(b),1)),
    Equal.trans(Bool,
      U64.test_bit(U64.and(Chess.color(b,Chess.get_turn(b)),Chess.get_kings(b)),U32.to_nat(E.src(b))),
      E.king_ok(b),True{},
      Equal.cong(U64,Bool,x => U64.test_bit(x,U32.to_nat(E.src(b))),
        U64.and(Chess.color(b,Chess.get_turn(b)),Chess.get_kings(b)),
        U64.and(Chess.get_kings(b),Chess.color(b,Chess.get_turn(b))),A.and_commute(Chess.color(b,Chess.get_turn(b)),Chess.get_kings(b))),
      king_guard(b,ks,guard)))
