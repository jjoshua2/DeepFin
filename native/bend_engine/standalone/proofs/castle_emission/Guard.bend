import Base
import ../../../legal_probe/Chess.bend as Chess
import ./Spec.bend as S
import ./Fresh.bend as H
import ./Membership.bend as M
import ../board/Spec.bend as B
import ../ray/Frame.bend as F
import ../castling/Route.bend as R
import ../castling/PROOF.bend as CastlingProof
import ../castling/LAWS.bend as Castling

def route_move(+white: Bool,+ks: Bool) ->
  {Chess.Ply{U32.add(Bool.pick(U32,white,0,56),4),U32.add(Bool.pick(U32,white,0,56),Bool.pick(U32,ks,6,2)),0,2} ==
    Chess.Ply{R.src(S.choose(white,ks)),R.dst(S.choose(white,ks)),0,2} : Chess.Ply}:
  match white ks:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

# Conversion certificate for Fresh.subset's structural Boolean representation.
def route_subset(+white: Bool,+ks: Bool) ->
  {H.subset(R.rook(S.choose(white,ks)),U64.shln(U64.from_u32(Bool.pick(U32,ks,96,14)),U32.to_nat(Bool.pick(U32,white,0,56)))) == True{} : Bool}:
  match white ks:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

def clear(+b: Chess.Board,+ks: Bool,e: {S.guard(b,ks) == True{} : Bool}) -> {S.path_clear(b,ks) == True{} : Bool}:
  F.right(S.rook_ok(b,ks),S.path_clear(b,ks),
    F.right(S.king_ok(b),Bool.and(S.rook_ok(b,ks),S.path_clear(b,ks)),
      F.right(S.rights_ok(b,ks),Bool.and(S.king_ok(b),Bool.and(S.rook_ok(b,ks),S.path_clear(b,ks))),e)))

def fresh(+b: Chess.Board,+ks: Bool,e: {S.guard(b,ks) == True{} : Bool}) ->
  {B.fresh_mask(b,R.rook(S.route(b,ks))) == True{} : Bool}:
  H.shrink(b,R.rook(S.route(b,ks)),S.between(b,ks),
    Equal.trans(Bool,B.fresh_mask(b,S.between(b,ks)),S.path_clear(b,ks),True{},
      Equal.sym(Bool,S.path_clear(b,ks),B.fresh_mask(b,S.between(b,ks)),H.zero_board(b,S.between(b,ks))),clear(b,ks,e)),
    route_subset(U32.is_eq(Chess.get_turn(b),1),ks))

def preserves(+b: Chess.Board,+ks: Bool,good: {B.valid(b) == True{} : Bool},e: {S.guard(b,ks) == True{} : Bool}) ->
  {B.valid(Chess.make_move(b,S.move(b,ks))) == True{} : Bool}:
  Equal.trans(Bool,B.valid(Chess.make_move(b,S.move(b,ks))),
    B.valid(Chess.make_move(b,Chess.Ply{R.src(S.route(b,ks)),R.dst(S.route(b,ks)),0,2})),True{},
    Equal.cong(Chess.Ply,Bool,m => B.valid(Chess.make_move(b,m)),S.move(b,ks),
      Chess.Ply{R.src(S.route(b,ks)),R.dst(S.route(b,ks)),0,2},route_move(U32.is_eq(Chess.get_turn(b),1),ks)),
    Castling.castling_make_move_preserves_representation(b,S.route(b,ks),good,fresh(b,ks,e)))

def consume(+b: Chess.Board,+ks: Bool,+m: Chess.Ply,good: {B.valid(b) == True{} : Bool},p: {S.guard(b,ks) == True{} : Bool} & {m == S.move(b,ks) : Chess.Ply}) ->
  {B.valid(Chess.make_move(b,m)) == True{} : Bool}:
  (guard,same) = p
  Equal.trans(Bool,B.valid(Chess.make_move(b,m)),B.valid(Chess.make_move(b,S.move(b,ks))),True{},
    Equal.cong(Chess.Ply,Bool,x => B.valid(Chess.make_move(b,x)),m,S.move(b,ks),same),preserves(b,ks,good,guard))

def emitted(+b: Chess.Board,+ks: Bool,table: Array<U64>,+tail: List<&2,Chess.Ply>,+m: Chess.Ply,
  good: {B.valid(b) == True{} : Bool},here: S.member(m,S.moves(Chess.castle_side(b,ks,(table,tail)))),absent: S.member(m,tail) -> Empty) ->
  {B.valid(Chess.make_move(b,m)) == True{} : Bool}:
  consume(b,ks,m,good,M.side(b,ks,table,tail,m,here,absent))
