# Initial rook-target emptiness is an explicit input condition, not inferred legality.
import Base
import ./Actual.bend as C
import ./Fresh.bend as Fresh
import ./Route.bend as R
import ../move_update/Spec.bend as S
import ../move_update/Clear.bend as Clear
import ../move_update/Selection.bend as Selection
import ../move_update/Ordinary.bend as Ordinary
import ../board/Spec.bend as B
import ../board/Actual.bend as BA
import ../../../legal_probe/Chess.bend as Chess

def first(+b: Chess.Board,+r: R.Route) -> Chess.Board:
  B.insert_mask(S.erase(b,C.mask(R.src(r),R.dst(r))),R.target(r),Selection.piece(b,R.src(r)),U32.is_eq(Chess.get_turn(b),1))

def first_valid(+b: Chess.Board,+r: R.Route,good: {B.valid(b) == True{} : Bool}) -> {B.valid(first(b,r)) == True{} : Bool}:
  Equal.trans(Bool,B.valid(first(b,r)),B.valid(S.replace(b,R.source(r),R.target(r),Selection.piece(b,R.src(r)),U32.is_eq(Chess.get_turn(b),1))),True{},
    Equal.cong(U64,Bool,m => B.valid(B.insert_mask(S.erase(b,m),R.target(r),Selection.piece(b,R.src(r)),U32.is_eq(Chess.get_turn(b),1))),C.mask(R.src(r),R.dst(r)),S.removal(R.source(r),R.target(r)),R.masks(r)),
    Clear.replace(b,R.source(r),R.target(r),Selection.piece(b,R.src(r)),U32.is_eq(Chess.get_turn(b),1),good))

def actual(+b: Chess.Board,+r: R.Route,good: {B.valid(b) == True{} : Bool},
  fresh: {B.fresh_mask(b,R.rook(r)) == True{} : Bool}) ->
  {B.valid(Chess.make_move(b,Chess.Ply{R.src(r),R.dst(r),0,2})) == True{} : Bool}:
  Equal.trans(Bool,B.valid(Chess.make_move(b,Chess.Ply{R.src(r),R.dst(r),0,2})),B.valid(C.expected(b,R.src(r),R.dst(r))),True{},
    Equal.cong(Chess.Board,Bool,x => B.valid(x),Chess.make_move(b,Chess.Ply{R.src(r),R.dst(r),0,2}),C.expected(b,R.src(r),R.dst(r)),C.actual(b,R.src(r),R.dst(r))),
    Equal.trans(Bool,B.valid(C.expected(b,R.src(r),R.dst(r))),B.valid(B.insert_mask(first(b,r),R.rook(r),B.Rook{},U32.is_eq(Chess.get_turn(b),1))),True{},
      BA.metadata(B.insert_mask(first(b,r),R.rook(r),B.Rook{},U32.is_eq(Chess.get_turn(b),1)),U32.xor(Chess.get_turn(b),1),Ordinary.rights(b,R.src(r),R.dst(r),B.tag(Selection.piece(b,R.src(r)))),Ordinary.ep(R.src(r),R.dst(r),B.tag(Selection.piece(b,R.src(r))))),
      BA.mask_valid(first(b,r),R.rook(r),B.Rook{},U32.is_eq(Chess.get_turn(b),1),first_valid(b,r,good),
        Fresh.after(b,C.mask(R.src(r),R.dst(r)),R.target(r),R.rook(r),Selection.piece(b,R.src(r)),U32.is_eq(Chess.get_turn(b),1),fresh,R.distinct(r)))))
