import Base
import ./Fusion.bend as F
import ./Intersection.bend as I
import ./Selector.bend as S
import ../color_counts/Actual.bend as A
import ../color_counts/U64Count.bend as U
import ../backrank_pawns/Mask.bend as M
import ../../../legal_probe/Chess.bend as Chess

def white(+b: Chess.Board) -> Nat: U.count(U64.and(Chess.get_pawns(b),Chess.get_white(b)))
def black(+b: Chess.Board) -> Nat: U.count(U64.and(Chess.get_pawns(b),Chess.get_black(b)))
def decreases(-b: Chess.Board,-m: Chess.Ply) -> Type:
  {Nat.is_le(white(Chess.make_move(b,m)),white(b)) == True{} : Bool} &
  {Nat.is_le(black(Chess.make_move(b,m)),black(b)) == True{} : Bool}
def model(+b: Chess.Board,+m: Chess.Ply,+color: U64,side: Bool) -> U64:
  Chess.update_piece(U64.and(Chess.get_pawns(b),color),M.remove(m),U64.bit(U32.to_nat(A.destination(m))),Bool.and(S.put(b,m),side))
def pawn_actual(+b: Chess.Board,m: Chess.Ply) ->
  {Chess.get_pawns(Chess.make_move(b,m)) == Chess.update_piece(Chess.get_pawns(b),M.remove(m),U64.bit(U32.to_nat(A.destination(m))),S.put(b,m)) : U64}:
  match m:
    case Chess.Ply{s,d,p,f}: {==}
def raw_fusion(+b: Chess.Board,m: Chess.Ply,+color: U64,+side: Bool) ->
  {U64.and(Chess.update_piece(Chess.get_pawns(b),M.remove(m),U64.bit(U32.to_nat(A.destination(m))),S.put(b,m)),
    Chess.update_piece(color,M.remove(m),U64.bit(U32.to_nat(A.destination(m))),side)) == model(b,m,color,side) : U64}:
  match m:
    case Chess.Ply{+s,+d,+p,+f}:
      F.fused(Chess.get_pawns(b),color,U64.bit(U32.to_nat(s)),
        U64.bit(U32.to_nat(Bool.pick(U32,U32.is_eq(f,1),U32.xor(d,8),d))),U64.bit(U32.to_nat(d)),
        Chess.select_u64(U32.is_eq(f,2),U64.bit(U32.to_nat(Bool.pick(U32,U32.is_eq(U32.and(d,7),6),U32.inc(d),U32.sub(d,2)))),U64.zero()),
        S.put(b,Chess.Ply{s,d,p,f}),side)
def frame(+b: Chess.Board,+m: Chess.Ply,+color: U64,+side: Bool,no: {U32.is_eq(A.flag(m),2) == False{} : Bool}) ->
  {U64.and(Chess.get_pawns(Chess.make_move(b,m)),Chess.update_piece(color,M.remove(m),A.added(m),side)) == model(b,m,color,side) : U64}:
  %Equal.sym(U64,Chess.get_pawns(Chess.make_move(b,m)),Chess.update_piece(Chess.get_pawns(b),M.remove(m),U64.bit(U32.to_nat(A.destination(m))),S.put(b,m)),pawn_actual(b,m)) :
    {U64.and(_,Chess.update_piece(color,M.remove(m),A.added(m),side)) == model(b,m,color,side) : U64}
  %Equal.sym(U64,A.added(m),U64.bit(U32.to_nat(A.destination(m))),A.noncastle_added(m,no)) :
    {U64.and(Chess.update_piece(Chess.get_pawns(b),M.remove(m),U64.bit(U32.to_nat(A.destination(m))),S.put(b,m)),
      Chess.update_piece(color,M.remove(m),_,side)) == model(b,m,color,side) : U64}
  raw_fusion(b,m,color,side)
def white_actual(+b: Chess.Board,+m: Chess.Ply,no: {U32.is_eq(A.flag(m),2) == False{} : Bool}) ->
  {U64.and(Chess.get_pawns(Chess.make_move(b,m)),Chess.get_white(Chess.make_move(b,m))) ==
    model(b,m,Chess.get_white(b),U32.is_eq(Chess.get_turn(b),1)) : U64}:
  Equal.trans(U64,U64.and(Chess.get_pawns(Chess.make_move(b,m)),Chess.get_white(Chess.make_move(b,m))),
    U64.and(Chess.get_pawns(Chess.make_move(b,m)),Chess.update_piece(Chess.get_white(b),M.remove(m),A.added(m),U32.is_eq(Chess.get_turn(b),1))),
    model(b,m,Chess.get_white(b),U32.is_eq(Chess.get_turn(b),1)),
    Equal.cong(U64,U64,x => U64.and(Chess.get_pawns(Chess.make_move(b,m)),x),Chess.get_white(Chess.make_move(b,m)),
      Chess.update_piece(Chess.get_white(b),M.remove(m),A.added(m),U32.is_eq(Chess.get_turn(b),1)),A.white_actual(b,m)),
    frame(b,m,Chess.get_white(b),U32.is_eq(Chess.get_turn(b),1),no))
def black_actual(+b: Chess.Board,+m: Chess.Ply,no: {U32.is_eq(A.flag(m),2) == False{} : Bool}) ->
  {U64.and(Chess.get_pawns(Chess.make_move(b,m)),Chess.get_black(Chess.make_move(b,m))) ==
    model(b,m,Chess.get_black(b),Bool.not(U32.is_eq(Chess.get_turn(b),1))) : U64}:
  Equal.trans(U64,U64.and(Chess.get_pawns(Chess.make_move(b,m)),Chess.get_black(Chess.make_move(b,m))),
    U64.and(Chess.get_pawns(Chess.make_move(b,m)),Chess.update_piece(Chess.get_black(b),M.remove(m),A.added(m),Bool.not(U32.is_eq(Chess.get_turn(b),1)))),
    model(b,m,Chess.get_black(b),Bool.not(U32.is_eq(Chess.get_turn(b),1))),
    Equal.cong(U64,U64,x => U64.and(Chess.get_pawns(Chess.make_move(b,m)),x),Chess.get_black(Chess.make_move(b,m)),
      Chess.update_piece(Chess.get_black(b),M.remove(m),A.added(m),Bool.not(U32.is_eq(Chess.get_turn(b),1))),A.black_actual(b,m)),
    frame(b,m,Chess.get_black(b),Bool.not(U32.is_eq(Chess.get_turn(b),1)),no))

def planes(+pawns: U64,+w: U64,+bl: U64,+remove: U64,+src: Nat,+dst: Nat,turn: Bool,put: Bool,
  +source_bound: {Nat.is_lt(src,64n) == True{} : Bool},target_bound: {Nat.is_lt(dst,64n) == True{} : Bool},
  owned: {U64.test_bit(Chess.select_u64(turn,w,bl),src) == True{} : Bool},
  pawn: S.certificate(pawns,src,put),erased: {U64.test_bit(remove,src) == True{} : Bool}) ->
  {Nat.is_le(U.count(Chess.update_piece(U64.and(pawns,w),remove,U64.bit(dst),Bool.and(put,turn))),U.count(U64.and(pawns,w))) == True{} : Bool} &
  {Nat.is_le(U.count(Chess.update_piece(U64.and(pawns,bl),remove,U64.bit(dst),Bool.and(put,Bool.not(turn)))),U.count(U64.and(pawns,bl))) == True{} : Bool}:
  match turn put:
    case True{} True{}:
      (U.replace_le(U64.and(pawns,w),remove,src,dst,source_bound,target_bound,I.owned(pawns,w,src,source_bound,pawn,owned),erased),
        A.inactive(U64.and(pawns,bl),remove,dst))
    case True{} False{}:
      (A.inactive(U64.and(pawns,w),remove,dst),A.inactive(U64.and(pawns,bl),remove,dst))
    case False{} True{}:
      (A.inactive(U64.and(pawns,w),remove,dst),
        U.replace_le(U64.and(pawns,bl),remove,src,dst,source_bound,target_bound,I.owned(pawns,bl,src,source_bound,pawn,owned),erased))
    case False{} False{}:
      (A.inactive(U64.and(pawns,w),remove,dst),A.inactive(U64.and(pawns,bl),remove,dst))

def lift_white(+b: Chess.Board,+m: Chess.Ply,no: {U32.is_eq(A.flag(m),2) == False{} : Bool},
  good: {Nat.is_le(U.count(model(b,m,Chess.get_white(b),U32.is_eq(Chess.get_turn(b),1))),white(b)) == True{} : Bool}) ->
  {Nat.is_le(white(Chess.make_move(b,m)),white(b)) == True{} : Bool}:
  %Equal.sym(U64,U64.and(Chess.get_pawns(Chess.make_move(b,m)),Chess.get_white(Chess.make_move(b,m))),
    model(b,m,Chess.get_white(b),U32.is_eq(Chess.get_turn(b),1)),white_actual(b,m,no)) :
    {Nat.is_le(U.count(_),white(b)) == True{} : Bool}
  good
def lift_black(+b: Chess.Board,+m: Chess.Ply,no: {U32.is_eq(A.flag(m),2) == False{} : Bool},
  good: {Nat.is_le(U.count(model(b,m,Chess.get_black(b),Bool.not(U32.is_eq(Chess.get_turn(b),1)))),black(b)) == True{} : Bool}) ->
  {Nat.is_le(black(Chess.make_move(b,m)),black(b)) == True{} : Bool}:
  %Equal.sym(U64,U64.and(Chess.get_pawns(Chess.make_move(b,m)),Chess.get_black(Chess.make_move(b,m))),
    model(b,m,Chess.get_black(b),Bool.not(U32.is_eq(Chess.get_turn(b),1))),black_actual(b,m,no)) :
    {Nat.is_le(U.count(_),black(b)) == True{} : Bool}
  good
def finish(+b: Chess.Board,+m: Chess.Ply,+no: {U32.is_eq(A.flag(m),2) == False{} : Bool},
  results: {Nat.is_le(U.count(model(b,m,Chess.get_white(b),U32.is_eq(Chess.get_turn(b),1))),white(b)) == True{} : Bool} &
    {Nat.is_le(U.count(model(b,m,Chess.get_black(b),Bool.not(U32.is_eq(Chess.get_turn(b),1)))),black(b)) == True{} : Bool}) -> decreases(b,m):
  (pw,pb)=results
  (lift_white(b,m,no,pw),lift_black(b,m,no,pb))
def post(+b: Chess.Board,+m: Chess.Ply,cert: A.emitted(b,m)) -> decreases(b,m):
  (+sb,(owned,(db,no)))=cert
  +put=S.put(b,m)
  finish(b,m,no,planes(Chess.get_pawns(b),Chess.get_white(b),Chess.get_black(b),M.remove(m),U32.to_nat(A.source(m)),U32.to_nat(A.destination(m)),
    U32.is_eq(Chess.get_turn(b),1),put,sb,db,owned,S.tagged(b,m,put,{==}),A.removed(m,sb)))
