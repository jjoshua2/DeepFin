import Base
import ./Algebra.bend as Algebra
import ./Count.bend as Count
import ./Frame.bend as Frame
import ./Intersection.bend as I
import ./Selector.bend as S
import ./Source.bend as Source
import ../color_counts/Actual.bend as A
import ../color_counts/U64Count.bend as U
import ../backrank_pawns/Mask.bend as M
import ../board/Spec.bend as Board
import ../../../legal_probe/Chess.bend as Chess

def white(+b: Chess.Board) -> Nat: U.count(U64.and(Chess.get_kings(b),Chess.get_white(b)))
def black(+b: Chess.Board) -> Nat: U.count(U64.and(Chess.get_kings(b),Chess.get_black(b)))
def preserved(-b: Chess.Board,-m: Chess.Ply) -> Type:
  {white(Chess.make_move(b,m)) == white(b) : Nat} & {black(Chess.make_move(b,m)) == black(b) : Nat}
def cap(m: Chess.Ply) -> U32:
  Chess.Ply{s,+d,p,f}=m
  Bool.pick(U32,U32.is_eq(f,1),U32.xor(d,8),d)
def target_clear(-b: Chess.Board,-m: Chess.Ply) -> Type:
  {U64.and(U64.bit(U32.to_nat(A.destination(m))),Chess.get_kings(b)) == U64.zero() : U64}
def cap_clear(-b: Chess.Board,-m: Chess.Ply) -> Type:
  {U64.and(U64.bit(U32.to_nat(cap(m))),Chess.get_kings(b)) == U64.zero() : U64}
def emitted(-b: Chess.Board,-m: Chess.Ply) -> Type:
  A.emitted(b,m) & (S.same(b,m) & (target_clear(b,m) & cap_clear(b,m)))
def simple_remove(+m: Chess.Ply,no: {U32.is_eq(A.flag(m),2) == False{} : Bool}) ->
  {M.remove(m) == U64.or(U64.or(U64.bit(U32.to_nat(A.source(m))),U64.bit(U32.to_nat(cap(m)))),U64.or(U64.bit(U32.to_nat(A.destination(m))),U64.zero())) : U64}:
  Chess.Ply{+s,+d,p,+f}=m
  %Equal.sym(Bool,U32.is_eq(f,2),False{},no) :
    {U64.or(U64.or(U64.bit(U32.to_nat(s)),U64.bit(U32.to_nat(cap(Chess.Ply{s,d,p,f})))),
      U64.or(U64.bit(U32.to_nat(d)),Chess.select_u64(_,U64.bit(U32.to_nat(Bool.pick(U32,U32.is_eq(U32.and(d,7),6),U32.inc(d),U32.sub(d,2)))),U64.zero()))) ==
      U64.or(U64.or(U64.bit(U32.to_nat(s)),U64.bit(U32.to_nat(cap(Chess.Ply{s,d,p,f})))),U64.or(U64.bit(U32.to_nat(d)),U64.zero())) : U64}
  {==}
def plane(+b: Chess.Board,+m: Chess.Ply,+color: U64,+side: Bool,
  +sb: {Nat.is_lt(U32.to_nat(A.source(m)),64n) == True{} : Bool},+db: {Nat.is_lt(U32.to_nat(A.destination(m)),64n) == True{} : Bool},
  no: {U32.is_eq(A.flag(m),2) == False{} : Bool},same: S.same(b,m),
  source_color: {U64.test_bit(color,U32.to_nat(A.source(m))) == side : Bool},
  +dst: {U64.and(U64.bit(U32.to_nat(A.destination(m))),Chess.get_kings(b)) == U64.zero() : U64},cap_good: cap_clear(b,m)) ->
  {U.count(Frame.model(b,m,color,side)) == U.count(U64.and(Chess.get_kings(b),color)) : Nat}:
  %Equal.sym(U64,M.remove(m),U64.or(U64.or(U64.bit(U32.to_nat(A.source(m))),U64.bit(U32.to_nat(cap(m)))),U64.or(U64.bit(U32.to_nat(A.destination(m))),U64.zero())),simple_remove(m,no)) :
    {U.count(Chess.update_piece(U64.and(Chess.get_kings(b),color),_,U64.bit(U32.to_nat(A.destination(m))),Bool.and(S.put(b,m),side))) == U.count(U64.and(Chess.get_kings(b),color)) : Nat}
  %Equal.sym(U64,Algebra.erase(Chess.get_kings(b),color,U64.bit(U32.to_nat(A.source(m))),U64.bit(U32.to_nat(cap(m))),U64.bit(U32.to_nat(A.destination(m)))),
    Algebra.single(Chess.get_kings(b),color,U64.bit(U32.to_nat(A.source(m)))),
    Algebra.strip(Chess.get_kings(b),color,U64.bit(U32.to_nat(A.source(m))),U64.bit(U32.to_nat(cap(m))),U64.bit(U32.to_nat(A.destination(m))),cap_good,dst)) :
    {U.count(U64.or(_,Chess.select_u64(Bool.and(S.put(b,m),side),U64.bit(U32.to_nat(A.destination(m))),U64.zero()))) == U.count(U64.and(Chess.get_kings(b),color)) : Nat}
  Count.update(U64.and(Chess.get_kings(b),color),U32.to_nat(A.source(m)),U32.to_nat(A.destination(m)),Bool.and(S.put(b,m),side),sb,db,
    I.source(Chess.get_kings(b),color,U32.to_nat(A.source(m)),S.put(b,m),side,sb,
      Equal.sym(Bool,S.put(b,m),U64.test_bit(Chess.get_kings(b),U32.to_nat(A.source(m))),same),source_color),
    I.fresh(Chess.get_kings(b),color,U32.to_nat(A.destination(m)),db,Count.clear_test(Chess.get_kings(b),U32.to_nat(A.destination(m)),dst)))
def finish(+b: Chess.Board,+m: Chess.Ply,+sb: {Nat.is_lt(U32.to_nat(A.source(m)),64n) == True{} : Bool},
  +db: {Nat.is_lt(U32.to_nat(A.destination(m)),64n) == True{} : Bool},+no: {U32.is_eq(A.flag(m),2) == False{} : Bool},
  +same: {S.put(b,m) == U64.test_bit(Chess.get_kings(b),U32.to_nat(A.source(m))) : Bool},
  +dst: {U64.and(U64.bit(U32.to_nat(A.destination(m))),Chess.get_kings(b)) == U64.zero() : U64},
  +cap_good: {U64.and(U64.bit(U32.to_nat(cap(m))),Chess.get_kings(b)) == U64.zero() : U64},
  colors: {U64.test_bit(Chess.get_white(b),U32.to_nat(A.source(m))) == U32.is_eq(Chess.get_turn(b),1) : Bool} &
    {U64.test_bit(Chess.get_black(b),U32.to_nat(A.source(m))) == Bool.not(U32.is_eq(Chess.get_turn(b),1)) : Bool}) -> preserved(b,m):
  (cw,cb)=colors
  (Equal.trans(Nat,white(Chess.make_move(b,m)),U.count(Frame.model(b,m,Chess.get_white(b),U32.is_eq(Chess.get_turn(b),1))),white(b),
     Equal.cong(U64,Nat,U.count,U64.and(Chess.get_kings(Chess.make_move(b,m)),Chess.get_white(Chess.make_move(b,m))),Frame.model(b,m,Chess.get_white(b),U32.is_eq(Chess.get_turn(b),1)),Frame.white_actual(b,m,no)),
     plane(b,m,Chess.get_white(b),U32.is_eq(Chess.get_turn(b),1),sb,db,no,same,cw,dst,cap_good)),
   Equal.trans(Nat,black(Chess.make_move(b,m)),U.count(Frame.model(b,m,Chess.get_black(b),Bool.not(U32.is_eq(Chess.get_turn(b),1)))),black(b),
     Equal.cong(U64,Nat,U.count,U64.and(Chess.get_kings(Chess.make_move(b,m)),Chess.get_black(Chess.make_move(b,m))),Frame.model(b,m,Chess.get_black(b),Bool.not(U32.is_eq(Chess.get_turn(b),1))),Frame.black_actual(b,m,no)),
     plane(b,m,Chess.get_black(b),Bool.not(U32.is_eq(Chess.get_turn(b),1)),sb,db,no,same,cb,dst,cap_good)))
def post(+b: Chess.Board,+m: Chess.Ply,rows: {Board.valid(b) == True{} : Bool},cert: emitted(b,m)) -> preserved(b,m):
  ((+sb,(owned,(db,no))),(same,(dst,cap_good)))=cert
  finish(b,m,sb,db,no,same,dst,cap_good,Source.colors(b,A.source(m),rows,sb,owned))
