# Candidate full fixed-white/fixed-black castling count preservation.
# The generic plane retains the necessary source-color equation.
import Base
import ./CastleClear.bend as Clear
import ./CastleStrip.bend as Strip
import ./CastleFrame.bend as CastleFrame
import ./Frame.bend as Frame
import ./Selector.bend as S
import ./Source.bend as Source
import ./Intersection.bend as I
import ./Count.bend as Count
import ./Actual.bend as K
import ../color_counts/U64Count.bend as U
import ../backrank_pawns/Mask.bend as M
import ../board/Spec.bend as B
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as Chess

def removed(+b: Chess.Board,+ks: Bool) -> U64:
  U64.or(U64.or(U64.bit(U32.to_nat(E.src(b))),U64.bit(U32.to_nat(E.dst(b,ks)))),
    U64.or(U64.bit(U32.to_nat(E.dst(b,ks))),U64.bit(U32.to_nat(E.rook(b,ks)))))

def reduced(+b: Chess.Board,+ks: Bool,+color: U64,+side: Bool,
  +rows: {B.valid(b) == True{} : Bool},+guard: {E.guard(b,ks) == True{} : Bool}) ->
  {Frame.model(b,E.move(b,ks),color,side) ==
   Chess.update_piece(U64.and(Chess.get_kings(b),color),U64.bit(U32.to_nat(E.src(b))),U64.bit(U32.to_nat(E.dst(b,ks))),
     Bool.and(S.put(b,E.move(b,ks)),side)) : U64}:
  Equal.trans(U64,Frame.model(b,E.move(b,ks),color,side),
    Chess.update_piece(U64.and(Chess.get_kings(b),color),removed(b,ks),U64.bit(U32.to_nat(E.dst(b,ks))),
      Bool.and(S.put(b,E.move(b,ks)),side)),
    Chess.update_piece(U64.and(Chess.get_kings(b),color),U64.bit(U32.to_nat(E.src(b))),U64.bit(U32.to_nat(E.dst(b,ks))),
      Bool.and(S.put(b,E.move(b,ks)),side)),
    Equal.cong(U64,U64,x => Chess.update_piece(U64.and(Chess.get_kings(b),color),x,U64.bit(U32.to_nat(E.dst(b,ks))),
      Bool.and(S.put(b,E.move(b,ks)),side)),M.remove(E.move(b,ks)),removed(b,ks),
      Clear.mask(U32.is_eq(Chess.get_turn(b),1),ks)),
    Equal.cong(U64,U64,x => U64.or(x,Chess.select_u64(Bool.and(S.put(b,E.move(b,ks)),side),U64.bit(U32.to_nat(E.dst(b,ks))),U64.zero())),
      U64.and_not(U64.and(Chess.get_kings(b),color),removed(b,ks)),
      U64.and_not(U64.and(Chess.get_kings(b),color),U64.bit(U32.to_nat(E.src(b)))),
      Strip.strip(Chess.get_kings(b),color,U64.bit(U32.to_nat(E.src(b))),U64.bit(U32.to_nat(E.dst(b,ks))),U64.bit(U32.to_nat(E.rook(b,ks))),
        Clear.destination(b,ks,rows,guard),Clear.rook_source(b,ks,rows,guard))))

def plane(+b: Chess.Board,+ks: Bool,+color: U64,+side: Bool,
  +rows: {B.valid(b) == True{} : Bool},+guard: {E.guard(b,ks) == True{} : Bool},
  source_color: {U64.test_bit(color,U32.to_nat(E.src(b))) == side : Bool}) ->
  {U.count(Frame.model(b,E.move(b,ks),color,side)) == U.count(U64.and(Chess.get_kings(b),color)) : Nat}:
  +sb=Clear.sb(U32.is_eq(Chess.get_turn(b),1))
  +db=Clear.db(U32.is_eq(Chess.get_turn(b),1),ks)
  Equal.trans(Nat,U.count(Frame.model(b,E.move(b,ks),color,side)),
    U.count(Chess.update_piece(U64.and(Chess.get_kings(b),color),U64.bit(U32.to_nat(E.src(b))),U64.bit(U32.to_nat(E.dst(b,ks))),
      Bool.and(S.put(b,E.move(b,ks)),side))),U.count(U64.and(Chess.get_kings(b),color)),
    Equal.cong(U64,Nat,U.count,Frame.model(b,E.move(b,ks),color,side),
      Chess.update_piece(U64.and(Chess.get_kings(b),color),U64.bit(U32.to_nat(E.src(b))),U64.bit(U32.to_nat(E.dst(b,ks))),
        Bool.and(S.put(b,E.move(b,ks)),side)),reduced(b,ks,color,side,rows,guard)),
    Count.update(U64.and(Chess.get_kings(b),color),U32.to_nat(E.src(b)),U32.to_nat(E.dst(b,ks)),
      Bool.and(S.put(b,E.move(b,ks)),side),sb,db,
      I.source(Chess.get_kings(b),color,U32.to_nat(E.src(b)),S.put(b,E.move(b,ks)),side,sb,
        Equal.sym(Bool,S.put(b,E.move(b,ks)),U64.test_bit(Chess.get_kings(b),U32.to_nat(E.src(b))),
          S.ordinary(b,E.src(b),E.dst(b,ks),2,rows,sb,Clear.source_owned(b,ks,guard))),source_color),
      I.fresh(Chess.get_kings(b),color,U32.to_nat(E.dst(b,ks)),db,
        Count.clear_test(Chess.get_kings(b),U32.to_nat(E.dst(b,ks)),Clear.destination(b,ks,rows,guard)))))

# Source.colors derives each fixed-color equation from rows and the same guard.
# Child-plane equalities come from the original candidate, not from premises.
def finish(+b: Chess.Board,+ks: Bool,
  +rows: {B.valid(b) == True{} : Bool},+guard: {E.guard(b,ks) == True{} : Bool},
  frames: CastleFrame.both(b,E.move(b,ks)),
  colors: {U64.test_bit(Chess.get_white(b),U32.to_nat(E.src(b))) == U32.is_eq(Chess.get_turn(b),1) : Bool} &
    {U64.test_bit(Chess.get_black(b),U32.to_nat(E.src(b))) == Bool.not(U32.is_eq(Chess.get_turn(b),1)) : Bool}) ->
  K.preserved(b,E.move(b,ks)):
  (fw,fb)=frames
  (cw,cb)=colors
  (Equal.trans(Nat,K.white(Chess.make_move(b,E.move(b,ks))),
     U.count(Frame.model(b,E.move(b,ks),Chess.get_white(b),U32.is_eq(Chess.get_turn(b),1))),K.white(b),
     Equal.cong(U64,Nat,U.count,U64.and(Chess.get_kings(Chess.make_move(b,E.move(b,ks))),Chess.get_white(Chess.make_move(b,E.move(b,ks)))),
       Frame.model(b,E.move(b,ks),Chess.get_white(b),U32.is_eq(Chess.get_turn(b),1)),fw),
     plane(b,ks,Chess.get_white(b),U32.is_eq(Chess.get_turn(b),1),rows,guard,cw)),
   Equal.trans(Nat,K.black(Chess.make_move(b,E.move(b,ks))),
     U.count(Frame.model(b,E.move(b,ks),Chess.get_black(b),Bool.not(U32.is_eq(Chess.get_turn(b),1)))),K.black(b),
     Equal.cong(U64,Nat,U.count,U64.and(Chess.get_kings(Chess.make_move(b,E.move(b,ks))),Chess.get_black(Chess.make_move(b,E.move(b,ks)))),
       Frame.model(b,E.move(b,ks),Chess.get_black(b),Bool.not(U32.is_eq(Chess.get_turn(b),1))),fb),
     plane(b,ks,Chess.get_black(b),Bool.not(U32.is_eq(Chess.get_turn(b),1)),rows,guard,cb)))

def preserved(+b: Chess.Board,+ks: Bool,
  +rows: {B.valid(b) == True{} : Bool},+guard: {E.guard(b,ks) == True{} : Bool}) ->
  K.preserved(b,E.move(b,ks)):
  finish(b,ks,rows,guard,CastleFrame.guarded(b,ks,rows,guard),
    Source.colors(b,E.src(b),rows,Clear.sb(U32.is_eq(Chess.get_turn(b),1)),Clear.source_owned(b,ks,guard)))
