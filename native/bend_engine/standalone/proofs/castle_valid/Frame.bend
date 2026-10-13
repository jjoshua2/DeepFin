import Base
import ../decoder/Bits.bend as B
import ../decoder/Spec.bend as Row
import ../board/Spec.bend as Board
import ../occupancy_insertion/Lattice.bend as L
import ../initialized_pawn_double/Algebra.bend as A
import ../backrank_pawns/Mask.bend as M
import ../../../legal_probe/Chess.bend as Chess

def and_true(a: Bool) -> {Bool.and(a,True{}) == a : Bool}:
  match a:
    case False{}: {==}
    case True{}: {==}
def or_false(a: Bool) -> {Bool.or(a,False{}) == a : Bool}:
  match a:
    case False{}: {==}
    case True{}: {==}
def word_keep(+n: Nat,+a: Word(n),+mask: Word(n),+i: Nat,
  miss: {B.get(n,mask,i) == False{} : Bool}) ->
  {B.get(n,Word.and(n,a,Word.not(n,mask)),i) == B.get(n,a,i) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a mask i:
        case WCon{ah,at} WCon{mh,mt} 0n:
          Equal.trans(Bool,Bool.and(ah,Bool.not(mh)),Bool.and(ah,True{}),ah,
            Equal.cong(Bool,Bool,x => Bool.and(ah,Bool.not(x)),mh,False{},miss),and_true(ah))
        case WCon{ah,at} WCon{mh,mt} 1n+j: word_keep(p,at,mt,j,miss)
def choose_keep(+al: Word(32n),+ah: Word(32n),+ml: Word(32n),+mh: Word(32n),+i: Nat,low: Bool,
  miss: {Bool.pick(Bool,low,B.get(32n,ml,i),B.get(32n,mh,Nat.sub(i,32n))) == False{} : Bool}) ->
  {Bool.pick(Bool,low,B.get(32n,Word.and(32n,al,Word.not(32n,ml)),i),B.get(32n,Word.and(32n,ah,Word.not(32n,mh)),Nat.sub(i,32n))) ==
    Bool.pick(Bool,low,B.get(32n,al,i),B.get(32n,ah,Nat.sub(i,32n))) : Bool}:
  match low:
    case True{}: word_keep(32n,al,ml,i,miss)
    case False{}: word_keep(32n,ah,mh,Nat.sub(i,32n),miss)
def get_keep(+a: U64,+mask: U64,+i: Nat,miss: {B.get64(mask,i) == False{} : Bool}) ->
  {B.get64(U64.and_not(a,mask),i) == B.get64(a,i) : Bool}:
  match a mask:
    case U64{U32{al},U32{ah}} U64{U32{ml},U32{mh}}: choose_keep(al,ah,ml,mh,i,Nat.is_lt(i,32n),miss)
def keep(+a: U64,+mask: U64,+i: Nat,+bound: {Nat.is_lt(i,64n) == True{} : Bool},
  miss: {U64.test_bit(mask,i) == False{} : Bool}) ->
  {U64.test_bit(U64.and_not(a,mask),i) == U64.test_bit(a,i) : Bool}:
  Equal.trans(Bool,U64.test_bit(U64.and_not(a,mask),i),B.get64(U64.and_not(a,mask),i),U64.test_bit(a,i),B.test64(U64.and_not(a,mask),i,bound),
    Equal.trans(Bool,B.get64(U64.and_not(a,mask),i),B.get64(a,i),U64.test_bit(a,i),
      get_keep(a,mask,i,Equal.trans(Bool,B.get64(mask,i),U64.test_bit(mask,i),False{},Equal.sym(Bool,U64.test_bit(mask,i),B.get64(mask,i),B.test64(mask,i,bound)),miss)),
      Equal.sym(Bool,U64.test_bit(a,i),B.get64(a,i),B.test64(a,i,bound))))
def update(+a: U64,+mask: U64,+target: U64,+on: Bool,+i: Nat,+bound: {Nat.is_lt(i,64n) == True{} : Bool},
  miss: {U64.test_bit(mask,i) == False{} : Bool},fresh: {U64.test_bit(target,i) == False{} : Bool}) ->
  {U64.test_bit(Chess.update_piece(a,mask,target,on),i) == U64.test_bit(a,i) : Bool}:
  Equal.trans(Bool,U64.test_bit(Chess.update_piece(a,mask,target,on),i),
    Bool.or(U64.test_bit(U64.and_not(a,mask),i),U64.test_bit(Chess.select_u64(on,target,U64.zero()),i)),U64.test_bit(a,i),
    L.test_or(U64.and_not(a,mask),Chess.select_u64(on,target,U64.zero()),i),
    Equal.trans(Bool,Bool.or(U64.test_bit(U64.and_not(a,mask),i),U64.test_bit(Chess.select_u64(on,target,U64.zero()),i)),
      Bool.or(U64.test_bit(a,i),U64.test_bit(Chess.select_u64(on,target,U64.zero()),i)),U64.test_bit(a,i),
      Equal.cong(Bool,Bool,x => Bool.or(x,U64.test_bit(Chess.select_u64(on,target,U64.zero()),i)),
        U64.test_bit(U64.and_not(a,mask),i),U64.test_bit(a,i),keep(a,mask,i,bound,miss)),
      Equal.trans(Bool,Bool.or(U64.test_bit(a,i),U64.test_bit(Chess.select_u64(on,target,U64.zero()),i)),
        Bool.or(U64.test_bit(a,i),False{}),U64.test_bit(a,i),
        Equal.cong(Bool,Bool,x => Bool.or(U64.test_bit(a,i),x),U64.test_bit(Chess.select_u64(on,target,U64.zero()),i),False{},A.selected_absent(on,target,i,fresh)),
        or_false(U64.test_bit(a,i)))))
def rook_to(+m: Chess.Ply) -> U64:
  Chess.Ply{+s,+d,p,f}=m
  Chess.select_u64(U32.is_eq(f,2),U64.bit(U32.to_nat(U32.div(U32.add(s,d),2))),U64.zero())
def added(+m: Chess.Ply) -> U64: U64.or(M.target(m),rook_to(m))
def pawns_actual(+b: Chess.Board,m: Chess.Ply) ->
  {Chess.get_pawns(Chess.make_move(b,m)) == Chess.update_piece(Chess.get_pawns(b),M.remove(m),M.target(m),U32.is_eq(M.put(b,m),0)) : U64}:
  Chess.Ply{s,d,p,f}=m
  {==}
def knights_actual(+b: Chess.Board,m: Chess.Ply) ->
  {Chess.get_knights(Chess.make_move(b,m)) == Chess.update_piece(Chess.get_knights(b),M.remove(m),M.target(m),U32.is_eq(M.put(b,m),1)) : U64}:
  Chess.Ply{s,d,p,f}=m
  {==}
def bishops_actual(+b: Chess.Board,m: Chess.Ply) ->
  {Chess.get_bishops(Chess.make_move(b,m)) == Chess.update_piece(Chess.get_bishops(b),M.remove(m),M.target(m),U32.is_eq(M.put(b,m),2)) : U64}:
  Chess.Ply{s,d,p,f}=m
  {==}
def rooks_actual(+b: Chess.Board,m: Chess.Ply) ->
  {Chess.get_rooks(Chess.make_move(b,m)) == U64.or(Chess.update_piece(Chess.get_rooks(b),M.remove(m),M.target(m),U32.is_eq(M.put(b,m),3)),rook_to(m)) : U64}:
  Chess.Ply{s,d,p,f}=m
  {==}
def queens_actual(+b: Chess.Board,m: Chess.Ply) ->
  {Chess.get_queens(Chess.make_move(b,m)) == Chess.update_piece(Chess.get_queens(b),M.remove(m),M.target(m),U32.is_eq(M.put(b,m),4)) : U64}:
  Chess.Ply{s,d,p,f}=m
  {==}
def kings_actual(+b: Chess.Board,m: Chess.Ply) ->
  {Chess.get_kings(Chess.make_move(b,m)) == Chess.update_piece(Chess.get_kings(b),M.remove(m),M.target(m),U32.is_eq(M.put(b,m),5)) : U64}:
  Chess.Ply{s,d,p,f}=m
  {==}
def white_actual(+b: Chess.Board,m: Chess.Ply) ->
  {Chess.get_white(Chess.make_move(b,m)) == Chess.update_piece(Chess.get_white(b),M.remove(m),added(m),U32.is_eq(Chess.get_turn(b),1)) : U64}:
  Chess.Ply{s,d,p,f}=m
  {==}
def black_actual(+b: Chess.Board,m: Chess.Ply) ->
  {Chess.get_black(Chess.make_move(b,m)) == Chess.update_piece(Chess.get_black(b),M.remove(m),added(m),Bool.not(U32.is_eq(Chess.get_turn(b),1))) : U64}:
  Chess.Ply{s,d,p,f}=m
  {==}
