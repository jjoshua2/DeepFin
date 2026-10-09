import Base
import ./U64Count.bend as U
import ../backrank_pawns/Mask.bend as M
import ../ray/Algebra.bend as A
import ../occupancy_insertion/Lattice.bend as L
import ../decoder/Bits.bend as B
import ../../../legal_probe/Chess.bend as Chess

def source(m: Chess.Ply) -> U32:
  Chess.Ply{s,d,p,f}=m
  s
def destination(m: Chess.Ply) -> U32:
  Chess.Ply{s,d,p,f}=m
  d
def flag(m: Chess.Ply) -> U32:
  Chess.Ply{s,d,p,f}=m
  f
def white(b: Chess.Board) -> Nat: U.count(Chess.get_white(b))
def black(b: Chess.Board) -> Nat: U.count(Chess.get_black(b))
def decreases(-b: Chess.Board,-m: Chess.Ply) -> Type:
  {Nat.is_le(white(Chess.make_move(b,m)),white(b)) == True{} : Bool} &
  {Nat.is_le(black(Chess.make_move(b,m)),black(b)) == True{} : Bool}
def emitted(-b: Chess.Board,-m: Chess.Ply) -> Type:
  {Nat.is_lt(U32.to_nat(source(m)),64n) == True{} : Bool} &
  ({U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(source(m))) == True{} : Bool} &
    ({Nat.is_lt(U32.to_nat(destination(m)),64n) == True{} : Bool} &
      {U32.is_eq(flag(m),2) == False{} : Bool}))

def bit_set(+i: Nat,+bound: {Nat.is_lt(i,64n) == True{} : Bool}) -> {U64.test_bit(U64.bit(i),i) == True{} : Bool}:
  Equal.trans(Bool,U64.test_bit(U64.bit(i),i),B.get(64n,U64.to_word(U64.bit(i)),i),True{},
    U.test_word(U64.bit(i),i,bound),
    Equal.trans(Bool,B.get(64n,U64.to_word(U64.bit(i)),i),B.get(64n,B.bit(64n,i),i),True{},
      Equal.cong(Word(64n),Bool,x => B.get(64n,x,i),U64.to_word(U64.bit(i)),B.bit(64n,i),U.bit_word(i,bound)),U.bit_self(64n,i,bound)))

def removed(+m: Chess.Ply,bound: {Nat.is_lt(U32.to_nat(source(m)),64n) == True{} : Bool}) ->
  {U64.test_bit(M.remove(m),U32.to_nat(source(m))) == True{} : Bool}:
  Chess.Ply{+s,+d,p,+f}=m
  +cap=Bool.pick(U32,U32.is_eq(f,1),U32.xor(d,8),d)
  +rook=Bool.pick(U32,U32.is_eq(U32.and(d,7),6),U32.inc(d),U32.sub(d,2))
  +head=U64.or(U64.bit(U32.to_nat(s)),U64.bit(U32.to_nat(cap)))
  +tail=U64.or(U64.bit(U32.to_nat(d)),Chess.select_u64(U32.is_eq(f,2),U64.bit(U32.to_nat(rook)),U64.zero()))
  L.insert_preserves(head,tail,U32.to_nat(s),
    L.insert_preserves(U64.bit(U32.to_nat(s)),U64.bit(U32.to_nat(cap)),U32.to_nat(s),bit_set(U32.to_nat(s),bound)))

def added(m: Chess.Ply) -> U64:
  Chess.Ply{+s,+d,p,f}=m
  U64.or(U64.bit(U32.to_nat(d)),Chess.select_u64(U32.is_eq(f,2),U64.bit(U32.to_nat(U32.div(U32.add(s,d),2))),U64.zero()))
def white_actual(+b: Chess.Board,+m: Chess.Ply) ->
  {Chess.get_white(Chess.make_move(b,m)) == Chess.update_piece(Chess.get_white(b),M.remove(m),added(m),U32.is_eq(Chess.get_turn(b),1)) : U64}:
  match m:
    case Chess.Ply{s,d,p,f}: {==}
def black_actual(+b: Chess.Board,+m: Chess.Ply) ->
  {Chess.get_black(Chess.make_move(b,m)) == Chess.update_piece(Chess.get_black(b),M.remove(m),added(m),Bool.not(U32.is_eq(Chess.get_turn(b),1))) : U64}:
  match m:
    case Chess.Ply{s,d,p,f}: {==}
def noncastle_added(+m: Chess.Ply,no: {U32.is_eq(flag(m),2) == False{} : Bool}) ->
  {added(m) == U64.bit(U32.to_nat(destination(m))) : U64}:
  Chess.Ply{+s,+d,p,+f}=m
  %Equal.sym(Bool,U32.is_eq(f,2),False{},no) :
    {U64.or(U64.bit(U32.to_nat(d)),Chess.select_u64(_,U64.bit(U32.to_nat(U32.div(U32.add(s,d),2))),U64.zero())) == U64.bit(U32.to_nat(d)) : U64}
  A.zero(U64.bit(U32.to_nat(d)))

def inactive(+a: U64,+remove: U64,+dst: Nat) ->
  {Nat.is_le(U.count(Chess.update_piece(a,remove,U64.bit(dst),False{})),U.count(a)) == True{} : Bool}:
  %Equal.sym(U64,U64.or(U64.and_not(a,remove),U64.zero()),U64.and_not(a,remove),A.zero(U64.and_not(a,remove))) :
    {Nat.is_le(U.count(_),U.count(a)) == True{} : Bool}
  U.delete_le(a,remove)
def planes(+w: U64,+bl: U64,+remove: U64,+src: Nat,+dst: Nat,turn: Bool,
  +source_bound: {Nat.is_lt(src,64n) == True{} : Bool},+target_bound: {Nat.is_lt(dst,64n) == True{} : Bool},
  owned: {U64.test_bit(Chess.select_u64(turn,w,bl),src) == True{} : Bool},+erased: {U64.test_bit(remove,src) == True{} : Bool}) ->
  {Nat.is_le(U.count(Chess.update_piece(w,remove,U64.bit(dst),turn)),U.count(w)) == True{} : Bool} &
  {Nat.is_le(U.count(Chess.update_piece(bl,remove,U64.bit(dst),Bool.not(turn))),U.count(bl)) == True{} : Bool}:
  match turn:
    case True{}:
      (U.replace_le(w,remove,src,dst,source_bound,target_bound,owned,erased),inactive(bl,remove,dst))
    case False{}:
      (inactive(w,remove,dst),U.replace_le(bl,remove,src,dst,source_bound,target_bound,owned,erased))
def lift_white(+b: Chess.Board,+m: Chess.Ply,no: {U32.is_eq(flag(m),2) == False{} : Bool},
  good: {Nat.is_le(U.count(Chess.update_piece(Chess.get_white(b),M.remove(m),U64.bit(U32.to_nat(destination(m))),U32.is_eq(Chess.get_turn(b),1))),white(b)) == True{} : Bool}) ->
  {Nat.is_le(white(Chess.make_move(b,m)),white(b)) == True{} : Bool}:
  %Equal.sym(U64,Chess.get_white(Chess.make_move(b,m)),Chess.update_piece(Chess.get_white(b),M.remove(m),added(m),U32.is_eq(Chess.get_turn(b),1)),white_actual(b,m)) :
    {Nat.is_le(U.count(_),white(b)) == True{} : Bool}
  %Equal.sym(U64,added(m),U64.bit(U32.to_nat(destination(m))),noncastle_added(m,no)) :
    {Nat.is_le(U.count(Chess.update_piece(Chess.get_white(b),M.remove(m),_,U32.is_eq(Chess.get_turn(b),1))),white(b)) == True{} : Bool}
  good
def lift_black(+b: Chess.Board,+m: Chess.Ply,no: {U32.is_eq(flag(m),2) == False{} : Bool},
  good: {Nat.is_le(U.count(Chess.update_piece(Chess.get_black(b),M.remove(m),U64.bit(U32.to_nat(destination(m))),Bool.not(U32.is_eq(Chess.get_turn(b),1)))),black(b)) == True{} : Bool}) ->
  {Nat.is_le(black(Chess.make_move(b,m)),black(b)) == True{} : Bool}:
  %Equal.sym(U64,Chess.get_black(Chess.make_move(b,m)),Chess.update_piece(Chess.get_black(b),M.remove(m),added(m),Bool.not(U32.is_eq(Chess.get_turn(b),1))),black_actual(b,m)) :
    {Nat.is_le(U.count(_),black(b)) == True{} : Bool}
  %Equal.sym(U64,added(m),U64.bit(U32.to_nat(destination(m))),noncastle_added(m,no)) :
    {Nat.is_le(U.count(Chess.update_piece(Chess.get_black(b),M.remove(m),_,Bool.not(U32.is_eq(Chess.get_turn(b),1)))),black(b)) == True{} : Bool}
  good
def finish(+b: Chess.Board,+m: Chess.Ply,+no: {U32.is_eq(flag(m),2) == False{} : Bool},
  results: {Nat.is_le(U.count(Chess.update_piece(Chess.get_white(b),M.remove(m),U64.bit(U32.to_nat(destination(m))),U32.is_eq(Chess.get_turn(b),1))),white(b)) == True{} : Bool} &
    {Nat.is_le(U.count(Chess.update_piece(Chess.get_black(b),M.remove(m),U64.bit(U32.to_nat(destination(m))),Bool.not(U32.is_eq(Chess.get_turn(b),1)))),black(b)) == True{} : Bool}) -> decreases(b,m):
  (pw,pb)=results
  (lift_white(b,m,no,pw),lift_black(b,m,no,pb))
def post(+b: Chess.Board,+m: Chess.Ply,cert: emitted(b,m)) -> decreases(b,m):
  (+sb,(owned,(+db,no)))=cert
  finish(b,m,no,planes(Chess.get_white(b),Chess.get_black(b),M.remove(m),U32.to_nat(source(m)),U32.to_nat(destination(m)),U32.is_eq(Chess.get_turn(b),1),sb,db,owned,removed(m,sb)))
