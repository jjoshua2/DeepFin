import Base
import ../candidate_king_destination/Algebra.bend as A
import ../candidate_king_destination/Bits.bend as Bits
import ../decoder/Bits.bend as B
import ../ray_clear_disjoint/Algebra.bend as D
import ../pawn_ep_occurrence/Bridge.bend as Kind
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as Chess

def mask() -> U64: U64.from_parts(4278190080,255)
def backrank(d: U32) -> Bool:
  +rank=U32.div(d,8)
  Bool.or(U32.is_zero(rank),U32.is_eq(rank,7))
def safe(-b: Chess.Board) -> Type:
  {U64.is_zero(U64.and(Chess.get_pawns(b),mask())) == True{} : Bool}
def target(m: Chess.Ply) -> U64:
  Chess.Ply{s,d,p,f}=m
  U64.bit(U32.to_nat(d))
def put(+b: Chess.Board,m: Chess.Ply) -> U32:
  Chess.Ply{s,d,+p,f}=m
  Bool.pick(U32,U32.is_zero(p),Chess.piece(b,s),p)
def addition(+b: Chess.Board,+m: Chess.Ply) -> U64:
  Chess.select_u64(U32.is_eq(put(b,m),0),target(m),U64.zero())
def emitted(b: Chess.Board,m: Chess.Ply) -> Type:
  D.disjoint(addition(b,m),mask())
def remove(m: Chess.Ply) -> U64:
  Chess.Ply{+s,+d,p,+f}=m
  +castle=U32.is_eq(f,2)
  +to=U64.bit(U32.to_nat(d))
  +cap=Bool.pick(U32,U32.is_eq(f,1),U32.xor(d,8),d)
  +rook=Bool.pick(U32,U32.is_eq(U32.and(d,7),6),U32.inc(d),U32.sub(d,2))
  U64.or(U64.or(U64.bit(U32.to_nat(s)),U64.bit(U32.to_nat(cap))),
    U64.or(to,Chess.select_u64(castle,U64.bit(U32.to_nat(rook)),U64.zero())))
def actual(+b: Chess.Board,+m: Chess.Ply) ->
  {Chess.get_pawns(Chess.make_move(b,m)) == U64.or(U64.and_not(Chess.get_pawns(b),remove(m)),addition(b,m)) : U64}:
  match m:
    case Chess.Ply{s,d,p,f}: {==}
def post(+b: Chess.Board,+m: Chess.Ply,parent: safe(b),added: emitted(b,m)) -> safe(Chess.make_move(b,m)):
  +kept=U64.and_not(Chess.get_pawns(b),remove(m))
  +next=U64.or(kept,addition(b,m))
  %Equal.sym(U64,Chess.get_pawns(Chess.make_move(b,m)),next,actual(b,m)) :
    {U64.is_zero(U64.and(_,mask())) == True{} : Bool}
  Equal.cong(U64,Bool,U64.is_zero,U64.and(next,mask()),U64.zero(),
    A.union_clear(kept,addition(b,m),mask(),
      A.gate_clear(Chess.get_pawns(b),U64.not(remove(m)),mask(),A.zero_sound(U64.and(Chess.get_pawns(b),mask()),parent)),added))

def pawn(+b: Chess.Board,+s: U32) -> Bool:
  U64.test_bit(Chess.get_pawns(b),U32.to_nat(s))
def decoder(+b: Chess.Board,+s: U32) -> {U32.is_eq(Chess.piece(b,s),0) == pawn(b,s) : Bool}:
  Kind.kind_zero(pawn(b,s),U64.test_bit(Chess.get_knights(b),U32.to_nat(s)),
    U64.test_bit(Chess.get_bishops(b),U32.to_nat(s)),U64.test_bit(Chess.get_rooks(b),U32.to_nat(s)),
    U64.test_bit(Chess.get_queens(b),U32.to_nat(s)))
def nonpawn(+b: Chess.Board,+s: U32,+d: U32,+flag: U32,no: {pawn(b,s) == False{} : Bool}) -> emitted(b,Chess.Ply{s,d,0,flag}):
  %Equal.sym(Bool,U32.is_eq(Chess.piece(b,s),0),False{},Equal.trans(Bool,U32.is_eq(Chess.piece(b,s),0),pawn(b,s),False{},decoder(b,s),no)) :
    D.disjoint(Chess.select_u64(_,U64.bit(U32.to_nat(d)),U64.zero()),mask())
  D.left_zero(mask())
def source_clear(+b: Chess.Board,+s: U32,+bound: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},
  set: {U64.test_bit(mask(),U32.to_nat(s)) == True{} : Bool},parent: safe(b)) -> {pawn(b,s) == False{} : Bool}:
  Equal.trans(Bool,pawn(b,s),B.get64(Chess.get_pawns(b),U32.to_nat(s)),False{},B.test64(Chess.get_pawns(b),U32.to_nat(s),bound),
    Bits.get_clear(mask(),Chess.get_pawns(b),U32.to_nat(s),A.flip(Chess.get_pawns(b),mask(),A.zero_sound(U64.and(Chess.get_pawns(b),mask()),parent)),
      Equal.trans(Bool,B.get64(mask(),U32.to_nat(s)),U64.test_bit(mask(),U32.to_nat(s)),True{},
        Equal.sym(Bool,U64.test_bit(mask(),U32.to_nat(s)),B.get64(mask(),U32.to_nat(s)),B.test64(mask(),U32.to_nat(s),bound)),set)))
def home_bound(white: Bool) -> {Nat.is_lt(U32.to_nat(U32.add(Bool.pick(U32,white,0,56),4)),64n) == True{} : Bool}:
  match white:
    case True{}: {==}
    case False{}: {==}
def home_set(white: Bool) -> {U64.test_bit(mask(),U32.to_nat(U32.add(Bool.pick(U32,white,0,56),4))) == True{} : Bool}:
  match white:
    case True{}: {==}
    case False{}: {==}
def castle(+b: Chess.Board,+ks: Bool,parent: safe(b)) -> emitted(b,E.move(b,ks)):
  nonpawn(b,E.src(b),E.dst(b,ks),2,source_clear(b,E.src(b),home_bound(U32.is_eq(Chess.get_turn(b),1)),home_set(U32.is_eq(Chess.get_turn(b),1)),parent))
