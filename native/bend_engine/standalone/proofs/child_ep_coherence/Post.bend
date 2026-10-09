# Board bit facts use the actual raw make_move update, with arbitrary old planes.
import Base
import ../layout/Domain.bend as Domain
import ../decoder/Bits.bend as B
import ../u64/Words.bend as Old
import ./Bits.bend as Bits
import ../move_update/Core.bend as Core
import ../ordinary_occupancy/Ordinary.bend as Occupancy
import ../ray/Algebra.bend as A
import ../bitboard_membership/AllEmittedSound.bend as All
import ../../../legal_probe/Chess.bend as C
import ../../Position.bend as Position

def absent(+b: C.Board,+mask: U64,+target: U64,+w: Bool,+turn: U32,+rights: U32,+ep: U32,+i: Nat,+bound: {Nat.is_lt(i,64n) == True{} : Bool},empty: {U64.test_bit(C.occupied(b),i) == False{} : Bool},miss: {U64.test_bit(target,i) == False{} : Bool}) -> {U64.test_bit(C.occupied(Core.raw(b,mask,target,0,w,turn,rights,ep)),i) == False{} : Bool}:
  Equal.trans(Bool,U64.test_bit(C.occupied(Core.raw(b,mask,target,0,w,turn,rights,ep)),i),U64.test_bit(C.update_piece(C.occupied(b),mask,target,True{}),i),False{},
    Equal.cong(U64,Bool,x => U64.test_bit(x,i),C.occupied(Core.raw(b,mask,target,0,w,turn,rights,ep)),C.update_piece(C.occupied(b),mask,target,True{}),Occupancy.raw_occupied(b,mask,target,0,w,turn,rights,ep)),Bits.update_absent(C.occupied(b),mask,target,True{},i,bound,empty,miss))

def cleared(+b: C.Board,+mask: U64,+target: U64,+w: Bool,+turn: U32,+rights: U32,+ep: U32,+i: Nat,+bound: {Nat.is_lt(i,64n) == True{} : Bool},hit: {U64.test_bit(mask,i) == True{} : Bool},miss: {U64.test_bit(target,i) == False{} : Bool}) -> {U64.test_bit(C.occupied(Core.raw(b,mask,target,0,w,turn,rights,ep)),i) == False{} : Bool}:
  Equal.trans(Bool,U64.test_bit(C.occupied(Core.raw(b,mask,target,0,w,turn,rights,ep)),i),U64.test_bit(C.update_piece(C.occupied(b),mask,target,True{}),i),False{},
    Equal.cong(U64,Bool,x => U64.test_bit(x,i),C.occupied(Core.raw(b,mask,target,0,w,turn,rights,ep)),C.update_piece(C.occupied(b),mask,target,True{}),Occupancy.raw_occupied(b,mask,target,0,w,turn,rights,ep)),Bits.update_removed(C.occupied(b),mask,target,True{},i,bound,hit,miss))

def pawn(+b: C.Board,+mask: U64,+target: U64,+w: Bool,+turn: U32,+rights: U32,+ep: U32,+i: Nat,hit: {U64.test_bit(target,i) == True{} : Bool}) -> {U64.test_bit(C.get_pawns(Core.raw(b,mask,target,0,w,turn,rights,ep)),i) == True{} : Bool}:
  match b:
    case C.Board{p,n,bi,r,q,k,wa,bl,ot,orr,oe}: Bits.update_present(p,mask,target,i,hit)

def piece(+b: C.Board,+sq: U32,yes: {U64.test_bit(C.get_pawns(b),U32.to_nat(sq)) == True{} : Bool}) -> {C.piece(b,sq) == 0 : U32}:
  %Equal.sym(Bool,U64.test_bit(C.get_pawns(b),U32.to_nat(sq)),True{},yes) :
    {Bool.pick(U32,_,0,Bool.pick(U32,U64.test_bit(C.get_knights(b),U32.to_nat(sq)),1,Bool.pick(U32,U64.test_bit(C.get_bishops(b),U32.to_nat(sq)),2,Bool.pick(U32,U64.test_bit(C.get_rooks(b),U32.to_nat(sq)),3,Bool.pick(U32,U64.test_bit(C.get_queens(b),U32.to_nat(sq)),4,5))))) == 0 : U32}
  {==}

def color(+board: C.Board,+mask: U64,+target: U64,+white: Bool,+rights: U32,+ep: U32,+i: Nat,hit: {U64.test_bit(target,i) == True{} : Bool}) ->
  {U64.test_bit(C.color(Core.raw(board,mask,target,0,white,Bool.to_u32(Bool.not(white)),rights,ep),U32.xor(Bool.to_u32(Bool.not(white)),1)),i) == True{} : Bool}:
  match board white:
    case C.Board{p,n,bi,r,q,k,+wa,bl,ot,orr,oe} True{}:
      %Equal.sym(U64,U64.or(target,U64.zero()),target,A.zero(target)) : {U64.test_bit(C.update_piece(wa,mask,_,True{}),i) == True{} : Bool}
      Bits.update_present(wa,mask,target,i,hit)
    case C.Board{p,n,bi,r,q,k,wa,+bl,ot,orr,oe} False{}:
      %Equal.sym(U64,U64.or(target,U64.zero()),target,A.zero(target)) : {U64.test_bit(C.update_piece(bl,mask,_,True{}),i) == True{} : Bool}
      Bits.update_present(bl,mask,target,i,hit)

def occupied_by(+b: C.Board,+dst: U32,+side: U32,color: {U64.test_bit(C.color(b,side),U32.to_nat(dst)) == True{} : Bool},piece: {C.piece(b,dst) == 0 : U32}) -> {Position.occupied_by(b,dst,0,side) == True{} : Bool}:
  All.bool_and_true(U64.test_bit(C.color(b,side),U32.to_nat(dst)),U32.is_eq(C.piece(b,dst),0),color,Equal.cong(U32,Bool,x => U32.is_eq(x,0),C.piece(b,dst),0,piece))

def not_false(a: Bool,eq: {a == False{} : Bool}) -> {Bool.not(a) == True{} : Bool}:
  %Equal.sym(Bool,a,False{},eq) : {Bool.not(_) == True{} : Bool}
  {==}

def turn(+b: C.Board,+mask: U64,+target: U64,+w: Bool,+t: U32,+rights: U32,+ep: U32) -> {C.get_turn(Core.raw(b,mask,target,0,w,t,rights,ep)) == t : U32}:
  match b:
    case C.Board{p,n,bi,r,q,k,wa,bl,ot,orr,oe}: {==}

def guard(+sq: U32,+b: C.Board,+t: U32) -> Bool:
  +white=U32.is_eq(t,1)
  +origin=Bool.pick(U32,white,U32.add(sq,8),U32.sub(sq,8))
  Bool.and(U32.is_eq(U32.div(sq,8),Bool.pick(U32,white,5,2)),
    Bool.and(Bool.not(U64.test_bit(C.occupied(b),U32.to_nat(sq))),
      Bool.and(Bool.not(U64.test_bit(C.occupied(b),U32.to_nat(origin))),Position.occupied_by(b,U32.xor(sq,8),0,U32.xor(t,1)))))

def unfolding(+sq: U32,+b: C.Board) -> Data: {Position.valid_ep(sq,b) == guard(sq,b,C.get_turn(b)) : Bool}

def unfold_cases(+sq: U32,+b: C.Board,n: Nat,same: {U32.to_nat(sq) == n : Nat},below: {Nat.is_lt(n,64n) == True{} : Bool}) -> unfolding(sq,b):
  match n:
    case 0n:
      %Equal.sym(U32,sq,0,Domain.injective(sq,0,same)) : unfolding(_,b)
      {==}
    case 1n:
      %Equal.sym(U32,sq,1,Domain.injective(sq,1,same)) : unfolding(_,b)
      {==}
    case 2n:
      %Equal.sym(U32,sq,2,Domain.injective(sq,2,same)) : unfolding(_,b)
      {==}
    case 3n:
      %Equal.sym(U32,sq,3,Domain.injective(sq,3,same)) : unfolding(_,b)
      {==}
    case 4n:
      %Equal.sym(U32,sq,4,Domain.injective(sq,4,same)) : unfolding(_,b)
      {==}
    case 5n:
      %Equal.sym(U32,sq,5,Domain.injective(sq,5,same)) : unfolding(_,b)
      {==}
    case 6n:
      %Equal.sym(U32,sq,6,Domain.injective(sq,6,same)) : unfolding(_,b)
      {==}
    case 7n:
      %Equal.sym(U32,sq,7,Domain.injective(sq,7,same)) : unfolding(_,b)
      {==}
    case 8n:
      %Equal.sym(U32,sq,8,Domain.injective(sq,8,same)) : unfolding(_,b)
      {==}
    case 9n:
      %Equal.sym(U32,sq,9,Domain.injective(sq,9,same)) : unfolding(_,b)
      {==}
    case 10n:
      %Equal.sym(U32,sq,10,Domain.injective(sq,10,same)) : unfolding(_,b)
      {==}
    case 11n:
      %Equal.sym(U32,sq,11,Domain.injective(sq,11,same)) : unfolding(_,b)
      {==}
    case 12n:
      %Equal.sym(U32,sq,12,Domain.injective(sq,12,same)) : unfolding(_,b)
      {==}
    case 13n:
      %Equal.sym(U32,sq,13,Domain.injective(sq,13,same)) : unfolding(_,b)
      {==}
    case 14n:
      %Equal.sym(U32,sq,14,Domain.injective(sq,14,same)) : unfolding(_,b)
      {==}
    case 15n:
      %Equal.sym(U32,sq,15,Domain.injective(sq,15,same)) : unfolding(_,b)
      {==}
    case 16n:
      %Equal.sym(U32,sq,16,Domain.injective(sq,16,same)) : unfolding(_,b)
      {==}
    case 17n:
      %Equal.sym(U32,sq,17,Domain.injective(sq,17,same)) : unfolding(_,b)
      {==}
    case 18n:
      %Equal.sym(U32,sq,18,Domain.injective(sq,18,same)) : unfolding(_,b)
      {==}
    case 19n:
      %Equal.sym(U32,sq,19,Domain.injective(sq,19,same)) : unfolding(_,b)
      {==}
    case 20n:
      %Equal.sym(U32,sq,20,Domain.injective(sq,20,same)) : unfolding(_,b)
      {==}
    case 21n:
      %Equal.sym(U32,sq,21,Domain.injective(sq,21,same)) : unfolding(_,b)
      {==}
    case 22n:
      %Equal.sym(U32,sq,22,Domain.injective(sq,22,same)) : unfolding(_,b)
      {==}
    case 23n:
      %Equal.sym(U32,sq,23,Domain.injective(sq,23,same)) : unfolding(_,b)
      {==}
    case 24n:
      %Equal.sym(U32,sq,24,Domain.injective(sq,24,same)) : unfolding(_,b)
      {==}
    case 25n:
      %Equal.sym(U32,sq,25,Domain.injective(sq,25,same)) : unfolding(_,b)
      {==}
    case 26n:
      %Equal.sym(U32,sq,26,Domain.injective(sq,26,same)) : unfolding(_,b)
      {==}
    case 27n:
      %Equal.sym(U32,sq,27,Domain.injective(sq,27,same)) : unfolding(_,b)
      {==}
    case 28n:
      %Equal.sym(U32,sq,28,Domain.injective(sq,28,same)) : unfolding(_,b)
      {==}
    case 29n:
      %Equal.sym(U32,sq,29,Domain.injective(sq,29,same)) : unfolding(_,b)
      {==}
    case 30n:
      %Equal.sym(U32,sq,30,Domain.injective(sq,30,same)) : unfolding(_,b)
      {==}
    case 31n:
      %Equal.sym(U32,sq,31,Domain.injective(sq,31,same)) : unfolding(_,b)
      {==}
    case 32n:
      %Equal.sym(U32,sq,32,Domain.injective(sq,32,same)) : unfolding(_,b)
      {==}
    case 33n:
      %Equal.sym(U32,sq,33,Domain.injective(sq,33,same)) : unfolding(_,b)
      {==}
    case 34n:
      %Equal.sym(U32,sq,34,Domain.injective(sq,34,same)) : unfolding(_,b)
      {==}
    case 35n:
      %Equal.sym(U32,sq,35,Domain.injective(sq,35,same)) : unfolding(_,b)
      {==}
    case 36n:
      %Equal.sym(U32,sq,36,Domain.injective(sq,36,same)) : unfolding(_,b)
      {==}
    case 37n:
      %Equal.sym(U32,sq,37,Domain.injective(sq,37,same)) : unfolding(_,b)
      {==}
    case 38n:
      %Equal.sym(U32,sq,38,Domain.injective(sq,38,same)) : unfolding(_,b)
      {==}
    case 39n:
      %Equal.sym(U32,sq,39,Domain.injective(sq,39,same)) : unfolding(_,b)
      {==}
    case 40n:
      %Equal.sym(U32,sq,40,Domain.injective(sq,40,same)) : unfolding(_,b)
      {==}
    case 41n:
      %Equal.sym(U32,sq,41,Domain.injective(sq,41,same)) : unfolding(_,b)
      {==}
    case 42n:
      %Equal.sym(U32,sq,42,Domain.injective(sq,42,same)) : unfolding(_,b)
      {==}
    case 43n:
      %Equal.sym(U32,sq,43,Domain.injective(sq,43,same)) : unfolding(_,b)
      {==}
    case 44n:
      %Equal.sym(U32,sq,44,Domain.injective(sq,44,same)) : unfolding(_,b)
      {==}
    case 45n:
      %Equal.sym(U32,sq,45,Domain.injective(sq,45,same)) : unfolding(_,b)
      {==}
    case 46n:
      %Equal.sym(U32,sq,46,Domain.injective(sq,46,same)) : unfolding(_,b)
      {==}
    case 47n:
      %Equal.sym(U32,sq,47,Domain.injective(sq,47,same)) : unfolding(_,b)
      {==}
    case 48n:
      %Equal.sym(U32,sq,48,Domain.injective(sq,48,same)) : unfolding(_,b)
      {==}
    case 49n:
      %Equal.sym(U32,sq,49,Domain.injective(sq,49,same)) : unfolding(_,b)
      {==}
    case 50n:
      %Equal.sym(U32,sq,50,Domain.injective(sq,50,same)) : unfolding(_,b)
      {==}
    case 51n:
      %Equal.sym(U32,sq,51,Domain.injective(sq,51,same)) : unfolding(_,b)
      {==}
    case 52n:
      %Equal.sym(U32,sq,52,Domain.injective(sq,52,same)) : unfolding(_,b)
      {==}
    case 53n:
      %Equal.sym(U32,sq,53,Domain.injective(sq,53,same)) : unfolding(_,b)
      {==}
    case 54n:
      %Equal.sym(U32,sq,54,Domain.injective(sq,54,same)) : unfolding(_,b)
      {==}
    case 55n:
      %Equal.sym(U32,sq,55,Domain.injective(sq,55,same)) : unfolding(_,b)
      {==}
    case 56n:
      %Equal.sym(U32,sq,56,Domain.injective(sq,56,same)) : unfolding(_,b)
      {==}
    case 57n:
      %Equal.sym(U32,sq,57,Domain.injective(sq,57,same)) : unfolding(_,b)
      {==}
    case 58n:
      %Equal.sym(U32,sq,58,Domain.injective(sq,58,same)) : unfolding(_,b)
      {==}
    case 59n:
      %Equal.sym(U32,sq,59,Domain.injective(sq,59,same)) : unfolding(_,b)
      {==}
    case 60n:
      %Equal.sym(U32,sq,60,Domain.injective(sq,60,same)) : unfolding(_,b)
      {==}
    case 61n:
      %Equal.sym(U32,sq,61,Domain.injective(sq,61,same)) : unfolding(_,b)
      {==}
    case 62n:
      %Equal.sym(U32,sq,62,Domain.injective(sq,62,same)) : unfolding(_,b)
      {==}
    case 63n:
      %Equal.sym(U32,sq,63,Domain.injective(sq,63,same)) : unfolding(_,b)
      {==}
    case 64n+p: B.impossible(unfolding(sq,b),p,below)

def unfold_ep(+sq: U32,+b: C.Board,below: {Nat.is_lt(U32.to_nat(sq),64n) == True{} : Bool}) -> unfolding(sq,b):
  unfold_cases(sq,b,U32.to_nat(sq),{==},below)
