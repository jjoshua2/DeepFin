# Generic board update: finite geometry is supplied by a source-file certificate.
import Base
import ./Ranks.bend as Ranks
import ./Files.bend as Files
import ./Geometry.bend as G
import ./Post.bend as Post
import ./Move.bend as Move
import ../move_update/Core.bend as Core
import ../move_update/Ordinary.bend as Ordinary
import ../occupancy_insertion/Lattice.bend as L
import ../initialized_pawn_double/Spec.bend as D
import ../bitboard_membership/AllEmittedSound.bend as All
import ../../../legal_probe/Chess.bend as C
import ../../Position.bend as Position

def board(+b: C.Board,+s: U32,+d: U32,+e: U32,+ep: Bool,+w: Bool) -> C.Board:
  Core.raw(b,Move.mask(s,d,ep),U64.bit(U32.to_nat(d)),0,w,Bool.to_u32(Bool.not(w)),Ordinary.rights(b,s,d,0),e)

def property(+b: C.Board,+s: U32,+ep: Bool,+w: Bool) -> Bool:
  Position.valid_ep(Move.midpoint(s,D.second(s,w)),Move.next(b,s,D.second(s,w),ep,w))

def guards(+b: C.Board,+s: U32,+d: U32,+e: U32,+ep: Bool,+w: Bool,empty: {U64.test_bit(C.occupied(b),U32.to_nat(e)) == False{} : Bool},cert: G.facts(s,d,e,w)) -> {Post.guard(e,board(b,s,d,e,ep,w),Bool.to_u32(Bool.not(w))) == True{} : Bool}:
  (rank,(origin,(victim,(first,(boundS,(boundD,(boundE,(selfS,(+selfD,(missS,missE))))))))))=cert
  All.bool_and_true(G.rank(e,w),
    Bool.and(Bool.not(U64.test_bit(C.occupied(board(b,s,d,e,ep,w)),U32.to_nat(e))),
      Bool.and(Bool.not(U64.test_bit(C.occupied(board(b,s,d,e,ep,w)),U32.to_nat(G.origin(e,w)))),Position.occupied_by(board(b,s,d,e,ep,w),U32.xor(e,8),0,U32.xor(Bool.to_u32(Bool.not(w)),1)))),rank,
    All.bool_and_true(Bool.not(U64.test_bit(C.occupied(board(b,s,d,e,ep,w)),U32.to_nat(e))),
      Bool.and(Bool.not(U64.test_bit(C.occupied(board(b,s,d,e,ep,w)),U32.to_nat(G.origin(e,w)))),Position.occupied_by(board(b,s,d,e,ep,w),U32.xor(e,8),0,U32.xor(Bool.to_u32(Bool.not(w)),1))),
      Post.not_false(U64.test_bit(C.occupied(board(b,s,d,e,ep,w)),U32.to_nat(e)),Post.absent(b,Move.mask(s,d,ep),U64.bit(U32.to_nat(d)),w,Bool.to_u32(Bool.not(w)),Ordinary.rights(b,s,d,0),e,U32.to_nat(e),boundE,empty,missE)),
      All.bool_and_true(Bool.not(U64.test_bit(C.occupied(board(b,s,d,e,ep,w)),U32.to_nat(G.origin(e,w)))),Position.occupied_by(board(b,s,d,e,ep,w),U32.xor(e,8),0,U32.xor(Bool.to_u32(Bool.not(w)),1)),
        Equal.trans(Bool,Bool.not(U64.test_bit(C.occupied(board(b,s,d,e,ep,w)),U32.to_nat(G.origin(e,w)))),Bool.not(U64.test_bit(C.occupied(board(b,s,d,e,ep,w)),U32.to_nat(s))),True{},
          Equal.cong(U32,Bool,x => Bool.not(U64.test_bit(C.occupied(board(b,s,d,e,ep,w)),U32.to_nat(x))),G.origin(e,w),s,origin),
          Post.not_false(U64.test_bit(C.occupied(board(b,s,d,e,ep,w)),U32.to_nat(s)),Post.cleared(b,Move.mask(s,d,ep),U64.bit(U32.to_nat(d)),w,Bool.to_u32(Bool.not(w)),Ordinary.rights(b,s,d,0),e,U32.to_nat(s),boundS,
            L.insert_preserves(U64.or(U64.bit(U32.to_nat(s)),U64.bit(U32.to_nat(Bool.pick(U32,ep,U32.xor(d,8),d)))),U64.or(U64.bit(U32.to_nat(d)),U64.zero()),U32.to_nat(s),
              L.insert_preserves(U64.bit(U32.to_nat(s)),U64.bit(U32.to_nat(Bool.pick(U32,ep,U32.xor(d,8),d))),U32.to_nat(s),selfS)),missS))),
        Equal.trans(Bool,Position.occupied_by(board(b,s,d,e,ep,w),U32.xor(e,8),0,U32.xor(Bool.to_u32(Bool.not(w)),1)),Position.occupied_by(board(b,s,d,e,ep,w),d,0,U32.xor(Bool.to_u32(Bool.not(w)),1)),True{},
          Equal.cong(U32,Bool,x => Position.occupied_by(board(b,s,d,e,ep,w),x,0,U32.xor(Bool.to_u32(Bool.not(w)),1)),U32.xor(e,8),d,victim),
          Post.occupied_by(board(b,s,d,e,ep,w),d,U32.xor(Bool.to_u32(Bool.not(w)),1),Post.color(b,Move.mask(s,d,ep),U64.bit(U32.to_nat(d)),w,Ordinary.rights(b,s,d,0),e,U32.to_nat(d),selfD),
            Post.piece(board(b,s,d,e,ep,w),d,Post.pawn(b,Move.mask(s,d,ep),U64.bit(U32.to_nat(d)),w,Bool.to_u32(Bool.not(w)),Ordinary.rights(b,s,d,0),e,U32.to_nat(d),selfD)))))))

def bound(+s: U32,+d: U32,+e: U32,+w: Bool,cert: G.facts(s,d,e,w)) -> {Nat.is_lt(U32.to_nat(e),64n) == True{} : Bool}:
  (rank,(origin,(victim,(first,(boundS,(boundD,(boundE,rest)))))))=cert
  boundE

def guarded(+b: C.Board,+s: U32,+d: U32,+e: U32,+ep: Bool,+w: Bool,empty: {U64.test_bit(C.occupied(b),U32.to_nat(e)) == False{} : Bool},below: {Nat.is_lt(U32.to_nat(e),64n) == True{} : Bool},cert: G.facts(s,d,e,w)) -> {Position.valid_ep(e,board(b,s,d,e,ep,w)) == True{} : Bool}:
  Equal.trans(Bool,Position.valid_ep(e,board(b,s,d,e,ep,w)),Post.guard(e,board(b,s,d,e,ep,w),C.get_turn(board(b,s,d,e,ep,w))),True{},
    Post.unfold_ep(e,board(b,s,d,e,ep,w),below),
    Equal.trans(Bool,Post.guard(e,board(b,s,d,e,ep,w),C.get_turn(board(b,s,d,e,ep,w))),Post.guard(e,board(b,s,d,e,ep,w),Bool.to_u32(Bool.not(w))),True{},
      Equal.cong(U32,Bool,x => Post.guard(e,board(b,s,d,e,ep,w),x),C.get_turn(board(b,s,d,e,ep,w)),Bool.to_u32(Bool.not(w)),Post.turn(b,Move.mask(s,d,ep),U64.bit(U32.to_nat(d)),w,Bool.to_u32(Bool.not(w)),Ordinary.rights(b,s,d,0),e)),guards(b,s,d,e,ep,w,empty,cert)))

def file(+b: C.Board,+ep: Bool,+f: Files.File,+w: Bool,empty: {U64.test_bit(C.occupied(b),U32.to_nat(D.first(Files.source(f,w),w))) == False{} : Bool}) -> {property(b,Files.source(f,w),ep,w) == True{} : Bool}:
  +s=Files.source(f,w)
  +d=D.second(s,w)
  +e=Move.midpoint(s,d)
  guarded(b,s,d,e,ep,w,Equal.trans(Bool,U64.test_bit(C.occupied(b),U32.to_nat(e)),U64.test_bit(C.occupied(b),U32.to_nat(D.first(s,w))),False{},
    Equal.cong(U32,Bool,x => U64.test_bit(C.occupied(b),U32.to_nat(x)),e,D.first(s,w),Equal.sym(U32,D.first(s,w),e,G.first(s,d,e,w,G.file(f,w)))),empty),bound(s,d,e,w,G.file(f,w)),G.file(f,w))

def reify(+b: C.Board,+s: U32,+ep: Bool,+w: Bool,empty: {U64.test_bit(C.occupied(b),U32.to_nat(D.first(s,w))) == False{} : Bool},cert: Exists(Files.File,f => {s == Files.source(f,w) : U32})) -> {property(b,s,ep,w) == True{} : Bool}:
  (+f,same)=cert
  +source_eq: {s == Files.source(f,w) : U32}=same
  %Equal.sym(U32,s,Files.source(f,w),source_eq) : {property(b,_,ep,w) == True{} : Bool}
  file(b,ep,f,w,Equal.trans(Bool,U64.test_bit(C.occupied(b),U32.to_nat(D.first(Files.source(f,w),w))),U64.test_bit(C.occupied(b),U32.to_nat(D.first(s,w))),False{},
    Equal.cong(U32,Bool,x => U64.test_bit(C.occupied(b),U32.to_nat(D.first(x,w))),Files.source(f,w),s,Equal.sym(U32,s,Files.source(f,w),source_eq)),empty))

def finite(+b: C.Board,+s: U32,+ep: Bool,+w: Bool,below: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},rank: {Ranks.start_rank(s,w) == True{} : Bool},empty: {U64.test_bit(C.occupied(b),U32.to_nat(D.first(s,w))) == False{} : Bool}) -> {property(b,s,ep,w) == True{} : Bool}:
  reify(b,s,ep,w,empty,Files.certify(s,w,below,rank))
