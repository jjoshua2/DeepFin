# Direct runtime witnesses; these do not assert initialized legal-list membership.
import Base
import ../destination_factorization/Spec.bend as Count
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as C
import ../../Position.bend as Position

def child(+b: C.Board,+m: C.Ply) -> Bool: Position.valid_ep(C.get_ep(C.make_move(b,m)),C.make_move(b,m))

def white(+ep: U32) -> C.Board:
  C.Board{U64.bit(8n),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.or(U64.bit(0n),U64.bit(60n)),U64.or(U64.bit(8n),U64.bit(0n)),U64.bit(60n),1,0,ep}
def black(+turn: U32) -> C.Board:
  C.Board{U64.bit(48n),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.or(U64.bit(0n),U64.bit(60n)),U64.bit(0n),U64.or(U64.bit(48n),U64.bit(60n)),turn,0,64}
def white_double() -> {child(white(64),C.Ply{8,24,0,0}) == True{} : Bool}: {==}
def black_double() -> {child(black(0),C.Ply{48,32,0,0}) == True{} : Bool}: {==}
def raw_ep_double() -> {child(white(24),C.Ply{8,24,0,1}) == True{} : Bool}: {==}
def raw_ep_parent_bad() -> {Position.valid_ep(24,white(24)) == False{} : Bool}: {==}
def wrong_promotion_child() -> {child(white(64),C.Ply{8,24,1,0}) == False{} : Bool}: {==}
def wrong_castle_child() -> {child(white(64),C.Ply{8,24,0,2}) == False{} : Bool}: {==}
def noncanonical_child() -> {child(black(2),C.Ply{48,32,0,0}) == False{} : Bool}: {==}

def repeated() -> List<&2,C.Ply>: C.Ply{8,16,0,1} <> C.Ply{8,16,0,1} <> Nil{}
def tagged() -> Array<U64> & List<&2,C.Ply>: C.scan_after(8,True{},16,repeated(),(ALeaf{U64.bit(47n)},U64.bit(16n)))
def promoted() -> Array<U64> & List<&2,C.Ply>: C.scan_after(48,True{},56,Nil{},(ALeaf{U64.bit(47n)},U64.or(U64.bit(56n),U64.bit(57n))))
def duplicate_count() -> {Count.count(E.moves(tagged()),C.Ply{8,16,0,1}) == 3n : Nat}: {==}
def wrong_ep_flag_absent() -> {Count.count(E.moves(tagged()),C.Ply{8,16,0,0}) == 0n : Nat}: {==}
def wrong_promotion_absent() -> {Count.count(E.moves(promoted()),C.Ply{48,56,0,0}) == 0n : Nat}: {==}
def promotion_raw_ep_flag_absent() -> {Count.count(E.moves(promoted()),C.Ply{48,56,1,1}) == 0n : Nat}: {==}
def jump() -> C.Board: C.Board{U64.bit(32n),U64.bit(48n),U64.zero(),U64.zero(),U64.zero(),U64.or(U64.bit(0n),U64.bit(60n)),U64.or(U64.bit(32n),U64.bit(0n)),U64.or(U64.bit(48n),U64.bit(60n)),1,0,64}
def arbitrary_jump_incoherent() -> {child(jump(),C.Ply{32,48,0,0}) == False{} : Bool}: {==}

def stale() -> C.Board: C.Board{U64.bit(8n),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.or(U64.bit(0n),U64.bit(24n)),U64.or(U64.bit(8n),U64.bit(0n)),U64.bit(24n),1,0,16}
def stale_parent_invalid() -> {Position.valid_ep(16,stale()) == False{} : Bool}: {==}
def stale_king_loss() -> {C.get_kings(C.make_move(stale(),C.Ply{8,16,0,1})) == U64.bit(0n) : U64}: {==}
