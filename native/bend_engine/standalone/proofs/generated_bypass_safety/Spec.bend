import Base
import ../storage/Observe.bend as O
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../candidate_bypass_classification/Generated.bend as G
import ../prepared_king_selection/Selection.bend as K
import ../prepared_ray_coverage/Query.bend as Q
import ../fast_full_equivalence/Equivalence.bend as F
import ../../../legal_probe/Chess.bend as Chess
import ../../Tables.bend as Tables

def scan(+c: O.Cells,+b: Chess.Board) -> List<&2,Chess.Ply>:
  E.moves(C.candidates(O.pack(c),b))

def candidates(+c: O.Cells,+b: Chess.Board) -> List<&2,Chess.Ply>:
  E.moves(G.produced(O.pack(c),b))

def old(+c: O.Cells,+b: Chess.Board) -> Data:
  Sigma<&2,&2,
    {Chess.attack(3,O.pack(c),U32.from_nat(K.index(b)),Chess.get_turn(b),Chess.occupied(b)) ==
      (O.pack(c),Tables.slider(U32.from_nat(K.index(b)),False{},Chess.occupied(b),False{})) : Array<U64> & U64},
    ignored =>
      {Chess.attack(2,O.pack(c),U32.from_nat(K.index(b)),Chess.get_turn(b),Chess.occupied(b)) ==
        (O.pack(c),Tables.slider(U32.from_nat(K.index(b)),True{},Chess.occupied(b),False{})) : Array<U64> & U64}>

def post(+c: O.Cells,+b: Chess.Board,+m: Chess.Ply) -> Type:
  Q.rook(c,K.index(b),U32.xor(Chess.get_turn(b),1),Chess.occupied(Chess.make_move(b,m))) &
  Q.bishop(c,K.index(b),U32.xor(Chess.get_turn(b),1),Chess.occupied(Chess.make_move(b,m)))

# Lookup contracts, not safety assumptions: exact table, original king index,
# opponent side and actual moved-board occupancy, only for actual scan bypasses.
def lookup(-c: O.Cells,-b: Chess.Board,-m: Chess.Ply) -> Type:
  E.member(m,scan(c,b)) ->
    {Chess.filter_requires(F.prepared_sensitive(c,b),m) == False{} : Bool} -> post(c,b,m)

def lookups(-c: O.Cells,-b: Chess.Board,+xs: List<&2,Chess.Ply>) -> Type:
  C.every(m => lookup(c,b,m),xs)

def initial(-c: O.Cells,-b: Chess.Board) -> Data:
  {Chess.in_check(O.pack(c),b,Chess.get_turn(b)) == (O.pack(c),False{}) : Array<U64> & Bool}
