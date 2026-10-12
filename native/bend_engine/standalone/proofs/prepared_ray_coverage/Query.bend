import Base
import ./Coverage.bend as C
import ../storage/Observe.bend as O
import ../accepted_castle/Spec.bend as S
import ../occupancy_insertion/Lattice.bend as L
import ../../../legal_probe/Chess.bend as Chess
import ../../Tables.bend as Tables

# Whole returned-pair contracts, at one supplied table, square, side and occupancy.
def rook(+c: O.Cells,+q: Nat,side: U32,+occ: U64) -> Type:
  {Chess.attack(3,O.pack(c),U32.from_nat(q),side,occ) ==
    (O.pack(c),Tables.slider(U32.from_nat(q),False{},occ,False{})) : Array<U64> & U64}
def bishop(+c: O.Cells,+q: Nat,side: U32,+occ: U64) -> Type:
  {Chess.attack(2,O.pack(c),U32.from_nat(q),side,occ) ==
    (O.pack(c),Tables.slider(U32.from_nat(q),True{},occ,False{})) : Array<U64> & U64}

# Chess.attack(4) feeds the returned rook table to bishop; exact table preservation
# is what permits these two separate contracts to compose.
def queen(+c: O.Cells,+q: Nat,+side: U32,+occ: U64,
  r: rook(c,q,side,occ),b: bishop(c,q,side,occ)) ->
  {Chess.attack(4,O.pack(c),U32.from_nat(q),side,occ) ==
    (O.pack(c),C.actual(q,occ)) : Array<U64> & U64}:
  %Equal.sym(Array<U64> & U64,Chess.attack(3,O.pack(c),U32.from_nat(q),side,occ),
    (O.pack(c),Tables.slider(U32.from_nat(q),False{},occ,False{})),r) :
    {Chess.queen_after(U32.from_nat(q),occ,_) == (O.pack(c),C.actual(q,occ)) : Array<U64> & U64}
  %Equal.sym(Array<U64> & U64,Chess.attack(2,O.pack(c),U32.from_nat(q),side,occ),
    (O.pack(c),Tables.slider(U32.from_nat(q),True{},occ,False{})),b) :
    {Chess.or_result(Tables.slider(U32.from_nat(q),False{},occ,False{}),_) ==
      (O.pack(c),C.actual(q,occ)) : Array<U64> & U64}
  {==}

def king(+b: Chess.Board) -> U32:
  U64.ctz(U64.and(Chess.get_kings(b),Chess.color(b,Chess.get_turn(b))))

def prepared(+c: O.Cells,+b: Chess.Board,+q: Nat,
  chosen: {king(b) == U32.from_nat(q) : U32},
  r: rook(c,q,Chess.get_turn(b),Chess.occupied(b)),
  d: bishop(c,q,Chess.get_turn(b),Chess.occupied(b))) ->
  {S.rays(c,b) == C.actual(q,Chess.occupied(b)) : U64}:
  Equal.trans(U64,S.rays(c,b),
    S.mask(Chess.attack(4,O.pack(c),U32.from_nat(q),Chess.get_turn(b),Chess.occupied(b))),
    C.actual(q,Chess.occupied(b)),
    Equal.cong(U32,U64,sq => S.mask(Chess.attack(4,O.pack(c),sq,Chess.get_turn(b),Chess.occupied(b))),
      king(b),U32.from_nat(q),chosen),
    Equal.cong(Array<U64> & U64,U64,S.mask,
      Chess.attack(4,O.pack(c),U32.from_nat(q),Chess.get_turn(b),Chess.occupied(b)),
      (O.pack(c),C.actual(q,Chess.occupied(b))),queen(c,q,Chess.get_turn(b),Chess.occupied(b),r,d)))

def coverage(+c: O.Cells,+b: Chess.Board,+q: Nat,+dir: Nat,
  qb: {Nat.is_lt(q,64n) == True{} : Bool},db: {Nat.is_lt(dir,8n) == True{} : Bool},
  chosen: {king(b) == U32.from_nat(q) : U32},
  r: rook(c,q,Chess.get_turn(b),Chess.occupied(b)),
  d: bishop(c,q,Chess.get_turn(b),Chess.occupied(b))) ->
  L.subset(C.path(q,dir,Chess.occupied(b)),S.rays(c,b)):
  L.rewrite(C.path(q,dir,Chess.occupied(b)),C.actual(q,Chess.occupied(b)),
    C.path(q,dir,Chess.occupied(b)),S.rays(c,b),
    {==},prepared(c,b,q,chosen,r,d),C.actual_coverage(q,dir,Chess.occupied(b),qb,db))
