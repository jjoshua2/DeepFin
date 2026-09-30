import Base
import ./PROOF.bend as Proof
import ./LAWS.bend as Laws
import ./Spec.bend as S
import ./Reduction.bend as R
import ./Queries.bend as Q
import ../../../legal_probe/Chess.bend as Chess
import ../../Position.bend as P

# Universally instantiated public laws, not just finite examples.
def reduce(+b: Chess.Board,+by: U32,+m: S.Masks) ->
  {Bool.not(U64.is_zero(U64.and(R.hits(b,m),Chess.color(b,by)))) == S.present(b,by,m) : Bool}:
  Laws.mask_reduction_has_square_witness(b,by,m)

def attacked(table: Array<U64>,+b: Chess.Board,+sq: U32,+by: U32) ->
  {Chess.attacked(table,b,sq,by) == Q.run(table,b,sq,by) : Array<U64> & Bool}:
  Laws.actual_attacked_uses_witness_reduction(table,b,sq,by)

def check(table: Array<U64>,+b: Chess.Board,+side: U32,+sq: Nat,bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  one: {U64.and(Chess.get_kings(b),Chess.color(b,side)) == U64.bit(sq) : U64}) ->
  {Chess.in_check(table,b,side) == Q.run(table,b,U32.from_nat(sq),U32.xor(side,1)) : Array<U64> & Bool}:
  Laws.single_king_in_check_uses_witness_reduction(table,b,side,sq,bound,one)

# No-king boards are not silently treated as a valid queried king at square zero.
def missing_king_index() -> {U64.ctz(U64.and(Chess.get_kings(P.empty()),Chess.color(P.empty(),1))) == 64 : U32}:
  {==}

# Multiple kings select the lowest occupied king bit; uniqueness must be supplied.
def two_kings() -> Chess.Board:
  P.put(5,True{},63,P.put(5,True{},0,P.empty()))

def first_king_only() -> {U64.ctz(U64.and(Chess.get_kings(two_kings()),Chess.color(two_kings(),1))) == 0 : U32}:
  {==}

# Queens can witness either slider class; wrong-color pieces must not witness it.
def queen() -> Chess.Board:
  P.put(4,False{},63,P.empty())

def diagonal_queen() -> {S.present(queen(),0,S.Masks{U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.bit(63n)}) == True{} : Bool}:
  {==}

def wrong_color() -> {S.present(queen(),1,S.Masks{U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.bit(63n)}) == False{} : Bool}:
  {==}
