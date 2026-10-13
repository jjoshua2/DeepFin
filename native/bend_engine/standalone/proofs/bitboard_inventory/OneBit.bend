# Exact actual enumeration for every singleton in-range bitboard.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ./Inventory.bend as Inventory
import ../destination_factorization/Spec.bend as S
import ../destination_factorization/Scan.bend as Scan
import ../destination_factorization/Emission.bend as E

def expand_one(+sq: Nat,+src: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>) ->
  {S.expand(U32.from_nat(sq) <> Nil{},src,pawn,ep,tail) ==
    Chess.put_move(S.promotes(pawn,U32.from_nat(sq)),src,U32.from_nat(sq),S.flag(pawn,U32.from_nat(sq),ep),tail) : List<&2,Chess.Ply>}:
  +promote = S.promotes(pawn,U32.from_nat(sq))
  +tag = S.flag(pawn,U32.from_nat(sq),ep)
  Equal.trans(List<&2,Chess.Ply>,S.expand(U32.from_nat(sq) <> Nil{},src,pawn,ep,tail),
    List.append(&2,Chess.Ply,S.block(promote,src,U32.from_nat(sq),tag),tail),
    Chess.put_move(promote,src,U32.from_nat(sq),tag,tail),
    {==},Equal.sym(List<&2,Chess.Ply>,Chess.put_move(promote,src,U32.from_nat(sq),tag,tail),
      List.append(&2,Chess.Ply,S.block(promote,src,U32.from_nat(sq),tag),tail),
      E.put(promote,src,U32.from_nat(sq),tag,tail)))

def scan_one_bit(table: Array<U64>,+sq: Nat,e: {Nat.is_lt(sq,64n) == True{} : Bool},+src: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>) ->
  {Chess.scan_after(src,pawn,ep,tail,(table,U64.bit(sq))) ==
    (table,Chess.put_move(S.promotes(pawn,U32.from_nat(sq)),src,U32.from_nat(sq),S.flag(pawn,U32.from_nat(sq),ep),tail)) : Array<U64> & List<&2,Chess.Ply>}:
  Equal.trans(Array<U64> & List<&2,Chess.Ply>,
    Chess.scan_after(src,pawn,ep,tail,(table,U64.bit(sq))),
    (table,S.expand(Chess.bit_squares(64n,U64.is_zero(U64.bit(sq)),U64.bit(sq),Nil{}),src,pawn,ep,tail)),
    (table,Chess.put_move(S.promotes(pawn,U32.from_nat(sq)),src,U32.from_nat(sq),S.flag(pawn,U32.from_nat(sq),ep),tail)),
    Scan.exact(table,U64.bit(sq),src,pawn,ep,tail),
    Equal.trans(Array<U64> & List<&2,Chess.Ply>,
      (table,S.expand(Chess.bit_squares(64n,U64.is_zero(U64.bit(sq)),U64.bit(sq),Nil{}),src,pawn,ep,tail)),
      (table,S.expand(U32.from_nat(sq) <> Nil{},src,pawn,ep,tail)),
      (table,Chess.put_move(S.promotes(pawn,U32.from_nat(sq)),src,U32.from_nat(sq),S.flag(pawn,U32.from_nat(sq),ep),tail)),
      Equal.cong(List<&2,U32>,Array<U64> & List<&2,Chess.Ply>,xs => (table,S.expand(xs,src,pawn,ep,tail)),
        Chess.bit_squares(64n,U64.is_zero(U64.bit(sq)),U64.bit(sq),Nil{}),U32.from_nat(sq) <> Nil{},
        Inventory.singleton_indices(sq,e)),
      Equal.cong(List<&2,Chess.Ply>,Array<U64> & List<&2,Chess.Ply>,ms => (table,ms),
        S.expand(U32.from_nat(sq) <> Nil{},src,pawn,ep,tail),
        Chess.put_move(S.promotes(pawn,U32.from_nat(sq)),src,U32.from_nat(sq),S.flag(pawn,U32.from_nat(sq),ep),tail),
        expand_one(sq,src,pawn,ep,tail))))

def empty_indices() ->
  {Chess.bit_squares(64n,U64.is_zero(U64.zero()),U64.zero(),Nil{}) == Nil{} : List<&2,U32>}:
  {==}

def scan_empty_targets(table: Array<U64>,+src: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>) ->
  {Chess.scan_after(src,pawn,ep,tail,(table,U64.zero())) == (table,tail) : Array<U64> & List<&2,Chess.Ply>}:
  Equal.trans(Array<U64> & List<&2,Chess.Ply>,
    Chess.scan_after(src,pawn,ep,tail,(table,U64.zero())),
    (table,S.expand(Chess.bit_squares(64n,U64.is_zero(U64.zero()),U64.zero(),Nil{}),src,pawn,ep,tail)),
    (table,tail),
    Scan.exact(table,U64.zero(),src,pawn,ep,tail),
    {==})

# The actual source scanner delegates its computed target mask to piece_targets.
# This corollary states the exact step result when that real producer returns a singleton.
def scan_step_onehot(+b: Chess.Board,+src: U32,table: Array<U64>,+tail: List<&2,Chess.Ply>,
  +sq: Nat,bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  target: {Chess.piece_targets(U64.test_bit(Chess.get_pawns(b),U32.to_nat(src)),table,b,src)
    == (table,U64.bit(sq)) : Array<U64> & U64}) ->
  {Chess.scan_step(b,src,(table,tail)) ==
    (table,Chess.put_move(S.promotes(U64.test_bit(Chess.get_pawns(b),U32.to_nat(src)),U32.from_nat(sq)),src,
      U32.from_nat(sq),S.flag(U64.test_bit(Chess.get_pawns(b),U32.to_nat(src)),U32.from_nat(sq),Chess.get_ep(b)),tail))
    : Array<U64> & List<&2,Chess.Ply>}:
  +pawn = U64.test_bit(Chess.get_pawns(b),U32.to_nat(src))
  +ep = Chess.get_ep(b)
  Equal.trans(Array<U64> & List<&2,Chess.Ply>,Chess.scan_step(b,src,(table,tail)),
    Chess.scan_after(src,pawn,ep,tail,Chess.piece_targets(pawn,table,b,src)),
    (table,Chess.put_move(S.promotes(pawn,U32.from_nat(sq)),src,U32.from_nat(sq),S.flag(pawn,U32.from_nat(sq),ep),tail)),
    {==},
    Equal.trans(Array<U64> & List<&2,Chess.Ply>,
      Chess.scan_after(src,pawn,ep,tail,Chess.piece_targets(pawn,table,b,src)),
      Chess.scan_after(src,pawn,ep,tail,(table,U64.bit(sq))),
      (table,Chess.put_move(S.promotes(pawn,U32.from_nat(sq)),src,U32.from_nat(sq),S.flag(pawn,U32.from_nat(sq),ep),tail)),
      Equal.cong(Array<U64> & U64,Array<U64> & List<&2,Chess.Ply>,r => Chess.scan_after(src,pawn,ep,tail,r),
        Chess.piece_targets(pawn,table,b,src),(table,U64.bit(sq)),target),
      scan_one_bit(table,sq,bound,src,pawn,ep,tail)))

def scan_step_empty(+b: Chess.Board,+src: U32,table: Array<U64>,+tail: List<&2,Chess.Ply>,
  target: {Chess.piece_targets(U64.test_bit(Chess.get_pawns(b),U32.to_nat(src)),table,b,src)
    == (table,U64.zero()) : Array<U64> & U64}) ->
  {Chess.scan_step(b,src,(table,tail)) == (table,tail) : Array<U64> & List<&2,Chess.Ply>}:
  +pawn = U64.test_bit(Chess.get_pawns(b),U32.to_nat(src))
  +ep = Chess.get_ep(b)
  Equal.trans(Array<U64> & List<&2,Chess.Ply>,Chess.scan_step(b,src,(table,tail)),
    Chess.scan_after(src,pawn,ep,tail,Chess.piece_targets(pawn,table,b,src)),
    (table,tail),{==},
    Equal.trans(Array<U64> & List<&2,Chess.Ply>,
      Chess.scan_after(src,pawn,ep,tail,Chess.piece_targets(pawn,table,b,src)),
      Chess.scan_after(src,pawn,ep,tail,(table,U64.zero())),(table,tail),
      Equal.cong(Array<U64> & U64,Array<U64> & List<&2,Chess.Ply>,r => Chess.scan_after(src,pawn,ep,tail,r),
        Chess.piece_targets(pawn,table,b,src),(table,U64.zero()),target),
      scan_empty_targets(table,src,pawn,ep,tail)))

# Connect a one-hot moving-side source mask through actual Chess.scan and its consumer.
def scan_one_source(+b: Chess.Board,table: Array<U64>,+src_sq: Nat,src_bound: {Nat.is_lt(src_sq,64n) == True{} : Bool},
  source: {Chess.color(b,Chess.get_turn(b)) == U64.bit(src_sq) : U64},+dst_sq: Nat,
  dst_bound: {Nat.is_lt(dst_sq,64n) == True{} : Bool},
  target: {Chess.piece_targets(U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(src_sq))),table,b,U32.from_nat(src_sq))
    == (table,U64.bit(dst_sq)) : Array<U64> & U64}) ->
  {Chess.scan(Chess.bit_squares(64n,U64.is_zero(Chess.color(b,Chess.get_turn(b))),Chess.color(b,Chess.get_turn(b)),Nil{}),b,(table,Nil{})) ==
    (table,Chess.put_move(S.promotes(U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(src_sq))),U32.from_nat(dst_sq)),
      U32.from_nat(src_sq),U32.from_nat(dst_sq),S.flag(U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(src_sq))),
        U32.from_nat(dst_sq),Chess.get_ep(b)),Nil{})) : Array<U64> & List<&2,Chess.Ply>}:
  +source_list = Equal.trans(List<&2,U32>,
    Chess.bit_squares(64n,U64.is_zero(Chess.color(b,Chess.get_turn(b))),Chess.color(b,Chess.get_turn(b)),Nil{}),
    Chess.bit_squares(64n,U64.is_zero(U64.bit(src_sq)),U64.bit(src_sq),Nil{}),
    U32.from_nat(src_sq) <> Nil{},
    Equal.cong(U64,List<&2,U32>,x => Chess.bit_squares(64n,U64.is_zero(x),x,Nil{}),
      Chess.color(b,Chess.get_turn(b)),U64.bit(src_sq),source),
    Inventory.singleton_indices(src_sq,src_bound))
  Equal.trans(Array<U64> & List<&2,Chess.Ply>,
    Chess.scan(Chess.bit_squares(64n,U64.is_zero(Chess.color(b,Chess.get_turn(b))),Chess.color(b,Chess.get_turn(b)),Nil{}),b,(table,Nil{})),
    Chess.scan(U32.from_nat(src_sq) <> Nil{},b,(table,Nil{})),
    (table,Chess.put_move(S.promotes(U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(src_sq))),U32.from_nat(dst_sq)),
      U32.from_nat(src_sq),U32.from_nat(dst_sq),S.flag(U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(src_sq))),
        U32.from_nat(dst_sq),Chess.get_ep(b)),Nil{})),
    Equal.cong(List<&2,U32>,Array<U64> & List<&2,Chess.Ply>,xs => Chess.scan(xs,b,(table,Nil{})),
      Chess.bit_squares(64n,U64.is_zero(Chess.color(b,Chess.get_turn(b))),Chess.color(b,Chess.get_turn(b)),Nil{}),
      U32.from_nat(src_sq) <> Nil{},source_list),
    Equal.trans(Array<U64> & List<&2,Chess.Ply>,
      Chess.scan(U32.from_nat(src_sq) <> Nil{},b,(table,Nil{})),
      Chess.scan_step(b,U32.from_nat(src_sq),(table,Nil{})),
      (table,Chess.put_move(S.promotes(U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(src_sq))),U32.from_nat(dst_sq)),
        U32.from_nat(src_sq),U32.from_nat(dst_sq),S.flag(U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(src_sq))),
          U32.from_nat(dst_sq),Chess.get_ep(b)),Nil{})),
      {==},scan_step_onehot(b,U32.from_nat(src_sq),table,Nil{},dst_sq,dst_bound,target)))

# The actual legal generator retains its actual castle-side and filter pipeline after
# this proved singleton scan prefix. Their complete semantics are not claimed here.
def legal_moves_one_source(+b: Chess.Board,table: Array<U64>,+src_sq: Nat,src_bound: {Nat.is_lt(src_sq,64n) == True{} : Bool},
  source: {Chess.color(b,Chess.get_turn(b)) == U64.bit(src_sq) : U64},+dst_sq: Nat,
  dst_bound: {Nat.is_lt(dst_sq,64n) == True{} : Bool},
  target: {Chess.piece_targets(U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(src_sq))),table,b,U32.from_nat(src_sq))
    == (table,U64.bit(dst_sq)) : Array<U64> & U64}) ->
  {Chess.legal_moves(table,b) == Chess.filter_prepare(b,Chess.castle_side(b,False{},Chess.castle_side(b,True{},
    (table,Chess.put_move(S.promotes(U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(src_sq))),U32.from_nat(dst_sq)),
      U32.from_nat(src_sq),U32.from_nat(dst_sq),S.flag(U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(src_sq))),
        U32.from_nat(dst_sq),Chess.get_ep(b)),Nil{}))))) : Array<U64> & List<&2,Chess.Ply>}:
  Equal.trans(Array<U64> & List<&2,Chess.Ply>,Chess.legal_moves(table,b),
    Chess.filter_prepare(b,Chess.castle_side(b,False{},Chess.castle_side(b,True{},
      Chess.scan(Chess.bit_squares(64n,U64.is_zero(Chess.color(b,Chess.get_turn(b))),Chess.color(b,Chess.get_turn(b)),Nil{}),b,(table,Nil{}))))),
    Chess.filter_prepare(b,Chess.castle_side(b,False{},Chess.castle_side(b,True{},
      (table,Chess.put_move(S.promotes(U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(src_sq))),U32.from_nat(dst_sq)),
        U32.from_nat(src_sq),U32.from_nat(dst_sq),S.flag(U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(src_sq))),
          U32.from_nat(dst_sq),Chess.get_ep(b)),Nil{}))))),
    {==},Equal.cong(Array<U64> & List<&2,Chess.Ply>,Array<U64> & List<&2,Chess.Ply>,
      r => Chess.filter_prepare(b,Chess.castle_side(b,False{},Chess.castle_side(b,True{},r))),
      Chess.scan(Chess.bit_squares(64n,U64.is_zero(Chess.color(b,Chess.get_turn(b))),Chess.color(b,Chess.get_turn(b)),Nil{}),b,(table,Nil{})),
      (table,Chess.put_move(S.promotes(U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(src_sq))),U32.from_nat(dst_sq)),
        U32.from_nat(src_sq),U32.from_nat(dst_sq),S.flag(U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(src_sq))),
          U32.from_nat(dst_sq),Chess.get_ep(b)),Nil{})),
      scan_one_source(b,table,src_sq,src_bound,source,dst_sq,dst_bound,target)))
