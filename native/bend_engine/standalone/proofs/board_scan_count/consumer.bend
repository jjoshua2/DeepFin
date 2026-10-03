# Import the all-source structural count and exact production caller.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../storage/Observe.bend as Observe
import ../destination_factorization/AfterCount.bend as After
import ../destination_factorization/Spec.bend as Spec
import ./Traversal.bend as T

def all_sources_any_tail(+xs: List<&2,U32>,+b: Chess.Board,+c: Observe.Cells,
  +tail: List<&2,Chess.Ply>,+query: Chess.Ply) ->
  {After.pair_count(Chess.scan(xs,b,(Observe.pack(c),tail)),query) ==
    T.scan_count_fold(xs,b,c,query,Spec.count(tail,query)) : Nat}:
  T.scan_count_cells(xs,b,c,tail,query)

def legal_moves_scan_count(+c: Observe.Cells,+b: Chess.Board,+query: Chess.Ply) ->
  {After.pair_count(Chess.scan(Chess.bit_squares(64n,
      U64.is_zero(Chess.color(b,Chess.get_turn(b))),Chess.color(b,Chess.get_turn(b)),Nil{}),
      b,(Observe.pack(c),Nil{})),query) ==
    T.scan_count_fold(Chess.bit_squares(64n,
      U64.is_zero(Chess.color(b,Chess.get_turn(b))),Chess.color(b,Chess.get_turn(b)),Nil{}),
      b,c,query,0n) : Nat}:
  T.legal_moves_raw_scan_count(c,b,query)

def legal_moves_calls_scanned_list(+c: Observe.Cells,+b: Chess.Board) ->
  {Chess.legal_moves(Observe.pack(c),b) == Chess.filter_prepare(b,Chess.castle_side(b,False{},
    Chess.castle_side(b,True{},Chess.scan(Chess.bit_squares(64n,
      U64.is_zero(Chess.color(b,Chess.get_turn(b))),Chess.color(b,Chess.get_turn(b)),Nil{}),
      b,(Observe.pack(c),Nil{}))))) : Array<U64> & List<&2,Chess.Ply>}:
  T.actual_legal_moves_scan_input(c,b)
