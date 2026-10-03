# Additive full-Ply occurrence accounting for actual Chess.scan.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../destination_factorization/AfterCount.bend as After
import ../destination_factorization/Spec.bend as Spec
import ../table_preservation/Core.bend as Core
import ../table_preservation/Targets.bend as TableTargets
import ../storage/Observe.bend as Observe
import ./StepProbe.bend as Step

def result_moves(r: Array<U64> & List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  (table,moves) = r
  moves

def replace_table(table: Array<U64>,r: Array<U64> & List<&2,Chess.Ply>,
  same: {Core.table(List<&2,Chess.Ply>,r) == table : Array<U64>}) ->
  {r == (table,result_moves(r)) : Array<U64> & List<&2,Chess.Ply>}:
  (old_table,moves) = r
  Equal.cong(Array<U64>,Array<U64> & List<&2,Chess.Ply>,t => (t,moves),old_table,table,same)

# The source code calls actual piece_targets then scan_after. TableTargets.step
# supplies the exact table projection contract; moves are the real scan_after
# output. This is explicit about state even if a future table cache changes.
def actual_step_pair(+b: Chess.Board,+src: U32,+c: Observe.Cells,
  +tail: List<&2,Chess.Ply>) ->
  {Chess.scan_step(b,src,(Observe.pack(c),tail)) ==
    (Observe.pack(c),result_moves(Chess.scan_step(b,src,(Observe.pack(c),tail)))) : Array<U64> & List<&2,Chess.Ply>}:
  replace_table(Observe.pack(c),Chess.scan_step(b,src,(Observe.pack(c),tail)),
    TableTargets.step(b,src,(Observe.pack(c),tail)))

def pair_count_matches_list(r: Array<U64> & List<&2,Chess.Ply>,+query: Chess.Ply) ->
  {After.pair_count(r,query) == Spec.count(result_moves(r),query) : Nat}:
  (table,moves) = r
  {==}

def actual_step_moves_count(+b: Chess.Board,+src: U32,+c: Observe.Cells,
  +tail: List<&2,Chess.Ply>,+query: Chess.Ply) ->
  {Spec.count(result_moves(Chess.scan_step(b,src,(Observe.pack(c),tail))),query) ==
    Step.source_count(src,U64.test_bit(Chess.get_pawns(b),U32.to_nat(src)),Chess.get_ep(b),query,
      Chess.piece_targets(U64.test_bit(Chess.get_pawns(b),U32.to_nat(src)),
        Observe.pack(c),b,src),Spec.count(tail,query)) : Nat}:
  Equal.trans(Nat,Spec.count(result_moves(Chess.scan_step(b,src,(Observe.pack(c),tail))),query),
    After.pair_count(Chess.scan_step(b,src,(Observe.pack(c),tail)),query),
    Step.source_count(src,U64.test_bit(Chess.get_pawns(b),U32.to_nat(src)),Chess.get_ep(b),query,
      Chess.piece_targets(U64.test_bit(Chess.get_pawns(b),U32.to_nat(src)),Observe.pack(c),b,src),
      Spec.count(tail,query)),
    Equal.sym(Nat,After.pair_count(Chess.scan_step(b,src,(Observe.pack(c),tail)),query),
      Spec.count(result_moves(Chess.scan_step(b,src,(Observe.pack(c),tail))),query),
      pair_count_matches_list(Chess.scan_step(b,src,(Observe.pack(c),tail)),query)),
    Step.actual_step_count(b,src,Observe.pack(c),tail,query))

# Right fold follows Chess.scan's actual recursion order. Its Nat accumulator
# is an occurrence count over complete Ply values; duplicates are retained.
def scan_count_fold(+xs: List<&2,U32>,+b: Chess.Board,+c: Observe.Cells,
  +query: Chess.Ply,+initial: Nat) -> Nat:
  match xs:
    case Nil{}: initial
    case Con{+src,+rest}:
      +pawn = U64.test_bit(Chess.get_pawns(b),U32.to_nat(src))
      scan_count_fold(rest,b,c,query,
        Step.source_count(src,pawn,Chess.get_ep(b),query,
          Chess.piece_targets(pawn,Observe.pack(c),b,src),initial))

def scan_count_cells(+xs: List<&2,U32>,+b: Chess.Board,+c: Observe.Cells,
  +tail: List<&2,Chess.Ply>,+query: Chess.Ply) ->
  {After.pair_count(Chess.scan(xs,b,(Observe.pack(c),tail)),query) ==
    scan_count_fold(xs,b,c,query,Spec.count(tail,query)) : Nat}:
  match xs:
    case Nil{}: {==}
    case Con{+src,+rest}:
      +pawn = U64.test_bit(Chess.get_pawns(b),U32.to_nat(src))
      +local_count = Step.source_count(src,pawn,Chess.get_ep(b),query,
        Chess.piece_targets(pawn,Observe.pack(c),b,src),Spec.count(tail,query))
      +step_exact = actual_step_pair(b,src,c,tail)
      +step_count = actual_step_moves_count(b,src,c,tail,query)
      +after_tail = result_moves(Chess.scan_step(b,src,(Observe.pack(c),tail)))
      +rest_count = scan_count_cells(rest,b,c,after_tail,query)
      Equal.trans(Nat,
        After.pair_count(Chess.scan(Con{src,rest},b,(Observe.pack(c),tail)),query),
        scan_count_fold(rest,b,c,query,Spec.count(after_tail,query)),
        scan_count_fold(Con{src,rest},b,c,query,Spec.count(tail,query)),
        Equal.trans(Nat,
          After.pair_count(Chess.scan(Con{src,rest},b,(Observe.pack(c),tail)),query),
          After.pair_count(Chess.scan(rest,b,(Observe.pack(c),after_tail)),query),
          scan_count_fold(rest,b,c,query,Spec.count(after_tail,query)),
          Equal.cong(Array<U64> & List<&2,Chess.Ply>,Nat,r => After.pair_count(Chess.scan(rest,b,r),query),
            Chess.scan_step(b,src,(Observe.pack(c),tail)),(Observe.pack(c),after_tail),step_exact),
          rest_count),
        Equal.trans(Nat,
          scan_count_fold(rest,b,c,query,Spec.count(after_tail,query)),
          scan_count_fold(rest,b,c,query,local_count),
          scan_count_fold(Con{src,rest},b,c,query,Spec.count(tail,query)),
          Equal.cong(Nat,Nat,n => scan_count_fold(rest,b,c,query,n),
            Spec.count(after_tail,query),local_count,step_count),
          {==}))

# Exact table-state invariant for the actual arbitrary-list traversal.
def actual_scan_table_unchanged(+xs: List<&2,U32>,+b: Chess.Board,
  r: Array<U64> & List<&2,Chess.Ply>) ->
  {Core.table(List<&2,Chess.Ply>,Chess.scan(xs,b,r)) ==
    Core.table(List<&2,Chess.Ply>,r) : Array<U64>}:
  TableTargets.scan(xs,b,r)



def actual_scan_count_for_table(+xs: List<&2,U32>,+b: Chess.Board,table: Array<U64>,
  +c: Observe.Cells,+table_image: {Observe.pack(c) == table : Array<U64>},
  +tail: List<&2,Chess.Ply>,+query: Chess.Ply) ->
  {After.pair_count(Chess.scan(xs,b,(table,tail)),query) ==
    scan_count_fold(xs,b,c,query,Spec.count(tail,query)) : Nat}:
  %table_image : {After.pair_count(Chess.scan(xs,b,(_,tail)),query) ==
    scan_count_fold(xs,b,c,query,Spec.count(tail,query)) : Nat}
  %table_image : {After.pair_count(Chess.scan(xs,b,(Observe.pack(c),tail)),query) ==
    scan_count_fold(xs,b,c,query,Spec.count(tail,query)) : Nat}
  scan_count_cells(xs,b,c,tail,query)

def legal_moves_raw_scan_count(+c: Observe.Cells,+b: Chess.Board,+query: Chess.Ply) ->
  {After.pair_count(Chess.scan(Chess.bit_squares(64n,
      U64.is_zero(Chess.color(b,Chess.get_turn(b))),Chess.color(b,Chess.get_turn(b)),Nil{}),b,
      (Observe.pack(c),Nil{})),query) ==
    scan_count_fold(Chess.bit_squares(64n,
      U64.is_zero(Chess.color(b,Chess.get_turn(b))),Chess.color(b,Chess.get_turn(b)),Nil{}),
      b,c,query,0n) : Nat}:
  scan_count_cells(Chess.bit_squares(64n,U64.is_zero(Chess.color(b,Chess.get_turn(b))),
    Chess.color(b,Chess.get_turn(b)),Nil{}),b,c,Nil{},query)

def actual_legal_moves_scan_input(+c: Observe.Cells,+b: Chess.Board) ->
  {Chess.legal_moves(Observe.pack(c),b) == Chess.filter_prepare(b,Chess.castle_side(b,False{},
    Chess.castle_side(b,True{},Chess.scan(Chess.bit_squares(64n,
      U64.is_zero(Chess.color(b,Chess.get_turn(b))),Chess.color(b,Chess.get_turn(b)),Nil{}),
      b,(Observe.pack(c),Nil{}))))) : Array<U64> & List<&2,Chess.Ply>}:
  {==}
