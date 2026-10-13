# The ordinary scan observes the same exact array at every source query.
import Base
import ./Spec.bend as S
import ./Emission.bend as Emit
import ../storage/Observe.bend as O
import ../storage/Representation.bend as Reify
import ../table_preservation/Core.bend as T
import ../table_preservation/Targets.bend as Targets
import ../bitboard_membership/AllEmittedSound.bend as All
import ../bitboard_membership/PrefixDecomposition.bend as Prefix
import ../child_metadata/Source.bend as Source
import ../child_metadata/Range.bend as R
import ../castle_chain/Spec.bend as C
import ../castle_chain/Chain.bend as Chain
import ../castle_chain/Lists.bend as Lists
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as Chess

def step_pawn(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,+c: O.Cells,+b: Chess.Board,+s: U32,
  +pawn: Bool,+tail: List<&2,Chess.Ply>,samepawn: {S.pawn(b,s) == pawn : Bool},
  bound: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},good: C.every(m => S.cells(c,b,m),tail),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> {T.table(List<&2,Chess.Ply>,r) == O.pack(c) : Array<U64>} -> C.every(m => S.cells(c,b,m),E.moves(r)) -> Goal(r)) ->
  Goal(Chess.scan_after(s,pawn,Chess.get_ep(b),tail,Chess.piece_targets(pawn,O.pack(c),b,s))):
  match pawn:
    case True{}:
      Emit.after(Goal,c,b,s,True{},Chess.get_ep(b),tail,Chess.pawn_targets(O.pack(c),b,s),
        Targets.pawn(O.pack(c),b,s),bound,samepawn,{==},good,next)
    case False{}:
      Emit.after(Goal,c,b,s,False{},Chess.get_ep(b),tail,Chess.piece_targets(False{},O.pack(c),b,s),
        Targets.piece(False{},O.pack(c),b,s),bound,samepawn,{==},good,next)

def step_fixed(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,+c: O.Cells,+b: Chess.Board,+s: U32,+tail: List<&2,Chess.Ply>,
  bound: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},good: C.every(m => S.cells(c,b,m),tail),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> {T.table(List<&2,Chess.Ply>,r) == O.pack(c) : Array<U64>} -> C.every(m => S.cells(c,b,m),E.moves(r)) -> Goal(r)) ->
  Goal(Chess.scan_step(b,s,(O.pack(c),tail))):
  step_pawn(Goal,c,b,s,S.pawn(b,s),tail,{==},bound,good,next)

def step(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,+c: O.Cells,+b: Chess.Board,+s: U32,r: Array<U64> & List<&2,Chess.Ply>,
  same: {T.table(List<&2,Chess.Ply>,r) == O.pack(c) : Array<U64>},bound: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},
  good: C.every(m => S.cells(c,b,m),E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> {T.table(List<&2,Chess.Ply>,r) == O.pack(c) : Array<U64>} -> C.every(m => S.cells(c,b,m),E.moves(r)) -> Goal(r)) ->
  Goal(Chess.scan_step(b,s,r)):
  (table,+tail)=r
  %Equal.sym(Array<U64>,table,O.pack(c),same) : Goal(Chess.scan_step(b,s,(_,tail)))
  step_fixed(Goal,c,b,s,tail,bound,good,next)

def scan(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,+c: O.Cells,keys: List<&2,U32>,+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,
  same: {T.table(List<&2,Chess.Ply>,r) == O.pack(c) : Array<U64>},+below: {All.all_below64(keys) == True{} : Bool},
  good: C.every(m => S.cells(c,b,m),E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> {T.table(List<&2,Chess.Ply>,r) == O.pack(c) : Array<U64>} -> C.every(m => S.cells(c,b,m),E.moves(r)) -> Goal(r)) ->
  Goal(Chess.scan(keys,b,r)):
  match keys:
    case Nil{}: next(r,same,good)
    case Con{+s,+rest}:
      +head=All.bool_and_true_left(Nat.is_lt(U32.to_nat(s),64n),All.all_below64(rest),below)
      +later=All.bool_and_true_right(Nat.is_lt(U32.to_nat(s),64n),All.all_below64(rest),below)
      step(z => Goal(Chess.scan(rest,b,z)),c,b,s,r,same,head,good,
        z => table_same => fields => scan(Goal,c,rest,b,z,table_same,later,fields,next))

def initial(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,+c: O.Cells,+b: Chess.Board,
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => S.cells(c,b,m),E.moves(r)) -> Goal(r)) ->
  Goal(C.candidates(O.pack(c),b)):
  +own=Chess.color(b,Chess.get_turn(b))
  scan(Goal,c,Chess.bit_squares(64n,U64.is_zero(own),own,Nil{}),b,(O.pack(c),Nil{}),{==},
    All.actual_bit_squares_below64(own),Unit{},r => same => fields => next(r,fields))

def castle_xor(white: Bool,ks: Bool) ->
  {S.trigger(U32.add(Bool.pick(U32,white,0,56),4),U32.add(Bool.pick(U32,white,0,56),Bool.pick(U32,ks,6,2))) == False{} : Bool}:
  match white ks:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

def castle(+c: O.Cells,+b: Chess.Board,+ks: Bool) -> S.cells(c,b,E.move(b,ks)):
  (Source.castle_bound(b,ks),
    %Equal.sym(Bool,S.trigger(R.src(E.move(b,ks)),S.dst(E.move(b,ks))),False{},
      castle_xor(U32.is_eq(Chess.get_turn(b),1),ks)) :
        S.gate(Bool.and(S.pawn(b,R.src(E.move(b,ks))),_),O.pack(c),b,R.src(E.move(b,ks)),S.dst(E.move(b,ks)))
    %Equal.sym(Bool,Bool.and(S.pawn(b,R.src(E.move(b,ks))),False{}),False{},
      Prefix.and_false_right(S.pawn(b,R.src(E.move(b,ks))))) :
        S.gate(_,O.pack(c),b,R.src(E.move(b,ks)),S.dst(E.move(b,ks)))
    Unit{})

def legal(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,+c: O.Cells,+b: Chess.Board,
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => S.cells(c,b,m),E.moves(r)) -> Goal(r)) ->
  Goal(Chess.legal_moves(O.pack(c),b)):
  initial(z => Goal(C.suffix(b,z)),c,b,
    r => fields => Chain.suffix(m => S.cells(c,b,m),Goal,b,r,fields,
      guard => castle(c,b,True{}),guard => castle(c,b,False{}),next))

def cells_member(+c: O.Cells,+b: Chess.Board,+m: Chess.Ply,
  here: E.member(m,E.moves(Chess.legal_moves(O.pack(c),b)))) -> S.cells(c,b,m):
  legal(r => E.member(m,E.moves(r)) -> S.cells(c,b,m),c,b,
    r => fields => Lists.extract(x => S.cells(c,b,x),r,m,fields))(here)

def lift(-table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,
  cert: Exists(O.Cells,c => {O.pack(c) == table : Array<U64>})) ->
  E.member(m,E.moves(Chess.legal_moves(table,b))) -> S.occurrence(table,b,m):
  (+c,e)=cert
  +same: {O.pack(c) == table : Array<U64>}=e
  %same : E.member(m,E.moves(Chess.legal_moves(_,b))) -> S.occurrence(table,b,m)
  %same : E.member(m,E.moves(Chess.legal_moves(O.pack(c),b))) -> S.occurrence(_,b,m)
  here => cells_member(c,b,m,here)

def member(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,
  here: E.member(m,E.moves(Chess.legal_moves(table,b)))) -> S.occurrence(table,b,m):
  lift(table,b,m,Reify.reify(table))(here)
