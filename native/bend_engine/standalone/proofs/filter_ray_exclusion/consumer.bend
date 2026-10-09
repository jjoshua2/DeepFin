import Base
import ./Filter.bend as F
import ../castle_safety/Spec.bend as S
import ../ray/Path.bend as P
import ../ray/Spec.bend as R
import ../ray_clear_disjoint/Algebra.bend as D
import ../occupancy_insertion/Lattice.bend as L
import ../../Tables.bend as Tables
import ../../../legal_probe/Chess.bend as Chess

def exclude(+b: Chess.Board,+rays: U64,+m: Chess.Ply,
  bound: {Nat.is_lt(U32.to_nat(S.source(m)),64n) == True{} : Bool},
  owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(S.source(m))) == True{} : Bool},
  bypass: {Chess.filter_requires(S.sensitive(b,rays),m) == False{} : Bool}) ->
  D.disjoint(U64.bit(U32.to_nat(S.source(m))),rays):
  match m:
    case Chess.Ply{src,dst,promotion,flag}: F.source_exclusion(b,rays,src,dst,promotion,flag,bound,owned,bypass)

def ray(+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+b: Chess.Board,+src: U32,+dst: U32,+rays: U64,+acc: U64,
  route: {R.trace(n,sq,dir,xs) == True{} : Bool},
  bound: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool},
  owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(src)) == True{} : Bool},
  bypass: {Chess.filter_requires(S.sensitive(b,rays),Chess.Ply{src,dst,0,0}) == False{} : Bool},
  coverage: L.subset(P.attack(xs,Chess.occupied(b)),rays)) ->
  L.subset(Tables.ray(n,False{},sq,dir,Chess.occupied(Chess.make_move(b,Chess.Ply{src,dst,0,0})),False{},acc),
    Tables.ray(n,False{},sq,dir,Chess.occupied(b),False{},acc)):
  F.actual_ray(n,sq,dir,xs,b,src,dst,rays,acc,route,bound,owned,bypass,coverage)

def board(white: U64,black: U64) -> Chess.Board:
  Chess.Board{U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),white,black,1,0,64}
def path() -> List<&2,Nat>: 1n <> 2n <> 3n <> Nil{}
def example() -> Chess.Board: board(U64.from_u32(12),U64.zero())
def owned() -> {U64.test_bit(Chess.color(example(),Chess.get_turn(example())),3n) == True{} : Bool}: {==}
def bypass() -> {Chess.filter_requires(S.sensitive(example(),U64.from_u32(6)),Chess.Ply{3,1,0,0}) == False{} : Bool}: {==}
def coverage() -> L.subset(P.attack(path(),Chess.occupied(example())),U64.from_u32(6)): {==}
def source_clear() -> D.disjoint(U64.bit(3n),U64.from_u32(6)):
  F.source_exclusion(example(),U64.from_u32(6),3,1,0,0,{==},owned(),bypass())
def covered_old_clear() -> D.disjoint(U64.bit(3n),P.attack(path(),Chess.occupied(example()))):
  F.path_disjoint(example(),U64.from_u32(6),path(),3,1,0,0,{==},owned(),bypass(),coverage())
def board_ray(+acc: U64) ->
  L.subset(Tables.ray(3n,False{},0,0,Chess.occupied(Chess.make_move(example(),Chess.Ply{3,1,0,0})),False{},acc),
    Tables.ray(3n,False{},0,0,Chess.occupied(example()),False{},acc)):
  F.actual_ray(3n,0,0,path(),example(),3,1,U64.from_u32(6),acc,{==},{==},owned(),bypass(),coverage())
def old_ray() -> {Tables.ray(3n,False{},0,0,Chess.occupied(example()),False{},U64.zero()) == U64.from_u32(6) : U64}: {==}
def new_ray() -> {Tables.ray(3n,False{},0,0,Chess.occupied(Chess.make_move(example(),Chess.Ply{3,1,0,0})),False{},U64.zero()) == U64.from_u32(2) : U64}: {==}

# Missing ownership: the actual filter can be False while src still lies in rays.
def unowned_bypass() ->
  {Chess.filter_requires(S.sensitive(board(U64.zero(),U64.zero()),U64.from_u32(4)),Chess.Ply{2,0,0,0}) == False{} : Bool}: {==}
def unowned_source() -> {U64.test_bit(U64.zero(),2n) == False{} : Bool}: {==}
def unowned_intersection() -> {U64.and(U64.bit(2n),U64.from_u32(4)) == U64.from_u32(4) : U64}: {==}

# Missing bypass: ownership holds but the actual filter is True.
def required_owned() -> {U64.test_bit(U64.from_u32(4),2n) == True{} : Bool}: {==}
def required_filter() ->
  {Chess.filter_requires(S.sensitive(board(U64.from_u32(4),U64.zero()),U64.from_u32(4)),Chess.Ply{2,0,0,0}) == True{} : Bool}: {==}
def required_intersection() -> {U64.and(U64.bit(2n),U64.from_u32(4)) == U64.from_u32(4) : U64}: {==}

# Coverage gap: owned source2 is excluded from rays2 but is the old first blocker.
def uncovered() -> Chess.Board: board(U64.from_u32(4),U64.zero())
def uncovered_owned() -> {U64.test_bit(Chess.color(uncovered(),Chess.get_turn(uncovered())),2n) == True{} : Bool}: {==}
def uncovered_bypass() ->
  {Chess.filter_requires(S.sensitive(uncovered(),U64.from_u32(2)),Chess.Ply{2,0,0,0}) == False{} : Bool}: {==}
def uncovered_exclusion() -> D.disjoint(U64.bit(2n),U64.from_u32(2)):
  F.source_exclusion(uncovered(),U64.from_u32(2),2,0,0,0,{==},uncovered_owned(),uncovered_bypass())
def uncovered_old() -> {Tables.ray(3n,False{},0,0,Chess.occupied(uncovered()),False{},U64.zero()) == U64.from_u32(6) : U64}: {==}
def uncovered_new() ->
  {Tables.ray(3n,False{},0,0,Chess.occupied(Chess.make_move(uncovered(),Chess.Ply{2,0,0,0})),False{},U64.zero()) == U64.from_u32(14) : U64}: {==}
def uncovered_intersection() ->
  {U64.and(Tables.ray(3n,False{},0,0,Chess.occupied(uncovered()),False{},U64.zero()),
    Tables.ray(3n,False{},0,0,Chess.occupied(Chess.make_move(uncovered(),Chess.Ply{2,0,0,0})),False{},U64.zero())) == U64.from_u32(6) : U64}: {==}
def coverage_false() ->
  {U64.is_eq(U64.and(U64.from_u32(2),P.attack(path(),Chess.occupied(uncovered()))),P.attack(path(),Chess.occupied(uncovered()))) == False{} : Bool}: {==}

def ep_required() ->
  {Chess.filter_requires(U64.zero(),Chess.Ply{3,1,0,1}) == True{} : Bool}: {==}
