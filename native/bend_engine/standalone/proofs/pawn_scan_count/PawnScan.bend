import Base
import ../../../legal_probe/Chess.bend as Chess
import ../attack_geometry/Initialized.bend as Init
import ../destination_factorization/QueryFactor.bend as Factor
import ../destination_factorization/AfterCount.bend as AfterCount
import ../destination_factorization/Spec.bend as Spec
import ../destination_factorization/QueryCount.bend as Q

def initialized_table(d: Nat,seed: U64,n: Nat,key: U32,at: U32,extra: Nat) -> Array<U64>:
  Init.run(d,seed,n,key,at,extra)

# Mirrors pawn_targets: guarded single/double advances plus pawn attacks against
# enemy non-Kings or the implementation-validated en-passant target.
def pawn_targets_formula(table: Array<U64>,+b: Chess.Board,+src: U32) -> Array<U64> & U64:
  +white = U32.is_eq(Chess.get_turn(b),1)
  +step = Bool.pick(U32,white,8,4294967288)
  +first = U32.add(src,step)
  +second = U32.add(first,step)
  +occ = Chess.occupied(b)
  +empty1 = Bool.and(U32.is_lt(first,64),Bool.not(U64.test_bit(occ,U32.to_nat(first))))
  start_rank = Bool.pick(U32,white,1,6)
  empty2 = Bool.and(empty1,Bool.and(U32.is_eq(U32.div(src,8),start_rank),
    Bool.not(U64.test_bit(occ,U32.to_nat(second)))))
  advances = U64.or(Chess.select_u64(empty1,U64.bit(U32.to_nat(first)),U64.zero()),
    Chess.select_u64(empty2,U64.bit(U32.to_nat(second)),U64.zero()))
  enemies = U64.and_not(Chess.color(b,U32.xor(Chess.get_turn(b),1)),Chess.get_kings(b))
  capturable = U64.or(enemies,Chess.ep_target(b))
  Chess.pawn_after(advances,capturable,
    Chess.attack(0,table,src,Chess.get_turn(b),occ))

def piece_targets_pawn(+d: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,
  +b: Chess.Board,+sq: Nat,depth: {d == 17n : Nat},full: {extra == 64n : Nat},
  bound: {Nat.is_lt(sq,64n) == True{} : Bool}) ->
  {Chess.piece_targets(True{},initialized_table(d,seed,n,key,at,extra),b,U32.from_nat(sq)) ==
    pawn_targets_formula(initialized_table(d,seed,n,key,at,extra),b,U32.from_nat(sq)) :
      Array<U64> & U64}:
  {==}

def scan_step_pawn_closed_count(+d: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,
  +b: Chess.Board,+sq: Nat,+query: Chess.Ply,+tail: List<&2,Chess.Ply>,+targets: U64,
  depth: {d == 17n : Nat},full: {extra == 64n : Nat},
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  pawn_present: {U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(sq))) == True{} : Bool},
  target_exact: {pawn_targets_formula(initialized_table(d,seed,n,key,at,extra),b,
    U32.from_nat(sq)) == (initialized_table(d,seed,n,key,at,extra),targets) : Array<U64> & U64}) ->
  {AfterCount.pair_count(Chess.scan_step(b,U32.from_nat(sq),
      (initialized_table(d,seed,n,key,at,extra),tail)),query) ==
    Nat.add(Bool.pick(Nat,Bool.and(
      Nat.is_lt(U32.to_nat(Q.ply_destination(query)),64n),
      U64.test_bit(targets,U32.to_nat(Q.ply_destination(query)))),
      Factor.query_emission_count(U32.from_nat(sq),True{},Chess.get_ep(b),query),0n),
      Spec.count(tail,query)) : Nat}:
  actual_targets: {Chess.piece_targets(True{},initialized_table(d,seed,n,key,at,extra),b,
    U32.from_nat(sq)) == pawn_targets_formula(initialized_table(d,seed,n,key,at,extra),b,
      U32.from_nat(sq)) : Array<U64> & U64} =
    piece_targets_pawn(d,seed,n,key,at,extra,b,sq,depth,full,bound)
  target_pair: {Chess.piece_targets(True{},initialized_table(d,seed,n,key,at,extra),b,
      U32.from_nat(sq)) == (initialized_table(d,seed,n,key,at,extra),targets) : Array<U64> & U64} =
    Equal.trans(Array<U64> & U64,
      Chess.piece_targets(True{},initialized_table(d,seed,n,key,at,extra),b,U32.from_nat(sq)),
      pawn_targets_formula(initialized_table(d,seed,n,key,at,extra),b,U32.from_nat(sq)),
      (initialized_table(d,seed,n,key,at,extra),targets),actual_targets,target_exact)
  step_expands: {Chess.scan_step(b,U32.from_nat(sq),(initialized_table(d,seed,n,key,at,extra),tail)) ==
    Chess.scan_after(U32.from_nat(sq),U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(sq))),Chess.get_ep(b),tail,
      Chess.piece_targets(U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(sq))),initialized_table(d,seed,n,key,at,extra),b,U32.from_nat(sq))) :
      Array<U64> & List<&2,Chess.Ply>} = {==}
  pawn_specialized: {Chess.scan_after(U32.from_nat(sq),U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(sq))),Chess.get_ep(b),tail,
      Chess.piece_targets(U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(sq))),initialized_table(d,seed,n,key,at,extra),b,U32.from_nat(sq))) ==
    Chess.scan_after(U32.from_nat(sq),True{},Chess.get_ep(b),tail,Chess.piece_targets(True{},initialized_table(d,seed,n,key,at,extra),b,U32.from_nat(sq))) :
      Array<U64> & List<&2,Chess.Ply>} =
    Equal.cong(Bool,Array<U64> & List<&2,Chess.Ply>,
      p => Chess.scan_after(U32.from_nat(sq),p,Chess.get_ep(b),tail,Chess.piece_targets(p,initialized_table(d,seed,n,key,at,extra),b,U32.from_nat(sq))),
      U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(sq))),True{},pawn_present)
  target_specialized: {Chess.scan_after(U32.from_nat(sq),True{},Chess.get_ep(b),tail,
      Chess.piece_targets(True{},initialized_table(d,seed,n,key,at,extra),b,U32.from_nat(sq))) ==
    Chess.scan_after(U32.from_nat(sq),True{},Chess.get_ep(b),tail,
      (initialized_table(d,seed,n,key,at,extra),targets)) :
      Array<U64> & List<&2,Chess.Ply>} =
    Equal.cong(Array<U64> & U64,Array<U64> & List<&2,Chess.Ply>,
      r => Chess.scan_after(U32.from_nat(sq),True{},Chess.get_ep(b),tail,r),
      Chess.piece_targets(True{},initialized_table(d,seed,n,key,at,extra),b,U32.from_nat(sq)),
      (initialized_table(d,seed,n,key,at,extra),targets),target_pair)
  step_exact: {Chess.scan_step(b,U32.from_nat(sq),(initialized_table(d,seed,n,key,at,extra),tail)) ==
    Chess.scan_after(U32.from_nat(sq),True{},Chess.get_ep(b),tail,(initialized_table(d,seed,n,key,at,extra),targets)) :
      Array<U64> & List<&2,Chess.Ply>} =
    Equal.trans(Array<U64> & List<&2,Chess.Ply>,
      Chess.scan_step(b,U32.from_nat(sq),(initialized_table(d,seed,n,key,at,extra),tail)),
      Chess.scan_after(U32.from_nat(sq),U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(sq))),
        Chess.get_ep(b),tail,Chess.piece_targets(U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(sq))),
          initialized_table(d,seed,n,key,at,extra),b,U32.from_nat(sq))),
      Chess.scan_after(U32.from_nat(sq),True{},Chess.get_ep(b),tail,
        (initialized_table(d,seed,n,key,at,extra),targets)),
      step_expands,Equal.trans(Array<U64> & List<&2,Chess.Ply>,
        Chess.scan_after(U32.from_nat(sq),U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(sq))),Chess.get_ep(b),tail,
          Chess.piece_targets(U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(sq))),initialized_table(d,seed,n,key,at,extra),b,U32.from_nat(sq))),
        Chess.scan_after(U32.from_nat(sq),True{},Chess.get_ep(b),tail,Chess.piece_targets(True{},initialized_table(d,seed,n,key,at,extra),b,U32.from_nat(sq))),
        Chess.scan_after(U32.from_nat(sq),True{},Chess.get_ep(b),tail,(initialized_table(d,seed,n,key,at,extra),targets)),
        pawn_specialized,target_specialized))

  Equal.trans(Nat,AfterCount.pair_count(Chess.scan_step(b,U32.from_nat(sq),(initialized_table(d,seed,n,key,at,extra),tail)),query),
    AfterCount.pair_count(Chess.scan_after(U32.from_nat(sq),True{},Chess.get_ep(b),tail,(initialized_table(d,seed,n,key,at,extra),targets)),query),
    Nat.add(Bool.pick(Nat,Bool.and(Nat.is_lt(U32.to_nat(Q.ply_destination(query)),64n),
      U64.test_bit(targets,U32.to_nat(Q.ply_destination(query)))),
      Factor.query_emission_count(U32.from_nat(sq),True{},Chess.get_ep(b),query),0n),Spec.count(tail,query)),
    Equal.cong(Array<U64> & List<&2,Chess.Ply>,Nat,r => AfterCount.pair_count(r,query),
      Chess.scan_step(b,U32.from_nat(sq),(initialized_table(d,seed,n,key,at,extra),tail)),
      Chess.scan_after(U32.from_nat(sq),True{},Chess.get_ep(b),tail,(initialized_table(d,seed,n,key,at,extra),targets)),step_exact),
    Factor.scan_after_closed_count(initialized_table(d,seed,n,key,at,extra),targets,U32.from_nat(sq),True{},Chess.get_ep(b),query,tail))
