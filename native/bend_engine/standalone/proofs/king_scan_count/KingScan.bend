# Compose checked initialized king geometry with the actual occupancy-filtered
# piece target producer and the closed full-Ply scan_after occurrence theorem.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../attack_geometry/consumer.bend as Attack
import ../attack_geometry/Geometry.bend as G
import ../attack_geometry/Initialized.bend as Init
import ../destination_factorization/QueryFactor.bend as Factor
import ../destination_factorization/AfterCount.bend as AfterCount
import ../destination_factorization/Spec.bend as Spec
import ../destination_factorization/QueryCount.bend as Q

def initialized_table(d: Nat,seed: U64,n: Nat,key: U32,at: U32,extra: Nat) -> Array<U64>:
  Init.run(d,seed,n,key,at,extra)

def own_excluded_king_targets(+b: Chess.Board,+sq: Nat) -> U64:
  U64.and_not(G.mask(G.King{},sq),U64.or(Chess.color(b,Chess.get_turn(b)),Chess.get_kings(b)))

def piece_targets_king(+d: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,
  +b: Chess.Board,+sq: Nat,
  depth: {d == 17n : Nat},full: {extra == 64n : Nat},
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  piece: {Chess.piece(b,U32.from_nat(sq)) == 5 : U32},
  king_present: {U64.test_bit(Chess.get_kings(b),U32.to_nat(U32.from_nat(sq))) == True{} : Bool}) ->
  {Chess.piece_targets(False{},initialized_table(d,seed,n,key,at,extra),b,U32.from_nat(sq)) ==
    (initialized_table(d,seed,n,key,at,extra),own_excluded_king_targets(b,sq)) : Array<U64> & U64}:
  +src = U32.from_nat(sq)
  geometry = Attack.initialized_all(d,seed,n,key,at,extra,G.King{},sq,Chess.occupied(b),depth,full,bound)
  piece_attack: {Chess.attack(Chess.piece(b,src),initialized_table(d,seed,n,key,at,extra),src,
    Chess.get_turn(b),Chess.occupied(b)) ==
    (initialized_table(d,seed,n,key,at,extra),G.mask(G.King{},sq)) : Array<U64> & U64} =
    Equal.trans(Array<U64> & U64,
      Chess.attack(Chess.piece(b,src),initialized_table(d,seed,n,key,at,extra),src,
        Chess.get_turn(b),Chess.occupied(b)),
      Chess.attack(5,initialized_table(d,seed,n,key,at,extra),src,Chess.get_turn(b),Chess.occupied(b)),
      (initialized_table(d,seed,n,key,at,extra),G.mask(G.King{},sq)),
      Equal.cong(U32,Array<U64> & U64,k => Chess.attack(k,initialized_table(d,seed,n,key,at,extra),
        src,Chess.get_turn(b),Chess.occupied(b)),Chess.piece(b,src),5,piece),geometry)
  filtered: {Chess.exclude_targets(U64.or(Chess.color(b,Chess.get_turn(b)),Chess.get_kings(b)),
      Chess.attack(Chess.piece(b,src),initialized_table(d,seed,n,key,at,extra),src,
        Chess.get_turn(b),Chess.occupied(b))) ==
    (initialized_table(d,seed,n,key,at,extra),own_excluded_king_targets(b,sq)) : Array<U64> & U64} =
    Equal.trans(Array<U64> & U64,
      Chess.exclude_targets(U64.or(Chess.color(b,Chess.get_turn(b)),Chess.get_kings(b)),
        Chess.attack(Chess.piece(b,src),initialized_table(d,seed,n,key,at,extra),src,
          Chess.get_turn(b),Chess.occupied(b))),
      Chess.exclude_targets(U64.or(Chess.color(b,Chess.get_turn(b)),Chess.get_kings(b)),
        (initialized_table(d,seed,n,key,at,extra),G.mask(G.King{},sq))),
      (initialized_table(d,seed,n,key,at,extra),own_excluded_king_targets(b,sq)),
      Equal.cong(Array<U64> & U64,Array<U64> & U64,
        r => Chess.exclude_targets(U64.or(Chess.color(b,Chess.get_turn(b)),Chess.get_kings(b)),r),
        Chess.attack(Chess.piece(b,src),initialized_table(d,seed,n,key,at,extra),src,
          Chess.get_turn(b),Chess.occupied(b)),
        (initialized_table(d,seed,n,key,at,extra),G.mask(G.King{},sq)),piece_attack),{==})
  Equal.trans(Array<U64> & U64,
    Chess.piece_targets(False{},initialized_table(d,seed,n,key,at,extra),b,src),
    Chess.exclude_targets(U64.or(Chess.color(b,Chess.get_turn(b)),Chess.get_kings(b)),
      Chess.attack(Chess.piece(b,src),initialized_table(d,seed,n,key,at,extra),src,
        Chess.get_turn(b),Chess.occupied(b))),
    (initialized_table(d,seed,n,key,at,extra),own_excluded_king_targets(b,sq)),
    {==},filtered)

def scan_step_king_closed_count(+d: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,
  +b: Chess.Board,+sq: Nat,+query: Chess.Ply,+tail: List<&2,Chess.Ply>,
  depth: {d == 17n : Nat},full: {extra == 64n : Nat},
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  piece: {Chess.piece(b,U32.from_nat(sq)) == 5 : U32},
  king_present: {U64.test_bit(Chess.get_kings(b),U32.to_nat(U32.from_nat(sq))) == True{} : Bool},
  pawn_clear: {U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(sq))) == False{} : Bool}) ->
  {AfterCount.pair_count(Chess.scan_step(b,U32.from_nat(sq),(initialized_table(d,seed,n,key,at,extra),tail)),query) ==
    Nat.add(Bool.pick(Nat,Bool.and(Nat.is_lt(U32.to_nat(Q.ply_destination(query)),64n),
      U64.test_bit(own_excluded_king_targets(b,sq),U32.to_nat(Q.ply_destination(query)))),
      Factor.query_emission_count(U32.from_nat(sq),False{},Chess.get_ep(b),query),0n),
      Spec.count(tail,query)) : Nat}:
  +src = U32.from_nat(sq)
  +targets = own_excluded_king_targets(b,sq)
  target_pair = piece_targets_king(d,seed,n,key,at,extra,b,sq,depth,full,bound,piece,king_present)
  step_expands: {Chess.scan_step(b,src,(initialized_table(d,seed,n,key,at,extra),tail)) ==
    Chess.scan_after(src,U64.test_bit(Chess.get_pawns(b),U32.to_nat(src)),Chess.get_ep(b),tail,
      Chess.piece_targets(U64.test_bit(Chess.get_pawns(b),U32.to_nat(src)),initialized_table(d,seed,n,key,at,extra),b,src)) :
      Array<U64> & List<&2,Chess.Ply>} = {==}
  pawn_specialized: {Chess.scan_after(src,U64.test_bit(Chess.get_pawns(b),U32.to_nat(src)),Chess.get_ep(b),tail,
      Chess.piece_targets(U64.test_bit(Chess.get_pawns(b),U32.to_nat(src)),initialized_table(d,seed,n,key,at,extra),b,src)) ==
    Chess.scan_after(src,False{},Chess.get_ep(b),tail,
      Chess.piece_targets(False{},initialized_table(d,seed,n,key,at,extra),b,src)) :
      Array<U64> & List<&2,Chess.Ply>} =
    Equal.cong(Bool,Array<U64> & List<&2,Chess.Ply>,
      pawn => Chess.scan_after(src,pawn,Chess.get_ep(b),tail,
        Chess.piece_targets(pawn,initialized_table(d,seed,n,key,at,extra),b,src)),
      U64.test_bit(Chess.get_pawns(b),U32.to_nat(src)),False{},pawn_clear)
  target_specialized: {Chess.scan_after(src,False{},Chess.get_ep(b),tail,
      Chess.piece_targets(False{},initialized_table(d,seed,n,key,at,extra),b,src)) ==
    Chess.scan_after(src,False{},Chess.get_ep(b),tail,
      (initialized_table(d,seed,n,key,at,extra),targets)) :
      Array<U64> & List<&2,Chess.Ply>} =
    Equal.cong(Array<U64> & U64,Array<U64> & List<&2,Chess.Ply>,
      target => Chess.scan_after(src,False{},Chess.get_ep(b),tail,target),
      Chess.piece_targets(False{},initialized_table(d,seed,n,key,at,extra),b,src),
      (initialized_table(d,seed,n,key,at,extra),targets),target_pair)
  step_exact: {Chess.scan_step(b,src,(initialized_table(d,seed,n,key,at,extra),tail)) ==
    Chess.scan_after(src,False{},Chess.get_ep(b),tail,(initialized_table(d,seed,n,key,at,extra),targets)) :
      Array<U64> & List<&2,Chess.Ply>} =
    Equal.trans(Array<U64> & List<&2,Chess.Ply>,
      Chess.scan_step(b,src,(initialized_table(d,seed,n,key,at,extra),tail)),
      Chess.scan_after(src,U64.test_bit(Chess.get_pawns(b),U32.to_nat(src)),Chess.get_ep(b),tail,
        Chess.piece_targets(U64.test_bit(Chess.get_pawns(b),U32.to_nat(src)),initialized_table(d,seed,n,key,at,extra),b,src)),
      Chess.scan_after(src,False{},Chess.get_ep(b),tail,(initialized_table(d,seed,n,key,at,extra),targets)),
      step_expands,Equal.trans(Array<U64> & List<&2,Chess.Ply>,
        Chess.scan_after(src,U64.test_bit(Chess.get_pawns(b),U32.to_nat(src)),Chess.get_ep(b),tail,
          Chess.piece_targets(U64.test_bit(Chess.get_pawns(b),U32.to_nat(src)),initialized_table(d,seed,n,key,at,extra),b,src)),
        Chess.scan_after(src,False{},Chess.get_ep(b),tail,
          Chess.piece_targets(False{},initialized_table(d,seed,n,key,at,extra),b,src)),
        Chess.scan_after(src,False{},Chess.get_ep(b),tail,(initialized_table(d,seed,n,key,at,extra),targets)),
        pawn_specialized,target_specialized))
  Equal.trans(Nat,
    AfterCount.pair_count(Chess.scan_step(b,src,(initialized_table(d,seed,n,key,at,extra),tail)),query),
    AfterCount.pair_count(Chess.scan_after(src,False{},Chess.get_ep(b),tail,
      (initialized_table(d,seed,n,key,at,extra),targets)),query),
    Nat.add(Bool.pick(Nat,Bool.and(Nat.is_lt(U32.to_nat(Q.ply_destination(query)),64n),
      U64.test_bit(targets,U32.to_nat(Q.ply_destination(query)))),
      Factor.query_emission_count(src,False{},Chess.get_ep(b),query),0n),
      Spec.count(tail,query)),
    Equal.cong(Array<U64> & List<&2,Chess.Ply>,Nat,r => AfterCount.pair_count(r,query),
      Chess.scan_step(b,src,(initialized_table(d,seed,n,key,at,extra),tail)),
      Chess.scan_after(src,False{},Chess.get_ep(b),tail,(initialized_table(d,seed,n,key,at,extra),targets)),step_exact),
    Factor.scan_after_closed_count(initialized_table(d,seed,n,key,at,extra),targets,src,False{},Chess.get_ep(b),query,tail))
