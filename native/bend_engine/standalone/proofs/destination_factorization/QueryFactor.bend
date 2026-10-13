# Factor one emission count through its exact U32 destination key.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../bitboard_membership/IndexEquality.bend as U32Eq
import ../bitboard_membership/PrefixDecomposition.bend as Prefix
import ../bitboard_membership/AllEmittedSound.bend as Squares
import ./Spec.bend as S
import ./QueryCount.bend as Q
import ./AfterCount.bend as AfterCount

def key_factor_classified(+src: U32,+pawn: Bool,+ep: U32,+dst: U32,+query: Chess.Ply,
  +qdst: U32,+same_dst: Bool,
  same_as_query: {same_dst == U32.is_eq(dst,qdst) : Bool},
  destination_as_same: {Q.destination_equal(dst,query) == same_dst : Bool}) ->
  {Q.emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query) ==
    Bool.pick(Nat,same_dst,
      Q.fields_emission_count(S.promotes(pawn,qdst),src,qdst,S.flag(pawn,qdst,ep),query),0n) : Nat}:
  match same_dst:
    case True{}:
      +eq_true: {U32.is_eq(dst,qdst) == True{} : Bool} =
        Equal.trans(Bool,U32.is_eq(dst,qdst),same_dst,True{},
          Equal.sym(Bool,same_dst,U32.is_eq(dst,qdst),same_as_query),{==})
      +dst_as_query: {dst == qdst : U32} = U32Eq.u32_eq_true_same(dst,qdst,eq_true)
      +fields_equal: {Q.fields_emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query) ==
        Q.fields_emission_count(S.promotes(pawn,qdst),src,qdst,S.flag(pawn,qdst,ep),query) : Nat} =
        Equal.cong(U32,Nat,x => Q.fields_emission_count(S.promotes(pawn,x),src,x,S.flag(pawn,x,ep),query),
          dst,qdst,dst_as_query)
      +as_pick_same: {Q.emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query) ==
        Bool.pick(Nat,same_dst,Q.fields_emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),0n) : Nat} =
        Equal.trans(Nat,
          Q.emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),
          Bool.pick(Nat,Q.destination_equal(dst,query),
            Q.fields_emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),0n),
          Bool.pick(Nat,same_dst,
            Q.fields_emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),0n),
          Q.emission_count_destination_factor(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),
          Equal.cong(Bool,Nat,b => Bool.pick(Nat,b,
            Q.fields_emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),0n),
            Q.destination_equal(dst,query),same_dst,destination_as_same))
      Equal.trans(Nat,
        Q.emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),
        Q.fields_emission_count(S.promotes(pawn,qdst),src,qdst,S.flag(pawn,qdst,ep),query),
        Bool.pick(Nat,True{},
          Q.fields_emission_count(S.promotes(pawn,qdst),src,qdst,S.flag(pawn,qdst,ep),query),0n),
        Equal.trans(Nat,
          Q.emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),
          Q.fields_emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),
          Q.fields_emission_count(S.promotes(pawn,qdst),src,qdst,S.flag(pawn,qdst,ep),query),
          Equal.trans(Nat,
            Q.emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),
            Bool.pick(Nat,True{},
              Q.fields_emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),0n),
            Q.fields_emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),
            as_pick_same,{==}),fields_equal),{==})
    case False{}:
      Equal.trans(Nat,
        Q.emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),
        Bool.pick(Nat,same_dst,
          Q.fields_emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),0n),
        Bool.pick(Nat,same_dst,
          Q.fields_emission_count(S.promotes(pawn,qdst),src,qdst,S.flag(pawn,qdst,ep),query),0n),
        Equal.trans(Nat,
          Q.emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),
          Bool.pick(Nat,Q.destination_equal(dst,query),
            Q.fields_emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),0n),
          Bool.pick(Nat,same_dst,
            Q.fields_emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),0n),
          Q.emission_count_destination_factor(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),
          Equal.cong(Bool,Nat,b => Bool.pick(Nat,b,
            Q.fields_emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),0n),
            Q.destination_equal(dst,query),same_dst,destination_as_same)),
        {==})

def query_emission_count(+src: U32,+pawn: Bool,+ep: U32,+query: Chess.Ply) -> Nat:
  Q.fields_emission_count(S.promotes(pawn,Q.ply_destination(query)),src,Q.ply_destination(query),
    S.flag(pawn,Q.ply_destination(query),ep),query)

def key_factor(+src: U32,+pawn: Bool,+ep: U32,+dst: U32,+query: Chess.Ply) ->
  {Q.emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query) ==
    Bool.pick(Nat,U32.is_eq(dst,Q.ply_destination(query)),query_emission_count(src,pawn,ep,query),0n) : Nat}:
  key_factor_classified(src,pawn,ep,dst,query,Q.ply_destination(query),
    U32.is_eq(dst,Q.ply_destination(query)),{==},Q.destination_equal_projection(dst,query))

def weighted_keys(keys: List<&2,U32>,+query: U32,+weight: Nat) -> Nat:
  match keys:
    case Nil{}: 0n
    case Con{head,tail}:
      Nat.add(Bool.pick(Nat,U32.is_eq(head,query),weight,0n),weighted_keys(tail,query,weight))

def list_emission_weighted(+keys: List<&2,U32>,+src: U32,+pawn: Bool,+ep: U32,+query: Chess.Ply) ->
  {Q.list_emission_count(keys,src,pawn,ep,query) ==
    weighted_keys(keys,Q.ply_destination(query),query_emission_count(src,pawn,ep,query)) : Nat}:
  match keys:
    case Nil{}: {==}
    case Con{+dst,+rest}:
      +head: {Q.emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query) ==
        Bool.pick(Nat,U32.is_eq(dst,Q.ply_destination(query)),query_emission_count(src,pawn,ep,query),0n) : Nat} =
        key_factor(src,pawn,ep,dst,query)
      +tail: {Q.list_emission_count(rest,src,pawn,ep,query) ==
        weighted_keys(rest,Q.ply_destination(query),query_emission_count(src,pawn,ep,query)) : Nat} =
        list_emission_weighted(rest,src,pawn,ep,query)
      +head_rewrite: {Q.list_emission_count(Con{dst,rest},src,pawn,ep,query) ==
        Nat.add(Bool.pick(Nat,U32.is_eq(dst,Q.ply_destination(query)),
          query_emission_count(src,pawn,ep,query),0n),
          Q.list_emission_count(rest,src,pawn,ep,query)) : Nat} =
        Equal.cong(Nat,Nat,x => Nat.add(x,Q.list_emission_count(rest,src,pawn,ep,query)),
          Q.emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),
          Bool.pick(Nat,U32.is_eq(dst,Q.ply_destination(query)),query_emission_count(src,pawn,ep,query),0n),head)
      +tail_rewrite: {Nat.add(Bool.pick(Nat,U32.is_eq(dst,Q.ply_destination(query)),
        query_emission_count(src,pawn,ep,query),0n),Q.list_emission_count(rest,src,pawn,ep,query)) ==
        Nat.add(Bool.pick(Nat,U32.is_eq(dst,Q.ply_destination(query)),
          query_emission_count(src,pawn,ep,query),0n),
          weighted_keys(rest,Q.ply_destination(query),query_emission_count(src,pawn,ep,query))) : Nat} =
        Equal.cong(Nat,Nat,x => Nat.add(Bool.pick(Nat,U32.is_eq(dst,Q.ply_destination(query)),
          query_emission_count(src,pawn,ep,query),0n),x),
          Q.list_emission_count(rest,src,pawn,ep,query),
          weighted_keys(rest,Q.ply_destination(query),query_emission_count(src,pawn,ep,query)),tail)
      Equal.trans(Nat,Q.list_emission_count(Con{dst,rest},src,pawn,ep,query),
        Nat.add(Bool.pick(Nat,U32.is_eq(dst,Q.ply_destination(query)),
          query_emission_count(src,pawn,ep,query),0n),
          Q.list_emission_count(rest,src,pawn,ep,query)),
        weighted_keys(Con{dst,rest},Q.ply_destination(query),query_emission_count(src,pawn,ep,query)),
        head_rewrite,tail_rewrite)

def scale_count(+n: Nat,+weight: Nat) -> Nat:
  match n:
    case 0n: 0n
    case 1n+rest: Nat.add(weight,scale_count(rest,weight))

def scale_add(+a: Nat,+b: Nat,+weight: Nat) ->
  {scale_count(Nat.add(a,b),weight) ==
    Nat.add(scale_count(a,weight),scale_count(b,weight)) : Nat}:
  match a:
    case 0n: {==}
    case 1n+rest:
      Equal.trans(Nat,
        scale_count(Nat.add(1n+rest,b),weight),
        Nat.add(weight,scale_count(Nat.add(rest,b),weight)),
        Nat.add(Nat.add(weight,scale_count(rest,weight)),scale_count(b,weight)),
        {==},
        Equal.trans(Nat,
          Nat.add(weight,scale_count(Nat.add(rest,b),weight)),
          Nat.add(weight,Nat.add(scale_count(rest,weight),scale_count(b,weight))),
          Nat.add(Nat.add(weight,scale_count(rest,weight)),scale_count(b,weight)),
          Equal.cong(Nat,Nat,x => Nat.add(weight,x),
            scale_count(Nat.add(rest,b),weight),
            Nat.add(scale_count(rest,weight),scale_count(b,weight)),scale_add(rest,b,weight)),
          Prefix.nat_add_assoc_rev(weight,scale_count(rest,weight),scale_count(b,weight))))

def factor_nat_add_zero(+n: Nat) -> {Nat.add(n,0n) == n : Nat}:
  match n:
    case 0n: {==}
    case 1n+rest:
      Equal.cong(Nat,Nat,x => 1n+x,Nat.add(rest,0n),rest,factor_nat_add_zero(rest))

def scale_bool(+hit: Bool,+weight: Nat) ->
  {scale_count(Bool.pick(Nat,hit,1n,0n),weight) ==
    Bool.pick(Nat,hit,weight,0n) : Nat}:
  match hit:
    case True{}: factor_nat_add_zero(weight)
    case False{}: {==}

def weighted_occurrences(+keys: List<&2,U32>,+query: U32,+weight: Nat) ->
  {weighted_keys(keys,query,weight) ==
    scale_count(Prefix.occurrences(keys,query),weight) : Nat}:
  match keys:
    case Nil{}: {==}
    case Con{+head,+tail}:
      +tail_scaled: {weighted_keys(tail,query,weight) ==
        scale_count(Prefix.occurrences(tail,query),weight) : Nat} =
        weighted_occurrences(tail,query,weight)
      +split: {scale_count(Prefix.occurrences(Con{head,tail},query),weight) ==
        Nat.add(scale_count(Bool.pick(Nat,U32.is_eq(head,query),1n,0n),weight),
          scale_count(Prefix.occurrences(tail,query),weight)) : Nat} =
        scale_add(Bool.pick(Nat,U32.is_eq(head,query),1n,0n),
          Prefix.occurrences(tail,query),weight)
      Equal.trans(Nat,
        weighted_keys(Con{head,tail},query,weight),
        Nat.add(Bool.pick(Nat,U32.is_eq(head,query),weight,0n),
          scale_count(Prefix.occurrences(tail,query),weight)),
        scale_count(Prefix.occurrences(Con{head,tail},query),weight),
        Equal.trans(Nat,
          weighted_keys(Con{head,tail},query,weight),
          Nat.add(Bool.pick(Nat,U32.is_eq(head,query),weight,0n),
            weighted_keys(tail,query,weight)),
          Nat.add(Bool.pick(Nat,U32.is_eq(head,query),weight,0n),
            scale_count(Prefix.occurrences(tail,query),weight)),
          {==},
          Equal.cong(Nat,Nat,x => Nat.add(Bool.pick(Nat,U32.is_eq(head,query),weight,0n),x),
            weighted_keys(tail,query,weight),
            scale_count(Prefix.occurrences(tail,query),weight),tail_scaled)),
        Equal.trans(Nat,
          Nat.add(Bool.pick(Nat,U32.is_eq(head,query),weight,0n),
            scale_count(Prefix.occurrences(tail,query),weight)),
          Nat.add(scale_count(Bool.pick(Nat,U32.is_eq(head,query),1n,0n),weight),
            scale_count(Prefix.occurrences(tail,query),weight)),
          scale_count(Prefix.occurrences(Con{head,tail},query),weight),
          Equal.cong(Nat,Nat,x => Nat.add(x,scale_count(Prefix.occurrences(tail,query),weight)),
            Bool.pick(Nat,U32.is_eq(head,query),weight,0n),
            scale_count(Bool.pick(Nat,U32.is_eq(head,query),1n,0n),weight),
            Equal.sym(Nat,
              scale_count(Bool.pick(Nat,U32.is_eq(head,query),1n,0n),weight),
              Bool.pick(Nat,U32.is_eq(head,query),weight,0n),
              scale_bool(U32.is_eq(head,query),weight))),
          Equal.sym(Nat,
            scale_count(Prefix.occurrences(Con{head,tail},query),weight),
            Nat.add(scale_count(Bool.pick(Nat,U32.is_eq(head,query),1n,0n),weight),
              scale_count(Prefix.occurrences(tail,query),weight)),split)))

def range_and_bit_classified(+bb: U64,+query: U32,+in_range: Bool,
  range_as_lt: {in_range == Nat.is_lt(U32.to_nat(query),64n) : Bool}) ->
  {Bool.and(in_range,U64.test_bit(bb,U32.to_nat(query))) ==
    U64.test_bit(bb,U32.to_nat(query)) : Bool}:
  match in_range:
    case True{}: {==}
    case False{}:
      +not_lt: {Nat.is_lt(U32.to_nat(query),64n) == False{} : Bool} =
        Equal.trans(Bool,Nat.is_lt(U32.to_nat(query),64n),in_range,False{},
          Equal.sym(Bool,in_range,Nat.is_lt(U32.to_nat(query),64n),range_as_lt),{==})
      +ge: {Nat.is_le(64n,U32.to_nat(query)) == True{} : Bool} =
        Squares.nat_not_lt_implies_reverse_le(U32.to_nat(query),64n,not_lt)
      +bit_false: {U64.test_bit(bb,U32.to_nat(query)) == False{} : Bool} =
        Squares.u64_test_bit_outside(bb,U32.to_nat(query),ge)
      Equal.trans(Bool,
        Bool.and(in_range,U64.test_bit(bb,U32.to_nat(query))),
        False{},U64.test_bit(bb,U32.to_nat(query)),{==},Equal.sym(Bool,U64.test_bit(bb,U32.to_nat(query)),False{},bit_false))

def range_and_bit(+bb: U64,+query: U32) ->
  {Bool.and(Nat.is_lt(U32.to_nat(query),64n),U64.test_bit(bb,U32.to_nat(query))) ==
    U64.test_bit(bb,U32.to_nat(query)) : Bool}:
  range_and_bit_classified(bb,query,Nat.is_lt(U32.to_nat(query),64n),{==})

def actual_keys_below64(+bb: U64) ->
  {Squares.all_below64(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{})) == True{} : Bool}:
  Squares.actual_bit_squares_below64(bb)

def actual_frequency_with_range(+bb: U64,+query: U32) ->
  {Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),query) ==
    Bool.pick(Nat,Bool.and(Nat.is_lt(U32.to_nat(query),64n),
      U64.test_bit(bb,U32.to_nat(query))),1n,0n) : Nat}:
  Equal.trans(Nat,
    Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),query),
    Bool.pick(Nat,U64.test_bit(bb,U32.to_nat(query)),1n,0n),
    Bool.pick(Nat,Bool.and(Nat.is_lt(U32.to_nat(query),64n),
      U64.test_bit(bb,U32.to_nat(query))),1n,0n),
    Squares.actual_bit_squares_frequency_all_u32(bb,query),
    Equal.cong(Bool,Nat,b => Bool.pick(Nat,b,1n,0n),
      U64.test_bit(bb,U32.to_nat(query)),
      Bool.and(Nat.is_lt(U32.to_nat(query),64n),U64.test_bit(bb,U32.to_nat(query))),
      Equal.sym(Bool,
        Bool.and(Nat.is_lt(U32.to_nat(query),64n),U64.test_bit(bb,U32.to_nat(query))),
        U64.test_bit(bb,U32.to_nat(query)),range_and_bit(bb,query))))

def actual_emission_count(+bb: U64,+src: U32,+pawn: Bool,+ep: U32,+query: Chess.Ply) ->
  {Q.list_emission_count(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),src,pawn,ep,query) ==
    Bool.pick(Nat,Bool.and(Nat.is_lt(U32.to_nat(Q.ply_destination(query)),64n),
      U64.test_bit(bb,U32.to_nat(Q.ply_destination(query)))),
      query_emission_count(src,pawn,ep,query),0n) : Nat}:
  +weighted: {Q.list_emission_count(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),src,pawn,ep,query) ==
    weighted_keys(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),Q.ply_destination(query),
      query_emission_count(src,pawn,ep,query)) : Nat} =
    list_emission_weighted(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),src,pawn,ep,query)
  +scaled: {weighted_keys(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),Q.ply_destination(query),
      query_emission_count(src,pawn,ep,query)) ==
    scale_count(Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),
      Q.ply_destination(query)),query_emission_count(src,pawn,ep,query)) : Nat} =
    weighted_occurrences(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),
      Q.ply_destination(query),query_emission_count(src,pawn,ep,query))
  +frequency: {Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),
      Q.ply_destination(query)) ==
    Bool.pick(Nat,Bool.and(Nat.is_lt(U32.to_nat(Q.ply_destination(query)),64n),
      U64.test_bit(bb,U32.to_nat(Q.ply_destination(query)))),1n,0n) : Nat} =
    actual_frequency_with_range(bb,Q.ply_destination(query))
  +scale_frequency: {scale_count(Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),
      Q.ply_destination(query)),query_emission_count(src,pawn,ep,query)) ==
    scale_count(Bool.pick(Nat,Bool.and(Nat.is_lt(U32.to_nat(Q.ply_destination(query)),64n),
      U64.test_bit(bb,U32.to_nat(Q.ply_destination(query)))),1n,0n),
      query_emission_count(src,pawn,ep,query)) : Nat} =
    Equal.cong(Nat,Nat,n => scale_count(n,query_emission_count(src,pawn,ep,query)),
      Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),Q.ply_destination(query)),
      Bool.pick(Nat,Bool.and(Nat.is_lt(U32.to_nat(Q.ply_destination(query)),64n),
        U64.test_bit(bb,U32.to_nat(Q.ply_destination(query)))),1n,0n),frequency)
  Equal.trans(Nat,
    Q.list_emission_count(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),src,pawn,ep,query),
    scale_count(Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),
      Q.ply_destination(query)),query_emission_count(src,pawn,ep,query)),
    Bool.pick(Nat,Bool.and(Nat.is_lt(U32.to_nat(Q.ply_destination(query)),64n),
      U64.test_bit(bb,U32.to_nat(Q.ply_destination(query)))),
      query_emission_count(src,pawn,ep,query),0n),
    Equal.trans(Nat,
      Q.list_emission_count(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),src,pawn,ep,query),
      weighted_keys(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),Q.ply_destination(query),
        query_emission_count(src,pawn,ep,query)),
      scale_count(Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),
        Q.ply_destination(query)),query_emission_count(src,pawn,ep,query)),weighted,scaled),
    Equal.trans(Nat,
      scale_count(Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),
        Q.ply_destination(query)),query_emission_count(src,pawn,ep,query)),
      scale_count(Bool.pick(Nat,Bool.and(Nat.is_lt(U32.to_nat(Q.ply_destination(query)),64n),
        U64.test_bit(bb,U32.to_nat(Q.ply_destination(query)))),1n,0n),
        query_emission_count(src,pawn,ep,query)),
      Bool.pick(Nat,Bool.and(Nat.is_lt(U32.to_nat(Q.ply_destination(query)),64n),
        U64.test_bit(bb,U32.to_nat(Q.ply_destination(query)))),
        query_emission_count(src,pawn,ep,query),0n),
      scale_frequency,scale_bool(
        Bool.and(Nat.is_lt(U32.to_nat(Q.ply_destination(query)),64n),
          U64.test_bit(bb,U32.to_nat(Q.ply_destination(query)))),
        query_emission_count(src,pawn,ep,query))))

def actual_tally_count(+bb: U64,+src: U32,+pawn: Bool,+ep: U32,+query: Chess.Ply,+total: Nat) ->
  {S.tally(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),src,pawn,ep,query,total) ==
    Nat.add(Bool.pick(Nat,Bool.and(Nat.is_lt(U32.to_nat(Q.ply_destination(query)),64n),
      U64.test_bit(bb,U32.to_nat(Q.ply_destination(query)))),
      query_emission_count(src,pawn,ep,query),0n),total) : Nat}:
  Equal.trans(Nat,
    S.tally(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),src,pawn,ep,query,total),
    Nat.add(Q.list_emission_count(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),src,pawn,ep,query),total),
    Nat.add(Bool.pick(Nat,Bool.and(Nat.is_lt(U32.to_nat(Q.ply_destination(query)),64n),
      U64.test_bit(bb,U32.to_nat(Q.ply_destination(query)))),
      query_emission_count(src,pawn,ep,query),0n),total),
    Q.tally_list_sum(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),src,pawn,ep,query,total),
    Equal.cong(Nat,Nat,n => Nat.add(n,total),
      Q.list_emission_count(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),src,pawn,ep,query),
      Bool.pick(Nat,Bool.and(Nat.is_lt(U32.to_nat(Q.ply_destination(query)),64n),
        U64.test_bit(bb,U32.to_nat(Q.ply_destination(query)))),
        query_emission_count(src,pawn,ep,query),0n),
      actual_emission_count(bb,src,pawn,ep,query)))

def scan_after_closed_count(table: Array<U64>,+targets: U64,+src: U32,+pawn: Bool,+ep: U32,
  +query: Chess.Ply,+tail: List<&2,Chess.Ply>) ->
  {AfterCount.pair_count(Chess.scan_after(src,pawn,ep,tail,(table,targets)),query) ==
    Nat.add(Bool.pick(Nat,Bool.and(Nat.is_lt(U32.to_nat(Q.ply_destination(query)),64n),
      U64.test_bit(targets,U32.to_nat(Q.ply_destination(query)))),
      query_emission_count(src,pawn,ep,query),0n),S.count(tail,query)) : Nat}:
  Equal.trans(Nat,
    AfterCount.pair_count(Chess.scan_after(src,pawn,ep,tail,(table,targets)),query),
    S.tally(Chess.bit_squares(64n,U64.is_zero(targets),targets,Nil{}),src,pawn,ep,query,
      S.count(tail,query)),
    Nat.add(Bool.pick(Nat,Bool.and(Nat.is_lt(U32.to_nat(Q.ply_destination(query)),64n),
      U64.test_bit(targets,U32.to_nat(Q.ply_destination(query)))),
      query_emission_count(src,pawn,ep,query),0n),S.count(tail,query)),
    AfterCount.scan_after_count(table,targets,src,pawn,ep,query,tail),
    actual_tally_count(targets,src,pawn,ep,query,S.count(tail,query)))
