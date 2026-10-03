# First reduce the full-Ply tally to an explicit sum of exact one-square contributions.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../bitboard_membership/PrefixDecomposition.bend as Prefix
import ../bitboard_membership/Cardinality.bend as Card
import ../bitboard_membership/IndexEquality.bend as U32Eq
import ./Spec.bend as S

def pick1(+hit: Bool) -> Nat:
  Bool.pick(Nat,hit,1n,0n)

def bump_pick(+hit: Bool,+n: Nat) -> {S.bump(hit,n) == Nat.add(pick1(hit),n) : Nat}:
  match hit:
    case True{}: {==}
    case False{}: {==}

def promotion_hits(+src: U32,+dst: U32,+query: Chess.Ply) -> List<&2,Bool>:
  Con{S.same(Chess.Ply{src,dst,1,0},query),
    Con{S.same(Chess.Ply{src,dst,2,0},query),
      Con{S.same(Chess.Ply{src,dst,3,0},query),
        Con{S.same(Chess.Ply{src,dst,4,0},query),Nil{}}}}}

def bump_hits(hits: List<&2,Bool>,+total: Nat) -> Nat:
  match hits:
    case Nil{}: total
    case Con{head,tail}: S.bump(head,bump_hits(tail,total))

def sum_hits(hits: List<&2,Bool>) -> Nat:
  match hits:
    case Nil{}: 0n
    case Con{head,tail}: Nat.add(pick1(head),sum_hits(tail))

def bump_hits_sum(+hits: List<&2,Bool>,+total: Nat) ->
  {bump_hits(hits,total) == Nat.add(sum_hits(hits),total) : Nat}:
  match hits:
    case Nil{}: {==}
    case Con{+head,+tail}:
      +rest = bump_hits_sum(tail,total)
      +bumped: {S.bump(head,bump_hits(tail,total)) ==
        Nat.add(pick1(head),Nat.add(sum_hits(tail),total)) : Nat} =
        Equal.trans(Nat,S.bump(head,bump_hits(tail,total)),
          S.bump(head,Nat.add(sum_hits(tail),total)),
          Nat.add(pick1(head),Nat.add(sum_hits(tail),total)),
          Equal.cong(Nat,Nat,n => S.bump(head,n),bump_hits(tail,total),
            Nat.add(sum_hits(tail),total),rest),
          bump_pick(head,Nat.add(sum_hits(tail),total)))
      Equal.trans(Nat,bump_hits(hits,total),
        Nat.add(pick1(head),Nat.add(sum_hits(tail),total)),
        Nat.add(sum_hits(hits),total),bumped,
        Prefix.nat_add_assoc_rev(pick1(head),sum_hits(tail),total))

def emission_count(promote: Bool,+src: U32,+dst: U32,+tag: U32,+query: Chess.Ply) -> Nat:
  match promote:
    case True{}: sum_hits(promotion_hits(src,dst,query))
    case False{}: pick1(S.same(Chess.Ply{src,dst,0,tag},query))

def contribution_sum(+promote: Bool,+src: U32,+dst: U32,+tag: U32,+query: Chess.Ply,+total: Nat) ->
  {S.contribution(promote,src,dst,tag,query,total) ==
    Nat.add(emission_count(promote,src,dst,tag,query),total) : Nat}:
  match promote:
    case False{}: bump_pick(S.same(Chess.Ply{src,dst,0,tag},query),total)
    case True{}:
      bump_hits_sum(promotion_hits(src,dst,query),total)

def list_emission_count(keys: List<&2,U32>,+src: U32,+pawn: Bool,+ep: U32,+query: Chess.Ply) -> Nat:
  match keys:
    case Nil{}: 0n
    case Con{+dst,rest}:
      Nat.add(emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),
        list_emission_count(rest,src,pawn,ep,query))

def tally_list_sum(+keys: List<&2,U32>,+src: U32,+pawn: Bool,+ep: U32,+query: Chess.Ply,+total: Nat) ->
  {S.tally(keys,src,pawn,ep,query,total) ==
    Nat.add(list_emission_count(keys,src,pawn,ep,query),total) : Nat}:
  match keys:
    case Nil{}: {==}
    case Con{+dst,+rest}:
      +rest_sum: {S.tally(rest,src,pawn,ep,query,total) ==
        Nat.add(list_emission_count(rest,src,pawn,ep,query),total) : Nat} =
        tally_list_sum(rest,src,pawn,ep,query,total)
      +head_sum: {S.contribution(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query,
          S.tally(rest,src,pawn,ep,query,total)) ==
        Nat.add(emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),
          S.tally(rest,src,pawn,ep,query,total)) : Nat} =
        contribution_sum(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query,
          S.tally(rest,src,pawn,ep,query,total))
      +rest_lift: {Nat.add(emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),
          S.tally(rest,src,pawn,ep,query,total)) ==
        Nat.add(emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),
          Nat.add(list_emission_count(rest,src,pawn,ep,query),total)) : Nat} =
        Equal.cong(Nat,Nat,n => Nat.add(emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),n),
          S.tally(rest,src,pawn,ep,query,total),
          Nat.add(list_emission_count(rest,src,pawn,ep,query),total),rest_sum)
      Equal.trans(Nat,S.tally(Con{dst,rest},src,pawn,ep,query,total),
        Nat.add(emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),
          Nat.add(list_emission_count(rest,src,pawn,ep,query),total)),
        Nat.add(list_emission_count(Con{dst,rest},src,pawn,ep,query),total),
        Equal.trans(Nat,S.tally(Con{dst,rest},src,pawn,ep,query,total),
          Nat.add(emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),
            S.tally(rest,src,pawn,ep,query,total)),
          Nat.add(emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),
            Nat.add(list_emission_count(rest,src,pawn,ep,query),total)),head_sum,rest_lift),
        Prefix.nat_add_assoc_rev(emission_count(S.promotes(pawn,dst),src,dst,S.flag(pawn,dst,ep),query),
          list_emission_count(rest,src,pawn,ep,query),total))


def and_assoc(+a: Bool,+b: Bool,+c: Bool) ->
  {Bool.and(Bool.and(a,b),c) == Bool.and(a,Bool.and(b,c)) : Bool}:
  match a b c:
    case True{} True{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} False{} True{}: {==}
    case True{} False{} False{}: {==}
    case False{} True{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} False{} False{}: {==}

def destination_equal(+dst: U32,+query: Chess.Ply) -> Bool:
  match query:
    case Chess.Ply{+qs,+qd,+qp,+qflag}: U32.is_eq(dst,qd)

def same_fields_without_destination(+src: U32,+dst: U32,+promotion: U32,+flag: U32,+query: Chess.Ply) -> Bool:
  match query:
    case Chess.Ply{+qs,+qd,+qp,+qflag}:
      Bool.and(U32.is_eq(promotion,qp),
        Bool.and(U32.is_eq(flag,qflag),U32.is_eq(src,qs)))

def same_split_destination(+src: U32,+dst: U32,+promotion: U32,+flag: U32,+query: Chess.Ply) ->
  {S.same(Chess.Ply{src,dst,promotion,flag},query) ==
    Bool.and(same_fields_without_destination(src,dst,promotion,flag,query),
      destination_equal(dst,query)) : Bool}:
  match query:
    case Chess.Ply{+qs,+qd,+qp,+qflag}:
      +p = U32.is_eq(promotion,qp)
      +f = U32.is_eq(flag,qflag)
      +s = U32.is_eq(src,qs)
      +d = U32.is_eq(dst,qd)
      +inner: {Bool.and(f,Bool.and(s,d)) == Bool.and(Bool.and(f,s),d) : Bool} =
        Equal.sym(Bool,Bool.and(Bool.and(f,s),d),Bool.and(f,Bool.and(s,d)),and_assoc(f,s,d))
      +middle: {Bool.and(p,Bool.and(f,Bool.and(s,d))) ==
        Bool.and(p,Bool.and(Bool.and(f,s),d)) : Bool} =
        Equal.cong(Bool,Bool,x => Bool.and(p,x),Bool.and(f,Bool.and(s,d)),
          Bool.and(Bool.and(f,s),d),inner)
      +outer: {Bool.and(p,Bool.and(Bool.and(f,s),d)) ==
        Bool.and(Bool.and(p,Bool.and(f,s)),d) : Bool} =
        Equal.sym(Bool,Bool.and(Bool.and(p,Bool.and(f,s)),d),
          Bool.and(p,Bool.and(Bool.and(f,s),d)),and_assoc(p,Bool.and(f,s),d))
      Equal.trans(Bool,S.same(Chess.Ply{src,dst,promotion,flag},query),
        Bool.and(p,Bool.and(Bool.and(f,s),d)),
        Bool.and(Bool.and(p,Bool.and(f,s)),d),
        Equal.trans(Bool,S.same(Chess.Ply{src,dst,promotion,flag},query),
          Bool.and(p,Bool.and(f,Bool.and(s,d))),
          Bool.and(p,Bool.and(Bool.and(f,s),d)),{==},middle),outer)

def and_true_right(+hit: Bool) -> {Bool.and(hit,True{}) == hit : Bool}:
  match hit:
    case True{}: {==}
    case False{}: {==}

def and_false_right(+hit: Bool) -> {Bool.and(hit,False{}) == False{} : Bool}:
  match hit:
    case True{}: {==}
    case False{}: {==}

def pick_and_factor(+hit: Bool,+destination: Bool) ->
  {pick1(Bool.and(hit,destination)) ==
    Bool.pick(Nat,destination,pick1(hit),0n) : Nat}:
  match destination:
    case True{}:
      Equal.cong(Bool,Nat,b => pick1(b),Bool.and(hit,True{}),hit,and_true_right(hit))
    case False{}:
      Equal.trans(Nat,pick1(Bool.and(hit,False{})),pick1(False{}),0n,
        Equal.cong(Bool,Nat,b => pick1(b),Bool.and(hit,False{}),False{},and_false_right(hit)),{==})

def same_pick_destination(+src: U32,+dst: U32,+promotion: U32,+flag: U32,+query: Chess.Ply) ->
  {pick1(S.same(Chess.Ply{src,dst,promotion,flag},query)) ==
    Bool.pick(Nat,destination_equal(dst,query),
      pick1(same_fields_without_destination(src,dst,promotion,flag,query)),0n) : Nat}:
  +split = same_split_destination(src,dst,promotion,flag,query)
  Equal.trans(Nat,pick1(S.same(Chess.Ply{src,dst,promotion,flag},query)),
    pick1(Bool.and(same_fields_without_destination(src,dst,promotion,flag,query),
      destination_equal(dst,query))),
    Bool.pick(Nat,destination_equal(dst,query),
      pick1(same_fields_without_destination(src,dst,promotion,flag,query)),0n),
    Equal.cong(Bool,Nat,b => pick1(b),S.same(Chess.Ply{src,dst,promotion,flag},query),
      Bool.and(same_fields_without_destination(src,dst,promotion,flag,query),
        destination_equal(dst,query)),split),
    pick_and_factor(same_fields_without_destination(src,dst,promotion,flag,query),destination_equal(dst,query)))

def promotion_fields(+src: U32,+dst: U32,+query: Chess.Ply) -> List<&2,Bool>:
  Con{same_fields_without_destination(src,dst,1,0,query),
    Con{same_fields_without_destination(src,dst,2,0,query),
      Con{same_fields_without_destination(src,dst,3,0,query),
        Con{same_fields_without_destination(src,dst,4,0,query),Nil{}}}}}

def factored_promotion_hits(+src: U32,+dst: U32,+query: Chess.Ply) ->
  {promotion_hits(src,dst,query) ==
    Con{Bool.and(same_fields_without_destination(src,dst,1,0,query),destination_equal(dst,query)),
      Con{Bool.and(same_fields_without_destination(src,dst,2,0,query),destination_equal(dst,query)),
        Con{Bool.and(same_fields_without_destination(src,dst,3,0,query),destination_equal(dst,query)),
          Con{Bool.and(same_fields_without_destination(src,dst,4,0,query),destination_equal(dst,query)),Nil{}}}}} :
    List<&2,Bool>}:
  +e4 = same_split_destination(src,dst,4,0,query)
  +t4: {Con{S.same(Chess.Ply{src,dst,4,0},query),Nil{}} ==
    Con{Bool.and(same_fields_without_destination(src,dst,4,0,query),destination_equal(dst,query)),Nil{}} : List<&2,Bool>} =
    Equal.cong(Bool,List<&2,Bool>,h => Con{h,Nil{}},
      S.same(Chess.Ply{src,dst,4,0},query),
      Bool.and(same_fields_without_destination(src,dst,4,0,query),destination_equal(dst,query)),e4)
  +e3 = same_split_destination(src,dst,3,0,query)
  +t3: {Con{S.same(Chess.Ply{src,dst,3,0},query),
      Con{S.same(Chess.Ply{src,dst,4,0},query),Nil{}}} ==
    Con{Bool.and(same_fields_without_destination(src,dst,3,0,query),destination_equal(dst,query)),
      Con{Bool.and(same_fields_without_destination(src,dst,4,0,query),destination_equal(dst,query)),Nil{}}} : List<&2,Bool>} =
    Equal.trans(List<&2,Bool>,
      Con{S.same(Chess.Ply{src,dst,3,0},query),Con{S.same(Chess.Ply{src,dst,4,0},query),Nil{}}},
      Con{S.same(Chess.Ply{src,dst,3,0},query),
        Con{Bool.and(same_fields_without_destination(src,dst,4,0,query),destination_equal(dst,query)),Nil{}}},
      Con{Bool.and(same_fields_without_destination(src,dst,3,0,query),destination_equal(dst,query)),
        Con{Bool.and(same_fields_without_destination(src,dst,4,0,query),destination_equal(dst,query)),Nil{}}},
      Equal.cong(List<&2,Bool>,List<&2,Bool>,xs => Con{S.same(Chess.Ply{src,dst,3,0},query),xs},
        Con{S.same(Chess.Ply{src,dst,4,0},query),Nil{}},
        Con{Bool.and(same_fields_without_destination(src,dst,4,0,query),destination_equal(dst,query)),Nil{}},t4),
      Equal.cong(Bool,List<&2,Bool>,h => Con{h,
        Con{Bool.and(same_fields_without_destination(src,dst,4,0,query),destination_equal(dst,query)),Nil{}}},
        S.same(Chess.Ply{src,dst,3,0},query),
        Bool.and(same_fields_without_destination(src,dst,3,0,query),destination_equal(dst,query)),e3))
  +e2 = same_split_destination(src,dst,2,0,query)
  +t2: {Con{S.same(Chess.Ply{src,dst,2,0},query),
      Con{S.same(Chess.Ply{src,dst,3,0},query),Con{S.same(Chess.Ply{src,dst,4,0},query),Nil{}}}} ==
    Con{Bool.and(same_fields_without_destination(src,dst,2,0,query),destination_equal(dst,query)),
      Con{Bool.and(same_fields_without_destination(src,dst,3,0,query),destination_equal(dst,query)),
        Con{Bool.and(same_fields_without_destination(src,dst,4,0,query),destination_equal(dst,query)),Nil{}}}} : List<&2,Bool>} =
    Equal.trans(List<&2,Bool>,
      Con{S.same(Chess.Ply{src,dst,2,0},query),
        Con{S.same(Chess.Ply{src,dst,3,0},query),Con{S.same(Chess.Ply{src,dst,4,0},query),Nil{}}}},
      Con{S.same(Chess.Ply{src,dst,2,0},query),
        Con{Bool.and(same_fields_without_destination(src,dst,3,0,query),destination_equal(dst,query)),
          Con{Bool.and(same_fields_without_destination(src,dst,4,0,query),destination_equal(dst,query)),Nil{}}}},
      Con{Bool.and(same_fields_without_destination(src,dst,2,0,query),destination_equal(dst,query)),
        Con{Bool.and(same_fields_without_destination(src,dst,3,0,query),destination_equal(dst,query)),
          Con{Bool.and(same_fields_without_destination(src,dst,4,0,query),destination_equal(dst,query)),Nil{}}}},
      Equal.cong(List<&2,Bool>,List<&2,Bool>,xs => Con{S.same(Chess.Ply{src,dst,2,0},query),xs},
        Con{S.same(Chess.Ply{src,dst,3,0},query),Con{S.same(Chess.Ply{src,dst,4,0},query),Nil{}}},
        Con{Bool.and(same_fields_without_destination(src,dst,3,0,query),destination_equal(dst,query)),
          Con{Bool.and(same_fields_without_destination(src,dst,4,0,query),destination_equal(dst,query)),Nil{}}},t3),
      Equal.cong(Bool,List<&2,Bool>,h => Con{h,
        Con{Bool.and(same_fields_without_destination(src,dst,3,0,query),destination_equal(dst,query)),
          Con{Bool.and(same_fields_without_destination(src,dst,4,0,query),destination_equal(dst,query)),Nil{}}}},
        S.same(Chess.Ply{src,dst,2,0},query),
        Bool.and(same_fields_without_destination(src,dst,2,0,query),destination_equal(dst,query)),e2))
  +e1 = same_split_destination(src,dst,1,0,query)
  Equal.trans(List<&2,Bool>,
    promotion_hits(src,dst,query),
    Con{S.same(Chess.Ply{src,dst,1,0},query),
      Con{Bool.and(same_fields_without_destination(src,dst,2,0,query),destination_equal(dst,query)),
        Con{Bool.and(same_fields_without_destination(src,dst,3,0,query),destination_equal(dst,query)),
          Con{Bool.and(same_fields_without_destination(src,dst,4,0,query),destination_equal(dst,query)),Nil{}}}}},
    Con{Bool.and(same_fields_without_destination(src,dst,1,0,query),destination_equal(dst,query)),
      Con{Bool.and(same_fields_without_destination(src,dst,2,0,query),destination_equal(dst,query)),
        Con{Bool.and(same_fields_without_destination(src,dst,3,0,query),destination_equal(dst,query)),
          Con{Bool.and(same_fields_without_destination(src,dst,4,0,query),destination_equal(dst,query)),Nil{}}}}},
    Equal.cong(List<&2,Bool>,List<&2,Bool>,xs => Con{S.same(Chess.Ply{src,dst,1,0},query),xs},
      Con{S.same(Chess.Ply{src,dst,2,0},query),
        Con{S.same(Chess.Ply{src,dst,3,0},query),Con{S.same(Chess.Ply{src,dst,4,0},query),Nil{}}}},
      Con{Bool.and(same_fields_without_destination(src,dst,2,0,query),destination_equal(dst,query)),
        Con{Bool.and(same_fields_without_destination(src,dst,3,0,query),destination_equal(dst,query)),
          Con{Bool.and(same_fields_without_destination(src,dst,4,0,query),destination_equal(dst,query)),Nil{}}}},t2),
    Equal.cong(Bool,List<&2,Bool>,h => Con{h,
      Con{Bool.and(same_fields_without_destination(src,dst,2,0,query),destination_equal(dst,query)),
        Con{Bool.and(same_fields_without_destination(src,dst,3,0,query),destination_equal(dst,query)),
          Con{Bool.and(same_fields_without_destination(src,dst,4,0,query),destination_equal(dst,query)),Nil{}}}}},
      S.same(Chess.Ply{src,dst,1,0},query),
      Bool.and(same_fields_without_destination(src,dst,1,0,query),destination_equal(dst,query)),e1))

def gate_hits(+hits: List<&2,Bool>,+destination: Bool) -> List<&2,Bool>:
  match hits:
    case Nil{}: Nil{}
    case Con{+head,+tail}: Con{Bool.and(head,destination),gate_hits(tail,destination)}

def gate_hits_true(+hits: List<&2,Bool>) ->
  {gate_hits(hits,True{}) == hits : List<&2,Bool>}:
  match hits:
    case Nil{}: {==}
    case Con{+head,+tail}:
      +head_eq = and_true_right(head)
      +tail_eq = gate_hits_true(tail)
      +tail_lift: {Con{Bool.and(head,True{}),gate_hits(tail,True{})} ==
        Con{Bool.and(head,True{}),tail} : List<&2,Bool>} =
        Equal.cong(List<&2,Bool>,List<&2,Bool>,xs => Con{Bool.and(head,True{}),xs},
          gate_hits(tail,True{}),tail,tail_eq)
      +head_lift: {Con{Bool.and(head,True{}),tail} == Con{head,tail} : List<&2,Bool>} =
        Equal.cong(Bool,List<&2,Bool>,h => Con{h,tail},Bool.and(head,True{}),head,head_eq)
      Equal.trans(List<&2,Bool>,gate_hits(Con{head,tail},True{}),
        Con{Bool.and(head,True{}),tail},Con{head,tail},
        Equal.trans(List<&2,Bool>,gate_hits(Con{head,tail},True{}),
          Con{Bool.and(head,True{}),gate_hits(tail,True{})},
          Con{Bool.and(head,True{}),tail},{==},tail_lift),head_lift)

def gate_hits_false_sum(+hits: List<&2,Bool>) ->
  {sum_hits(gate_hits(hits,False{})) == 0n : Nat}:
  match hits:
    case Nil{}: {==}
    case Con{+head,+tail}:
      +head_zero: {pick1(Bool.and(head,False{})) == 0n : Nat} =
        Equal.cong(Bool,Nat,b => pick1(b),Bool.and(head,False{}),False{},and_false_right(head))
      +tail_sum = gate_hits_false_sum(tail)
      +drop_head: {Nat.add(pick1(Bool.and(head,False{})),
          sum_hits(gate_hits(tail,False{}))) ==
        Nat.add(0n,sum_hits(gate_hits(tail,False{}))) : Nat} =
        Equal.cong(Nat,Nat,n => Nat.add(n,sum_hits(gate_hits(tail,False{}))),
          pick1(Bool.and(head,False{})),0n,head_zero)
      Equal.trans(Nat,sum_hits(gate_hits(Con{head,tail},False{})),
        Nat.add(0n,sum_hits(gate_hits(tail,False{}))),0n,
        Equal.trans(Nat,sum_hits(gate_hits(Con{head,tail},False{})),
          Nat.add(pick1(Bool.and(head,False{})),sum_hits(gate_hits(tail,False{}))),
          Nat.add(0n,sum_hits(gate_hits(tail,False{}))),{==},drop_head),
        Equal.trans(Nat,Nat.add(0n,sum_hits(gate_hits(tail,False{}))),
          sum_hits(gate_hits(tail,False{})),0n,
          {==},Equal.cong(Nat,Nat,n => Nat.add(0n,n),
            sum_hits(gate_hits(tail,False{})),0n,tail_sum)))

def gate_hits_sum(+hits: List<&2,Bool>,+destination: Bool) ->
  {sum_hits(gate_hits(hits,destination)) ==
    Bool.pick(Nat,destination,sum_hits(hits),0n) : Nat}:
  match destination:
    case True{}:
      Equal.cong(List<&2,Bool>,Nat,xs => sum_hits(xs),gate_hits(hits,True{}),hits,gate_hits_true(hits))
    case False{}: gate_hits_false_sum(hits)

def fields_emission_count(promote: Bool,+src: U32,+dst: U32,+tag: U32,+query: Chess.Ply) -> Nat:
  match promote:
    case True{}: sum_hits(promotion_fields(src,dst,query))
    case False{}: pick1(same_fields_without_destination(src,dst,0,tag,query))

def emission_count_destination_factor(+promote: Bool,+src: U32,+dst: U32,+tag: U32,+query: Chess.Ply) ->
  {emission_count(promote,src,dst,tag,query) ==
    Bool.pick(Nat,destination_equal(dst,query),
      fields_emission_count(promote,src,dst,tag,query),0n) : Nat}:
  match promote:
    case False{}: same_pick_destination(src,dst,0,tag,query)
    case True{}:
      +factored = factored_promotion_hits(src,dst,query)
      +as_gates: {sum_hits(promotion_hits(src,dst,query)) ==
        sum_hits(gate_hits(promotion_fields(src,dst,query),destination_equal(dst,query))) : Nat} =
        Equal.cong(List<&2,Bool>,Nat,xs => sum_hits(xs),
          promotion_hits(src,dst,query),
          gate_hits(promotion_fields(src,dst,query),destination_equal(dst,query)),factored)
      +gates_sum: {sum_hits(gate_hits(promotion_fields(src,dst,query),destination_equal(dst,query))) ==
        Bool.pick(Nat,destination_equal(dst,query),sum_hits(promotion_fields(src,dst,query)),0n) : Nat} =
        gate_hits_sum(promotion_fields(src,dst,query),destination_equal(dst,query))
      Equal.trans(Nat,
        emission_count(True{},src,dst,tag,query),
        sum_hits(gate_hits(promotion_fields(src,dst,query),destination_equal(dst,query))),
        Bool.pick(Nat,destination_equal(dst,query),fields_emission_count(True{},src,dst,tag,query),0n),
        Equal.trans(Nat,
          emission_count(True{},src,dst,tag,query),
          sum_hits(promotion_hits(src,dst,query)),
          sum_hits(gate_hits(promotion_fields(src,dst,query),destination_equal(dst,query))),
          {==},as_gates),
        Equal.trans(Nat,
          sum_hits(gate_hits(promotion_fields(src,dst,query),destination_equal(dst,query))),
          Bool.pick(Nat,destination_equal(dst,query),sum_hits(promotion_fields(src,dst,query)),0n),
          Bool.pick(Nat,destination_equal(dst,query),fields_emission_count(True{},src,dst,tag,query),0n),
          gates_sum,{==}))

def ply_destination(+query: Chess.Ply) -> U32:
  match query:
    case Chess.Ply{+qs,+qd,+qp,+qflag}: qd

def destination_equal_projection(+dst: U32,+query: Chess.Ply) ->
  {destination_equal(dst,query) == U32.is_eq(dst,ply_destination(query)) : Bool}:
  match query:
    case Chess.Ply{+qs,+qd,+qp,+qflag}: {==}
