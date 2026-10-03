# Exact accumulator-aware prefix, list cardinality, and occurrence decomposition
# for actual Chess.bit_squares. This retains list order and all duplicate entries.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ./BitSquaresStep.bend as PriorStep
import ./ActualClearBit.bend as ActualClear
import ../bitboard_ctz_structural/WordCtzCore.bend as Ctz

def append(+xs: List<&2,U32>,+ys: List<&2,U32>) -> List<&2,U32>:
  match xs:
    case Nil{}: ys
    case Con{head,tail}: Con{head,append(tail,ys)}

def emitted_prefix(+n: Nat,+empty: Bool,+bb: U64) -> List<&2,U32>:
  match n:
    case 0n: Nil{}
    case 1n+p:
      match empty:
        case True{}: Nil{}
        case False{}:
          +rest = U64.clear_lsb(bb)
          append(emitted_prefix(p,U64.is_zero(rest),rest),U64.ctz(bb) <> Nil{})

def append_cons_right(+xs: List<&2,U32>,+head: U32,+tail: List<&2,U32>) ->
  {append(xs,Con{head,tail}) == append(append(xs,head <> Nil{}),tail) : List<&2,U32>}:
  match xs:
    case Nil{}: {==}
    case Con{item,rest}:
      Equal.cong(List<&2,U32>,List<&2,U32>,
        ys => Con{item,ys},
        append(rest,Con{head,tail}),
        append(append(rest,head <> Nil{}),tail),
        append_cons_right(rest,head,tail))

def bit_squares_as_prefix(+n: Nat,+empty: Bool,+bb: U64,+acc: List<&2,U32>) ->
  {Chess.bit_squares(n,empty,bb,acc) == append(emitted_prefix(n,empty,bb),acc) : List<&2,U32>}:
  match n:
    case 0n: {==}
    case 1n+p:
      match empty:
        case True{}: {==}
        case False{}:
          +rest = U64.clear_lsb(bb)
          Equal.trans(List<&2,U32>,
            Chess.bit_squares(1n+p,False{},bb,acc),
            append(emitted_prefix(p,U64.is_zero(rest),rest),U64.ctz(bb) <> acc),
            append(emitted_prefix(1n+p,False{},bb),acc),
            bit_squares_as_prefix(p,U64.is_zero(rest),rest,U64.ctz(bb) <> acc),
            append_cons_right(emitted_prefix(p,U64.is_zero(rest),rest),U64.ctz(bb),acc))

def list_length(+xs: List<&2,U32>) -> Nat:
  match xs:
    case Nil{}: 0n
    case Con{head,tail}: Nat.add(1n,list_length(tail))

def list_length_append(+xs: List<&2,U32>,+ys: List<&2,U32>) ->
  {list_length(append(xs,ys)) == Nat.add(list_length(xs),list_length(ys)) : Nat}:
  match xs:
    case Nil{}: {==}
    case Con{head,tail}:
      Equal.cong(Nat,Nat,k => Nat.add(1n,k),
        list_length(append(tail,ys)),
        Nat.add(list_length(tail),list_length(ys)),
        list_length_append(tail,ys))

def bit_squares_cardinality_acc(+n: Nat,+empty: Bool,+bb: U64,+acc: List<&2,U32>) ->
  {list_length(Chess.bit_squares(n,empty,bb,acc)) ==
    Nat.add(list_length(emitted_prefix(n,empty,bb)),list_length(acc)) : Nat}:
  Equal.trans(Nat,
    list_length(Chess.bit_squares(n,empty,bb,acc)),
    list_length(append(emitted_prefix(n,empty,bb),acc)),
    Nat.add(list_length(emitted_prefix(n,empty,bb)),list_length(acc)),
    Equal.cong(List<&2,U32>,Nat,xs => list_length(xs),
      Chess.bit_squares(n,empty,bb,acc),append(emitted_prefix(n,empty,bb),acc),
      bit_squares_as_prefix(n,empty,bb,acc)),
    list_length_append(emitted_prefix(n,empty,bb),acc))

def occurrences(+xs: List<&2,U32>,+query: U32) -> Nat:
  match xs:
    case Nil{}: 0n
    case Con{head,tail}:
      Nat.add(Bool.pick(Nat,U32.is_eq(head,query),1n,0n),occurrences(tail,query))

def nat_add_assoc(+a: Nat,+b: Nat,+c: Nat) ->
  {Nat.add(Nat.add(a,b),c) == Nat.add(a,Nat.add(b,c)) : Nat}:
  match a:
    case 0n: {==}
    case 1n+p:
      Equal.cong(Nat,Nat,k => 1n+k,
        Nat.add(Nat.add(p,b),c),Nat.add(p,Nat.add(b,c)),
        nat_add_assoc(p,b,c))

def nat_add_assoc_rev(+a: Nat,+b: Nat,+c: Nat) ->
  {Nat.add(a,Nat.add(b,c)) == Nat.add(Nat.add(a,b),c) : Nat}:
  Equal.sym(Nat,Nat.add(Nat.add(a,b),c),Nat.add(a,Nat.add(b,c)),nat_add_assoc(a,b,c))

def occurrences_append(+xs: List<&2,U32>,+ys: List<&2,U32>,+query: U32) ->
  {occurrences(append(xs,ys),query) ==
    Nat.add(occurrences(xs,query),occurrences(ys,query)) : Nat}:
  match xs:
    case Nil{}: {==}
    case Con{head,tail}:
      Equal.trans(Nat,
        occurrences(append(xs,ys),query),
        Nat.add(Bool.pick(Nat,U32.is_eq(head,query),1n,0n),
          occurrences(append(tail,ys),query)),
        Nat.add(occurrences(xs,query),occurrences(ys,query)),
        {==},
        Equal.trans(Nat,
          Nat.add(Bool.pick(Nat,U32.is_eq(head,query),1n,0n),
            occurrences(append(tail,ys),query)),
          Nat.add(Bool.pick(Nat,U32.is_eq(head,query),1n,0n),
            Nat.add(occurrences(tail,query),occurrences(ys,query))),
          Nat.add(occurrences(xs,query),occurrences(ys,query)),
          Equal.cong(Nat,Nat,k => Nat.add(Bool.pick(Nat,U32.is_eq(head,query),1n,0n),k),
            occurrences(append(tail,ys),query),
            Nat.add(occurrences(tail,query),occurrences(ys,query)),
            occurrences_append(tail,ys,query)),
          nat_add_assoc_rev(Bool.pick(Nat,U32.is_eq(head,query),1n,0n),
            occurrences(tail,query),occurrences(ys,query))))

def bit_squares_occurrences_acc(+n: Nat,+empty: Bool,+bb: U64,+acc: List<&2,U32>,+query: U32) ->
  {occurrences(Chess.bit_squares(n,empty,bb,acc),query) ==
    Nat.add(occurrences(emitted_prefix(n,empty,bb),query),occurrences(acc,query)) : Nat}:
  Equal.trans(Nat,
    occurrences(Chess.bit_squares(n,empty,bb,acc),query),
    occurrences(append(emitted_prefix(n,empty,bb),acc),query),
    Nat.add(occurrences(emitted_prefix(n,empty,bb),query),occurrences(acc,query)),
    Equal.cong(List<&2,U32>,Nat,xs => occurrences(xs,query),
      Chess.bit_squares(n,empty,bb,acc),append(emitted_prefix(n,empty,bb),acc),
      bit_squares_as_prefix(n,empty,bb,acc)),
    occurrences_append(emitted_prefix(n,empty,bb),acc,query))

def and_false_right(+b: Bool) ->
  {Bool.and(b,False{}) == False{} : Bool}:
  match b:
    case False{}: {==}
    case True{}: {==}

def not_nat_eq_refl(+n: Nat) ->
  {Bool.not(Nat.is_eq(n,n)) == False{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p: not_nat_eq_refl(p)

def first_step_selected_key_is_live_bit(+bb: U64,+e: {U64.is_zero(bb) == False{} : Bool}) ->
  {U64.test_bit(bb,U32.to_nat(U64.ctz(bb))) == True{} : Bool}:
  PriorStep.bit_squares_step_head_is_set(bb,e)

def first_step_selected_key_is_removed(+bb: U64,+e: {U64.is_zero(bb) == False{} : Bool}) ->
  {U64.test_bit(U64.clear_lsb(bb),U32.to_nat(U64.ctz(bb))) == False{} : Bool}:
  Equal.trans(Bool,
    U64.test_bit(U64.clear_lsb(bb),U32.to_nat(U64.ctz(bb))),
    Bool.and(U64.test_bit(bb,U32.to_nat(U64.ctz(bb))),
      Bool.not(Nat.is_eq(U32.to_nat(U64.ctz(bb)),U32.to_nat(U64.ctz(bb))))),
    False{},
    ActualClear.actual_clear_lsb_bit(bb,U32.to_nat(U64.ctz(bb)),Ctz.actual_ctz_range(bb,e),e),
    Equal.trans(Bool,
      Bool.and(U64.test_bit(bb,U32.to_nat(U64.ctz(bb))),
        Bool.not(Nat.is_eq(U32.to_nat(U64.ctz(bb)),U32.to_nat(U64.ctz(bb))))),
      Bool.and(U64.test_bit(bb,U32.to_nat(U64.ctz(bb))),False{}),
      False{},
      Equal.cong(Bool,Bool,b => Bool.and(U64.test_bit(bb,U32.to_nat(U64.ctz(bb))),b),
        Bool.not(Nat.is_eq(U32.to_nat(U64.ctz(bb)),U32.to_nat(U64.ctz(bb)))),
        False{},not_nat_eq_refl(U32.to_nat(U64.ctz(bb)))),
      and_false_right(U64.test_bit(bb,U32.to_nat(U64.ctz(bb))))))
