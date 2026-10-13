# Exact emitted list cardinality from the structural Word count budget.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../u64/Words.bend as Old
import ../fill_interval/Naturals.bend as NatProof
import ../bitboard_inventory/ClearStep.bend as ClearStep
import ../bitboard_zero_bridge/Bridge.bend as ZeroBridge
import ./RemainingBudget.bend as Budget
import ./PrefixDecomposition.bend as Prefix

def nat_add_zero_right(+n: Nat) -> {Nat.add(n,0n) == n : Nat}:
  match n:
    case 0n: {==}
    case 1n+p:
      Equal.cong(Nat,Nat,n => Nat.add(1n,n),Nat.add(p,0n),p,nat_add_zero_right(p))

def nat_add_right_succ(+a: Nat,+b: Nat) ->
  {Nat.add(a,Nat.add(1n,b)) == Nat.add(1n,Nat.add(a,b)) : Nat}:
  match a:
    case 0n: {==}
    case 1n+p:
      Equal.cong(Nat,Nat,n => Nat.add(1n,n),Nat.add(p,Nat.add(1n,b)),Nat.add(1n,Nat.add(p,b)),
        nat_add_right_succ(p,b))

def nat_add_comm(+a: Nat,+b: Nat) -> {Nat.add(a,b) == Nat.add(b,a) : Nat}:
  match a:
    case 0n:
      Equal.sym(Nat,Nat.add(b,0n),b,nat_add_zero_right(b))
    case 1n+p:
      Equal.trans(Nat,Nat.add(1n+p,b),Nat.add(1n,Nat.add(b,p)),Nat.add(b,1n+p),
        Equal.trans(Nat,Nat.add(1n+p,b),Nat.add(1n,Nat.add(p,b)),Nat.add(1n,Nat.add(b,p)),
          {==},
          Equal.cong(Nat,Nat,n => Nat.add(1n,n),Nat.add(p,b),Nat.add(b,p),nat_add_comm(p,b))),
        Equal.sym(Nat,Nat.add(b,Nat.add(1n,p)),Nat.add(1n,Nat.add(b,p)),nat_add_right_succ(b,p)))

def lt_succ_one_false(+p: Nat) -> {Nat.is_lt(1n+p,1n) == False{} : Bool}:
  match p:
    case 0n: {==}
    case 1n+q: {==}

def count_lt_one_zero(+c: Nat,+e: {Nat.is_lt(c,1n) == True{} : Bool}) -> {c == 0n : Nat}:
  match c:
    case 0n: {==}
    case 1n+p:
      Old.false_true({1n+p == 0n : Nat},Equal.trans(Bool,False{},Nat.is_lt(1n+p,1n),True{},
        Equal.sym(Bool,Nat.is_lt(1n+p,1n),False{},lt_succ_one_false(p)),e))

def zero_word_count(+n: Nat,+w: Word(n),+e: {ClearStep.is_zero_word(n,w) == True{} : Bool}) ->
  {Word.count(n,w) == 0n : Nat}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w:
        case WCon{False{},tail}:
          Equal.trans(Nat,Word.count(1n+p,w),Word.count(p,tail),0n,{==},
            zero_word_count(p,tail,e))
        case WCon{True{},tail}: Old.false_true({Word.count(1n+p,w) == 0n : Nat},e)

def u64_zero_count(+a: U64,+e: {U64.is_zero(a) == True{} : Bool}) ->
  {Word.count(64n,U64.to_word(a)) == 0n : Nat}:
  zero_word_count(64n,U64.to_word(a),
    Equal.trans(Bool,ClearStep.is_zero_word(64n,U64.to_word(a)),U64.is_zero(a),True{},
      Equal.sym(Bool,U64.is_zero(a),ClearStep.is_zero_word(64n,U64.to_word(a)),ZeroBridge.u64_is_zero_word(a)),e))

def emitted_prefix_count_nonzero(+p: Nat,+bb: U64,
  +budget: {Nat.is_lt(Word.count(64n,U64.to_word(bb)),Nat.add(1n,Nat.add(1n,p))) == True{} : Bool},
  +e: {U64.is_zero(bb) == False{} : Bool},
  +ih: {Prefix.list_length(Prefix.emitted_prefix(p,U64.is_zero(U64.clear_lsb(bb)),U64.clear_lsb(bb))) ==
    Word.count(64n,U64.to_word(U64.clear_lsb(bb))) : Nat}) ->
  {Prefix.list_length(Prefix.emitted_prefix(1n+p,False{},bb)) == Word.count(64n,U64.to_word(bb)) : Nat}:
  +rest = U64.clear_lsb(bb)
  +next_budget: {Nat.is_lt(Word.count(64n,U64.to_word(rest)),Nat.add(1n,p)) == True{} : Bool} =
    Budget.clear_count_budget(bb,1n+p,budget,e)
  Equal.trans(Nat,
    Prefix.list_length(Prefix.emitted_prefix(1n+p,False{},bb)),
    Prefix.list_length(Prefix.append(Prefix.emitted_prefix(p,U64.is_zero(rest),rest),U64.ctz(bb) <> Nil{})),
    Word.count(64n,U64.to_word(bb)),
    {==},
    Equal.trans(Nat,
      Prefix.list_length(Prefix.append(Prefix.emitted_prefix(p,U64.is_zero(rest),rest),U64.ctz(bb) <> Nil{})),
      Nat.add(Prefix.list_length(Prefix.emitted_prefix(p,U64.is_zero(rest),rest)),
        Prefix.list_length(U64.ctz(bb) <> Nil{})),
      Word.count(64n,U64.to_word(bb)),
      Prefix.list_length_append(Prefix.emitted_prefix(p,U64.is_zero(rest),rest),U64.ctz(bb) <> Nil{}),
      Equal.trans(Nat,
        Nat.add(Prefix.list_length(Prefix.emitted_prefix(p,U64.is_zero(rest),rest)),
          Prefix.list_length(U64.ctz(bb) <> Nil{})),
        Nat.add(Prefix.list_length(Prefix.emitted_prefix(p,U64.is_zero(rest),rest)),1n),
        Word.count(64n,U64.to_word(bb)),
        Equal.cong(Nat,Nat,n => Nat.add(Prefix.list_length(Prefix.emitted_prefix(p,U64.is_zero(rest),rest)),n),
          Prefix.list_length(U64.ctz(bb) <> Nil{}),1n,{==}),
        Equal.trans(Nat,
          Nat.add(Prefix.list_length(Prefix.emitted_prefix(p,U64.is_zero(rest),rest)),1n),
          Nat.add(1n,Word.count(64n,U64.to_word(rest))),
          Word.count(64n,U64.to_word(bb)),
          Equal.trans(Nat,
            Nat.add(Prefix.list_length(Prefix.emitted_prefix(p,U64.is_zero(rest),rest)),1n),
            Nat.add(Word.count(64n,U64.to_word(rest)),1n),
            Nat.add(1n,Word.count(64n,U64.to_word(rest))),
            Equal.cong(Nat,Nat,n => Nat.add(n,1n),
              Prefix.list_length(Prefix.emitted_prefix(p,U64.is_zero(rest),rest)),
              Word.count(64n,U64.to_word(rest)),
              ih),
            nat_add_comm(Word.count(64n,U64.to_word(rest)),1n)),
          Equal.sym(Nat,Word.count(64n,U64.to_word(bb)),
            Nat.add(1n,Word.count(64n,U64.to_word(rest))),
            Budget.nonzero_clear_word_count(bb,e))))))

def emitted_prefix_count_cases(+n: Nat,+empty: Bool,+bb: U64,
  +budget: {Nat.is_lt(Word.count(64n,U64.to_word(bb)),Nat.add(1n,n)) == True{} : Bool},
  +same: {empty == U64.is_zero(bb) : Bool}) ->
  {Prefix.list_length(Prefix.emitted_prefix(n,empty,bb)) == Word.count(64n,U64.to_word(bb)) : Nat}:
  match n:
    case 0n:
      Equal.trans(Nat,Prefix.list_length(Prefix.emitted_prefix(0n,empty,bb)),0n,
        Word.count(64n,U64.to_word(bb)),{==},
        Equal.sym(Nat,Word.count(64n,U64.to_word(bb)),0n,
          count_lt_one_zero(Word.count(64n,U64.to_word(bb)),budget)))
    case 1n+p:
      match empty:
        case True{}:
          Equal.trans(Nat,Prefix.list_length(Prefix.emitted_prefix(1n+p,True{},bb)),0n,
            Word.count(64n,U64.to_word(bb)),{==},
            Equal.sym(Nat,Word.count(64n,U64.to_word(bb)),0n,
              u64_zero_count(bb,Equal.trans(Bool,U64.is_zero(bb),empty,True{},
                Equal.sym(Bool,empty,U64.is_zero(bb),same),{==}))))
        case False{}:
          +rest = U64.clear_lsb(bb)
          +next_budget: {Nat.is_lt(Word.count(64n,U64.to_word(rest)),Nat.add(1n,p)) == True{} : Bool} =
            Budget.clear_count_budget(bb,1n+p,budget,
              Equal.trans(Bool,U64.is_zero(bb),empty,False{},Equal.sym(Bool,empty,U64.is_zero(bb),same),{==}))
          emitted_prefix_count_nonzero(p,bb,budget,
            Equal.trans(Bool,U64.is_zero(bb),empty,False{},Equal.sym(Bool,empty,U64.is_zero(bb),same),{==}),
            emitted_prefix_count_cases(p,U64.is_zero(rest),rest,next_budget,{==}))

def emitted_prefix_count(+n: Nat,+bb: U64,
  +budget: {Nat.is_lt(Word.count(64n,U64.to_word(bb)),Nat.add(1n,n)) == True{} : Bool}) ->
  {Prefix.list_length(Prefix.emitted_prefix(n,U64.is_zero(bb),bb)) == Word.count(64n,U64.to_word(bb)) : Nat}:
  emitted_prefix_count_cases(n,U64.is_zero(bb),bb,budget,{==})

def bit_squares_cardinality(+n: Nat,+bb: U64,
  +budget: {Nat.is_lt(Word.count(64n,U64.to_word(bb)),Nat.add(1n,n)) == True{} : Bool}) ->
  {Prefix.list_length(Chess.bit_squares(n,U64.is_zero(bb),bb,Nil{})) == Word.count(64n,U64.to_word(bb)) : Nat}:
  Equal.trans(Nat,
    Prefix.list_length(Chess.bit_squares(n,U64.is_zero(bb),bb,Nil{})),
    Nat.add(Prefix.list_length(Prefix.emitted_prefix(n,U64.is_zero(bb),bb)),0n),
    Word.count(64n,U64.to_word(bb)),
    Prefix.bit_squares_cardinality_acc(n,U64.is_zero(bb),bb,Nil{}),
    Equal.trans(Nat,
      Nat.add(Prefix.list_length(Prefix.emitted_prefix(n,U64.is_zero(bb),bb)),0n),
      Prefix.list_length(Prefix.emitted_prefix(n,U64.is_zero(bb),bb)),
      Word.count(64n,U64.to_word(bb)),
      nat_add_zero_right(Prefix.list_length(Prefix.emitted_prefix(n,U64.is_zero(bb),bb))),
      emitted_prefix_count(n,bb,budget)))

def le_weaken_right_succ(+a: Nat,+b: Nat,+e: {Nat.is_le(a,b) == True{} : Bool}) ->
  {Nat.is_le(a,1n+b) == True{} : Bool}:
  match a b:
    case 0n _: {==}
    case 1n 0n: {==}
    case 1n+1n+p 0n: Old.false_true({Nat.is_le(1n+1n+p,1n) == True{} : Bool},e)
    case 1n+p 1n+q: le_weaken_right_succ(p,q,e)

def count_le_width(+n: Nat,+w: Word(n)) -> {Nat.is_le(Word.count(n,w),n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w:
        case WCon{False{},tail}:
          le_weaken_right_succ(Word.count(p,tail),p,count_le_width(p,tail))
        case WCon{True{},tail}:
          Equal.trans(Bool,Nat.is_le(Nat.add(1n,Word.count(p,tail)),1n+p),
            Nat.is_le(Word.count(p,tail),p),True{},
            {==},count_le_width(p,tail))

def le_succ_pair(+a: Nat,+b: Nat) ->
  {Nat.is_le(1n+a,1n+b) == Nat.is_le(a,b) : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+q: {==}
    case 1n+p 0n: {==}
    case 1n+p 1n+q: {==}

def count_lt_width_succ(+n: Nat,+w: Word(n)) ->
  {Nat.is_lt(Word.count(n,w),1n+n) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_lt(Word.count(n,w),1n+n),
    Nat.is_le(1n+Word.count(n,w),1n+n),True{},
    Equal.sym(Bool,Nat.is_le(1n+Word.count(n,w),1n+n),
      Nat.is_lt(Word.count(n,w),1n+n),NatProof.suc_le(Word.count(n,w),1n+n)),
    Equal.trans(Bool,Nat.is_le(1n+Word.count(n,w),1n+n),Nat.is_le(Word.count(n,w),n),True{},
      le_succ_pair(Word.count(n,w),n),count_le_width(n,w)))

def actual_bit_squares_cardinality(+bb: U64) ->
  {Prefix.list_length(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{})) == Word.count(64n,U64.to_word(bb)) : Nat}:
  bit_squares_cardinality(64n,bb,count_lt_width_succ(64n,U64.to_word(bb)))
