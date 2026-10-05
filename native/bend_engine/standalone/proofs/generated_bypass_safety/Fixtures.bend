import Base
import ./Spec.bend as S
import ./consumer.bend as Bridge
import ./Forced.bend as Forced
import ../candidate_bypass_classification/consumer.bend as Classification
import ../storage/Observe.bend as O
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../ordinary_nonslider_preservation/consumer.bend as Ordinary
import ../promotion_bypass_preservation/consumer.bend as Promotion
import ../promotion/Spec.bend as P
import ../fast_full_equivalence/Equivalence.bend as F
import ../../../legal_probe/Chess.bend as Chess

# The retained tail is arbitrary, duplicated and unclassified.
def duplicate_tail() -> List<&2,Chess.Ply>:
  Chess.Ply{99,100,99,99} <> Chess.Ply{99,100,99,99} <> Nil{}

def ep_full_step(+c: O.Cells,+b: Chess.Board,+src: U32,+dst: U32) ->
  {Chess.filter_fast_step(Chess.filter_requires(F.prepared_sensitive(c,b),Chess.Ply{src,dst,0,1}),b,Chess.Ply{src,dst,0,1},
      (O.pack(c),duplicate_tail())) == Chess.filter_step(b,Chess.Ply{src,dst,0,1},(O.pack(c),duplicate_tail())) : Array<U64> & List<&2,Chess.Ply>}:
  Forced.ep_step(c,b,Chess.Ply{src,dst,0,1},(src,(dst,{==})),(O.pack(c),duplicate_tail()))

def guarded_castle_full_step() ->
  {Chess.filter_fast_step(Chess.filter_requires(F.prepared_sensitive(O.Cell{U64.zero()},Classification.fixture_board()),Chess.Ply{4,6,0,2}),
      Classification.fixture_board(),Chess.Ply{4,6,0,2},(ALeaf{U64.zero()},duplicate_tail())) ==
    Chess.filter_step(Classification.fixture_board(),Chess.Ply{4,6,0,2},(ALeaf{U64.zero()},duplicate_tail())) : Array<U64> & List<&2,Chess.Ply>}:
  Forced.castle_step(O.Cell{U64.zero()},Classification.fixture_board(),Chess.Ply{4,6,0,2},Inl{({==},{==})},
    (ALeaf{U64.zero()},duplicate_tail()))

def reordered_promotions() -> List<&2,Chess.Ply>:
  P.ply(49,57,P.Queen{}) <> P.ply(49,57,P.Knight{}) <> P.ply(49,57,P.Queen{}) <>
    P.ply(49,57,P.Bishop{}) <> P.ply(49,57,P.Rook{}) <> Nil{}

def promotion_repeated_order() ->
  {Chess.filter_fast(reordered_promotions(),Promotion.fixture_board(),F.prepared_sensitive(Promotion.fixture_cells(),Promotion.fixture_board()),
      (O.pack(Promotion.fixture_cells()),duplicate_tail())) ==
    (O.pack(Promotion.fixture_cells()),P.ply(49,57,P.Rook{}) <> P.ply(49,57,P.Bishop{}) <> P.ply(49,57,P.Queen{}) <>
      P.ply(49,57,P.Knight{}) <> P.ply(49,57,P.Queen{}) <> duplicate_tail()) : Array<U64> & List<&2,Chess.Ply>}:
  {==}

def repeated_ordinary() -> List<&2,Chess.Ply>:
  Chess.Ply{19,34,0,0} <> Chess.Ply{19,34,0,0} <> Chess.Ply{19,34,0,0} <> Nil{}

def ordinary_repeated_order() ->
  {Chess.filter_fast(repeated_ordinary(),Ordinary.fixture_board(),F.prepared_sensitive(Ordinary.fixture_cells(),Ordinary.fixture_board()),
      (O.pack(Ordinary.fixture_cells()),duplicate_tail())) ==
    (O.pack(Ordinary.fixture_cells()),Chess.Ply{19,34,0,0} <> Chess.Ply{19,34,0,0} <> Chess.Ply{19,34,0,0} <> duplicate_tail()) : Array<U64> & List<&2,Chess.Ply>}:
  {==}

def ordinary_member() ->
  E.member(Chess.Ply{19,34,0,0},S.candidates(Ordinary.fixture_cells(),Ordinary.fixture_board())):
  Inl{{==}}

def ordinary_old() -> S.old(Ordinary.fixture_cells(),Ordinary.fixture_board()):
  (Ordinary.fixture_old_rook(),Ordinary.fixture_old_bishop())

def ordinary_lookup() -> S.lookup(Ordinary.fixture_cells(),Ordinary.fixture_board(),Chess.Ply{19,34,0,0}):
  here => bypass => (Ordinary.fixture_new_rook(),Ordinary.fixture_new_bishop())

def ordinary_actual_safe() ->
  {Chess.in_check(O.pack(Ordinary.fixture_cells()),Chess.make_move(Ordinary.fixture_board(),Chess.Ply{19,34,0,0}),1) ==
    (O.pack(Ordinary.fixture_cells()),False{}) : Array<U64> & Bool}:
  Bridge.actual_candidate_safe(Ordinary.fixture_cells(),Ordinary.fixture_board(),Chess.Ply{19,34,0,0},True{},
    ordinary_member(),Ordinary.fixture_bypass(),Ordinary.fixture_valid(),Ordinary.fixture_nonempty(),{==},
    ordinary_old(),Ordinary.fixture_initial(),ordinary_lookup())

def promotion_member(+choice: P.Choice) ->
  E.member(P.ply(49,57,choice),S.candidates(Promotion.fixture_cells(),Promotion.fixture_board())):
  match choice:
    case P.Knight{}: Inl{{==}}
    case P.Bishop{}: Inr{Inl{{==}}}
    case P.Rook{}: Inr{Inr{Inl{{==}}}}
    case P.Queen{}: Inr{Inr{Inr{Inl{{==}}}}}

def promotion_old() -> S.old(Promotion.fixture_cells(),Promotion.fixture_board()):
  (Promotion.fixture_old_rook(),Promotion.fixture_old_bishop())

def promotion_post(+choice: P.Choice) ->
  S.post(Promotion.fixture_cells(),Promotion.fixture_board(),P.ply(49,57,choice)):
  (Promotion.fixture_new_rook(choice),Promotion.fixture_new_bishop(choice))

def promotion_lookup(+choice: P.Choice) ->
  S.lookup(Promotion.fixture_cells(),Promotion.fixture_board(),P.ply(49,57,choice)):
  here => bypass => promotion_post(choice)

# This conclusion invokes the new generated-origin/classification/no-check join.

def promotion_actual_safe(+choice: P.Choice) ->
  {Chess.in_check(O.pack(Promotion.fixture_cells()),Chess.make_move(Promotion.fixture_board(),P.ply(49,57,choice)),1) ==
    (O.pack(Promotion.fixture_cells()),False{}) : Array<U64> & Bool}:
  Bridge.actual_candidate_safe(Promotion.fixture_cells(),Promotion.fixture_board(),P.ply(49,57,choice),True{},
    promotion_member(choice),Promotion.fixture_bypass(choice),Promotion.fixture_valid(),Promotion.fixture_nonempty(),{==},
    promotion_old(),Promotion.fixture_initial(),promotion_lookup(choice))

def promotion_origins() ->
  C.every(m => E.member(m,S.candidates(Promotion.fixture_cells(),Promotion.fixture_board())),reordered_promotions()):
  (promotion_member(P.Queen{}),(promotion_member(P.Knight{}),(promotion_member(P.Queen{}),
    (promotion_member(P.Bishop{}),(promotion_member(P.Rook{}),Unit{})))))

def promotion_lookups() ->
  S.lookups(Promotion.fixture_cells(),Promotion.fixture_board(),reordered_promotions()):
  (promotion_lookup(P.Queen{}),(promotion_lookup(P.Knight{}),(promotion_lookup(P.Queen{}),
    (promotion_lookup(P.Bishop{}),(promotion_lookup(P.Rook{}),Unit{})))))

def promotion_repeated_fast_full() ->
  {Chess.filter_fast(reordered_promotions(),Promotion.fixture_board(),F.prepared_sensitive(Promotion.fixture_cells(),Promotion.fixture_board()),
      (O.pack(Promotion.fixture_cells()),duplicate_tail())) ==
    Chess.filter_legal(reordered_promotions(),Promotion.fixture_board(),(O.pack(Promotion.fixture_cells()),duplicate_tail())) : Array<U64> & List<&2,Chess.Ply>}:
  Bridge.member_list_fast_full(Promotion.fixture_cells(),Promotion.fixture_board(),True{},reordered_promotions(),duplicate_tail(),
    promotion_origins(),Promotion.fixture_valid(),Promotion.fixture_nonempty(),{==},promotion_old(),Promotion.fixture_initial(),promotion_lookups())

def generated_promotion_lookups() ->
  S.lookups(Promotion.fixture_cells(),Promotion.fixture_board(),S.candidates(Promotion.fixture_cells(),Promotion.fixture_board())):
  (promotion_lookup(P.Knight{}),(promotion_lookup(P.Bishop{}),(promotion_lookup(P.Rook{}),(promotion_lookup(P.Queen{}),Unit{}))))

def promotion_generated_fast_full() ->
  {Chess.filter_fast(S.candidates(Promotion.fixture_cells(),Promotion.fixture_board()),Promotion.fixture_board(),
      F.prepared_sensitive(Promotion.fixture_cells(),Promotion.fixture_board()),(O.pack(Promotion.fixture_cells()),duplicate_tail())) ==
    Chess.filter_legal(S.candidates(Promotion.fixture_cells(),Promotion.fixture_board()),Promotion.fixture_board(),
      (O.pack(Promotion.fixture_cells()),duplicate_tail())) : Array<U64> & List<&2,Chess.Ply>}:
  Bridge.generated_fast_full(Promotion.fixture_cells(),Promotion.fixture_board(),True{},duplicate_tail(),
    Promotion.fixture_valid(),Promotion.fixture_nonempty(),{==},promotion_old(),Promotion.fixture_initial(),generated_promotion_lookups())

def promotion_actual_legal_full() ->
  {Chess.legal_moves(O.pack(Promotion.fixture_cells()),Promotion.fixture_board()) ==
    Chess.filter_legal(S.candidates(Promotion.fixture_cells(),Promotion.fixture_board()),Promotion.fixture_board(),
      (O.pack(Promotion.fixture_cells()),Nil{})) : Array<U64> & List<&2,Chess.Ply>}:
  Bridge.actual_legal_full(Promotion.fixture_cells(),Promotion.fixture_board(),True{},
    Promotion.fixture_valid(),Promotion.fixture_nonempty(),{==},promotion_old(),Promotion.fixture_initial(),generated_promotion_lookups())

def ordinary_origins() ->
  C.every(m => E.member(m,S.candidates(Ordinary.fixture_cells(),Ordinary.fixture_board())),repeated_ordinary()):
  (ordinary_member(),(ordinary_member(),(ordinary_member(),Unit{})))

def ordinary_lookups() -> S.lookups(Ordinary.fixture_cells(),Ordinary.fixture_board(),repeated_ordinary()):
  (ordinary_lookup(),(ordinary_lookup(),(ordinary_lookup(),Unit{})))

def ordinary_repeated_fast_full() ->
  {Chess.filter_fast(repeated_ordinary(),Ordinary.fixture_board(),F.prepared_sensitive(Ordinary.fixture_cells(),Ordinary.fixture_board()),
      (O.pack(Ordinary.fixture_cells()),duplicate_tail())) ==
    Chess.filter_legal(repeated_ordinary(),Ordinary.fixture_board(),(O.pack(Ordinary.fixture_cells()),duplicate_tail())) : Array<U64> & List<&2,Chess.Ply>}:
  Bridge.member_list_fast_full(Ordinary.fixture_cells(),Ordinary.fixture_board(),True{},repeated_ordinary(),duplicate_tail(),
    ordinary_origins(),Ordinary.fixture_valid(),Ordinary.fixture_nonempty(),{==},ordinary_old(),Ordinary.fixture_initial(),ordinary_lookups())
