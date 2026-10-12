import Base
import ./Equivalence.bend as E
import ../../../legal_probe/Chess.bend as Chess
import ../accepted_castle/Spec.bend as S
import ../castle_chain/Spec.bend as C
import ../storage/Observe.bend as Observe
import ../u64/Words.bend as W

def actual_fast_full(+xs: List<&2,Chess.Ply>,+c: Observe.Cells,+b: Chess.Board,
  +sensitive: U64,+tail: List<&2,Chess.Ply>,
  premise: C.every(m => E.bypass_safe(c,b,sensitive,m),xs)) ->
  {Chess.filter_fast(xs,b,sensitive,(Observe.pack(c),tail)) ==
    Chess.filter_legal(xs,b,(Observe.pack(c),tail)) : Array<U64> & List<&2,Chess.Ply>}:
  E.actual_fast_full(xs,c,b,sensitive,tail,premise)

def actual_prepare_full(+c: Observe.Cells,+b: Chess.Board,+xs: List<&2,Chess.Ply>,
  premise: C.every(m => E.bypass_safe(c,b,E.prepared_sensitive(c,b),m),xs)) ->
  {Chess.filter_prepare(b,(Observe.pack(c),xs)) ==
    Chess.filter_legal(xs,b,(Observe.pack(c),Nil{})) : Array<U64> & List<&2,Chess.Ply>}:
  E.actual_prepare_full(c,b,xs,premise)

# A concrete mixed order, repeated candidate, and repeated untouched tail.
# Explicit actual check-answer witnesses are assumptions, not board-validity claims.
def duplicate_order_tail(+c: Observe.Cells,+b: Chess.Board,
  safe_m_first: S.safe(c,b,Chess.Ply{12,28,0,0}),
  safe_m_last: S.safe(c,b,Chess.Ply{12,28,0,0}),
  safe_n: S.safe(c,b,Chess.Ply{13,29,0,0})) ->
  {Chess.filter_fast(
    Con{Chess.Ply{12,28,0,0},Con{Chess.Ply{13,29,0,0},Con{Chess.Ply{12,28,0,0},Nil{}}}},
    b,U64.zero,(Observe.pack(c),
      Con{Chess.Ply{13,29,0,0},Con{Chess.Ply{12,28,0,0},Con{Chess.Ply{12,28,0,0},Nil{}}}})) ==
    Chess.filter_legal(
    Con{Chess.Ply{12,28,0,0},Con{Chess.Ply{13,29,0,0},Con{Chess.Ply{12,28,0,0},Nil{}}}},
    b,(Observe.pack(c),
      Con{Chess.Ply{13,29,0,0},Con{Chess.Ply{12,28,0,0},Con{Chess.Ply{12,28,0,0},Nil{}}}}))
    : Array<U64> & List<&2,Chess.Ply>}:
  E.actual_fast_full(
    Con{Chess.Ply{12,28,0,0},Con{Chess.Ply{13,29,0,0},Con{Chess.Ply{12,28,0,0},Nil{}}}},
    c,b,U64.zero,
    Con{Chess.Ply{13,29,0,0},Con{Chess.Ply{12,28,0,0},Con{Chess.Ply{12,28,0,0},Nil{}}}},
    (e => safe_m_first,(e => safe_n,(e => safe_m_last,Unit{}))))


# Closed implementation counterexample to deleting the bypass-retention premise.
# The board and lookup table are deliberately arbitrary; no geometry assumption.
def counter_cells() -> Observe.Cells:
  Observe.Cell{U64.from_u32(2)}

def counter_board() -> Chess.Board:
  Chess.Board{U64.zero,U64.from_u32(2),U64.zero,U64.zero,U64.zero,
    U64.one,U64.one,U64.from_u32(2),1,0,64}

def counter_move() -> Chess.Ply:
  Chess.Ply{2,3,0,0}

def counter_requires_false() ->
  {Chess.filter_requires(U64.one,counter_move()) == False{} : Bool}:
  {==}

def counter_checked_true() ->
  {S.checked(counter_cells(),counter_board(),counter_move()) == True{} : Bool}:
  {==}

def counter_fast_retains() ->
  {Chess.filter_fast(Con{counter_move(),Nil{}},counter_board(),U64.one,
    (Observe.pack(counter_cells()),Nil{})) ==
    (Observe.pack(counter_cells()),Con{counter_move(),Nil{}})
    : Array<U64> & List<&2,Chess.Ply>}:
  {==}

def counter_full_rejects() ->
  {Chess.filter_legal(Con{counter_move(),Nil{}},counter_board(),
    (Observe.pack(counter_cells()),Nil{})) ==
    (Observe.pack(counter_cells()),Nil{})
    : Array<U64> & List<&2,Chess.Ply>}:
  {==}

# Replacing the False antecedent with True makes this wrong premise vacuous.
def counter_wrong_guard_holds() ->
  C.every(m => {Chess.filter_requires(U64.one,m) == True{} : Bool} ->
    S.safe(counter_cells(),counter_board(),m),Con{counter_move(),Nil{}}):
  (r => W.false_true(S.safe(counter_cells(),counter_board(),counter_move()),
    Equal.trans(Bool,False{},Chess.filter_requires(U64.one,counter_move()),True{},
      Equal.sym(Bool,Chess.filter_requires(U64.one,counter_move()),False{},counter_requires_false()),
      r)),Unit{})

# Closed counterexample to replacing prepare's exact mask by all set bits.
def prepare_counter_cells() -> Observe.Cells:
  Observe.Branch{Observe.Cell{U64.zero},Observe.Cell{U64.from_u32(2)}}

def prepare_counter_board() -> Chess.Board:
  Chess.Board{U64.zero,U64.from_u32(2),U64.zero,U64.zero,U64.zero,
    U64.one,U64.zero,U64.from_u32(2),1,0,64}

def prepare_counter_move() -> Chess.Ply:
  Chess.Ply{0,3,0,0}

def prepare_counter_initial_false() ->
  {S.initial(prepare_counter_cells(),prepare_counter_board()) == False{} : Bool}:
  {==}

def prepare_counter_sensitive_zero() ->
  {E.prepared_sensitive(prepare_counter_cells(),prepare_counter_board()) == U64.zero : U64}:
  {==}

def prepare_counter_checked_true() ->
  {S.checked(prepare_counter_cells(),prepare_counter_board(),prepare_counter_move()) == True{} : Bool}:
  {==}

def prepare_counter_all_requires_true() ->
  {Chess.filter_requires(U64{4294967295,4294967295},prepare_counter_move()) == True{} : Bool}:
  {==}

def prepare_counter_weakened_premise_holds() ->
  C.every(m => E.bypass_safe(prepare_counter_cells(),prepare_counter_board(),
    U64{4294967295,4294967295},m),Con{prepare_counter_move(),Nil{}}):
  (r => W.false_true(S.safe(prepare_counter_cells(),prepare_counter_board(),prepare_counter_move()),
    Equal.trans(Bool,False{},
      Chess.filter_requires(U64{4294967295,4294967295},prepare_counter_move()),True{},
      Equal.sym(Bool,Chess.filter_requires(U64{4294967295,4294967295},prepare_counter_move()),False{},r),
      prepare_counter_all_requires_true())),Unit{})

def prepare_counter_retains() ->
  {Chess.filter_prepare(prepare_counter_board(),
    (Observe.pack(prepare_counter_cells()),Con{prepare_counter_move(),Nil{}})) ==
    (Observe.pack(prepare_counter_cells()),Con{prepare_counter_move(),Nil{}})
    : Array<U64> & List<&2,Chess.Ply>}:
  {==}

def prepare_counter_full_rejects() ->
  {Chess.filter_legal(Con{prepare_counter_move(),Nil{}},prepare_counter_board(),
    (Observe.pack(prepare_counter_cells()),Nil{})) ==
    (Observe.pack(prepare_counter_cells()),Nil{})
    : Array<U64> & List<&2,Chess.Ply>}:
  {==}

# Distinguishable candidates make the order control a false equality.
def order_counter_board() -> Chess.Board:
  Chess.Board{U64.zero,U64.zero,U64.zero,U64.zero,U64.zero,
    U64.zero,U64.zero,U64.zero,1,0,64}

def order_counter_original() ->
  {Chess.filter_legal(
    Con{Chess.Ply{12,28,0,0},Con{Chess.Ply{13,29,0,0},Con{Chess.Ply{12,28,0,0},Nil{}}}},
    order_counter_board(),(Observe.pack(Observe.Cell{U64.zero}),Nil{})) ==
    (Observe.pack(Observe.Cell{U64.zero}),
      Con{Chess.Ply{12,28,0,0},Con{Chess.Ply{13,29,0,0},Con{Chess.Ply{12,28,0,0},Nil{}}}})
    : Array<U64> & List<&2,Chess.Ply>}:
  {==}

def order_counter_reordered() ->
  {Chess.filter_legal(
    Con{Chess.Ply{13,29,0,0},Con{Chess.Ply{12,28,0,0},Con{Chess.Ply{12,28,0,0},Nil{}}}},
    order_counter_board(),(Observe.pack(Observe.Cell{U64.zero}),Nil{})) ==
    (Observe.pack(Observe.Cell{U64.zero}),
      Con{Chess.Ply{12,28,0,0},Con{Chess.Ply{12,28,0,0},Con{Chess.Ply{13,29,0,0},Nil{}}}})
    : Array<U64> & List<&2,Chess.Ply>}:
  {==}
