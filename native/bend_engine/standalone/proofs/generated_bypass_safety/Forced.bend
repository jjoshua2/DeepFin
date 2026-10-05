import Base
import ./Spec.bend as S
import ../storage/Observe.bend as O
import ../castle_chain/Spec.bend as C
import ../accepted_castle/Spec.bend as Prepared
import ../candidate_bypass_classification/Fields.bend as Fields
import ../candidate_bypass_classification/Bypass.bend as Bypass
import ../castle_safety/Mask.bend as Mask
import ../fast_full_equivalence/Equivalence.bend as F
import ../u64/Words.bend as W
import ../../../legal_probe/Chess.bend as Chess

# Forced paths use the actual full check, with the exact supplied pair.
def step(+c: O.Cells,+b: Chess.Board,+m: Chess.Ply,
  required: {Chess.filter_requires(F.prepared_sensitive(c,b),m) == True{} : Bool},
  r: Array<U64> & List<&2,Chess.Ply>) ->
  {Chess.filter_fast_step(Chess.filter_requires(F.prepared_sensitive(c,b),m),b,m,r) ==
    Chess.filter_step(b,m,r) : Array<U64> & List<&2,Chess.Ply>}:
  %Equal.sym(Bool,Chess.filter_requires(F.prepared_sensitive(c,b),m),True{},required) :
    {Chess.filter_fast_step(_,b,m,r) == Chess.filter_step(b,m,r) : Array<U64> & List<&2,Chess.Ply>}
  {==}

# These occurrences need no moved-board lookup facts: the antecedent is empty.
def lookup(+c: O.Cells,+b: Chess.Board,+m: Chess.Ply,
  +required: {Chess.filter_requires(F.prepared_sensitive(c,b),m) == True{} : Bool}) -> S.lookup(c,b,m):
  here => bypass => W.false_true(S.post(c,b,m),Equal.trans(Bool,False{},
    Chess.filter_requires(F.prepared_sensitive(c,b),m),True{},
    Equal.sym(Bool,Chess.filter_requires(F.prepared_sensitive(c,b),m),False{},bypass),required))

def ep_lookup(+c: O.Cells,+b: Chess.Board,+m: Chess.Ply,ep: Fields.ep(m)) -> S.lookup(c,b,m):
  lookup(c,b,m,Bypass.ep_required(m,F.prepared_sensitive(c,b),ep))

def castle_lookup(+c: O.Cells,+b: Chess.Board,+m: Chess.Ply,castle: C.castle(b,m)) -> S.lookup(c,b,m):
  lookup(c,b,m,Mask.certificate(b,m,Prepared.rays(c,b),castle))

def ep_step(+c: O.Cells,+b: Chess.Board,+m: Chess.Ply,ep: Fields.ep(m),
  r: Array<U64> & List<&2,Chess.Ply>) ->
  {Chess.filter_fast_step(Chess.filter_requires(F.prepared_sensitive(c,b),m),b,m,r) ==
    Chess.filter_step(b,m,r) : Array<U64> & List<&2,Chess.Ply>}:
  step(c,b,m,Bypass.ep_required(m,F.prepared_sensitive(c,b),ep),r)

def castle_step(+c: O.Cells,+b: Chess.Board,+m: Chess.Ply,castle: C.castle(b,m),
  r: Array<U64> & List<&2,Chess.Ply>) ->
  {Chess.filter_fast_step(Chess.filter_requires(F.prepared_sensitive(c,b),m),b,m,r) ==
    Chess.filter_step(b,m,r) : Array<U64> & List<&2,Chess.Ply>}:
  step(c,b,m,Mask.certificate(b,m,Prepared.rays(c,b),castle),r)
