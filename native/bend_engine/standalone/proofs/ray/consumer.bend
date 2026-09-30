import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Spec.bend as S
import ./Rays.bend as Rays
import ./Path.bend as Path
import ./Grid.bend as Grid
import ./Termination.bend as End
import ../../Tables.bend as Tables
import ../../../legal_probe/Chess.bend as Chess
import ../data/Spec.bend as Data
import ../layout/Spec.bend as Layout

# Universal importing use; no desired attack, mask or storage certificate supplied.
def initialized(+d: Nat,+seed: U64,+before: Nat,+after: Nat,+k: Nat,+extra: Nat,+start: Nat,+occ: U64,
  depth: {d == 17n : Nat},budget: {Nat.is_lt(Nat.add(Data.count(before,after),k),129n) == True{} : Bool},
  ebudget: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}) ->
  {Chess.slide(Data.run(Array.new(U64,d,seed),before,after,k,extra,start),U32.from_nat(Nat.add(before,k)),occ) ==
    (Data.run(Array.new(U64,d,seed),before,after,k,extra,start),S.rays(Nat.add(before,k),occ)) : Array<U64> & U64}:
  Laws.initialized_lookup_matches_independent_rays(d,seed,before,after,k,extra,start,occ,depth,budget,ebudget)

def ray(+sq: Nat,+dir: Nat,+occ: U64,+acc: U64,
  sb: {Nat.is_lt(sq,64n) == True{} : Bool},db: {Nat.is_lt(dir,8n) == True{} : Bool}) ->
  {Tables.ray(7n,False{},U32.from_nat(sq),U32.from_nat(dir),occ,False{},acc) == U64.or(acc,Path.attack(Rays.path(7n,sq,dir),occ)) : U64}:
  Laws.ray_matches_independent_path(sq,dir,occ,acc,sb,db)

def unmasked(+key: Nat,+occ: U64,valid: {Nat.is_lt(key,128n) == True{} : Bool}) ->
  {S.actual_slider(key,occ) == S.rays(key,occ) : U64}:
  Laws.slider_matches_independent_rays(key,occ,valid)

def masked(+key: Nat,+occ: U64,valid: {Nat.is_lt(key,128n) == True{} : Bool}) ->
  {S.actual_slider(key,U64.and(occ,Layout.mask(U32.from_nat(key)))) == S.rays(key,occ) : U64}:
  Laws.relevant_mask_preserves_attacks(key,occ,valid)

def edge(+sq: Nat,+dir: Nat,sb: {Nat.is_lt(sq,64n) == True{} : Bool},db: {Nat.is_lt(dir,8n) == True{} : Bool}) ->
  {Grid.next(End.endpoint(Rays.path(7n,sq,dir),sq),dir) == 64n : Nat}:
  Laws.seven_steps_reach_edge(sq,dir,sb,db)

# Satisfiable full pipeline and both extreme selected keys.
def first_budget() -> {Nat.is_lt(Nat.add(Data.count(0n,127n),0n),129n) == True{} : Bool}: {==}
def last_budget() -> {Nat.is_lt(Nat.add(Data.count(127n,0n),0n),129n) == True{} : Bool}: {==}
def full_extras() -> {Nat.is_le(Nat.add(64n,0n),64n) == True{} : Bool}: {==}
# Explicit blocker inclusion, stopping and terminal-edge irrelevance.
def first_blocker() -> {Path.attack(1n <> 2n <> 3n <> Nil{},U64.bit(2n)) == U64.or(U64.bit(1n),U64.bit(2n)) : U64}: {==}
def terminal_blocker() -> {S.rays(0n,U64.zero()) == S.rays(0n,U64.bit(7n)) : U64}: {==}
