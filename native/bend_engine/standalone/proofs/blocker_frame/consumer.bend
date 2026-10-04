import Base
import ./Blocker.bend as B
import ../ray/Path.bend as P
import ../ray/Spec.bend as S
import ../../Tables.bend as Tables

def path_frame(+xs: List<&2,Nat>,+a: U64,+b: U64,
  e: {B.agree(xs,a,b) == True{} : Bool}) -> {P.attack(xs,a) == P.attack(xs,b) : U64}:
  B.attack_equal(xs,a,b,e)

def actual_ray(+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+a: U64,+b: U64,+acc: U64,
  route: {S.trace(n,sq,dir,xs) == True{} : Bool},
  e: {B.agree(xs,a,b) == True{} : Bool}) ->
  {Tables.ray(n,False{},sq,dir,a,False{},acc) ==
    Tables.ray(n,False{},sq,dir,b,False{},acc) : U64}:
  B.actual_ray(n,sq,dir,xs,a,b,acc,route,e)

def weakening_path() -> List<&2,Nat>:
  1n <> 2n <> 3n <> 4n <> Nil{}

def weakening_new_agree() ->
  {B.agree(weakening_path(),U64.from_u32(4),U64.from_u32(12)) == True{} : Bool}: {==}

def weakening_old_disagrees() ->
  {P.agree(weakening_path(),U64.from_u32(4),U64.from_u32(12)) == False{} : Bool}: {==}

def weakening_attack_a() ->
  {P.attack(weakening_path(),U64.from_u32(4)) == U64.from_u32(6) : U64}: {==}

def weakening_attack_b() ->
  {P.attack(weakening_path(),U64.from_u32(12)) == U64.from_u32(6) : U64}: {==}

def weakening_actual_equal(+acc: U64) ->
  {Tables.ray(4n,False{},0,0,U64.from_u32(4),False{},acc) ==
    Tables.ray(4n,False{},0,0,U64.from_u32(12),False{},acc) : U64}:
  B.actual_ray(4n,0,0,weakening_path(),U64.from_u32(4),U64.from_u32(12),acc,{==},
    weakening_new_agree())

def weakening_actual_a() ->
  {Tables.ray(4n,False{},0,0,U64.from_u32(4),False{},U64.zero()) == U64.from_u32(6) : U64}: {==}

def weakening_actual_b() ->
  {Tables.ray(4n,False{},0,0,U64.from_u32(12),False{},U64.zero()) == U64.from_u32(6) : U64}: {==}

# A one-sided blocker before the final square must fail the premise.
def one_sided_rejected() ->
  {B.agree(1n <> 2n <> Nil{},U64.zero(),U64.from_u32(2)) == False{} : Bool}: {==}

def reverse_one_sided_rejected() ->
  {B.agree(1n <> 2n <> Nil{},U64.from_u32(2),U64.zero()) == False{} : Bool}: {==}

def reachable_attack_a() ->
  {P.attack(1n <> 2n <> Nil{},U64.zero()) == U64.from_u32(6) : U64}: {==}

def reachable_attack_b() ->
  {P.attack(1n <> 2n <> Nil{},U64.from_u32(2)) == U64.from_u32(2) : U64}: {==}

def actual_reachable_a() ->
  {Tables.ray(2n,False{},0,0,U64.zero(),False{},U64.zero()) == U64.from_u32(6) : U64}: {==}

def actual_reachable_b() ->
  {Tables.ray(2n,False{},0,0,U64.from_u32(2),False{},U64.zero()) == U64.from_u32(2) : U64}: {==}

def mismatch_after_clear_rejected() ->
  {B.agree(weakening_path(),U64.zero(),U64.from_u32(4)) == False{} : Bool}: {==}

def empty_agree(+a: U64,+b: U64) ->
  {B.agree(Nil{},a,b) == True{} : Bool}: {==}

def terminal_agree(+x: Nat,+a: U64,+b: U64) ->
  {B.agree(x <> Nil{},a,b) == True{} : Bool}: {==}
