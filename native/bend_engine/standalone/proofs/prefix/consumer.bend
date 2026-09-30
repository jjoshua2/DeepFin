import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Spec.bend as S
import ../layout/Spec.bend as Layout
import ../../Tables.bend as Tables

# The certificates are produced, not additional assumptions on real arrays.
def any_buffer(+n: Nat, +k: Nat, a: Array<U64>, +valid: {Nat.is_lt(Nat.add(n,k),129n) == True{} : Bool}) ->
  {Tables.tables(n,U32.from_nat(k),S.prefix(k),a) == S.scheduled(n,k,a) : Array<U64>}:
  Laws.tables_follow_prefix(n,k,a,valid)

def rook_bishop_boundary() ->
  {U32.is_le(U32.add(S.prefix(63n),Layout.size(63)),S.prefix(64n)) == True{} : Bool}:
  Laws.blocks_ordered(63n,64n,{==},{==})

def distant_blocks() ->
  {U32.is_le(U32.add(S.prefix(0n),Layout.size(0)),S.prefix(127n)) == True{} : Bool}:
  Laws.blocks_ordered(0n,127n,{==},{==})

def last_step() ->
  {U64.add(U64.from_u32(S.prefix(127n)),U64.from_u32(Layout.size(127))) == U64.from_u32(S.prefix(128n)) : U64}:
  Laws.step_no_overflow(127n,{==})
