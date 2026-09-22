import Base
import ./LAWS.bend as Laws
import ./Spec.bend as S
import ./Facts.bend as Facts
import ./Size.bend as Size
import ./Certified.bend as Certified
import ./Monotone.bend as Monotone
import ./Nats.bend as Nats
import ./Builder.bend as Builder
import ../layout/Spec.bend as Layout
import ../layout/Low.bend as Low
import ../../Tables.bend as Tables

def Laws.prefix_widening(k,valid):
  Facts.wide(k,Facts.states(k,valid))

def Laws.prefix_in_allocation(k,valid):
  (Facts.lower(k,Facts.states(k,valid)),Facts.upper(k,Facts.states(k,valid)))

def Laws.step_no_overflow(k,valid):
  Equal.trans(U64,U64.add(U64.from_u32(S.prefix(k)),U64.from_u32(Layout.size(U32.from_nat(k)))),
    U64.add(U64.from_u32(S.prefix(k)),U64.from_u32(S.size(k))),U64.from_u32(S.prefix(1n+k)),
    Equal.cong(U32,U64,x => U64.add(U64.from_u32(S.prefix(k)),U64.from_u32(x)),
      Layout.size(U32.from_nat(k)),S.size(k),Certified.size(k,valid)),Facts.no_wrap(k,Facts.transitions(k,valid)))

def Laws.blocks_ordered(i,j,before,valid):
  +ivalid = Low.le_lt(i,j,128n,Nats.lt_to_le(i,j,before),valid)
  +step = Equal.cong(U32,U32,x => U32.add(S.prefix(i),x),
    Layout.size(U32.from_nat(i)),S.size(i),Certified.size(i,ivalid))
  %Equal.sym(U32,U32.add(S.prefix(i),Layout.size(U32.from_nat(i))),S.prefix(1n+i),step) :
    {U32.is_le(_,S.prefix(j)) == True{} : Bool}
  Monotone.ordered(1n+i,j,Nats.lt_suc(i,j,before),Nats.weaken(j,128n,valid))

def Laws.tables_follow_prefix(n,k,a,valid):
  Builder.refine(n,k,a,Certified.make(n,k,valid))

def Laws.logical_end():
  {==}
