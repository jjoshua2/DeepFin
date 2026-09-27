import Base
import ./LAWS.bend as Laws
import ./Cases.bend as Cases
import ./Domain.bend as Domain
import ./Spec.bend as Spec
import ./Geometry.bend as Geometry
import ./Consequences.bend as Consequences

def Laws.mask_population(key,valid):
  %Domain.recover(key,valid) :
    {U64.popcount(Spec.mask(_)) == U32.from_nat(Geometry.population(U32.to_nat(key))) : U32}
  Cases.counts(U32.to_nat(key),valid)

def Laws.tight_population_bound(key,valid):
  Consequences.population(key,valid,Laws.mask_population(key,valid))

def Laws.shift_in_range(key,valid):
  Consequences.safe_shift(key,valid,Laws.mask_population(key,valid))

def Laws.size_positive(key,valid):
  Consequences.positive(key,valid,Laws.mask_population(key,valid))

def Laws.tight_size_bound(key,valid):
  Consequences.size_bound(key,valid,Laws.mask_population(key,valid))

def Laws.size_matches_capacity(key,valid):
  Consequences.capacity(key,valid,Laws.mask_population(key,valid))

def Laws.full_index_bound(key,occ,valid):
  Consequences.full_index(key,occ,valid,Laws.mask_population(key,valid))

def Laws.lookup_index_bound(key,occ,valid):
  Consequences.lookup_index(key,occ,valid,Laws.mask_population(key,valid))
