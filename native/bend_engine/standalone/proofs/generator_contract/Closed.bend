# Candidate source proof. Qualification is separate; this is not an accepted law.
# No token guard, assumed table, law declaration, axiom, or foreign witness.
import Base
import ../../Tables.bend as Tables
import ../initialized_attacks/Build.bend as Init

def public_builder_matches_recipe() ->
  {Tables.build() == Init.run(17n,U64.zero(),128n,64n) : Array<U64>}:
  {==}
