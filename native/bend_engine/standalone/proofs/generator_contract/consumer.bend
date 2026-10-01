# An independent importing caller must check the full closed proposition.
import Base
import ../../Tables.bend as Tables
import ../initialized_attacks/Build.bend as Init
import ./Closed.bend as Closed

def actual_public_builder() ->
  {Tables.build() == Init.run(17n,U64.zero(),128n,64n) : Array<U64>}:
  Closed.public_builder_matches_recipe()
