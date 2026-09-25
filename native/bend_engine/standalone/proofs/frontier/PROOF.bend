import Base
import ./LAWS.bend as Laws
import ./Traversal.bend as T

def Laws.layout_preserves_frontier(text,state,good):
  T.all(text,state,good)

def Laws.accepted_prefix_has_frontier(prefix,suffix,accepted):
  T.accepted_prefix(prefix,suffix,accepted)

def Laws.live_typed_insertion_is_fresh(prefix,piece,white,live):
  T.initialized_typed(prefix,piece,white,live)

def Laws.accepted_placement_has_consistent_board(text):
  T.initialized_result(text)
