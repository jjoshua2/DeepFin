import Base
import ./LAWS.bend as Laws
import ./Actual.bend as A
import ./Typed.bend as Typed
import ./Metadata.bend as Metadata
import ../../Position.bend as Position

def Laws.layout_append_exact(prefix,suffix,l): A.append(prefix,suffix,l)
def Laws.invalid_layout_rejected(s,l,e): A.rejected(Position.layout(s,l),A.invalid(s,l,e))
def Laws.accepted_layout_has_valid_prefix(prefix,suffix,l,e): A.accepted_prefix(prefix,suffix,l,e)
def Laws.layout_preserves_metadata(s,l): Metadata.metadata(s,l)
def Laws.typed_piece_transition(p,white,b,file,rank,valid): Typed.step(p,white,b,file,rank,valid)
