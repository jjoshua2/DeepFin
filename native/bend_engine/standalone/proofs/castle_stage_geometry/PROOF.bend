import Base
import ./LAWS.bend as Laws
import ./Inputs.bend as Inputs
import ../castle_kings/Spec.bend as K
import ../castle_kings/Check.bend as Check
import ../initialized_attacks/PROOF.bend as InitializedProof
import ../initialized_attacks/LAWS.bend as Initialized

def Laws.initialized_castling_stage_check_matches_geometry(d,seed,n,extra,b,white,ks,stage,depth,tables,full,good,turn,one,guard):
  Initialized.initialized_single_king_check_matches_geometry(d,seed,n,extra,
    K.stage_board(b,white,ks,stage),U32.to_nat(K.stage_square(white,ks,stage)),white,
    depth,tables,full,Check.bounded(white,ks,stage),
    Inputs.singleton(b,white,ks,stage,good,turn,one,guard))
