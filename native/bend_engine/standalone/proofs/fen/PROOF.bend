import Base
import ./LAWS.bend as Laws
import ./Consistency.bend as C
import ./Assembly.bend as A

def Laws.fen_result_is_consistent(pieces,turn,castle,target,half,full):
  C.actual(pieces,turn,castle,target,half,full)

def Laws.rejected_placement_rejects_fen(pieces,turn,castle,target,half,full,rejected):
  A.rejected(pieces,turn,castle,target,half,full,rejected)

def Laws.validated_fields_construct_exact_game(pieces,turn,castle,target,half,full,b,t,r,e,h,f,pb,pt,pr,pe,ph,pf,hb,fp,fb):
  A.fields(pieces,turn,castle,target,half,full,b,t,r,e,h,f,pb,pt,pr,pe,ph,pf,hb,fp,fb)
