import Base
import ./LAWS.bend as Laws
import ./Actual.bend as Actual
import ./Fill.bend as Fill

def Laws.write_preserves_other_location(a,i,j,v,e):
  Actual.frame(a,i,j,v,e)

def Laws.write_preserves_separation(a,i,j,at,v):
  Actual.stable(a,i,j,at,v)

def Laws.fill_preserves_unwritten_location(a,n,at,sq,bishop,mask,subset,j,e):
  Fill.frame(a,n,at,sq,bishop,mask,subset,j,e)

def Laws.fill_preserves_separation(a,n,at,sq,bishop,mask,subset,i,j):
  Fill.stable(a,n,at,sq,bishop,mask,subset,i,j)
